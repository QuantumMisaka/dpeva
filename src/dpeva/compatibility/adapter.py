"""Immutable DeepMD-kit command and capability adapter."""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field

from dpeva.constants import VALID_DP_BACKENDS

from .deepmd import (
    CapabilityKey,
    CapabilityMatrix,
    CapabilityRecord,
    CapabilityUnavailable,
)


def _with_log(command: str, log_file: str | None) -> str:
    if log_file is None:
        return command
    return f"{command} > {shlex.quote(log_file)} 2>&1"


@dataclass(frozen=True)
class DeepMDAdapter:
    """Build commands for one backend without shared mutable state.

    The adapter owns only backend selection and the immutable-in-use capability
    view.  Callers that have enough scientific context to construct a complete
    :class:`CapabilityKey` must call :meth:`preflight` before building a
    command.  Legacy manager paths intentionally do not invent missing key
    dimensions.
    """

    backend: str
    matrix: CapabilityMatrix = field(default_factory=CapabilityMatrix.load_default)
    allow_experimental: bool = False
    legacy_unchecked: bool = field(default=False, repr=False)

    @classmethod
    def for_legacy_unchecked(
        cls,
        backend: str,
        matrix: CapabilityMatrix | None = None,
        *,
        allow_experimental: bool = False,
    ) -> "DeepMDAdapter":
        """Create the temporary adapter for pre-schema legacy manager paths."""

        return cls(
            backend,
            matrix or CapabilityMatrix.load_default(),
            allow_experimental,
            legacy_unchecked=True,
        )

    def __post_init__(self) -> None:
        if self.backend not in VALID_DP_BACKENDS:
            raise ValueError(
                f"Invalid backend '{self.backend}'. Valid options: {VALID_DP_BACKENDS}"
            )

    @property
    def base_command(self) -> tuple[str, str]:
        return ("dp", f"--{self.backend}")

    def preflight(
        self,
        key_or_matrix: CapabilityKey | CapabilityMatrix,
        key: CapabilityKey | None = None,
        allow_experimental: bool | None = None,
        operation: str | None = None,
    ) -> CapabilityRecord:
        """Authorize one exact capability key for this backend.

        ``preflight(key)`` is the canonical form.  The two-positional form
        ``preflight(matrix, key)`` remains accepted for the initial adapter
        design's call sites while the adapter is being introduced.
        """

        if isinstance(key_or_matrix, CapabilityMatrix):
            if key is None:
                raise TypeError("preflight(matrix, key) requires a capability key")
            matrix = key_or_matrix
            capability_key = key
        else:
            if key is not None:
                raise TypeError("preflight(key) does not accept a second positional key")
            matrix = self.matrix
            capability_key = key_or_matrix

        if capability_key.backend != self.backend:
            raise CapabilityUnavailable(
                f"adapter backend {self.backend} does not match {capability_key.backend}"
            )
        if operation is not None and capability_key.operation != operation:
            raise CapabilityUnavailable(
                f"capability key operation {capability_key.operation!r} does not match "
                f"command operation {operation!r}"
            )
        return matrix.require(
            capability_key,
            allow_experimental=(
                self.allow_experimental
                if allow_experimental is None
                else allow_experimental
            ),
        )

    def _authorize(
        self, operation: str, capability_key: CapabilityKey | None
    ) -> None:
        if self.legacy_unchecked:
            return
        if capability_key is None:
            raise CapabilityUnavailable(
                f"{operation} command requires a complete capability_key"
            )
        self.preflight(capability_key, operation=operation)

    def train(
        self,
        input_file: str,
        finetune_path: str | None = None,
        init_model_path: str | None = None,
        skip_neighbor_stat: bool = False,
        log_file: str | None = None,
        *,
        capability_key: CapabilityKey | None = None,
    ) -> str:
        # DeepMD exposes fine-tuning as a separate policy capability even
        # though both routes use the ``train`` executable subcommand.  Never
        # let a normal-train key silently authorize a fine-tune invocation.
        operation = "fine-tune" if finetune_path else "train"
        self._authorize(operation, capability_key)
        argv = [*self.base_command, "train", input_file]
        if skip_neighbor_stat:
            argv.append("--skip-neighbor-stat")
        if finetune_path:
            argv.extend(["--finetune", finetune_path])
        elif init_model_path:
            argv.extend(["--init-model", init_model_path])
        return _with_log(shlex.join(argv), log_file)

    def freeze(
        self,
        output: str | None = None,
        *,
        capability_key: CapabilityKey | None = None,
    ) -> str:
        self._authorize("freeze", capability_key)
        argv = [*self.base_command, "freeze"]
        if output:
            argv.extend(["-o", output])
        return shlex.join(argv)

    def training_outputs(self, checkpoint_prefix: str = "model.ckpt") -> tuple[str, ...]:
        """Return outputs provable for the installed backend command contract.

        PyTorch freeze output is model-dependent (``.pth`` or ``.pt2``), so its
        stable regular checkpoint is the certifying output. Other backends have
        deterministic default freeze output names in the installed entrypoints.
        """

        if not isinstance(checkpoint_prefix, str) or not checkpoint_prefix.strip():
            raise CapabilityUnavailable("training save_ckpt must be a non-empty string")
        contracts = {
            "pt": (f"{checkpoint_prefix}.pt",),
            "tf": ("frozen_model.pb",),
            "pt-expt": ("frozen_model.pte",),
            "jax": ("frozen_model.hlo",),
            "pd": ("frozen_model.json", "frozen_model.pdiparams"),
        }
        try:
            return contracts[self.backend]
        except KeyError:
            raise CapabilityUnavailable(
                f"no proven training output contract for backend {self.backend!r}"
            ) from None

    def test(
        self,
        model: str,
        system: str,
        prefix: str,
        head: str | None = None,
        log_file: str | None = None,
        *,
        capability_key: CapabilityKey | None = None,
    ) -> str:
        self._authorize("test", capability_key)
        argv = [*self.base_command, "test", "-s", system, "-m", model, "-d", prefix]
        if head:
            argv.extend(["--head", head])
        return _with_log(shlex.join(argv), log_file)

    def eval_desc(
        self,
        model: str,
        system: str,
        output: str,
        head: str | None = None,
        log_file: str | None = None,
        *,
        capability_key: CapabilityKey | None = None,
    ) -> str:
        self._authorize("eval-desc", capability_key)
        argv = [*self.base_command, "eval-desc", "-s", system, "-m", model, "-o", output]
        if head:
            argv.extend(["--head", head])
        return _with_log(shlex.join(argv), log_file)

    def embed(
        self,
        model: str,
        system: str,
        output: str,
        head: str | None = None,
        dtype: str = "fp32",
        log_file: str | None = None,
        *,
        capability_key: CapabilityKey | None = None,
    ) -> str:
        self._authorize("embed", capability_key)
        argv = [
            *self.base_command,
            "embed",
            "-s",
            system,
            "-m",
            model,
            "-o",
            output,
            "--dtype",
            dtype,
        ]
        if head:
            argv.extend(["--head", head])
        return _with_log(shlex.join(argv), log_file)


__all__ = ["DeepMDAdapter"]
