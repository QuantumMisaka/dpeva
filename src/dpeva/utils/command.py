"""Deprecated DeepMD command facade with the historical API preserved."""

from __future__ import annotations

from typing import ClassVar, Optional
import warnings

from dpeva.compatibility.adapter import DeepMDAdapter
from dpeva.constants import DEFAULT_DP_BACKEND, VALID_DP_BACKENDS


class DPCommandBuilder:
    """Compatibility facade for the original stateful command builder.

    New code should inject an independent :class:`DeepMDAdapter`.  The
    optional keyword-only ``backend`` arguments are a stateless bridge for
    callers migrating away from the historical ``set_backend`` state.
    """

    DEFAULT_BACKEND: ClassVar[str] = DEFAULT_DP_BACKEND
    VALID_BACKENDS: ClassVar[tuple[str, ...]] = tuple(VALID_DP_BACKENDS)
    _backend: ClassVar[str] = DEFAULT_BACKEND

    @classmethod
    def set_backend(cls, backend: str) -> None:
        """Set the backend used by calls that omit ``backend``."""
        if backend not in cls.VALID_BACKENDS:
            raise ValueError(
                f"Invalid backend '{backend}'. Valid options: {cls.VALID_BACKENDS}"
            )
        cls._backend = backend

    @classmethod
    def _resolve_backend(cls, backend: Optional[str]) -> str:
        return cls._backend if backend is None else backend

    @classmethod
    def _adapter(cls, backend: Optional[str] = None) -> DeepMDAdapter:
        return DeepMDAdapter.for_legacy_unchecked(cls._resolve_backend(backend))

    @staticmethod
    def _warn_deprecated() -> None:
        warnings.warn(
            "DPCommandBuilder is deprecated; inject DeepMDAdapter instead",
            DeprecationWarning,
            stacklevel=3,
        )

    @classmethod
    def _get_base_cmd(cls, backend: Optional[str] = None) -> str:
        return " ".join(cls._adapter(backend).base_command)

    @classmethod
    def train(
        cls,
        input_file: str,
        finetune_path: Optional[str] = None,
        init_model_path: Optional[str] = None,
        skip_neighbor_stat: bool = False,
        log_file: Optional[str] = None,
        *,
        backend: Optional[str] = None,
    ) -> str:
        cls._warn_deprecated()
        return cls._adapter(backend).train(
            input_file, finetune_path, init_model_path, skip_neighbor_stat, log_file
        )

    @classmethod
    def freeze(cls, output: Optional[str] = None, *, backend: Optional[str] = None) -> str:
        cls._warn_deprecated()
        return cls._adapter(backend).freeze(output)

    @classmethod
    def eval_desc(
        cls,
        model: str,
        system: str,
        output: str,
        head: Optional[str] = None,
        log_file: Optional[str] = None,
        *,
        backend: Optional[str] = None,
    ) -> str:
        cls._warn_deprecated()
        return cls._adapter(backend).eval_desc(model, system, output, head, log_file)

    @classmethod
    def embed(
        cls,
        model: str,
        system: str,
        output: str,
        head: Optional[str] = None,
        dtype: str = "fp32",
        log_file: Optional[str] = None,
        *,
        backend: Optional[str] = None,
    ) -> str:
        cls._warn_deprecated()
        return cls._adapter(backend).embed(model, system, output, head, dtype, log_file)

    @classmethod
    def test(
        cls,
        model: str,
        system: str,
        prefix: str,
        head: Optional[str] = None,
        log_file: Optional[str] = None,
        *,
        backend: Optional[str] = None,
    ) -> str:
        cls._warn_deprecated()
        return cls._adapter(backend).test(model, system, prefix, head, log_file)
