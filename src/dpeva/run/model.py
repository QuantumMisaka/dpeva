"""Explicit model artifact identity and the legacy inference discovery bridge.

The reference is deliberately a small, closed schema.  A reference describes
what a caller intends to execute; it does not download or otherwise resolve a
pretrained alias.  Resolution therefore has to happen before a workflow is
submitted to either local or Slurm execution.
"""

from __future__ import annotations

import hashlib
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, model_validator


class ModelArtifactKind(str, Enum):
    CHECKPOINT = "checkpoint"
    FROZEN = "frozen"
    EXPORTABLE = "exportable"
    PRETRAINED_ALIAS = "pretrained-alias"


class ModelRole(str, Enum):
    REGULAR = "regular"
    EMA = "ema"


class ModelArtifactRef(BaseModel):
    """Machine-readable identity for one model artifact.

    ``supported_operations`` is intentionally producer-declared.  An empty
    list does not mean "all operations"; it means that the reference has not
    established an executable contract and must be rejected by
    :func:`require_operation`.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    kind: ModelArtifactKind
    family: StrictStr
    backend: StrictStr
    path: StrictStr | None = None
    alias: StrictStr | None = None
    resolved_path: StrictStr | None = None
    checksum: StrictStr | None = None
    head: StrictStr | None = None
    role: ModelRole = ModelRole.REGULAR
    deepmd_version: StrictStr | None = None
    producer_run: StrictStr | None = None
    supported_operations: list[StrictStr] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_identity(self) -> "ModelArtifactRef":
        if self.kind is ModelArtifactKind.PRETRAINED_ALIAS:
            if not self.alias:
                raise ValueError("pretrained-alias requires alias")
        elif not self.path and not self.resolved_path:
            raise ValueError(f"{self.kind.value} requires path or resolved_path")

        if self.checksum is not None:
            if len(self.checksum) != 64 or any(
                character not in "0123456789abcdef" for character in self.checksum
            ):
                raise ValueError("checksum must be a lowercase SHA-256 hexadecimal digest")

        if len(set(self.supported_operations)) != len(self.supported_operations):
            raise ValueError("supported_operations must not contain duplicates")
        if any(not operation or not isinstance(operation, str) for operation in self.supported_operations):
            raise ValueError("supported_operations must contain non-empty strings")
        return self


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model_ref(path: Path) -> ModelArtifactRef:
    """Load and strictly validate a JSON model reference."""

    return ModelArtifactRef.model_validate_json(path.read_text(encoding="utf-8"))


def resolve_model_refs(
    work_dir: Path, *, family: str, backend: str
) -> list[ModelArtifactRef]:
    """Discover legacy checkpoint files without assuming contiguous indices.

    This is a compatibility bridge for existing work directories only.  It
    does not infer a concrete model family and assigns the sole operation that
    this bridge is used for (``test``).  Both regular and EMA checkpoints are
    represented independently.
    """

    work_dir = Path(work_dir)
    if not work_dir.is_dir():
        return []

    refs: list[ModelArtifactRef] = []
    model_dirs = sorted(
        (path for path in work_dir.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    )
    for model_dir in model_dirs:
        for filename, role in (
            ("model.ckpt.pt", ModelRole.REGULAR),
            ("model_ema.ckpt.pt", ModelRole.EMA),
        ):
            path = model_dir / filename
            if path.is_file():
                refs.append(
                    ModelArtifactRef(
                        kind=ModelArtifactKind.CHECKPOINT,
                        family=family,
                        backend=backend,
                        path=str(path),
                        checksum=_sha256(path),
                        role=role,
                        supported_operations=["test"],
                    )
                )
    return refs


def require_operation(ref: ModelArtifactRef, operation: str) -> None:
    """Fail closed unless an artifact explicitly supports ``operation``."""

    if not isinstance(operation, str) or not operation:
        raise ValueError("operation must be a non-empty string")
    if ref.kind is ModelArtifactKind.PRETRAINED_ALIAS and not ref.resolved_path:
        raise ValueError("resolve pretrained alias before execution")
    if operation not in ref.supported_operations:
        raise ValueError(f"model artifact does not declare operation: {operation}")
