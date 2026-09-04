"""Strict, packaged DeepMD capability declarations.

The matrix is deliberately data-driven and small.  A version range identifies
the release lane; it is not a claim that every command in that release is
supported.  Querying a key therefore always returns one exact record, and
``require`` applies the fail-closed status policy before a workflow builds a
command.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, ValidationError, model_validator


CapabilityStatus = Literal[
    "supported",
    "experimental",
    "unsupported",
    "blocked-upstream",
]


class CapabilityKey(BaseModel):
    """The complete identity of one compatibility claim."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    operation: StrictStr = Field(min_length=1)
    backend: StrictStr = Field(min_length=1)
    model_family: StrictStr = Field(min_length=1)
    artifact: StrictStr = Field(min_length=1)
    data_format: StrictStr = Field(min_length=1)
    environment: StrictStr = Field(min_length=1)


class CapabilityEvidence(BaseModel):
    """References to the two evidence layers used for promotion."""

    model_config = ConfigDict(extra="forbid")

    cpu_contract: StrictStr = Field(min_length=1)
    sai_qualification: StrictStr | None = None


class CapabilityRecord(BaseModel):
    """One exact capability status and its verification metadata."""

    model_config = ConfigDict(extra="forbid")

    key: CapabilityKey
    status: CapabilityStatus
    version_range: StrictStr = Field(min_length=1)
    verification_command: StrictStr = Field(min_length=1)
    evidence_ref: CapabilityEvidence | None = None
    upstream_issue: StrictStr | None = None

    @model_validator(mode="after")
    def validate_status_metadata(self) -> "CapabilityRecord":
        if self.status == "blocked-upstream" and not self.upstream_issue:
            raise ValueError("blocked-upstream capability requires upstream_issue")
        if self.status != "blocked-upstream" and self.upstream_issue is not None:
            raise ValueError("upstream_issue is only valid for blocked-upstream capability")
        return self


class _CapabilityManifest(BaseModel):
    """Private on-disk envelope; unknown schema fields are never ignored."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    deepmd_version: Literal["3.2"] = "3.2"
    records: list[CapabilityRecord]

    @model_validator(mode="after")
    def validate_records(self) -> "_CapabilityManifest":
        keys = [_key_identity(record.key) for record in self.records]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate capability key")
        if not self.records:
            raise ValueError("capability manifest must contain records")
        ranges = {record.version_range for record in self.records}
        if len(ranges) != 1:
            raise ValueError("capability records must use one version range")
        return self


def _key_identity(key: CapabilityKey) -> tuple[str, ...]:
    return (
        key.operation,
        key.backend,
        key.model_family,
        key.artifact,
        key.data_format,
        key.environment,
    )


class CapabilityUnavailable(RuntimeError):
    """Raised when a capability is absent or not authorized for execution."""


class CapabilityMatrix:
    """Read-only view of a versioned capability manifest."""

    VERSION_RANGE = ">=3.2,<3.3"

    def __init__(self, manifest: _CapabilityManifest) -> None:
        self._manifest = manifest
        self.records = tuple(manifest.records)
        self._by_key = {_key_identity(record.key): record for record in self.records}

    @classmethod
    def from_payload(cls, payload: object) -> "CapabilityMatrix":
        """Validate a decoded manifest payload and build an immutable query view."""

        manifest = _CapabilityManifest.model_validate(payload)
        if {record.version_range for record in manifest.records} != {cls.VERSION_RANGE}:
            raise ValueError(f"capability version_range must be {cls.VERSION_RANGE!r}")
        return cls(manifest)

    @classmethod
    def load(cls, path: str | Path) -> "CapabilityMatrix":
        """Load and strictly validate a manifest from an explicit JSON path."""

        manifest_path = Path(path)
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot load capability manifest {manifest_path}: {exc}") from exc
        return cls.from_payload(payload)

    @classmethod
    def load_default(cls) -> "CapabilityMatrix":
        """Load the packaged default without depending on the current directory."""

        resource = resources.files("dpeva.compatibility").joinpath("deepmd-3.2.json")
        try:
            payload = json.loads(resource.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot load packaged capability manifest: {exc}") from exc
        return cls.from_payload(payload)

    def get(self, key: CapabilityKey) -> CapabilityRecord:
        """Return the one exact record for ``key`` or fail closed."""

        if not isinstance(key, CapabilityKey):
            raise TypeError("key must be a CapabilityKey")
        record = self._by_key.get(_key_identity(key))
        if record is None:
            raise CapabilityUnavailable(f"no capability record for key {key.model_dump()}")
        return record

    def require(
        self, key: CapabilityKey, *, allow_experimental: bool = False
    ) -> CapabilityRecord:
        """Authorize a key according to its manifest status."""

        record = self.get(key)
        if record.status == "experimental" and not allow_experimental:
            raise CapabilityUnavailable(
                f"capability is experimental; explicit allow_experimental is required: {key.model_dump()}"
            )
        if record.status in {"unsupported", "blocked-upstream"}:
            suffix = f" ({record.upstream_issue})" if record.upstream_issue else ""
            raise CapabilityUnavailable(
                f"capability is {record.status}{suffix}: {key.model_dump()}"
            )
        return record


__all__ = [
    "CapabilityEvidence",
    "CapabilityKey",
    "CapabilityMatrix",
    "CapabilityRecord",
    "CapabilityUnavailable",
]
