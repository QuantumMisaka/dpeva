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

from pydantic import BaseModel, ConfigDict, Field, StrictStr, model_validator


CapabilityStatus = Literal[
    "supported",
    "experimental",
    "unsupported",
    "blocked-upstream",
]
EvidenceKind = Literal["cpu-contract", "sai-v100-qualification"]
VerificationStatus = Literal["implemented", "planned", "blocked"]
SAI_VERIFICATION_CASES = frozenset(
    {
        "preflight", "pip-freeze", "deepmd-version", "torch-cuda", "gpu",
        "pt-test", "pt-test-ema", "pt-eval-desc", "pt-eval-desc-ema",
        "pt-embed", "pt-embed-ema", "dpa4c-periodic-eval-desc",
    }
)


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

    model_config = ConfigDict(extra="forbid", frozen=True)

    cpu_contract: StrictStr | None = Field(default=None, min_length=1)
    sai_qualification: StrictStr | None = None


class CapabilityRecord(BaseModel):
    """One exact capability status and its verification metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: CapabilityKey
    status: CapabilityStatus
    version_range: StrictStr = Field(min_length=1)
    verification_command: StrictStr | None = Field(default=None, min_length=1)
    required_evidence: tuple[EvidenceKind, ...] = ()
    verification_status: VerificationStatus = "implemented"
    evidence_ref: CapabilityEvidence | None = None
    upstream_issue: StrictStr | None = None
    covered_roles: tuple[Literal["regular", "ema"], ...] | None = None
    sai_verification_cases: tuple[StrictStr, ...] | None = None

    @model_validator(mode="after")
    def validate_status_metadata(self) -> "CapabilityRecord":
        if self.verification_status in {"planned", "blocked"} and self.verification_command is not None:
            raise ValueError("planned/blocked capability must not declare verification_command")
        if self.verification_status == "implemented" and not self.verification_command:
            raise ValueError("implemented capability requires verification_command")
        if len(self.required_evidence) != len(set(self.required_evidence)):
            raise ValueError("required_evidence must not contain duplicates")
        if self.sai_verification_cases is not None:
            if not self.sai_verification_cases:
                raise ValueError("sai_verification_cases must not be empty")
            if any(case not in SAI_VERIFICATION_CASES for case in self.sai_verification_cases):
                raise ValueError("sai_verification_cases contains an unknown qualification case")
            if len(set(self.sai_verification_cases)) != len(self.sai_verification_cases):
                raise ValueError("sai_verification_cases must not contain duplicates")
        if self.status == "blocked-upstream" and not self.upstream_issue:
            raise ValueError("blocked-upstream capability requires upstream_issue")
        if self.status != "blocked-upstream" and self.upstream_issue is not None:
            raise ValueError("upstream_issue is only valid for blocked-upstream capability")
        if self.key.operation == "candidate-evaluation":
            if self.covered_roles != ("regular", "ema"):
                raise ValueError(
                    "candidate-evaluation must explicitly cover regular and ema roles"
                )
        elif self.covered_roles is not None:
            raise ValueError("covered_roles is only valid for candidate-evaluation")
        if self.status == "supported" and self.verification_status != "implemented":
            raise ValueError("supported capability requires implemented verification")
        if self.status == "supported":
            if not self.required_evidence:
                raise ValueError("supported capability requires required_evidence")
            if "sai-v100-qualification" in self.required_evidence and not self.sai_verification_cases:
                raise ValueError("supported SAI capability requires sai_verification_cases")
            missing = [kind for kind in self.required_evidence if not self.evidence_ref]
            if missing:
                raise ValueError("supported capability requires evidence_ref")
            if self.evidence_ref:
                for kind in self.required_evidence:
                    if kind == "cpu-contract" and not self.evidence_ref.cpu_contract:
                        raise ValueError("supported capability requires CPU evidence reference")
                    if kind == "sai-v100-qualification" and not self.evidence_ref.sai_qualification:
                        raise ValueError("supported capability requires SAI evidence reference")
        return self


class _CapabilityManifest(BaseModel):
    """Private on-disk envelope; unknown schema fields are never ignored."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    deepmd_version: Literal["3.2"] = "3.2"
    records: tuple[CapabilityRecord, ...]

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


def _evidence_path(reference: str, repo_root: Path) -> Path | None:
    """Resolve a repository-local JSON evidence reference, fail closed."""

    if not reference or "#" in reference or "://" in reference:
        return None
    path = (repo_root / reference).resolve()
    try:
        path.relative_to(repo_root.resolve())
    except ValueError:
        return None
    return path if path.suffix == ".json" and path.is_file() else None


def _read_json_evidence(reference: str | None, repo_root: Path) -> dict[str, object] | None:
    if not reference:
        return None
    path = _evidence_path(reference, repo_root)
    if path is None:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _evidence_matches(record: CapabilityRecord, payload: dict[str, object], *, sai: bool) -> bool:
    # Parse through the one producer schema.  A qualification aggregate is
    # accepted only when it contains validated per-capability attestations;
    # command records and prose reports are deliberately insufficient.
    from .attestation import CapabilityAttestation

    is_aggregate = "attestations" in payload
    if is_aggregate:
        if (
            payload.get("schema_version") != "1.0"
            or payload.get("status") != "finished"
        ):
            return False
        if sai:
            if payload.get("qualification") != "deepmd-3.2-sai-v100":
                return False
            aggregate_job = payload.get("job_id")
            aggregate_gpu = payload.get("gpu")
            if not isinstance(aggregate_job, (int, str)) or isinstance(aggregate_job, bool) or not str(aggregate_job).isdigit():
                return False
            if not isinstance(aggregate_gpu, str) or "v100" not in aggregate_gpu.lower():
                return False
        raw = payload.get("attestations")
        if not isinstance(raw, list):
            return False
        candidates = raw
    else:
        candidates = [payload]
    expected_source = "sai-v100-qualification" if sai else "cpu-contract"
    if is_aggregate and not record.sai_verification_cases and sai:
        return False
    matches = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        try:
            attestation = CapabilityAttestation.model_validate(candidate)
        except Exception:
            return False
        if is_aggregate and sai and (
            str(attestation.job_id) != str(payload.get("job_id"))
            or attestation.gpu != payload.get("gpu")
        ):
            return False
        if (
            attestation.source == expected_source
            and attestation.capability_key == record.key
            and attestation.verification_command == record.verification_command
        ):
            matches.append(attestation)
    if len(matches) != 1 and not (sai and len(matches) == len(record.sai_verification_cases or ())):
        return False
    if sai:
        expected_cases = tuple(record.sai_verification_cases or ())
        observed_cases = tuple(attestation.case for attestation in matches)
        return len(matches) == len(expected_cases) and len(set(observed_cases)) == len(observed_cases) and set(observed_cases) == set(expected_cases)
    return len(matches) == 1


def validate_promotion_evidence(record: CapabilityRecord, repo_root: str | Path) -> bool:
    """Return whether one record has exact, repository-local promotion evidence.

    This is intentionally a pure read-only gate.  It never changes the
    manifest, accepts no report anchors, and requires a separate JSON object
    for each declared evidence kind.
    """

    if not isinstance(record, CapabilityRecord):
        return False
    if record.verification_status != "implemented" or not record.verification_command:
        return False if record.status == "supported" else True
    if record.status != "supported":
        return True
    refs = record.evidence_ref
    if refs is None:
        return False
    root = Path(repo_root).expanduser().resolve()
    for kind in record.required_evidence:
        reference = refs.cpu_contract if kind == "cpu-contract" else refs.sai_qualification
        payload = _read_json_evidence(reference, root)
        if payload is None or not _evidence_matches(record, payload, sai=kind == "sai-v100-qualification"):
            return False
    return True

__all__ = [
    "CapabilityEvidence",
    "CapabilityKey",
    "CapabilityMatrix",
    "CapabilityRecord",
    "CapabilityUnavailable",
    "validate_promotion_evidence",
]
