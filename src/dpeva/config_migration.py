"""Explicit migration of legacy configuration fields.

Migration is deliberately separate from Pydantic validation.  It accepts only
the legacy aliases that have a documented destination and returns a deep copy,
so callers can retain the exact source JSON for run evidence.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MigrationWarning:
    """A legacy field accepted for one migration boundary."""

    field: str
    replacement: str
    removal_version: str = "1.0"


@dataclass(frozen=True)
class MigrationResult:
    """Normalized configuration and the compatibility warnings it generated."""

    normalized: dict[str, Any]
    warnings: tuple[MigrationWarning, ...]


_FLAT_SUBMISSION_KEYS = (
    "backend",
    "slurm_config",
    "env_setup",
    "slurm_array",
    "slurm_array_task_limit",
)


def migrate_legacy_config(raw: dict[str, Any]) -> MigrationResult:
    """Move supported top-level submission aliases under ``submission``.

    The input mapping and all nested values are copied before any changes are
    made.  When both forms are supplied, equal values are accepted once with
    a warning; differing values are rejected instead of silently choosing one.
    Unknown fields are intentionally left for strict Pydantic models to reject.
    """

    normalized = deepcopy(raw)
    warnings: list[MigrationWarning] = []
    has_nested_submission = "submission" in normalized
    submission = normalized.get("submission", {})
    if not isinstance(submission, dict):
        if any(key in normalized for key in _FLAT_SUBMISSION_KEYS):
            raise ValueError("submission must be an object when legacy fields are present")
        return MigrationResult(normalized=normalized, warnings=())

    submission = deepcopy(submission)
    for key in _FLAT_SUBMISSION_KEYS:
        if key not in normalized:
            continue
        # ``backend`` is also a workflow-native field for exploration
        # (currently ``atst-tools``).  Only the two historical submission
        # backend values are unambiguous aliases.
        if key == "backend" and normalized[key] not in {"local", "slurm"}:
            continue
        value = normalized.pop(key)
        if key in submission and submission[key] != value:
            raise ValueError(f"conflicting legacy field '{key}' and submission.{key}")
        submission[key] = value
        warnings.append(
            MigrationWarning(field=key, replacement=f"submission.{key}")
        )

    if has_nested_submission or submission:
        normalized["submission"] = submission
    return MigrationResult(normalized=normalized, warnings=tuple(warnings))
