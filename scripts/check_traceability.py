#!/usr/bin/env python3
"""Validate the repository's capability-to-evidence traceability registry.

The registry is deliberately a path/schema check.  It does not inspect source
text or attempt to infer whether a capability is implemented or scientifically
qualified.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


_ENTRY_FIELDS = frozenset(
    {
        "capability_id",
        "code_paths",
        "test_paths",
        "documentation_paths",
        "owner",
        "evidence_path",
    }
)
_PATH_FIELDS = ("code_paths", "test_paths", "documentation_paths")


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _path_failure(
    value: Any,
    *,
    field: str,
    capability_id: str,
    repo_root: Path,
) -> str | None:
    """Return a failure for one repository-relative file reference."""

    label = f"{field}: {capability_id}: {value!r}"
    if not _non_empty_string(value):
        return f"invalid {label}: expected a non-empty path"

    relative = Path(value)
    if relative.is_absolute():
        return f"invalid {label}: path must be repository-relative"
    # Reject lexical traversal even when it happens to resolve back inside the
    # repository.  This keeps the manifest readable and prevents ambiguity.
    if ".." in relative.parts:
        return f"invalid {label}: lexical '..' escape"

    root = repo_root.resolve()
    candidate = repo_root / relative
    try:
        resolved = candidate.resolve(strict=False)
        resolved.relative_to(root)
    except (OSError, ValueError):
        return f"invalid {label}: resolved path escapes repository"
    if not candidate.exists():
        return f"invalid {label}: path does not exist"
    if not candidate.is_file():
        return f"invalid {label}: expected a file"
    return None


def validate_traceability(path: Path, repo_root: Path) -> list[str]:
    """Return all schema and path failures in a capability registry.

    The function reports malformed input as findings instead of raising so it
    can be used by both a CI gate and focused unit tests.  Every path is
    checked as a file inside ``repo_root`` after both lexical and resolved
    containment checks; symlinks pointing outside the repository are rejected.
    """

    failures: list[str] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [f"cannot read registry {path}: {exc}"]

    if not isinstance(payload, list) or not payload:
        return ["registry root must be a non-empty JSON array"]

    seen: set[str] = set()
    for index, entry in enumerate(payload):
        prefix = f"entry[{index}]"
        if not isinstance(entry, dict):
            failures.append(f"{prefix} must be an object")
            continue

        unknown = set(entry) - _ENTRY_FIELDS
        missing = _ENTRY_FIELDS - set(entry)
        if unknown:
            failures.append(f"{prefix} has unknown field(s): {', '.join(sorted(unknown))}")
        if missing:
            failures.append(f"{prefix} is missing field(s): {', '.join(sorted(missing))}")
        if unknown or missing:
            # A malformed entry cannot be checked safely by the field loop.
            continue

        capability_id = entry["capability_id"]
        if not _non_empty_string(capability_id):
            failures.append(f"{prefix} capability_id must be a non-empty string")
            capability_name = f"entry[{index}]"
        else:
            capability_name = capability_id
            if capability_id in seen:
                failures.append(f"duplicate capability_id: {capability_id}")
            seen.add(capability_id)

        if not _non_empty_string(entry["owner"]):
            failures.append(f"missing owner: {capability_name}")

        for field in _PATH_FIELDS:
            values = entry[field]
            if not isinstance(values, list) or not values:
                failures.append(f"{field} must be a non-empty array: {capability_name}")
                continue
            for value in values:
                failure = _path_failure(
                    value,
                    field=field,
                    capability_id=capability_name,
                    repo_root=repo_root,
                )
                if failure:
                    failures.append(failure)

        evidence_failure = _path_failure(
            entry["evidence_path"],
            field="evidence_path",
            capability_id=capability_name,
            repo_root=repo_root,
        )
        if evidence_failure:
            failures.append(evidence_failure)

    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "registry",
        nargs="?",
        default="docs/governance/traceability/capability-evidence.json",
        help="repository-relative capability registry",
    )
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parent.parent
    registry = Path(args.registry)
    if not registry.is_absolute():
        registry = repo_root / registry
    failures = validate_traceability(registry, repo_root=repo_root)
    if failures:
        for failure in failures:
            print(f"traceability: {failure}", file=sys.stderr)
        return 1
    print(f"traceability check passed: {registry.relative_to(repo_root)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
