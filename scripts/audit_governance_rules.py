#!/usr/bin/env python3
"""Report stale or unenforced governance rules without changing policy files.

The registry is intentionally small and describes enforcement mechanisms, not
the complete governance specification.  Normal audits are informational;
``--strict`` is reserved for an explicit release review.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any


_FIELDS = frozenset(
    {
        "rule_id",
        "owner",
        "basis",
        "enforcement_paths",
        "last_reviewed",
        "review_interval_days",
    }
)
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DEFAULT_REGISTRY = "docs/governance/rules.json"


def _finding(rule_id: str, reason: str, detail: str) -> dict[str, str]:
    return {"rule_id": rule_id, "reason": reason, "detail": detail}


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _valid_repo_file(value: Any, repo_root: Path) -> bool:
    if not _non_empty_string(value):
        return False
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        return False
    root = repo_root.resolve()
    candidate = repo_root / relative
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError):
        return False
    return resolved.is_file()


def _parse_date(value: Any) -> date | None:
    if not isinstance(value, str) or not _ISO_DATE.fullmatch(value):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _read_registry(path: Path) -> tuple[list[Any], list[dict[str, str]]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [], [_finding("<registry>", "invalid-registry", str(exc))]
    if not isinstance(payload, list) or not payload:
        return [], [_finding("<registry>", "invalid-schema", "root must be a non-empty array")]
    return payload, []


def _audit_record(
    record: Any,
    index: int,
    seen: set[str],
    repo_root: Path,
    today: date,
) -> list[dict[str, str]]:
    label = f"<entry:{index}>"
    if not isinstance(record, dict):
        return [_finding(label, "invalid-schema", "record must be an object")]

    findings: list[dict[str, str]] = []
    rule_id = record.get("rule_id")
    if _non_empty_string(rule_id):
        rule_id = rule_id.strip()
        if rule_id in seen:
            findings.append(_finding(rule_id, "duplicate-rule-id", "rule_id must be unique"))
        seen.add(rule_id)
        label = rule_id
    else:
        findings.append(_finding(label, "invalid-rule-id", "rule_id must be non-empty"))

    unknown = sorted(set(record) - _FIELDS)
    missing = sorted(_FIELDS - set(record))
    if unknown or missing:
        details: list[str] = []
        if unknown:
            details.append(f"unknown fields: {', '.join(unknown)}")
        if missing:
            details.append(f"missing fields: {', '.join(missing)}")
        findings.append(_finding(label, "invalid-schema", "; ".join(details)))
        return findings

    if not _non_empty_string(record["owner"]):
        findings.append(_finding(label, "missing-owner", "owner must be non-empty"))
    if not _non_empty_string(record["basis"]):
        findings.append(_finding(label, "missing-basis", "basis must be non-empty"))

    paths = record["enforcement_paths"]
    valid_paths = isinstance(paths, list) and bool(paths)
    if not valid_paths:
        findings.append(
            _finding(label, "missing-enforcement", "enforcement_paths must be a non-empty array")
        )
    else:
        for path in paths:
            if not _valid_repo_file(path, repo_root):
                findings.append(
                    _finding(
                        label,
                        "invalid-enforcement",
                        f"path is not a contained file: {path!r}",
                    )
                )

    reviewed = _parse_date(record["last_reviewed"])
    if reviewed is None:
        findings.append(
            _finding(label, "invalid-date", "last_reviewed must be strict YYYY-MM-DD")
        )

    interval = record["review_interval_days"]
    if isinstance(interval, bool) or not isinstance(interval, int) or interval <= 0:
        findings.append(
            _finding(label, "invalid-interval", "review_interval_days must be a positive integer")
        )

    valid_interval = isinstance(interval, int) and not isinstance(interval, bool) and interval > 0
    if reviewed is not None and valid_interval:
        deadline = reviewed + timedelta(days=interval)
        if today > deadline:
            findings.append(_finding(label, "review-overdue", deadline.isoformat()))
    return findings


def _audit(registry: Path, repo_root: Path, today: date) -> tuple[int, list[dict[str, str]]]:
    records, findings = _read_registry(registry)
    seen: set[str] = set()
    for index, record in enumerate(records):
        findings.extend(_audit_record(record, index, seen, repo_root, today))
    return len(records), findings


def audit_rules(registry: Path, repo_root: Path, today: date) -> list[dict[str, str]]:
    """Return deterministic findings while leaving the registry untouched."""

    _active, findings = _audit(Path(registry), Path(repo_root), today)
    return findings


def build_report(registry: Path, repo_root: Path, today: date) -> dict[str, Any]:
    """Build the stable JSON report consumed by the quarterly workflow."""

    active, findings = _audit(Path(registry), Path(repo_root), today)
    return {"active": active, "findings": findings, "reviewed_at": today.isoformat()}


def _render_text(report: dict[str, Any]) -> str:
    lines = [f"active: {report['active']}", f"reviewed_at: {report['reviewed_at']}"]
    if report["findings"]:
        lines.append("findings:")
        lines.extend(
            f"- {item['rule_id']}: {item['reason']}: {item['detail']}"
            for item in report["findings"]
        )
    else:
        lines.append("findings: none")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registry", nargs="?", default=_DEFAULT_REGISTRY)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--output", type=Path, help="write the report to this existing directory")
    parser.add_argument("--strict", action="store_true", help="return 1 when findings exist")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parent.parent
    registry = Path(args.registry)
    if not registry.is_absolute():
        registry = repo_root / registry
    report = build_report(registry, repo_root, date.today())
    if args.format == "json":
        rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    else:
        rendered = _render_text(report)

    if args.output is not None:
        output = args.output
        if output.resolve(strict=False) == registry.resolve(strict=False):
            parser.error("--output must not overwrite the governance registry")
        if not output.parent.is_dir():
            parser.error(f"output directory does not exist: {output.parent}")
        try:
            output.write_text(rendered, encoding="utf-8")
        except OSError as exc:
            parser.error(f"cannot write report: {exc}")
    else:
        sys.stdout.write(rendered)
    return 1 if args.strict and report["findings"] else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
