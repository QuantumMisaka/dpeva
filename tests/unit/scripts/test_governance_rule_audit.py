from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from scripts.audit_governance_rules import audit_rules, build_report, main


def write_rule_registry(root: Path, **overrides: object) -> Path:
    entry = {
        "rule_id": "GATE-TEST",
        "owner": "Test Owner",
        "basis": "test evidence",
        "enforcement_paths": ["scripts/gate.sh"],
        "trigger_paths": ["scripts/gate.sh"],
        "last_reviewed": "2026-09-04",
        "review_interval_days": 90,
    }
    entry.update(overrides)
    path = root / "rules.json"
    path.write_text(json.dumps([entry]), encoding="utf-8")
    return path


def test_overdue_rule_is_reported(tmp_path: Path) -> None:
    registry = write_rule_registry(tmp_path, last_reviewed="2025-01-01")
    findings = audit_rules(registry, repo_root=Path.cwd(), today=date(2026, 9, 4))
    assert findings[0]["reason"] == "review-overdue"


def test_audit_is_report_only() -> None:
    registry = Path("docs/governance/rules.json")
    before = registry.read_bytes()
    audit_rules(registry, Path.cwd(), date.today())
    assert registry.read_bytes() == before


def test_report_has_stable_shape_and_active_count(tmp_path: Path) -> None:
    registry = write_rule_registry(tmp_path)
    report = build_report(registry, repo_root=Path.cwd(), today=date(2026, 9, 4))
    assert list(report) == ["active", "findings", "reviewed_at"]
    assert report["active"] == 1
    assert report["reviewed_at"] == "2026-09-04"


def test_default_mode_reports_without_failing_and_strict_fails(tmp_path: Path, capsys) -> None:
    registry = write_rule_registry(tmp_path, last_reviewed="2025-01-01")
    assert main([str(registry), "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["findings"]
    assert main([str(registry), "--format", "json", "--strict"]) == 1


def test_output_writes_only_explicit_report_file(tmp_path: Path) -> None:
    registry = write_rule_registry(tmp_path)
    output = tmp_path / "report.json"
    assert main([str(registry), "--format", "json", "--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["active"] == 1


def test_schema_requires_exact_fields_and_unique_ids(tmp_path: Path) -> None:
    registry = write_rule_registry(tmp_path, unexpected="value")
    payload = json.loads(registry.read_text(encoding="utf-8"))
    payload.append(dict(payload[0], unexpected=None))
    registry.write_text(json.dumps(payload), encoding="utf-8")
    findings = audit_rules(registry, Path.cwd(), date(2026, 9, 4))
    reasons = {finding["reason"] for finding in findings}
    assert "invalid-schema" in reasons
    assert "duplicate-rule-id" in reasons


def test_schema_rejects_bad_dates_intervals_and_paths(tmp_path: Path) -> None:
    registry = write_rule_registry(
        tmp_path,
        last_reviewed="20260904",
        review_interval_days=True,
        enforcement_paths=["../outside", "/tmp/outside"],
        trigger_paths=["../outside", "/tmp/outside"],
    )
    findings = audit_rules(registry, Path.cwd(), date(2026, 9, 4))
    reasons = {finding["reason"] for finding in findings}
    assert {"invalid-date", "invalid-interval", "invalid-enforcement", "missing-trigger"} <= reasons


def test_schema_rejects_symlink_escape(tmp_path: Path) -> None:
    inside = tmp_path / "inside.py"
    inside.write_text("fixture", encoding="utf-8")
    outside = tmp_path.parent / "outside-governance-rule.py"
    outside.write_text("outside", encoding="utf-8")
    link = tmp_path / "linked.py"
    link.symlink_to(outside)
    registry = write_rule_registry(
        tmp_path,
        enforcement_paths=["linked.py"],
        trigger_paths=["inside.py"],
    )
    findings = audit_rules(registry, repo_root=tmp_path, today=date(2026, 9, 4))
    assert any(finding["reason"] == "invalid-enforcement" for finding in findings)


def test_missing_trigger_evidence_is_reported(tmp_path: Path) -> None:
    registry = write_rule_registry(tmp_path, trigger_paths=[])
    findings = audit_rules(registry, repo_root=Path.cwd(), today=date(2026, 9, 4))
    assert any(finding["reason"] == "missing-trigger" for finding in findings)


def test_rule_cap_is_one_registry_finding_and_strict_only(tmp_path: Path) -> None:
    registry = tmp_path / "rules.json"
    entry = json.loads(write_rule_registry(tmp_path).read_text(encoding="utf-8"))[0]
    registry.write_text(
        json.dumps([dict(entry, rule_id=f"GATE-{index}") for index in range(9)]),
        encoding="utf-8",
    )
    findings = audit_rules(registry, repo_root=Path.cwd(), today=date(2026, 9, 4))
    cap_findings = [finding for finding in findings if finding["reason"] == "too-many-active-rules"]
    assert len(cap_findings) == 1
    assert main([str(registry), "--format", "json"]) == 0
    assert main([str(registry), "--format", "json", "--strict"]) == 1
