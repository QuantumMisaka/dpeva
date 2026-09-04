from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import dpeva.cli as cli
from dpeva.config_migration import MigrationResult, MigrationWarning
from dpeva.run.context import RunContext, RunOptions
from dpeva.run.doctor import DoctorCheck, build_doctor_report
from dpeva.run.status import RunState


def test_migration_result_keeps_exact_raw_input_and_schema_metadata() -> None:
    raw = {"backend": "local", "data_path": "data"}
    result = MigrationResult(
        normalized={"submission": {"backend": "local"}, "data_path": "data"},
        warnings=(MigrationWarning(field="backend", replacement="submission.backend"),),
        original=raw,
        input_schema_version="1.0",
    )

    assert result.original == raw
    assert result.input_schema_version == "1.0"


def test_cli_config_loader_reads_source_once(monkeypatch, tmp_path: Path) -> None:
    calls = {"count": 0}
    raw = {"data_path": "data", "backend": "local"}

    def load_once(_path):
        calls["count"] += 1
        return raw

    monkeypatch.setattr(cli, "load_json_config", load_once)
    monkeypatch.setattr(cli, "resolve_config_paths", lambda mapping, _path: mapping)

    result = cli.load_and_resolve_config(str(tmp_path / "config.json"))

    assert calls["count"] == 1
    assert result.original == raw
    assert result.normalized["submission"]["backend"] == "local"


def test_context_persists_config_metadata_reference_and_payload(tmp_path: Path) -> None:
    context = RunContext.create(
        tmp_path,
        "feature",
        RunOptions(run_id="metadata"),
        {"backend": "local"},
        {"submission": {"backend": "local"}},
        config_metadata={
            "schema_version": "1.0",
            "input_schema_version": "1.0",
            "migration_warnings": [],
        },
    )

    manifest = json.loads((context.run_dir / "run.json").read_text())
    assert manifest["config"]["metadata"] == "config.metadata.json"
    assert "environment" not in manifest
    assert json.loads((context.run_dir / "config.metadata.json").read_text())["schema_version"] == "1.0"


def test_force_archives_config_metadata_reference(tmp_path: Path) -> None:
    initial = RunContext.create(
        tmp_path,
        "feature",
        RunOptions(run_id="metadata-force"),
        {"x": 1},
        {"x": 1},
        config_metadata={"schema_version": "1.0", "input_schema_version": "1.0"},
    )
    previous = json.loads((initial.run_dir / "run.json").read_text())
    forced = RunContext.create(
        tmp_path,
        "feature",
        RunOptions(run_id="metadata-force", force=True, reason="metadata retry"),
        {"x": 2},
        {"x": 2},
        config_metadata={"schema_version": "1.0", "input_schema_version": "1.0"},
    )

    archive = json.loads((forced.run_dir / "attempts/attempt-0001.json").read_text())
    assert archive["config"]["metadata"] == previous["config"]["metadata"]
    assert (forced.run_dir / forced.recorder.manifest.config["metadata"]).is_file()


def test_resume_submitted_run_rejects_before_new_context(tmp_path: Path) -> None:
    context = RunContext.create(tmp_path, "feature", RunOptions(run_id="submitted"), {}, {})
    context.recorder.transition(RunState.VALIDATED)
    context.recorder.transition(RunState.SUBMITTED)

    with pytest.raises(ValueError, match="submitted.*scheduler"):
        RunContext.create(
            tmp_path,
            "feature",
            RunOptions(run_id="submitted", resume=True),
            {},
            {},
        )


def test_doctor_keeps_optional_hardware_failure_out_of_required_status() -> None:
    report = build_doctor_report(
        checks=[
            DoctorCheck(name="deepmd", status="ok", version="3.2.0", detail="ok"),
            DoctorCheck(name="deepmd.cli.test", status="ok", detail="ok"),
            DoctorCheck(name="cuda", status="missing", detail="not installed", required=False),
        ]
    )

    assert report.status == "ok"


def test_input_identity_is_relative_and_dataset_identity_is_structural(tmp_path: Path) -> None:
    dataset = tmp_path / "data"
    dataset.mkdir()
    (dataset / "type.raw").write_text("0\n")
    model = tmp_path / "model.pt"
    model.write_bytes(b"model")
    context = RunContext.create(
        tmp_path,
        "feature",
        RunOptions(run_id="inputs"),
        {},
        {},
        inputs=[
            {"kind": "dataset", "ref": "data", "identity": "structural-sha256:abc", "identity_scope": "bounded-structural"},
            {"kind": "model", "ref": "model.pt", "identity": "sha256:def", "identity_scope": "full-content"},
        ],
    )

    payload = json.loads((context.run_dir / "run.json").read_text())
    assert all(not value.startswith("/") for item in payload["inputs"] for value in item.values())
    assert payload["inputs"][0]["identity_scope"] == "bounded-structural"


def test_doctor_default_probes_required_operation_surfaces(monkeypatch) -> None:
    calls: list[list[str]] = []

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr("dpeva.run.doctor._probe_python_package", lambda *a, **k: DoctorCheck(name=a[0], status="ok", detail="ok", required=k.get("required", True)))
    report = build_doctor_report(run=run, include_optional=False)

    names = {check.name for check in report.checks}
    assert {"deepmd", "deepmd.cli.test", "deepmd.cli.eval-desc", "deepmd.cli.embed"} <= names
    assert ["dp", "test", "-h"] in calls
    assert ["dp", "eval-desc", "-h"] in calls
    assert ["dp", "embed", "-h"] in calls
