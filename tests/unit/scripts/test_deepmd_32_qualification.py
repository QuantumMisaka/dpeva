"""Local fail-closed tests for the DeepMD 3.2 qualification harness."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.validation.collect_deepmd_32_qualification import collect_qualification
from scripts.validation.prepare_deepmd_32_qualification import prepare
from scripts.validation.submit_deepmd_32_qualification import parse_job_id, submit


def _write_command_result(root: Path, case: str, *, returncode: int = 0, artifacts: list[str] | None = None) -> None:
    checks = [{"path": str(root / artifact), "exists": True} for artifact in artifacts or []]
    path = root / "commands" / f"{case}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"case": case, "returncode": returncode, "status": "finished" if returncode == 0 else "failed", "artifact_checks": checks}), encoding="utf-8")


def _complete_environment(root: Path) -> None:
    environment = root / "environment"
    environment.mkdir(parents=True, exist_ok=True)
    for name, content in {"pip-freeze.json": "{}\n", "deepmd-version.json": "{}\n", "torch-cuda.json": "{}\n", "gpu.json": "{}\n"}.items():
        (environment / name).write_text(content, encoding="utf-8")


def test_collector_refuses_partial_success(tmp_path: Path) -> None:
    _write_command_result(tmp_path, "pt-test", artifacts=["results.e.out"])
    _write_command_result(tmp_path, "dpa4c-periodic-eval-desc", returncode=1)
    (tmp_path / "submission.json").write_text('{"job_id":"123"}', encoding="utf-8")
    report = collect_qualification(tmp_path, job_id="123", gpu="Tesla V100-SXM2-32GB")
    assert report["status"] == "failed"
    assert report["job_id"] == "123"
    assert "dpa4c-periodic-eval-desc" in report["failed_commands"]


def test_collector_rejects_mutable_latest(tmp_path: Path) -> None:
    latest = tmp_path / "latest"
    latest.mkdir()
    with pytest.raises(ValueError, match="mutable"):
        collect_qualification(latest)


def test_prepare_records_models_without_copying(tmp_path: Path) -> None:
    model_root = tmp_path / "models"
    model_root.mkdir()
    (model_root / "model.ckpt.pt").write_bytes(b"regular")
    (model_root / "model_ema.ckpt.pt").write_bytes(b"ema")
    output = tmp_path / "build" / "input.json"
    payload = prepare(model_root, output)
    assert payload["fixture"]["type_map"] == ["Fe", "C", "H", "O"]
    assert payload["fixture"]["periodic"] is True
    assert Path(payload["models"]["regular"]["path"]) == model_root / "model.ckpt.pt"
    assert not (output.parent / "input" / "model.ckpt.pt").exists()
    with pytest.raises(FileExistsError):
        prepare(model_root, output)


def test_job_id_parser_is_fail_closed() -> None:
    assert parse_job_id("Submitted batch job 123\n") == "123"
    with pytest.raises(ValueError):
        parse_job_id("sbatch: error: partition 4V100 invalid")
    with pytest.raises(ValueError):
        parse_job_id("Submitted batch job 1\nSubmitted batch job 2")


def test_submit_dry_run_writes_immutable_reference(tmp_path: Path) -> None:
    model_root = tmp_path / "models"
    model_root.mkdir()
    for name in ("model.ckpt.pt", "model_ema.ckpt.pt"):
        (model_root / name).write_bytes(name.encode())
    input_path = tmp_path / "input.json"
    prepare(model_root, input_path)
    slurm = tmp_path / "job.slurm"
    slurm.write_text("#!/bin/bash\n", encoding="utf-8")
    ref = tmp_path / "latest.json"
    result = submit(input_path, slurm, ref, job_root=tmp_path / "external", dry_run=True)
    assert result["status"] == "dry-run"
    assert json.loads(ref.read_text(encoding="utf-8"))["job_dir"] != str(tmp_path / "external" / "latest")
    assert (Path(result["job_dir"]) / "submission.json").is_file()


def test_recorded_runner_argv_and_missing_artifact(tmp_path: Path) -> None:
    config = tmp_path / "input.json"
    config.write_text(json.dumps({"schema_version": "1.0", "fixture": {"path": str(tmp_path)}, "models": {"regular": {"path": str(tmp_path / "r"), "sha256": ""}, "ema": {"path": str(tmp_path / "e"), "sha256": ""}}}), encoding="utf-8")
    from scripts.validation.run_recorded_command import run_recorded_command

    result = run_recorded_command(config, tmp_path / "job", "pt-test")
    assert result["status"] == "failed"
    assert result["returncode"] != 0
