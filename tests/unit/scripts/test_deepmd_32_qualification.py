"""Local fail-closed tests for the DeepMD 3.2 qualification harness."""

from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts.validation.collect_deepmd_32_qualification import collect_qualification
from scripts.validation.prepare_deepmd_32_qualification import prepare
from scripts.validation.submit_deepmd_32_qualification import parse_job_id, submit
from dpeva.compatibility import CapabilityEvidence, CapabilityKey, CapabilityRecord, validate_promotion_evidence


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_dir():
        for child in sorted(path.rglob("*")):
            if child.is_file():
                digest.update(str(child.relative_to(path)).encode())
                digest.update(child.read_bytes())
        return digest.hexdigest()
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(autouse=True)
def _dpa4c_model_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Give preparation tests an explicit disposable DPA4C model input."""

    model = tmp_path / "dpa4c-model.pt"
    model.write_bytes(b"dpa4c fixture")
    monkeypatch.setenv("DPEVA_DEEPMD_DPA4C_MODEL", str(model))


def _write_command_result(root: Path, case: str, *, returncode: int = 0, artifacts: list[str] | None = None) -> None:
    checks = []
    for artifact in artifacts or []:
        path_value = root / artifact
        path_value.parent.mkdir(parents=True, exist_ok=True)
        path_value.write_bytes(b"artifact")
        checks.append({"path": str(path_value), "exists": True, "sha256": hashlib.sha256(b"artifact").hexdigest()})
    path = root / "commands" / f"{case}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps({"schema_version": "1.0", "case": case, "argv": [case], "job_id": "123", "started_at": now, "ended_at": now, "returncode": returncode, "status": "finished" if returncode == 0 else "failed", "declared_artifacts": artifacts or [], "artifact_checks": checks}), encoding="utf-8")


def _complete_environment(root: Path) -> None:
    environment = root / "environment"
    environment.mkdir(parents=True, exist_ok=True)
    values = {
        "pip-freeze.json": {"schema_version": "1.0", "case": "pip-freeze", "returncode": 0, "value": ""},
        "deepmd-version.json": {"schema_version": "1.0", "case": "deepmd-version", "returncode": 0, "value": "DeePMD-kit v3.2.0"},
        "torch-cuda.json": {"schema_version": "1.0", "case": "torch-cuda", "returncode": 0, "value": {"available": True, "cuda": "12.6", "torch": "2.0"}},
        "gpu.json": {"schema_version": "1.0", "case": "gpu", "returncode": 0, "value": "GPU 0: Tesla V100"},
    }
    for name, content in values.items():
        (environment / name).write_text(json.dumps(content) + "\n", encoding="utf-8")


def test_collector_refuses_partial_success(tmp_path: Path) -> None:
    _write_command_result(tmp_path, "pt-test", artifacts=["results.e.out"])
    _write_command_result(tmp_path, "dpa4c-periodic-eval-desc", returncode=1)
    (tmp_path / "submission.json").write_text('{"job_id":"123"}', encoding="utf-8")
    report = collect_qualification(tmp_path, job_id="123", gpu="Tesla V100-SXM2-32GB")
    assert report["status"] == "failed"
    assert report["job_id"] == "123"
    assert "dpa4c-periodic-eval-desc" in report["failed_commands"]
    assert report["attestations"] == []
    with pytest.raises(RuntimeError, match="incomplete"):
        collect_qualification(tmp_path, job_id="123", require_complete=True)


def test_collector_rejects_mutable_latest(tmp_path: Path) -> None:
    latest = tmp_path / "latest"
    latest.mkdir()
    with pytest.raises(ValueError, match="mutable"):
        collect_qualification(latest)


def test_inspection_does_not_freeze_incomplete_final(tmp_path: Path) -> None:
    (tmp_path / "launch.json").write_text('{"schema_version":"1.0"}', encoding="utf-8")
    report = collect_qualification(tmp_path)
    assert report["status"] == "failed"
    assert not (tmp_path / "qualification.json").exists()


def test_collector_rejects_unrecorded_job_dir_cli(tmp_path: Path) -> None:
    (tmp_path / "launch.json").write_text('{"schema_version":"1.0"}', encoding="utf-8")
    from scripts.validation.collect_deepmd_32_qualification import main

    with pytest.raises(ValueError, match="recorded submission"):
        main(["--job-dir", str(tmp_path)])


def test_collector_accepts_directory_artifact_only_after_complete_records(tmp_path: Path) -> None:
    from scripts.validation.collect_deepmd_32_qualification import REQUIRED_CASES, _artifact_sha256

    model_root = tmp_path / "models"
    model_root.mkdir()
    for name in ("model.ckpt.pt", "model_ema.ckpt.pt"):
        (model_root / name).write_bytes(name.encode())
    input_path = tmp_path / "input.json"
    prepare(model_root, input_path)
    launch = tmp_path / "launch.json"
    launch.write_text(json.dumps({"schema_version": "1.0", "input_path": str(input_path), "input_sha256": _sha256(input_path)}), encoding="utf-8")
    for case in REQUIRED_CASES:
        artifact_dir = tmp_path / "artifacts" / case
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "result").write_text(case, encoding="utf-8")
        now = datetime.now(timezone.utc).isoformat()
        artifact = {"path": str(artifact_dir), "exists": True, "sha256": _artifact_sha256(artifact_dir)}
        record = {"schema_version": "1.0", "case": case, "argv": [case], "job_id": "123", "started_at": now, "ended_at": now, "returncode": 0, "status": "finished", "declared_artifacts": [str(artifact_dir)], "artifact_checks": [artifact]}
        path = tmp_path / "commands" / f"{case}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record), encoding="utf-8")
    _complete_environment(tmp_path)
    (tmp_path / "submission.json").write_text(json.dumps({"job_id": "123"}), encoding="utf-8")
    mismatch = collect_qualification(tmp_path, job_id="123", gpu="Tesla A100")
    assert mismatch["status"] == "failed"
    assert mismatch["attestations"] == []
    assert any("GPU expectation" in error for error in mismatch["invalid_evidence"])
    command_path = tmp_path / "commands" / "pt-test.json"
    command = json.loads(command_path.read_text(encoding="utf-8"))
    command["status"] = "failed"
    command_path.write_text(json.dumps(command), encoding="utf-8")
    bad_status = collect_qualification(tmp_path, job_id="123")
    assert bad_status["status"] == "failed"
    assert bad_status["attestations"] == []
    command["status"] = "finished"
    command_path.write_text(json.dumps(command), encoding="utf-8")
    report = collect_qualification(tmp_path, job_id="123", gpu=" GPU 0: Tesla V100 ", finalize=True)
    assert report["status"] == "finished"
    assert (tmp_path / "qualification.json").is_file()
    assert len(report["attestations"]) == 7
    key = CapabilityKey(operation="test", backend="pt", model_family="DPA4", artifact="checkpoint", data_format="deepmd/npy", environment="cpu")
    record = CapabilityRecord(
        key=key, status="supported", version_range=">=3.2,<3.3",
        verification_command="pytest tests/contract/deepmd/test_cli_contract.py::test_pt_test_requires_numeric_output -q",
        required_evidence=("sai-v100-qualification",), verification_status="implemented",
        evidence_ref=CapabilityEvidence(sai_qualification="qualification.json"),
        sai_verification_cases=("pt-test", "pt-test-ema"),
    )
    assert validate_promotion_evidence(record, tmp_path)


def test_collector_rejects_missing_and_nonnumeric_job_identity(tmp_path: Path) -> None:
    """A complete-looking command set cannot promote without bound numeric IDs."""

    (tmp_path / "submission.json").write_text(json.dumps({"job_id": "not-a-job"}), encoding="utf-8")
    report = collect_qualification(tmp_path, job_id="", finalize=False)
    assert report["status"] == "failed"
    assert report["attestations"] == []
    assert any("job_id must be numeric" in error for error in report["invalid_evidence"])


def test_prepare_records_models_without_copying(tmp_path: Path) -> None:
    model_root = tmp_path / "models"
    model_root.mkdir()
    (model_root / "model.ckpt.pt").write_bytes(b"regular")
    (model_root / "model_ema.ckpt.pt").write_bytes(b"ema")
    output = tmp_path / "build" / "input.json"
    payload = prepare(model_root, output)
    assert payload["fixture"]["type_map"] == ["Fe", "C", "H", "O"]
    assert payload["fixture"]["periodic"] is True
    assert payload["fixture"]["sha256"] == _sha256(Path(payload["fixture"]["path"]))
    assert payload["dpa4c_model_sha256"] == _sha256(Path(payload["dpa4c_model_path"]))
    assert Path(payload["models"]["regular"]["path"]) == model_root / "model.ckpt.pt"
    assert not (output.parent / "input" / "model.ckpt.pt").exists()
    with pytest.raises(FileExistsError):
        prepare(model_root, output)


def test_prepare_requires_dpa4c_model_reference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_root = tmp_path / "models"
    model_root.mkdir()
    (model_root / "model.ckpt.pt").write_bytes(b"regular")
    (model_root / "model_ema.ckpt.pt").write_bytes(b"ema")
    monkeypatch.delenv("DPEVA_DEEPMD_DPA4C_MODEL")
    with pytest.raises(FileNotFoundError, match="DPEVA_DEEPMD_DPA4C_MODEL"):
        prepare(model_root, tmp_path / "input.json")


def test_prepared_fixture_is_real_periodic_deepmd_npy(tmp_path: Path) -> None:
    dpdata = pytest.importorskip("dpdata")
    model_root = tmp_path / "models"
    model_root.mkdir()
    (model_root / "model.ckpt.pt").write_bytes(b"regular")
    (model_root / "model_ema.ckpt.pt").write_bytes(b"ema")
    payload = prepare(model_root, tmp_path / "input.json")
    system = dpdata.LabeledSystem(payload["fixture"]["path"], fmt="deepmd/npy")
    assert system.get_nframes() == 1
    assert system.get_ntypes() == 4
    assert system["cells"].shape == (1, 3, 3)


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
    slurm.write_text("\n".join(("#!/bin/bash", "#SBATCH --partition=4V100", "#SBATCH --nodes=1", "#SBATCH --ntasks=1", "#SBATCH --gpus-per-node=1", "#SBATCH --qos=improper-gpu", "#SBATCH --time=00:30:00")) + "\n", encoding="utf-8")
    ref = tmp_path / "latest.json"
    result = submit(input_path, slurm, ref, job_root=tmp_path / "external", dry_run=True)
    assert result["status"] == "dry-run"
    assert json.loads(ref.read_text(encoding="utf-8"))["job_dir"] != str(tmp_path / "external" / "latest")
    assert (Path(result["job_dir"]) / "submission.json").is_file()
    launch = json.loads((Path(result["job_dir"]) / "launch.json").read_text(encoding="utf-8"))
    submission = json.loads((Path(result["job_dir"]) / "submission.json").read_text(encoding="utf-8"))
    assert launch["qualification_env_name"] == "dpeva-dpa4-320"
    assert submission["qualification_env_name"] == launch["qualification_env_name"]


def test_submit_uses_nil_export_without_slurm_login_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SAI cancels ``--export=NONE`` jobs before their batch step starts."""

    model_root = tmp_path / "models"
    model_root.mkdir()
    for name in ("model.ckpt.pt", "model_ema.ckpt.pt"):
        (model_root / name).write_bytes(name.encode())
    input_path = tmp_path / "input.json"
    prepare(model_root, input_path)
    slurm = tmp_path / "job.slurm"
    slurm.write_text(
        "\n".join(
            (
                "#!/bin/bash",
                "#SBATCH --partition=4V100",
                "#SBATCH --nodes=1",
                "#SBATCH --ntasks=1",
                "#SBATCH --gpus-per-node=1",
                "#SBATCH --qos=improper-gpu",
                "#SBATCH --time=00:30:00",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> object:
        from subprocess import CompletedProcess

        calls.append(command)
        return CompletedProcess(command, 0, "Submitted batch job 123\n", "")

    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.setattr(
        "scripts.validation.submit_deepmd_32_qualification.subprocess.run", fake_run
    )

    submit(
        input_path,
        slurm,
        tmp_path / "latest.json",
        job_root=tmp_path / "external",
    )

    assert calls[0][0:2] == ["sbatch", "--export=NIL"]
    assert "--export=NONE" not in calls[0]
    submission_path = next((tmp_path / "external").glob("deepmd-32-*/submission.json"))
    job_dir = Path(json.loads(submission_path.read_text(encoding="utf-8"))["job_dir"])
    assert f"--output={job_dir}/slurm-%j.out" in calls[0]
    assert f"--error={job_dir}/slurm-%j.err" in calls[0]
    assert calls[0][-4] == str(slurm)
    assert calls[0][-1] == str(slurm.parent.parent.parent.resolve())


def test_submit_rehashes_fixture_before_submission(tmp_path: Path) -> None:
    model_root = tmp_path / "models"
    model_root.mkdir()
    for name in ("model.ckpt.pt", "model_ema.ckpt.pt"):
        (model_root / name).write_bytes(name.encode())
    input_path = tmp_path / "input.json"
    payload = prepare(model_root, input_path)
    (Path(payload["fixture"]["path"]) / "type.raw").write_text("0 1 2 9\n", encoding="utf-8")
    slurm = tmp_path / "job.slurm"
    slurm.write_text("\n".join(("#!/bin/bash", "#SBATCH --partition=4V100", "#SBATCH --nodes=1", "#SBATCH --ntasks=1", "#SBATCH --gpus-per-node=1", "#SBATCH --qos=improper-gpu", "#SBATCH --time=00:30:00")) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="periodic fixture"):
        submit(input_path, slurm, tmp_path / "latest.json", job_root=tmp_path / "external", dry_run=True)


def test_slurm_script_selects_qualified_environment_before_source() -> None:
    script = Path("scripts/validation/run_deepmd_32_qualification.slurm").read_text(encoding="utf-8")
    assert script.startswith("#!/bin/bash\n")
    assert '[[ "$#" -ne 3 ]]' in script
    assert 'readonly REPO_ROOT="$3"' in script
    assert "BASH_SOURCE" not in script
    assert 'export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"' in script
    assert script.index("PYTHONPATH") < script.index("source \"$REPO_ROOT/scripts/env/dpeva-dpa4.env\"")
    assert 'export DPEVA_DPA4_ENV_NAME="dpeva-dpa4-320"' in script
    assert script.index("DPEVA_DPA4_ENV_NAME") < script.index("source \"$REPO_ROOT/scripts/env/dpeva-dpa4.env\"")


def test_preflight_rejects_wrong_qualification_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from subprocess import CompletedProcess
    from scripts.validation.run_recorded_command import _sha256, run_recorded_command

    model_root = tmp_path / "models"
    model_root.mkdir()
    for name in ("model.ckpt.pt", "model_ema.ckpt.pt"):
        (model_root / name).write_bytes(name.encode())
    fixture = tmp_path / "input"
    (fixture / "set.000").mkdir(parents=True)
    (fixture / "type.raw").write_text("0\n", encoding="utf-8")
    (fixture / "type_map.raw").write_text("Fe\n", encoding="utf-8")
    config = tmp_path / "input.json"
    config.write_text(json.dumps({"schema_version": "1.0", "fixture": {"path": str(fixture)}, "models": {"regular": {"path": str(model_root / "model.ckpt.pt"), "sha256": _sha256(model_root / "model.ckpt.pt")}, "ema": {"path": str(model_root / "model_ema.ckpt.pt"), "sha256": _sha256(model_root / "model_ema.ckpt.pt")}}}), encoding="utf-8")
    script = Path("scripts/validation/run_deepmd_32_qualification.slurm").resolve()
    launch = tmp_path / "job" / "launch.json"
    launch.parent.mkdir()
    launch.write_text(json.dumps({"schema_version": "1.0", "qualification_env_name": "dpeva-dpa4-320", "input_path": str(config), "input_sha256": _sha256(config), "slurm_script_path": str(script), "slurm_script_sha256": _sha256(script), "expected_deepmd_version": "DeePMD-kit v3.2.0", "expected_gpu": "V100", "job_dir": str(launch.parent)}), encoding="utf-8")
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "dpeva-dpa4")
    monkeypatch.setenv("CONDA_PREFIX", "/opt/conda/envs/dpeva-dpa4")

    def fake_run(argv: list[str], **_kwargs: object) -> CompletedProcess[str]:
        if argv[:2] == ["dp", "--version"]:
            return CompletedProcess(argv, 0, "DeePMD-kit v3.2.0\n", "")
        if argv[:2] == ["nvidia-smi", "-L"]:
            return CompletedProcess(argv, 0, "GPU 0: Tesla V100\n", "")
        return CompletedProcess(argv, 0, '{"torch":"2.0","cuda":"12.6","available":true}\n', "")

    monkeypatch.setattr("scripts.validation.run_recorded_command.subprocess.run", fake_run)
    result = run_recorded_command(config, launch.parent, "preflight")
    assert result["status"] == "failed"
    assert "environment mismatch" in result["error"]


@pytest.mark.parametrize("cpu_directive", ["#SBATCH --cpus=1", "#SBATCH --cpus-per-task=1", "#SBATCH --cpus-per-task 1"])
def test_submit_rejects_cpu_directives(tmp_path: Path, cpu_directive: str) -> None:
    model_root = tmp_path / "models"
    model_root.mkdir()
    for name in ("model.ckpt.pt", "model_ema.ckpt.pt"):
        (model_root / name).write_bytes(name.encode())
    input_path = tmp_path / "input.json"
    prepare(model_root, input_path)
    slurm = tmp_path / "job.slurm"
    slurm.write_text("\n".join(("#!/bin/bash", "#SBATCH --partition=4V100", "#SBATCH --nodes=1", "#SBATCH --ntasks=1", "#SBATCH --gpus-per-node=1", "#SBATCH --qos=improper-gpu", "#SBATCH --time=00:30:00", cpu_directive)) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bounded SAI"):
        submit(input_path, slurm, tmp_path / "latest.json", job_root=tmp_path / "external", dry_run=True)


@pytest.mark.parametrize("memory_directive", ["#SBATCH --mem=1G", "#SBATCH --mem-per-cpu=1G", "#SBATCH --mem-per-gpu=1G", "#SBATCH --mem-per-cpu 1G"])
def test_submit_rejects_all_memory_directives(tmp_path: Path, memory_directive: str) -> None:
    model_root = tmp_path / "models"
    model_root.mkdir()
    for name in ("model.ckpt.pt", "model_ema.ckpt.pt"):
        (model_root / name).write_bytes(name.encode())
    input_path = tmp_path / "input.json"
    prepare(model_root, input_path)
    slurm = tmp_path / "job.slurm"
    slurm.write_text("\n".join(("#!/bin/bash", "#SBATCH --partition=4V100", "#SBATCH --nodes=1", "#SBATCH --ntasks=1", "#SBATCH --gpus-per-node=1", "#SBATCH --qos=improper-gpu", "#SBATCH --time=00:30:00", memory_directive)) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bounded SAI"):
        submit(input_path, slurm, tmp_path / "latest.json", job_root=tmp_path / "external", dry_run=True)


def test_recorded_runner_argv_and_missing_artifact(tmp_path: Path) -> None:
    config = tmp_path / "input.json"
    config.write_text(json.dumps({"schema_version": "1.0", "fixture": {"path": str(tmp_path)}, "models": {"regular": {"path": str(tmp_path / "r"), "sha256": ""}, "ema": {"path": str(tmp_path / "e"), "sha256": ""}}}), encoding="utf-8")
    from scripts.validation.run_recorded_command import run_recorded_command

    result = run_recorded_command(config, tmp_path / "job", "pt-test")
    assert result["status"] == "failed"
    assert result["returncode"] != 0


def test_preflight_writes_record_from_fresh_job_dir(tmp_path: Path) -> None:
    from scripts.validation.run_recorded_command import run_recorded_command

    config = tmp_path / "input.json"
    config.write_text('{"schema_version":"1.0"}', encoding="utf-8")
    job_dir = tmp_path / "fresh-job"
    job_dir.mkdir()
    result = run_recorded_command(config, job_dir, "preflight")
    assert result["status"] == "failed"
    assert (job_dir / "commands" / "preflight.json").is_file()
