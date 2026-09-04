import json
import subprocess
import sys
from pathlib import Path

import pytest

from dpeva.config import FeatureConfig, InferenceConfig
from dpeva.run.context import RunOptions
from dpeva.run.artifacts import ArtifactValidationError
from dpeva.utils.exceptions import PartialWorkflowError, WorkflowError
from dpeva.workflows.feature import FeatureWorkflow
from dpeva.workflows.infer import InferenceWorkflow


def _feature_config(tmp_path, *, backend="local", savedir=None):
    data = tmp_path / "data"
    data.mkdir(parents=True, exist_ok=True)
    (data / "type.raw").write_text("0\n")
    model = tmp_path / "model.pt"
    model.write_bytes(b"model")
    return FeatureConfig(
        data_path=data,
        model_path=model,
        savedir=savedir or tmp_path / "features",
        submission={"backend": backend},
    )


def _infer_config(tmp_path, *, backend="local"):
    data = tmp_path / "data"
    data.mkdir(parents=True, exist_ok=True)
    (data / "type.raw").write_text("0\n")
    work = tmp_path / "work"
    model = work / "0" / "model.ckpt.pt"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"model")
    return InferenceConfig(
        work_dir=work,
        data_path=data,
        task_name="test_val",
        submission={"backend": backend},
    )


def test_feature_failure_writes_failed_manifest(tmp_path, monkeypatch) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "type.raw").write_text("0\n")
    model = tmp_path / "model.pt"
    model.write_bytes(b"model")
    output = tmp_path / "desc"

    def fail_submit(*args, **kwargs):
        raise subprocess.CalledProcessError(2, ["bash", "run_evaldesc.sh"])

    monkeypatch.setattr("dpeva.submission.manager.JobManager.submit", fail_submit)
    config = FeatureConfig(data_path=data, model_path=model, savedir=output)
    with pytest.raises(subprocess.CalledProcessError):
        FeatureWorkflow(
            config,
            original_config=config.model_dump(mode="json"),
            run_options=RunOptions(run_id="feature-failure"),
        ).run()
    payload = json.loads((output / ".dpeva/runs/feature-failure/run.json").read_text())
    assert payload["status"] == "failed"
    assert payload["failure"]["category"] == "EXECUTION"


def test_infer_mixed_children_write_partial_manifest(tmp_path, monkeypatch) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "type.raw").write_text("0\n")
    work = tmp_path / "work"
    for index in (0, 1):
        model = work / str(index) / "model.ckpt.pt"
        model.parent.mkdir(parents=True)
        model.write_bytes(b"model")

    def mixed_submit(self, script_path, working_dir="."):
        directory = Path(working_dir)
        if directory.parts[-2] == "0":
            (directory / "results.e.out").write_text("0 0\n")
            return ""
        raise subprocess.CalledProcessError(4, ["bash", str(script_path)])

    monkeypatch.setattr("dpeva.submission.manager.JobManager.submit", mixed_submit)
    config = InferenceConfig(work_dir=work, data_path=data, task_name="test_val")
    with pytest.raises(PartialWorkflowError):
        InferenceWorkflow(
            config,
            original_config=config.model_dump(mode="json"),
            run_options=RunOptions(run_id="infer-partial"),
        ).run()
    payload = json.loads((work / ".dpeva/runs/infer-partial/run.json").read_text())
    assert payload["status"] == "partial"
    assert [job["status"] for job in payload["jobs"]] == ["finished", "failed"]
    assert payload["artifacts"][0]["status"] == "verified"


def test_infer_all_children_failure_writes_failed_manifest(tmp_path, monkeypatch) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "type.raw").write_text("0\n")
    work = tmp_path / "work"
    for index in (0, 1):
        model = work / str(index) / "model.ckpt.pt"
        model.parent.mkdir(parents=True)
        model.write_bytes(b"model")

    def fail_submit(self, script_path, working_dir="."):
        raise subprocess.CalledProcessError(4, ["bash", str(script_path)])

    monkeypatch.setattr("dpeva.submission.manager.JobManager.submit", fail_submit)
    config = InferenceConfig(work_dir=work, data_path=data, task_name="test_val")
    with pytest.raises(WorkflowError, match="all inference jobs failed"):
        InferenceWorkflow(
            config,
            original_config=config.model_dump(mode="json"),
            run_options=RunOptions(run_id="infer-failed"),
        ).run()
    payload = json.loads((work / ".dpeva/runs/infer-failed/run.json").read_text())
    assert payload["status"] == "failed"
    assert [job["status"] for job in payload["jobs"]] == ["failed", "failed"]


def test_feature_success_manifest_contains_verified_output(tmp_path, monkeypatch) -> None:
    config = _feature_config(tmp_path)

    def submit(self, script_path, working_dir="."):
        Path(working_dir, "features.npy").write_bytes(b"feature")
        return ""

    monkeypatch.setattr("dpeva.submission.manager.JobManager.submit", submit)
    FeatureWorkflow(config, run_options=RunOptions(run_id="feature-success")).run()
    payload = json.loads(
        (config.savedir / ".dpeva/runs/feature-success/run.json").read_text()
    )
    assert payload["status"] == "finished"
    assert payload["artifacts"][0]["status"] == "verified"


def test_feature_missing_output_is_artifact_failure(tmp_path, monkeypatch) -> None:
    config = _feature_config(tmp_path)
    monkeypatch.setattr("dpeva.submission.manager.JobManager.submit", lambda *a, **k: "")
    with pytest.raises(ArtifactValidationError, match="missing or empty"):
        FeatureWorkflow(config, run_options=RunOptions(run_id="feature-empty")).run()
    payload = json.loads(
        (config.savedir / ".dpeva/runs/feature-empty/run.json").read_text()
    )
    assert payload["status"] == "failed"
    assert payload["failure"]["category"] == "ARTIFACT"


def test_feature_multi_pool_requires_each_pool(tmp_path, monkeypatch) -> None:
    data = tmp_path / "data"
    for pool in ("pool0", "pool1"):
        system = data / pool / "system"
        system.mkdir(parents=True)
        (system / "type.raw").write_text("0\n")
    model = tmp_path / "model.pt"
    model.write_bytes(b"model")
    config = FeatureConfig(
        data_path=data,
        model_path=model,
        savedir=tmp_path / "features",
        submission={"backend": "local"},
    )

    def submit(self, script_path, working_dir="."):
        Path(working_dir, "pool0").mkdir(parents=True, exist_ok=True)
        Path(working_dir, "pool0", "features.npy").write_bytes(b"feature")
        return ""

    monkeypatch.setattr("dpeva.submission.manager.JobManager.submit", submit)
    with pytest.raises(ArtifactValidationError, match="pool1"):
        FeatureWorkflow(config, run_options=RunOptions(run_id="feature-pools")).run()
    payload = json.loads(
        (config.savedir / ".dpeva/runs/feature-pools/run.json").read_text()
    )
    assert payload["failure"]["category"] == "ARTIFACT"


def test_infer_success_manifest_and_artifact(tmp_path, monkeypatch) -> None:
    config = _infer_config(tmp_path)

    def submit(self, script_path, working_dir="."):
        Path(working_dir, "results.e.out").write_text("prediction\n")
        return ""

    monkeypatch.setattr("dpeva.submission.manager.JobManager.submit", submit)
    InferenceWorkflow(config, run_options=RunOptions(run_id="infer-success")).run()
    payload = json.loads(
        (config.work_dir / ".dpeva/runs/infer-success/run.json").read_text()
    )
    assert payload["status"] == "finished"
    assert payload["jobs"][0]["status"] == "finished"
    assert payload["artifacts"][0]["status"] == "verified"


def test_infer_empty_output_is_artifact_failure(tmp_path, monkeypatch) -> None:
    config = _infer_config(tmp_path)
    monkeypatch.setattr("dpeva.submission.manager.JobManager.submit", lambda *a, **k: "")
    with pytest.raises(WorkflowError, match="all inference jobs failed"):
        InferenceWorkflow(config, run_options=RunOptions(run_id="infer-empty")).run()
    payload = json.loads(
        (config.work_dir / ".dpeva/runs/infer-empty/run.json").read_text()
    )
    assert payload["failure"]["category"] == "ARTIFACT"
    assert payload["jobs"][0]["failure_category"] == "ARTIFACT"


def test_slurm_feature_and_infer_record_parsed_ids(tmp_path, monkeypatch) -> None:
    feature = _feature_config(tmp_path / "feature", backend="slurm")
    infer = _infer_config(tmp_path / "infer", backend="slurm")

    def submit(self, script_path, working_dir="."):
        return "Submitted batch job 8123"

    monkeypatch.setattr("dpeva.submission.manager.JobManager.submit", submit)
    FeatureWorkflow(feature, run_options=RunOptions(run_id="feature-slurm")).run()
    InferenceWorkflow(infer, run_options=RunOptions(run_id="infer-slurm")).run()
    f_payload = json.loads(
        (feature.savedir / ".dpeva/runs/feature-slurm/run.json").read_text()
    )
    i_payload = json.loads(
        (infer.work_dir / ".dpeva/runs/infer-slurm/run.json").read_text()
    )
    assert f_payload["status"] == i_payload["status"] == "submitted"
    assert f_payload["jobs"][0]["job_id"] == i_payload["jobs"][0]["job_id"] == "8123"


def test_cli_partial_exit_and_snapshots(tmp_path, monkeypatch) -> None:
    config_dir = tmp_path / "cli"
    config_dir.mkdir()
    data = config_dir / "data"
    data.mkdir()
    (data / "type.raw").write_text("0\n")
    work = config_dir / "work"
    for index in (0, 1):
        model = work / str(index) / "model.ckpt.pt"
        model.parent.mkdir(parents=True)
        model.write_bytes(b"model")
    config_path = config_dir / "infer.json"
    config_path.write_text(
        json.dumps(
            {
                "work_dir": "work",
                "data_path": "data",
                "task_name": "test_val",
                "submission": {"backend": "local"},
            }
        )
    )

    allow_all = {"value": False}

    def submit(self, script_path, working_dir="."):
        directory = Path(working_dir)
        if allow_all["value"] or directory.parts[-2] == "0":
            Path(working_dir, "results.e.out").write_text("prediction\n")
            return ""
        raise subprocess.CalledProcessError(2, ["bash", str(script_path)])

    monkeypatch.setattr("dpeva.submission.manager.JobManager.submit", submit)
    monkeypatch.setattr(sys, "argv", ["dpeva", "--no-banner", "infer", str(config_path), "--run-id", "cli-partial"])
    with pytest.raises(SystemExit) as exc:
        __import__("dpeva.cli", fromlist=["main"]).main()
    assert exc.value.code == 1
    run_dir = work / ".dpeva/runs/cli-partial"
    payload = json.loads((run_dir / "run.json").read_text())
    assert payload["status"] == "partial"
    original = json.loads((run_dir / "config.original.json").read_text())
    resolved = json.loads((run_dir / "config.resolved.json").read_text())
    assert original["work_dir"] == "work"
    assert resolved["work_dir"] == str(work)

    # Existing evidence is immutable by default; an incomplete partial run is
    # explicitly resumable, and force creates a new audited attempt.
    with pytest.raises(SystemExit) as exc:
        __import__("dpeva.cli", fromlist=["main"]).main()
    assert exc.value.code == 1

    allow_all["value"] = True
    monkeypatch.setattr(
        sys,
        "argv",
        ["dpeva", "--no-banner", "infer", str(config_path), "--run-id", "cli-partial", "--resume"],
    )
    __import__("dpeva.cli", fromlist=["main"]).main()
    payload = json.loads((run_dir / "run.json").read_text())
    assert payload["status"] == "finished"
    assert any(event["kind"] == "resume" for event in payload["events"])

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dpeva", "--no-banner", "infer", str(config_path),
            "--run-id", "cli-partial", "--force", "--reason", "verified retry",
        ],
    )
    __import__("dpeva.cli", fromlist=["main"]).main()
    payload = json.loads((run_dir / "run.json").read_text())
    assert payload["status"] == "finished"
    assert any(event["kind"] == "force" and event["reason"] == "verified retry" for event in payload["events"])
