import json
import subprocess
from pathlib import Path

import pytest

from dpeva.config import FeatureConfig, InferenceConfig
from dpeva.run.context import RunOptions
from dpeva.utils.exceptions import PartialWorkflowError, WorkflowError
from dpeva.workflows.feature import FeatureWorkflow
from dpeva.workflows.infer import InferenceWorkflow


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
