import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from dpeva import __version__
from dpeva.compatibility import DeepMDAdapter
from dpeva.config import FeatureConfig, InferenceConfig
from dpeva.constants import LOG_FILE_FEATURE, LOG_FILE_INFER, WORKFLOW_FINISHED_TAG
from dpeva.run.context import RunOptions
from dpeva.run.artifacts import ArtifactValidationError, validate_feature_outputs
from dpeva.utils.logs import close_workflow_logger
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
    retained = payload["artifacts"][0].copy()

    def resume_submit(self, script_path, working_dir="."):
        directory = Path(working_dir)
        if directory.parts[-2] == "1":
            (directory / "results.e.out").write_text("1 1\n")
        return ""

    monkeypatch.setattr("dpeva.submission.manager.JobManager.submit", resume_submit)
    with pytest.raises(PartialWorkflowError):
        InferenceWorkflow(
            config,
            original_config=config.model_dump(mode="json"),
            run_options=RunOptions(run_id="infer-partial", resume=True),
        ).run()
    resumed = json.loads((work / ".dpeva/runs/infer-partial/run.json").read_text())
    assert resumed["status"] == "partial"
    assert resumed["artifacts"][0] == retained
    assert [item["path"] for item in resumed["artifacts"] if item["kind"] == "inference"] == [
        "0/test_val/results.e.out",
        "1/test_val/results.e.out",
    ]


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

    def fake_eval_desc(self, *, output, **kwargs):
        target = shlex.quote(str(Path(output) / "features.npy"))
        return f"printf feature > {target}"

    monkeypatch.setattr(DeepMDAdapter, "eval_desc", fake_eval_desc)
    try:
        FeatureWorkflow(config, run_options=RunOptions(run_id="feature-success")).run()
    finally:
        close_workflow_logger("dpeva", str(config.savedir / LOG_FILE_FEATURE))
    payload = json.loads(
        (config.savedir / ".dpeva/runs/feature-success/run.json").read_text()
    )
    assert payload["status"] == "finished"
    assert payload["source"]["dpeva_version"] == __version__
    assert payload["source"]["package_version"] == __version__
    assert len(payload["source"]["git_commit"]) == 40
    assert all(not value.startswith("/") for item in payload["inputs"] for value in item.values())
    assert payload["inputs"][0]["identity_scope"] == "bounded-structural"
    assert payload["inputs"][1]["identity_scope"] == "full-content"
    assert payload["artifacts"][0]["status"] == "verified"
    feature_log = config.savedir / LOG_FILE_FEATURE
    assert feature_log.read_text(encoding="utf-8").count(WORKFLOW_FINISHED_TAG) == 1
    log_artifacts = [item for item in payload["artifacts"] if item["kind"] == "log"]
    assert log_artifacts
    for artifact in log_artifacts:
        final_bytes = (config.savedir / artifact["path"]).read_bytes()
        assert artifact["checksum"] == hashlib.sha256(final_bytes).hexdigest()


def test_infer_real_local_children_do_not_leak_marker_on_partial(
    tmp_path, monkeypatch
) -> None:
    config = _infer_config(tmp_path)
    second = config.work_dir / "1" / "model.ckpt.pt"
    second.parent.mkdir(parents=True)
    second.write_bytes(b"model")

    def mixed_test(self, *, model, **kwargs):
        if Path(model).parent.name == "0":
            return "printf 'prediction\\n' > results.e.out"
        return "false"

    monkeypatch.setattr(DeepMDAdapter, "test", mixed_test)
    try:
        with pytest.raises(PartialWorkflowError):
            InferenceWorkflow(
                config, run_options=RunOptions(run_id="infer-real-partial")
            ).run()
    finally:
        close_workflow_logger("dpeva", str(config.work_dir / LOG_FILE_INFER))

    payload = json.loads(
        (config.work_dir / ".dpeva/runs/infer-real-partial/run.json").read_text()
    )
    assert payload["status"] == "partial"
    assert [job["status"] for job in payload["jobs"]] == ["finished", "failed"]
    assert WORKFLOW_FINISHED_TAG not in (
        config.work_dir / LOG_FILE_INFER
    ).read_text(encoding="utf-8")


def test_infer_real_local_child_does_not_leak_marker_on_analysis_failure(
    tmp_path, monkeypatch
) -> None:
    config = _infer_config(tmp_path)
    config.auto_analysis = True

    def successful_test(self, **kwargs):
        return "printf 'prediction\\n' > results.e.out"

    def fail_analysis(self):
        raise WorkflowError("analysis failed")

    monkeypatch.setattr(DeepMDAdapter, "test", successful_test)
    monkeypatch.setattr(InferenceWorkflow, "analyze_results", fail_analysis)
    try:
        with pytest.raises(WorkflowError, match="analysis failed"):
            InferenceWorkflow(
                config, run_options=RunOptions(run_id="infer-analysis-failed")
            ).run()
    finally:
        close_workflow_logger("dpeva", str(config.work_dir / LOG_FILE_INFER))

    payload = json.loads(
        (config.work_dir / ".dpeva/runs/infer-analysis-failed/run.json").read_text()
    )
    assert payload["status"] == "failed"
    assert WORKFLOW_FINISHED_TAG not in (
        config.work_dir / LOG_FILE_INFER
    ).read_text(encoding="utf-8")


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


def test_feature_preexisting_output_is_not_attributed_to_noop_attempt(
    tmp_path, monkeypatch
) -> None:
    config = _feature_config(tmp_path)
    config.savedir.mkdir(parents=True)
    stale = config.savedir / "features.npy"
    stale.write_bytes(b"feature")
    before = stale.stat()
    monkeypatch.setattr("dpeva.submission.manager.JobManager.submit", lambda *a, **k: "")

    with pytest.raises(ArtifactValidationError, match="current attempt"):
        FeatureWorkflow(config, run_options=RunOptions(run_id="feature-stale")).run()

    after = stale.stat()
    assert (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns) == (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    payload = json.loads(
        (config.savedir / ".dpeva/runs/feature-stale/run.json").read_text()
    )
    assert payload["status"] == "failed"
    assert not [item for item in payload["artifacts"] if item["kind"] == "feature"]


def test_feature_identical_bytes_rewritten_in_attempt_are_fresh(tmp_path, monkeypatch) -> None:
    config = _feature_config(tmp_path)
    config.savedir.mkdir(parents=True)
    output = config.savedir / "features.npy"
    output.write_bytes(b"same bytes")
    (config.savedir / "untouched.npy").write_bytes(b"stale neighbor")
    (config.savedir / "eval_desc.err").write_text("old log\n", encoding="utf-8")

    def rewrite(self, script_path, working_dir="."):
        Path(working_dir, "features.npy").write_bytes(b"same bytes")
        return ""

    monkeypatch.setattr("dpeva.submission.manager.JobManager.submit", rewrite)
    FeatureWorkflow(config, run_options=RunOptions(run_id="feature-rewrite")).run()

    payload = json.loads(
        (config.savedir / ".dpeva/runs/feature-rewrite/run.json").read_text()
    )
    assert payload["status"] == "finished"
    assert [item["path"] for item in payload["artifacts"] if item["kind"] == "feature"] == [
        "features.npy"
    ]
    assert "eval_desc.err" not in [item["path"] for item in payload["artifacts"]]


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
    assert payload["source"]["dpeva_version"] == __version__
    assert payload["source"]["package_version"] == __version__
    assert all(not value.startswith("/") for item in payload["inputs"] for value in item.values())
    assert payload["inputs"][0]["identity_scope"] == "bounded-structural"
    assert payload["inputs"][1]["identity_scope"] == "full-content"
    assert payload["jobs"][0]["status"] == "finished"
    assert payload["artifacts"][0]["status"] == "verified"


def test_infer_explicit_same_basename_regular_and_ema_refs_execute_with_distinct_evidence(
    tmp_path, monkeypatch
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "type.raw").write_text("0\n")
    work = tmp_path / "work"
    first = tmp_path / "regular" / "model.pt"
    second = tmp_path / "ema" / "model.pt"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"regular")
    second.write_bytes(b"ema")
    refs = []
    for path, role in ((first, "regular"), (second, "ema")):
        ref_path = tmp_path / f"{role}.json"
        ref_path.write_text(
            json.dumps(
                {
                    "kind": "checkpoint",
                    "family": "DPA4C",
                    "backend": "pt-expt",
                    "path": path.relative_to(tmp_path).as_posix(),
                    "checksum": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "role": role,
                    "supported_operations": ["test"],
                }
            ),
            encoding="utf-8",
        )
        refs.append(ref_path)

    commands: list[str] = []

    def submit(self, script_path, working_dir="."):
        commands.append(Path(script_path).read_text(encoding="utf-8"))
        Path(working_dir, "results.e.out").write_text("prediction\n")
        return ""

    monkeypatch.setattr("dpeva.submission.manager.JobManager.submit", submit)
    config = InferenceConfig(
        work_dir=work,
        data_path=data,
        task_name="test_val",
        dp_backend="pt-expt",
        model_ref_paths=refs,
        submission={"backend": "local"},
    )
    workflow = InferenceWorkflow(config, run_options=RunOptions(run_id="infer-explicit-ensemble"))
    assert [ref.role.value for ref in workflow.model_refs] == ["regular", "ema"]
    workflow.run()

    payload = json.loads(
        (work / ".dpeva/runs/infer-explicit-ensemble/run.json").read_text()
    )
    assert payload["status"] == "finished"
    assert [job["status"] for job in payload["jobs"]] == ["finished", "finished"]
    model_inputs = [item for item in payload["inputs"] if item["kind"] == "model"]
    assert len(model_inputs) == 2
    assert model_inputs[0]["ref"] != model_inputs[1]["ref"]
    assert str(first) in commands[0]
    assert str(second) in commands[1]


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


def test_infer_preexisting_output_is_not_attributed_to_noop_attempt(
    tmp_path, monkeypatch
) -> None:
    config = _infer_config(tmp_path)
    output = config.work_dir / "0" / config.task_name / "results.e.out"
    output.parent.mkdir(parents=True)
    output.write_text("stale prediction\n", encoding="utf-8")
    monkeypatch.setattr("dpeva.submission.manager.JobManager.submit", lambda *a, **k: "")

    with pytest.raises(WorkflowError, match="all inference jobs failed"):
        InferenceWorkflow(config, run_options=RunOptions(run_id="infer-stale")).run()

    payload = json.loads(
        (config.work_dir / ".dpeva/runs/infer-stale/run.json").read_text()
    )
    assert payload["failure"]["category"] == "ARTIFACT"
    assert payload["jobs"][0]["failure_category"] == "ARTIFACT"
    assert not [item for item in payload["artifacts"] if item["kind"] == "inference"]


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
    assert payload["config"]["metadata"] == "config.metadata.json"
    metadata = json.loads((run_dir / "config.metadata.json").read_text())
    assert metadata["schema_version"] == "1.0"
    assert metadata["input_schema_version"] == "1.0"
    assert metadata["migration_warnings"] == []

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


@pytest.mark.parametrize("response", [None, "not an sbatch response"])
def test_feature_malformed_slurm_response_is_failed(tmp_path, monkeypatch, response) -> None:
    config = _feature_config(tmp_path, backend="slurm")
    monkeypatch.setattr("dpeva.submission.manager.JobManager.submit", lambda *a, **k: response)
    with pytest.raises((TypeError, ValueError)):
        FeatureWorkflow(config, run_options=RunOptions(run_id="feature-bad-slurm")).run()
    payload = json.loads(
        (config.savedir / ".dpeva/runs/feature-bad-slurm/run.json").read_text()
    )
    assert payload["status"] == "failed"
    assert payload["failure"]["category"] == "EXECUTION"


def test_infer_slurm_mixed_submission_stays_submitted(tmp_path, monkeypatch) -> None:
    config = _infer_config(tmp_path, backend="slurm")
    model = config.work_dir / "1" / "model.ckpt.pt"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"model")
    calls = {"count": 0}

    def submit(self, script_path, working_dir="."):
        calls["count"] += 1
        if calls["count"] == 1:
            return "Submitted batch job 8123"
        raise subprocess.CalledProcessError(2, ["sbatch", str(script_path)])

    monkeypatch.setattr("dpeva.submission.manager.JobManager.submit", submit)
    with pytest.raises(PartialWorkflowError):
        InferenceWorkflow(config, run_options=RunOptions(run_id="infer-slurm-partial")).run()
    run_path = config.work_dir / ".dpeva/runs/infer-slurm-partial/run.json"
    payload = json.loads(run_path.read_text())
    assert payload["status"] == "submitted"
    assert payload["failure"] is None
    assert payload["jobs"][0]["job_id"] == "8123"
    assert payload["jobs"][0]["status"] == "submitted"
    assert payload["jobs"][1]["status"] == "failed"
    assert payload["jobs"][1]["failure_category"] == "EXECUTION"
    assert all(event["state"] not in {"running", "partial"} for event in payload["events"])


def test_feature_resume_of_submitted_slurm_rejects_without_new_job(tmp_path, monkeypatch) -> None:
    config = _feature_config(tmp_path, backend="slurm")
    monkeypatch.setattr(
        "dpeva.submission.manager.JobManager.submit",
        lambda *a, **k: "Submitted batch job 8125",
    )
    FeatureWorkflow(config, run_options=RunOptions(run_id="feature-slurm-resume")).run()
    calls = {"submit": 0}
    monkeypatch.setattr(
        "dpeva.submission.manager.JobManager.submit",
        lambda *a, **k: calls.__setitem__("submit", calls["submit"] + 1),
    )
    with pytest.raises(ValueError, match="submitted.*scheduler"):
        FeatureWorkflow(
            config,
            run_options=RunOptions(run_id="feature-slurm-resume", resume=True),
        ).run()
    payload = json.loads(
        (config.savedir / ".dpeva/runs/feature-slurm-resume/run.json").read_text()
    )
    assert payload["status"] == "submitted"
    assert [job["job_id"] for job in payload["jobs"]] == ["8125"]
    assert calls["submit"] == 0


def test_infer_slurm_all_fail_is_execution_failure(tmp_path, monkeypatch) -> None:
    config = _infer_config(tmp_path, backend="slurm")

    def fail_submit(*args, **kwargs):
        raise subprocess.CalledProcessError(2, ["sbatch"])

    monkeypatch.setattr("dpeva.submission.manager.JobManager.submit", fail_submit)
    with pytest.raises(WorkflowError, match="all inference jobs failed"):
        InferenceWorkflow(
            config, run_options=RunOptions(run_id="infer-slurm-failed")
        ).run()
    payload = json.loads(
        (config.work_dir / ".dpeva/runs/infer-slurm-failed/run.json").read_text()
    )
    assert payload["status"] == "failed"
    assert payload["failure"]["category"] == "EXECUTION"
    assert payload["jobs"][0]["failure_category"] == "EXECUTION"


@pytest.mark.parametrize("response", [None, "sbatch output without a job id"])
def test_infer_malformed_slurm_response_is_execution_failure(
    tmp_path, monkeypatch, response
) -> None:
    config = _infer_config(tmp_path, backend="slurm")
    monkeypatch.setattr(
        "dpeva.submission.manager.JobManager.submit", lambda *a, **k: response
    )
    with pytest.raises(WorkflowError, match="all inference jobs failed"):
        InferenceWorkflow(
            config, run_options=RunOptions(run_id="infer-slurm-malformed")
        ).run()
    payload = json.loads(
        (config.work_dir / ".dpeva/runs/infer-slurm-malformed/run.json").read_text()
    )
    assert payload["status"] == "failed"
    assert payload["failure"]["category"] == "EXECUTION"
    assert payload["jobs"][0]["failure_category"] == "EXECUTION"


def test_infer_resume_of_submitted_slurm_rejects_without_new_job(tmp_path, monkeypatch) -> None:
    config = _infer_config(tmp_path, backend="slurm")
    monkeypatch.setattr(
        "dpeva.submission.manager.JobManager.submit",
        lambda *a, **k: "Submitted batch job 8124",
    )
    InferenceWorkflow(config, run_options=RunOptions(run_id="infer-slurm-resume")).run()
    calls = {"submit": 0}
    monkeypatch.setattr(
        "dpeva.submission.manager.JobManager.submit",
        lambda *a, **k: calls.__setitem__("submit", calls["submit"] + 1),
    )
    with pytest.raises(ValueError, match="submitted.*scheduler"):
        InferenceWorkflow(
            config,
            run_options=RunOptions(run_id="infer-slurm-resume", resume=True),
        ).run()
    payload = json.loads(
        (config.work_dir / ".dpeva/runs/infer-slurm-resume/run.json").read_text()
    )
    assert payload["status"] == "submitted"
    assert [job["job_id"] for job in payload["jobs"]] == ["8124"]
    assert calls["submit"] == 0


def test_infer_analysis_failure_preserves_artifacts_and_failed_state(tmp_path, monkeypatch) -> None:
    config = _infer_config(tmp_path)
    config.auto_analysis = True

    def submit(self, script_path, working_dir="."):
        Path(working_dir, "results.e.out").write_text("prediction\n")
        return ""

    monkeypatch.setattr("dpeva.submission.manager.JobManager.submit", submit)
    workflow = InferenceWorkflow(
        config,
        run_options=RunOptions(run_id="infer-analysis-failed"),
    )
    workflow.analyze_results = lambda: (_ for _ in ()).throw(RuntimeError("analysis failed"))
    with pytest.raises(RuntimeError, match="analysis failed"):
        workflow.run()
    payload = json.loads(
        (config.work_dir / ".dpeva/runs/infer-analysis-failed/run.json").read_text()
    )
    assert payload["status"] == "failed"
    assert payload["failure"]["category"] == "EXECUTION"
    assert payload["failure"]["message"] == "analysis failed"
    assert payload["artifacts"][0]["status"] == "verified"


def test_infer_mixed_artifact_and_execution_failures_are_deterministic(tmp_path, monkeypatch) -> None:
    config = _infer_config(tmp_path)
    model = config.work_dir / "1" / "model.ckpt.pt"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"model")
    model = config.work_dir / "2" / "model.ckpt.pt"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"model")

    def submit(self, script_path, working_dir="."):
        index = Path(working_dir).parts[-2]
        if index == "0":
            Path(working_dir, "results.e.out").write_text("prediction\n")
            return ""
        if index == "1":
            Path(working_dir, "results.e.out").touch()
            return ""
        raise subprocess.CalledProcessError(3, ["bash", str(script_path)])

    monkeypatch.setattr("dpeva.submission.manager.JobManager.submit", submit)
    with pytest.raises(PartialWorkflowError):
        InferenceWorkflow(config, run_options=RunOptions(run_id="infer-mixed-failures")).run()
    payload = json.loads(
        (config.work_dir / ".dpeva/runs/infer-mixed-failures/run.json").read_text()
    )
    assert payload["status"] == "partial"
    assert payload["failure"]["category"] == "EXECUTION"
    assert [job["failure_category"] for job in payload["jobs"][1:]] == ["ARTIFACT", "EXECUTION"]


def test_embed_validator_requires_exact_embedding_per_pool(tmp_path) -> None:
    for pool in ("pool0", "pool1"):
        pool_dir = tmp_path / pool
        pool_dir.mkdir()
        (pool_dir / "embedding.hdf5").write_bytes(b"hdf5")
    outputs = validate_feature_outputs(
        tmp_path, "embed", expected_pools=["pool0", "pool1"]
    )
    assert [path.name for path in outputs] == ["embedding.hdf5", "embedding.hdf5"]


@pytest.mark.parametrize("layout", ["missing", "wrong-location"])
def test_embed_validator_rejects_missing_or_recursive_output(tmp_path, layout) -> None:
    (tmp_path / "pool0").mkdir()
    if layout == "wrong-location":
        nested = tmp_path / "pool0" / "nested"
        nested.mkdir()
        (nested / "embedding.hdf5").write_bytes(b"hdf5")
    with pytest.raises(ArtifactValidationError, match="pool0"):
        validate_feature_outputs(tmp_path, "embed", expected_pools=["pool0"])
