import json
import logging
from unittest.mock import patch

import pytest

from dpeva.constants import WORKFLOW_FINISHED_TAG
from dpeva.utils.exceptions import WorkflowError


@pytest.fixture
def isolated_feature_loggers():
    loggers = [logging.getLogger(name) for name in ("dpeva", "dpeva.workflows.feature")]
    original = [
        (logger, logger.propagate, logger.handlers[:], logger.level)
        for logger in loggers
    ]
    try:
        for logger in loggers:
            logger.handlers.clear()
            logger.propagate = True
            logger.setLevel(logging.INFO)
        yield
    finally:
        for logger, propagate, handlers, level in original:
            for added in set(logger.handlers) - set(handlers):
                added.close()
            logger.handlers[:] = handlers
            logger.propagate = propagate
            logger.setLevel(level)


def test_slurm_training_submission_does_not_emit_completion_marker(tmp_path, caplog):
    from dpeva.workflows.train import TrainingWorkflow

    input_json = tmp_path / "input.json"
    input_json.write_text(json.dumps({"training": {}, "model": {}}), encoding="utf-8")
    config = {
        "base_model_path": str(tmp_path / "model.ckpt.pt"),
        "input_json_path": str(input_json),
        "work_dir": str(tmp_path / "work"),
        "num_models": 3,
        "model_head": "head",
        "training_mode": "init",
        "submission": {"backend": "slurm"},
    }
    (tmp_path / "model.ckpt.pt").write_text("model", encoding="utf-8")
    with patch("dpeva.workflows.train.setup_workflow_logger"), \
         patch("dpeva.workflows.train.TrainingExecutionManager") as manager_cls, \
         patch("dpeva.workflows.train.TrainingIOManager") as io_cls:
        io_cls.return_value.create_task_dir.return_value = str(tmp_path / "work" / "0")
        io_cls.return_value.copy_base_model.return_value = "model.ckpt.pt"
        manager_cls.return_value.generate_script.return_value = "train.slurm"
        TrainingWorkflow(config).run()
    assert WORKFLOW_FINISHED_TAG not in caplog.text


def test_feature_workflow_local_rejects_empty_recursion_output(
    tmp_path, caplog, isolated_feature_loggers
):
    from dpeva.workflows.feature import FeatureWorkflow

    data = tmp_path / "data"
    data.mkdir()
    (tmp_path / "model.pt").touch()
    config = {
        "data_path": str(data),
        "model_path": str(tmp_path / "model.pt"),
        "savedir": str(tmp_path / "out"),
        "mode": "python",
        "submission": {"backend": "local"},
    }
    with patch("dpeva.workflows.feature.DescriptorGenerator"), \
         patch("dpeva.feature.managers.FeatureExecutionManager.run_local_python_recursion"):
        from dpeva.run.artifacts import ArtifactValidationError

        with pytest.raises(ArtifactValidationError, match="missing or empty"):
            FeatureWorkflow(config).run()
    assert WORKFLOW_FINISHED_TAG not in caplog.text


def test_feature_workflow_propagates_recursion_failure_without_marker(
    tmp_path, caplog, isolated_feature_loggers
):
    from dpeva.workflows.feature import FeatureWorkflow

    data = tmp_path / "data"
    data.mkdir()
    (tmp_path / "model.pt").touch()
    config = {
        "data_path": str(data),
        "model_path": str(tmp_path / "model.pt"),
        "savedir": str(tmp_path / "out"),
        "mode": "python",
        "submission": {"backend": "local"},
    }
    with patch("dpeva.workflows.feature.DescriptorGenerator"), patch(
        "dpeva.feature.managers.FeatureExecutionManager.run_local_python_recursion",
        side_effect=WorkflowError("leaf system failed"),
    ):
        with pytest.raises(WorkflowError, match="leaf system failed"):
            FeatureWorkflow(config).run()
    assert WORKFLOW_FINISHED_TAG not in caplog.text


def test_feature_workflow_emits_one_marker_only_after_verified_finished(
    tmp_path, caplog, isolated_feature_loggers
):
    from dpeva.workflows.feature import FeatureWorkflow

    data = tmp_path / "data"
    data.mkdir()
    model = tmp_path / "model.pt"
    model.write_bytes(b"model")
    output = tmp_path / "out"
    config = {
        "data_path": str(data),
        "model_path": str(model),
        "savedir": str(output),
        "mode": "python",
        "submission": {"backend": "local"},
    }

    def produce(*args, **kwargs):
        output.mkdir(parents=True, exist_ok=True)
        (output / "features.npy").write_bytes(b"feature")

    caplog.set_level(logging.INFO)
    with patch("dpeva.workflows.feature.setup_workflow_logger"), patch(
        "dpeva.workflows.feature.DescriptorGenerator"
    ), patch(
        "dpeva.feature.managers.FeatureExecutionManager.run_local_python_recursion",
        side_effect=produce,
    ):
        FeatureWorkflow(config).run()

    markers = [record for record in caplog.records if record.message == WORKFLOW_FINISHED_TAG]
    assert len(markers) == 1
    manifests = list((output / ".dpeva" / "runs").glob("*/run.json"))
    payload = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert payload["status"] == "finished"
    assert [item["path"] for item in payload["artifacts"] if item["kind"] == "feature"] == [
        "features.npy"
    ]


@pytest.mark.parametrize("feature_exporter", ["eval_desc", "embed"])
def test_multi_pool_checks_require_every_pool(feature_exporter, tmp_path):
    from dpeva.feature.managers import FeatureExecutionManager

    with patch("dpeva.feature.managers.JobManager") as job_cls:
        manager = FeatureExecutionManager("local", {}, "", "pt", 1)
        manager.submit_cli_job(
            str(tmp_path / "data"), str(tmp_path / "out"), "model.pt", "head",
            ["pool1", "pool2"], feature_exporter=feature_exporter,
        )
        command = job_cls.return_value.generate_script.call_args.args[0].command
    assert command.count("grep -q .") == 2 if feature_exporter == "eval_desc" else command.count("test -s") == 2
