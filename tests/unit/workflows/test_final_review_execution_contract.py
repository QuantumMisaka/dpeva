import json
from unittest.mock import patch

import pytest

from dpeva.constants import WORKFLOW_FINISHED_TAG
from dpeva.utils.exceptions import WorkflowError


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


def test_feature_workflow_local_rejects_empty_recursion_output(tmp_path, caplog):
    from dpeva.workflows.feature import FeatureWorkflow

    data = tmp_path / "data"
    data.mkdir()
    config = {
        "data_path": str(data),
        "model_path": str(tmp_path / "model.pt"),
        "savedir": str(tmp_path / "out"),
        "mode": "python",
        "submission": {"backend": "local"},
    }
    with patch("dpeva.workflows.feature.DescriptorGenerator"), \
         patch("dpeva.feature.managers.FeatureExecutionManager.run_local_python_recursion"):
        with pytest.raises(WorkflowError, match="no non-empty"):
            FeatureWorkflow(config).run()
    assert WORKFLOW_FINISHED_TAG not in caplog.text


def test_feature_workflow_propagates_recursion_failure_without_marker(tmp_path, caplog):
    from dpeva.workflows.feature import FeatureWorkflow

    data = tmp_path / "data"
    data.mkdir()
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
