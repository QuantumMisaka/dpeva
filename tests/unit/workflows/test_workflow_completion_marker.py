from unittest.mock import MagicMock, patch

from dpeva.constants import WORKFLOW_FINISHED_TAG


def test_training_script_contains_completion_marker(tmp_path):
    with patch("dpeva.training.managers.JobManager") as MockJobManager:
        mock_job_manager = MockJobManager.return_value
        mock_job_manager.generate_script = MagicMock()

        from dpeva.training.managers import TrainingExecutionManager

        mgr = TrainingExecutionManager(
            backend="local",
            slurm_config={},
            env_setup="",
            dp_backend="pt",
            template_path=None,
        )

        mgr.generate_script(task_idx=0, task_dir=str(tmp_path), base_model_name=None, omp_threads=4)

        job_config = mock_job_manager.generate_script.call_args[0][0]
        assert "dp --pt train" in job_config.command
        assert job_config.command.index("dp --pt train") < job_config.command.index("test -s model.ckpt.pt")
        assert job_config.command.index("test -s lcurve.out") < job_config.command.index(WORKFLOW_FINISHED_TAG)


def test_inference_command_appends_completion_marker(tmp_path):
    model_path = tmp_path / "model.ckpt.pt"
    model_path.write_text("x", encoding="utf-8")
    data_path = tmp_path / "data"
    data_path.mkdir()

    with patch("dpeva.inference.managers.JobManager") as MockJobManager:
        mock_job_manager = MockJobManager.return_value
        mock_job_manager.generate_script = MagicMock()
        mock_job_manager.submit = MagicMock()

        from dpeva.inference.managers import InferenceExecutionManager

        mgr = InferenceExecutionManager(
            backend="local",
            slurm_config={},
            env_setup="",
            dp_backend="pt",
            omp_threads=4,
        )

        mgr.submit_jobs(
            models_paths=[str(model_path)],
            data_path=str(data_path),
            work_dir=str(tmp_path / "work"),
            task_name="test_val",
            head="results",
            results_prefix="results",
        )

        job_config = mock_job_manager.generate_script.call_args[0][0]
        assert "dp --pt test" in job_config.command
        assert job_config.command.index("dp --pt test") < job_config.command.index("compgen -G")
        assert job_config.command.index("compgen -G") < job_config.command.index(WORKFLOW_FINISHED_TAG)


def test_feature_eval_desc_command_appends_completion_marker(tmp_path):
    data_path = tmp_path / "data"
    data_path.mkdir()
    (data_path / "type.raw").write_text("0\n", encoding="utf-8")

    with patch("dpeva.feature.managers.JobManager") as MockJobManager:
        mock_job_manager = MockJobManager.return_value
        mock_job_manager.generate_script = MagicMock()
        mock_job_manager.submit = MagicMock()

        from dpeva.feature.managers import FeatureExecutionManager

        mgr = FeatureExecutionManager(
            backend="local",
            slurm_config={},
            env_setup="",
            dp_backend="pt",
            omp_threads=4,
        )

        mgr.submit_cli_job(
            data_path=str(data_path),
            output_dir=str(tmp_path / "desc"),
            model_path=str(tmp_path / "base.pt"),
            head="results",
            sub_pools=[],
            blocking=True,
        )

        job_config = mock_job_manager.generate_script.call_args[0][0]
        assert "dp --pt eval-desc" in job_config.command
        assert job_config.command.index("dp --pt eval-desc") < job_config.command.index("find ")
        assert job_config.command.index("find ") < job_config.command.index(WORKFLOW_FINISHED_TAG)


def test_feature_python_worker_script_contains_completion_marker(tmp_path):
    data_path = tmp_path / "data"
    data_path.mkdir()
    out_dir = tmp_path / "desc"

    with patch("dpeva.feature.managers.JobManager") as MockJobManager:
        mock_job_manager = MockJobManager.return_value
        mock_job_manager.submit_python_script = MagicMock()

        from dpeva.feature.managers import FeatureExecutionManager

        mgr = FeatureExecutionManager(
            backend="slurm",
            slurm_config={},
            env_setup="",
            dp_backend="pt",
            omp_threads=4,
        )

        mgr.submit_python_slurm_job(
            data_path=str(data_path),
            output_dir=str(out_dir),
            model_path=str(tmp_path / "base.pt"),
            head="results",
            batch_size=1,
            output_mode="atomic",
        )

        worker_script_content = mock_job_manager.submit_python_script.call_args[0][0]
        assert WORKFLOW_FINISHED_TAG in worker_script_content
        assert worker_script_content.index("run_local_python_recursion") < worker_script_content.index(WORKFLOW_FINISHED_TAG)
