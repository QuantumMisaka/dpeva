
import pytest
import subprocess
from unittest.mock import patch

from dpeva.inference.managers import InferenceExecutionManager
from dpeva.compatibility import DeepMDAdapter


def _replace_dp_command(generated: str, replacement: str) -> str:
    command = next(line for line in generated.splitlines() if line.startswith("dp "))
    return generated.replace(command, replacement, 1)

class TestInferenceExecutionManager:

    @pytest.fixture
    def mock_job_manager(self):
        with patch("dpeva.inference.managers.JobManager") as mock:
            yield mock

    @pytest.fixture
    def manager_slurm(self, mock_job_manager):
        return InferenceExecutionManager(
            backend="slurm",
            slurm_config={"partition": "gpu"},
            env_setup="export FOO=bar",
            dp_backend="pt",
            omp_threads=4
        )

    @pytest.fixture
    def manager_local(self, mock_job_manager):
        return InferenceExecutionManager(
            backend="local",
            slurm_config={},
            env_setup="",
            dp_backend="pt",
            omp_threads=4
        )

    def test_submit_jobs_slurm_parallel(self, manager_slurm, tmp_path):
        """
        Verify that Slurm backend submits jobs in parallel (one job per model).
        This ensures the 'One-Job-Per-Model' design.
        """
        # Setup dummy models
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        models = []
        for i in range(3):
            p = work_dir / f"model_{i}.pt"
            p.touch()
            models.append(str(p))

        # Call submit_jobs
        manager_slurm.submit_jobs(
            models_paths=models,
            data_path=str(tmp_path / "data"),
            work_dir=str(work_dir),
            task_name="test_task",
            head="head",
            results_prefix="res"
        )

        # Verify JobManager was initialized correctly
        assert manager_slurm.job_manager.generate_script.call_count == 3
        
        # Verify submit was called 3 times (parallel submission loop)
        assert manager_slurm.job_manager.submit.call_count == 3
        
        # Verify submission arguments
        # We expect 3 calls, each with a different script path
        calls = manager_slurm.job_manager.submit.call_args_list
        assert len(calls) == 3
        
        # Check that we are submitting different scripts in different directories
        submitted_dirs = [c[1]['working_dir'] for c in calls]
        assert len(set(submitted_dirs)) == 3
        assert all("test_task" in d for d in submitted_dirs)

        # Verify JobConfig details for one of the models
        # manager_slurm.job_manager.generate_script was called 3 times.
        gen_calls = manager_slurm.job_manager.generate_script.call_args_list
        job_config = gen_calls[0][0][0]
        
        # DeepMDAdapter builds the model/system command before the output guard.
        assert "dp --pt test" in job_config.command
        assert str(work_dir / "model_0.pt") in job_config.command
        assert "-s" in job_config.command
        assert "res" in job_config.command
        assert "head" in job_config.command
        assert "compgen -G" in job_config.command
        assert job_config.command.index("compgen -G") < job_config.command.index("DPEVA_TAG: WORKFLOW_FINISHED")

    def test_submit_jobs_local_sequential(self, manager_local, tmp_path):
        """
        Verify that Local backend submits jobs (sequentially in current impl).
        """
        # Setup dummy models
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        models = []
        for i in range(2):
            p = work_dir / f"model_{i}.pt"
            p.touch()
            models.append(str(p))

        # Call submit_jobs
        manager_local.submit_jobs(
            models_paths=models,
            data_path=str(tmp_path / "data"),
            work_dir=str(work_dir),
            task_name="test_task",
            head="head",
            results_prefix="res"
        )

        # Verify submit was called 2 times
        assert manager_local.job_manager.submit.call_count == 2

    def test_empty_inference_artifact_does_not_emit_finished(self, manager_local, tmp_path):
        work_dir = tmp_path / "work"
        model = work_dir / "model_0.pt"
        work_dir.mkdir()
        model.touch()

        manager_local.submit_jobs(
            models_paths=[str(model)],
            data_path=str(tmp_path / "data"),
            work_dir=str(work_dir),
            task_name="task",
            head="head",
            results_prefix="results",
        )

        task_dir = work_dir / "0" / "task"
        (task_dir / "results.0.out").touch()
        generated = manager_local.job_manager.generate_script.call_args[0][0].command
        command = _replace_dp_command(generated, "true")
        result = subprocess.run(
            ["bash", "-c", "set -Eeuo pipefail\n" + command],
            cwd=task_dir,
            text=True,
            capture_output=True,
        )

        assert result.returncode != 0
        assert "DPEVA_TAG: WORKFLOW_FINISHED" not in result.stdout

    def test_stale_nonempty_inference_artifact_does_not_emit_finished(self, manager_local, tmp_path):
        work_dir = tmp_path / "work"
        model = work_dir / "model_0.pt"
        work_dir.mkdir()
        model.touch()

        manager_local.submit_jobs(
            models_paths=[str(model)],
            data_path=str(tmp_path / "data"),
            work_dir=str(work_dir),
            task_name="task",
            head="head",
            results_prefix="results",
        )

        task_dir = work_dir / "0" / "task"
        (task_dir / "results.0.out").write_text("prediction\n", encoding="utf-8")
        generated = manager_local.job_manager.generate_script.call_args[0][0].command
        command = _replace_dp_command(generated, "true")
        result = subprocess.run(
            ["bash", "-c", "set -Eeuo pipefail\n" + command],
            cwd=task_dir,
            text=True,
            capture_output=True,
        )

        assert result.returncode != 0
        assert "DPEVA_TAG: WORKFLOW_FINISHED" not in result.stdout

    def test_identical_inference_rewrite_emits_finished(self, manager_local, tmp_path):
        work_dir = tmp_path / "work"
        model = work_dir / "model_0.pt"
        work_dir.mkdir()
        model.touch()

        manager_local.submit_jobs(
            models_paths=[str(model)],
            data_path=str(tmp_path / "data"),
            work_dir=str(work_dir),
            task_name="task",
            head="head",
            results_prefix="results",
        )

        task_dir = work_dir / "0" / "task"
        output = task_dir / "results.0.out"
        output.write_text("prediction\n", encoding="utf-8")
        generated = manager_local.job_manager.generate_script.call_args[0][0].command
        command = _replace_dp_command(
            generated, "printf 'prediction\\n' > results.0.out"
        )
        result = subprocess.run(
            ["bash", "-c", "set -Eeuo pipefail\n" + command],
            cwd=task_dir,
            text=True,
            capture_output=True,
        )

        assert result.returncode == 0
        assert "DPEVA_TAG: WORKFLOW_FINISHED" in result.stdout

    def test_injected_adapter_owns_command_backend(self, manager_local, tmp_path):
        manager_local.adapter = DeepMDAdapter.for_legacy_unchecked("tf")
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        model = work_dir / "model_0.pt"
        model.touch()

        manager_local.submit_jobs(
            models_paths=[str(model)],
            data_path=str(tmp_path / "data"),
            work_dir=str(work_dir),
            task_name="task",
            head="head",
            results_prefix="res",
        )

        command = manager_local.job_manager.generate_script.call_args[0][0].command
        assert "dp --tf test" in command

    def test_submit_jobs_uses_default_env_setup_for_local_backend(self, manager_local, tmp_path):
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        model = work_dir / "model_0.pt"
        model.touch()

        manager_local.submit_jobs(
            models_paths=[str(model)],
            data_path=str(tmp_path / "data"),
            work_dir=str(work_dir),
            task_name="task",
            head="head",
            results_prefix="res",
        )

        job_config = manager_local.job_manager.generate_script.call_args[0][0]
        assert job_config.env_setup == "export OMP_NUM_THREADS=4"

    def test_submit_jobs_skips_missing_models_and_logs_warning(self, manager_local, tmp_path):
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        valid_model = work_dir / "model_0.pt"
        valid_model.touch()
        missing_model = work_dir / "model_1.pt"

        with patch.object(manager_local.logger, "warning") as mock_warning:
            manager_local.submit_jobs(
                models_paths=[str(valid_model), str(missing_model)],
                data_path=str(tmp_path / "data"),
                work_dir=str(work_dir),
                task_name="task",
                head="head",
                results_prefix="res",
            )

        assert manager_local.job_manager.generate_script.call_count == 1
        assert manager_local.job_manager.submit.call_count == 1
        mock_warning.assert_called_once()

    def test_submit_jobs_without_task_name_uses_model_index_directory(self, manager_local, tmp_path):
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        model = work_dir / "model_0.pt"
        model.touch()

        manager_local.submit_jobs(
            models_paths=[str(model)],
            data_path=str(tmp_path / "data"),
            work_dir=str(work_dir),
            task_name="",
            head="head",
            results_prefix="res",
        )

        submit_kwargs = manager_local.job_manager.submit.call_args.kwargs
        assert submit_kwargs["working_dir"] == str(work_dir / "0")

    def test_submit_jobs_generates_local_shell_script(self, manager_local, tmp_path):
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        model = work_dir / "model_0.pt"
        model.touch()

        manager_local.submit_jobs(
            models_paths=[str(model)],
            data_path=str(tmp_path / "data"),
            work_dir=str(work_dir),
            task_name="task",
            head="head",
            results_prefix="res",
        )

        script_path = manager_local.job_manager.generate_script.call_args[0][1]
        assert script_path.endswith("run_test.sh")
