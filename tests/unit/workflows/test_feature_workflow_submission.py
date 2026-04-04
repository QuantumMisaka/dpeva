
import pytest
from unittest.mock import patch
from dpeva.workflows.feature import FeatureWorkflow

class TestFeatureWorkflowSubmission:

    @pytest.fixture
    def config(self, tmp_path):
        (tmp_path / "data").mkdir()
        (tmp_path / "model.pb").touch()
        return {
            "data_path": str(tmp_path / "data"),
            "model_path": str(tmp_path / "model.pb"),
            "model_head": "head",
            "savedir": str(tmp_path / "savedir"),
            "submission": {
                "backend": "slurm",
                "slurm_config": {"partition": "gpu"}
            }
        }

    @patch("dpeva.feature.managers.JobManager")
    def test_submit_cli_mode_slurm(self, MockJobManager, config, tmp_path):
        """
        Verify FeatureWorkflow CLI mode submission to Slurm.
        It should submit 'dp eval-desc' command.
        """
        config["mode"] = "cli"
        workflow = FeatureWorkflow(config)
        
        # Access the execution manager's job manager mock
        # Note: workflow.execution_manager initializes its own JobManager.
        # We need to mock the JobManager class used inside FeatureExecutionManager.
        
        # Run
        workflow.run()
        
        mock_job_manager = workflow.execution_manager.job_manager
        
        # Verify script generation
        mock_job_manager.generate_script.assert_called_once()
        job_config = mock_job_manager.generate_script.call_args[0][0]
        
        # Check command
        assert "dp" in job_config.command # dp eval-desc ...
        assert "eval-desc" in job_config.command # We can't check 'eval-desc' if using DPCommandBuilder alias, but command string should have it
        assert str(config["model_path"]) in job_config.command
        
        # Verify submission
        mock_job_manager.submit.assert_called_once()
        script_path = mock_job_manager.submit.call_args[0][0]
        assert script_path.endswith("run_evaldesc.slurm")

    @patch("dpeva.feature.managers.JobManager")
    def test_submit_python_mode_slurm(self, MockJobManager, config, tmp_path):
        """
        Verify FeatureWorkflow Python mode submission to Slurm.
        It should submit a python script (worker) that runs the recursion.
        """
        config["mode"] = "python"
        workflow = FeatureWorkflow(config)
        
        # Run
        workflow.run()
        
        mock_job_manager = workflow.execution_manager.job_manager
        
        # Verify python script submission
        # submit_python_script calls generate_script then submit
        mock_job_manager.submit_python_script.assert_called_once()
        
        args = mock_job_manager.submit_python_script.call_args
        worker_content = args[0][0]
        script_name = args[0][1]
        
        assert "from dpeva.feature.generator import DescriptorGenerator" in worker_content
        assert "run_local_python_recursion" in worker_content
        assert script_name == "run_desc_worker.py"

    @patch("dpeva.workflows.feature.DescriptorGenerator")
    def test_run_python_mode_local(self, MockGenerator, config, tmp_path):
        """
        Verify FeatureWorkflow Python mode local execution.
        It should run recursion locally without submission.
        """
        config["mode"] = "python"
        config["submission"]["backend"] = "local"
        
        # Mock execution manager's run_local_python_recursion to avoid FS operations
        with patch("dpeva.feature.managers.FeatureExecutionManager.run_local_python_recursion") as mock_recursion:
            workflow = FeatureWorkflow(config)
            workflow.run()
            
            mock_recursion.assert_called_once()
            # The MockGenerator here is the patched object from dpeva.workflows.feature
            # workflow.run calls DescriptorGenerator(...) which returns a Mock instance
            # That instance is passed to run_local_python_recursion
            
            assert mock_recursion.call_count == 1

    @patch("dpeva.feature.managers.JobManager")
    def test_run_python_mode_slurm_uses_worker_submission(self, MockJobManager, config):
        config["mode"] = "python"
        workflow = FeatureWorkflow(config)

        workflow.run()

        workflow.execution_manager.job_manager.submit_python_script.assert_called_once()
        workflow.execution_manager.job_manager.submit.assert_not_called()

    @patch("dpeva.feature.managers.JobManager")
    def test_cli_mode_passes_empty_sub_pools_when_single_pool(self, MockJobManager, config):
        config["mode"] = "cli"
        with patch("dpeva.workflows.feature.FeatureIOManager") as mock_io_cls:
            mock_io_cls.return_value.detect_multi_pool_structure.return_value = []
            workflow = FeatureWorkflow(config)
            workflow.run()

        job_config = workflow.execution_manager.job_manager.generate_script.call_args[0][0]
        assert "eval-desc" in job_config.command
        assert "DPEVA_TAG: WORKFLOW_FINISHED" in job_config.command
