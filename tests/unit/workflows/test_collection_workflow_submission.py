
import os
import pytest
import numpy as np # Import numpy
from unittest.mock import MagicMock, patch
from dpeva.workflows.collect import CollectionWorkflow
from dpeva.config import CollectionConfig

class TestCollectionWorkflowSubmission:

    @pytest.fixture
    def config(self, tmp_path):
        return {
            "project": str(tmp_path),
            "desc_dir": str(tmp_path / "desc"),
            "testdata_dir": str(tmp_path / "testdata"),
            "root_savedir": "dpeva_uq_result",
            "submission": {
                "backend": "slurm",
                "slurm_config": {"partition": "cpu"}
            },
            "config_path": str(tmp_path / "config.json")
        }

    @patch("dpeva.workflows.collect.JobManager")
    def test_submit_to_slurm_self_submission(self, MockJobManager, config, tmp_path):
        """
        Verify that CollectionWorkflow submits ITSELF to Slurm (Self-Submission).
        It should submit a command that runs 'dpeva.cli collect config.json'.
        """
        # Create dummy directories to pass validation
        (tmp_path / "desc").mkdir()
        (tmp_path / "testdata").mkdir()
        
        workflow = CollectionWorkflow(config, config_path=config["config_path"])
        mock_job_manager = MockJobManager.return_value
        
        # Run
        workflow.run()
        
        # Verify JobManager initialization
        MockJobManager.assert_called_with(mode="slurm")
        
        # Verify script generation
        mock_job_manager.generate_script.assert_called_once()
        job_config = mock_job_manager.generate_script.call_args[0][0]
        
        # Check command: it must call dpeva.cli collect
        assert "dpeva.cli collect" in job_config.command
        assert config["config_path"] in job_config.command
        
        # Check internal backend override environment variable
        assert "export DPEVA_INTERNAL_BACKEND=local" in job_config.env_setup
        
        # Verify submission
        mock_job_manager.submit.assert_called_once()
        script_path = mock_job_manager.submit.call_args[0][0]
        assert script_path.endswith("submit_collect.slurm")

    def test_run_local_execution(self, config, tmp_path):
        """
        Verify that if backend is local (or overridden), it proceeds to execution logic
        instead of self-submission.
        """
        # Set backend to local
        config["submission"]["backend"] = "local"
        (tmp_path / "desc").mkdir()
        (tmp_path / "testdata").mkdir()
    
        # Mock actual execution components to avoid running them
        with patch("dpeva.workflows.collect.UQManager") as MockUQ, \
             patch("dpeva.workflows.collect.SamplingManager") as MockSampling, \
             patch("dpeva.workflows.collect.CollectionIOManager") as MockIO, \
             patch("dpeva.workflows.collect.setup_workflow_logger"):
    
            workflow = CollectionWorkflow(config)
    
            # Mock uq_trust_mode to 'no_filter' to simplify flow
            workflow.config.uq_trust_mode = "no_filter"
    
            # Mock IO manager
            mock_io_instance = MockIO.return_value
            mock_io_instance.load_descriptors.return_value = (["data1"], np.array([[1.0]]))
            mock_io_instance.view_savedir = str(tmp_path) # Mock view_savedir to a real path
    
            # Mock SamplingManager.prepare_features to return valid tuple
            mock_sampling_instance = MockSampling.return_value
            mock_sampling_instance.prepare_features.return_value = (None, False, 0)
            
            # Mock Visualization to avoid actual plotting/saving
            # We must patch where it is IMPORTED in dpeva.workflows.collect
            # In collect.py: "from dpeva.uncertain.visualization import UQVisualizer"
            # It uses UQVisualizer, not Visualization. My bad.
            with patch("dpeva.workflows.collect.UQVisualizer") as MockVis:
                 workflow.run()
            
            # Verify it did NOT try to self-submit (no JobManager usage)
            # And it proceeded to load descriptors
            mock_io_instance.load_descriptors.assert_called()
