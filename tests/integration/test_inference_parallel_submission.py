import os
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from dpeva.workflows.infer import InferenceWorkflow

@pytest.fixture
def slurm_config(tmp_path):
    """Setup config for Slurm testing."""
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    
    # Create 3 dummy models
    for i in range(3):
        (work_dir / str(i)).mkdir()
        (work_dir / str(i) / "model.ckpt.pt").touch()
    
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    config = {
        "work_dir": str(work_dir),
        "data_path": str(data_dir),
        "task_name": "test_task",
        "submission": {
            "backend": "slurm",
            "slurm_config": {
                "partition": "gpu",
                "ntasks": 1,
                "gpus_per_node": 1
            }
        }
    }
    
    return config, work_dir

def test_inference_parallel_submission(slurm_config):
    """Test that InferenceWorkflow submits parallel jobs in Slurm mode."""
    config_dict, work_dir = slurm_config
    
    # Mock JobManager BEFORE initializing workflow
    with patch("dpeva.inference.managers.JobManager") as MockJobManager:
        mock_job_manager_cls = MockJobManager.return_value
        # The JobManager() call returns an instance, we need to mock THAT instance
        mock_job_instance = MockJobManager.return_value
        
        # Initialize workflow
        workflow = InferenceWorkflow(config_dict)
        
        # Run workflow
        workflow.run()
        
        # Verify that submit was called 3 times (once per model)
        assert mock_job_instance.submit.call_count == 3
        
        # Verify script generation
        assert mock_job_instance.generate_script.call_count == 3
        
        # Check generated job configs
        # Get all call args to generate_script
        calls = mock_job_instance.generate_script.call_args_list
        
        for i, call in enumerate(calls):
            job_config = call[0][0]
            script_path = call[0][1]
            
            # Verify job name is unique
            assert job_config.job_name == f"dp_test_{i}"
            
            # Verify command calls dp test directly
            # Note: DPCommandBuilder might add --pt or other flags, so we check for key components
            assert "dp" in job_config.command
            assert "test" in job_config.command
            assert str(work_dir / str(i) / "model.ckpt.pt") in job_config.command
            
            # Verify output/error logs are generic (or task specific if implemented)
            # Current implementation uses "test_job.out" in task dir
            assert job_config.output_log == "test_job.out"
            
            # Verify Slurm parameters are passed
            assert job_config.partition == "gpu"
            assert job_config.gpus_per_node == 1
            
            # Verify script path is in task directory
            expected_task_dir = work_dir / str(i) / "test_task"
            assert str(expected_task_dir) in script_path
            
def test_inference_local_submission(slurm_config):
    """Test that InferenceWorkflow submits sequential jobs in Local mode."""
    config_dict, work_dir = slurm_config
    config_dict["submission"]["backend"] = "local"
    
    with patch("dpeva.inference.managers.JobManager") as MockJobManager:
        mock_job_instance = MockJobManager.return_value
        
        workflow = InferenceWorkflow(config_dict)
        workflow.run()
        
        # Should still call submit 3 times
        assert mock_job_instance.submit.call_count == 3
        
        # Verify backend mode
        assert workflow.execution_manager.backend == "local"
