import os
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from dpeva.workflows.infer import InferenceWorkflow

@pytest.fixture
def mock_slurm_env(tmp_path):
    """Setup mock environment for Slurm testing."""
    # Create dummy models
    models_dir = tmp_path / "work"
    models_dir.mkdir()
    (models_dir / "0").mkdir()
    (models_dir / "0" / "model.ckpt.pt").touch()
    
    # Create dummy data
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    config = {
        "work_dir": str(models_dir),
        "data_path": str(data_dir),
        "task_name": "test_task",
        "submission": {
            "backend": "slurm",
            "slurm_config": {
                "partition": "gpu",
                "ntasks": 1
            }
        }
    }
    
    config_path = tmp_path / "infer.json"
    with open(config_path, "w") as f:
        json.dump(config, f)
        
    return config, config_path

def test_inference_self_submission(mock_slurm_env):
    """Test that InferenceWorkflow self-submits when in Slurm mode."""
    config_dict, config_path = mock_slurm_env
    
    # Initialize workflow with config_path
    workflow = InferenceWorkflow(config_dict, config_path=str(config_path))
    
    # Mock JobManager to intercept submission
    with patch("dpeva.workflows.infer.JobManager") as MockJobManager:
        mock_manager_instance = MockJobManager.return_value
        
        # Run workflow
        # Ensure DPEVA_INTERNAL_BACKEND is NOT set
        with patch.dict(os.environ, {}, clear=True):
             # Also need to ensure os.environ.get("DPEVA_INTERNAL_BACKEND") returns None
             # The patch above clears env, but let's be explicit if needed.
             pass
             
        workflow.run()
        
        # Verify self-submission
        assert mock_manager_instance.generate_script.called
        assert mock_manager_instance.submit.called
        
        # Check script content in JobConfig
        call_args = mock_manager_instance.generate_script.call_args
        job_config = call_args[0][0]
        
        assert "dpeva.cli infer" in job_config.command
        assert str(config_path) in job_config.command
        assert "export DPEVA_INTERNAL_BACKEND=local" in job_config.env_setup

def test_inference_internal_execution(mock_slurm_env):
    """Test that InferenceWorkflow executes logic when running internally."""
    config_dict, config_path = mock_slurm_env
    
    workflow = InferenceWorkflow(config_dict, config_path=str(config_path))
    
    # Mock ExecutionManager to verify actual jobs are submitted
    with patch.object(workflow.execution_manager, "submit_jobs") as mock_submit:
        # Simulate internal backend environment
        with patch.dict(os.environ, {"DPEVA_INTERNAL_BACKEND": "local"}):
            workflow.run()
            
        # Should call submit_jobs, NOT self-submit
        assert mock_submit.called
        assert workflow.execution_manager.backend == "local"
