import pytest
import os
from unittest.mock import MagicMock, patch
from dpeva.workflows.infer import InferenceWorkflow

def create_mock_models(base_dir, num_models=3):
    """Helper to create dummy model files."""
    for i in range(num_models):
        model_dir = base_dir / str(i)
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "model.ckpt.pt").touch()

def test_init_model_discovery(tmp_path):
    """Verify workflow finds models in 0/, 1/ subdirectories."""
    create_mock_models(tmp_path, num_models=3)
    
    config = {
        "work_dir": str(tmp_path),
        "data_path": str(tmp_path / "data"), # Dummy
        "task_name": "test_task"
    }
    
    workflow = InferenceWorkflow(config)
    assert len(workflow.models_paths) == 3
    # Check if paths are correct (order might vary so check set or existence)
    expected_paths = {str(tmp_path / str(i) / "model.ckpt.pt") for i in range(3)}
    assert set(workflow.models_paths) == expected_paths

def test_run_command_generation(tmp_path, mock_job_manager):
    """Verify dp test command construction."""
    create_mock_models(tmp_path, num_models=1)
    data_path = tmp_path / "test_data"
    data_path.mkdir()
    
    config = {
        "work_dir": str(tmp_path),
        "data_path": str(data_path),
        "task_name": "test_val",
        "model_head": "MyHead",
        "submission": {"backend": "slurm"}
    }
    
    workflow = InferenceWorkflow(config)
    workflow.run()
    
    # Verify JobManager.submit was called
    assert mock_job_manager.generate_script.called
    assert mock_job_manager.submit.called
    
    # Inspect the JobConfig passed to generate_script
    call_args = mock_job_manager.generate_script.call_args
    job_config = call_args[0][0] # First arg is job_config
    
    # Command path checks
    assert str(data_path) in job_config.command
    assert str(tmp_path / "0" / "model.ckpt.pt") in job_config.command
    assert "--head MyHead" in job_config.command
    
    assert "-d results" in job_config.command

def test_no_models_found(tmp_path, caplog):
    """Verify error logging when no models are found."""
    import logging
    caplog.set_level(logging.ERROR)

    data_path = tmp_path / "data"
    data_path.mkdir(exist_ok=True)
    
    config = {
        "work_dir": str(tmp_path),
        "data_path": str(data_path)
    }
    
    workflow = InferenceWorkflow(config)
    
    # Use patch to verify logging call, avoiding caplog issues if propagation is disabled
    with patch.object(workflow.logger, 'error') as mock_error:
        workflow.run()
        mock_error.assert_called_with("No models provided for inference.")
