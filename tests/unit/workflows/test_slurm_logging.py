
import pytest
import logging
import os
from unittest.mock import patch, MagicMock
from dpeva.workflows.collect import CollectionWorkflow

@pytest.fixture
def reset_dpeva_logger():
    """Resets the dpeva logger to a clean state."""
    logger = logging.getLogger("dpeva")
    logger.handlers = []
    logger.propagate = True
    return logger

@pytest.fixture
def base_config(tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    desc_dir = tmp_path / "desc"
    desc_dir.mkdir()
    testdata_dir = tmp_path / "testdata"
    testdata_dir.mkdir()
    
    return {
        "project": str(project_dir),
        "desc_dir": str(desc_dir),
        "testdata_dir": str(testdata_dir),
        "uq_trust_mode": "manual",
        "uq_qbc_trust_lo": 0.1,
        "uq_qbc_trust_width": 0.1,
        "uq_rnd_rescaled_trust_lo": 0.1,
        "uq_rnd_rescaled_trust_width": 0.1,
        "config_path": "dummy_config.json"
    }

def test_slurm_submission_logging(base_config, reset_dpeva_logger, caplog):
    """
    Test that when backend='slurm':
    1. configure_logging is skipped (propagate remains True).
    2. Logs are generated during submission.
    """
    config = base_config.copy()
    config["backend"] = "slurm"
    
    # Initialize Workflow
    # Should NOT call configure_logging
    # Ensure backend override is cleared
    with patch.dict(os.environ):
        if "DPEVA_INTERNAL_BACKEND" in os.environ:
            del os.environ["DPEVA_INTERNAL_BACKEND"]
            
        wf = CollectionWorkflow(config)
    
        dpeva_logger = logging.getLogger("dpeva")
        assert dpeva_logger.propagate is True, "Propagate should be True for slurm backend"
        # Note: File handler might exist if previous tests didn't clean up, but we reset it.
        assert len(dpeva_logger.handlers) == 0, "No file handlers should be added for slurm backend"
        
        # Mock JobManager
        with patch("dpeva.workflows.collect.JobManager") as MockManager:
            instance = MockManager.return_value
            instance.submit.return_value = "Submitted batch job 12345"
            
            # Run Workflow (Submission)
            with caplog.at_level(logging.INFO):
                wf.run()
                
            # Verify JobManager was called
            instance.submit.assert_called_once()
            
            # Verify our explicit log message appears
            assert "CollectionWorkflow submitted successfully to Slurm" in caplog.text
            assert "Job script:" in caplog.text

def test_local_backend_logging(base_config, reset_dpeva_logger):
    """Test that local backend DOES configure logging."""
    config = base_config.copy()
    config["backend"] = "local"
    
    wf = CollectionWorkflow(config)
    
    dpeva_logger = logging.getLogger("dpeva")
    # Should call configure_logging, which sets propagate=False
    assert dpeva_logger.propagate is False, "Propagate should be False for local backend"
    # Should add a FileHandler
    assert len(dpeva_logger.handlers) >= 1, "File handler should be added"
