
import pytest
import logging
import os
from pydantic import ValidationError
from dpeva.workflows.collect import CollectionWorkflow

# Mock config for testing
@pytest.fixture
def base_config(tmp_path):
    desc_dir = tmp_path / "desc_dir"
    desc_dir.mkdir()
    testdata_dir = tmp_path / "testdata"
    testdata_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    
    return {
        "project": str(project_dir),
        "desc_dir": str(desc_dir),
        "testdata_dir": str(testdata_dir),
        "uq_select_scheme": "tangent_lo",
        "backend": "local"
    }

def test_manual_mode_missing_lo(base_config):
    """Test that manual mode requires 'lo'."""
    config = base_config.copy()
    config["uq_trust_mode"] = "manual"
    # Missing lo
    
    with pytest.raises(ValidationError) as excinfo:
        CollectionWorkflow(config)
    assert "uq_qbc_trust_lo must be provided" in str(excinfo.value)

def test_manual_mode_valid(base_config):
    """Test that manual mode works with 'lo' and 'width'."""
    config = base_config.copy()
    config["uq_trust_mode"] = "manual"
    config["uq_qbc_trust_lo"] = 0.1
    config["uq_qbc_trust_width"] = 0.1
    config["uq_rnd_rescaled_trust_lo"] = 0.1
    config["uq_rnd_rescaled_trust_width"] = 0.1
    
    wf = CollectionWorkflow(config)
    assert wf.uq_qbc_trust_lo == 0.1
    assert wf.uq_qbc_trust_hi == 0.2
    assert wf.uq_rnd_trust_lo == 0.1

def test_auto_mode_valid(base_config):
    """Test that auto mode works without 'lo'."""
    config = base_config.copy()
    config["uq_trust_mode"] = "auto"
    config["uq_trust_ratio"] = 0.5
    config["uq_trust_width"] = 0.2
    
    wf = CollectionWorkflow(config)
    # In auto mode, lo should be None initially
    assert wf.uq_qbc_trust_lo is None
    assert wf.uq_rnd_trust_lo is None
    # Params should be stored
    assert wf.uq_qbc_params["ratio"] == 0.5
    assert wf.uq_qbc_params["width"] == 0.2

def test_default_mode_fallback(base_config):
    """Test default mode is now auto (Pydantic default)."""
    config = base_config.copy()
    # No uq_trust_mode provided, should default to auto
    
    wf = CollectionWorkflow(config)
    assert wf.uq_trust_mode == "auto"

def test_slurm_missing_config_path(base_config):
    """Test that Slurm backend requires config_path."""
    config = base_config.copy()
    config["backend"] = "slurm"
    # Provide valid UQ params to pass init validation
    config["uq_trust_mode"] = "manual"
    config["uq_qbc_trust_lo"] = 0.1
    config["uq_qbc_trust_width"] = 0.1
    config["uq_rnd_rescaled_trust_lo"] = 0.1
    config["uq_rnd_rescaled_trust_width"] = 0.1
    
    # The check is in run(), not init.
    wf = CollectionWorkflow(config)
    
    with pytest.raises(ValueError, match="config_path is missing"):
        wf.run()

def test_slurm_valid_config_path(base_config):
    """Test that Slurm backend accepts config_path."""
    config = base_config.copy()
    config["backend"] = "slurm"
    # Provide valid UQ params
    config["uq_trust_mode"] = "manual"
    config["uq_qbc_trust_lo"] = 0.1
    config["uq_qbc_trust_width"] = 0.1
    config["uq_rnd_rescaled_trust_lo"] = 0.1
    config["uq_rnd_rescaled_trust_width"] = 0.1
    
    # We just check init doesn't fail.
    # We can't easily call run() because it tries to submit.
    # But we can verify backend is set.
    wf = CollectionWorkflow(config, config_path="dummy_config.json")
    assert wf.backend == "slurm"
    assert wf.config_path == "dummy_config.json"
