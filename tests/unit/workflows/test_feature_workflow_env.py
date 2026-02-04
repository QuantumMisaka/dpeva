import pytest
import os
from unittest.mock import patch, MagicMock
from dpeva.workflows.feature import FeatureWorkflow

def test_feature_workflow_cli_env(real_config_loader, tmp_path, mock_job_manager):
    """
    Test CLI mode feature generation with environment setup.
    Config: test/desc-test/config_cli_test.json
    """
    config = real_config_loader(
        "desc-test/config_cli_test.json",
        mock_data_mapping={
            "datadir": "data",
            "modelpath": "model.pt",
            "savedir": "savedir"
        }
    )

    # Polyfill data_path which FeatureWorkflow expects
    config["data_path"] = config["datadir"]
    
    # Fix aliases
    if "modelpath" in config:
        config["model_path"] = config.pop("modelpath")
    if "head" in config:
        config["model_head"] = config.pop("head")
    
    # Mock data dir and model
    # (tmp_path / "data").mkdir(exist_ok=True) # Handled by loader
    if not (tmp_path / "model.pt").exists():
        (tmp_path / "model.pt").touch()
    
    # Create a mock system to trigger processing
    (tmp_path / "data" / "sys1").mkdir()
    
    # Mock DescriptorGenerator since FeatureWorkflow delegates to it
    # But FeatureWorkflow handles the orchestration and JobManager config passing.
    # Let's see if we can test Workflow directly.
    # If FeatureWorkflow calls generator.run(), we mock generator.
    
    with patch("dpeva.workflows.feature.DescriptorGenerator") as MockGen:
        workflow = FeatureWorkflow(config)
        
        # Verify mode
        assert workflow.mode == "cli"
        
        workflow.run()
        
        # Verify generator init
        assert MockGen.call_count == 1
        _, kwargs = MockGen.call_args
        
        # Check environment setup passed to generator
        assert "source /opt/envs/deepmd3.1.2.env" in kwargs["env_setup"]
        assert "export DP_INTERFACE_PREC=high" in kwargs["env_setup"]
        
        # Verify generator run called
        assert MockGen.return_value.run_cli_generation.called
        
        # Check args
        args = MockGen.return_value.run_cli_generation.call_args
        assert args[0][0] == config["data_path"]
        assert args[0][1] == config["savedir"]

def test_feature_workflow_python_mode_data_format(tmp_path):
    """Test Python mode passes data_format correctly."""
    config_dict = {
        "data_path": str(tmp_path / "data"),
        "model_path": str(tmp_path / "model.pt"),
        "model_head": "OC20M",
        "mode": "python",
        "savedir": str(tmp_path / "savedir")
    }
    
    (tmp_path / "data").mkdir()
    (tmp_path / "model.pt").touch()
    
    with patch("dpeva.workflows.feature.DescriptorGenerator") as MockGen:
        workflow = FeatureWorkflow(config_dict)
        
        workflow.run()
        
        MockGen.return_value.run_python_generation.assert_called_with(
            data_path=str(tmp_path / "data"),
            output_dir=str(tmp_path / "savedir"),
            data_format="auto",
            output_mode="atomic"
        )
