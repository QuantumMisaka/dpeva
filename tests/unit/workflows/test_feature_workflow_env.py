import pytest
from unittest.mock import patch, MagicMock
from dpeva.workflows.feature import FeatureWorkflow

def test_feature_workflow_cli_env(tmp_path):
    """
    Test CLI mode feature generation with environment setup.
    """
    config_dict = {
        "data_path": str(tmp_path / "data"),
        "model_path": str(tmp_path / "model.pt"),
        "model_head": "OC20M",
        "mode": "cli",
        "savedir": str(tmp_path / "savedir"),
        "submission": {
            "backend": "local",
            "env_setup": "export TEST=1"
        }
    }
    
    (tmp_path / "data").mkdir()
    (tmp_path / "model.pt").touch()
    
    with patch("dpeva.workflows.feature.FeatureExecutionManager") as MockExec:
        with patch("dpeva.workflows.feature.FeatureIOManager") as MockIO:
            workflow = FeatureWorkflow(config_dict)
            
            # Mock IO return
            MockIO.return_value.detect_multi_pool_structure.return_value = ["pool1"]
            
            workflow.run()
            
            # Verify Execution Manager Init
            MockExec.assert_called_once()
            _, kwargs = MockExec.call_args
            assert kwargs["env_setup"] == "export TEST=1"
            
            # Verify submit_cli_job called
            exec_instance = MockExec.return_value
            exec_instance.submit_cli_job.assert_called_with(
                data_path=str(tmp_path / "data"),
                output_dir=str(tmp_path / "savedir"),
                model_path=str(tmp_path / "model.pt"),
                head="OC20M",
                sub_pools=["pool1"]
            )

def test_feature_workflow_python_mode(tmp_path):
    """Test Python mode orchestration."""
    config_dict = {
        "data_path": str(tmp_path / "data"),
        "model_path": str(tmp_path / "model.pt"),
        "model_head": "OC20M",
        "mode": "python",
        "savedir": str(tmp_path / "savedir"),
        "submission": {
            "backend": "local"
        }
    }
    
    (tmp_path / "data").mkdir()
    
    with patch("dpeva.workflows.feature.FeatureExecutionManager") as MockExec:
        # Mock backend attribute
        exec_instance = MockExec.return_value
        exec_instance.backend = "local"
        
        with patch("dpeva.workflows.feature.DescriptorGenerator") as MockGen:
            workflow = FeatureWorkflow(config_dict)
            workflow.run()
            
            # Verify Generator Init
            MockGen.assert_called_with(
                model_path=str(tmp_path / "model.pt"),
                head="OC20M",
                batch_size=1000,
                omp_threads=4 # Default
            )
            
            # Verify run_local_python_recursion called
            exec_instance.run_local_python_recursion.assert_called()
