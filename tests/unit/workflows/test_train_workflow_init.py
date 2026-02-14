import pytest
import os
import json
from unittest.mock import patch, MagicMock
from dpeva.workflows.train import TrainingWorkflow

def test_training_workflow_init_multi_model(real_config_loader, tmp_path):
    """
    Test initialization using real multi-model config from test/test-v2-7.
    """
    # Load config and remap paths to tmp_path
    data_dir = tmp_path / "sampled_dpdata"
    data_dir.mkdir()
    
    # Create input.json
    input_json_path = tmp_path / "input.json"
    with open(input_json_path, "w") as f:
        json.dump({"training": {"numb_steps": 100}, "model": {"type_map": ["O", "H"]}}, f)
        
    config = real_config_loader(
        "test-v2-7/config.json",
        mock_data_mapping={
            "training_data_path": "sampled_dpdata"
        }
    )

    # Manually fix input_json_path which is relative in config
    config["input_json_path"] = str(input_json_path)
    config["work_dir"] = str(tmp_path)
    
    # Fix missing/renamed fields due to alias removal
    if "finetune_head_name" in config:
        config["model_head"] = config.pop("finetune_head_name")
    if "mode" in config:
        config["training_mode"] = config.pop("mode")

    # Patch Managers
    with patch("dpeva.workflows.train.TrainingConfigManager") as MockConfigManager, \
         patch("dpeva.workflows.train.TrainingExecutionManager") as MockExecutionManager, \
         patch("dpeva.workflows.train.TrainingIOManager") as MockIOManager:
         
        # Setup Mocks
        mock_config_manager = MockConfigManager.return_value
        mock_config_manager.prepare_task_configs.return_value = [{} for _ in range(4)]
        
        mock_io_manager = MockIOManager.return_value
        mock_io_manager.create_task_dir.side_effect = lambda i: os.path.join(str(tmp_path), str(i))
        mock_io_manager.copy_base_model.return_value = "model.ckpt"
        
        mock_exec_manager = MockExecutionManager.return_value
        mock_exec_manager.generate_script.return_value = "script.sh"

        # Initialize Workflow
        workflow = TrainingWorkflow(config)
        
        # Assertions on workflow state
        assert workflow.num_models == 4
        assert len(workflow.seeds) == 4
        assert workflow.mode == "init"
        
        # Run
        workflow.run()
        
        # Verify Interactions
        assert mock_io_manager.configure_logging.called
        assert mock_config_manager.prepare_task_configs.called
        assert mock_io_manager.create_task_dir.call_count == 4
        assert mock_io_manager.save_task_config.call_count == 4
        assert mock_io_manager.copy_base_model.call_count == 4
        assert mock_exec_manager.generate_script.call_count == 4
        assert mock_exec_manager.submit_jobs.called
