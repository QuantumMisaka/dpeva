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
    # We map 'training_data_path' to a temp dir
    data_dir = tmp_path / "sampled_dpdata"
    data_dir.mkdir()
    
    # We also need input.json to exist (TrainingWorkflow reads it via Trainer or passes path)
    # Actually TrainingWorkflow passes config to Trainer.
    # Let's check config content.
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
    
    # Initialize Workflow
    workflow = TrainingWorkflow(config)
    
    # Assertions on workflow state
    assert workflow.num_models == 4
    assert len(workflow.seeds) == 4
    assert workflow.mode == "init"
    
    # Test run logic (partial)
    # We mock ParallelTrainer to verify it gets called correctly
    with patch("dpeva.workflows.train.ParallelTrainer") as MockTrainer:
        workflow.run()
        
        # Verify Trainer was initialized with correct paths
        assert MockTrainer.call_count == 1
        _, kwargs = MockTrainer.call_args
        assert kwargs["num_models"] == 4
        assert kwargs["work_dir"] == str(tmp_path)
        
        # Verify trainer.train() was called
        assert MockTrainer.return_value.train.called
