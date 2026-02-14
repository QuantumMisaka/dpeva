import pytest
import os
from unittest.mock import MagicMock, patch
from dpeva.workflows.collect import CollectionWorkflow

def test_collect_single_pool_routing(real_config_loader, tmp_path):
    """
    Test routing for Single Data Pool configuration.
    Config: test/test-in-single-datapool/collect_config_single.json
    """
    # Load config
    config = real_config_loader(
        "test-in-single-datapool/collect_config_single.json",
        mock_data_mapping={
            "project": "project",
            "desc_dir": "desc_pool",
            "testdata_dir": "test_data",
            "root_savedir": "savedir"
        }
    )
    
    # Mock existence of dirs (Handled by real_config_loader or we ensure they exist)
    # (tmp_path / "test_data").mkdir(exist_ok=True)
    # (tmp_path / "desc_pool").mkdir(exist_ok=True)
    
    # Init workflow
    # We might need to mock UQManager if it does heavy lifting in __init__
    with patch("dpeva.workflows.collect.UQManager") as MockUQ:
        workflow = CollectionWorkflow(config)
        
        # Verify Single Pool characteristics
        # Config params are now in workflow.config
        assert workflow.config.uq_select_scheme == "tangent_lo"
        assert workflow.config.uq_trust_mode == "auto"
        
        # Check config attributes (using uq_trust_ratio instead of removed global_trust_ratio)
        assert workflow.config.uq_trust_ratio is not None

def test_collect_multi_pool_routing(real_config_loader, tmp_path):
    """
    Test routing for Multi Data Pool configuration.
    Config: test/test-for-multiple-datapool/joint_collect_config.json
    """
    config = real_config_loader(
        "test-for-multiple-datapool/joint_collect_config.json",
        mock_data_mapping={
             "project": "project",
             "desc_dir": "desc_pool",
             "testdata_dir": "test_data",
             "training_data_dir": "train_data", 
             "training_desc_dir": "desc_train",
             "root_savedir": "savedir"
        }
    )
    
    # (tmp_path / "test_data").mkdir(exist_ok=True)
    # (tmp_path / "train_data").mkdir(exist_ok=True)
    
    with patch("dpeva.workflows.collect.UQManager") as MockUQ:
        workflow = CollectionWorkflow(config)
        
        # Verify separation
        if workflow.config.training_data_dir:
            assert str(workflow.config.training_data_dir) == config["training_data_dir"]

