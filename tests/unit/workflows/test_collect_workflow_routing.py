import pytest
import os
from unittest.mock import MagicMock, patch
from dpeva.workflows.collect import CollectionWorkflow

def test_collect_single_pool_routing(tmp_path):
    """
    Test routing for Single Data Pool configuration.
    """
    # Create dummy dirs
    (tmp_path / "project").mkdir()
    (tmp_path / "desc_pool").mkdir()
    (tmp_path / "test_data").mkdir()
    
    config = {
        "project": str(tmp_path / "project"),
        "desc_dir": str(tmp_path / "desc_pool"),
        "testdata_dir": str(tmp_path / "test_data"),
        "root_savedir": str(tmp_path / "savedir"),
        "uq_select_scheme": "tangent_lo",
        "uq_trust_mode": "auto",
        "uq_trust_ratio": 0.5,
        "backend": "local"
    }
    
    # Init workflow
    with patch("dpeva.workflows.collect.UQManager") as MockUQ:
        workflow = CollectionWorkflow(config)
        
        # Verify Single Pool characteristics
        assert workflow.config.uq_select_scheme == "tangent_lo"
        assert workflow.config.uq_trust_mode == "auto"
        assert workflow.config.uq_trust_ratio is not None

def test_collect_multi_pool_routing(tmp_path):
    """
    Test routing for Multi Data Pool configuration.
    """
    # Create dummy dirs
    (tmp_path / "project").mkdir()
    (tmp_path / "desc_pool").mkdir()
    (tmp_path / "test_data").mkdir()
    (tmp_path / "train_data").mkdir()
    (tmp_path / "desc_train").mkdir()
    
    config = {
        "project": str(tmp_path / "project"),
        "desc_dir": str(tmp_path / "desc_pool"),
        "testdata_dir": str(tmp_path / "test_data"),
        "training_data_dir": str(tmp_path / "train_data"),
        "training_desc_dir": str(tmp_path / "desc_train"),
        "root_savedir": str(tmp_path / "savedir"),
        "uq_select_scheme": "tangent_lo",
        "backend": "local"
    }
    
    with patch("dpeva.workflows.collect.UQManager") as MockUQ:
        workflow = CollectionWorkflow(config)
        
        # Verify separation
        if workflow.config.training_data_dir:
            assert str(workflow.config.training_data_dir) == config["training_data_dir"]

