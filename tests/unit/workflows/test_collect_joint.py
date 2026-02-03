import pytest
import numpy as np
import pandas as pd
import os
from unittest.mock import MagicMock, patch
from dpeva.workflows.collect import CollectionWorkflow

class TestCollectionWorkflowJoint:
    
    @pytest.fixture
    def mock_config(self, tmp_path):
        return {
            "project": str(tmp_path / "project"),
            "testdata_dir": str(tmp_path / "testdata"),
            "desc_dir": str(tmp_path / "desc_dir"), # Required key
            "training_desc_dir": str(tmp_path / "training_desc"), # Triggers joint sampling
            "backend": "local",
            "uq_qbc_trust_lo": 0.1,
            "uq_qbc_trust_hi": 0.3,
            "uq_rnd_rescaled_trust_lo": 0.1,
            "uq_rnd_rescaled_trust_hi": 0.3,
            "uq_trust_width": 0.2,
            "uq_trust_mode": "manual",
            "select_n": 10
        }

    # _load_candidate_descriptors seems to be internal. 
    # Let's check source code if it exists or if logic is inline.
    # From log: AttributeError: ... does not have the attribute '_load_candidate_descriptors'
    # It seems I assumed this method exists but it might not.
    # In collect.py, candidate loading is likely done inside _perform_direct_selection or similar.
    
    # We will patch methods that definitely exist or are imported.
    # 'dpeva.workflows.collect.DIRECTSampler' is good.
    # For data loading, maybe we can mock pd.read_csv or np.load? 
    # Or patch the method that calls them.
    
    # Let's try to patch `_perform_direct_selection` to see if it triggers joint logic.
    # But wait, we want to test INSIDE `_perform_direct_selection`.
    
    # Let's patch `dpeva.workflows.collect.load_descriptors` if it exists, or `np.load`.
    # Actually, in previous `run_uq_collect.py` logs, we saw:
    # "Loading candidate descriptors from ..."
    # "Loading training descriptors from ..."
    
    # These logs come from `_perform_direct_selection` (or similar).
    
    @patch("dpeva.workflows.collect.CollectionWorkflow._load_descriptors")
    def test_joint_sampling_trigger(self, mock_load_desc, mock_config):
        """
        Verify that joint sampling logic is triggered when training_desc_dir is present.
        """
        # Ensure project dir exists for validation
        os.makedirs(mock_config["project"], exist_ok=True)
        os.makedirs(mock_config["desc_dir"], exist_ok=True)
        os.makedirs(mock_config["testdata_dir"], exist_ok=True)
        
        # Mock loading training descriptors (second call to _load_descriptors usually, or specific call)
        # _load_descriptors returns (names, data)
        # We need it to return something for training data
        
        def side_effect(desc_dir, *args, **kwargs):
            if desc_dir == mock_config["training_desc_dir"]:
                return [], np.random.rand(5, 128) # 5 training frames
            return [], np.random.rand(10, 128) # default
            
        mock_load_desc.side_effect = side_effect
        
        # Initialize workflow
        wf = CollectionWorkflow(mock_config)
        
        # Prepare inputs for _prepare_features_for_direct
        # Create candidate dataframe with features efficiently to avoid PerformanceWarning
        cols = [f"desc_stru_{i}" for i in range(128)]
        
        data = {"idx": range(2)}
        # Add random features to data dict
        for c in cols:
            data[c] = np.random.rand(2)
            
        df_candidate = pd.DataFrame(data)
        
        # Mock df_desc columns
        df_desc = pd.DataFrame(columns=cols)
            
        # Run method
        features, use_joint, n_cand = wf._prepare_features_for_direct(df_candidate, df_desc)
        
        # Assertions
        assert use_joint is True
        assert n_cand == 2
        # Total features = 2 candidate + 5 training = 7
        assert features.shape == (7, 128)
        
        # Verify _load_descriptors was called with training dir
        # Check call args
        calls = mock_load_desc.call_args_list
        # Should be at least one call with training_desc_dir
        training_call_found = False
        for call in calls:
            args, _ = call
            if args[0] == mock_config["training_desc_dir"]:
                training_call_found = True
                break
        assert training_call_found

    def test_joint_sampling_no_training_data(self, mock_config):
        """
        Verify fallback to normal sampling if training_desc_dir is missing.
        """
        # Ensure project dir exists
        os.makedirs(mock_config["project"], exist_ok=True)
        os.makedirs(mock_config["desc_dir"], exist_ok=True)
        os.makedirs(mock_config["testdata_dir"], exist_ok=True)
        
        mock_config.pop("training_desc_dir")
        wf = CollectionWorkflow(mock_config)
        
        # Prepare inputs
        cols = [f"desc_stru_{i}" for i in range(128)]
        data = {"idx": range(2)}
        for c in cols:
            data[c] = np.random.rand(2)
        
        df_candidate = pd.DataFrame(data)
        df_desc = pd.DataFrame(columns=cols)
            
        with patch("dpeva.workflows.collect.CollectionWorkflow._load_descriptors") as mock_load:
            features, use_joint, n_cand = wf._prepare_features_for_direct(df_candidate, df_desc)
            
            assert use_joint is False
            assert features.shape == (2, 128)
            assert not mock_load.called