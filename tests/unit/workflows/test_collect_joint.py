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
    
    @patch("dpeva.workflows.collect.CollectionWorkflow._log_initial_stats")
    @patch("dpeva.sampling.manager.SamplingManager.execute_sampling")
    @patch("dpeva.workflows.collect.UQVisualizer")
    @patch("dpeva.io.collection.load_systems")
    @patch("dpeva.uncertain.manager.UQManager.load_predictions")
    @patch("dpeva.uncertain.manager.UQManager.run_analysis")
    @patch("dpeva.uncertain.manager.UQManager.run_filtering")
    def test_joint_sampling_trigger(self, mock_filtering, mock_run_analysis, mock_load_preds, mock_load_sys, mock_vis, mock_execute_sampling, mock_log_stats, mock_config):
        """
        Verify that setting training_desc_dir triggers joint sampling mode.
        """
        # Setup mocks
        os.makedirs(mock_config["project"], exist_ok=True)
        os.makedirs(mock_config["desc_dir"], exist_ok=True)
        os.makedirs(mock_config["testdata_dir"], exist_ok=True)
        
        # Mock load_predictions
        pred_obj = MagicMock()
        pred_obj.dataname_list = [["c1-0", 10]]
        pred_obj.datanames_nframe = {"c1-0": 1}
        mock_load_preds.return_value = ([pred_obj], False)
        
        # Mock run_analysis
        mock_run_analysis.return_value = ({
            "uq_qbc_for": np.array([0.1]),
            "uq_rnd_for": np.array([0.1])
        }, np.array([0.1]))
        
        # Mock run_filtering to return non-empty candidate df WITH descriptor columns
        # Prepare descriptor columns
        cols = [f"desc_stru_{i}" for i in range(128)]
        data = {"dataname": ["c1-0"]}
        for c in cols:
            data[c] = np.random.rand(1)
        
        df_dummy_cand = pd.DataFrame(data)
        mock_filtering.return_value = (df_dummy_cand, pd.DataFrame(), pd.DataFrame(), MagicMock())
        
        # Mock execute_sampling return
        mock_execute_sampling.return_value = {
            "selected_indices": [0],
            "pca_features": np.random.rand(1, 2),
            "explained_variance": np.array([0.1]),
            "random_indices": [0],
            "scores_direct": np.array([0.5]),
            "scores_random": np.array([0.1]),
            "full_pca_features": np.random.rand(1, 2)
        }
        
        # Patch load_descriptors on IOManager
        with patch("dpeva.io.collection.CollectionIOManager.load_descriptors") as mock_load:
            # First call (candidate), Second call (training)
            mock_load.side_effect = [
                (["c1"], np.random.rand(1, 10)), # Candidate
                (["t1"], np.random.rand(1, 10))  # Training
            ]
            
            wf = CollectionWorkflow(mock_config)
            wf.run()
            
            # Verify execute_sampling was called with MERGED features and n_candidates
            mock_execute_sampling.assert_called()
            call_args = mock_execute_sampling.call_args
            args = call_args.args
            kwargs = call_args.kwargs
            
            # args[0] is features. 
            # Candidate: 1 frame. Training: 1 frame.
            # Merged should be 2 frames.
            features = args[0]
            assert features.shape[0] == 2, f"Expected 2 frames (1 cand + 1 train), got {features.shape[0]}"
            
            # Verify n_candidates passed
            # n_candidates is no longer passed to execute_sampling
            assert "n_candidates" not in kwargs
            
            # Verify n_candidates is set in manager
            assert wf.sampling_manager.n_candidates == 1
            
            # Verify load_descriptors was called twice (once for candidate, once for training)
            assert mock_load.call_count == 2

    def test_joint_sampling_no_training_data(self, mock_config):
        """
        Verify fallback to normal sampling if training_desc_dir is missing.
        """
        # Ensure project dir exists
        os.makedirs(mock_config["project"], exist_ok=True)
        os.makedirs(mock_config["desc_dir"], exist_ok=True)
        os.makedirs(mock_config["testdata_dir"], exist_ok=True)
    
        mock_config.pop("training_desc_dir")
        # Ensure we are not in joint mode config-wise
        
        # We need to mock _log_initial_stats, DIRECTSampler, UQVisualizer, load_systems, load_descriptors
        with patch("dpeva.workflows.collect.CollectionWorkflow._log_initial_stats"), \
             patch("dpeva.sampling.manager.DIRECTSampler") as mock_sampler, \
             patch("dpeva.workflows.collect.UQVisualizer"), \
             patch("dpeva.io.collection.load_systems"), \
             patch("dpeva.uncertain.manager.UQManager.load_predictions") as mock_load_preds, \
             patch("dpeva.uncertain.manager.UQManager.run_analysis") as mock_run_analysis, \
             patch("dpeva.uncertain.manager.UQManager.run_filtering") as mock_filtering, \
             patch("dpeva.io.collection.CollectionIOManager.load_descriptors") as mock_load_desc:
            
            # Mock load_predictions
            pred_obj = MagicMock()
            pred_obj.dataname_list = [["c1-0", 10]]
            pred_obj.datanames_nframe = {"c1-0": 1}
            mock_load_preds.return_value = ([pred_obj], False)
            
            # Mock run_analysis
            mock_run_analysis.return_value = ({
                "uq_qbc_for": np.array([0.1]),
                "uq_rnd_for": np.array([0.1])
            }, np.array([0.1]))
            
            # Mock run_filtering to return non-empty candidate df WITH columns
            cols = [f"desc_stru_{i}" for i in range(128)]
            data = {"dataname": ["c1-0"]}
            for c in cols:
                data[c] = np.random.rand(1)
            df_dummy_cand = pd.DataFrame(data)
            
            mock_filtering.return_value = (df_dummy_cand, pd.DataFrame(), pd.DataFrame(), MagicMock())
            
            mock_load_desc.return_value = (["c1"], np.random.rand(1, 10))
            
            mock_sampler_instance = MagicMock()
            mock_sampler.return_value = mock_sampler_instance
            mock_sampler_instance.fit_transform.return_value = {
                "selected_indices": [0], 
                "PCAfeatures": np.random.rand(1, 2)
            }
            mock_sampler_instance.pca.pca.explained_variance_ = np.array([10, 5])
            
            wf = CollectionWorkflow(mock_config)
            wf.run()
            
            # Verify fit_transform (normal) called, not fit_transform_joint
            mock_sampler_instance.fit_transform.assert_called()
            mock_sampler_instance.fit_transform_joint.assert_not_called()