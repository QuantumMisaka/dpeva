import pytest
import pandas as pd
import numpy as np
import os
from unittest.mock import MagicMock, patch
from dpeva.workflows.collect import CollectionWorkflow

class TestCollectionWorkflow2Direct:
    
    @pytest.fixture
    def mock_config(self, tmp_path):
        return {
            "project": str(tmp_path / "project"),
            "testdata_dir": str(tmp_path / "testdata"),
            "desc_dir": str(tmp_path / "desc_dir"),
            "backend": "local",
            "uq_trust_mode": "no_filter",
            "sampler_type": "2-direct",
            "step1_n_clusters": 2,
            "step2_n_clusters": 2,
            "step2_k": 1,
            "num_selection": 5
        }

    @patch("dpeva.workflows.collect.CollectionWorkflow._log_initial_stats")
    @patch("dpeva.sampling.manager.SamplingManager.execute_sampling")
    @patch("dpeva.workflows.collect.UQVisualizer")
    @patch("dpeva.io.collection.load_systems")
    @patch("dpeva.io.collection.CollectionIOManager.load_atomic_features")
    @patch("dpeva.uncertain.manager.UQManager.load_predictions")
    def test_run_2direct(self, mock_load_preds, mock_load_atomic, mock_load_sys, mock_vis, mock_execute_sampling, mock_log_stats, mock_config):
        """
        Test the main run logic for 2-step DIRECT.
        """
        # Ensure config sets sampler_type to 2-direct
        mock_config["sampler_type"] = "2-direct"
        
        # Setup mocks
        os.makedirs(mock_config["project"], exist_ok=True)
        os.makedirs(mock_config["desc_dir"], exist_ok=True)
        os.makedirs(mock_config["testdata_dir"], exist_ok=True)
        
        # Mock load_predictions (return dummy predictions)
        # We need at least 2 models for UQ
        pred_data_1 = MagicMock()
        pred_data_1.force = {'pred_fx': np.random.rand(10, 1), 'pred_fy': np.random.rand(10, 1), 'pred_fz': np.random.rand(10, 1)}
        pred_data_1.dataname_list = [["s-0", 0, 10]]
        pred_data_1.datanames_nframe = {"s-0": 1}
        pred_data_1.has_ground_truth = False
        
        pred_data_2 = MagicMock()
        pred_data_2.force = {'pred_fx': np.random.rand(10, 1), 'pred_fy': np.random.rand(10, 1), 'pred_fz': np.random.rand(10, 1)}
        pred_data_2.dataname_list = [["s-0", 0, 10]]
        pred_data_2.datanames_nframe = {"s-0": 1}
        pred_data_2.has_ground_truth = False

        mock_load_preds.return_value = ([pred_data_1, pred_data_2], False)
        
        # Mock load_atomic_features
        n_frames = 10
        datanames = [f"s-{i}" for i in range(n_frames)]
        
        X_atom_list = [np.random.rand(10, 5) for _ in range(n_frames)]
        n_atoms_list = [10] * n_frames
        mock_load_atomic.return_value = (X_atom_list, n_atoms_list)
        
        # Mock execute_sampling return
        mock_execute_sampling.return_value = {
            "selected_indices": [0],
            "pca_features": np.random.rand(10, 2),
            "explained_variance": np.array([0.1]),
            "random_indices": [0],
            "scores_direct": np.array([0.5]),
            "scores_random": np.array([0.1]),
            "full_pca_features": np.random.rand(10, 2)
        }

        # Mock IO Manager's load_descriptors
        with patch("dpeva.io.collection.CollectionIOManager.load_descriptors") as mock_load_desc:
            mock_load_desc.return_value = (datanames, np.random.rand(n_frames, 10))
            
            wf = CollectionWorkflow(mock_config)
            wf.run()
            
            # Verify that execute_sampling was called
            # This implicitly confirms that the workflow reached the sampling stage
            mock_execute_sampling.assert_called()
            
            # Optionally check if load_atomic_features was called, which is specific to 2-direct
            mock_load_atomic.assert_called()
