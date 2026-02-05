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

    @patch("dpeva.workflows.collect.CollectionWorkflow._load_descriptors")
    @patch("dpeva.workflows.collect.CollectionWorkflow._load_atomic_features_for_candidates")
    @patch("dpeva.workflows.collect.TwoStepDIRECTSampler")
    @patch("dpeva.workflows.collect.UQVisualizer")
    @patch("dpeva.workflows.collect.load_systems")
    def test_run_2direct(self, mock_load_sys, mock_vis, mock_sampler_cls, mock_load_atomic, mock_load_desc, mock_config):
        os.makedirs(mock_config["project"], exist_ok=True)
        os.makedirs(mock_config["desc_dir"], exist_ok=True)
        os.makedirs(mock_config["testdata_dir"], exist_ok=True)
        
        # Mock _load_descriptors (Step 1 structural features)
        # Returns: desc_datanames, desc_stru
        n_frames = 10
        desc_datanames = [f"sys-{i}" for i in range(n_frames)]
        desc_stru = np.random.rand(n_frames, 10)
        mock_load_desc.return_value = (desc_datanames, desc_stru)
        
        # Mock _load_atomic_features_for_candidates
        X_atom_list = [np.random.rand(3, 5) for _ in range(n_frames)]
        n_atoms_list = [3] * n_frames
        mock_load_atomic.return_value = (X_atom_list, n_atoms_list)
        
        # Mock TwoStepDIRECTSampler instance
        mock_sampler_instance = MagicMock()
        mock_sampler_cls.return_value = mock_sampler_instance
        
        # fit_transform returns dict
        mock_sampler_instance.fit_transform.return_value = {
            "selected_indices": [0, 1],
            "step1_labels": [0]*5 + [1]*5,
            "PCAfeatures": np.random.rand(n_frames, 2)
        }
        
        # Mock step1_sampler.pca.pca.explained_variance_ for visualization
        mock_sampler_instance.step1_sampler.pca.pca.explained_variance_ = np.array([10, 5, 1])
        
        # Mock Visualizer
        mock_vis_instance = MagicMock()
        mock_vis.return_value = mock_vis_instance
        mock_vis_instance.plot_pca_analysis.return_value = pd.DataFrame()
        
        # Mock load_systems
        mock_load_sys.return_value = [] # Skip export logic
        
        wf = CollectionWorkflow(mock_config)
        wf.run()
        
        # Assertions
        mock_sampler_cls.assert_called_once()
        mock_load_atomic.assert_called_once()
        mock_sampler_instance.fit_transform.assert_called_once()
        
        # Check args to fit_transform
        call_args = mock_sampler_instance.fit_transform.call_args
        assert len(call_args[0]) == 3 # X_stru, X_atom_list, n_atoms_list
