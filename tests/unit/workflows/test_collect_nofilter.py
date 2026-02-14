import pytest
import numpy as np
import pandas as pd
import os
from unittest.mock import MagicMock, patch
from dpeva.workflows.collect import CollectionWorkflow
from dpeva.sampling.clustering import BirchClustering

class TestCollectionWorkflowDirectModes:
    
    @pytest.fixture
    def mock_config(self, tmp_path):
        return {
            "project": str(tmp_path / "project"),
            "testdata_dir": str(tmp_path / "testdata"),
            "desc_dir": str(tmp_path / "desc_dir"),
            "backend": "local",
            "uq_trust_mode": "no_filter",
            "direct_k": 1,
            # We will set n_clusters/num_selection in specific tests
        }

    @patch("dpeva.workflows.collect.CollectionWorkflow._log_initial_stats")
    @patch("dpeva.sampling.manager.DIRECTSampler")
    @patch("dpeva.workflows.collect.UQVisualizer")
    @patch("dpeva.io.collection.load_systems")
    def test_direct_mode_explicit(self, mock_load_sys, mock_vis, mock_sampler, mock_log_stats, mock_config):
        """Test Explicit Mode: direct_n_clusters is set."""
        mock_config["direct_n_clusters"] = 10
        
        # Setup mocks
        os.makedirs(mock_config["project"], exist_ok=True)
        os.makedirs(mock_config["desc_dir"], exist_ok=True)
        os.makedirs(mock_config["testdata_dir"], exist_ok=True)
        
        # Mock load_descriptors via IOManager patch or simple return
        # Since we can't easily patch IOManager method on instance, we rely on patching load_systems if used,
        # OR we patch CollectionIOManager.load_descriptors
        
        with patch("dpeva.io.collection.CollectionIOManager.load_descriptors") as mock_load_desc:
            mock_load_desc.return_value = (["s-0"], np.random.rand(1, 10))
            
            mock_sampler_instance = MagicMock()
            mock_sampler.return_value = mock_sampler_instance
            mock_sampler_instance.fit_transform.return_value = {"selected_indices": [0], "PCAfeatures": np.random.rand(1, 2)}
            mock_sampler_instance.pca.pca.explained_variance_ = np.array([10, 5])
            
            wf = CollectionWorkflow(mock_config)
            wf.run()
            
            # Verify
            call_kwargs = mock_sampler.call_args.kwargs
            clustering_obj = call_kwargs['clustering']
            assert isinstance(clustering_obj, BirchClustering)
            assert clustering_obj.n == 10

    @patch("dpeva.workflows.collect.CollectionWorkflow._log_initial_stats")
    @patch("dpeva.sampling.manager.DIRECTSampler")
    @patch("dpeva.workflows.collect.UQVisualizer")
    @patch("dpeva.io.collection.load_systems")
    def test_direct_mode_dynamic(self, mock_load_sys, mock_vis, mock_sampler, mock_log_stats, mock_config):
        """Test Dynamic Mode: Both None."""
        # Both None
        
        # Setup mocks
        os.makedirs(mock_config["project"], exist_ok=True)
        os.makedirs(mock_config["desc_dir"], exist_ok=True)
        os.makedirs(mock_config["testdata_dir"], exist_ok=True)
        
        with patch("dpeva.io.collection.CollectionIOManager.load_descriptors") as mock_load_desc:
            mock_load_desc.return_value = (["s-0"], np.random.rand(1, 10))
            
            mock_sampler_instance = MagicMock()
            mock_sampler.return_value = mock_sampler_instance
            mock_sampler_instance.fit_transform.return_value = {"selected_indices": [0], "PCAfeatures": np.random.rand(1, 2)}
            mock_sampler_instance.pca.pca.explained_variance_ = np.array([10, 5])
            
            wf = CollectionWorkflow(mock_config)
            wf.run()
            
            # Verify
            call_kwargs = mock_sampler.call_args.kwargs
            clustering_obj = call_kwargs['clustering']
            assert clustering_obj.n is None
