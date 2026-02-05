import pytest
import numpy as np
import pandas as pd
import os
from unittest.mock import MagicMock, patch
from dpeva.workflows.collect import CollectionWorkflow, BirchClustering

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

    @patch("dpeva.workflows.collect.CollectionWorkflow._run_nofilter_setup")
    @patch("dpeva.workflows.collect.DIRECTSampler")
    @patch("dpeva.workflows.collect.UQVisualizer")
    @patch("dpeva.workflows.collect.load_systems")
    def test_direct_mode_explicit(self, mock_load_sys, mock_vis, mock_sampler, mock_nofilter, mock_config):
        """Test Explicit Mode: direct_n_clusters is set."""
        mock_config["direct_n_clusters"] = 10
        
        # Setup mocks
        os.makedirs(mock_config["project"], exist_ok=True)
        os.makedirs(mock_config["desc_dir"], exist_ok=True)
        os.makedirs(mock_config["testdata_dir"], exist_ok=True)
        
        df_dummy = pd.DataFrame({"dataname": ["s-0"], "uq_identity": ["candidate"]})
        stats_init = pd.DataFrame({"num_systems": [1], "num_frames": [1]}, index=["root"])
        stats_init.index.name = "pool_name"
        
        mock_nofilter.return_value = (df_dummy, df_dummy, df_dummy, ["s"], df_dummy, stats_init, False)
        
        mock_sampler_instance = MagicMock()
        mock_sampler.return_value = mock_sampler_instance
        mock_sampler_instance.fit_transform.return_value = {"selected_indices": [0], "PCAfeatures": np.random.rand(1, 2)}
        mock_sampler_instance.pca.pca.explained_variance_ = np.array([10, 5])
        
        mock_load_sys.return_value = []
        
        wf = CollectionWorkflow(mock_config)
        wf.run()
        
        # Verify
        # Check that BirchClustering was called with n=10
        # DIRECTSampler is instantiated in run(). We can check call args of DIRECTSampler
        # args[0] is structure_encoder (None), kwargs['clustering'] is Birch
        
        call_kwargs = mock_sampler.call_args.kwargs
        clustering_obj = call_kwargs['clustering']
        assert isinstance(clustering_obj, BirchClustering)
        assert clustering_obj.n == 10

    @patch("dpeva.workflows.collect.CollectionWorkflow._run_nofilter_setup")
    @patch("dpeva.workflows.collect.DIRECTSampler")
    @patch("dpeva.workflows.collect.UQVisualizer")
    @patch("dpeva.workflows.collect.load_systems")
    def test_direct_mode_dynamic(self, mock_load_sys, mock_vis, mock_sampler, mock_nofilter, mock_config):
        """Test Dynamic Mode: Both None."""
        # Both None
        
        # Setup mocks
        os.makedirs(mock_config["project"], exist_ok=True)
        os.makedirs(mock_config["desc_dir"], exist_ok=True)
        os.makedirs(mock_config["testdata_dir"], exist_ok=True)
        
        df_dummy = pd.DataFrame({"dataname": ["s-0"], "uq_identity": ["candidate"]})
        stats_init = pd.DataFrame({"num_systems": [1], "num_frames": [1]}, index=["root"])
        stats_init.index.name = "pool_name"
        mock_nofilter.return_value = (df_dummy, df_dummy, df_dummy, ["s"], df_dummy, stats_init, False)
        
        mock_sampler_instance = MagicMock()
        mock_sampler.return_value = mock_sampler_instance
        mock_sampler_instance.fit_transform.return_value = {"selected_indices": [0], "PCAfeatures": np.random.rand(1, 2)}
        mock_sampler_instance.pca.pca.explained_variance_ = np.array([10, 5])
        mock_load_sys.return_value = []
        
        wf = CollectionWorkflow(mock_config)
        wf.run()
        
        # Verify
        call_kwargs = mock_sampler.call_args.kwargs
        clustering_obj = call_kwargs['clustering']
        assert clustering_obj.n is None
