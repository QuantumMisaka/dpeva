
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from dpeva.sampling.manager import SamplingManager

@pytest.fixture
def sampling_manager():
    config = {
        "sampler_type": "direct",
        "direct_n_clusters": 5,
        "direct_k": 1
    }
    return SamplingManager(config)

def test_prepare_features_joint(sampling_manager):
    df_cand = pd.DataFrame({"desc_stru_0": [1, 2], "desc_stru_1": [3, 4]})
    df_desc = df_cand.copy() # Columns match
    
    train_feat = np.array([[5, 6]])
    
    feat, joint, n = sampling_manager.prepare_features(df_cand, df_desc, train_feat)
    
    assert joint is True
    assert feat.shape == (3, 2) # 2 cand + 1 train
    assert n == 2

def test_execute_sampling_direct(sampling_manager):
    features = np.random.rand(10, 5)
    
    with patch("dpeva.sampling.manager.DIRECTSampler") as mock_sampler:
        mock_instance = mock_sampler.return_value
        mock_instance.fit_transform.return_value = {
            "selected_indices": [0, 1],
            "PCAfeatures": np.zeros((10, 2))
        }
        mock_instance.pca.pca.explained_variance_ = np.array([1.0, 0.5])
        
        res = sampling_manager.execute_sampling(features)
        
        idx = res["selected_indices"]
        assert len(idx) == 2
        mock_instance.fit_transform.assert_called_once()

def test_execute_sampling_joint():
    config = {
        "sampler_type": "direct",
        "direct_n_clusters": 5,
        "direct_k": 1,
        "direct_thr_init": 0.1
    }
    manager = SamplingManager(config)
    
    # 10 candidate features, 5 training features (appended)
    # Total features passed to execute_sampling is 15
    features = np.random.rand(15, 5)
    background_features = np.random.rand(5, 5)
    
    with patch("dpeva.sampling.manager.DIRECTSampler") as mock_sampler:
        mock_instance = mock_sampler.return_value
        # Mock return: include some training data indices (>= 10)
        # 0, 1 are candidates; 12, 14 are training data
        mock_instance.fit_transform.return_value = {
            "selected_indices": [0, 1, 12, 14], 
            "PCAfeatures": np.zeros((15, 2))
        }
        # Need to mock pca attribute for background transform
        mock_pca = MagicMock()
        mock_pca.transform.return_value = np.zeros((5, 2))
        mock_pca.pca.explained_variance_ = np.array([1.0, 0.5])
        mock_instance.pca = mock_pca
        
        manager.n_candidates = 10
        res = manager.execute_sampling(features, background_features=background_features)
        
        # Verify fit_transform called WITHOUT n_candidates
        mock_instance.fit_transform.assert_called_once()
        args, kwargs = mock_instance.fit_transform.call_args
        assert "n_candidates" not in kwargs, "n_candidates should not be passed to fit_transform"
        
        # Verify filtering: indices >= 10 should be removed
        selected = res["selected_indices"]
        assert selected == [0, 1]
        
        assert "full_pca_features" in res
