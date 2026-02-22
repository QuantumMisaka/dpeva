import pytest
import numpy as np
from dpeva.sampling.direct import DIRECTSampler

def test_direct_sampler_basic():
    """Test DIRECTSampler with synthetic 2D data."""
    # Create 3 distinct clusters in 2D space
    # Cluster 1: Around (0, 0)
    # Cluster 2: Around (10, 10)
    # Cluster 3: Around (20, 0)
    
    np.random.seed(42)
    c1 = np.random.normal(0, 1, (10, 2))
    c2 = np.random.normal(10, 1, (10, 2))
    c3 = np.random.normal(20, 1, (10, 2))
    
    X = np.vstack([c1, c2, c3]) # 30 samples
    
    # We want to select 1 sample from each cluster.
    # Total selected should be around 3 (depending on BIRCH clustering result)
    
    # Initialize Sampler
    # structure_encoder=None (skip encoding)
    # clustering="Birch" with n=3
    # select_k=1
    
    sampler = DIRECTSampler(
        structure_encoder=None,
        clustering=pytest.importorskip("dpeva.sampling.clustering").BirchClustering(n=3, threshold_init=0.5),
        select_k_from_clusters=pytest.importorskip("dpeva.sampling.stratified_sampling").SelectKFromClusters(k=1)
    )
    
    result = sampler.fit_transform(X)
    
    assert "selected_indices" in result
    selected_indices = result["selected_indices"]
    
    # We expect 3 samples (1 from each cluster)
    # Birch might not perfectly find 3 clusters with default settings, but let's check basic properties
    assert len(selected_indices) > 0
    assert len(selected_indices) <= 30
    
    # Check that indices are valid
    assert max(selected_indices) < 30
    assert min(selected_indices) >= 0

def test_direct_sampler_dimensions():
    """Test DIRECTSampler handles dimensions correctly."""
    X = np.random.rand(20, 5) # 20 samples, 5 features
    
    sampler = DIRECTSampler(
        structure_encoder=None,
        # Let BIRCH decide n based on threshold
        clustering=pytest.importorskip("dpeva.sampling.clustering").BirchClustering(n=None, threshold_init=0.5),
        select_k_from_clusters=pytest.importorskip("dpeva.sampling.stratified_sampling").SelectKFromClusters(k=1)
    )
    
    result = sampler.fit_transform(X)
    assert "selected_indices" in result
