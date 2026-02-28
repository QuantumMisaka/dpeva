
import pytest
import numpy as np
import logging
from unittest.mock import MagicMock, patch
from dpeva.sampling.clustering import BirchClustering

class TestBirchClustering:
    
    def test_fit_n_clusters_iteration(self):
        """Test that threshold decreases to reach n clusters."""
        X = np.random.rand(100, 2)
        
        clusterer = BirchClustering(n=5, threshold_init=10.0, max_iter=10)
        
        with patch("dpeva.sampling.clustering.Birch") as mock_birch:
            # Simulation:
            # Call 1 (Initial): 1 cluster
            m1 = MagicMock()
            m1.subcluster_labels_ = [0]*100 
            m1.fit.return_value = m1
            
            # Call 2 (Iter 0): 2 clusters
            m2 = MagicMock()
            m2.subcluster_labels_ = [0]*50 + [1]*50
            m2.fit.return_value = m2
            
            # Call 3 (Iter 1): 5 clusters (Success)
            m3 = MagicMock()
            m3.subcluster_labels_ = [0, 1, 2, 3, 4]*20
            m3.fit.return_value = m3
            m3.predict.return_value = np.zeros(100)
            m3.subcluster_centers_ = np.zeros((5, 2))
            
            mock_birch.side_effect = [m1, m2, m3]
            
            clusterer.transform(X)
            
            assert mock_birch.call_count == 3

    def test_min_threshold_break(self, caplog):
        """Test breaking when threshold is too low."""
        X = np.random.rand(10, 2)
        # Init < Min
        clusterer = BirchClustering(n=10, threshold_init=1e-4, min_threshold=1e-3)
        
        with patch("dpeva.sampling.clustering.Birch") as mock_birch:
             m = MagicMock()
             m.subcluster_labels_ = [0] # 1 cluster
             m.fit.return_value = m
             m.predict.return_value = np.zeros(10)
             m.subcluster_centers_ = np.zeros((1, 2))
             
             mock_birch.return_value = m
             
             with caplog.at_level(logging.WARNING):
                 clusterer.transform(X)
             
             assert "dropped below min_threshold" in caplog.text

    def test_max_iter_break(self, caplog):
        """Test breaking when max_iter reached."""
        X = np.random.rand(10, 2)
        clusterer = BirchClustering(n=10, threshold_init=1.0, max_iter=2)
        
        with patch("dpeva.sampling.clustering.Birch") as mock_birch:
             m = MagicMock()
             m.subcluster_labels_ = [0] # Always 1 cluster
             m.fit.return_value = m
             m.predict.return_value = np.zeros(10)
             m.subcluster_centers_ = np.zeros((1, 2))
             
             mock_birch.return_value = m
             
             with caplog.at_level(logging.WARNING):
                 clusterer.transform(X)
                 
             assert "failed to reach target" in caplog.text
             assert mock_birch.call_count == 3 # Initial + 2 iterations
