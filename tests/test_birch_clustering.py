
import unittest
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
import logging

# Ensure logging is configured to see output if needed
logging.basicConfig(level=logging.INFO)

from dpeva.sampling.clustering import BirchClustering

class TestBirchClustering(unittest.TestCase):
    def setUp(self):
        # Create a logger capture
        self.logger = logging.getLogger("dpeva.sampling.clustering")
        self.logger.setLevel(logging.INFO)
        
    def test_empty_input(self):
        """Test that empty input raises ValueError."""
        birch = BirchClustering(n=10)
        with self.assertRaises(ValueError):
            birch.transform(np.array([]))
            
    def test_insufficient_samples(self):
        """Test behavior when n_samples < n_clusters."""
        X = np.random.rand(5, 2)
        birch = BirchClustering(n=10)
        
        with self.assertLogs("dpeva.sampling.clustering", level="WARNING") as cm:
            res = birch.transform(X)
            
        # Check if warning was logged
        logs = [o for o in cm.output]
        found = any("Number of samples (5) is less than target clusters (10)" in o for o in logs)
        if not found:
            print(f"Captured logs: {logs}")
        self.assertTrue(found, f"Warning not found in logs: {logs}")
        
        # Check if n was adjusted to 5 (since each sample becomes a cluster)
        self.assertEqual(len(set(res["labels"])), 5)

    def test_convergence_warning_suppression(self):
        """Test that sklearn ConvergenceWarning is suppressed."""
        X = np.random.rand(100, 2)
        # Use aggressive parameters to force warning
        birch = BirchClustering(n=50, threshold_init=0.5, max_iter=5)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always") # Cause all warnings to always be triggered.
            birch.transform(X)
            
            # Check for ConvergenceWarning
            convergence_warnings = [x for x in w if issubclass(x.category, ConvergenceWarning)]
            if len(convergence_warnings) > 0:
                print(f"Captured warnings: {[str(x.message) for x in convergence_warnings]}")
            self.assertEqual(len(convergence_warnings), 0, "ConvergenceWarning should be suppressed")

    def test_threshold_iteration(self):
        """Test that threshold is decreased when target clusters not reached."""
        X = np.random.rand(100, 2)
        birch = BirchClustering(n=10, threshold_init=10.0, max_iter=10)
        
        with self.assertLogs("dpeva.sampling.clustering", level="INFO") as cm:
            res = birch.transform(X)
            
        # Check logs for iteration
        iteration_logs = [o for o in cm.output if "Iteration" in o]
        self.assertTrue(len(iteration_logs) > 0, "Should have iterated at least once")
        
        # Check final result
        self.assertGreaterEqual(len(set(res["labels"])), 1)

if __name__ == "__main__":
    unittest.main()
