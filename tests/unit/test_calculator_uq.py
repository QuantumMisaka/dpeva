import pytest
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Any
from dpeva.uncertain.calculator import UQCalculator

# --- Test Utilities & Golden Values ---

def calculate_expected_qbc(f_ensemble: np.ndarray) -> np.ndarray:
    """
    Manually calculates QbC force UQ (golden reference).
    
    Formula: sqrt( mean( (f_i - f_mean)^2 ) ) over models i=1,2,3
    Note: Code uses mean over models for each component (x,y,z), then sums them, then sqrt.
    
    Code Logic:
    fx_qbc_sq_diff = mean((fx_i - fx_mean)^2)
    f_qbc_stddiff = sqrt(fx_qbc_sq_diff + fy_qbc_sq_diff + fz_qbc_sq_diff)
    
    Args:
        f_ensemble: Shape (N_models=3, N_atoms, 3)
        
    Returns:
        Shape (N_atoms,)
    """
    f_mean = np.mean(f_ensemble, axis=0) # (N_atoms, 3)
    sq_diff = (f_ensemble - f_mean)**2 # (3, N_atoms, 3)
    mean_sq_diff = np.mean(sq_diff, axis=0) # (N_atoms, 3) -> fx_var, fy_var, fz_var
    sum_var = np.sum(mean_sq_diff, axis=1) # (N_atoms,)
    return np.sqrt(sum_var)

def calculate_expected_rnd(f_main: np.ndarray, f_ensemble: np.ndarray) -> np.ndarray:
    """
    Manually calculates RND force UQ (golden reference).
    
    Formula: sqrt( mean( (f_i - f_main)^2 ) ) over models i=1,2,3
    
    Args:
        f_main: Shape (N_atoms, 3)
        f_ensemble: Shape (N_models=3, N_atoms, 3)
        
    Returns:
        Shape (N_atoms,)
    """
    sq_diff = (f_ensemble - f_main)**2 # (3, N_atoms, 3)
    mean_sq_diff = np.mean(sq_diff, axis=0) # (N_atoms, 3)
    sum_var = np.sum(mean_sq_diff, axis=1) # (N_atoms,)
    return np.sqrt(sum_var)

# --- Tests ---

class TestUQCalculator:
    """Tests for UQCalculator class."""

    def test_compute_qbc_rnd_golden_value(self, mock_predictions_factory):
        """
        Verifies compute_qbc_rnd against manually calculated golden values.
        Using fixed random seed for reproducibility.
        """
        np.random.seed(42)
        n_atoms = 10
        
        # Generate random forces
        f0 = np.random.randn(n_atoms, 3)
        f1 = np.random.randn(n_atoms, 3)
        f2 = np.random.randn(n_atoms, 3)
        f3 = np.random.randn(n_atoms, 3)
        
        p0 = mock_predictions_factory(f0, [n_atoms])
        p1 = mock_predictions_factory(f1, [n_atoms])
        p2 = mock_predictions_factory(f2, [n_atoms])
        p3 = mock_predictions_factory(f3, [n_atoms])
        
        calc = UQCalculator()
        results = calc.compute_qbc_rnd([p0, p1, p2, p3])
        
        # Golden Calculation
        f_ensemble = np.stack([f1, f2, f3])
        expected_qbc = np.max(calculate_expected_qbc(f_ensemble)) # Max per structure
        expected_rnd = np.max(calculate_expected_rnd(f0, f_ensemble)) # Max per structure
        
        # Assertions with tight tolerance
        np.testing.assert_allclose(results["uq_qbc_for"][0], expected_qbc, rtol=1e-5, atol=1e-6, err_msg="QbC mismatch")
        np.testing.assert_allclose(results["uq_rnd_for"][0], expected_rnd, rtol=1e-5, atol=1e-6, err_msg="RND mismatch")
        
    def test_compute_qbc_rnd_multiple_frames(self, mock_predictions_factory):
        """Test with multiple frames to ensure aggregation is correct."""
        atom_counts = [2, 3] # Frame 0: 2 atoms, Frame 1: 3 atoms
        n_total = 5
        
        f0 = np.zeros((n_total, 3))
        # Frame 0: High variance
        f1 = np.zeros((n_total, 3)); f1[0:2] = 1.0
        f2 = np.zeros((n_total, 3)); f2[0:2] = -1.0
        f3 = np.zeros((n_total, 3)); f3[0:2] = 0.0
        
        # Frame 1: Zero variance (all match f0)
        # (Already zero initialized)
        
        p0 = mock_predictions_factory(f0, atom_counts)
        p1 = mock_predictions_factory(f1, atom_counts)
        p2 = mock_predictions_factory(f2, atom_counts)
        p3 = mock_predictions_factory(f3, atom_counts)
        
        calc = UQCalculator()
        results = calc.compute_qbc_rnd([p0, p1, p2, p3])
        
        assert len(results["uq_qbc_for"]) == 2
        
        # Frame 0: High UQ
        assert results["uq_qbc_for"][0] > 0.1
        # Frame 1: Zero UQ
        assert results["uq_qbc_for"][1] == 0.0

    def test_compute_qbc_rnd_with_ground_truth_and_diff(self, mock_predictions_factory):
        """Test calculation when diff_fx is available in predictions."""
        n_atoms = 5
        f0 = np.zeros((n_atoms, 3))
        # Ground truth different from f0, so diff_fx will be non-zero
        gt = np.ones((n_atoms, 3)) 
        
        # This factory will set diff_fx = f0 - gt = -1
        p0 = mock_predictions_factory(f0, [n_atoms], has_gt=True, gt_forces=gt)
        p1 = mock_predictions_factory(f0, [n_atoms])
        p2 = mock_predictions_factory(f0, [n_atoms])
        p3 = mock_predictions_factory(f0, [n_atoms])
        
        calc = UQCalculator()
        results = calc.compute_qbc_rnd([p0, p1, p2, p3])
        
        # diff_maxf_0_frame: max(sqrt(fx^2+fy^2+fz^2))
        # diff = [-1, -1, -1] -> norm = sqrt(3)
        expected_diff = np.sqrt(3)
        assert np.isclose(results["diff_maxf_0_frame"][0], expected_diff)
        
        # diff_rmsf_0_frame: sqrt(mean(diff^2))
        # diff^2 = 3. mean = 3. sqrt = sqrt(3)
        assert np.isclose(results["diff_rmsf_0_frame"][0], expected_diff)

    def test_compute_qbc_rnd_no_ground_truth(self, mock_predictions_factory):
        """Test calculation when ground truth is missing."""
        n_atoms = 5
        f0 = np.zeros((n_atoms, 3))
        
        # Explicitly set has_gt=False, so diff_fx will be None
        p0 = mock_predictions_factory(f0, [n_atoms], has_gt=False)
        p1 = mock_predictions_factory(f0, [n_atoms])
        p2 = mock_predictions_factory(f0, [n_atoms])
        p3 = mock_predictions_factory(f0, [n_atoms])
        
        calc = UQCalculator()
        results = calc.compute_qbc_rnd([p0, p1, p2, p3])
        
        assert results["diff_maxf_0_frame"][0] == 0.0
        assert results["diff_rmsf_0_frame"][0] == 0.0

    @pytest.mark.parametrize("input_val, expected_lo", [
        # Peak at 0.5, drops to half height around 0.5 + HWHM
        # Gaussian sigma=0.1 -> HWHM approx 0.1177
        # Trust Lo should be approx 0.5 + 0.1177 = 0.6177
        (0.5, 0.617),
    ])
    def test_calculate_trust_lo_gaussian(self, input_val, expected_lo):
        """Test auto-threshold on ideal Gaussian distribution."""
        np.random.seed(42)
        # Generate Gaussian data
        data = np.random.normal(loc=input_val, scale=0.1, size=5000)
        data = np.clip(data, 0, 2.0)
        
        calc = UQCalculator()
        trust_lo = calc.calculate_trust_lo(data, ratio=0.5, bound=(0, 2.0), grid_size=2000)
        
        # Allow some tolerance due to KDE approximation and sampling noise
        assert np.isclose(trust_lo, expected_lo, atol=0.05)

    def test_calculate_trust_lo_edge_cases(self):
        """Test empty/sparse/singular data."""
        calc = UQCalculator()
        
        # Empty
        assert calc.calculate_trust_lo(np.array([])) is None
        
        # Too few points
        assert calc.calculate_trust_lo(np.array([0.5])) is None
        
        # Singular (all points same) - KDE should fail gracefully or produce delta
        # Scipy gaussian_kde raises LinAlgError on singular matrix if no noise
        # Our code catches Exception and returns None
        data_singular = np.ones(100) * 0.5
        assert calc.calculate_trust_lo(data_singular) is None

    def test_align_scales(self):
        """Test RobustScaler alignment."""
        calc = UQCalculator()
        np.random.seed(42)
        
        # qbc: median=10, iqr=2
        qbc = np.random.normal(10, 1.5, 1000) 
        
        # rnd: median=0, iqr=1
        rnd = np.random.normal(0, 0.75, 1000)
        
        rnd_rescaled = calc.align_scales(qbc, rnd)
        
        # Check median alignment
        assert np.isclose(np.median(rnd_rescaled), np.median(qbc), atol=0.2)
        
        # Check IQR alignment
        q3_q, q1_q = np.percentile(qbc, [75, 25])
        q3_r, q1_r = np.percentile(rnd_rescaled, [75, 25])
        assert np.isclose(q3_r - q1_r, q3_q - q1_q, atol=0.2)

    def test_compute_qbc_rnd_robustness_nan(self, mock_predictions_factory, caplog):
        """
        Test 'Clamp-and-Clean' strategy:
        1. Clamp: Handles negative squares (simulated via manual hacking if needed, but here we check result).
        2. Clean: Handles NaN inputs gracefully by converting to Infinity.
        """
        n_atoms = 5
        # Model 0 has NaN
        f0 = np.zeros((n_atoms, 3))
        f0[0, 0] = np.nan
        
        p0 = mock_predictions_factory(f0, [n_atoms])
        p1 = mock_predictions_factory(np.zeros((n_atoms, 3)), [n_atoms])
        p2 = mock_predictions_factory(np.zeros((n_atoms, 3)), [n_atoms])
        p3 = mock_predictions_factory(np.zeros((n_atoms, 3)), [n_atoms])
        
        calc = UQCalculator()
        
        # Should not crash
        with caplog.at_level(logging.WARNING):
            results = calc.compute_qbc_rnd([p0, p1, p2, p3])
            
        # Verify Warning
        assert "NaNs detected" in caplog.text
        
        # Verify Output is clean
        assert not np.isnan(results["uq_qbc_for"]).any()
        assert not np.isnan(results["uq_rnd_for"]).any()
        # The nan input should result in Infinity after cleaning
        assert results["uq_rnd_for"][0] == np.inf
        
        # Check that Infinity is correctly handled by RobustScaler (it should remain large/Inf)
        # We need enough data points for scaler to find valid median/IQR
        # Let's create a mix of valid data and one Inf
        valid_data = np.random.normal(0, 1, 100)
        mixed_data = np.concatenate([valid_data, [np.inf]])
        
        # Mock other dimension
        valid_other = np.random.normal(0, 1, 101)
        
        # Align scales
        rescaled = calc.align_scales(valid_other, mixed_data)
        
        # The Inf value should result in Inf (or very large) after scaling
        # RobustScaler subtracts median (finite) and divides by IQR (finite) -> Inf
        assert rescaled[-1] == np.inf

    def test_compute_qbc_rnd_robustness_negative_variance(self, mock_predictions_factory):
        """
        Test 'Clamp' strategy:
        Simulate a case where floating point error might cause sum of squares to be slightly negative.
        Since we can't easily force numpy to make errors, we trust the logic change (np.maximum(..., 0)).
        But we can verify that very small differences don't cause issues.
        """
        n_atoms = 1
        # Very close values
        val = 1.0
        delta = 1e-16 # Below machine epsilon for some ops?
        
        f0 = np.array([[val, 0, 0]])
        f1 = np.array([[val + delta, 0, 0]])
        f2 = np.array([[val - delta, 0, 0]])
        f3 = np.array([[val, 0, 0]])
        
        p0 = mock_predictions_factory(f0, [n_atoms])
        p1 = mock_predictions_factory(f1, [n_atoms])
        p2 = mock_predictions_factory(f2, [n_atoms])
        p3 = mock_predictions_factory(f3, [n_atoms])
        
        calc = UQCalculator()
        results = calc.compute_qbc_rnd([p0, p1, p2, p3])
        
        # Should be a very small number, not NaN
        assert not np.isnan(results["uq_qbc_for"][0])
        assert results["uq_qbc_for"][0] >= 0.0
