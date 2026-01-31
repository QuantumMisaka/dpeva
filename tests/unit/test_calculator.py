import pytest
import numpy as np
from dpeva.uncertain.calculator import UQCalculator

def test_compute_qbc_rnd(mock_predictions_factory):
    """Test QbC and RND calculation logic with simple controlled data."""
    calc = UQCalculator()
    
    # Setup: 1 Frame, 2 Atoms
    # Model 0 (Main): [1, 0, 0], [0, 1, 0]
    # Model 1: [1.1, 0, 0], [0, 1.1, 0]
    # Model 2: [0.9, 0, 0], [0, 0.9, 0]
    # Model 3: [1.0, 0, 0], [0, 1.0, 0]
    
    # Ground Truth: [1, 0, 0], [0, 1, 0] (Same as Model 0 -> Diff = 0)
    
    gt = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    
    p0 = mock_predictions_factory(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), [2], gt_forces=gt)
    p1 = mock_predictions_factory(np.array([[1.1, 0.0, 0.0], [0.0, 1.1, 0.0]]), [2])
    p2 = mock_predictions_factory(np.array([[0.9, 0.0, 0.0], [0.0, 0.9, 0.0]]), [2])
    p3 = mock_predictions_factory(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), [2])
    
    results = calc.compute_qbc_rnd(p0, p1, p2, p3)
    
    # Check Keys
    assert "uq_qbc_for" in results
    assert "uq_rnd_for" in results
    assert "diff_maxf_0_frame" in results
    
    # Check Shapes
    assert results["uq_qbc_for"].shape == (1,)
    assert results["uq_rnd_for"].shape == (1,)
    
    # Verify QbC Logic
    # Ensemble Mean: (1.1 + 0.9 + 1.0) / 3 = 1.0
    # Variance (sq_diff):
    #   M1: (1.1-1.0)^2 = 0.01
    #   M2: (0.9-1.0)^2 = 0.01
    #   M3: (1.0-1.0)^2 = 0.00
    #   Mean Sq Diff = 0.02 / 3 ≈ 0.00667
    #   Sqrt ≈ 0.0816
    
    expected_qbc = np.sqrt(0.02 / 3)
    assert np.isclose(results["uq_qbc_for"][0], expected_qbc, atol=1e-4)
    
    # Verify RND Logic
    # Deviation from M0 (1.0):
    #   M1: (1.1-1.0)^2 = 0.01
    #   M2: (0.9-1.0)^2 = 0.01
    #   M3: (1.0-1.0)^2 = 0.00
    #   Mean Sq Diff = 0.00667
    #   Sqrt ≈ 0.0816
    
    expected_rnd = np.sqrt(0.02 / 3)
    assert np.isclose(results["uq_rnd_for"][0], expected_rnd, atol=1e-4)

    # Verify Diff (Should be 0 as P0 == GT)
    assert np.isclose(results["diff_maxf_0_frame"][0], 0.0, atol=1e-6)

def test_calculate_trust_lo_simple():
    """Test auto-threshold calculation with a simple peak."""
    calc = UQCalculator()
    
    # Create a synthetic distribution: Peak at 0.5, decays to right
    # We simulate this by drawing samples from a normal distribution centered at 0.5
    np.random.seed(42)
    data = np.random.normal(loc=0.5, scale=0.1, size=1000)
    data = data[(data >= 0) & (data <= 2.0)] # Clip to bound
    
    # Calculate threshold
    # Ratio 0.5 means we look for density = 0.5 * peak_density
    trust_lo = calc.calculate_trust_lo(data, ratio=0.5, bound=(0, 2.0))
    
    assert trust_lo is not None
    assert trust_lo > 0.5 # Must be to the right of peak
    assert trust_lo < 1.0 # Should be reasonably close

def test_calculate_trust_lo_fallback():
    """Test fallback when data is empty."""
    calc = UQCalculator()
    data = np.array([])
    trust_lo = calc.calculate_trust_lo(data)
    assert trust_lo is None
