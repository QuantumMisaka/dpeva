import pytest
import pandas as pd
import numpy as np
from dpeva.uncertain.filter import UQFilter

@pytest.fixture
def uq_dataframe():
    """Create a DataFrame with points covering various regions."""
    # lo=0.1, hi=0.2
    data = {
        "dataname": ["p1", "p2", "p3", "p4", "p5"],
        "uq_qbc_for":        [0.05, 0.15, 0.25, 0.15, 0.19],
        "uq_rnd_for_rescaled": [0.05, 0.15, 0.05, 0.25, 0.19]
    }
    return pd.DataFrame(data)

def test_filter_strict(uq_dataframe):
    """Test strict filtering scheme."""
    # Strict: Both x and y must be in [lo, hi]
    # lo=0.1, hi=0.2
    
    # p1(0.05, 0.05): accurate (<lo)
    # p2(0.15, 0.15): candidate (in range)
    # p3(0.25, 0.05): failed (>hi)
    # p4(0.15, 0.25): failed (>hi)
    # p5(0.19, 0.19): candidate (in range)
    
    uq_filter = UQFilter(scheme="strict", trust_lo=0.1, trust_hi=0.2)
    cand, acc, fail = uq_filter.filter(uq_dataframe)
    
    assert "p2" in cand["dataname"].values
    assert "p5" in cand["dataname"].values
    assert "p1" in acc["dataname"].values
    assert "p3" in fail["dataname"].values
    assert "p4" in fail["dataname"].values
    
    assert len(cand) == 2
    assert len(acc) == 1
    assert len(fail) == 2

def test_filter_tangent_lo(uq_dataframe):
    """Test tangent_lo filtering scheme."""
    # Tangent Lo: 
    # Box: [0, 0.2] x [0, 0.2]
    # Tangent Cut: Eliminates corner near (0.2, 0.2)? Or near (0.1, 0.1)?
    # Logic: (x-lo)*(lo-hi) + (y-lo)*(lo-hi) <= 0
    # Let lo=0.1, hi=0.2. lo-hi = -0.1
    # (x-0.1)*(-0.1) + (y-0.1)*(-0.1) <= 0
    # -0.1(x-0.1 + y-0.1) <= 0
    # x + y - 0.2 >= 0  => x + y >= 0.2
    
    # Wait, the logic in source code is:
    # (uq_x - uq_x_lo)*(uq_x_lo - uq_x_hi) + (uq_y - uq_y_lo)*(uq_y_lo - uq_y_hi) <= 0
    # Vector A = (x-lo, y-lo)
    # Vector B = (lo-hi, lo-hi) = (-0.1, -0.1)
    # A . B <= 0  => A and B angle >= 90 deg
    # Basically means (x-lo, y-lo) points generally in direction of (hi-lo, hi-lo)? No.
    # If dot product is negative, vectors are opposing.
    # B points "down-left". So A must point "up-right" for dot product < 0? No.
    # If A points "up-right" (x>lo, y>lo), then x-lo > 0, y-lo > 0.
    # Dot product: (pos)(-neg) + (pos)(-neg) = neg. Correct.
    # So if x > lo and y > lo, condition holds.
    
    # Let's re-evaluate logic:
    # mask_candidate = in_box & tangent_cond
    # in_box: x <= hi, y <= hi
    # tangent_cond: x + y >= 2*lo (derived from dot prod <= 0)
    
    # So candidate region: inside box [0, hi]x[0, hi], AND x+y >= 2*lo
    # Accurate region: inside box, but x+y < 2*lo
    
    # p1(0.05, 0.05): sum=0.1. 2*lo=0.2. sum < 0.2. -> Accurate
    # p2(0.15, 0.15): sum=0.3. > 0.2. In box. -> Candidate
    # p3(0.25, 0.05): x > hi -> Failed
    # p4(0.15, 0.25): y > hi -> Failed
    # p5(0.19, 0.19): sum=0.38. > 0.2. In box. -> Candidate
    
    uq_filter = UQFilter(scheme="tangent_lo", trust_lo=0.1, trust_hi=0.2)
    cand, acc, fail = uq_filter.filter(uq_dataframe)
    
    assert "p2" in cand["dataname"].values
    assert "p5" in cand["dataname"].values
    assert "p1" in acc["dataname"].values
    assert "p3" in fail["dataname"].values
    
    assert len(cand) == 2
    assert len(acc) == 1
    assert len(fail) == 2

def test_filter_input_validation():
    """Test validation of inputs."""
    with pytest.raises(ValueError, match="not supported"):
        UQFilter(scheme="invalid_scheme")
        
    with pytest.raises(ValueError, match="Low trust threshold should be lower"):
        UQFilter(scheme="strict", trust_lo=0.2, trust_hi=0.1)
