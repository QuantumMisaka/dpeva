import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from dpeva.uncertain.manager import UQManager
from dpeva.io.types import PredictionData

@pytest.fixture
def uq_manager(tmp_path):
    config = {
        "trust_mode": "auto",
        "auto_bounds": {"qbc": {"lo_min": 0.1}},
        "qbc_params": {"ratio": 0.5},
        "rnd_params": {}
    }
    return UQManager(str(tmp_path), "test_res", "head", config, 4)

def test_auto_threshold(uq_manager):
    # Mock calculator methods
    uq_manager.calculator.calculate_trust_lo = MagicMock(side_effect=[0.05, 0.2]) # QbC=0.05, RND=0.2
    
    uq_results = {"uq_qbc_for": np.array([0.1])}
    uq_rnd = np.array([0.2])
    
    uq_manager.run_auto_threshold(uq_results, uq_rnd)
    
    # Check clamping: 0.05 < 0.1 (min) -> clamped to 0.1
    assert uq_manager.qbc_params["lo"] == 0.1
    # Check normal: 0.2 (no bounds) -> 0.2
    assert uq_manager.rnd_params["lo"] == 0.2

def test_load_predictions(uq_manager):
    with patch("dpeva.uncertain.manager.DPTestResultParser") as mock_parser:
        mock_parser.return_value.parse.return_value = {
            "energy": [], "force": [], "virial": [], "has_ground_truth": True,
            "dataname_list": [], "datanames_nframe": {}
        }
        
        preds, has_gt = uq_manager.load_predictions()
        assert len(preds) == 4
        assert has_gt is True
