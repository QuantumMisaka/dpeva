
import pytest
import numpy as np
import os
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


def test_load_predictions_requires_all_models_have_ground_truth(uq_manager):
    with patch("dpeva.uncertain.manager.DPTestResultParser") as mock_parser:
        mock_parser.return_value.parse.side_effect = [
            {"energy": [], "force": [], "virial": [], "has_ground_truth": True, "dataname_list": [], "datanames_nframe": {}},
            {"energy": [], "force": [], "virial": [], "has_ground_truth": True, "dataname_list": [], "datanames_nframe": {}},
            {"energy": [], "force": [], "virial": [], "has_ground_truth": False, "dataname_list": [], "datanames_nframe": {}},
            {"energy": [], "force": [], "virial": [], "has_ground_truth": True, "dataname_list": [], "datanames_nframe": {}},
        ]
        _, has_gt = uq_manager.load_predictions()
        assert has_gt is False


def test_load_predictions_logs_warning_when_any_model_lacks_ground_truth(uq_manager):
    with patch("dpeva.uncertain.manager.DPTestResultParser") as mock_parser:
        mock_parser.return_value.parse.side_effect = [
            {"energy": [], "force": [], "virial": [], "has_ground_truth": True, "dataname_list": [], "datanames_nframe": {}},
            {"energy": [], "force": [], "virial": [], "has_ground_truth": False, "dataname_list": [], "datanames_nframe": {}},
            {"energy": [], "force": [], "virial": [], "has_ground_truth": True, "dataname_list": [], "datanames_nframe": {}},
            {"energy": [], "force": [], "virial": [], "has_ground_truth": True, "dataname_list": [], "datanames_nframe": {}},
        ]
        with patch.object(uq_manager.logger, "warning") as mock_warning:
            _, has_gt = uq_manager.load_predictions()
        assert has_gt is False
        mock_warning.assert_called_once_with(
            "Detected missing/invalid ground truth in target pool (including near-zero energy labels <1e-4). Treating the pool as unlabeled and enabling no-label analysis/plot branches."
        )

def test_verify_atom_counts(tmp_path):
    """Test optional atom count verification logic."""
    testdata_dir = tmp_path / "testdata"
    testdata_dir.mkdir()
    
    # Setup UQManager with testdata_dir
    config = {}
    manager = UQManager(str(tmp_path), "res", "head", config, 1, testdata_dir=str(testdata_dir))
    
    # Mock dataname_list: [name, frame_idx, natom]
    # sys1: Correct (3 parsed, 3 real)
    # sys2: Incorrect (5 parsed, 10 real)
    # sys3: Missing in testdata (should skip/warn debug)
    dataname_list = [
        ["sys1", 0, 3],
        ["sys2", 0, 5],
        ["sys3", 0, 8]
    ]
    
    # Mock load_systems
    # sys1 -> 3 atoms
    # sys2 -> 10 atoms
    mock_sys1 = {"atom_types": [0, 0, 1]}
    mock_sys2 = {"atom_types": [0]*10}
    
    def side_effect_load(path):
        name = os.path.basename(path)
        if name == "sys1":
            return [mock_sys1]
        elif name == "sys2":
            return [mock_sys2]
        else:
            # Emulate load_systems behavior for missing file? 
            # Usually it raises or returns empty if not found?
            # Our code checks os.path.exists first.
            return []

    # Create dummy directories for sys1 and sys2 so os.path.exists passes
    (testdata_dir / "sys1").mkdir()
    (testdata_dir / "sys2").mkdir()
    
    # Patch load_systems where it is defined, not where it is used (since it's a local import)
    # Actually, since it is imported inside the function using `from dpeva.io.dataset import load_systems`,
    # patching `dpeva.io.dataset.load_systems` is the way to go.
    with patch("dpeva.io.dataset.load_systems", side_effect=side_effect_load):
        with patch.object(manager.logger, 'error') as mock_error:
            with patch.object(manager.logger, 'warning') as mock_warn:
                manager._verify_atom_counts_list(dataname_list)
                
                # Assertions
                # Sys2 mismatch should trigger error
                mock_error.assert_called_with("ATOM COUNT MISMATCH for sys2: Parsed=5, Actual=10!")
                
                # Sys3 missing should NOT trigger error (just debug skip, or loop continue)
                # Ensure no other errors
                assert mock_error.call_count == 1
                
                # Final summary warning about mismatch
                mock_warn.assert_any_call("Verification completed: 1 matched, 1 MISMATCHES.")
