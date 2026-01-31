import pytest
import numpy as np
import os
import json
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from unittest.mock import MagicMock, patch

# Project Root Calculation
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TEST_DATA_ROOT = PROJECT_ROOT / "test"

@dataclass
class MockDPTestResults:
    """Mock class for DPTestResults to be used in tests."""
    data_f: Dict[str, np.ndarray]
    dataname_list: List[List[Any]]
    has_ground_truth: bool = True
    diff_fx: Optional[np.ndarray] = None
    diff_fy: Optional[np.ndarray] = None
    diff_fz: Optional[np.ndarray] = None

    @property
    def force(self):
        """Alias for data_f to match PredictionData interface."""
        return self.data_f

    @property
    def energy(self):
        """Alias for data_e (not used in current mock but good for completeness)."""
        return None

@pytest.fixture
def mock_predictions_factory():
    def _create(
        pred_forces: np.ndarray,
        atom_counts: List[int],
        has_gt: bool = True,
        gt_forces: Optional[np.ndarray] = None
    ):
        """
        Creates a MockDPTestResults object.
        
        Args:
            pred_forces: (N_atoms, 3) array of predicted forces.
            atom_counts: List of atom counts per frame (e.g. [2, 3] means frame 0 has 2 atoms, frame 1 has 3).
            has_gt: Whether ground truth exists.
            gt_forces: (N_atoms, 3) array of ground truth forces.
        """
        n_atoms_total = sum(atom_counts)
        assert pred_forces.shape == (n_atoms_total, 3)
        
        data_f = {
            'pred_fx': pred_forces[:, 0],
            'pred_fy': pred_forces[:, 1],
            'pred_fz': pred_forces[:, 2]
        }
        
        diff_fx, diff_fy, diff_fz = None, None, None
        
        if has_gt and gt_forces is not None:
            assert gt_forces.shape == (n_atoms_total, 3)
            diff_fx = pred_forces[:, 0] - gt_forces[:, 0]
            diff_fy = pred_forces[:, 1] - gt_forces[:, 1]
            diff_fz = pred_forces[:, 2] - gt_forces[:, 2]
            
            # Populate data_f with ground truth for new UQCalculator logic
            data_f['data_fx'] = gt_forces[:, 0]
            data_f['data_fy'] = gt_forces[:, 1]
            data_f['data_fz'] = gt_forces[:, 2]
        
        elif has_gt and gt_forces is None:
             # Fallback: If has_gt claims True but no GT provided, fill with zeros to satisfy UQCalculator key checks
             # This handles tests that don't care about GT values but use default has_gt=True
             data_f['data_fx'] = np.zeros_like(pred_forces[:, 0])
             data_f['data_fy'] = np.zeros_like(pred_forces[:, 1])
             data_f['data_fz'] = np.zeros_like(pred_forces[:, 2])
            
        # Construct dataname_list: [name, frame_idx, natom]
        dataname_list = []
        for i, natom in enumerate(atom_counts):
            dataname_list.append([f"sys-{i}", i, natom])
            
        return MockDPTestResults(
            data_f=data_f,
            dataname_list=dataname_list,
            has_ground_truth=has_gt,
            diff_fx=diff_fx,
            diff_fy=diff_fy,
            diff_fz=diff_fz
        )
    return _create

@pytest.fixture
def mock_dptest_output_dir(tmp_path):
    """
    Copies a subset of real dp test results to a temp directory.
    Source: test/dptest-results-labeled/results.*.out
    """
    source_dir = TEST_DATA_ROOT / "dptest-results-labeled"
    dest_dir = tmp_path / "dptest_results"
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy first N lines of output files to save time/space
    for ext in ["e", "f", "v", "e_peratom", "v_peratom"]:
        fname = f"results.{ext}.out"
        src_file = source_dir / fname
        if src_file.exists():
            with open(src_file, "r") as f_in, open(dest_dir / fname, "w") as f_out:
                # Header + 200 lines to ensure enough data
                try:
                    head = [next(f_in) for _ in range(201)]
                    f_out.writelines(head)
                except StopIteration:
                    pass # End of file
                
    return dest_dir

@pytest.fixture
def real_config_loader(tmp_path):
    """
    Loads a config file from test/ directory and resolves paths to point to 
    either tmp_path or keeps them absolute if they exist.
    """
    def _load(rel_path, mock_data_mapping=None):
        """
        rel_path: path relative to dpeva/test/
        mock_data_mapping: dict mapping 'config_key' -> 'tmp_path_subdir'
        """
        config_path = TEST_DATA_ROOT / rel_path
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found at {config_path}")
            
        with open(config_path, "r") as f:
            config = json.load(f)
            
        # Basic path resolution
        if mock_data_mapping:
            for key, sub_dir in mock_data_mapping.items():
                if key in config:
                    target_dir = tmp_path / sub_dir
                    target_dir.mkdir(parents=True, exist_ok=True)
                    config[key] = str(target_dir)
        
        # Also inject project root if needed
        config["project_root"] = str(tmp_path)
        
        return config
    return _load

@pytest.fixture
def mock_job_manager():
    """Mocks JobManager to prevent actual submission."""
    with patch("dpeva.workflows.infer.JobManager") as mock_cls:
        manager_instance = mock_cls.return_value
        yield manager_instance

@pytest.fixture
def mock_job_manager_train():
    """Mocks JobManager for training workflow."""
    with patch("dpeva.training.trainer.JobManager") as mock_cls:
        manager_instance = mock_cls.return_value
        yield manager_instance
