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
    Creates a temp directory with dummy dp test results.
    Generates random data instead of copying from external source.
    """
    dest_dir = tmp_path / "dptest_results"
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    n_frames = 10
    n_atoms = 5
    n_lines_peratom = n_frames * n_atoms
    
    # 1. Energy Per Atom (Used for loading data_e)
    # 2 cols: data_e, pred_e
    with open(dest_dir / "results.e_peratom.out", "w") as f:
        # Add system info comment
        f.write(f"# /mock/pool/sys: 0\n")
        for _ in range(n_lines_peratom):
            f.write(f"{np.random.rand():.4f} {np.random.rand():.4f}\n")

    # 2. Force (Used for loading data_f)
    # 6 cols: data_fx, data_fy, data_fz, pred_fx, pred_fy, pred_fz
    with open(dest_dir / "results.f.out", "w") as f:
        f.write("# Mock Force\n")
        for _ in range(n_lines_peratom): 
            f.write(" ".join([f"{np.random.rand():.4f}" for _ in range(6)]) + "\n")

    # 3. Virial Per Atom (Used for loading data_v if present)
    # 18 cols: 9 data, 9 pred
    with open(dest_dir / "results.v_peratom.out", "w") as f:
         f.write("# Mock V Per Atom\n")
         for _ in range(n_lines_peratom):
             f.write(" ".join([f"{np.random.rand():.4f}" for _ in range(18)]) + "\n")

    # Create other files just in case, though parser seems to prioritize above
    with open(dest_dir / "results.e.out", "w") as f:
        f.write("# Mock E Frame\n")
        for _ in range(n_frames):
            f.write(f"{np.random.rand():.4f} {np.random.rand():.4f}\n")

    with open(dest_dir / "results.v.out", "w") as f:
        f.write("# Mock V Frame\n")
        for _ in range(n_frames):
            f.write(" ".join([f"{np.random.rand():.4f}" for _ in range(18)]) + "\n")
                 
    return dest_dir

@pytest.fixture
def mock_job_manager():
    """Mocks JobManager to prevent actual submission."""
    with patch("dpeva.inference.managers.JobManager") as mock_cls:
        manager_instance = mock_cls.return_value
        yield manager_instance

