import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from scipy.linalg import lstsq
from collections import Counter
import logging

class StatsCalculator:
    """
    Calculates statistical metrics for inference results.
    """
    
    def __init__(self, energy_per_atom: np.ndarray, 
                 force_flat: np.ndarray, 
                 virial_per_atom: Optional[np.ndarray] = None,
                 energy_true: Optional[np.ndarray] = None,
                 force_true: Optional[np.ndarray] = None,
                 virial_true: Optional[np.ndarray] = None,
                 atom_counts_list: Optional[List[Dict[str, int]]] = None,
                 atom_num_list: Optional[List[int]] = None,
                 ref_energies: Optional[Dict[str, float]] = None):
        """
        Args:
            energy_per_atom: Predicted energy per atom.
            force_flat: Predicted forces (flattened).
            virial_per_atom: Predicted virial per atom (optional).
            energy_true: Ground truth energy per atom (optional).
            force_true: Ground truth forces (flattened) (optional).
            virial_true: Ground truth virial per atom (optional).
            atom_counts_list: List of atom counts dict for each frame (e.g. [{"H": 2, "O": 1}, ...]).
                              Required for cohesive energy calculation via Least Squares.
            atom_num_list: List of total atom numbers for each frame.
            ref_energies: Dictionary of atomic reference energies (e.g. {"H": -13.6}).
                          If provided, overrides Least Squares fitting.
        """
        self.e_pred = energy_per_atom
        self.f_pred = force_flat
        self.v_pred = virial_per_atom
        
        self.e_true = energy_true
        self.f_true = force_true
        self.v_true = virial_true
        
        self.has_truth = (self.e_true is not None)
        self.atom_counts_list = atom_counts_list
        self.atom_num_list = atom_num_list
        self.ref_energies = ref_energies
        
        self.logger = logging.getLogger(__name__)

        # Pre-calculate diffs if truth is available
        self.e_diff = None
        self.f_diff = None
        self.v_diff = None
        
        if self.has_truth:
            self.e_diff = self.e_pred - self.e_true
            if self.f_pred is not None and self.f_true is not None:
                self.f_diff = self.f_pred - self.f_true
            if self.v_pred is not None and self.v_true is not None:
                self.v_diff = self.v_pred - self.v_true
        
    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute RMSE and MAE for Energy, Force, and Virial.

        Returns:
            Dict[str, float]: Dictionary containing MAE and RMSE metrics (e.g. "e_mae", "e_rmse").
        """
        if not self.has_truth:
            return {}
            
        metrics = {}
        
        # Energy
        metrics["e_mae"] = np.mean(np.abs(self.e_diff))
        metrics["e_rmse"] = np.sqrt(np.mean(self.e_diff**2))
        
        # Force
        if self.f_diff is not None:
            metrics["f_mae"] = np.mean(np.abs(self.f_diff))
            metrics["f_rmse"] = np.sqrt(np.mean(self.f_diff**2))
        
        # Virial
        if self.v_diff is not None:
            metrics["v_mae"] = np.mean(np.abs(self.v_diff))
            metrics["v_rmse"] = np.sqrt(np.mean(self.v_diff**2))
            
        return metrics

    def get_distribution_stats(self, data: Optional[np.ndarray], label: str) -> Dict:
        """
        Get detailed distribution statistics.
        """
        if data is None:
            return {}
            
        df = pd.Series(data, name=label)
        desc = df.describe().to_dict()
        
        return {
            "all": desc
        }

    def compute_force_magnitude(self, force_flat: np.ndarray) -> np.ndarray:
        """
        Compute force magnitude from flattened force array.
        """
        if force_flat is None:
            return None
        f_vec = force_flat.reshape(-1, 3)
        return np.linalg.norm(f_vec, axis=1)

    def compute_relative_energy(self, energy: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute relative energy (Cohesive Energy).
        If atom composition is available, use Least Squares to fit atomic energies E0,
        then subtract sum(N_i * E0_i).

        Args:
            energy (np.ndarray): Energy per atom.

        Returns:
            Optional[np.ndarray]: Cohesive energy per atom, or None if calculation fails.
        """
        if self.atom_counts_list is None or self.atom_num_list is None:
            self.logger.warning("No atom counts info provided. Skipping relative energy calculation.")
            return None
            
        try:
            # 1. Collect unique elements
            unique_elements = sorted(list(set(
                e for counts in self.atom_counts_list for e in counts.keys()
            )))
            element_map = {e: i for i, e in enumerate(unique_elements)}
            
            n_frames = len(energy)
            if n_frames != len(self.atom_counts_list):
                self.logger.warning(f"Mismatch between energy length ({n_frames}) and atom counts length ({len(self.atom_counts_list)}). "
                                    f"Likely using training set for atom counts vs test set for inference. "
                                    f"Skipping relative energy calculation.")
                return None
            
            # 2. Build Matrix A and Vector b
            # A: (N_frames, N_elements), b: Total Energy (N_frames,)
            A = np.zeros((n_frames, len(unique_elements)))
            b = energy * np.array(self.atom_num_list) # Total energy for fitting
            
            for idx, counts in enumerate(self.atom_counts_list):
                for elem, count in counts.items():
                    A[idx, element_map[elem]] = count
                    
            # 3. Determine E0
            E0_dict = {}
            if self.ref_energies is not None:
                # Check coverage
                missing = [e for e in unique_elements if e not in self.ref_energies]
                if not missing:
                    E0_dict = self.ref_energies
                    self.logger.info(f"Using provided Atomic Energies (E0): {E0_dict}")
                else:
                    self.logger.warning(f"Ref energies missing for {missing}. Falling back to LS fitting.")
            
            if not E0_dict:
                 # Solve Ax = b for E0
                 # lstsq returns: x, residuals, rank, s
                 x, _, _, _ = lstsq(A, b)
                 E0_dict = {elem: float(val) for elem, val in zip(unique_elements, x)}
                 self.logger.info(f"Fitted Atomic Energies (E0): {E0_dict}")
            
            # 4. Compute Reference Energy for each frame (Total Energy)
            ref_energies_total = []
            for counts in self.atom_counts_list:
                ref_e = sum(counts.get(e, 0) * E0_dict.get(e, 0.0) for e in unique_elements)
                ref_energies_total.append(ref_e)
            
            ref_energies_total = np.array(ref_energies_total)
            
            # 5. Compute Cohesive Energy (per atom)
            # E_coh_per_atom = (E_total_pred - E_ref_total) / N_atoms
            # Note: energy input is per_atom, so E_total_pred = energy * N_atoms
            
            # Correct formula: 
            # E_coh_per_atom = (energy * N_atoms - ref_energies_total) / N_atoms
            #                = energy - (ref_energies_total / N_atoms)
            
            ref_energies_per_atom = ref_energies_total / np.array(self.atom_num_list)
            cohesive_energy_per_atom = energy - ref_energies_per_atom
            
            return cohesive_energy_per_atom
            
        except Exception as e:
            self.logger.error(f"Relative energy calculation failed: {e}", exc_info=True)
            return None
