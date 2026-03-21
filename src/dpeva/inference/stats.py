import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from scipy.linalg import lstsq
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
                 ref_energies: Optional[Dict[str, float]] = None,
                 enable_cohesive_energy: bool = True,
                 allow_ref_energy_lstsq_completion: bool = False):
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
        self.enable_cohesive_energy = enable_cohesive_energy
        self.allow_ref_energy_lstsq_completion = allow_ref_energy_lstsq_completion
        self._cohesive_ref_energies_cache: Optional[Dict[str, float]] = None
        
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

    def _fit_e0_by_lstsq(self, A: np.ndarray, b: np.ndarray, unknown_elements: List[str]) -> Optional[Dict[str, float]]:
        if A.shape[1] == 0:
            return {}
        if A.shape[0] < A.shape[1]:
            self.logger.warning(
                f"Cohesive energy skipped: insufficient samples for lstsq "
                f"(samples={A.shape[0]}, unknown_elements={A.shape[1]})."
            )
            return None
        try:
            cond_number = np.linalg.cond(A)
            if not np.isfinite(cond_number) or cond_number > 1e12:
                self.logger.warning(
                    f"Cohesive energy skipped: ill-conditioned lstsq matrix (cond={cond_number:.3e})."
                )
                return None
            x, _, rank, _ = lstsq(A, b)
            if rank < A.shape[1]:
                self.logger.warning(
                    f"Cohesive energy skipped: rank-deficient lstsq matrix "
                    f"(rank={rank}, unknown_elements={A.shape[1]})."
                )
                return None
            return {elem: float(val) for elem, val in zip(unknown_elements, x)}
        except Exception as e:
            self.logger.warning(f"Cohesive energy skipped: lstsq fitting failed ({e}).")
            return None

    def _resolve_reference_energies(self, energy: np.ndarray) -> Optional[Dict[str, float]]:
        unique_elements = sorted(list(set(e for counts in self.atom_counts_list for e in counts.keys())))
        element_map = {e: i for i, e in enumerate(unique_elements)}
        n_frames = len(energy)
        atom_nums_arr = np.asarray(self.atom_num_list, dtype=float)
        if np.any(atom_nums_arr <= 0):
            self.logger.warning("Cohesive energy skipped: non-positive atom count found.")
            return None

        A_full = np.zeros((n_frames, len(unique_elements)))
        b_total = np.asarray(energy, dtype=float) * atom_nums_arr
        for idx, counts in enumerate(self.atom_counts_list):
            for elem, count in counts.items():
                A_full[idx, element_map[elem]] = count

        provided = self.ref_energies or {}
        known_elements = [e for e in unique_elements if e in provided]
        unknown_elements = [e for e in unique_elements if e not in provided]

        if not unknown_elements and known_elements:
            e0 = {e: float(provided[e]) for e in unique_elements}
            self.logger.info(f"Using provided Atomic Energies (E0): {e0}")
            return e0

        if unknown_elements and not self.allow_ref_energy_lstsq_completion:
            self.logger.warning(
                f"Cohesive energy skipped: missing ref energies for {unknown_elements} "
                "and allow_ref_energy_lstsq_completion is disabled."
            )
            return None

        A_unknown = np.zeros((n_frames, len(unknown_elements)))
        if known_elements:
            b_adjusted = b_total.copy()
            for elem in known_elements:
                b_adjusted -= A_full[:, element_map[elem]] * float(provided[elem])
        else:
            b_adjusted = b_total

        for idx, elem in enumerate(unknown_elements):
            A_unknown[:, idx] = A_full[:, element_map[elem]]

        fitted_unknown = self._fit_e0_by_lstsq(A_unknown, b_adjusted, unknown_elements)
        if fitted_unknown is None:
            return None

        e0 = {elem: float(provided[elem]) for elem in known_elements}
        e0.update(fitted_unknown)
        self.logger.info(f"Using mixed Atomic Energies (E0): {e0}")
        return e0

    def compute_relative_energy(self, energy: np.ndarray) -> Optional[np.ndarray]:
        """Compute cohesive energy per atom with mixed ref-energy and lstsq fallback policy."""
        if not self.enable_cohesive_energy:
            self.logger.info("Cohesive energy disabled by configuration.")
            return None
        if self.atom_counts_list is None or self.atom_num_list is None:
            self.logger.warning("Cohesive energy skipped: no atom composition info provided.")
            return None
        if len(energy) != len(self.atom_counts_list) or len(self.atom_counts_list) != len(self.atom_num_list):
            self.logger.warning(
                f"Cohesive energy skipped: frame mismatch (energy={len(energy)}, "
                f"composition={len(self.atom_counts_list)}, atom_nums={len(self.atom_num_list)})."
            )
            return None
        finite_mask = np.isfinite(np.asarray(energy, dtype=float))
        if not np.all(finite_mask):
            self.logger.warning(
                f"Cohesive energy input contains non-finite values "
                f"({int(np.size(finite_mask) - np.count_nonzero(finite_mask))}/{np.size(finite_mask)})."
            )
        try:
            if self._cohesive_ref_energies_cache is None:
                self._cohesive_ref_energies_cache = self._resolve_reference_energies(energy)
            e0_dict = self._cohesive_ref_energies_cache
            if e0_dict is None:
                return None
            unique_elements = sorted(list(set(e for counts in self.atom_counts_list for e in counts.keys())))
            ref_energies_total = []
            for counts in self.atom_counts_list:
                ref_total = sum(counts.get(elem, 0) * e0_dict.get(elem, 0.0) for elem in unique_elements)
                ref_energies_total.append(ref_total)
            ref_energies_per_atom = np.asarray(ref_energies_total, dtype=float) / np.asarray(self.atom_num_list, dtype=float)
            cohesive_energy_per_atom = np.asarray(energy, dtype=float) - ref_energies_per_atom
            cohesive_finite = np.isfinite(cohesive_energy_per_atom)
            if not np.all(cohesive_finite):
                self.logger.warning(
                    f"Cohesive energy contains non-finite values "
                    f"({int(np.size(cohesive_finite) - np.count_nonzero(cohesive_finite))}/{np.size(cohesive_finite)})."
                )
            return cohesive_energy_per_atom
        except Exception as e:
            self.logger.error(f"Relative energy calculation failed: {e}", exc_info=True)
            return None
