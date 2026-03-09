"""
Abacus Post-Processor
=====================

Handles the processing of ABACUS calculation results, including convergence checks,
metric calculation, and data cleaning.
"""

import os
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import dpdata
from ase import Atoms
from collections import Counter
from scipy.linalg import lstsq

logger = logging.getLogger(__name__)

EV_A3_TO_GPA = 160.21766208

class AbacusPostProcessor:
    """
    Process ABACUS results: Check convergence, Parse data, Clean data.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with configuration.
        
        Args:
            config:
                - cleaning_thresholds: Dict with keys 'energy', 'force', 'stress', 'max_atoms'.
                  Note: 'energy' threshold refers to Cohesive Energy per atom (eV/atom).
                - ref_energies: Dict mapping element to reference energy (E0).
        """
        self.config = config
        self.thresholds = config.get("cleaning_thresholds", {
            "cohesive_energy": float("nan"),  # Cohesive Energy per atom (eV/atom). NaN/None means skip.
            "force": 40.0,      # eV/Angstrom
            "stress": 40.0,     # GPa
            "max_atoms": 255
        })
        self.ref_energies = config.get("ref_energies", {})

    def check_convergence(self, task_dir: Path) -> bool:
        """
        Check if an ABACUS calculation has converged.
        Looks for "charge density convergence is achieved" in running_scf.log.
        """
        task_dir = Path(task_dir)
        # Find running_scf.log
        # Usually in OUT.suffix/running_scf.log
        # We search recursively or check standard location
        log_files = list(task_dir.glob("OUT.*/running_scf.log"))
        if not log_files:
            # Fallback to abacus.out in root if running_scf not found (older versions?)
            log_files = list(task_dir.glob("abacus.out"))
        
        for log_file in log_files:
            try:
                with open(log_file, 'r') as f:
                    content = f.read()
                    if "charge density convergence is achieved" in content:
                        return True
                    if "convergence has not been achieved" in content:
                        return False
            except Exception as e:
                logger.warning(f"Error reading log file {log_file}: {e}")
                continue
        
        return False

    def load_data(self, task_dir: Path) -> Optional[dpdata.LabeledSystem]:
        """
        Load data from a converged task directory using dpdata.
        """
        try:
            # dpdata expects the directory containing INPUT/STRU/OUT.*
            # fmt='abacus/scf'
            ls = dpdata.LabeledSystem(str(task_dir), fmt='abacus/scf')
            if len(ls) > 0:
                return ls
        except Exception as e:
            logger.error(f"Failed to load data from {task_dir}: {e}")
        return None

    def compute_metrics(self, systems: dpdata.MultiSystems) -> pd.DataFrame:
        """
        Compute metrics (Cohesive Energy, Pressure, Max Force) for a MultiSystems object.
        Returns a DataFrame.
        """
        data_list = []
        for si, s in enumerate(systems):
            atom_names = s["atom_names"]
            atom_types = s["atom_types"]
            elements = [atom_names[t] for t in atom_types]
            n_frames = s.get_nframes()
            
            for fi in range(n_frames):
                energy = float(s["energies"][fi])
                forces = s["forces"][fi]
                virial = s["virials"][fi]
                cells = s["cells"][fi]
                volume = float(np.abs(np.linalg.det(cells)))
                
                # Pressure (GPa)
                # stress = virial / volume (if volume > 0)
                # pressure = trace(stress) / 3
                # 1 eV/A^3 = 160.21766208 GPa
                if volume > 1e-6:
                    stress_tensor = virial / volume
                    pressure_gpa = float(np.trace(stress_tensor) / 3.0) * EV_A3_TO_GPA
                else:
                    stress_tensor = np.zeros((3,3))
                    pressure_gpa = 0.0
                
                max_force = float(np.linalg.norm(forces, axis=1).max())
                num_atoms = len(elements)
                energy_per_atom = energy / num_atoms
                
                data_list.append({
                    "sys_idx": si,
                    "frame_idx": fi,
                    "elements": elements,
                    "num_atoms": num_atoms,
                    "energy": energy,
                    "energy_per_atom": energy_per_atom,
                    "max_force": max_force,
                    "pressure_gpa": pressure_gpa,
                    "volume": volume
                })
        
        df = pd.DataFrame(data_list)
        
        # Compute Cohesive Energy
        # 1. Determine E0 (Reference Energy)
        unique_elements = sorted(list(set([e for row in df["elements"] for e in row])))
        e0_map = self.ref_energies.copy()
        
        # Identify missing E0
        missing_elements = [e for e in unique_elements if e not in e0_map]
        
        if missing_elements:
            logger.info(f"Missing reference energies for {missing_elements}, fitting via Least Squares...")
            # Solve A * x = b
            # A: counts of each element in each frame
            # x: E0 for each element
            # b: Total energy of each frame
            A = np.zeros((len(df), len(unique_elements)))
            b = df["energy"].values
            elem_to_idx = {e: i for i, e in enumerate(unique_elements)}
            
            for i, row in df.iterrows():
                counts = Counter(row["elements"])
                for e, count in counts.items():
                    A[i, elem_to_idx[e]] = count
            
            x, residuals, rank, s = lstsq(A, b)
            fitted_e0 = {e: x[i] for i, e in enumerate(unique_elements)}
            logger.info(f"Fitted E0: {fitted_e0}")
            
            # Update map (prioritize existing config, fill missing with fitted)
            for e in missing_elements:
                e0_map[e] = fitted_e0[e]
        else:
            fitted_e0 = {}

        # 2. Calculate Cohesive Energy
        if missing_elements and not fitted_e0:
             # If fitting failed or wasn't performed, we cannot filter by cohesive energy
             logger.warning("Could not determine reference energies. Cohesive energy filtering will be skipped.")
             df["cohesive_energy_per_atom"] = np.nan
        else:
            def calc_coh(row):
                ref = sum(e0_map.get(e, 0.0) for e in row["elements"])
                return (row["energy"] - ref) / row["num_atoms"]
            
            df["cohesive_energy_per_atom"] = df.apply(calc_coh, axis=1)
        
        return df

    def filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter DataFrame based on thresholds.
        """
        mask = np.ones(len(df), dtype=bool)
        
        # Cohesive Energy Filter
        # Use 'cohesive_energy' key (preferred) or legacy 'energy' key
        ce_thr = self.thresholds.get("cohesive_energy", self.thresholds.get("energy"))
        
        if ce_thr is not None and not np.isnan(ce_thr):
            if "cohesive_energy_per_atom" in df.columns and not df["cohesive_energy_per_atom"].isna().all():
                mask &= (df["cohesive_energy_per_atom"] <= ce_thr)
            else:
                logger.warning("Cohesive energy not available. Skipping energy filtering.")
        
        if "force" in self.thresholds:
            mask &= (df["max_force"] <= self.thresholds["force"])
            
        if "stress" in self.thresholds:
            mask &= (np.abs(df["pressure_gpa"]) <= self.thresholds["stress"])
            
        if "max_atoms" in self.thresholds:
            mask &= (df["num_atoms"] <= self.thresholds["max_atoms"])
            
        cleaned_df = df[mask].copy()
        logger.info(f"Filtered data: {len(df)} -> {len(cleaned_df)} frames")
        return cleaned_df

    def export_data(self, systems: dpdata.MultiSystems, df_clean: pd.DataFrame, output_dir: Path, format: str = "deepmd/npy"):
        """
        Export cleaned data to disk.
        """
        if df_clean.empty:
            logger.warning("No data to export.")
            return

        # Reconstruct MultiSystems from cleaned indices
        # This is tricky because dpdata structures are nested (System -> Frame)
        # But MultiSystems is a list of Systems.
        # If we filter frames within a System, we need to create new Systems.
        # dpdata doesn't easily support slicing frames to create new Systems without re-parsing or deep copying.
        
        # Efficient approach: Group by sys_idx
        grouped = df_clean.groupby("sys_idx")
        new_ms = dpdata.MultiSystems()
        
        for sys_idx, group in grouped:
            original_sys = systems[sys_idx]
            frame_indices = group["frame_idx"].values.astype(int)
            # dpdata System supports slicing: sys.sub_system(indices)
            try:
                sub_sys = original_sys.sub_system(frame_indices)
                new_ms.append(sub_sys)
            except Exception as e:
                logger.error(f"Failed to slice system {sys_idx}: {e}")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            new_ms.to(format, str(output_dir))
            logger.info(f"Exported {len(new_ms)} systems to {output_dir}")
        except Exception as e:
            logger.error(f"Export failed: {e}")

