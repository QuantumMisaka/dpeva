"""
Abacus Input Generator
======================

Handles the generation of ABACUS input files (INPUT, STRU, KPT) from atomic structures.
"""

import os
import logging
from typing import Dict, List, Optional, Any, Union
from copy import deepcopy
from pathlib import Path

import numpy as np
from ase import Atoms
try:
    from ase.io.abacus import write_input, write_abacus
except ImportError:
    raise ImportError(
        "Could not import ase.io.abacus. This module requires the 'ase-abacus' plugin.\n"
        "Please install it via:\n"
        "  pip install git+https://gitlab.com/1041176461/ase-abacus.git"
    )

from dpeva.labeling.structure import StructureAnalyzer

# Configure logging
logger = logging.getLogger(__name__)

class AbacusGenerator:
    """
    Generates ABACUS input files for a set of atomic structures.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the generator with configuration.

        Args:
            config (Dict): Configuration dictionary containing:
                - dft_params: Dict of ABACUS INPUT parameters.
                - pp_map: Dict mapping element symbol to PP filename.
                - orb_map: Dict mapping element symbol to Orb filename.
                - pp_dir: Directory containing PP files.
                - orb_dir: Directory containing Orb files.
                - kpt_criteria: Criteria for K-point generation (default: 25).
                - vacuum_thickness: Minimum vacuum thickness (default: 6.3).
        """
        self.config = config
        self.dft_params = config.get("dft_params", {})
        self.pp_map = config.get("pp_map", {})
        self.orb_map = config.get("orb_map", {})
        self.mag_map = config.get("mag_map", {})
        self.pp_dir = config.get("pp_dir", "")
        self.orb_dir = config.get("orb_dir", "")
        self.kpt_criteria = config.get("kpt_criteria", 25)
        
        # Instantiate analyzer as a helper (though Manager should use it primarily)
        # We keep it here to support the fallback in generate()
        self.analyzer = StructureAnalyzer(
            vacuum_thickness=config.get("vacuum_thickness", 6.3)
        )

    def _set_magmom(self, atoms: Atoms):
        """
        Set initial magnetic moments based on config (optional).
        Currently supports Fe-C-H-O logic if provided in config.
        """
        if not self.mag_map:
            return

        init_magmom = atoms.get_initial_magnetic_moments()
        # If no initial magmom set (all zeros), initialize
        if np.all(init_magmom == 0):
             init_magmom = np.zeros(len(atoms))

        for symbol, mag in self.mag_map.items():
            indices = [atom.index for atom in atoms if atom.symbol == symbol]
            init_magmom[indices] = mag
        
        atoms.set_initial_magnetic_moments(init_magmom)

    def _set_kpoints(self, atoms: Atoms, vacuum_status: List[bool], is_cluster: bool = False) -> List[int]:
        """
        Calculate K-points based on lattice size and criteria.
        """
        kpoints = [1, 1, 1]
        if is_cluster:
            return kpoints
        
        cell_par = atoms.cell.cellpar()
        for dim in range(3):
            if vacuum_status[dim]:
                kpoints[dim] = 1
            else:
                length = cell_par[dim]
                if length < 1e-3:
                    kpoints[dim] = 1
                else:
                    kpoints[dim] = int(self.kpt_criteria / length) + 1
        return kpoints

    def _write_kpt_file(self, filename: str, kpoints: List[int]):
        """Write KPT file."""
        with open(filename, 'w') as f:
            f.write("K_POINTS\n0\nGamma\n")
            f.write(f"{kpoints[0]} {kpoints[1]} {kpoints[2]} 0 0 0\n")

    def generate(self, atoms: Atoms, output_dir: Union[str, Path], task_name: str, 
                 stru_type: str = None, vacuum_status: List[bool] = None, dataset_name: str = "unknown", system_name: str = "unknown"):
        """
        Generate input files for a single structure.
        
        Args:
            atoms: ASE Atoms object (preprocessed).
            output_dir: Directory to save files.
            task_name: Name of the task.
            stru_type: Pre-determined structure type (optional).
            vacuum_status: Pre-determined vacuum status (optional).
            dataset_name: Name of the dataset this task belongs to.
            system_name: Name of the system this task belongs to.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # If type/status not provided, analyze locally (fallback)
        if stru_type is None or vacuum_status is None:
            atoms, stru_type, vacuum_status = self.analyzer.analyze(atoms)
        
        # Metadata Injection (New feature)
        # Extract frame index from task_name if possible (format: sys_idx)
        try:
            frame_idx = int(task_name.split("_")[-1])
        except (ValueError, IndexError):
            frame_idx = -1

        import json
        meta = {
            "dataset_name": dataset_name,
            "system_name": system_name,
            "stru_type": stru_type,
            "task_name": task_name,
            "frame_idx": frame_idx
        }
        try:
            with open(output_dir / "task_meta.json", "w") as f:
                json.dump(meta, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to write task_meta.json: {e}")

        input_params = deepcopy(self.dft_params)
        
        # Layer specific params
        if stru_type == "layer":
            try:
                vac_dim = vacuum_status.index(True)
                input_params.update({
                    'efield_flag': 1,
                    'dip_cor_flag': 1,
                    'efield_dir': vac_dim
                })
            except ValueError:
                pass # Should not happen if logic is consistent

        # K-Points
        is_cluster = (stru_type in ["cluster", "cubic_cluster"])
        kpoints = self._set_kpoints(atoms, vacuum_status, is_cluster)
        input_params['kpts'] = kpoints
        
        # Gamma Only Check
        vac_count = sum(vacuum_status)
        if vac_count == 3 or kpoints == [1, 1, 1] or is_cluster:
            input_params['gamma_only'] = 1
        
        # Magnetism
        self._set_magmom(atoms)
        
        # Write Files
        input_params['pp'] = self.pp_map
        input_params['basis'] = self.orb_map
        input_params['pseudo_dir'] = self.pp_dir
        input_params['basis_dir'] = self.orb_dir 
        
        with open(output_dir / "INPUT", "w") as f:
            write_input(f, parameters=input_params)
            if 'orbital_dir' not in input_params and self.orb_dir:
                 f.write(f"orbital_dir {self.orb_dir}\n")

        with open(output_dir / "STRU", "w") as f:
            write_abacus(f, atoms, pp=self.pp_map, basis=self.orb_map)
            
        self._write_kpt_file(output_dir / "KPT", kpoints)
        
        # Dump viz files
        atoms.write(output_dir / f"{task_name}.cif")
        atoms.write(output_dir / f"{task_name}.extxyz")
        
        return stru_type

