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
from ase.io.abacus import write_input, write_abacus # ase-abacus is needed for now

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
        self.vacuum_thickness = config.get("vacuum_thickness", 6.3)
        
        # Cubic cluster detection params
        self.cubic_symprec_decimal = 0
        self.cubic_min_length = 9.8
        self.cubic_tol = 6.3

    def _judge_vacuum(self, atoms: Atoms) -> List[bool]:
        """
        Detect vacuum along x, y, z axes.
        """
        atoms_copy = atoms.copy()
        atoms_copy.center()
        vacuum_status = []
        for dim in range(3):
            dim_pos = atoms_copy.positions[:, dim]
            dim_pos_gap = max(dim_pos) - min(dim_pos)
            dim_lat = atoms_copy.cell[dim, dim]
            if dim_pos_gap > dim_lat - self.vacuum_thickness:
                vacuum_status.append(False)
            else:
                vacuum_status.append(True)
        return vacuum_status

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

    def _swap_crystal_lattice(self, atoms: Atoms, swap_indices: List[int]) -> Atoms:
        """Swap lattice vectors (e.g., to align vacuum with Z)."""
        new_atoms = atoms.copy()
        cell_par = new_atoms.cell.cellpar()
        
        # Swap lengths
        cell_par[swap_indices[0]], cell_par[swap_indices[1]] = \
            cell_par[swap_indices[1]], cell_par[swap_indices[0]]
        # Swap angles (indices 3,4,5 correspond to alpha, beta, gamma)
        # alpha(3) is angle between b,c; beta(4) a,c; gamma(5) a,b
        # This simple swap logic assumes orthorhombic-like or simple swap needs
        # For full crystallography, use spglib or ase.build tools if needed.
        # Following original script logic for now.
        cell_par[swap_indices[0] + 3], cell_par[swap_indices[1] + 3] = \
            cell_par[swap_indices[1] + 3], cell_par[swap_indices[0] + 3]
            
        new_atoms.set_cell(cell_par)
        
        # Swap scaled positions
        scaled_pos = atoms.get_scaled_positions()
        scaled_pos[:, swap_indices[0]], scaled_pos[:, swap_indices[1]] = \
            scaled_pos[:, swap_indices[1]], scaled_pos[:, swap_indices[0]]
        new_atoms.set_scaled_positions(scaled_pos)
        
        return new_atoms

    def _preprocess_structure(self, atoms: Atoms) -> Atoms:
        """
        Wrap, center, and ensure atoms are not split across boundaries.
        """
        atoms.wrap()
        atoms.center()
        
        scaled_pos = atoms.get_scaled_positions()
        shifted = False
        for dim in range(3):
            # Check if atoms are concentrated at boundaries
            # Heuristic: if < 45% of atoms are in the middle 50% (0.25-0.75)
            mask_middle = (scaled_pos[:, dim] > 0.25) & (scaled_pos[:, dim] < 0.75)
            if np.sum(mask_middle) / len(atoms) < 0.45:
                # Shift atoms from > 0.67 to left by 1.0
                mask_high = scaled_pos[:, dim] > 0.67
                scaled_pos[mask_high, dim] -= 1.0
                shifted = True
        
        if shifted:
            atoms.set_scaled_positions(scaled_pos)
            atoms.center()
        
        return atoms

    def generate(self, atoms: Atoms, output_dir: Union[str, Path], task_name: str):
        """
        Generate input files for a single structure.

        Args:
            atoms: ASE Atoms object.
            output_dir: Directory to save files.
            task_name: Name of the task (used for file naming if needed).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        atoms = self._preprocess_structure(atoms)
        vacuum_status = self._judge_vacuum(atoms)
        
        input_params = deepcopy(self.dft_params)
        
        # Dimension Analysis & Classification
        stru_type = "bulk"
        vac_count = sum(vacuum_status)
        
        if vac_count == 3:
            stru_type = "cluster"
        elif vac_count == 2:
            stru_type = "string"
            # Move extended dim to Z (2)
            # vac_status has one False (the extended dim)
            extend_dim = vacuum_status.index(False)
            if extend_dim != 2:
                atoms = self._swap_crystal_lattice(atoms, [extend_dim, 2])
                vacuum_status[extend_dim], vacuum_status[2] = vacuum_status[2], vacuum_status[extend_dim]
        elif vac_count == 1:
            stru_type = "layer"
            # Move vacuum dim to Z (2)
            # The original script logic for layer swaps the longer in-plane vector to C-axis (Z-axis)
            # if necessary, and then updates vacuum_status.
            
            vac_dim = vacuum_status.index(True)
            plane_dims = [i for i in range(3) if i != vac_dim]
            lengths = atoms.cell.cellpar()[:3]
            long_dim = plane_dims[0] if lengths[plane_dims[0]] >= lengths[plane_dims[1]] else plane_dims[1]
            
            if long_dim != 2:
                atoms = self._swap_crystal_lattice(atoms, [long_dim, 2])
                vacuum_status[long_dim], vacuum_status[2] = vacuum_status[2], vacuum_status[long_dim]
            
            # Recalc vacuum dim after swap
            vac_dim = vacuum_status.index(True)
            input_params.update({
                'efield_flag': 1,
                'dip_cor_flag': 1,
                'efield_dir': vac_dim
            })

        elif vac_count == 0:
            stru_type = "bulk"
        
        # Cubic Cluster Check
        if vac_count != 0:
            cell_par = atoms.cell.cellpar()
            is_cubic = (
                (min(cell_par[:3]) >= self.cubic_min_length) and
                (abs(atoms.cell[0,0] - atoms.cell[1,1]) < 1e-6) and
                (abs(atoms.cell[1,1] - atoms.cell[2,2]) < 1e-6) and
                (abs(min(atoms.cell.flatten())) < 1e-6) # Check off-diagonal approx 0
            )
            if is_cubic:
                stru_type = "cubic_cluster"

        # K-Points
        is_cluster = (stru_type in ["cluster", "cubic_cluster"])
        kpoints = self._set_kpoints(atoms, vacuum_status, is_cluster)
        input_params['kpts'] = kpoints
        
        if vac_count == 3 or kpoints == [1, 1, 1] or is_cluster:
            input_params['gamma_only'] = 1
        
        # Magnetism
        self._set_magmom(atoms)
        
        # Write Files
        # INPUT
        # Add pp_dir/orb_dir to params if not present (though ABACUS usually reads from INPUT)
        
        # We also need to map pp and basis for each element
        # 'pp': {'Fe': 'Fe.upf', ...}
        # 'basis': {'Fe': 'Fe.orb', ...}
        input_params['pp'] = self.pp_map
        input_params['basis'] = self.orb_map
        input_params['pseudo_dir'] = self.pp_dir
        input_params['basis_dir'] = self.orb_dir # ASE might not write this standard key
        
        with open(output_dir / "INPUT", "w") as f:
            write_input(f, parameters=input_params)
            # Manually append orbital_dir if ASE doesn't (ASE's write_input is basic)
            if 'orbital_dir' not in input_params and self.orb_dir:
                 f.write(f"orbital_dir {self.orb_dir}\n")

        # STRU
        with open(output_dir / "STRU", "w") as f:
            write_abacus(f, atoms, pp=self.pp_map, basis=self.orb_map)
            
        # KPT
        self._write_kpt_file(output_dir / "KPT", kpoints)
        
        # Dump viz files
        atoms.write(output_dir / f"{task_name}.cif")
        atoms.write(output_dir / f"{task_name}.extxyz")
        
        return stru_type

