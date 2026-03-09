"""
Structure Analysis and Preprocessing Module
===========================================

Provides tools for analyzing atomic structures, determining their dimensionality,
and preprocessing them for DFT calculations.
"""

import logging
from typing import List, Tuple, Optional
import numpy as np
from ase import Atoms

logger = logging.getLogger(__name__)

class StructureAnalyzer:
    """
    Analyzer for atomic structures.
    Responsible for:
    1. Preprocessing (wrap, center, shift)
    2. Vacuum detection
    3. Dimensionality classification (Bulk, Layer, String, Cluster)
    4. Standardization (aligning vacuum to Z-axis)
    """

    def __init__(self, vacuum_thickness: float = 6.3, 
                 cubic_min_length: float = 9.8, 
                 cubic_tol: float = 6.3):
        """
        Initialize the analyzer.

        Args:
            vacuum_thickness (float): Minimum thickness to be considered vacuum (Angstrom).
            cubic_min_length (float): Minimum length for cubic cluster detection.
            cubic_tol (float): Tolerance for cubic cluster detection.
        """
        self.vacuum_thickness = vacuum_thickness
        self.cubic_min_length = cubic_min_length
        self.cubic_tol = cubic_tol

    def preprocess(self, atoms: Atoms) -> Atoms:
        """
        Wrap, center, and ensure atoms are not split across boundaries.
        """
        atoms = atoms.copy()
        atoms.wrap()
        atoms.center()
        
        scaled_pos = atoms.get_scaled_positions()
        shifted = False
        for dim in range(3):
            # Check if atoms are concentrated at boundaries
            # Heuristic: if < 45% of atoms are in the middle 50% (0.25-0.75)
            mask_middle = (scaled_pos[:, dim] > 0.25) & (scaled_pos[:, dim] < 0.75)
            if len(atoms) > 0 and np.sum(mask_middle) / len(atoms) < 0.45:
                # Shift atoms from > 0.67 to left by 1.0
                mask_high = scaled_pos[:, dim] > 0.67
                scaled_pos[mask_high, dim] -= 1.0
                shifted = True
        
        if shifted:
            atoms.set_scaled_positions(scaled_pos)
            atoms.center()
        
        return atoms

    def judge_vacuum(self, atoms: Atoms) -> List[bool]:
        """
        Detect vacuum along x, y, z axes.
        """
        atoms_copy = atoms.copy()
        atoms_copy.center()
        vacuum_status = []
        for dim in range(3):
            dim_pos = atoms_copy.positions[:, dim]
            if len(dim_pos) == 0:
                # Empty structure? Treat as vacuum
                vacuum_status.append(True)
                continue
                
            dim_pos_gap = max(dim_pos) - min(dim_pos)
            dim_lat = atoms_copy.cell[dim, dim]
            if dim_pos_gap > dim_lat - self.vacuum_thickness:
                vacuum_status.append(False)
            else:
                vacuum_status.append(True)
        return vacuum_status

    def swap_crystal_lattice(self, atoms: Atoms, swap_indices: List[int]) -> Atoms:
        """Swap lattice vectors (e.g., to align vacuum with Z)."""
        new_atoms = atoms.copy()
        cell_par = new_atoms.cell.cellpar()
        
        # Swap lengths
        cell_par[swap_indices[0]], cell_par[swap_indices[1]] = \
            cell_par[swap_indices[1]], cell_par[swap_indices[0]]
        # Swap angles (indices 3,4,5 correspond to alpha, beta, gamma)
        cell_par[swap_indices[0] + 3], cell_par[swap_indices[1] + 3] = \
            cell_par[swap_indices[1] + 3], cell_par[swap_indices[0] + 3]
            
        new_atoms.set_cell(cell_par)
        
        # Swap scaled positions
        scaled_pos = atoms.get_scaled_positions()
        scaled_pos[:, swap_indices[0]], scaled_pos[:, swap_indices[1]] = \
            scaled_pos[:, swap_indices[1]], scaled_pos[:, swap_indices[0]]
        new_atoms.set_scaled_positions(scaled_pos)
        
        return new_atoms

    def analyze(self, atoms: Atoms) -> Tuple[Atoms, str, List[bool]]:
        """
        Analyze and preprocess structure to determine type and vacuum status.
        
        Returns:
            Tuple containing:
            - processed_atoms: Centered and wrapped atoms object
            - stru_type: Determined structure type (bulk, cluster, cubic_cluster, layer, string)
            - vacuum_status: List of booleans indicating vacuum along x, y, z
        """
        # Preprocess
        processed_atoms = self.preprocess(atoms)
        vacuum_status = self.judge_vacuum(processed_atoms)
        vac_count = sum(vacuum_status)
        
        stru_type = "bulk"
        
        if vac_count == 3:
            stru_type = "cluster"
        elif vac_count == 2:
            stru_type = "string"
            # Move extended dim to Z (2) if needed
            extend_dim = vacuum_status.index(False)
            if extend_dim != 2:
                processed_atoms = self.swap_crystal_lattice(processed_atoms, [extend_dim, 2])
                vacuum_status[extend_dim], vacuum_status[2] = vacuum_status[2], vacuum_status[extend_dim]
                
        elif vac_count == 1:
            stru_type = "layer"
            # Identify in-plane dimensions
            vac_dim = vacuum_status.index(True)
            plane_dims = [i for i in range(3) if i != vac_dim]
            lengths = processed_atoms.cell.cellpar()[:3]
            # Identify longer dimension
            long_dim = plane_dims[0] if lengths[plane_dims[0]] >= lengths[plane_dims[1]] else plane_dims[1]
            
            # Swap longer dim to Z (2) if needed
            if long_dim != 2:
                processed_atoms = self.swap_crystal_lattice(processed_atoms, [long_dim, 2])
                vacuum_status[long_dim], vacuum_status[2] = vacuum_status[2], vacuum_status[long_dim]
        
        # Check for Cubic Cluster
        if vac_count != 0:
            cell_par = processed_atoms.cell.cellpar()
            is_cubic = (
                (min(cell_par[:3]) >= self.cubic_min_length) and
                (abs(processed_atoms.cell[0,0] - processed_atoms.cell[1,1]) < 1e-6) and
                (abs(processed_atoms.cell[1,1] - processed_atoms.cell[2,2]) < 1e-6) and
                (abs(min(processed_atoms.cell.flatten())) < 1e-6) # Check off-diagonal approx 0
            )
            if is_cubic:
                stru_type = "cubic_cluster"
                
        return processed_atoms, stru_type, vacuum_status
