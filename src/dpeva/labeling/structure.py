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

    def align_axis_to_z(self, atoms: Atoms, axis: int) -> Atoms:
        """
        Rotate the cell to align the specified axis to the Z-axis.
        Performs a cyclic permutation to ensure det(R)=1 (proper rotation).
        
        Args:
            atoms: Atoms object
            axis: The axis index (0, 1, or 2) to align to Z.
            
        Returns:
            Rotated Atoms object with axis aligned to Z.
        """
        if axis == 2:
            return atoms.copy()
            
        new_atoms = atoms.copy()
        
        # Determine cyclic permutation to map 'axis' to index 2 (Z)
        # We want New[2] to correspond to Old[axis].
        # Cyclic permutations of (0, 1, 2) are:
        # (0, 1, 2) -> Identity
        # (1, 2, 0) -> 0 moves to 2, 1 moves to 0, 2 moves to 1.
        # (2, 0, 1) -> 0 moves to 1, 1 moves to 2, 2 moves to 0.
        
        if axis == 0:
            # We want Old[0] at New[2].
            # Permutation: [1, 2, 0]
            # New[0] <= Old[1]
            # New[1] <= Old[2]
            # New[2] <= Old[0]
            perm = [1, 2, 0]
        elif axis == 1:
            # We want Old[1] at New[2].
            # Permutation: [2, 0, 1]
            # New[0] <= Old[2]
            # New[1] <= Old[0]
            # New[2] <= Old[1]
            perm = [2, 0, 1]
        else:
            raise ValueError("Axis must be 0, 1, or 2")
            
        # Apply permutation to cell parameters
        old_cellpar = new_atoms.cell.cellpar()
        new_cellpar = np.zeros_like(old_cellpar)
        
        # Lengths (0-2)
        new_cellpar[0] = old_cellpar[perm[0]]
        new_cellpar[1] = old_cellpar[perm[1]]
        new_cellpar[2] = old_cellpar[perm[2]]
        
        # Angles (3-5)
        # Helper to map vector indices to angle index
        # (1,2)->3 (alpha), (0,2)->4 (beta), (0,1)->5 (gamma)
        def get_angle_index(i, j):
            pair = tuple(sorted((i, j)))
            if pair == (1, 2): return 3
            if pair == (0, 2): return 4
            if pair == (0, 1): return 5
            return -1
            
        # New alpha (angle between New[1] and New[2])
        new_cellpar[3] = old_cellpar[get_angle_index(perm[1], perm[2])]
        # New beta (angle between New[0] and New[2])
        new_cellpar[4] = old_cellpar[get_angle_index(perm[0], perm[2])]
        # New gamma (angle between New[0] and New[1])
        new_cellpar[5] = old_cellpar[get_angle_index(perm[0], perm[1])]
        
        new_atoms.set_cell(new_cellpar)
        
        # Apply permutation to scaled positions
        # Ensure wrapping to [0, 1) before rotation to avoid C-component drift
        old_scaled = atoms.get_scaled_positions(wrap=True)
        new_scaled = np.zeros_like(old_scaled)
        new_scaled[:, 0] = old_scaled[:, perm[0]]
        new_scaled[:, 1] = old_scaled[:, perm[1]]
        new_scaled[:, 2] = old_scaled[:, perm[2]]
        
        new_atoms.set_scaled_positions(new_scaled)
        
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
                processed_atoms = self.align_axis_to_z(processed_atoms, extend_dim)
                # Update vacuum status after rotation
                # Since we rotated cyclicly:
                # If X (0) -> Z (2). [1, 2, 0].
                # New vac[0] = Old vac[1]
                # New vac[1] = Old vac[2]
                # New vac[2] = Old vac[0]
                # We can re-judge vacuum to be safe and simple
                vacuum_status = self.judge_vacuum(processed_atoms)
                
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
                processed_atoms = self.align_axis_to_z(processed_atoms, long_dim)
                # Re-judge vacuum status
                vacuum_status = self.judge_vacuum(processed_atoms)
        
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
