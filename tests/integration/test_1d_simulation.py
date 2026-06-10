
import numpy as np
from ase import Atoms
from dpeva.labeling.structure import StructureAnalyzer

def create_1d_structure_along_axis(axis=0):
    """
    Create a 1D structure (nanowire) along a specific axis.
    20 atoms, 3 elements (Fe, C, O).
    """
    np.random.seed(42)
    n_atoms = 20
    
    # Define dimensions
    # Periodic dimension length
    L_periodic = 10.0
    # Vacuum dimension length
    L_vacuum = 50.0
    
    cell_dims = [0, 0, 0]
    
    # Set cell dimensions based on axis
    for i in range(3):
        if i == axis:
            cell_dims[i] = L_periodic
        else:
            cell_dims[i] = L_vacuum
            
    # Generate positions
    # Spread along periodic axis
    # Confined in vacuum axes (center of cell)
    positions = []
    symbols = []
    elements = ['Fe', 'C', 'O']
    
    for i in range(n_atoms):
        pos = [0.0, 0.0, 0.0]
        for dim in range(3):
            if dim == axis:
                # Random position along the wire
                pos[dim] = np.random.uniform(0, L_periodic)
            else:
                # Centered with small thermal fluctuation/thickness
                # Center is L_vacuum / 2
                center = L_vacuum / 2.0
                # Fluctuation +/- 1.0 Angstrom
                pos[dim] = np.random.uniform(center - 1.0, center + 1.0)
        
        positions.append(pos)
        symbols.append(elements[i % 3])
        
    atoms = Atoms(symbols=symbols, positions=positions, cell=cell_dims, pbc=True)
    return atoms

def test_1d_structure_preprocessing_x_axis():
    """
    Test preprocessing of a 1D structure aligned along the X-axis.
    It should be rotated to align with the Z-axis.
    """
    # 1. Create 1D structure along X-axis (index 0)
    atoms_x = create_1d_structure_along_axis(axis=0)
    
    # Verify initial state
    # Vacuum should be [False, True, True] roughly?
    # Analyzer uses vacuum thickness 6.3
    # X len=10. Spread ~10. Gap ~0. -> Not vacuum.
    # Y len=50. Spread ~2. Gap ~48 > 6.3. -> Vacuum.
    # Z len=50. Spread ~2. Gap ~48 > 6.3. -> Vacuum.
    # So initial vacuum: [False, True, True]
    
    analyzer = StructureAnalyzer()
    
    # 2. Run analysis
    processed_atoms, stru_type, vacuum_status = analyzer.analyze(atoms_x)
    
    # 3. Verify Structure Type
    print(f"Detected Type: {stru_type}")
    print(f"Vacuum Status: {vacuum_status}")
    
    assert stru_type == "string"
    
    # 4. Verify Rotation
    # The non-vacuum axis should now be Z (index 2).
    # So vacuum_status should be [True, True, False]
    assert vacuum_status == [True, True, False]
    
    # Verify Cell Dimensions
    # Original X (10.0) should be at Z (index 2)
    # Original Y, Z (50.0) should be at X, Y
    cell_par = processed_atoms.cell.cellpar()
    print(f"New Cell Params: {cell_par}")
    
    assert np.isclose(cell_par[2], 10.0)
    assert np.isclose(cell_par[0], 50.0)
    assert np.isclose(cell_par[1], 50.0)
    
    # 5. Verify Coordinates Integrity
    # Check if atoms are still within the cell (wrapped)
    scaled_pos = processed_atoms.get_scaled_positions()
    assert np.all(scaled_pos >= -1e-5)
    assert np.all(scaled_pos < 1.0 + 1e-5)
    
    # Check wire confinement
    # The wire should now be along Z.
    # So X and Y coordinates should be centered (approx 0.5 in scaled)
    # Width was ~2.0 A in 50.0 A cell => 0.04 in scaled units.
    # So X and Y scaled positions should be close to 0.5.
    assert np.all(np.abs(scaled_pos[:, 0] - 0.5) < 0.1)
    assert np.all(np.abs(scaled_pos[:, 1] - 0.5) < 0.1)
    
    # Z coordinates should be spread out (uniform)
    z_spread = np.max(scaled_pos[:, 2]) - np.min(scaled_pos[:, 2])
    assert z_spread > 0.5 # Should occupy significant portion of Z

def test_1d_structure_preprocessing_z_axis():
    """
    Test preprocessing of a 1D structure already along the Z-axis.
    It should remain unchanged.
    """
    # 1. Create 1D structure along Z-axis (index 2)
    atoms_z = create_1d_structure_along_axis(axis=2)
    
    analyzer = StructureAnalyzer()
    
    # 2. Run analysis
    processed_atoms, stru_type, vacuum_status = analyzer.analyze(atoms_z)
    
    # 3. Verify
    assert stru_type == "string"
    # Vacuum in X, Y. Not Z.
    assert vacuum_status == [True, True, False]
    
    # Cell should be [50, 50, 10]
    cell_par = processed_atoms.cell.cellpar()
    assert np.isclose(cell_par[2], 10.0)
    assert np.isclose(cell_par[0], 50.0)
    
    # Verify it didn't rotate unnecessarily
    # (Implicitly checked by dimensions)

if __name__ == "__main__":
    # Manually run if executed as script
    try:
        test_1d_structure_preprocessing_x_axis()
        print("test_1d_structure_preprocessing_x_axis PASSED")
        test_1d_structure_preprocessing_z_axis()
        print("test_1d_structure_preprocessing_z_axis PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        raise
