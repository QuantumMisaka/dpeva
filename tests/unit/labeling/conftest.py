
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

@pytest.fixture
def mock_atoms_cluster():
    """Create a cluster-like Atoms object (large vacuum)."""
    atoms = Atoms('Fe2', positions=[[0, 0, 0], [2, 0, 0]], cell=[20, 20, 20], pbc=True)
    return atoms

@pytest.fixture
def mock_atoms_layer():
    """Create a layer-like Atoms object (vacuum in Z)."""
    atoms = Atoms('Fe4', positions=[[0, 0, 0], [2, 0, 0], [0, 2, 0], [2, 2, 0]], 
                 cell=[4, 4, 20], pbc=True)
    return atoms

@pytest.fixture
def mock_atoms_bulk():
    """Create a bulk-like Atoms object."""
    atoms = Atoms('Fe2', positions=[[0, 0, 0], [1.5, 1.5, 1.5]], 
                 cell=[3, 3, 3], pbc=True)
    return atoms

@pytest.fixture
def mock_dpdata_system():
    """Mock a dpdata.System object."""
    sys = MagicMock()
    sys.get_nframes.return_value = 2
    
    # Mock atoms
    atoms = Atoms('Fe2', positions=[[0, 0, 0], [2, 0, 0]], cell=[10, 10, 10], pbc=True)
    sys.to_ase_structure.return_value = [atoms, atoms]
    
    # Mock attributes
    sys.target_name = "sys_test"
    
    return sys

@pytest.fixture
def mock_multisystems(mock_dpdata_system):
    """Mock a dpdata.MultiSystems object."""
    ms = MagicMock()
    ms.__len__.return_value = 1
    ms.__iter__.return_value = iter([mock_dpdata_system])
    # Also make it subscriptable
    ms.__getitem__.return_value = mock_dpdata_system
    return ms
