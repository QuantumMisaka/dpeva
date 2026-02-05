import pytest
import numpy as np
from dpeva.sampling.two_step_direct import TwoStepDIRECTSampler
from dpeva.sampling.direct import BirchClustering

class TestTwoStepDIRECTSampler:
    
    @pytest.fixture
    def mock_data(self):
        # Create synthetic data
        np.random.seed(42)
        n_structures = 20
        dim_stru = 10
        dim_atom = 5
        
        X_stru = np.random.rand(n_structures, dim_stru)
        
        X_atom_list = []
        n_atoms_list = []
        
        for i in range(n_structures):
            n_atoms = np.random.randint(2, 6) # At least 2 atoms
            X_atom_list.append(np.random.rand(n_atoms, dim_atom))
            n_atoms_list.append(n_atoms)
            
        return X_stru, X_atom_list, n_atoms_list

    def test_init(self):
        sampler = TwoStepDIRECTSampler()
        assert sampler.step1_clustering.n is None
        assert sampler.step2_clustering.n is None
        assert sampler.step2_selection.k == 5
        assert sampler.step2_selection.selection_criteria == "smallest"

    def test_fit_transform_basic(self, mock_data):
        X_stru, X_atom_list, n_atoms_list = mock_data
        
        # Override for small data
        # Use simple params to avoid convergence warnings
        sampler = TwoStepDIRECTSampler()
        sampler.step1_clustering.n = 3
        sampler.step2_clustering.n = 3
        sampler.step2_selection.k = 1
        
        result = sampler.fit_transform(X_stru, X_atom_list, n_atoms_list)
        
        assert "selected_indices" in result
        assert "step1_labels" in result
        assert "PCAfeatures" in result
        
        selected = result["selected_indices"]
        assert len(selected) > 0
        assert len(selected) <= 20
        
        # Check uniqueness
        assert len(selected) == len(set(selected))
        
        # Check step1 labels
        assert len(result["step1_labels"]) == 20
        
    def test_selection_logic_smallest(self):
        # Test if it actually picks the smallest atoms
        # Create 2 structures, same structural feature (same cluster)
        # Struct 0: 10 atoms
        # Struct 1: 2 atoms
        # Atomic features: make them similar so they fall in same atomic cluster
        
        # Use slightly different values to avoid 0 variance (PCA failure)
        X_stru = np.array([[1.0, 1.0], [1.01, 1.01]]) 
        
        # Atomic features: clustered around 0
        X_atom_0 = np.random.normal(loc=0, scale=0.01, size=(10, 2))
        X_atom_1 = np.random.normal(loc=0, scale=0.01, size=(2, 2))
        
        X_atom_list = [X_atom_0, X_atom_1]
        n_atoms_list = [10, 2]
        
        sampler = TwoStepDIRECTSampler()
        sampler.step1_clustering.n = 1
        sampler.step2_clustering.n = 1
        sampler.step2_selection.k = 1 # Select 1 from atomic cluster
        
        result = sampler.fit_transform(X_stru, X_atom_list, n_atoms_list)
        
        selected = result["selected_indices"]
        # Should pick struct 1 because it has fewer atoms (2 < 10)
        assert 1 in selected
        assert 0 not in selected
