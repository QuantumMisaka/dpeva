
import os
import shutil
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from dpeva.io.collection import CollectionIOManager

class TestCollectionIOManagerFull:
    
    @pytest.fixture
    def manager(self, tmp_path):
        project = tmp_path / "project"
        root = "dpeva_uq"
        return CollectionIOManager(str(project), root)

    def test_ensure_dirs(self, manager):
        """Test directory creation."""
        manager.ensure_dirs()
        assert os.path.exists(manager.view_savedir)
        assert os.path.exists(manager.dpdata_savedir)
        assert os.path.exists(manager.df_savedir)

    def test_count_frames(self, manager, tmp_path):
        """Test counting frames."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        with patch("dpeva.io.collection.load_systems") as mock_load:
            mock_load.return_value = [MagicMock(__len__=lambda x: 10), MagicMock(__len__=lambda x: 5)]
            count = manager.count_frames(str(data_dir))
            assert count == 15
            
            # Error case
            mock_load.side_effect = Exception("Failed")
            count = manager.count_frames(str(data_dir))
            assert count == 0

    def test_load_descriptors_specific(self, manager, tmp_path):
        """Test loading specific descriptors."""
        desc_dir = tmp_path / "desc"
        desc_dir.mkdir()
        
        # Create dummy descriptors (N_frames, N_atoms, N_features)
        # Using 1 atom for simplicity
        np.save(desc_dir / "sys1.npy", np.random.rand(10, 1, 128))
        np.save(desc_dir / "sys2.npy", np.random.rand(5, 1, 128))
        
        targets = ["sys1"]
        names, data = manager.load_descriptors(str(desc_dir), target_names=targets)
        
        assert len(names) == 10
        assert data.shape == (10, 128) # Normalized descriptors, not modulo

    def test_load_descriptors_glob(self, manager, tmp_path):
        """Test loading all descriptors via glob."""
        desc_dir = tmp_path / "desc_glob"
        desc_dir.mkdir()
        
        np.save(desc_dir / "sys1.npy", np.random.rand(10, 1, 128))
        
        names, data = manager.load_descriptors(str(desc_dir))
        
        assert len(names) == 10

    def test_load_descriptors_mismatch(self, manager, tmp_path):
        """Test handling of frame mismatch."""
        desc_dir = tmp_path / "desc_mismatch"
        desc_dir.mkdir()
        
        np.save(desc_dir / "sys1.npy", np.random.rand(10, 1, 128))
        
        # Expected 5, got 10 -> Truncate
        expected = {"sys1": 5}
        names, data = manager.load_descriptors(str(desc_dir), target_names=["sys1"], expected_frames=expected)
        assert len(names) == 5
        
        # Expected 15, got 10 -> Error
        expected = {"sys1": 15}
        with pytest.raises(ValueError):
            manager.load_descriptors(str(desc_dir), target_names=["sys1"], expected_frames=expected)

    def test_load_atomic_features(self, manager, tmp_path):
        """Test loading atomic features."""
        desc_dir = tmp_path / "desc_atom"
        desc_dir.mkdir()
        
        # Create atomic features (list of arrays usually, but saved as object array or just array if uniform)
        # Here we simulate array of arrays logic? 
        # load_atomic_features expects .npy to contain a list/array where each element is atomic features for a frame
        
        # Frame 0: 2 atoms, 5 features
        # Frame 1: 2 atoms, 5 features
        feat = np.random.rand(2, 2, 5) 
        np.save(desc_dir / "sys1.npy", feat)
        
        df = pd.DataFrame({"dataname": ["sys1-0", "sys1-1"]})
        
        X, n = manager.load_atomic_features(str(desc_dir), df)
        
        assert len(X) == 2
        assert n == [2, 2]

    def test_export_dpdata(self, manager, tmp_path):
        """Test exporting to dpdata."""
        testdata_dir = tmp_path / "testdata"
        testdata_dir.mkdir()
        
        df_final = pd.DataFrame({"dataname": ["sys1-0", "sys1-2"]}) # Select frame 0 and 2
        unique_systems = ["sys1"]
        
        with patch("dpeva.io.collection.load_systems") as mock_load:
            sys_mock = MagicMock()
            sys_mock.target_name = "sys1"
            sys_mock.__len__.return_value = 5 # 5 frames total
            
            # sub_system returns a new system
            sub_mock = MagicMock()
            sys_mock.sub_system.return_value = sub_mock
            
            mock_load.return_value = [sys_mock]
            
            manager.ensure_dirs()
            manager.export_dpdata(str(testdata_dir), df_final, unique_systems)
            
            # Verify calls
            # Sampled: 0, 2
            sys_mock.sub_system.assert_any_call([0, 2])
            # Other: 1, 3, 4
            sys_mock.sub_system.assert_any_call([1, 3, 4])
            
            assert sub_mock.to_deepmd_npy.call_count == 2
