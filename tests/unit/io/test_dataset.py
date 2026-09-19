import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from dpeva.io.dataset import DatasetLoadError, load_systems

class TestDatasetLoader:
    @pytest.fixture
    def data_dir(self, tmp_path):
        """Create a temporary directory structure mimicking dpdata."""
        root = tmp_path / "data"
        root.mkdir()
        
        # System 1: deepmd/npy (set.000 folder)
        sys1 = root / "sys1"
        sys1.mkdir()
        (sys1 / "type.raw").write_text("0 1")
        (sys1 / "set.000").mkdir()
        np.save(sys1 / "set.000" / "coord.npy", np.random.rand(10, 2, 3))
        np.save(sys1 / "set.000" / "box.npy", np.random.rand(10, 3, 3))
        np.save(sys1 / "set.000" / "energy.npy", np.random.rand(10))
        np.save(sys1 / "set.000" / "force.npy", np.random.rand(10, 2, 3))
        np.save(sys1 / "set.000" / "virial.npy", np.random.rand(10, 3, 3))
        
        # System 2: deepmd/npy/mixed (coord.npy directly in root? No, mixed is hdf5 or specific structure)
        # Actually mixed usually refers to MultiSystems loading multiple formats or folders.
        # But here we test auto-detection of standard npy structure.
        
        # Let's create another one
        sys2 = root / "sys2"
        sys2.mkdir()
        (sys2 / "type.raw").write_text("0")
        (sys2 / "set.000").mkdir()
        np.save(sys2 / "set.000" / "coord.npy", np.random.rand(5, 1, 3))
        np.save(sys2 / "set.000" / "box.npy", np.random.rand(5, 3, 3))
        
        return root

    def test_load_systems_auto(self, data_dir):
        """Test auto-detection of systems."""
        systems = load_systems(str(data_dir), fmt="auto")
        assert len(systems) == 2
        names = sorted([s.target_name for s in systems])
        assert names == ["sys1", "sys2"]
        
    def test_load_systems_target(self, data_dir):
        """Test loading specific target systems."""
        systems = load_systems(str(data_dir), fmt="auto", target_systems=["sys1"])
        assert len(systems) == 1
        assert systems[0].target_name == "sys1"

    def test_load_systems_target_with_redundant_root_prefix(self, tmp_path):
        data_dir = tmp_path / "other_dpdata"
        (data_dir / "sys1").mkdir(parents=True)

        with patch("dpeva.io.dataset._load_single_path") as mock_load:
            mock_sys = MagicMock()
            mock_load.return_value = mock_sys

            systems = load_systems(
                str(data_dir),
                fmt="auto",
                target_systems=["other_dpdata/sys1"],
            )

            assert len(systems) == 1
            mock_load.assert_called_once()
            call_args, _ = mock_load.call_args
            assert call_args[0] == str(data_dir / "sys1")
            assert call_args[1] == "other_dpdata/sys1"

    def test_load_systems_target_multipool_hierarchy_preserved(self, tmp_path):
        data_dir = tmp_path / "test_data"
        (data_dir / "poolA" / "sys1").mkdir(parents=True)

        with patch("dpeva.io.dataset._load_single_path") as mock_load:
            mock_sys = MagicMock()
            mock_load.return_value = mock_sys

            systems = load_systems(
                str(data_dir),
                fmt="auto",
                target_systems=["poolA/sys1"],
            )

            assert len(systems) == 1
            mock_load.assert_called_once()
            call_args, _ = mock_load.call_args
            assert call_args[0] == str(data_dir / "poolA" / "sys1")
            assert call_args[1] == "poolA/sys1"

    def test_load_systems_invalid_path(self, data_dir):
        """Test behavior with invalid path."""
        with pytest.raises(DatasetLoadError, match="auto-discover"):
            load_systems(str(data_dir / "nonexistent"), fmt="auto")

    def test_load_systems_invalid_path_can_be_tolerated(self, data_dir):
        """Callers that deliberately tolerate an empty dataset opt in explicitly."""
        systems = load_systems(
            str(data_dir / "nonexistent"), fmt="auto", on_empty="warn"
        )
        assert systems == []

    def test_load_systems_rejects_unknown_empty_policy(self, data_dir):
        with pytest.raises(ValueError, match="on_empty must be one of"):
            load_systems(str(data_dir), fmt="auto", on_empty="ignore")

    def test_load_single_system_optimization(self, data_dir):
        """Test the optimization path for loading a single system directory."""
        # sys1 is a valid system directory (has type.raw)
        sys1_path = data_dir / "sys1"
        
        # When calling load_systems on sys1_path, it should detect it's a single system
        # and return [sys] instead of scanning subdirs (like set.000)
        
        # We need to mock _load_single_path to verify it's called with the root dir
        with patch("dpeva.io.dataset._load_single_path") as mock_load:
            mock_sys = MagicMock()
            mock_load.return_value = mock_sys
            
            systems = load_systems(str(sys1_path), fmt="auto")
            
            assert len(systems) == 1
            assert systems[0] == mock_sys
            
            # Verify it was called with sys1_path, not sys1_path/set.000
            mock_load.assert_called_once()
            args, _ = mock_load.call_args
            assert args[0] == str(sys1_path)

    def test_load_systems_mixed_format(self, data_dir):
        """Test loading mixed format via MultiSystems."""
        # Mock dpdata.MultiSystems.from_file
        with patch("dpeva.io.dataset.dpdata.MultiSystems.from_file") as mock_multi:
            mock_ms = MagicMock()
            mock_ms.__len__.return_value = 2
            mock_ms.__iter__.return_value = [MagicMock(), MagicMock()]
            mock_multi.return_value = mock_ms
            
            systems = load_systems(str(data_dir), fmt="auto")
            
            assert len(systems) == 2
            # Should have tried deepmd/npy/mixed first
            mock_multi.assert_any_call(str(data_dir), fmt="deepmd/npy/mixed")

    def test_load_systems_fallback_filtering(self, data_dir):
        """Test fallback scanning filters out set.* directories."""
        # Add a set.000 dir to root data_dir (simulating a system dir treated as root)
        (data_dir / "set.000").mkdir()
        (data_dir / "set.000" / "coord.npy").touch() # dummy
        
        # Also add a valid subdir "subsys"
        (data_dir / "subsys").mkdir()
        
        # Mock _load_single_path to succeed for subsys and fail for others
        def side_effect(path, name, fmt="auto"):
            if "set.000" in path:
                raise ValueError("Should not be called")
            if "subsys" in path:
                return MagicMock()
            if "sys1" in path or "sys2" in path: # existing fixtures
                return MagicMock()
            raise ValueError(f"Unknown path {path}")
            
        with patch("dpeva.io.dataset._load_single_path", side_effect=side_effect) as mock_load:
            # We also need to mock MultiSystems to fail so it goes to fallback
            with patch("dpeva.io.dataset.dpdata.MultiSystems.from_file", side_effect=Exception("Fail")):
                systems = load_systems(str(data_dir), fmt="auto")

                # The fallback scan must load the real system directories and
                # never treat the internal set.000 folder as a system.
                assert len(systems) == 3
                loaded_paths = [call[0][0] for call in mock_load.call_args_list]
                assert any("sys1" in path for path in loaded_paths)
                assert any("subsys" in path for path in loaded_paths)
                assert not any("set.000" in path for path in loaded_paths)
 

    def test_fix_duplicate_atom_names(self):
        """Test duplicate atom name merging logic."""
        # Create a mock system
        sys = MagicMock()
        sys.__getitem__.side_effect = lambda k: {
            "atom_names": ["Fe", "Fe", "O"],
            "atom_types": np.array([0, 0, 1, 1, 2, 0]), # indices into atom_names
            # 0->Fe, 1->Fe, 2->O.
            # Real types: Fe, Fe, O, Fe, O, Fe
        }[k]
        sys.data = {}
        
        from dpeva.io.dataset import _fix_duplicate_atom_names
        _fix_duplicate_atom_names(sys, "test_sys")
        
        # Expected:
        # New names: ["Fe", "O"]
        # Old map: 0->0 (Fe->Fe), 1->0 (Fe->Fe), 2->1 (O->O)
        # Old types: 0, 0, 1, 1, 2, 0
        # New types: 0, 0, 0, 0, 1, 0
        
        assert sys.data["atom_names"] == ["Fe", "O"]
        np.testing.assert_array_equal(sys.data["atom_types"], np.array([0, 0, 0, 0, 1, 0]))
