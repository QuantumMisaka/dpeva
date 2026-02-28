import pytest
import numpy as np
import os
from unittest.mock import MagicMock, patch
from dpeva.io.dataproc import DPTestResultParser

def create_dummy_files(dirname, head="results", n_frames=2, n_atoms=3):
    """Creates dummy output files for testing."""
    e_file = os.path.join(dirname, f"{head}.e_peratom.out")
    f_file = os.path.join(dirname, f"{head}.f.out")
    
    # Create Energy File with comments
    with open(e_file, "w") as f:
        f.write("# system1: 0\n")
        f.write("# system2: 1\n") # Frame 0 is system1, Frame 1 is system2? No, indices are usually start lines?

        
        # Let's mimic standard output
        # # pool/sys1: 0
        # 1.0 1.1
        # 1.0 1.1
        # # pool/sys2: 3
        # 2.0 2.1
        
        f.write("# pool/sys1: 0\n")
        for _ in range(n_frames):
            f.write("1.0 1.1\n")
            
        f.write(f"# pool/sys2: {n_frames + 1}\n")
        for _ in range(n_frames):
            f.write("2.0 2.1\n")
            
    # Create Force File
    # Format: fx fy fz pfx pfy pfz
    total_lines = (n_frames * 2) # 2 systems
    with open(f_file, "w") as f:
        for _ in range(total_lines):
            # 3 atoms per frame -> Flattened? No, force file is usually [n_frames, 3*n_atoms + 3*n_atoms] ?
            # Actually genfromtxt loads it.
            # If standard deepmd, it's one line per frame containing all atoms flattened?
            # Or one line per atom?
            # DPTestResultParser uses `np.genfromtxt(names=["data_fx", ...])`.
            # This implies columnar data.
            # Usually .f.out is (N_frames * N_atoms, 6).
            
            for _ in range(n_atoms):
                f.write("0.1 0.2 0.3 0.11 0.21 0.31\n")

class TestDPTestResultParser:
    
    @pytest.fixture
    def result_dir(self, tmp_path):
        d = tmp_path / "results"
        d.mkdir()
        return d

    def test_parser_basic(self, result_dir):
        """
        Test parsing of basic energy and force files.
        Verifies that the parser correctly extracts system names and ground truth status.
        """
        create_dummy_files(str(result_dir), n_frames=1, n_atoms=2)
        
        parser = DPTestResultParser(result_dir=str(result_dir), head="results")
        results = parser.parse()
        
        assert results["energy"] is not None, "Energy data should be parsed"
        assert results["force"] is not None, "Force data should be parsed"
        assert results["has_ground_truth"] is True, "Should detect ground truth from non-zero values"
        
        # Check dataname list
        # Expected: 2 systems, 1 frame each.
        # sys1: frame 0. sys2: frame 0.
        assert len(results["dataname_list"]) == 2, "Should parse 2 frames from dataname list"
        assert results["dataname_list"][0][0] == "pool/sys1"
        assert results["dataname_list"][1][0] == "pool/sys2"

    def test_parser_no_ground_truth(self, result_dir):
        """
        Test parsing when data columns are zero.
        Should set has_ground_truth to False.
        """
        # Create zero-filled files
        e_file = result_dir / "results.e_peratom.out"
        f_file = result_dir / "results.f.out"
        
        with open(e_file, "w") as f:
            f.write("# sys1: 0\n")
            f.write("0.0 1.0\n")
            
        with open(f_file, "w") as f:
            # 1 atom
            f.write("0.0 0.0 0.0 1.0 1.0 1.0\n")
            
        parser = DPTestResultParser(result_dir=str(result_dir))
        results = parser.parse()
        
        assert results["has_ground_truth"] is False, "Should detect NO ground truth when columns are zero"

    def test_parse_missing_energy(self, result_dir):
        """Test missing energy file."""
        parser = DPTestResultParser(str(result_dir))
        with pytest.raises(FileNotFoundError):
            parser.parse()

    def test_parse_no_force(self, result_dir, caplog):
        """Test missing force file (should warn, not fail)."""
        # Create energy file
        e_file = result_dir / "results.e_peratom.out"
        e_file.write_text("# sys:0:1\n1.0 1.1")
        
        parser = DPTestResultParser(str(result_dir))
        
        # Mock _get_dataname_info to avoid parsing empty file correctly
        with patch.object(parser, '_get_dataname_info', return_value=([], {})):
            with patch.object(parser, '_check_ground_truth'):
                data = parser.parse()
                
        assert data["force"] is None
        assert "Force file not found" in caplog.text

    def test_check_ground_truth(self, result_dir):
        """Test ground truth detection."""
        parser = DPTestResultParser(str(result_dir))
        
        # Case 1: All zeros -> No GT
        parser.data_e = np.zeros((10,), dtype=[('data_e', 'f4'), ('pred_e', 'f4')])
        parser.data_f = np.zeros((10,), dtype=[('data_fx', 'f4'), ('data_fy', 'f4'), ('data_fz', 'f4'), 
                                               ('pred_fx', 'f4'), ('pred_fy', 'f4'), ('pred_fz', 'f4')])
        
        parser._check_ground_truth()
        assert parser.has_ground_truth is False
        
        # Case 2: Non-zero -> Has GT
        parser.data_e['data_e'][0] = 1.0
        parser._check_ground_truth()
        assert parser.has_ground_truth is True

    def test_get_dataname_info(self, result_dir):
        """Test dataname extraction from comments."""
        # Create dummy file with comments
        content = """# data/sys1:0:2
1.0 1.1
1.0 1.1
# data/sys2:0:3
1.0 1.1
"""
        f_path = result_dir / "test.out"
        f_path.write_text(content)
        
        parser = DPTestResultParser(str(result_dir))
        # Mock _get_natom_from_name to return something
        with patch.object(parser, '_get_natom_from_name', side_effect=[2, 3]):
            lst, dct = parser._get_dataname_info(str(f_path))
            
        assert len(lst) == 3 # 2 frames for sys1 + 1 frame for sys2
        assert dct["data/sys1"] == 2
        assert dct["data/sys2"] == 1
        
        # Verify structure: [name, idx, natom]
        assert lst[0] == ["data/sys1", 0, 2]
        assert lst[1] == ["data/sys1", 1, 2]
        assert lst[2] == ["data/sys2", 0, 3]

    def test_get_natom_fallback(self, result_dir):
        """Test atom count estimation fallback."""
        parser = DPTestResultParser(str(result_dir), type_map=["H", "O"])
        
        # Invalid format
        n = parser._get_natom_from_name("InvalidName")
        assert n == 1 # Fallback
