
import pytest
import os
from dpeva.io.dataproc import DPTestResultParser

def create_dummy_files(dirname, head="results", systems=None):
    """
    Creates dummy output files for testing.
    
    Args:
        dirname (str): Output directory.
        head (str): File prefix.
        systems (list): List of dicts [{"name": "sys1", "n_frames": 2, "n_atoms": 3}, ...]
    """
    if systems is None:
        systems = [{"name": "pool/sys1", "n_frames": 1, "n_atoms": 2},
                   {"name": "pool/sys2", "n_frames": 1, "n_atoms": 2}]
                   
    e_file = os.path.join(dirname, f"{head}.e_peratom.out")
    f_file = os.path.join(dirname, f"{head}.f.out")
    
    current_e_line = 0
    
    with open(e_file, "w") as fe, open(f_file, "w") as ff:
        for sys in systems:
            name = sys["name"]
            n_frames = sys["n_frames"]
            n_atoms = sys["n_atoms"]
            
            # Write Header
            # Using current_e_line as index
            fe.write(f"# {name}: {current_e_line}\n")
            ff.write(f"# {name}: 0\n") # Force file index usually restarts or follows? 
            # In deepmd, force file index is also line number. 
            # But here we just need headers to mark sections.
            
            # Write Data
            for _ in range(n_frames):
                fe.write("1.0 1.1\n") # 1 line per frame
                current_e_line += 1
                
                for _ in range(n_atoms):
                    ff.write("0.1 0.2 0.3 0.11 0.21 0.31\n") # 1 line per atom

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
        systems = [
            {"name": "pool/sys1", "n_frames": 2, "n_atoms": 3},
            {"name": "pool/sys2", "n_frames": 1, "n_atoms": 5}
        ]
        create_dummy_files(str(result_dir), head="results", systems=systems)
        
        parser = DPTestResultParser(result_dir=str(result_dir), head="results")
        results = parser.parse()
        
        assert results["energy"] is not None
        assert results["force"] is not None
        
        # Check dataname list
        # Sys1: 2 frames, natom 3
        # Sys2: 1 frame, natom 5
        d_list = results["dataname_list"]
        assert len(d_list) == 3 # 2+1 frames
        
        # Frame 0 of sys1
        assert d_list[0][0] == "pool/sys1"
        assert d_list[0][2] == 3 # natom
        
        # Frame 1 of sys1
        assert d_list[1][0] == "pool/sys1"
        assert d_list[1][2] == 3 # natom
        
        # Frame 0 of sys2
        assert d_list[2][0] == "pool/sys2"
        assert d_list[2][2] == 5 # natom

    def test_structure_based_atom_parsing(self, result_dir):
        """
        Test robust atom count calculation based on line ratios.
        This verifies the fix for "natom=1" fallback issue.
        """
        # Create a system with a name that would fail regex parsing (no numbers)
        systems = [
            {"name": "pure_iron_liquid", "n_frames": 10, "n_atoms": 128}
        ]
        create_dummy_files(str(result_dir), head="results", systems=systems)
        
        parser = DPTestResultParser(str(result_dir))
        results = parser.parse()
        
        # Check if natom is correctly calculated as 128
        assert len(results["dataname_list"]) == 10
        for item in results["dataname_list"]:
            assert item[0] == "pure_iron_liquid"
            assert item[2] == 128, f"Expected 128 atoms, got {item[2]}"

    def test_parser_no_ground_truth(self, result_dir):
        """Test parsing when data columns are zero."""
        e_file = result_dir / "results.e_peratom.out"
        f_file = result_dir / "results.f.out"
        
        with open(e_file, "w") as f:
            f.write("# sys1: 0\n")
            f.write("0.0 1.0\n")
            
        with open(f_file, "w") as f:
            f.write("# sys1: 0\n")
            f.write("0.0 0.0 0.0 1.0 1.0 1.0\n") # 1 atom
            
        parser = DPTestResultParser(result_dir=str(result_dir))
        results = parser.parse()
        
        assert results["has_ground_truth"] is False

    def test_parser_no_ground_truth_when_any_energy_frame_is_zero(self, result_dir):
        e_file = result_dir / "results.e_peratom.out"
        f_file = result_dir / "results.f.out"

        with open(e_file, "w") as f:
            f.write("# sys1: 0\n")
            f.write("0.0 1.0\n")
            f.write("0.8 1.0\n")

        with open(f_file, "w") as f:
            f.write("# sys1: 0\n")
            f.write("0.2 0.2 0.2 0.3 0.3 0.3\n")
            f.write("0.4 0.4 0.4 0.5 0.5 0.5\n")

        parser = DPTestResultParser(result_dir=str(result_dir))
        results = parser.parse()

        assert results["has_ground_truth"] is False

    def test_parser_keeps_strict_less_than_threshold_for_zero(self, result_dir):
        e_file = result_dir / "results.e_peratom.out"
        f_file = result_dir / "results.f.out"

        with open(e_file, "w") as f:
            f.write("# sys1: 0\n")
            f.write("0.0001 1.0\n")

        with open(f_file, "w") as f:
            f.write("# sys1: 0\n")
            f.write("0.1 0.2 0.3 0.2 0.3 0.4\n")

        parser = DPTestResultParser(result_dir=str(result_dir))
        results = parser.parse()

        assert results["has_ground_truth"] is True

    def test_parser_keeps_ground_truth_when_only_part_force_is_near_zero(self, result_dir):
        e_file = result_dir / "results.e_peratom.out"
        f_file = result_dir / "results.f.out"

        with open(e_file, "w") as f:
            f.write("# sys1: 0\n")
            f.write("0.8 1.0\n")
            f.write("0.9 1.0\n")

        with open(f_file, "w") as f:
            f.write("# sys1: 0\n")
            f.write("0.0 0.0 0.0 0.2 0.2 0.2\n")
            f.write("0.3 0.4 0.5 0.2 0.3 0.4\n")

        parser = DPTestResultParser(result_dir=str(result_dir))
        results = parser.parse()

        assert results["has_ground_truth"] is True

    def test_parser_no_ground_truth_when_data_equals_prediction(self, result_dir):
        e_file = result_dir / "results.e_peratom.out"
        f_file = result_dir / "results.f.out"

        with open(e_file, "w") as f:
            f.write("# sys1: 0\n")
            f.write("1.23 1.23\n")

        with open(f_file, "w") as f:
            f.write("# sys1: 0\n")
            f.write("0.11 0.22 0.33 0.11 0.22 0.33\n")

        parser = DPTestResultParser(result_dir=str(result_dir))
        results = parser.parse()
        assert results["has_ground_truth"] is False

    def test_parse_missing_energy(self, result_dir):
        """Test missing energy file."""
        parser = DPTestResultParser(str(result_dir))
        with pytest.raises(FileNotFoundError):
            parser.parse()

    def test_parse_no_force(self, result_dir, caplog):
        """Test missing force file (should warn, not fail, fallback natom=1)."""
        e_file = result_dir / "results.e_peratom.out"
        e_file.write_text("# sys:0\n1.0 1.1")
        
        parser = DPTestResultParser(str(result_dir))
        
        # We need to mock parse_indices inside _get_dataname_info or just let it run
        # Since force file is missing, f_indices will be empty
        # Logic should handle it gracefully
        
        data = parser.parse()
                
        assert data["force"] is None
        assert "Force file not found" in caplog.text
        # Fallback natom should be 1
        assert data["dataname_list"][0][2] == 1

    def test_get_dataname_info_logic(self, result_dir):
        """Directly test _get_dataname_info method."""
        e_file = result_dir / "test.e"
        f_file = result_dir / "test.f"
        
        e_file.write_text("# sys1:0\n1.0 1.0\n1.0 1.0\n# sys2:2\n1.0 1.0\n") # 2 frames sys1, 1 frame sys2
        
        # Sys1: 2 frames, 4 atoms -> 8 force lines
        # Sys2: 1 frame, 2 atoms -> 2 force lines
        f_content = "# sys1:0\n" + ("0.0 0.0 0.0 0.0 0.0 0.0\n" * 8) + \
                    "# sys2:8\n" + ("0.0 0.0 0.0 0.0 0.0 0.0\n" * 2)
        f_file.write_text(f_content)
        
        parser = DPTestResultParser(str(result_dir))
        lst, dct = parser._get_dataname_info(str(e_file), str(f_file))
        
        assert dct["sys1"] == 2
        assert dct["sys2"] == 1
        
        # Verify atoms
        # sys1
        assert lst[0][2] == 4
        assert lst[1][2] == 4
        # sys2
        assert lst[2][2] == 2
