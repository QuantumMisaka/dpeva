import pytest
import numpy as np
import os
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

def test_parser_basic(tmp_path):
    """
    Test parsing of basic energy and force files.
    Verifies that the parser correctly extracts system names and ground truth status.
    """
    create_dummy_files(tmp_path, n_frames=1, n_atoms=2)
    
    parser = DPTestResultParser(result_dir=str(tmp_path), head="results")
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

def test_parser_no_ground_truth(tmp_path):
    """
    Test parsing when data columns are zero.
    Should set has_ground_truth to False.
    """
    # Create zero-filled files
    e_file = os.path.join(tmp_path, "results.e_peratom.out")
    f_file = os.path.join(tmp_path, "results.f.out")
    
    with open(e_file, "w") as f:
        f.write("# sys1: 0\n")
        f.write("0.0 1.0\n")
        
    with open(f_file, "w") as f:
        # 1 atom
        f.write("0.0 0.0 0.0 1.0 1.0 1.0\n")
        
    parser = DPTestResultParser(result_dir=str(tmp_path))
    results = parser.parse()
    
    assert results["has_ground_truth"] is False, "Should detect NO ground truth when columns are zero"
