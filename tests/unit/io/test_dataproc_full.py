
import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from dpeva.io.dataproc import DPTestResultParser

class TestDPTestResultParser:
    
    @pytest.fixture
    def result_dir(self, tmp_path):
        d = tmp_path / "results"
        d.mkdir()
        return d

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
