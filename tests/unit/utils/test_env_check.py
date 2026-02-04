import pytest
import warnings
from unittest.mock import patch, MagicMock
from dpeva.utils.env_check import check_deepmd_version

class TestEnvCheck:
    
    @patch("subprocess.check_output")
    def test_version_ok(self, mock_subprocess):
        # Case: Version is newer (3.2.0 > 3.1.2)
        mock_subprocess.return_value = "DeePMD-kit v3.2.0"
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_deepmd_version()
            assert len(w) == 0

    @patch("subprocess.check_output")
    def test_version_exact(self, mock_subprocess):
        # Case: Version is exact match (3.1.2 == 3.1.2)
        mock_subprocess.return_value = "DeePMD-kit v3.1.2"
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_deepmd_version()
            assert len(w) == 0

    @patch("subprocess.check_output")
    def test_version_old(self, mock_subprocess):
        # Case: Version is older (2.2.9 < 3.1.2)
        mock_subprocess.return_value = "DeePMD-kit v2.2.9"
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_deepmd_version()
            assert len(w) == 1
            assert "older than the recommended version" in str(w[0].message)

    @patch("subprocess.check_output")
    def test_dp_not_found(self, mock_subprocess):
        # Case: dp command not found
        mock_subprocess.side_effect = FileNotFoundError
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_deepmd_version()
            assert len(w) == 1
            assert "command not found" in str(w[0].message)

    @patch("subprocess.check_output")
    def test_parse_complex_version(self, mock_subprocess):
        # Case: Complex version string
        mock_subprocess.return_value = "DeePMD-kit 3.0.0-beta.1"
        # 3.0.0-beta.1 < 3.1.2
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_deepmd_version()
            assert len(w) == 1
            assert "older than" in str(w[0].message)

if __name__ == "__main__":
    pytest.main([__file__])
