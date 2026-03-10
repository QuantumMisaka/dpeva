
import pytest
from unittest.mock import MagicMock, patch, call
from pathlib import Path
from dpeva.workflows.labeling import LabelingWorkflow
from dpeva.config import LabelingConfig

class TestLabelingWorkflow:
    
    @pytest.fixture
    def config(self, tmp_path):
        return LabelingConfig(
            work_dir=str(tmp_path),
            input_data_path=str(tmp_path / "data"),
            submission={"backend": "local"},
            dft_params={},
            attempt_params=[],
            pp_dir="/tmp/pp",
            orb_dir="/tmp/orb"
        )

    @patch("dpeva.workflows.labeling.load_systems")
    @patch("dpeva.workflows.labeling.LabelingManager")
    def test_wf_001_single_pool_routing(self, MockManager, mock_load, config, tmp_path):
        """
        WF-001: Test Single-Pool detection (Root is System).
        """
        # Setup
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "type.raw").touch()
        
        import dpdata
        mock_sys = MagicMock(spec=dpdata.System)
        mock_load.return_value = [mock_sys]
        
        # Act
        wf = LabelingWorkflow(config)
        wf.run()
        
        # Assert
        # Check prepare_tasks call
        manager = MockManager.return_value
        manager.prepare_tasks.assert_called_once()
        dataset_map = manager.prepare_tasks.call_args[0][0]
        
        assert "data" in dataset_map # Dataset name is dir name
        assert len(dataset_map) == 1

    @patch("dpeva.workflows.labeling.load_systems")
    @patch("dpeva.workflows.labeling.LabelingManager")
    def test_wf_002_multi_pool_routing(self, MockManager, mock_load, config, tmp_path):
        """
        WF-002: Test Multi-Pool detection (Root contains Datasets).
        """
        # Setup
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "DS1").mkdir()
        (data_dir / "DS2").mkdir()
        
        import dpdata
        mock_sys = MagicMock(spec=dpdata.System)
        mock_load.return_value = [mock_sys]
        
        # Act
        wf = LabelingWorkflow(config)
        wf.run()
        
        # Assert
        manager = MockManager.return_value
        dataset_map = manager.prepare_tasks.call_args[0][0]
        
        assert "DS1" in dataset_map
        assert "DS2" in dataset_map
        assert len(dataset_map) == 2

    @patch("dpeva.workflows.labeling.load_systems")
    def test_wf_003_empty_directory(self, mock_load, config, tmp_path):
        """
        WF-003: Test handling of empty input directory.
        """
        # Setup
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        # Act & Assert
        wf = LabelingWorkflow(config)
        with pytest.raises(ValueError, match="No valid systems found"):
            wf.run()

    @patch("dpeva.workflows.labeling.DataIntegrationManager")
    @patch("dpeva.workflows.labeling.load_systems")
    @patch("dpeva.workflows.labeling.LabelingManager")
    def test_wf_004_integration_enabled(self, MockManager, mock_load, MockIntegrationManager, tmp_path):
        config = LabelingConfig(
            work_dir=str(tmp_path),
            input_data_path=str(tmp_path / "data"),
            submission={"backend": "local"},
            dft_params={},
            attempt_params=[],
            pp_dir="/tmp/pp",
            orb_dir="/tmp/orb",
            integration_enabled=True,
        )
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "type.raw").touch()

        import dpdata
        mock_sys = MagicMock(spec=dpdata.System)
        mock_load.return_value = [mock_sys]

        manager = MockManager.return_value
        bundle_dir = tmp_path / "bundle"
        bundle_dir.mkdir()
        manager.prepare_tasks.return_value = [bundle_dir]

        wf = LabelingWorkflow(config)
        wf.run()

        MockIntegrationManager.return_value.integrate.assert_called_once()
