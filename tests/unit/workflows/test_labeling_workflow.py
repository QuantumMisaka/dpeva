
import pytest
from unittest.mock import MagicMock, patch
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
        manager = MockManager.return_value
        manager.prepare_tasks.return_value = []
        
        wf = LabelingWorkflow(config)
        wf.run()

        manager.prepare_tasks.assert_called_once()
        dataset_map = manager.prepare_tasks.call_args[0][0]

        assert "data" in dataset_map
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
        manager = MockManager.return_value
        manager.prepare_tasks.return_value = []
        
        wf = LabelingWorkflow(config)
        wf.run()
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

        MockIntegrationManager.assert_called_once_with(
            deduplicate=False,
            output_format=config.integration_output_format,
        )
        MockIntegrationManager.return_value.integrate.assert_called_once()

    @patch("dpeva.workflows.labeling.load_systems")
    @patch("dpeva.workflows.labeling.LabelingManager")
    def test_wf_stage_prepare(self, MockManager, mock_load, config, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "type.raw").touch()

        import dpdata
        mock_sys = MagicMock(spec=dpdata.System)
        mock_load.return_value = [mock_sys]

        manager = MockManager.return_value
        packed_dir = tmp_path / "inputs" / "N_10_0"
        packed_dir.mkdir(parents=True)
        manager.prepare_tasks.return_value = [packed_dir]

        wf = LabelingWorkflow(config)
        prepared = wf.run_prepare()

        assert prepared == [packed_dir]
        manager.prepare_tasks.assert_called_once()

    def test_wf_stage_execute_requires_prepare(self, config):
        wf = LabelingWorkflow(config)
        with pytest.raises(ValueError, match="Please run prepare stage first"):
            wf.run_execute()

    @patch("dpeva.workflows.labeling.DataIntegrationManager")
    @patch("dpeva.workflows.labeling.LabelingManager")
    def test_wf_stage_postprocess(self, MockManager, MockIntegrationManager, config):
        config.integration_enabled = True
        wf = LabelingWorkflow(config)
        wf.run_postprocess()

        manager = MockManager.return_value
        manager.collect_and_export.assert_called_once()
        MockIntegrationManager.assert_called_once_with(
            deduplicate=False,
            output_format=config.integration_output_format,
        )
        MockIntegrationManager.return_value.integrate.assert_called_once()

    @patch("dpeva.workflows.labeling.close_workflow_logger")
    @patch("dpeva.workflows.labeling.setup_workflow_logger")
    @patch("dpeva.workflows.labeling.load_systems")
    @patch("dpeva.workflows.labeling.LabelingManager")
    def test_wf_stage_prepare_creates_stage_log(self, MockManager, mock_load, mock_setup_logger, mock_close_logger, config, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "type.raw").touch()
        import dpdata
        mock_sys = MagicMock(spec=dpdata.System)
        mock_load.return_value = [mock_sys]
        MockManager.return_value.prepare_tasks.return_value = []

        wf = LabelingWorkflow(config)
        wf.run_prepare()

        mock_setup_logger.assert_called_once_with("dpeva", str(config.work_dir), "labeling_prepare.log", capture_stdout=True)
        mock_close_logger.assert_called_once_with("dpeva", str(tmp_path / "labeling_prepare.log"))

    @patch("dpeva.workflows.labeling.close_workflow_logger")
    @patch("dpeva.workflows.labeling.setup_workflow_logger")
    @patch("dpeva.workflows.labeling.LabelingManager")
    def test_wf_stage_execute_creates_stage_log(self, MockManager, mock_setup_logger, mock_close_logger, config, tmp_path):
        inputs_dir = tmp_path / "inputs" / "N_50_0"
        inputs_dir.mkdir(parents=True)
        (inputs_dir / "task_a").mkdir()
        MockManager.return_value.process_results.return_value = ([], [])

        wf = LabelingWorkflow(config)
        wf.run_execute()

        mock_setup_logger.assert_called_once_with("dpeva", str(config.work_dir), "labeling_execute.log", capture_stdout=True)
        mock_close_logger.assert_called_once_with("dpeva", str(tmp_path / "labeling_execute.log"))

    @patch("dpeva.workflows.labeling.close_workflow_logger")
    @patch("dpeva.workflows.labeling.setup_workflow_logger")
    @patch("dpeva.workflows.labeling.DataIntegrationManager")
    @patch("dpeva.workflows.labeling.LabelingManager")
    def test_wf_stage_postprocess_creates_stage_log(self, MockManager, MockIntegrationManager, mock_setup_logger, mock_close_logger, config, tmp_path):
        config.integration_enabled = True
        wf = LabelingWorkflow(config)
        wf.run_postprocess()

        mock_setup_logger.assert_called_once_with("dpeva", str(config.work_dir), "labeling_postprocess.log", capture_stdout=True)
        mock_close_logger.assert_called_once_with("dpeva", str(tmp_path / "labeling_postprocess.log"))
