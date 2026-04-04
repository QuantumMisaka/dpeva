
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
    @patch("dpeva.workflows.labeling.LabelingManager")
    def test_wf_single_pool_dataset_routing(self, MockManager, mock_load, config, tmp_path):
        data_dir = tmp_path / "data"
        pool_dir = data_dir / "pool1"
        data_dir.mkdir()
        pool_dir.mkdir()
        (pool_dir / "type.raw").touch()

        import dpdata

        mock_sys = MagicMock(spec=dpdata.System)
        mock_load.return_value = [mock_sys]
        MockManager.return_value.prepare_tasks.return_value = []

        wf = LabelingWorkflow(config)
        wf.run()

        dataset_map = MockManager.return_value.prepare_tasks.call_args[0][0]
        assert list(dataset_map.keys()) == ["data"]

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

    def test_wf_load_dataset_map_raises_for_missing_input_path(self, config, tmp_path):
        wf = LabelingWorkflow(config)
        with pytest.raises(FileNotFoundError, match="Input path not found"):
            wf._load_dataset_map()

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
        (bundle_dir / "task_a").mkdir()
        manager.prepare_tasks.return_value = [bundle_dir]
        manager.process_results.return_value = ([], [])
        manager.extract_results.return_value = ([], [], [])

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

    @patch("dpeva.workflows.labeling.LabelingManager")
    def test_wf_stage_extract(self, MockManager, config, tmp_path):
        inputs_dir = tmp_path / "inputs" / "N_50_0"
        inputs_dir.mkdir(parents=True)
        (inputs_dir / "task_a").mkdir()
        MockManager.return_value.extract_results.return_value = ([], [], [])

        wf = LabelingWorkflow(config)
        wf.run_extract()

        MockManager.return_value.extract_results.assert_called_once()

    @patch("dpeva.workflows.labeling.LabelingManager")
    def test_wf_stage_execute_extract_postprocess_sequence(self, MockManager, config, tmp_path):
        inputs_dir = tmp_path / "inputs" / "N_50_0"
        inputs_dir.mkdir(parents=True)
        (inputs_dir / "task_a").mkdir()
        manager = MockManager.return_value
        manager.process_results.return_value = ([], [])
        manager.extract_results.return_value = ([], [], [])

        wf = LabelingWorkflow(config)
        wf.run_execute()
        wf.run_extract()
        wf.run_postprocess()

        manager.process_results.assert_called_once()
        manager.extract_results.assert_called_once()
        manager.collect_and_export.assert_called_once()

    @patch("dpeva.workflows.labeling.logger.warning")
    @patch("dpeva.workflows.labeling.LabelingManager")
    def test_wf_execute_warns_when_attempt_params_empty(self, MockManager, mock_warning, config, tmp_path):
        inputs_dir = tmp_path / "inputs" / "N_50_0"
        inputs_dir.mkdir(parents=True)
        (inputs_dir / "task_a").mkdir()
        config.attempt_params = []
        MockManager.return_value.process_results.return_value = ([], [])

        wf = LabelingWorkflow(config)
        wf.run_execute()

        assert any("No attempt_params defined" in call.args[0] for call in mock_warning.call_args_list)

    @patch("dpeva.workflows.labeling.LabelingManager")
    def test_collect_active_job_dirs_only_returns_non_empty_bundles(self, MockManager, config, tmp_path):
        active = tmp_path / "inputs" / "N_10_0"
        inactive = tmp_path / "inputs" / "N_20_0"
        active.mkdir(parents=True)
        inactive.mkdir(parents=True)
        (active / "task_a").mkdir()

        result = LabelingWorkflow._collect_active_job_dirs([active, inactive])

        assert result == [active]

    @patch("dpeva.workflows.labeling.logger.error")
    @patch("dpeva.workflows.labeling.LabelingManager")
    def test_submit_job_dirs_continues_after_partial_failure(self, MockManager, mock_error, config, tmp_path):
        job_a = tmp_path / "N_10_0"
        job_b = tmp_path / "N_20_0"
        job_a.mkdir()
        job_b.mkdir()
        manager = MockManager.return_value
        manager.generate_runner_script.side_effect = ["print('a')", "print('b')"]

        wf = LabelingWorkflow(config)
        wf.job_manager.submit_python_script = MagicMock(side_effect=["123", RuntimeError("submit failed")])

        job_ids = wf._submit_job_dirs([job_a, job_b], attempt=0)

        assert job_ids == ["123"]
        mock_error.assert_called_once()

    @patch("dpeva.workflows.labeling.time.sleep")
    @patch("dpeva.workflows.labeling.subprocess.run", side_effect=RuntimeError("squeue down"))
    @patch("dpeva.workflows.labeling.logger.error")
    def test_monitor_slurm_jobs_logs_query_failure_and_continues(self, mock_error, mock_run, mock_sleep, config):
        wf = LabelingWorkflow(config)
        with patch.object(wf, "_monitor_slurm_jobs", wraps=wf._monitor_slurm_jobs) as wrapped:
            # Stop after one failure cycle by raising from sleep
            mock_sleep.side_effect = RuntimeError("stop-loop")
            with pytest.raises(RuntimeError, match="stop-loop"):
                wrapped(["job-123"], interval=1)

        mock_error.assert_called_once()

    @patch("dpeva.workflows.labeling.logger.info")
    @patch("dpeva.workflows.labeling.LabelingManager")
    def test_run_execute_short_circuits_when_all_tasks_converged(self, MockManager, mock_info, config, tmp_path):
        inputs_dir = tmp_path / "inputs" / "N_50_0"
        inputs_dir.mkdir(parents=True)
        # no subdirs -> no active jobs
        wf = LabelingWorkflow(config)
        wf.run_execute()

        assert any("All tasks converged." in call.args[0] for call in mock_info.call_args_list)

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

    @patch("dpeva.workflows.labeling.close_workflow_logger")
    @patch("dpeva.workflows.labeling.setup_workflow_logger")
    @patch("dpeva.workflows.labeling.LabelingManager")
    def test_wf_stage_extract_creates_stage_log(self, MockManager, mock_setup_logger, mock_close_logger, config, tmp_path):
        inputs_dir = tmp_path / "inputs" / "N_50_0"
        inputs_dir.mkdir(parents=True)
        (inputs_dir / "task_a").mkdir()
        MockManager.return_value.extract_results.return_value = ([], [], [])

        wf = LabelingWorkflow(config)
        wf.run_extract()

        mock_setup_logger.assert_called_once_with("dpeva", str(config.work_dir), "labeling_extract.log", capture_stdout=True)
        mock_close_logger.assert_called_once_with("dpeva", str(tmp_path / "labeling_extract.log"))
