import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from dpeva.workflows.analysis import AnalysisWorkflow

class TestAnalysisWorkflow:
    
    @pytest.fixture
    def config(self, tmp_path):
        return {
            "result_dir": str(tmp_path / "results"),
            "output_dir": str(tmp_path / "analysis"),
            "type_map": ["O", "H"],
            "ref_energies": {"O": -10.0, "H": -5.0}
        }

    @patch("dpeva.workflows.analysis.UnifiedAnalysisManager")
    @patch("dpeva.workflows.analysis.AnalysisIOManager")
    @patch("dpeva.workflows.analysis.setup_workflow_logger")
    @patch("dpeva.workflows.analysis.close_workflow_logger")
    def test_run_success(self, mock_close_logger, mock_setup_logger, MockIOManager, MockManager, config):
        """Test successful execution of analysis workflow."""
        # Setup mocks
        mock_io = MockIOManager.return_value
        mock_manager = MockManager.return_value
        
        # Mock load_data return
        mock_data = {"energy": {"pred_e": np.array([1.0])}}
        mock_parser = MagicMock()
        mock_parser.get_composition_list.return_value = (None, None)
        mock_io.load_data.return_value = (mock_data, mock_parser)
        
        # Mock compute_metrics return
        mock_stats_export = {"energy": {}}
        mock_metrics = {"e_mae": 0.1}
        mock_stats_calc = MagicMock()
        mock_e_rel_pred = np.array([-0.1])
        mock_e_rel_true = np.array([-0.11])
        mock_manager.analyze_model.return_value = (
            mock_stats_export, 
            mock_metrics, 
            mock_stats_calc, 
            mock_e_rel_pred, 
            mock_e_rel_true
        )
        
        # Initialize Workflow
        workflow = AnalysisWorkflow(config)
        
        # Run
        workflow.run()
        
        # Verify Interactions
        mock_setup_logger.assert_called_once()
        MockManager.assert_called_once_with(
            ref_energies=config["ref_energies"],
            enable_cohesive_energy=True,
            allow_ref_energy_lstsq_completion=False,
            slow_plot_threshold_seconds=60.0,
            enhanced_parity_renderer="auto",
        )
        mock_io.load_data.assert_called_with(config["result_dir"], config["type_map"], "results")
        
        assert mock_manager.analyze_model.call_args.kwargs["plot_level"] == "full"
        
        mock_io.save_metrics.assert_called_with(mock_metrics)
        mock_io.save_summary_csv.assert_called_with(mock_metrics)
        mock_io.save_stats_desc.assert_called() # cohesive stats
        
        mock_close_logger.assert_called_once()

    @patch("dpeva.workflows.analysis.UnifiedAnalysisManager")
    @patch("dpeva.workflows.analysis.AnalysisIOManager")
    @patch("dpeva.workflows.analysis.setup_workflow_logger")
    @patch("dpeva.workflows.analysis.close_workflow_logger")
    def test_run_logs_stage_and_total_elapsed(self, mock_close_logger, mock_setup_logger, MockIOManager, MockManager, config):
        mock_io = MockIOManager.return_value
        mock_manager = MockManager.return_value
        mock_parser = MagicMock()
        mock_parser.get_composition_list.return_value = (None, None)
        mock_io.load_data.return_value = ({"energy": {"pred_e": np.array([1.0])}}, mock_parser)
        mock_manager.analyze_model.return_value = ({}, {"e_mae": 0.1}, MagicMock(), np.array([-0.1]), np.array([-0.2]))
        workflow = AnalysisWorkflow(config)
        with patch.object(workflow.logger, "info") as mock_info:
            workflow.run()
        logged_messages = [call.args[0] for call in mock_info.call_args_list if call.args]
        assert any("Stage[parse] Start parsing results" in msg for msg in logged_messages)
        assert any("Stage[parse] Finished in" in msg for msg in logged_messages)
        assert any("Stage[composition] Start loading composition information" in msg for msg in logged_messages)
        assert any("Stage[composition] Finished in" in msg for msg in logged_messages)
        assert any("Stage[statistics+plot] Start statistics calculation and plotting" in msg for msg in logged_messages)
        assert any("Stage[statistics+plot] Finished in" in msg for msg in logged_messages)
        assert any("Analysis total elapsed time:" in msg for msg in logged_messages)
        mock_close_logger.assert_called_once()

    @patch("dpeva.workflows.analysis.AnalysisIOManager")
    @patch("dpeva.workflows.analysis.setup_workflow_logger")
    @patch("dpeva.workflows.analysis.close_workflow_logger")
    def test_run_failure(self, mock_close_logger, mock_setup_logger, MockIOManager, config):
        """Test failure handling."""
        mock_io = MockIOManager.return_value
        mock_io.load_data.side_effect = Exception("Parsing failed")
        
        workflow = AnalysisWorkflow(config)
        
        with pytest.raises(Exception, match="Parsing failed"):
            workflow.run()
            
        mock_close_logger.assert_called_once()

    @patch("dpeva.workflows.analysis.DatasetAnalysisManager")
    @patch("dpeva.workflows.analysis.AnalysisIOManager")
    @patch("dpeva.workflows.analysis.setup_workflow_logger")
    @patch("dpeva.workflows.analysis.close_workflow_logger")
    def test_run_dataset_mode(self, mock_close_logger, mock_setup_logger, MockIOManager, MockDatasetManager, tmp_path):
        config = {
            "mode": "dataset",
            "dataset_dir": str(tmp_path / "dataset"),
            "output_dir": str(tmp_path / "analysis"),
        }

        workflow = AnalysisWorkflow(config)
        workflow.run()

        mock_setup_logger.assert_called_once()
        MockIOManager.return_value.load_data.assert_not_called()
        MockDatasetManager.return_value.analyze.assert_called_once_with(
            workflow.config.dataset_dir,
            workflow.config.output_dir,
            plot_level="full",
        )
        mock_close_logger.assert_called_once()

    @patch("dpeva.workflows.analysis.DatasetAnalysisManager")
    @patch("dpeva.workflows.analysis.AnalysisIOManager")
    @patch("dpeva.workflows.analysis.setup_workflow_logger")
    @patch("dpeva.workflows.analysis.close_workflow_logger")
    def test_run_dataset_mode_with_optional_type_map(self, mock_close_logger, mock_setup_logger, MockIOManager, MockDatasetManager, tmp_path):
        config = {
            "mode": "dataset",
            "dataset_dir": str(tmp_path / "dataset"),
            "output_dir": str(tmp_path / "analysis"),
            "type_map": ["O", "H"],
            "plot_level": "basic",
        }
        workflow = AnalysisWorkflow(config)
        workflow.run()
        MockDatasetManager.return_value.analyze.assert_called_once_with(
            workflow.config.dataset_dir,
            workflow.config.output_dir,
            plot_level="basic",
        )
        mock_close_logger.assert_called_once()

    @patch("dpeva.workflows.analysis.UnifiedAnalysisManager")
    @patch("dpeva.workflows.analysis.AnalysisIOManager")
    @patch("dpeva.workflows.analysis.setup_workflow_logger")
    @patch("dpeva.workflows.analysis.close_workflow_logger")
    def test_run_uses_data_path_composition(self, mock_close_logger, mock_setup_logger, MockIOManager, MockManager, tmp_path):
        config = {
            "result_dir": str(tmp_path / "results"),
            "output_dir": str(tmp_path / "analysis"),
            "type_map": ["O", "H"],
            "data_path": str(tmp_path / "dpdata"),
            "ref_energies": {"O": -10.0, "H": -5.0},
        }
        mock_io = MockIOManager.return_value
        mock_manager = MockManager.return_value
        mock_parser = MagicMock()
        mock_io.load_data.return_value = ({"energy": {"pred_e": np.array([1.0])}}, mock_parser)
        mock_io.load_composition_info.return_value = ([{"O": 1, "H": 2}], [3])
        mock_manager.analyze_model.return_value = ({}, {"e_mae": 0.1}, MagicMock(), np.array([-0.1]), np.array([-0.2]))
        workflow = AnalysisWorkflow(config)
        workflow.run()
        mock_io.load_composition_info.assert_called_once_with(config["data_path"])
        mock_parser.get_composition_list.assert_not_called()
        mock_close_logger.assert_called_once()

    @patch("dpeva.workflows.analysis.UnifiedAnalysisManager")
    @patch("dpeva.workflows.analysis.AnalysisIOManager")
    @patch("dpeva.workflows.analysis.setup_workflow_logger")
    @patch("dpeva.workflows.analysis.close_workflow_logger")
    def test_run_passes_basic_plot_level(self, mock_close_logger, mock_setup_logger, MockIOManager, MockManager, tmp_path):
        config = {
            "result_dir": str(tmp_path / "results"),
            "output_dir": str(tmp_path / "analysis"),
            "type_map": ["O", "H"],
            "plot_level": "basic",
        }
        mock_io = MockIOManager.return_value
        mock_manager = MockManager.return_value
        mock_parser = MagicMock()
        mock_parser.get_composition_list.return_value = (None, None)
        mock_io.load_data.return_value = ({"energy": {"pred_e": np.array([1.0])}}, mock_parser)
        mock_manager.analyze_model.return_value = ({}, {"e_mae": 0.1}, MagicMock(), np.array([-0.1]), np.array([-0.2]))
        workflow = AnalysisWorkflow(config)
        workflow.run()
        assert mock_manager.analyze_model.call_args.kwargs["plot_level"] == "basic"
        mock_close_logger.assert_called_once()

    @patch("dpeva.workflows.analysis.UnifiedAnalysisManager")
    @patch("dpeva.workflows.analysis.AnalysisIOManager")
    @patch("dpeva.workflows.analysis.setup_workflow_logger")
    @patch("dpeva.workflows.analysis.close_workflow_logger")
    def test_run_logs_plot_level_scope(self, mock_close_logger, mock_setup_logger, MockIOManager, MockManager, tmp_path):
        config = {
            "result_dir": str(tmp_path / "results"),
            "output_dir": str(tmp_path / "analysis"),
            "type_map": ["O", "H"],
            "plot_level": "full",
        }
        mock_io = MockIOManager.return_value
        mock_manager = MockManager.return_value
        mock_parser = MagicMock()
        mock_parser.get_composition_list.return_value = (None, None)
        mock_io.load_data.return_value = ({"energy": {"pred_e": np.array([1.0])}}, mock_parser)
        mock_manager.analyze_model.return_value = ({}, {"e_mae": 0.1}, MagicMock(), np.array([-0.1]), np.array([-0.2]))
        workflow = AnalysisWorkflow(config)
        with patch.object(workflow.logger, "info") as mock_info:
            workflow.run()
        logged_messages = [call.args[0] for call in mock_info.call_args_list if call.args]
        assert any("Plot level 'full': full: basic + overlay、with_error、enhanced parity" in msg for msg in logged_messages)
        mock_close_logger.assert_called_once()

    @patch("dpeva.workflows.analysis.UnifiedAnalysisManager")
    @patch("dpeva.workflows.analysis.AnalysisIOManager")
    def test_init_passes_slow_plot_threshold_to_manager(self, MockIOManager, MockManager, tmp_path):
        config = {
            "result_dir": str(tmp_path / "results"),
            "output_dir": str(tmp_path / "analysis"),
            "type_map": ["O", "H"],
            "slow_plot_threshold_seconds": 12.5,
        }
        AnalysisWorkflow(config)
        MockIOManager.assert_called_once()
        MockManager.assert_called_once_with(
            ref_energies={},
            enable_cohesive_energy=True,
            allow_ref_energy_lstsq_completion=False,
            slow_plot_threshold_seconds=12.5,
            enhanced_parity_renderer="auto",
        )

    @patch("dpeva.workflows.analysis.UnifiedAnalysisManager")
    @patch("dpeva.workflows.analysis.AnalysisIOManager")
    def test_init_passes_enhanced_parity_renderer_to_manager(
        self, MockIOManager, MockManager, tmp_path
    ):
        config = {
            "result_dir": str(tmp_path / "results"),
            "output_dir": str(tmp_path / "analysis"),
            "type_map": ["O", "H"],
            "enhanced_parity_renderer": "scatter",
        }
        AnalysisWorkflow(config)
        MockIOManager.assert_called_once()
        MockManager.assert_called_once_with(
            ref_energies={},
            enable_cohesive_energy=True,
            allow_ref_energy_lstsq_completion=False,
            slow_plot_threshold_seconds=60.0,
            enhanced_parity_renderer="scatter",
        )

    @patch("dpeva.workflows.analysis.UnifiedAnalysisManager")
    @patch("dpeva.workflows.analysis.AnalysisIOManager")
    @patch("dpeva.workflows.analysis.setup_workflow_logger")
    @patch("dpeva.workflows.analysis.close_workflow_logger")
    def test_run_skips_metric_save_when_empty_metrics(self, mock_close_logger, mock_setup_logger, MockIOManager, MockManager, config):
        mock_io = MockIOManager.return_value
        mock_manager = MockManager.return_value
        mock_parser = MagicMock()
        mock_parser.get_composition_list.return_value = (None, None)
        mock_io.load_data.return_value = ({"energy": {"pred_e": np.array([1.0])}}, mock_parser)
        mock_manager.analyze_model.return_value = ({}, {}, MagicMock(), None, None)
        workflow = AnalysisWorkflow(config)
        workflow.run()
        mock_io.save_metrics.assert_not_called()
        mock_io.save_summary_csv.assert_not_called()
        mock_io.save_stats_desc.assert_not_called()
        mock_close_logger.assert_called_once()

    @patch("dpeva.workflows.analysis.JobManager")
    @patch("dpeva.workflows.analysis.setup_workflow_logger")
    @patch("dpeva.workflows.analysis.close_workflow_logger")
    def test_run_submits_to_slurm_when_backend_is_slurm(self, mock_close_logger, mock_setup_logger, MockJobManager, tmp_path):
        config = {
            "result_dir": str(tmp_path / "results"),
            "output_dir": str(tmp_path / "analysis"),
            "type_map": ["O", "H"],
            "submission": {
                "backend": "slurm",
                "slurm_config": {"partition": "cpu"}
            },
        }
        with patch.dict("os.environ", {}, clear=True):
            workflow = AnalysisWorkflow(config, config_path=str(tmp_path / "config.json"))
            workflow.run()
        MockJobManager.assert_called_once_with(mode="slurm")
        mock_job_manager = MockJobManager.return_value
        mock_job_manager.generate_script.assert_called_once()
        job_config = mock_job_manager.generate_script.call_args[0][0]
        assert "dpeva.cli analysis" in job_config.command
        assert str(tmp_path / "config.json") in job_config.command
        assert "export DPEVA_INTERNAL_BACKEND=local" in job_config.env_setup
        mock_job_manager.submit.assert_called_once()
        mock_setup_logger.assert_not_called()
        mock_close_logger.assert_not_called()

    @patch("dpeva.workflows.analysis.UnifiedAnalysisManager")
    @patch("dpeva.workflows.analysis.AnalysisIOManager")
    @patch("dpeva.workflows.analysis.setup_workflow_logger")
    @patch("dpeva.workflows.analysis.close_workflow_logger")
    def test_run_worker_mode_executes_local_even_if_config_is_slurm(self, mock_close_logger, mock_setup_logger, MockIOManager, MockManager, tmp_path):
        config = {
            "result_dir": str(tmp_path / "results"),
            "output_dir": str(tmp_path / "analysis"),
            "type_map": ["O", "H"],
            "submission": {"backend": "slurm"},
        }
        mock_io = MockIOManager.return_value
        mock_manager = MockManager.return_value
        mock_parser = MagicMock()
        mock_parser.get_composition_list.return_value = (None, None)
        mock_io.load_data.return_value = ({"energy": {"pred_e": np.array([1.0])}}, mock_parser)
        mock_manager.analyze_model.return_value = ({}, {"e_mae": 0.1}, MagicMock(), np.array([-0.1]), np.array([-0.2]))
        with patch.dict("os.environ", {"DPEVA_INTERNAL_BACKEND": "local"}):
            workflow = AnalysisWorkflow(config, config_path=str(tmp_path / "config.json"))
            workflow.run()
        mock_io.load_data.assert_called_once()
        mock_setup_logger.assert_called_once()
        mock_close_logger.assert_called_once()
