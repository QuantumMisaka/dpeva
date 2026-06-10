import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from dpeva.constants import WORKFLOW_FINISHED_TAG
from dpeva.workflows.analysis import AnalysisWorkflow, _set_config_path_if_missing


class _FrozenConfig:
    def __init__(self):
        self._config_path = None

    @property
    def config_path(self):
        return self._config_path

    @config_path.setter
    def config_path(self, _value):
        raise TypeError("frozen")


@pytest.fixture
def config(tmp_path):
    return {
        "result_dir": str(tmp_path / "results"),
        "output_dir": str(tmp_path / "analysis"),
        "type_map": ["O", "H"],
        "ref_energies": {"O": -10.0, "H": -5.0},
    }


def test_set_config_path_if_missing_logs_warning_for_frozen_config(tmp_path):
    config = _FrozenConfig()
    logger = MagicMock()
    _set_config_path_if_missing(config, str(tmp_path / "config.json"), logger)
    logger.warning.assert_called_once()
    assert config.config_path is None


@patch("dpeva.workflows.analysis.UnifiedAnalysisManager")
@patch("dpeva.workflows.analysis.AnalysisIOManager")
@patch("dpeva.workflows.analysis.setup_workflow_logger")
@patch("dpeva.workflows.analysis.close_workflow_logger")
def test_run_success(
    mock_close_logger,
    mock_setup_logger,
    mock_io_manager_cls,
    mock_manager_cls,
    config,
):
    mock_io = mock_io_manager_cls.return_value
    mock_manager = mock_manager_cls.return_value
    mock_parser = MagicMock()
    mock_parser.get_composition_list.return_value = (None, None)
    mock_io.load_data.return_value = ({"energy": {"pred_e": np.array([1.0])}}, mock_parser)
    mock_metrics = {"e_mae": 0.1}
    mock_manager.analyze_model.return_value = (
        {"energy": {}},
        mock_metrics,
        MagicMock(),
        np.array([-0.1]),
        np.array([-0.11]),
    )

    workflow = AnalysisWorkflow(config)
    workflow.run()

    mock_setup_logger.assert_called_once()
    mock_manager_cls.assert_called_once_with(
        ref_energies=config["ref_energies"],
        enable_cohesive_energy=True,
        allow_ref_energy_lstsq_completion=False,
        slow_plot_threshold_seconds=60.0,
        enhanced_parity_renderer="auto",
    )
    mock_io.load_data.assert_called_with(config["result_dir"], config["type_map"], "results")
    assert mock_manager.analyze_model.call_args.kwargs["plot_level"] == "full"
    mock_io.save_metrics.assert_called_once_with(mock_metrics)
    mock_io.save_summary_csv.assert_called_once_with(mock_metrics)
    mock_io.save_stats_desc.assert_called_once()
    mock_close_logger.assert_called_once()


@patch("dpeva.workflows.analysis.UnifiedAnalysisManager")
@patch("dpeva.workflows.analysis.AnalysisIOManager")
@patch("dpeva.workflows.analysis.setup_workflow_logger")
@patch("dpeva.workflows.analysis.close_workflow_logger")
def test_run_logs_stage_total_elapsed_and_completion_marker(
    mock_close_logger,
    mock_setup_logger,
    mock_io_manager_cls,
    mock_manager_cls,
    config,
):
    mock_io = mock_io_manager_cls.return_value
    mock_manager = mock_manager_cls.return_value
    mock_parser = MagicMock()
    mock_parser.get_composition_list.return_value = (None, None)
    mock_io.load_data.return_value = ({"energy": {"pred_e": np.array([1.0])}}, mock_parser)
    mock_manager.analyze_model.return_value = (
        {},
        {"e_mae": 0.1},
        MagicMock(),
        np.array([-0.1]),
        np.array([-0.2]),
    )

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
    assert WORKFLOW_FINISHED_TAG in logged_messages
    mock_close_logger.assert_called_once()


@patch("dpeva.workflows.analysis.AnalysisIOManager")
@patch("dpeva.workflows.analysis.setup_workflow_logger")
@patch("dpeva.workflows.analysis.close_workflow_logger")
def test_run_failure(mock_close_logger, mock_setup_logger, mock_io_manager_cls, config):
    mock_io = mock_io_manager_cls.return_value
    mock_io.load_data.side_effect = Exception("Parsing failed")

    workflow = AnalysisWorkflow(config)

    with pytest.raises(Exception, match="Parsing failed"):
        workflow.run()

    mock_close_logger.assert_called_once()


@patch("dpeva.workflows.analysis.UnifiedAnalysisManager")
@patch("dpeva.workflows.analysis.DatasetAnalysisManager")
@patch("dpeva.workflows.analysis.AnalysisIOManager")
@patch("dpeva.workflows.analysis.setup_workflow_logger")
@patch("dpeva.workflows.analysis.close_workflow_logger")
def test_run_dataset_mode(
    mock_close_logger,
    mock_setup_logger,
    mock_io_manager_cls,
    mock_dataset_manager_cls,
    mock_manager_cls,
    tmp_path,
):
    config = {
        "mode": "dataset",
        "dataset_dir": str(tmp_path / "dataset"),
        "output_dir": str(tmp_path / "analysis"),
    }

    workflow = AnalysisWorkflow(config)
    workflow.run()

    mock_setup_logger.assert_called_once()
    mock_io_manager_cls.return_value.load_data.assert_not_called()
    mock_dataset_manager_cls.return_value.analyze.assert_called_once_with(
        workflow.config.dataset_dir,
        workflow.config.output_dir,
        plot_level="full",
    )
    mock_close_logger.assert_called_once()


@patch("dpeva.workflows.analysis.UnifiedAnalysisManager")
@patch("dpeva.workflows.analysis.DatasetAnalysisManager")
@patch("dpeva.workflows.analysis.AnalysisIOManager")
@patch("dpeva.workflows.analysis.setup_workflow_logger")
@patch("dpeva.workflows.analysis.close_workflow_logger")
def test_run_dataset_mode_with_optional_type_map(
    mock_close_logger,
    mock_setup_logger,
    mock_io_manager_cls,
    mock_dataset_manager_cls,
    mock_manager_cls,
    tmp_path,
):
    config = {
        "mode": "dataset",
        "dataset_dir": str(tmp_path / "dataset"),
        "output_dir": str(tmp_path / "analysis"),
        "type_map": ["O", "H"],
        "plot_level": "basic",
    }

    workflow = AnalysisWorkflow(config)
    workflow.run()

    mock_dataset_manager_cls.return_value.analyze.assert_called_once_with(
        workflow.config.dataset_dir,
        workflow.config.output_dir,
        plot_level="basic",
    )
    mock_close_logger.assert_called_once()


@patch("dpeva.workflows.analysis.UnifiedAnalysisManager")
@patch("dpeva.workflows.analysis.AnalysisIOManager")
@patch("dpeva.workflows.analysis.setup_workflow_logger")
@patch("dpeva.workflows.analysis.close_workflow_logger")
def test_run_prefers_parser_aligned_composition_when_data_path_order_mismatches(
    mock_close_logger,
    mock_setup_logger,
    mock_io_manager_cls,
    mock_manager_cls,
    tmp_path,
):
    config = {
        "result_dir": str(tmp_path / "results"),
        "output_dir": str(tmp_path / "analysis"),
        "type_map": ["O", "H"],
        "data_path": str(tmp_path / "dpdata"),
        "ref_energies": {"O": -10.0, "H": -5.0},
    }
    mock_io = mock_io_manager_cls.return_value
    mock_manager = mock_manager_cls.return_value
    mock_parser = MagicMock()
    mock_io.load_data.return_value = ({"energy": {"pred_e": np.array([1.0])}}, mock_parser)
    mock_parser.get_composition_list.return_value = ([{"O": 1, "H": 2}], [3])
    mock_io.load_composition_info.return_value = ([{"O": 2, "H": 1}], [3])
    mock_manager.analyze_model.return_value = (
        {},
        {"e_mae": 0.1},
        MagicMock(),
        np.array([-0.1]),
        np.array([-0.2]),
    )

    workflow = AnalysisWorkflow(config)
    workflow.run()

    mock_io.load_composition_info.assert_called_once_with(config["data_path"])
    assert mock_manager.analyze_model.call_args.kwargs["atom_counts_list"] == [{"O": 1, "H": 2}]
    assert mock_manager.analyze_model.call_args.kwargs["atom_num_list"] == [3]
    mock_close_logger.assert_called_once()


@patch("dpeva.workflows.analysis.UnifiedAnalysisManager")
@patch("dpeva.workflows.analysis.AnalysisIOManager")
@patch("dpeva.workflows.analysis.setup_workflow_logger")
@patch("dpeva.workflows.analysis.close_workflow_logger")
def test_run_falls_back_to_data_path_composition_when_parser_counts_invalid(
    mock_close_logger,
    mock_setup_logger,
    mock_io_manager_cls,
    mock_manager_cls,
    tmp_path,
):
    config = {
        "result_dir": str(tmp_path / "results"),
        "output_dir": str(tmp_path / "analysis"),
        "type_map": ["O", "H"],
        "data_path": str(tmp_path / "dpdata"),
        "ref_energies": {"O": -10.0, "H": -5.0},
    }
    mock_io = mock_io_manager_cls.return_value
    mock_manager = mock_manager_cls.return_value
    mock_parser = MagicMock()
    mock_io.load_data.return_value = ({"energy": {"pred_e": np.array([1.0])}}, mock_parser)
    mock_parser.get_composition_list.return_value = ([{}], [0])
    mock_io.load_composition_info.return_value = ([{"O": 1, "H": 2}], [3])
    mock_manager.analyze_model.return_value = (
        {},
        {"e_mae": 0.1},
        MagicMock(),
        np.array([-0.1]),
        np.array([-0.2]),
    )

    workflow = AnalysisWorkflow(config)
    workflow.run()

    assert mock_manager.analyze_model.call_args.kwargs["atom_counts_list"] == [{"O": 1, "H": 2}]
    assert mock_manager.analyze_model.call_args.kwargs["atom_num_list"] == [3]
    mock_close_logger.assert_called_once()


@patch("dpeva.workflows.analysis.UnifiedAnalysisManager")
@patch("dpeva.workflows.analysis.AnalysisIOManager")
@patch("dpeva.workflows.analysis.setup_workflow_logger")
@patch("dpeva.workflows.analysis.close_workflow_logger")
def test_run_passes_basic_plot_level(
    mock_close_logger,
    mock_setup_logger,
    mock_io_manager_cls,
    mock_manager_cls,
    tmp_path,
):
    config = {
        "result_dir": str(tmp_path / "results"),
        "output_dir": str(tmp_path / "analysis"),
        "type_map": ["O", "H"],
        "plot_level": "basic",
    }
    mock_io = mock_io_manager_cls.return_value
    mock_manager = mock_manager_cls.return_value
    mock_parser = MagicMock()
    mock_parser.get_composition_list.return_value = (None, None)
    mock_io.load_data.return_value = ({"energy": {"pred_e": np.array([1.0])}}, mock_parser)
    mock_manager.analyze_model.return_value = (
        {},
        {"e_mae": 0.1},
        MagicMock(),
        np.array([-0.1]),
        np.array([-0.2]),
    )

    workflow = AnalysisWorkflow(config)
    workflow.run()

    assert mock_manager.analyze_model.call_args.kwargs["plot_level"] == "basic"
    mock_close_logger.assert_called_once()


@patch("dpeva.workflows.analysis.UnifiedAnalysisManager")
@patch("dpeva.workflows.analysis.AnalysisIOManager")
@patch("dpeva.workflows.analysis.setup_workflow_logger")
@patch("dpeva.workflows.analysis.close_workflow_logger")
def test_run_logs_plot_level_scope(
    mock_close_logger,
    mock_setup_logger,
    mock_io_manager_cls,
    mock_manager_cls,
    tmp_path,
):
    config = {
        "result_dir": str(tmp_path / "results"),
        "output_dir": str(tmp_path / "analysis"),
        "type_map": ["O", "H"],
        "plot_level": "full",
    }
    mock_io = mock_io_manager_cls.return_value
    mock_manager = mock_manager_cls.return_value
    mock_parser = MagicMock()
    mock_parser.get_composition_list.return_value = (None, None)
    mock_io.load_data.return_value = ({"energy": {"pred_e": np.array([1.0])}}, mock_parser)
    mock_manager.analyze_model.return_value = (
        {},
        {"e_mae": 0.1},
        MagicMock(),
        np.array([-0.1]),
        np.array([-0.2]),
    )

    workflow = AnalysisWorkflow(config)
    with patch.object(workflow.logger, "info") as mock_info:
        workflow.run()

    logged_messages = [call.args[0] for call in mock_info.call_args_list if call.args]
    assert any(
        "Plot level 'full': full: basic + overlay、with_error、enhanced parity" in msg
        for msg in logged_messages
    )
    mock_close_logger.assert_called_once()


@patch("dpeva.workflows.analysis.UnifiedAnalysisManager")
@patch("dpeva.workflows.analysis.AnalysisIOManager")
def test_init_passes_slow_plot_threshold_to_manager(
    mock_io_manager_cls,
    mock_manager_cls,
    tmp_path,
):
    config = {
        "result_dir": str(tmp_path / "results"),
        "output_dir": str(tmp_path / "analysis"),
        "type_map": ["O", "H"],
        "slow_plot_threshold_seconds": 12.5,
    }

    AnalysisWorkflow(config)

    mock_io_manager_cls.assert_called_once()
    mock_manager_cls.assert_called_once_with(
        ref_energies={},
        enable_cohesive_energy=True,
        allow_ref_energy_lstsq_completion=False,
        slow_plot_threshold_seconds=12.5,
        enhanced_parity_renderer="auto",
    )


@patch("dpeva.workflows.analysis.UnifiedAnalysisManager")
@patch("dpeva.workflows.analysis.AnalysisIOManager")
def test_init_passes_enhanced_parity_renderer_to_manager(
    mock_io_manager_cls,
    mock_manager_cls,
    tmp_path,
):
    config = {
        "result_dir": str(tmp_path / "results"),
        "output_dir": str(tmp_path / "analysis"),
        "type_map": ["O", "H"],
        "enhanced_parity_renderer": "scatter",
    }

    AnalysisWorkflow(config)

    mock_io_manager_cls.assert_called_once()
    mock_manager_cls.assert_called_once_with(
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
def test_run_skips_metric_save_when_empty_metrics(
    mock_close_logger,
    mock_setup_logger,
    mock_io_manager_cls,
    mock_manager_cls,
    config,
):
    mock_io = mock_io_manager_cls.return_value
    mock_manager = mock_manager_cls.return_value
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
def test_run_submits_to_slurm_when_backend_is_slurm(
    mock_close_logger,
    mock_setup_logger,
    mock_job_manager_cls,
    tmp_path,
):
    config = {
        "result_dir": str(tmp_path / "results"),
        "output_dir": str(tmp_path / "analysis"),
        "type_map": ["O", "H"],
        "submission": {
            "backend": "slurm",
            "slurm_config": {"partition": "cpu"},
        },
    }

    with patch.dict("os.environ", {}, clear=True):
        workflow = AnalysisWorkflow(config, config_path=str(tmp_path / "config.json"))
        workflow.run()

    mock_job_manager_cls.assert_called_once_with(mode="slurm")
    mock_job_manager = mock_job_manager_cls.return_value
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
def test_run_worker_mode_executes_local_even_if_config_is_slurm(
    mock_close_logger,
    mock_setup_logger,
    mock_io_manager_cls,
    mock_manager_cls,
    tmp_path,
):
    config = {
        "result_dir": str(tmp_path / "results"),
        "output_dir": str(tmp_path / "analysis"),
        "type_map": ["O", "H"],
        "submission": {"backend": "slurm"},
    }
    mock_io = mock_io_manager_cls.return_value
    mock_manager = mock_manager_cls.return_value
    mock_parser = MagicMock()
    mock_parser.get_composition_list.return_value = (None, None)
    mock_io.load_data.return_value = ({"energy": {"pred_e": np.array([1.0])}}, mock_parser)
    mock_manager.analyze_model.return_value = (
        {},
        {"e_mae": 0.1},
        MagicMock(),
        np.array([-0.1]),
        np.array([-0.2]),
    )

    with patch.dict("os.environ", {"DPEVA_INTERNAL_BACKEND": "local"}):
        workflow = AnalysisWorkflow(config, config_path=str(tmp_path / "config.json"))
        workflow.run()

    mock_io.load_data.assert_called_once()
    mock_setup_logger.assert_called_once()
    mock_close_logger.assert_called_once()
