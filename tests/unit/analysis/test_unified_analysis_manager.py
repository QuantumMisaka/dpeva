from unittest.mock import MagicMock, patch

import numpy as np

from dpeva.analysis.managers import UnifiedAnalysisManager


def test_analyze_model_warns_when_plot_is_slow_with_threshold_and_basic_hint(tmp_path):
    data = {
        "energy": {"pred_e": np.array([1.0])},
        "force": None,
        "virial": None,
        "has_ground_truth": False,
    }
    manager = UnifiedAnalysisManager(
        enable_cohesive_energy=False,
        slow_plot_threshold_seconds=1.0,
    )

    mock_stats_calc = MagicMock()
    mock_stats_calc.e_true = None
    mock_stats_calc.e_pred = np.array([1.0])
    mock_stats_calc.f_true = None
    mock_stats_calc.f_pred = None
    mock_stats_calc.v_true = None
    mock_stats_calc.v_pred = None
    mock_stats_calc.compute_relative_energy.return_value = None
    mock_stats_calc.get_distribution_stats.return_value = {"count": 1}

    with patch("dpeva.analysis.managers.StatsCalculator", return_value=mock_stats_calc), patch(
        "dpeva.analysis.managers.InferenceVisualizer"
    ) as MockVisualizer, patch(
        "dpeva.analysis.managers.time.perf_counter", side_effect=[0.0, 1.5]
    ), patch.object(
        manager.logger, "warning"
    ) as mock_warning:
        manager.analyze_model(data=data, output_dir=str(tmp_path), plot_level="basic")

    MockVisualizer.return_value.plot_distribution.assert_called_once()
    warning_messages = [call.args[0] for call in mock_warning.call_args_list]
    assert any("dist_predicted_energy" in msg for msg in warning_messages)
    assert any("elapsed=1.500s" in msg for msg in warning_messages)
    assert any("threshold=1.000s" in msg for msg in warning_messages)
    assert any("plot_level='basic'" in msg for msg in warning_messages)


def test_analyze_model_downgrades_plot_failures_to_warning_and_keeps_metrics(tmp_path):
    data = {
        "energy": {"pred_e": np.array([1.0]), "data_e": np.array([1.2])},
        "force": None,
        "virial": None,
        "has_ground_truth": True,
    }
    manager = UnifiedAnalysisManager(enable_cohesive_energy=False, slow_plot_threshold_seconds=60.0)

    mock_stats_calc = MagicMock()
    mock_stats_calc.e_true = np.array([1.2])
    mock_stats_calc.e_pred = np.array([1.0])
    mock_stats_calc.f_true = None
    mock_stats_calc.f_pred = None
    mock_stats_calc.v_true = None
    mock_stats_calc.v_pred = None
    mock_stats_calc.compute_metrics.return_value = {"e_mae": 0.2}
    mock_stats_calc.compute_relative_energy.return_value = None
    mock_stats_calc.get_distribution_stats.return_value = {"count": 1}

    with patch("dpeva.analysis.managers.StatsCalculator", return_value=mock_stats_calc), patch(
        "dpeva.analysis.managers.InferenceVisualizer"
    ) as mock_visualizer_cls, patch.object(manager, "save_statistics") as mock_save_stats, patch.object(
        manager.logger, "warning"
    ) as mock_warning:
        visualizer = mock_visualizer_cls.return_value
        visualizer.plot_parity.side_effect = RuntimeError("plot broke")

        stats_export, metrics, _, _, _ = manager.analyze_model(
            data=data,
            output_dir=str(tmp_path),
            plot_level="basic",
        )

    assert metrics["e_mae"] == 0.2
    assert stats_export["energy"] == {"count": 1}
    mock_save_stats.assert_called_once()
    assert any(
        "Plot 'parity_Energy' failed and was skipped: plot broke" in call.args[0]
        for call in mock_warning.call_args_list
    )


def test_analyze_model_full_plot_level_calls_enhanced_parity(tmp_path):
    data = {
        "energy": {"pred_e": np.array([1.0]), "data_e": np.array([1.1])},
        "force": None,
        "virial": None,
        "has_ground_truth": True,
    }
    manager = UnifiedAnalysisManager(enable_cohesive_energy=False)

    mock_stats_calc = MagicMock()
    mock_stats_calc.e_true = np.array([1.1])
    mock_stats_calc.e_pred = np.array([1.0])
    mock_stats_calc.f_true = None
    mock_stats_calc.f_pred = None
    mock_stats_calc.v_true = None
    mock_stats_calc.v_pred = None
    mock_stats_calc.compute_metrics.return_value = {"e_mae": 0.1}
    mock_stats_calc.compute_relative_energy.return_value = None
    mock_stats_calc.get_distribution_stats.return_value = {"count": 1}

    with patch("dpeva.analysis.managers.StatsCalculator", return_value=mock_stats_calc), patch(
        "dpeva.analysis.managers.InferenceVisualizer"
    ) as mock_visualizer_cls, patch.object(manager, "save_statistics"):
        visualizer = mock_visualizer_cls.return_value
        manager.analyze_model(data=data, output_dir=str(tmp_path), plot_level="full")

    visualizer.plot_parity.assert_called_once()
    visualizer.plot_parity_enhanced.assert_called_once()


def test_analyze_model_without_ground_truth_only_plots_predicted_distribution(tmp_path):
    data = {
        "energy": {"pred_e": np.array([1.0, 1.2])},
        "force": None,
        "virial": None,
        "has_ground_truth": False,
    }
    manager = UnifiedAnalysisManager(enable_cohesive_energy=False)

    mock_stats_calc = MagicMock()
    mock_stats_calc.e_true = None
    mock_stats_calc.e_pred = np.array([1.0, 1.2])
    mock_stats_calc.f_true = None
    mock_stats_calc.f_pred = None
    mock_stats_calc.v_true = None
    mock_stats_calc.v_pred = None
    mock_stats_calc.compute_relative_energy.return_value = None
    mock_stats_calc.get_distribution_stats.return_value = {"count": 2}

    with patch("dpeva.analysis.managers.StatsCalculator", return_value=mock_stats_calc), patch(
        "dpeva.analysis.managers.InferenceVisualizer"
    ) as mock_visualizer_cls, patch.object(manager, "save_statistics"):
        visualizer = mock_visualizer_cls.return_value
        stats_export, metrics, _, e_rel_pred, e_rel_true = manager.analyze_model(
            data=data,
            output_dir=str(tmp_path),
            plot_level="basic",
        )

    assert metrics == {}
    assert e_rel_pred is None
    assert e_rel_true is None
    assert stats_export["energy"] == {"count": 2}
    visualizer.plot_distribution.assert_called_once()
    visualizer.plot_parity.assert_not_called()
    visualizer.plot_error_distribution.assert_not_called()


def test_analyze_model_disables_composition_on_frame_mismatch(tmp_path):
    data = {
        "energy": {"pred_e": np.array([1.0, 1.1]), "data_e": np.array([1.0, 1.1])},
        "force": None,
        "virial": None,
        "has_ground_truth": True,
    }
    manager = UnifiedAnalysisManager(enable_cohesive_energy=True)

    mock_stats_calc = MagicMock()
    mock_stats_calc.e_true = np.array([1.0, 1.1])
    mock_stats_calc.e_pred = np.array([1.0, 1.1])
    mock_stats_calc.f_true = None
    mock_stats_calc.f_pred = None
    mock_stats_calc.v_true = None
    mock_stats_calc.v_pred = None
    mock_stats_calc.compute_metrics.return_value = {"e_mae": 0.0}
    mock_stats_calc.compute_relative_energy.return_value = None
    mock_stats_calc.get_distribution_stats.return_value = {"count": 2}

    with patch("dpeva.analysis.managers.StatsCalculator", return_value=mock_stats_calc) as mock_stats_cls, patch(
        "dpeva.analysis.managers.InferenceVisualizer"
    ), patch.object(manager, "save_statistics"), patch.object(manager.logger, "warning") as mock_warning:
        manager.analyze_model(
            data=data,
            output_dir=str(tmp_path),
            atom_counts_list=[{"H": 1}],
            atom_num_list=[1],
            plot_level="basic",
        )

    kwargs = mock_stats_cls.call_args.kwargs
    assert kwargs["atom_counts_list"] is None
    assert kwargs["atom_num_list"] is None
    assert any("Frame count mismatch" in call.args[0] for call in mock_warning.call_args_list)


def test_analyze_model_skips_cohesive_family_when_relative_energy_unavailable(tmp_path):
    data = {
        "energy": {"pred_e": np.array([1.0]), "data_e": np.array([1.1])},
        "force": None,
        "virial": None,
        "has_ground_truth": True,
    }
    manager = UnifiedAnalysisManager(enable_cohesive_energy=True)

    mock_stats_calc = MagicMock()
    mock_stats_calc.e_true = np.array([1.1])
    mock_stats_calc.e_pred = np.array([1.0])
    mock_stats_calc.f_true = None
    mock_stats_calc.f_pred = None
    mock_stats_calc.v_true = None
    mock_stats_calc.v_pred = None
    mock_stats_calc.compute_metrics.return_value = {"e_mae": 0.1}
    mock_stats_calc.compute_relative_energy.return_value = None
    mock_stats_calc.get_distribution_stats.return_value = {"count": 1}

    with patch("dpeva.analysis.managers.StatsCalculator", return_value=mock_stats_calc), patch(
        "dpeva.analysis.managers.InferenceVisualizer"
    ) as mock_visualizer_cls, patch.object(manager, "save_statistics"):
        visualizer = mock_visualizer_cls.return_value
        _, _, _, e_rel_pred, e_rel_true = manager.analyze_model(
            data=data,
            output_dir=str(tmp_path),
            plot_level="full",
        )

    assert e_rel_pred is None
    assert e_rel_true is None
    visualizer.plot_parity_enhanced.assert_called_once()
    cohesive_calls = [call.args[2] for call in visualizer.plot_parity.call_args_list if len(call.args) >= 3]
    assert "Cohesive Energy" not in cohesive_calls
