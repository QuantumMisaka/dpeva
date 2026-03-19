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
