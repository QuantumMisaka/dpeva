import logging
import numpy as np
from unittest.mock import patch

from dpeva.inference.visualizer import InferenceVisualizer


def test_plot_parity_filters_non_finite_and_writes_file(tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    viz = InferenceVisualizer(str(tmp_path))
    y_true = np.array([0.0, 1.0, np.nan, 2.0, np.inf])
    y_pred = np.array([0.0, 1.1, 0.2, np.nan, 2.1])
    viz.plot_parity(y_true, y_pred, "Energy", "eV/atom")
    assert (tmp_path / "parity_energy.png").exists()
    assert "filtered non-finite paired values" in caplog.text


def test_plot_parity_skips_without_finite_pairs(tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    viz = InferenceVisualizer(str(tmp_path))
    y_true = np.array([np.nan, np.inf])
    y_pred = np.array([np.nan, np.inf])
    viz.plot_parity(y_true, y_pred, "Energy", "eV/atom")
    assert not (tmp_path / "parity_energy.png").exists()
    assert "no finite paired values" in caplog.text


def test_plot_distribution_skips_without_finite_values(tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    viz = InferenceVisualizer(str(tmp_path))
    values = np.array([np.nan, np.inf, -np.inf])
    viz.plot_distribution(values, "Predicted Energy", "eV/atom")
    assert not (tmp_path / "dist_predicted_energy.png").exists()
    assert "no finite values" in caplog.text


def test_plot_parity_enhanced_and_overlay_outputs(tmp_path):
    viz = InferenceVisualizer(str(tmp_path))
    y_true = np.array([0.0, 1.0, 2.0, 3.0])
    y_pred = np.array([0.1, 0.9, 2.2, 2.9])
    viz.plot_parity_enhanced(y_true, y_pred, "Energy", "eV/atom")
    viz.plot_distribution_overlay(y_pred, y_true, "Energy", "eV/atom")
    viz.plot_distribution_with_error(y_pred, y_true, y_pred - y_true, "Energy", "eV/atom")
    assert (tmp_path / "parity_energy_enhanced.png").exists()
    assert (tmp_path / "dist_energy_overlay.png").exists()
    assert (tmp_path / "dist_energy_with_error.png").exists()


def test_stats_text_excludes_quantiles(tmp_path):
    viz = InferenceVisualizer(str(tmp_path))
    text = viz._stats_text(np.array([1.0, 2.0, 3.0]), "Sample")
    assert "25%" not in text
    assert "50%" not in text
    assert "75%" not in text


def test_error_distribution_does_not_render_stats_box(tmp_path):
    viz = InferenceVisualizer(str(tmp_path))
    with patch.object(viz, "_add_stats_box") as mock_box:
        viz.plot_error_distribution(np.array([0.1, -0.1, 0.2]), "Energy", "eV/atom")
        mock_box.assert_not_called()


def test_enhanced_parity_does_not_render_stats_box(tmp_path):
    viz = InferenceVisualizer(str(tmp_path))
    with patch.object(viz, "_add_stats_box") as mock_box:
        viz.plot_parity_enhanced(np.array([1.0, 2.0, 3.0]), np.array([1.1, 1.9, 3.2]), "Energy", "eV/atom")
        mock_box.assert_not_called()


def test_enhanced_parity_includes_error_inset_render_path(tmp_path):
    viz = InferenceVisualizer(str(tmp_path))
    y_true = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    y_pred = np.array([-1.9, -1.2, 0.1, 1.2, 1.8, 3.1])
    viz.plot_parity_enhanced(y_true, y_pred, "Virial", "eV")
    assert (tmp_path / "parity_virial_enhanced.png").exists()
