import logging
import numpy as np
from unittest.mock import MagicMock, patch
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.figure import Figure

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
    viz.plot_distribution_with_error(
        y_pred, y_true, y_pred - y_true, "Energy", "eV/atom"
    )
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


def test_enhanced_parity_renders_stats_box(tmp_path):
        viz = InferenceVisualizer(str(tmp_path))
        with patch.object(viz, "_add_stats_box") as mock_box:
            viz.plot_parity_enhanced(
                np.array([1.0, 2.0, 3.0]), np.array([1.1, 1.9, 3.2]), "Energy", "eV/atom"
            )
            mock_box.assert_called_once()
            
            # test regular parity also renders stats box
            mock_box.reset_mock()
            viz.plot_parity(
                np.array([1.0, 2.0, 3.0]), np.array([1.1, 1.9, 3.2]), "Energy", "eV/atom"
            )
            mock_box.assert_called_once()


def test_enhanced_parity_includes_error_inset_render_path(tmp_path):
    viz = InferenceVisualizer(str(tmp_path))
    y_true = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    y_pred = np.array([-1.9, -1.2, 0.1, 1.2, 1.8, 3.1])
    viz.plot_parity_enhanced(y_true, y_pred, "Virial", "eV")
    assert (tmp_path / "parity_virial_enhanced.png").exists()


def test_distribution_with_error_avoids_tight_layout_warning(tmp_path, recwarn):
    viz = InferenceVisualizer(str(tmp_path))
    y_true = np.array([0.0, 1.0, 2.0, 3.0])
    y_pred = np.array([0.1, 0.9, 2.2, 2.9])
    viz.plot_distribution_with_error(
        y_pred, y_true, y_pred - y_true, "Energy", "eV/atom"
    )
    assert not any("tight_layout" in str(w.message) for w in recwarn)


def test_plot_distribution_uses_lightweight_histogram_kde_template(tmp_path):
    viz = InferenceVisualizer(str(tmp_path))
    with patch("dpeva.inference.visualizer.sns.histplot") as mock_histplot:
        viz.plot_distribution(
            np.array([0.0, 0.2, 0.5, 1.0]), "Predicted Energy", "eV/atom"
        )
    kwargs = mock_histplot.call_args.kwargs
    assert kwargs["kde"] is True
    assert kwargs["stat"] == "density"
    assert kwargs["element"] == "step"
    assert kwargs["fill"] is True
    assert kwargs["bins"] == "fd"
    assert kwargs["kde_kws"]["cut"] == 0


def test_plot_parity_enhanced_reuses_histogram_kde_template(tmp_path):
    viz = InferenceVisualizer(str(tmp_path))
    with patch("dpeva.inference.visualizer.sns.histplot") as mock_histplot:
        viz.plot_parity_enhanced(
            np.array([0.0, 1.0, 2.0, 3.0]),
            np.array([0.1, 0.9, 2.2, 2.9]),
            "Energy",
            "eV/atom",
        )
    assert mock_histplot.call_count == 3
    for call in mock_histplot.call_args_list:
        kwargs = call.kwargs
        assert kwargs["kde"] is True
        assert kwargs["stat"] == "density"
        assert kwargs["element"] == "step"
        assert kwargs["fill"] is True
        assert kwargs["bins"] == "fd"


def test_force_quantity_profile_keeps_enhanced_scatter_weight(tmp_path):
    viz = InferenceVisualizer(str(tmp_path))
    force_profile = viz._get_parity_profile("Force")
    force_enhanced = viz._get_parity_profile("Force", enhanced=True)
    cohesive_enhanced = viz._get_parity_profile("Cohesive Energy", enhanced=True)
    assert force_enhanced["scatter_color"] == force_profile["scatter_color"]
    assert force_enhanced["main_scatter_alpha"] >= force_profile["scatter_alpha"]
    assert force_enhanced["main_scatter_size"] >= force_profile["scatter_size"]
    assert (
        force_enhanced["main_scatter_alpha"] > cohesive_enhanced["main_scatter_alpha"]
    )


def test_plot_parity_uses_force_quantity_profile(tmp_path):
    viz = InferenceVisualizer(str(tmp_path))
    profile = viz._get_parity_profile("Force")
    with patch("matplotlib.axes._axes.Axes.scatter") as mock_scatter:
        viz.plot_parity(
            np.array([0.0, 0.5, 1.0, 1.5]),
            np.array([0.1, 0.55, 0.95, 1.45]),
            "Force",
            "eV/Å",
        )
    kwargs = mock_scatter.call_args.kwargs
    assert kwargs["alpha"] == profile["scatter_alpha"]
    assert kwargs["s"] == profile["scatter_size"]
    assert kwargs["c"] == profile["scatter_color"]


def test_plot_parity_enhanced_uses_force_quantity_profile(tmp_path):
    viz = InferenceVisualizer(str(tmp_path))
    profile = viz._get_parity_profile("Force", enhanced=True)
    mock_collection = MagicMock()
    mock_collection.get_array.return_value = np.array([1.0, 3.0, 5.0])
    with patch(
        "matplotlib.axes._axes.Axes.hexbin", return_value=mock_collection
    ) as mock_hexbin, patch.object(
        Figure, "colorbar"
    ):
        viz.plot_parity_enhanced(
            np.array([0.0, 0.5, 1.0, 1.5]),
            np.array([0.1, 0.55, 0.95, 1.45]),
            "Force",
            "eV/Å",
        )
    kwargs = mock_hexbin.call_args.kwargs
    assert profile["main_density_mode"] == "hexbin"
    assert kwargs["gridsize"] == profile["main_density_gridsize"]
    assert kwargs["mincnt"] == profile["main_density_mincnt"]
    assert kwargs["cmap"] == profile["main_density_cmap"]
    assert mock_collection.set_norm.call_args.args[0].__class__.__name__ == "LogNorm"


def test_hexbin_mode_forces_log_norm_in_enhanced_profile(tmp_path):
    viz = InferenceVisualizer(str(tmp_path), enhanced_parity_renderer="hexbin")
    profile = viz._get_parity_profile("Energy", enhanced=True)
    assert profile["main_density_mode"] == "hexbin"
    assert profile["main_density_norm"] == "log"


def test_force_enhanced_profile_uses_panel_policy_and_layout_override(tmp_path):
    viz = InferenceVisualizer(str(tmp_path))
    profile = viz._get_parity_profile("Force", enhanced=True)
    assert profile["side_panels_enabled"] is False
    assert profile["error_inset_enabled"] is True
    assert profile["main_density_mode"] == "hexbin"
    assert profile["colorbar_enabled"] is True
    assert profile["main_density_cmap"] == "viridis"
    assert profile["main_density_gridsize"] == 60
    assert profile["main_density_mincnt"] == 1
    assert profile["main_overlay_scatter_enabled"] is False
    assert profile["hexbin_width_ratios"][-1] < 0.3


def test_virial_enhanced_profile_matches_force_hexbin_policy(tmp_path):
    viz = InferenceVisualizer(str(tmp_path))
    profile = viz._get_parity_profile("Virial", enhanced=True)
    assert profile["main_density_mode"] == "hexbin"
    assert profile["main_density_cmap"] == "viridis"
    assert profile["main_density_gridsize"] == 60
    assert profile["main_density_mincnt"] == 1
    assert profile["main_overlay_scatter_enabled"] is False


def test_plot_parity_enhanced_force_renders_histogram_error_panel_only(tmp_path):
    viz = InferenceVisualizer(str(tmp_path))
    with patch.object(viz, "_plot_histogram_with_kde") as mock_hist, patch.object(
        Figure, "colorbar"
    ):
        viz.plot_parity_enhanced(
            np.array([0.0, 0.5, 1.0, 1.5]),
            np.array([0.1, 0.55, 0.95, 1.45]),
            "Force",
            "eV/Å",
        )
    assert mock_hist.call_count == 1


def test_plot_parity_enhanced_force_adds_colorbar_sidebar(tmp_path):
    viz = InferenceVisualizer(str(tmp_path))
    mock_colorbar = MagicMock()
    with patch.object(
        Figure, "colorbar", return_value=mock_colorbar
    ) as mock_colorbar_call:
        viz.plot_parity_enhanced(
            np.array([0.0, 0.5, 1.0, 1.5]),
            np.array([0.1, 0.55, 0.95, 1.45]),
            "Force",
            "eV/Å",
        )
    assert mock_colorbar_call.called
    assert mock_colorbar_call.call_args.kwargs["orientation"] == "vertical"


def test_plot_parity_enhanced_force_uses_vertical_error_zero_line(tmp_path):
    viz = InferenceVisualizer(str(tmp_path))
    with patch("matplotlib.axes._axes.Axes.axhline") as mock_axhline, patch(
        "matplotlib.axes._axes.Axes.axvline"
    ) as mock_axvline:
        viz.plot_parity_enhanced(
            np.array([0.0, 0.5, 1.0, 1.5]),
            np.array([0.1, 0.55, 0.95, 1.45]),
            "Force",
            "eV/Å",
        )
    assert not mock_axhline.called
    assert mock_axvline.called


def test_plot_parity_enhanced_force_uses_sidebar_axis_labels(tmp_path):
    viz = InferenceVisualizer(str(tmp_path))
    with patch.object(Figure, "savefig", autospec=True), patch(
        "matplotlib.pyplot.close"
    ):
        viz.plot_parity_enhanced(
            np.array([0.0, 0.5, 1.0, 1.5]),
            np.array([0.1, 0.55, 0.95, 1.45]),
            "Force",
            "eV/Å",
        )
    fig = plt.gcf()
    ax_err = fig.axes[1]
    ax_cbar = fig.axes[2]
    assert ax_err.get_xlabel() == "Error Density"
    assert ax_err.get_ylabel() == ""
    assert ax_cbar.get_xlabel() == ""
    assert ax_cbar.get_ylabel() == "Counts Per Hexbin"
    plt.close(fig)


def test_plot_parity_enhanced_force_no_overlay_scatter_on_hexbin(tmp_path):
    viz = InferenceVisualizer(str(tmp_path))
    mock_collection = MagicMock()
    mock_collection.get_array.return_value = np.array([1.0, 3.0, 5.0])
    with patch(
        "matplotlib.axes._axes.Axes.hexbin", return_value=mock_collection
    ) as mock_hexbin, patch(
        "matplotlib.axes._axes.Axes.scatter"
    ) as mock_scatter, patch.object(
        Figure, "colorbar"
    ):
        viz.plot_parity_enhanced(
            np.array([0.0, 0.5, 1.0, 1.5]),
            np.array([0.1, 0.55, 0.95, 1.45]),
            "Force",
            "eV/Å",
        )
    assert mock_hexbin.called
    assert not mock_scatter.called
    assert "C" not in mock_hexbin.call_args.kwargs
    assert "reduce_C_function" not in mock_hexbin.call_args.kwargs


def test_plot_parity_energy_uses_non_scientific_formatter(tmp_path):
    viz = InferenceVisualizer(str(tmp_path))
    with patch.object(viz, "_apply_scalar_formatter") as mock_formatter:
        viz.plot_parity(
            np.array([0.0, 0.5, 1.0, 1.5]),
            np.array([0.1, 0.55, 0.95, 1.45]),
            "Energy",
            "eV/atom",
        )
    for call in mock_formatter.call_args_list:
        assert call.kwargs["scientific_enabled"] is False


def test_plot_parity_enhanced_energy_uses_non_scientific_formatter(tmp_path):
    viz = InferenceVisualizer(str(tmp_path))
    with patch.object(viz, "_apply_scalar_formatter") as mock_formatter:
        viz.plot_parity_enhanced(
            np.array([0.0, 0.5, 1.0, 1.5]),
            np.array([0.1, 0.55, 0.95, 1.45]),
            "Energy",
            "eV/atom",
        )
    for call in mock_formatter.call_args_list:
        assert call.kwargs["scientific_enabled"] is False


def test_plot_parity_enhanced_energy_keeps_scatter_main_layer(tmp_path):
    viz = InferenceVisualizer(str(tmp_path))
    with patch("matplotlib.axes._axes.Axes.scatter") as mock_scatter:
        viz.plot_parity_enhanced(
            np.array([0.0, 0.5, 1.0, 1.5]),
            np.array([0.1, 0.55, 0.95, 1.45]),
            "Energy",
            "eV/atom",
        )
    assert mock_scatter.called


def test_scatter_enhanced_axes_align_to_main_panel(tmp_path):
    viz = InferenceVisualizer(str(tmp_path))
    profile = viz._get_parity_profile("Cohesive Energy", enhanced=True)
    fig = plt.figure(figsize=profile["figure_size"])
    ax_main, ax_top, ax_right, ax_err, _ = viz._create_enhanced_parity_axes(fig, profile)
    ax_main.set_xlim(-7, -1)
    ax_main.set_ylim(-7, -1)
    ax_main.set_aspect("equal", adjustable="box")
    fig.canvas.draw()
    initial_main_box = ax_main.get_position()
    initial_top_box = ax_top.get_position()
    initial_right_box = ax_right.get_position()
    expected_width = min(
        max(
            initial_top_box.height,
            initial_main_box.width * profile["scatter_sidebar_width_scale"],
        ),
        max(
            initial_right_box.x1
            - (initial_main_box.x1 + profile["aligned_sidebar_gap"]),
            0.05,
        ),
    )
    viz._align_scatter_enhanced_axes(fig, ax_main, ax_top, ax_right, ax_err)

    main_box = ax_main.get_position()
    top_box = ax_top.get_position()
    right_box = ax_right.get_position()
    err_box = ax_err.get_position()

    assert abs(top_box.x0 - main_box.x0) < 1e-6
    assert abs(top_box.x1 - main_box.x1) < 1e-6
    assert abs(right_box.y0 - main_box.y0) < 1e-6
    assert abs(right_box.height - main_box.height) < 1e-6
    assert abs(right_box.width - expected_width) < 1e-6
    assert abs(err_box.y1 - top_box.y1) < 1e-6
    assert abs(err_box.height - top_box.height) < 1e-6
    assert abs(err_box.width - expected_width) < 1e-6
    assert abs(err_box.x0 - right_box.x0) < 1e-6
    assert abs(err_box.x1 - right_box.x1) < 1e-6
    assert (right_box.x0 - main_box.x1) >= 0.0
    plt.close(fig)


def test_hexbin_enhanced_sidebar_moves_closer_to_main_panel(tmp_path):
    viz = InferenceVisualizer(str(tmp_path))
    profile = viz._get_parity_profile("Force", enhanced=True)
    fig = plt.figure(figsize=profile["figure_size"])
    ax_main, _, _, ax_err, ax_cbar = viz._create_enhanced_parity_axes(fig, profile)
    ax_main.set_xlim(-50, 50)
    ax_main.set_ylim(-50, 50)
    ax_main.set_aspect("equal", adjustable="box")
    viz._align_hexbin_sidebar_axes(fig, ax_main, ax_err, ax_cbar, profile)

    main_box = ax_main.get_position()
    err_box = ax_err.get_position()
    cbar_box = ax_cbar.get_position()

    assert cbar_box.width < err_box.width
    assert abs((cbar_box.x0 + cbar_box.x1) / 2 - (err_box.x0 + err_box.x1) / 2) < 1e-6
    assert abs((err_box.x0 - main_box.x1) - 0.05) < 1e-6
    plt.close(fig)


def test_cohesive_energy_scatter_profile_uses_compact_sidebar_and_restrained_ticks(tmp_path):
    viz = InferenceVisualizer(str(tmp_path))
    profile = viz._get_parity_profile("Cohesive Energy", enhanced=True)
    assert profile["main_density_mode"] == "scatter"
    assert profile["width_ratios"][-1] == 0.27
    assert profile["density_tick_count"] == 2
    assert abs(profile["panel_title_y"] - 1.015) < 1e-6
    assert profile["top_panel_ylabel"] == "True Density"
    assert profile["right_panel_xlabel"] == "Predicted Density"
    assert profile["error_panel_xlabel_template"] == ""
    assert profile["error_panel_ylabel"] == "Error Density"
    assert profile["aligned_sidebar_gap"] <= 0.04
    assert profile["scatter_sidebar_width_scale"] >= 0.24


def test_plot_parity_enhanced_energy_uses_axis_label_semantics(tmp_path):
    viz = InferenceVisualizer(str(tmp_path))
    with patch.object(Figure, "savefig", autospec=True), patch(
        "matplotlib.pyplot.close"
    ):
        viz.plot_parity_enhanced(
            np.array([0.0, 0.5, 1.0, 1.5]),
            np.array([0.1, 0.55, 0.95, 1.45]),
            "Energy",
            "eV/atom",
        )
    fig = plt.gcf()
    ax_top = fig.axes[0]
    ax_main = fig.axes[1]
    ax_right = fig.axes[2]
    ax_err = fig.axes[3]
    assert ax_top.get_xlabel() == ""
    assert ax_top.get_ylabel() == "True Density"
    assert ax_right.get_xlabel() == "Predicted Density"
    assert ax_err.get_xlabel() == ""
    assert ax_err.get_ylabel() == "Error Density"
    assert to_hex(ax_err.yaxis.label.get_color()) == to_hex(plt.rcParams["axes.labelcolor"])
    assert abs(ax_right.get_position().height - ax_main.get_position().height) < 1e-6
    plt.close(fig)


def test_apply_axis_fonts_supports_panel_title_y(tmp_path):
    viz = InferenceVisualizer(str(tmp_path))
    fig, ax = plt.subplots()
    viz._apply_axis_fonts(ax, title="Panel", title_y=1.01)
    assert abs(ax.title.get_position()[1] - 1.01) < 1e-6
    plt.close(fig)


def test_apply_scalar_formatter_can_hide_panel_offset_text(tmp_path):
    viz = InferenceVisualizer(str(tmp_path))
    fig, ax = plt.subplots()
    viz._apply_scalar_formatter(
        ax,
        axis="x",
        powerlimits=(-2, 3),
        scientific_enabled=False,
        suppress_offset_text=True,
    )
    assert ax.xaxis.offsetText.get_visible() is False
    plt.close(fig)


def test_plot_parity_enhanced_respects_explicit_renderer_override(tmp_path):
    viz = InferenceVisualizer(str(tmp_path), enhanced_parity_renderer="scatter")
    profile = viz._get_parity_profile("Force", enhanced=True)
    assert profile["renderer_policy"] == "scatter"
    assert profile["main_density_mode"] == "scatter"
    with patch("matplotlib.axes._axes.Axes.scatter") as mock_scatter, patch(
        "matplotlib.axes._axes.Axes.hexbin"
    ) as mock_hexbin:
        viz.plot_parity_enhanced(
            np.array([0.0, 0.5, 1.0, 1.5]),
            np.array([0.1, 0.55, 0.95, 1.45]),
            "Force",
            "eV/Å",
        )
    assert mock_scatter.called
    assert not mock_hexbin.called


def test_overlay_distribution_uses_tight_bbox_when_legend_outside(tmp_path):
    viz = InferenceVisualizer(str(tmp_path))
    viz.distribution_profile["overlay_legend_outside"] = True
    viz.distribution_profile["overlay_save_bbox_inches"] = "tight"
    with patch.object(Figure, "savefig") as mock_savefig:
        viz.plot_distribution_overlay(
            np.array([0.2, 0.5, 0.8]),
            np.array([0.1, 0.4, 0.9]),
            "Energy",
            "eV/atom",
        )
    assert mock_savefig.call_args.kwargs["bbox_inches"] == "tight"


def test_overlay_distribution_skips_tight_bbox_when_legend_inside(tmp_path):
    viz = InferenceVisualizer(str(tmp_path))
    viz.distribution_profile["overlay_legend_outside"] = False
    with patch.object(Figure, "savefig") as mock_savefig:
        viz.plot_distribution_overlay(
            np.array([0.2, 0.5, 0.8]),
            np.array([0.1, 0.4, 0.9]),
            "Energy",
            "eV/atom",
        )
    assert "bbox_inches" not in mock_savefig.call_args.kwargs


def test_distribution_with_error_uses_profile_driven_layout_values(tmp_path):
    viz = InferenceVisualizer(str(tmp_path))
    viz.distribution_profile["with_error_stats_anchor_x"] = 0.11
    viz.distribution_profile["with_error_stats_anchor_y"] = 0.27
    viz.distribution_profile["with_error_zero_linewidth"] = 2.4
    with patch.object(viz, "_add_stats_box") as mock_stats_box, patch(
        "matplotlib.axes._axes.Axes.axvline"
    ) as mock_axvline:
        viz.plot_distribution_with_error(
            np.array([0.1, 0.9, 2.2, 2.9]),
            np.array([0.0, 1.0, 2.0, 3.0]),
            np.array([0.1, -0.1, 0.2, -0.1]),
            "Energy",
            "eV/atom",
        )
    stats_kwargs = mock_stats_box.call_args.kwargs
    assert stats_kwargs["x"] == 0.11
    assert stats_kwargs["y"] == 0.27
    assert mock_axvline.call_args.kwargs["linewidth"] == 2.4
