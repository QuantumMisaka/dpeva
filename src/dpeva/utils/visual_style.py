import math

import matplotlib.pyplot as plt
import seaborn as sns


def get_publication_font_hierarchy(font_size: int = 12) -> dict[str, float]:
    """
    Build a publication-oriented font hierarchy from a base font size.

    Args:
        font_size (int): Base font size.

    Returns:
        dict[str, float]: Derived font sizes for plot elements.
    """
    base = float(font_size)
    return {
        "base": base,
        "title": round(base * 1.3, 2),
        "label": round(base * 1.08, 2),
        "tick": round(base * 0.96, 2),
        "legend": round(base * 0.94, 2),
        "legend_title": round(base, 2),
        "annotation": round(base * 0.9, 2),
    }


def scale_font_hierarchy(
    fonts: dict[str, float],
    *,
    title_scale: float = 1.0,
    label_scale: float = 1.0,
    tick_scale: float = 1.0,
    legend_scale: float = 1.0,
    legend_title_scale: float = 1.0,
) -> dict[str, float]:
    """
    Scale a font hierarchy while keeping the same semantic keys.

    Args:
        fonts (dict[str, float]): Source font hierarchy.
        title_scale (float): Title scale factor.
        label_scale (float): Axis label scale factor.
        tick_scale (float): Tick label scale factor.
        legend_scale (float): Legend label scale factor.
        legend_title_scale (float): Legend title scale factor.

    Returns:
        dict[str, float]: Scaled font hierarchy.
    """
    return {
        "base": fonts["base"],
        "title": round(fonts["title"] * title_scale, 2),
        "label": round(fonts["label"] * label_scale, 2),
        "tick": round(fonts["tick"] * tick_scale, 2),
        "legend": round(fonts["legend"] * legend_scale, 2),
        "legend_title": round(fonts["legend_title"] * legend_title_scale, 2),
        "annotation": fonts["annotation"],
    }


def get_collection_pca_scatter_profile(
    font_size: int | float = 12,
) -> dict[str, tuple[float, float] | dict[str, float]]:
    """
    Build the shared publication profile for Collection PCA scatter plots.

    Args:
        font_size (int | float): Base font size.

    Returns:
        dict[str, tuple[float, float] | dict[str, float]]: Shared PCA scatter profile.
    """
    base_fonts = get_publication_font_hierarchy(font_size)
    return {
        "figure_size": (11, 9),
        "axis_margins": (0.02, 0.02),
        "tick_target_count": 8,
        "fonts": scale_font_hierarchy(
            base_fonts,
            title_scale=1.24,
            label_scale=1.2,
            tick_scale=1.2,
            legend_scale=1.08,
            legend_title_scale=1.08,
        ),
    }


def get_analysis_distribution_profile(
    font_size: int | float = 12,
) -> dict[str, float | str | bool | tuple[float, float] | dict[str, float]]:
    """
    Build the shared publication profile for Analysis distribution plots.

    Args:
        font_size (int | float): Base font size.

    Returns:
        dict[str, float | str | bool | tuple[float, float] | dict[str, float]]:
            Shared distribution profile for single, overlay, and with-error plots.
    """
    base_fonts = get_publication_font_hierarchy(font_size)
    return {
        "figure_size": (8.8, 6.0),
        "comparison_figure_size": (11.8, 5.9),
        "with_error_width_ratios": (3.7, 1.4),
        "with_error_wspace": 0.42,
        "with_error_stats_anchor_x": 0.03,
        "with_error_stats_anchor_y": 0.38,
        "with_error_zero_linewidth": 1.2,
        "with_error_layout": {
            "left": 0.07,
            "right": 0.98,
            "top": 0.91,
            "bottom": 0.12,
        },
        "stat": "density",
        "bins": "fd",
        "element": "step",
        "fill": True,
        "hist_alpha": 0.22,
        "bw_adjust": 0.9,
        "step_linewidth": 1.3,
        "grid_alpha": 0.24,
        "legend_framealpha": 0.92,
        "overlay_legend_loc": "upper left",
        "overlay_legend_outside": True,
        "overlay_save_bbox_inches": "tight",
        "legend_bbox_anchor": (1.02, 1.0),
        "main_panel_right_margin": 0.84,
        "stats_anchor_x": 1.02,
        "stats_anchor_y": 0.66,
        "fonts": scale_font_hierarchy(
            base_fonts,
            title_scale=1.1,
            label_scale=1.12,
            tick_scale=1.08,
            legend_scale=1.04,
            legend_title_scale=1.02,
        ),
    }


def get_analysis_parity_profile(
    font_size: int | float = 12,
) -> dict[str, object]:
    """
    Build the shared publication profile for Analysis parity plots.

    Args:
        font_size (int | float): Base font size.

    Returns:
        dict[str, object]: Shared parity and enhanced-parity profile metadata.
    """
    base_fonts = get_publication_font_hierarchy(font_size)
    main_fonts = scale_font_hierarchy(
        base_fonts,
        title_scale=1.12,
        label_scale=1.14,
        tick_scale=1.1,
        legend_scale=1.02,
        legend_title_scale=1.0,
    )
    enhanced_fonts = scale_font_hierarchy(
        base_fonts,
        title_scale=1.14,
        label_scale=1.18,
        tick_scale=1.12,
        legend_scale=1.04,
        legend_title_scale=1.02,
    )
    panel_fonts = scale_font_hierarchy(
        base_fonts,
        title_scale=0.88,
        label_scale=0.92,
        tick_scale=0.88,
        legend_scale=0.9,
        legend_title_scale=0.9,
    )
    base_profile = {
        "figure_size": (6.8, 6.45),
        "scatter_alpha": 0.34,
        "scatter_size": 15.0,
        "scatter_color": "#1d4ed8",
        "identity_color": "#374151",
        "identity_alpha": 0.88,
        "identity_linewidth": 1.5,
        "grid_alpha": 0.22,
        "main_tick_count": 5,
        "panel_tick_count": 3,
        "fonts": main_fonts,
        "scientific_enabled": True,
        "scientific_powerlimits": (-3, 4),
        "enhanced": {
            "figure_size": (9.6, 7.9),
            "width_ratios": [1.0, 0.28],
            "height_ratios": [0.32, 1.0],
            "wspace": 0.045,
            "hspace": 0.1,
            "layout_padding": {
                "left": 0.095,
                "right": 0.955,
                "top": 0.905,
                "bottom": 0.105,
            },
            "renderer_policy": "auto",
            "renderer_default": "scatter",
            "main_scatter_alpha": 0.34,
            "main_scatter_size": 15.5,
            "main_density_mode": "scatter",
            "main_density_gridsize": 60,
            "main_density_mincnt": 1,
            "main_density_cmap": "cividis",
            "main_density_norm": "linear",
            "main_density_norm_gamma": 0.8,
            "main_density_linewidths": 0.0,
            "main_overlay_scatter_enabled": False,
            "main_overlay_scatter_alpha": 0.12,
            "main_overlay_scatter_size": 6.2,
            "main_overlay_scatter_color": "#1d4ed8",
            "panel_hist_alpha": 0.2,
            "error_hist_alpha": 0.18,
            "panel_tick_count": 3,
            "density_tick_count": 2,
            "error_tick_count": 2,
            "error_zero_linewidth": 1.1,
            "grid_alpha": 0.2,
            "suptitle_y": 1.000,
            "panel_title_pad": 5.0,
            "panel_title_y": 1.02,
            "side_panels_enabled": True,
            "error_inset_enabled": True,
            "colorbar_enabled": False,
            "colorbar_title": "Sample Density",
            "colorbar_orientation": "horizontal",
            "colorbar_tick_count": 4,
            "colorbar_label_pad": 8.0,
            "hexbin_width_ratios": [1.0, 0.24],
            "hexbin_wspace": 0.12,
            "hexbin_sidebar_height_ratios": [0.38, 0.62],
            "hexbin_sidebar_hspace": 0.28,
            "hexbin_colorbar_width_ratio": 1.0,
            "hexbin_colorbar_align": "center",
            "hexbin_violin_width": 0.74,
            "hexbin_violin_alpha": 0.34,
            "hexbin_violin_linewidth": 1.05,
            "scatter_sidebar_width_scale": 0.22,
            "scientific_enabled": True,
            "colors": {
                "true": "#ef4444",
                "pred": "#1d4ed8",
                "error": "#d97706",
                "identity": "#374151",
            },
            "fonts": enhanced_fonts,
            "panel_fonts": panel_fonts,
            "scientific_powerlimits": (-3, 4),
            "main_title": "{label} Enhanced Parity",
            "top_panel_title": "True",
            "right_panel_title": "Predicted",
            "error_panel_title": "Error Density",
            "top_panel_ylabel": "True Density",
            "top_panel_ylabel_pad": 7.0,
            "right_panel_xlabel": "Predicted Density",
            "right_panel_xlabel_pad": 4.0,
            "error_panel_xlabel_template": "",
            "error_panel_xlabel_pad": 1.5,
            "error_panel_ylabel": "Error Density",
            "error_panel_ylabel_pad": 7.0,
            "error_tick_pad": -3.0,
            "aligned_sidebar_gap": 0.040,
            "hexbin_aligned_sidebar_gap": 0.015,
        },
    }
    quantity_overrides = {
        "energy": {
            "scatter_alpha": 0.35,
            "scatter_size": 14.5,
            "scientific_enabled": False,
            "scientific_powerlimits": (-2, 3),
            "enhanced": {
                "figure_size": (9.2, 7.75),
                "width_ratios": [1.0, 0.27],
                "height_ratios": [0.29, 1.0],
                "main_scatter_alpha": 0.35,
                "main_scatter_size": 15.5,
                "panel_hist_alpha": 0.18,
                "error_hist_alpha": 0.17,
                "density_tick_count": 2,
                "scientific_enabled": False,
                "scientific_powerlimits": (-2, 3),
                "panel_title_pad": 4.0,
                "panel_title_y": 1.015,
                "suptitle_y": 1.000,
            },
        },
        "cohesive_energy": {
            "scatter_alpha": 0.4,
            "scatter_size": 16.0,
            "scientific_enabled": False,
            "scientific_powerlimits": (-2, 3),
            "enhanced": {
                "figure_size": (9.1, 7.7),
                "width_ratios": [1.0, 0.27],
                "height_ratios": [0.28, 1.0],
                "main_scatter_alpha": 0.38,
                "main_scatter_size": 15.8,
                "panel_hist_alpha": 0.17,
                "error_hist_alpha": 0.15,
                "panel_tick_count": 4,
                "density_tick_count": 2,
                "error_tick_count": 3,
                "scientific_enabled": False,
                "scientific_powerlimits": (-2, 3),
                "panel_title_pad": 3.6,
                "panel_title_y": 1.015,
                "suptitle_y": 1.000,
                "scatter_sidebar_width_scale": 0.24,
            },
        },
        "force": {
            "scatter_alpha": 0.42,
            "scatter_size": 16.5,
            "main_tick_count": 6,
            "scientific_powerlimits": (-3, 4),
            "enhanced": {
                "figure_size": (9.8, 7.4),
                "main_scatter_alpha": 0.42,
                "main_scatter_size": 17.0,
                "renderer_default": "hexbin",
                "main_density_mode": "hexbin",
                "main_density_gridsize": 60,
                "main_density_mincnt": 1,
                "main_density_cmap": "viridis",
                "main_density_norm": "power",
                "main_density_norm_gamma": 0.72,
                "panel_hist_alpha": 0.22,
                "error_hist_alpha": 0.18,
                "side_panels_enabled": False,
                "error_inset_enabled": True,
                "colorbar_enabled": True,
                "colorbar_title": "Counts Per Hexbin",
                "colorbar_orientation": "vertical",
                "hexbin_width_ratios": [1.0, 0.24],
                "hexbin_sidebar_height_ratios": [0.38, 0.62],
                "hexbin_sidebar_hspace": 0.45,
                "hexbin_colorbar_width_ratio": 0.46,
                "hexbin_colorbar_align": "center",
                "hexbin_aligned_sidebar_gap": 0.05,
                "panel_tick_count": 3,
                "error_tick_count": 2,
                "scientific_powerlimits": (-3, 4),
            },
        },
        "virial": {
            "scatter_alpha": 0.36,
            "scatter_size": 15.5,
            "main_tick_count": 5,
            "scientific_powerlimits": (-2, 3),
            "enhanced": {
                "figure_size": (9.6, 7.3),
                "main_scatter_alpha": 0.36,
                "main_scatter_size": 16.0,
                "renderer_default": "hexbin",
                "main_density_mode": "hexbin",
                "main_density_gridsize": 60,
                "main_density_mincnt": 1,
                "main_density_cmap": "viridis",
                "main_density_norm": "power",
                "main_density_norm_gamma": 0.78,
                "panel_hist_alpha": 0.19,
                "error_hist_alpha": 0.17,
                "side_panels_enabled": False,
                "error_inset_enabled": True,
                "colorbar_enabled": True,
                "colorbar_title": "Counts Per Hexbin",
                "colorbar_orientation": "vertical",
                "hexbin_width_ratios": [1.0, 0.24],
                "hexbin_sidebar_height_ratios": [0.37, 0.63],
                "hexbin_sidebar_hspace": 0.45,
                "hexbin_colorbar_width_ratio": 0.46,
                "hexbin_colorbar_align": "center",
                "hexbin_aligned_sidebar_gap": 0.05,
                "panel_tick_count": 3,
                "error_tick_count": 2,
                "scientific_powerlimits": (-2, 3),
            },
        },
    }
    return {
        **base_profile,
        "quantity_overrides": quantity_overrides,
    }


def get_analysis_enhanced_parity_profile(
    font_size: int | float = 12,
) -> dict[
    str, float | tuple[float, float] | list[float] | dict[str, float] | dict[str, str]
]:
    """
    Build the shared publication profile for Analysis enhanced parity plots.

    Args:
        font_size (int | float): Base font size.

    Returns:
        dict[str, float | tuple[float, float] | list[float] | dict[str, float] | dict[str, str]]:
            Shared enhanced parity layout and typography profile.
    """
    parity_profile = get_analysis_parity_profile(font_size)
    enhanced_profile = parity_profile["enhanced"]
    if not isinstance(enhanced_profile, dict):
        raise TypeError("Analysis enhanced parity profile must be a dictionary.")
    return enhanced_profile


def resolve_linked_tick_step(
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    target_tick_count: int = 6,
) -> float:
    """
    Resolve a shared major tick step for paired axes.

    Args:
        x_limits (tuple[float, float]): X-axis limits.
        y_limits (tuple[float, float]): Y-axis limits.
        target_tick_count (int): Approximate number of ticks to display.

    Returns:
        float: A shared "nice" tick step.
    """
    spans = []
    for start, end in (x_limits, y_limits):
        if math.isfinite(start) and math.isfinite(end):
            span = abs(float(end) - float(start))
            if span > 0:
                spans.append(span)
    if not spans:
        return 1.0
    max_span = max(spans)
    raw_step = max_span / max(target_tick_count - 1, 1)
    exponent = math.floor(math.log10(raw_step)) if raw_step > 0 else 0
    magnitude = 10**exponent
    for multiplier in (1, 2, 5, 10):
        step = multiplier * magnitude
        if raw_step <= step:
            return step
    return 10 * magnitude


def get_legend_layout(
    item_count: int,
    max_rows: int = 8,
    max_cols: int = 4,
) -> dict[str, float | int | tuple[float, float] | str]:
    """
    Compute a balanced legend layout for dense multi-series plots.

    Args:
        item_count (int): Number of legend items.
        max_rows (int): Preferred maximum legend rows.
        max_cols (int): Upper bound of legend columns.

    Returns:
        dict[str, float | int | tuple[float, float] | str]: Legend layout metadata.
    """
    safe_item_count = max(int(item_count), 0)
    if safe_item_count == 0:
        return {
            "ncol": 1,
            "loc": "upper center",
            "bbox_to_anchor": (0.5, -0.1),
            "bottom_margin": 0.18,
        }
    safe_max_rows = max(int(max_rows), 1)
    safe_max_cols = max(int(max_cols), 1)
    ncol = min(max(math.ceil(safe_item_count / safe_max_rows), 1), safe_max_cols)
    nrows = math.ceil(safe_item_count / ncol)
    bottom_margin = min(0.18 + max(nrows - 1, 0) * 0.06, 0.42)
    return {
        "ncol": ncol,
        "loc": "upper center",
        "bbox_to_anchor": (0.5, -0.12),
        "bottom_margin": bottom_margin,
    }


def set_visual_style(
    font_size: int = 12, context: str = "paper", style: str = "whitegrid"
):
    """
    Sets the global visualization style for DPEVA plots.

    Args:
        font_size (int): Base font size.
        context (str): Seaborn context ('paper', 'notebook', 'talk', 'poster').
        style (str): Seaborn style ('whitegrid', 'darkgrid', 'white', 'dark', 'ticks').
    """
    fonts = get_publication_font_hierarchy(font_size)
    sns.set_theme(style=style, context=context)
    plt.rcParams.update(
        {
            "font.size": fonts["base"],
            "axes.labelsize": fonts["label"],
            "axes.titlesize": fonts["title"],
            "axes.titleweight": "semibold",
            "xtick.labelsize": fonts["tick"],
            "ytick.labelsize": fonts["tick"],
            "legend.fontsize": fonts["legend"],
            "legend.title_fontsize": fonts["legend_title"],
            "figure.titlesize": fonts["title"],
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.top": False,
            "ytick.right": False,
            "lines.linewidth": 1.5,
            "axes.linewidth": 1.0,
            "grid.linewidth": 0.8,
            "grid.alpha": 0.35,
            "xtick.major.size": 4.0,
            "ytick.major.size": 4.0,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
