import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
from matplotlib import gridspec
from matplotlib.colors import LogNorm, Normalize, PowerNorm
from matplotlib.ticker import (
    FormatStrFormatter,
    LogFormatterSciNotation,
    LogLocator,
    MaxNLocator,
    ScalarFormatter,
)
from dpeva.constants import FIG_DPI
from dpeva.utils.visual_style import (
    get_analysis_distribution_profile,
    get_analysis_parity_profile,
    set_visual_style,
)


class InferenceVisualizer:
    """
    Visualization tools for inference results (Parity Plots, Error Distributions).
    """

    def __init__(
        self,
        output_dir: str,
        dpi: int = FIG_DPI,
        enhanced_parity_renderer: str = "auto",
    ):
        """
        Initialize the InferenceVisualizer.

        Args:
            output_dir (str): Directory to save plots.
            dpi (int, optional): Resolution for saved images. Defaults to FIG_DPI.
        """
        self.output_dir = output_dir
        self.dpi = dpi
        self.logger = logging.getLogger(__name__)
        set_visual_style()
        self.distribution_profile = get_analysis_distribution_profile()
        self.parity_profile = get_analysis_parity_profile()
        self.enhanced_parity_renderer = enhanced_parity_renderer

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def _filter_finite_1d(
        self, data: np.ndarray, label: str, context: str
    ) -> Optional[np.ndarray]:
        values = np.asarray(data, dtype=float).reshape(-1)
        finite_mask = np.isfinite(values)
        finite_count = int(np.count_nonzero(finite_mask))
        if finite_count == 0:
            self.logger.warning(f"{context} skipped for {label}: no finite values.")
            return None
        if finite_count < values.size:
            self.logger.warning(
                f"{context} for {label}: filtered non-finite values "
                f"({values.size - finite_count}/{values.size})."
            )
        return values[finite_mask]

    def _filter_finite_pair(
        self, x: np.ndarray, y: np.ndarray, label: str, context: str
    ):
        x_arr = np.asarray(x, dtype=float).reshape(-1)
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        if x_arr.size == 0 or y_arr.size == 0:
            self.logger.warning(f"{context} skipped for {label}: empty input.")
            return None, None
        valid_mask = np.isfinite(x_arr) & np.isfinite(y_arr)
        valid_count = int(np.count_nonzero(valid_mask))
        if valid_count == 0:
            self.logger.warning(
                f"{context} skipped for {label}: no finite paired values."
            )
            return None, None
        if valid_count < x_arr.size:
            self.logger.warning(
                f"{context} for {label}: filtered non-finite paired values "
                f"({x_arr.size - valid_count}/{x_arr.size})."
            )
        return x_arr[valid_mask], y_arr[valid_mask]

    def _stats_text(self, data: np.ndarray, name: str, include_name: bool = True) -> str:
        values = np.asarray(data, dtype=float).reshape(-1)
        desc = {
            "count": float(values.size),
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
        lines = []
        if include_name:
            lines.append(name)
        lines.extend([
            f"count={desc['count']:.0f}",
            f"mean={desc['mean']:.4f}",
            f"std={desc['std']:.4f}",
            f"min={desc['min']:.4f}",
            f"max={desc['max']:.4f}"
        ])
        return "\n".join(lines)

    def _add_stats_box(self, ax, text: str, x: float = 0.02, y: float = 0.98, fontsize: float = 12, ha: str = "left"):
        ax.text(
            x,
            y,
            text,
            transform=ax.transAxes,
            va="top",
            ha=ha,
            fontsize=fontsize,
            bbox={
                "boxstyle": "round",
                "facecolor": "white",
                "alpha": 0.8,
                "edgecolor": "#999999",
            },
        )

    def _apply_axis_fonts(
        self,
        ax,
        *,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        fonts: Optional[dict[str, float]] = None,
        title_color: Optional[str] = None,
        title_pad: float = 6.0,
        title_y: Optional[float] = None,
    ):
        axis_fonts = fonts or self.distribution_profile["fonts"]
        if title is not None:
            title_kwargs = {"fontsize": axis_fonts["title"], "pad": title_pad}
            if title_color is not None:
                title_kwargs["color"] = title_color
            if title_y is not None:
                title_kwargs["y"] = title_y
            ax.set_title(title, **title_kwargs)
        if xlabel is not None:
            ax.set_xlabel(xlabel, fontsize=axis_fonts["label"])
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize=axis_fonts["label"])
        ax.tick_params(axis="both", labelsize=axis_fonts["tick"])

    def _plot_histogram_with_kde(
        self,
        ax,
        data: np.ndarray,
        *,
        color: str,
        label: Optional[str] = None,
        orientation: str = "vertical",
        alpha: Optional[float] = None,
    ):
        profile = self.distribution_profile
        histplot_kwargs = {
            "kde": True,
            "stat": profile["stat"],
            "color": color,
            "alpha": profile["hist_alpha"] if alpha is None else alpha,
            "element": profile["element"],
            "fill": profile["fill"],
            "bins": profile["bins"],
            "kde_kws": {"bw_adjust": profile["bw_adjust"], "cut": 0},
            "line_kws": {"linewidth": profile["step_linewidth"]},
            "ax": ax,
        }
        if label is not None:
            histplot_kwargs["label"] = label
        if orientation == "horizontal":
            sns.histplot(y=data, **histplot_kwargs)
        else:
            sns.histplot(data, **histplot_kwargs)

    def _resolve_parity_quantity(self, label: str) -> str:
        normalized = label.strip().lower().replace("-", " ").replace("_", " ")
        if "cohesive" in normalized:
            return "cohesive_energy"
        if "force" in normalized:
            return "force"
        if "virial" in normalized:
            return "virial"
        return "energy"

    def _merge_profile(self, base: dict, override: dict) -> dict:
        merged = {}
        for key, value in base.items():
            if isinstance(value, dict):
                merged[key] = dict(value)
            elif isinstance(value, list):
                merged[key] = list(value)
            else:
                merged[key] = value
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                nested = dict(merged[key])
                nested.update(value)
                merged[key] = nested
            else:
                merged[key] = value
        return merged

    def _get_parity_profile(self, label: str, *, enhanced: bool = False) -> dict:
        quantity = self._resolve_parity_quantity(label)
        quantity_override = self.parity_profile["quantity_overrides"].get(quantity, {})
        merged_profile = self._merge_profile(self.parity_profile, quantity_override)
        if not enhanced:
            return merged_profile
        enhanced_profile = dict(merged_profile["enhanced"])
        enhanced_profile["main_tick_count"] = merged_profile["main_tick_count"]
        enhanced_profile["scatter_color"] = merged_profile["scatter_color"]
        enhanced_profile["identity_color"] = merged_profile["identity_color"]
        enhanced_profile["identity_alpha"] = merged_profile["identity_alpha"]
        enhanced_profile["identity_linewidth"] = merged_profile["identity_linewidth"]
        enhanced_profile["scientific_enabled"] = enhanced_profile.get(
            "scientific_enabled",
            merged_profile.get("scientific_enabled", True),
        )
        enhanced_profile["scientific_powerlimits"] = enhanced_profile.get(
            "scientific_powerlimits",
            merged_profile["scientific_powerlimits"],
        )
        enhanced_profile["side_panels_enabled"] = enhanced_profile.get(
            "side_panels_enabled",
            True,
        )
        enhanced_profile["error_inset_enabled"] = enhanced_profile.get(
            "error_inset_enabled",
            True,
        )
        enhanced_profile["renderer_policy"] = enhanced_profile.get("renderer_policy", "auto")
        renderer_override = str(self.enhanced_parity_renderer).lower()
        if renderer_override not in {"auto", "scatter", "hexbin"}:
            renderer_override = "auto"
        renderer_default = enhanced_profile.get(
            "renderer_default",
            enhanced_profile.get("main_density_mode", "scatter"),
        )
        enhanced_profile["renderer_default"] = renderer_default
        enhanced_profile["renderer_policy"] = renderer_override
        enhanced_profile["main_density_mode"] = (
            renderer_default if renderer_override == "auto" else renderer_override
        )
        if enhanced_profile["main_density_mode"] == "hexbin":
            enhanced_profile["main_density_norm"] = "log"
        enhanced_profile["colorbar_enabled"] = bool(
            enhanced_profile.get("colorbar_enabled", False)
            and enhanced_profile["main_density_mode"] == "hexbin"
        )
        return enhanced_profile

    def _resolve_axis_limits(
        self, y_true_valid: np.ndarray, y_pred_valid: np.ndarray
    ) -> tuple[float, float]:
        vmin = min(y_true_valid.min(), y_pred_valid.min())
        vmax = max(y_true_valid.max(), y_pred_valid.max())
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            raise ValueError("Axis bounds are non-finite.")
        if vmin == vmax:
            delta = max(abs(vmin) * 0.05, 1e-6)
            vmin -= delta
            vmax += delta
        margin = (vmax - vmin) * 0.05
        return vmin - margin, vmax + margin

    def _apply_scalar_formatter(
        self,
        ax,
        *,
        axis: str,
        powerlimits: tuple[int, int],
        scientific_enabled: bool = True,
        suppress_offset_text: bool = False,
        plain_decimal_places: int = 2,
    ):
        axis_obj = getattr(ax, f"{axis}axis")
        if suppress_offset_text and not scientific_enabled:
            axis_obj.set_major_formatter(
                FormatStrFormatter(f"%.{plain_decimal_places}f")
            )
            axis_obj.offsetText.set_visible(False)
            return
        formatter = ScalarFormatter(useOffset=False)
        formatter.set_scientific(scientific_enabled)
        if scientific_enabled:
            formatter.set_powerlimits(powerlimits)
        axis_obj.set_major_formatter(formatter)
        if suppress_offset_text:
            axis_obj.offsetText.set_visible(False)

    def _plot_parity_main_layer(
        self,
        ax,
        y_true_valid: np.ndarray,
        y_pred_valid: np.ndarray,
        profile: dict,
    ):
        density_mode = profile.get("main_density_mode", "scatter")
        if density_mode == "hexbin":
            density_artist = ax.hexbin(
                y_true_valid,
                y_pred_valid,
                gridsize=profile.get("main_density_gridsize", 60),
                mincnt=profile.get("main_density_mincnt", 1),
                cmap=profile.get("main_density_cmap", "Blues"),
                linewidths=profile.get("main_density_linewidths", 0.0),
                rasterized=True,
            )
            density_norm = self._build_density_norm(profile, density_artist)
            if density_norm is not None:
                density_artist.set_norm(density_norm)
            return density_artist
        ax.scatter(
            y_true_valid,
            y_pred_valid,
            alpha=profile["main_scatter_alpha"],
            s=profile["main_scatter_size"],
            c=profile["scatter_color"],
            edgecolors="none",
            rasterized=True,
        )
        return None

    def _build_density_norm(self, profile: dict, density_artist):
        norm_name = str(profile.get("main_density_norm", "linear")).lower()
        density_values = density_artist.get_array()
        if density_values is None or len(density_values) == 0:
            return None
        vmax = float(np.nanmax(density_values))
        if not np.isfinite(vmax) or vmax <= 0:
            return None
        if norm_name == "log":
            positive_values = density_values[density_values > 0]
            if positive_values.size == 0:
                return None
            vmin = float(np.nanmin(positive_values))
            return LogNorm(vmin=max(vmin, 1.0), vmax=vmax)
        if norm_name == "power":
            gamma = float(profile.get("main_density_norm_gamma", 0.8))
            return PowerNorm(gamma=gamma, vmin=0, vmax=vmax)
        return Normalize(vmin=0, vmax=vmax)

    def _create_enhanced_parity_axes(self, fig, profile: dict):
        padding = profile.get(
            "layout_padding",
            {"left": 0.09, "right": 0.97, "top": 0.91, "bottom": 0.11},
        )
        ax_top = None
        ax_right = None
        ax_err = None
        ax_cbar = None
        if profile["side_panels_enabled"]:
            gs = fig.add_gridspec(
                2,
                2,
                left=padding["left"],
                right=padding["right"],
                top=padding["top"],
                bottom=padding["bottom"],
                width_ratios=profile["width_ratios"],
                height_ratios=profile["height_ratios"],
                wspace=profile["wspace"],
                hspace=profile["hspace"],
            )
            ax_top = fig.add_subplot(gs[0, 0])
            ax_main = fig.add_subplot(gs[1, 0], sharex=ax_top)
            ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
            if profile["error_inset_enabled"]:
                ax_err = fig.add_subplot(gs[0, 1])
            return ax_main, ax_top, ax_right, ax_err, ax_cbar
        if profile["error_inset_enabled"]:
            gs = fig.add_gridspec(
                1,
                2,
                left=padding["left"],
                right=padding["right"],
                top=padding["top"],
                bottom=padding["bottom"],
                width_ratios=profile.get("hexbin_width_ratios", [1.0, 0.24]),
                wspace=profile.get("hexbin_wspace", profile["wspace"]),
            )
            ax_main = fig.add_subplot(gs[0, 0])
            sidebar = gs[0, 1].subgridspec(
                2,
                1,
                height_ratios=profile.get(
                    "hexbin_sidebar_height_ratios", [0.42, 0.58]
                ),
                hspace=profile.get("hexbin_sidebar_hspace", 0.34),
            )
            ax_err = fig.add_subplot(sidebar[0, 0])
            if profile.get("colorbar_enabled", False):
                ax_cbar = fig.add_subplot(sidebar[1, 0])
            return ax_main, ax_top, ax_right, ax_err, ax_cbar
        gs = fig.add_gridspec(
            1,
            1,
            left=padding["left"],
            right=padding["right"],
            top=padding["top"],
            bottom=padding["bottom"],
        )
        ax_main = fig.add_subplot(gs[0, 0])
        return ax_main, ax_top, ax_right, ax_err, ax_cbar

    def _format_error_axis_label(self, unit: str, profile: dict) -> str:
        template = str(profile.get("error_panel_xlabel_template", "Error ({unit})"))
        try:
            return template.format(unit=unit)
        except (KeyError, IndexError, ValueError):
            return f"Error ({unit})"

    def _align_scatter_enhanced_axes(self, fig, ax_main, ax_top, ax_right, ax_err):
        if ax_top is None or ax_right is None:
            return
        fig.canvas.draw()
        main_box = ax_main.get_position()
        top_box = ax_top.get_position()
        right_box = ax_right.get_position()
        right_gap = float(getattr(ax_main, "_dpeva_aligned_sidebar_gap", 0.022))
        width_scale = float(getattr(ax_main, "_dpeva_scatter_sidebar_width_scale", 0.2))
        desired_width = max(top_box.height, main_box.width * width_scale)
        right_x1 = min(right_box.x1, main_box.x1 + right_gap + desired_width)
        right_x0 = min(main_box.x1 + right_gap, right_x1 - 0.05)
        right_width = max(0.01, min(desired_width, right_x1 - right_x0))
        right_x0 = right_x1 - right_width
        right_height = main_box.height
        right_bottom_y0 = main_box.y0
        ax_top.set_position(
            [main_box.x0, top_box.y0, main_box.width, top_box.height]
        )
        ax_right.set_position(
            [right_x0, right_bottom_y0, right_width, right_height]
        )
        if ax_err is not None:
            ax_err.set_position(
                [right_x0, top_box.y0, right_width, top_box.height]
            )

    def _align_hexbin_sidebar_axes(self, fig, ax_main, ax_err, ax_cbar, profile: dict):
        if ax_err is None:
            return
        fig.canvas.draw()
        main_box = ax_main.get_position()
        err_box = ax_err.get_position()
        gap = float(profile.get("hexbin_aligned_sidebar_gap", 0.02))
        right_x1 = ax_cbar.get_position().x1 if ax_cbar is not None else err_box.x1
        right_x0 = min(main_box.x1 + gap, right_x1 - 0.05)
        new_width = right_x1 - right_x0
        
        if ax_cbar is not None:
            height_ratios = profile.get("hexbin_sidebar_height_ratios", [0.35, 0.65])
            hspace = profile.get("hexbin_sidebar_hspace", 0.25)
            r0, r1 = height_ratios[0], height_ratios[1]
            total_r = r0 + r1
            base_h = main_box.height / (total_r + hspace * total_r / 2)
            
            h1 = r0 * base_h
            h2 = r1 * base_h
            
            err_y0 = main_box.y1 - h1
            cbar_y0 = main_box.y0
            
            ax_err.set_position([right_x0, err_y0, new_width, h1])
            
            cbar_width = new_width * float(profile.get("hexbin_colorbar_width_ratio", 1.0))
            cbar_width = min(max(cbar_width, 0.018), new_width)
            align = str(profile.get("hexbin_colorbar_align", "center")).lower()
            if align == "right":
                cbar_x0 = right_x0 + new_width - cbar_width
            elif align == "left":
                cbar_x0 = right_x0
            else:
                cbar_x0 = right_x0 + (new_width - cbar_width) / 2
            ax_cbar.set_position([cbar_x0, cbar_y0, cbar_width, h2])
        else:
            ax_err.set_position([right_x0, main_box.y1 - err_box.height, new_width, err_box.height])

    def plot_parity(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        label: str,
        unit: str,
        title: str = None,
    ):
        """
        Plot Parity (Diagonal) plot: Predicted vs Ground Truth.

        Args:
            y_true (np.ndarray): Ground truth values.
            y_pred (np.ndarray): Predicted values.
            label (str): Label for the quantity (e.g. "Energy").
            unit (str): Unit string (e.g. "eV/atom").
            title (str, optional): Custom title. Defaults to None.
        """
        y_true_valid, y_pred_valid = self._filter_finite_pair(
            y_true, y_pred, label, "Parity plot"
        )
        if y_true_valid is None:
            return
        profile = self._get_parity_profile(label)
        fonts = profile["fonts"]
        try:
            vmin, vmax = self._resolve_axis_limits(y_true_valid, y_pred_valid)
        except ValueError:
            self.logger.warning(
                f"Parity plot skipped for {label}: axis bounds are non-finite."
            )
            return
        fig, ax = plt.subplots(figsize=profile["figure_size"])
        ax.plot(
            [vmin, vmax],
            [vmin, vmax],
            linestyle="--",
            lw=profile["identity_linewidth"],
            color=profile["identity_color"],
            alpha=profile["identity_alpha"],
        )
        ax.scatter(
            y_true_valid,
            y_pred_valid,
            alpha=profile["scatter_alpha"],
            s=profile["scatter_size"],
            c=profile["scatter_color"],
            edgecolors="none",
            rasterized=True,
        )
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(vmin, vmax)
        ax.set_aspect("equal", adjustable="box")
        self._apply_axis_fonts(
            ax,
            title=title or f"{label} Parity Plot",
            xlabel=f"True {label} ({unit})",
            ylabel=f"Predicted {label} ({unit})",
            fonts=fonts,
        )
        ax.grid(True, alpha=profile["grid_alpha"])
        ax.xaxis.set_major_locator(MaxNLocator(profile["main_tick_count"]))
        ax.yaxis.set_major_locator(MaxNLocator(profile["main_tick_count"]))
        self._apply_scalar_formatter(
            ax,
            axis="x",
            powerlimits=profile["scientific_powerlimits"],
            scientific_enabled=profile.get("scientific_enabled", True),
        )
        self._apply_scalar_formatter(
            ax,
            axis="y",
            powerlimits=profile["scientific_powerlimits"],
            scientific_enabled=profile.get("scientific_enabled", True),
        )
        
        err_values = y_pred_valid - y_true_valid
        max_err = np.max(np.abs(err_values))
        p99_err = np.percentile(np.abs(err_values), 99)
        mae = np.mean(np.abs(err_values))
        rmse = np.sqrt(np.mean(err_values**2))
        stats_text = (
            f"Max Err: {max_err:.4f}\n"
            f"99% Err: {p99_err:.4f}\n"
            f"MAE: {mae:.4f}\n"
            f"RMSE: {rmse:.4f}"
        )
        self._add_stats_box(ax, stats_text, fontsize=fonts["label"])
        
        fig.tight_layout()
        filename = f"parity_{label.lower().replace(' ', '_')}.png"
        fig.savefig(
            os.path.join(self.output_dir, filename), dpi=self.dpi
        )
        plt.close(fig)

    def plot_parity_enhanced(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        label: str,
        unit: str,
        title: str = None,
    ):
        """Plot parity with marginal distributions for true and predicted values."""
        y_true_valid, y_pred_valid = self._filter_finite_pair(
            y_true, y_pred, label, "Enhanced parity plot"
        )
        if y_true_valid is None:
            return
        try:
            vmin, vmax = self._resolve_axis_limits(y_true_valid, y_pred_valid)
        except ValueError:
            self.logger.warning(
                f"Enhanced parity plot skipped for {label}: axis bounds are non-finite."
            )
            return
        profile = self._get_parity_profile(label, enhanced=True)
        fonts = profile["fonts"]
        panel_fonts = profile["panel_fonts"]
        colors = profile["colors"]
        density_tick_count = profile.get("density_tick_count", profile["panel_tick_count"])

        fig = plt.figure(figsize=profile["figure_size"])
        ax_main, ax_top, ax_right, ax_err, ax_cbar = self._create_enhanced_parity_axes(
            fig, profile
        )
        setattr(
            ax_main,
            "_dpeva_aligned_sidebar_gap",
            float(profile.get("aligned_sidebar_gap", 0.022)),
        )
        setattr(
            ax_main,
            "_dpeva_scatter_sidebar_width_scale",
            float(profile.get("scatter_sidebar_width_scale", 0.2)),
        )

        ax_main.plot(
            [vmin, vmax],
            [vmin, vmax],
            linestyle="--",
            lw=profile["identity_linewidth"],
            color=profile["identity_color"],
            alpha=profile["identity_alpha"],
        )
        density_artist = self._plot_parity_main_layer(
            ax_main, y_true_valid, y_pred_valid, profile
        )
        ax_main.set_xlim(vmin, vmax)
        ax_main.set_ylim(vmin, vmax)
        ax_main.set_aspect("equal", adjustable="box")
        self._apply_axis_fonts(
            ax_main,
            xlabel=f"True {label} ({unit})",
            ylabel=f"Predicted {label} ({unit})",
            fonts=fonts,
        )
        ax_main.grid(True, alpha=profile["grid_alpha"])
        ax_main.xaxis.set_major_locator(MaxNLocator(profile["main_tick_count"]))
        ax_main.yaxis.set_major_locator(MaxNLocator(profile["main_tick_count"]))
        self._apply_scalar_formatter(
            ax_main,
            axis="x",
            powerlimits=profile["scientific_powerlimits"],
            scientific_enabled=profile.get("scientific_enabled", True),
        )
        self._apply_scalar_formatter(
            ax_main,
            axis="y",
            powerlimits=profile["scientific_powerlimits"],
            scientific_enabled=profile.get("scientific_enabled", True),
        )
        
        err_values = y_pred_valid - y_true_valid
        
        max_err = np.max(np.abs(err_values))
        p99_err = np.percentile(np.abs(err_values), 99)
        mae = np.mean(np.abs(err_values))
        rmse = np.sqrt(np.mean(err_values**2))
        stats_text = (
            f"Max Err: {max_err:.4f}\n"
            f"99% Err: {p99_err:.4f}\n"
            f"MAE: {mae:.4f}\n"
            f"RMSE: {rmse:.4f}"
        )
        self._add_stats_box(ax_main, stats_text, fontsize=fonts["label"])
        
        if ax_top is not None:
            self._plot_histogram_with_kde(
                ax_top,
                y_true_valid,
                color=colors["true"],
                alpha=profile["panel_hist_alpha"],
            )
            self._apply_axis_fonts(
                ax_top,
                ylabel=profile.get("top_panel_ylabel", "True Density"),
                fonts=panel_fonts,
            )
            ax_top.yaxis.label.set_color(colors["true"])
            ax_top.yaxis.label.set_fontsize(panel_fonts["title"])
            ax_top.yaxis.labelpad = profile.get("top_panel_ylabel_pad", 7.0)
            ax_top.grid(True, alpha=profile["grid_alpha"])
            ax_top.tick_params(
                axis="x",
                labelbottom=False,
                bottom=False,
                labeltop=False,
                top=False,
            )
            ax_top.tick_params(axis="y", length=2.8, pad=1.5)
            ax_top.xaxis.set_major_locator(MaxNLocator(4))
            ax_top.yaxis.set_major_locator(MaxNLocator(density_tick_count))
            self._apply_scalar_formatter(
                ax_top,
                axis="y",
                powerlimits=profile["scientific_powerlimits"],
                scientific_enabled=profile.get("scientific_enabled", True),
                suppress_offset_text=True,
            )

        if ax_right is not None:
            self._plot_histogram_with_kde(
                ax_right,
                y_pred_valid,
                color=colors["pred"],
                orientation="horizontal",
                alpha=profile["panel_hist_alpha"],
            )
            self._apply_axis_fonts(
                ax_right,
                xlabel=profile.get("right_panel_xlabel", profile["right_panel_title"]),
                fonts=panel_fonts,
            )
            ax_right.xaxis.label.set_color(colors["pred"])
            ax_right.xaxis.label.set_fontsize(panel_fonts["title"])
            ax_right.xaxis.labelpad = profile.get("right_panel_xlabel_pad", 4.0)
            ax_right.grid(True, alpha=profile["grid_alpha"])
            ax_right.tick_params(axis="y", labelleft=False, left=False)
            ax_right.tick_params(axis="x", length=2.8)
            ax_right.xaxis.set_major_locator(MaxNLocator(density_tick_count))
            self._apply_scalar_formatter(
                ax_right,
                axis="x",
                powerlimits=profile["scientific_powerlimits"],
                scientific_enabled=profile.get("scientific_enabled", True),
                suppress_offset_text=True,
            )

        if ax_err is not None:
            error_xlabel = self._format_error_axis_label(unit, profile)
            is_hexbin_sidebar = ax_top is None and ax_right is None
            
            p_high = np.percentile(np.abs(err_values), 99)
            if p_high > 0:
                err_limit = p_high * 1.5
                visible_err_values = err_values[(err_values >= -err_limit) & (err_values <= err_limit)]
            else:
                visible_err_values = err_values

            self._plot_histogram_with_kde(
                ax_err,
                visible_err_values,
                color=colors["error"],
                orientation="vertical",
                alpha=profile["error_hist_alpha"],
            )
            
            if p_high > 0:
                ax_err.set_xlim(-p_high * 1.5, p_high * 1.5)

            ax_err.axvline(
                0.0,
                color=profile["identity_color"],
                linestyle="--",
                linewidth=profile["error_zero_linewidth"],
            )
            self._apply_axis_fonts(
                ax_err,
                xlabel=(
                    (error_xlabel or profile.get("error_panel_title", "Error"))
                    if is_hexbin_sidebar
                    else error_xlabel
                ),
                ylabel=profile.get("error_panel_ylabel", "Error Density"),
                fonts=panel_fonts,
            )
            ax_err.grid(True, alpha=profile["grid_alpha"])
            if is_hexbin_sidebar:
                ax_err.set_ylabel("")
                ax_err.xaxis.label.set_fontsize(panel_fonts["title"])
                ax_err.xaxis.labelpad = profile.get("error_panel_xlabel_pad", 1.5)
                ax_err.xaxis.set_major_locator(MaxNLocator(profile["error_tick_count"]))
                self._apply_scalar_formatter(
                    ax_err,
                    axis="x",
                    powerlimits=profile["scientific_powerlimits"],
                    scientific_enabled=profile.get("scientific_enabled", True),
                    suppress_offset_text=True,
                )
                ax_err.tick_params(axis="x", labelsize=panel_fonts["tick"], pad=1.5)
                ax_err.tick_params(axis="y", left=False, labelleft=False, length=0)
            else:
                ax_err.xaxis.set_label_position("top")
                ax_err.xaxis.labelpad = (
                    profile.get("error_panel_xlabel_pad", 1.5) if error_xlabel else 0.0
                )
                ax_err.yaxis.label.set_fontsize(panel_fonts["title"])
                ax_err.yaxis.labelpad = profile.get("error_panel_ylabel_pad", 7.0)
                ax_err.xaxis.set_major_locator(MaxNLocator(profile["error_tick_count"]))
                ax_err.yaxis.set_major_locator(MaxNLocator(density_tick_count))
                self._apply_scalar_formatter(
                    ax_err,
                    axis="x",
                    powerlimits=profile["scientific_powerlimits"],
                    scientific_enabled=profile.get("scientific_enabled", True),
                    suppress_offset_text=True,
                )
                ax_err.xaxis.tick_top()
                ax_err.tick_params(
                    axis="x",
                    labelsize=panel_fonts["tick"],
                    pad=profile.get("error_tick_pad", -3.0),
                    labelbottom=False,
                )
                ax_err.tick_params(axis="y", labelleft=False, left=False, length=0)

        if ax_cbar is not None and density_artist is not None:
            colorbar_orientation = profile.get("colorbar_orientation", "horizontal")
            colorbar = fig.colorbar(
                density_artist,
                cax=ax_cbar,
                orientation=colorbar_orientation,
            )
            colorbar.set_label("")
            if colorbar_orientation == "vertical":
                colorbar.ax.set_xlabel("")
                colorbar.ax.set_ylabel(
                    profile.get("colorbar_title", "Counts Per Hexbin"),
                    fontsize=panel_fonts["title"],
                    labelpad=profile.get("colorbar_label_pad", 6.0),
                )
                colorbar.ax.yaxis.set_label_position("left")
                colorbar.ax.tick_params(
                    axis="x", bottom=False, labelbottom=False, length=0
                )
                colorbar.ax.tick_params(
                    axis="y", labelsize=panel_fonts["tick"], pad=1.5
                )
            else:
                colorbar.ax.set_ylabel("")
                colorbar.ax.set_xlabel(
                    profile.get("colorbar_title", "Sample Density"),
                    fontsize=panel_fonts["title"],
                    labelpad=profile.get("colorbar_label_pad", 6.0),
                )
                colorbar.ax.tick_params(axis="x", labelsize=panel_fonts["tick"], pad=1.5)
                colorbar.ax.tick_params(axis="y", left=False, labelleft=False, length=0)
            if isinstance(density_artist.norm, LogNorm):
                colorbar.locator = LogLocator(base=10)
                colorbar.formatter = LogFormatterSciNotation(base=10)
                colorbar.update_ticks()
            else:
                locator = MaxNLocator(profile.get("colorbar_tick_count", 4))
                if colorbar_orientation == "vertical":
                    colorbar.ax.yaxis.set_major_locator(locator)
                else:
                    colorbar.ax.xaxis.set_major_locator(locator)

        if ax_top is not None and ax_right is not None:
            self._align_scatter_enhanced_axes(fig, ax_main, ax_top, ax_right, ax_err)
        elif ax_err is not None:
            self._align_hexbin_sidebar_axes(fig, ax_main, ax_err, ax_cbar, profile)

        if title:
            fig.suptitle(title, y=profile["suptitle_y"], fontsize=fonts["title"], fontweight="bold")
        else:
            fig.suptitle(
                profile["main_title"].format(label=label),
                y=profile["suptitle_y"],
                fontsize=fonts["title"],
                fontweight="bold",
            )

        filename = f"parity_{label.lower().replace(' ', '_')}_enhanced.png"
        fig.savefig(
            os.path.join(self.output_dir, filename), dpi=self.dpi
        )
        plt.close(fig)

    def plot_distribution(
        self,
        data: np.ndarray,
        label: str,
        unit: str,
        color: str = "blue",
        title: str = None,
        highlight_outliers: bool = False,
        outlier_mask: Optional[np.ndarray] = None,
        show_legend: bool = False,
        show_stats: bool = True,
    ):
        """
        Plot KDE distribution of data.

        Args:
            data (np.ndarray): Data to plot.
            label (str): Label for the quantity.
            unit (str): Unit string.
            color (str, optional): Plot color. Defaults to 'blue'.
            title (str, optional): Custom title. Defaults to None.
            highlight_outliers (bool, optional): Whether to highlight outliers. Defaults to False.
            outlier_mask (np.ndarray, optional): Boolean mask for outliers. Defaults to None.
        """
        filtered_data = self._filter_finite_1d(data, label, "Distribution plot")
        if filtered_data is None:
            return
        profile = self.distribution_profile
        fonts = profile["fonts"]
        fig, ax = plt.subplots(figsize=profile["figure_size"])
        self._plot_histogram_with_kde(ax, filtered_data, color=color)

        if highlight_outliers and outlier_mask is not None:
            outlier_mask_arr = np.asarray(outlier_mask, dtype=bool).reshape(-1)
            if outlier_mask_arr.size == filtered_data.size and np.any(outlier_mask_arr):
                clean_data = filtered_data[~outlier_mask_arr]
                if clean_data.size > 0:
                    sns.kdeplot(
                        clean_data,
                        color="green",
                        linestyle="--",
                        linewidth=profile["step_linewidth"],
                        bw_adjust=profile["bw_adjust"],
                        cut=0,
                        label="Clean Data (No Outliers)",
                        ax=ax,
                    )
            elif outlier_mask_arr.size != filtered_data.size:
                self.logger.warning(
                    f"Distribution plot for {label}: outlier mask size mismatch "
                    f"({outlier_mask_arr.size} vs {filtered_data.size}), skip outlier highlighting."
                )

        if title:
            plot_title = title
        else:
            plot_title = f"{label} Distribution"
        self._apply_axis_fonts(
            ax,
            title=plot_title,
            xlabel=f"{label} ({unit})",
            ylabel="Density",
            fonts=fonts,
        )
        if show_stats:
            hist, _ = np.histogram(filtered_data, bins=10)
            mid_idx = len(hist) // 2
            left_density = np.max(hist[:mid_idx]) if len(hist[:mid_idx]) > 0 else 0
            right_density = np.max(hist[mid_idx:]) if len(hist[mid_idx:]) > 0 else 0
            
            if left_density > right_density:
                stats_x = 0.98
                stats_ha = "right"
            else:
                stats_x = 0.02
                stats_ha = "left"
            
            stats_fontsize = fonts.get("legend", 12) * 1.15
            self._add_stats_box(
                ax, self._stats_text(filtered_data, label, include_name=False), x=stats_x, y=0.98, fontsize=stats_fontsize, ha=stats_ha
            )

        if show_legend:
            ax.legend(
                loc="upper right", frameon=True, framealpha=profile["legend_framealpha"]
            )
        ax.grid(True, alpha=profile["grid_alpha"])
        fig.tight_layout()

        filename = f"dist_{label.lower().replace(' ', '_')}.png"
        fig.savefig(os.path.join(self.output_dir, filename), dpi=self.dpi)
        plt.close(fig)

    def plot_distribution_overlay(
        self,
        pred_data: np.ndarray,
        true_data: np.ndarray,
        label: str,
        unit: str,
        pred_label: str = "Predicted",
        true_label: str = "True",
        pred_color: str = "#2563eb",
        true_color: str = "#ef4444",
        show_stats: bool = True,
    ):
        """Plot predicted/true distributions in one panel for direct comparison."""
        pred_values = self._filter_finite_1d(
            pred_data, f"{pred_label} {label}", "Overlay distribution plot"
        )
        true_values = self._filter_finite_1d(
            true_data, f"{true_label} {label}", "Overlay distribution plot"
        )
        if pred_values is None or true_values is None:
            return
        profile = self.distribution_profile
        fonts = profile["fonts"]
        fig, ax = plt.subplots(figsize=profile["figure_size"])
        self._plot_histogram_with_kde(
            ax, pred_values, color=pred_color, label=pred_label
        )
        self._plot_histogram_with_kde(
            ax, true_values, color=true_color, label=true_label
        )
        self._apply_axis_fonts(
            ax,
            title=f"{label} Distribution Overlay",
            xlabel=f"{label} ({unit})",
            ylabel="Density",
            fonts=fonts,
        )
        ax.grid(True, alpha=profile["grid_alpha"])
        legend_kwargs = {
            "loc": profile["overlay_legend_loc"],
            "frameon": True,
            "framealpha": profile["legend_framealpha"],
        }
        if profile["overlay_legend_outside"]:
            legend_kwargs["bbox_to_anchor"] = profile["legend_bbox_anchor"]
            legend_kwargs["borderaxespad"] = 0.0
        ax.legend(**legend_kwargs)
        if show_stats:
            stats = (
                self._stats_text(pred_values, pred_label)
                + "\n\n"
                + self._stats_text(true_values, true_label)
            )
            self._add_stats_box(
                ax, stats, x=profile["stats_anchor_x"], y=profile["stats_anchor_y"]
            )
        fig.tight_layout(rect=[0.0, 0.0, profile["main_panel_right_margin"], 1.0])
        filename = f"dist_{label.lower().replace(' ', '_')}_overlay.png"
        save_kwargs = {"dpi": self.dpi}
        if profile["overlay_legend_outside"]:
            save_kwargs["bbox_inches"] = profile["overlay_save_bbox_inches"]
        fig.savefig(
            os.path.join(self.output_dir, filename),
            **save_kwargs,
        )
        plt.close(fig)

    def plot_distribution_with_error(
        self,
        pred_data: np.ndarray,
        true_data: np.ndarray,
        error_data: np.ndarray,
        label: str,
        unit: str,
        pred_label: str = "Predicted",
        true_label: str = "True",
        pred_color: str = "#2563eb",
        true_color: str = "#ef4444",
        error_color: str = "#f59e0b",
        show_stats: bool = True,
    ):
        """Plot distribution comparison with a compact side error-distribution panel."""
        pred_values = self._filter_finite_1d(
            pred_data, f"{pred_label} {label}", "Distribution-with-error plot"
        )
        true_values = self._filter_finite_1d(
            true_data, f"{true_label} {label}", "Distribution-with-error plot"
        )
        err_values = self._filter_finite_1d(
            error_data, f"{label} Error", "Distribution-with-error plot"
        )
        if pred_values is None or true_values is None or err_values is None:
            return

        profile = self.distribution_profile
        fonts = profile["fonts"]
        fig = plt.figure(figsize=profile["comparison_figure_size"])
        gs = gridspec.GridSpec(
            1,
            2,
            width_ratios=profile["with_error_width_ratios"],
            wspace=profile["with_error_wspace"],
        )
        ax_main = fig.add_subplot(gs[0, 0])
        ax_err = fig.add_subplot(gs[0, 1])

        self._plot_histogram_with_kde(
            ax_main, pred_values, color=pred_color, label=pred_label
        )
        self._plot_histogram_with_kde(
            ax_main, true_values, color=true_color, label=true_label
        )
        self._apply_axis_fonts(
            ax_main,
            title=f"{label} Distribution with Error",
            xlabel=f"{label} ({unit})",
            ylabel="Density",
            fonts=fonts,
        )
        ax_main.grid(True, alpha=profile["grid_alpha"])
        ax_main.legend(
            loc="upper left", frameon=True, framealpha=profile["legend_framealpha"]
        )
        if show_stats:
            stats = (
                self._stats_text(pred_values, pred_label)
                + "\n\n"
                + self._stats_text(true_values, true_label)
            )
            self._add_stats_box(
                ax_main,
                stats,
                x=profile["with_error_stats_anchor_x"],
                y=profile["with_error_stats_anchor_y"],
            )

        self._plot_histogram_with_kde(ax_err, err_values, color=error_color)
        ax_err.axvline(
            0.0,
            color="#374151",
            linestyle="--",
            linewidth=profile["with_error_zero_linewidth"],
        )
        self._apply_axis_fonts(
            ax_err,
            title="Error Density",
            xlabel=f"Error ({unit})",
            ylabel="Density",
            fonts=fonts,
        )
        ax_err.grid(True, alpha=profile["grid_alpha"])
        fig.subplots_adjust(
            left=profile["with_error_layout"]["left"],
            right=profile["with_error_layout"]["right"],
            top=profile["with_error_layout"]["top"],
            bottom=profile["with_error_layout"]["bottom"],
            wspace=profile["with_error_wspace"],
        )
        filename = f"dist_{label.lower().replace(' ', '_')}_with_error.png"
        fig.savefig(os.path.join(self.output_dir, filename), dpi=self.dpi)
        plt.close(fig)

    def plot_error_distribution(self, error: np.ndarray, label: str, unit: str):
        """
        Plot error distribution (Predicted - True).

        Args:
            error (np.ndarray): Error values.
            label (str): Label for the quantity.
            unit (str): Unit string.
        """
        filtered_error = self._filter_finite_1d(
            error, f"{label} Error", "Error distribution plot"
        )
        if filtered_error is None:
            return
        profile = self.distribution_profile
        fonts = profile["fonts"]
        fig, ax = plt.subplots(figsize=profile["figure_size"])
        self._plot_histogram_with_kde(ax, filtered_error, color="red")
        ax.axvline(0, color="k", linestyle="--", lw=1)
        self._apply_axis_fonts(
            ax,
            title=f"{label} Error Distribution",
            xlabel=f"{label} Error ({unit})",
            ylabel="Density",
            fonts=fonts,
        )
        ax.grid(True, alpha=profile["grid_alpha"])
        fig.tight_layout()

        filename = f"error_dist_{label.lower().replace(' ', '_')}.png"
        fig.savefig(os.path.join(self.output_dir, filename), dpi=self.dpi)
        plt.close(fig)
