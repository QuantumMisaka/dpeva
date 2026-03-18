import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
from matplotlib import gridspec
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from dpeva.constants import FIG_DPI

class InferenceVisualizer:
    """
    Visualization tools for inference results (Parity Plots, Error Distributions).
    """
    
    def __init__(self, output_dir: str, dpi: int = FIG_DPI):
        """
        Initialize the InferenceVisualizer.

        Args:
            output_dir (str): Directory to save plots.
            dpi (int, optional): Resolution for saved images. Defaults to FIG_DPI.
        """
        self.output_dir = output_dir
        self.dpi = dpi
        self.logger = logging.getLogger(__name__)
        
        # Style settings
        from dpeva.utils.visual_style import set_visual_style
        set_visual_style()
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def _filter_finite_1d(self, data: np.ndarray, label: str, context: str) -> Optional[np.ndarray]:
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

    def _filter_finite_pair(self, x: np.ndarray, y: np.ndarray, label: str, context: str):
        x_arr = np.asarray(x, dtype=float).reshape(-1)
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        if x_arr.size == 0 or y_arr.size == 0:
            self.logger.warning(f"{context} skipped for {label}: empty input.")
            return None, None
        valid_mask = np.isfinite(x_arr) & np.isfinite(y_arr)
        valid_count = int(np.count_nonzero(valid_mask))
        if valid_count == 0:
            self.logger.warning(f"{context} skipped for {label}: no finite paired values.")
            return None, None
        if valid_count < x_arr.size:
            self.logger.warning(
                f"{context} for {label}: filtered non-finite paired values "
                f"({x_arr.size - valid_count}/{x_arr.size})."
            )
        return x_arr[valid_mask], y_arr[valid_mask]

    def _stats_text(self, data: np.ndarray, name: str) -> str:
        values = np.asarray(data, dtype=float).reshape(-1)
        desc = {
            "count": float(values.size),
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
        return (
            f"{name}\n"
            f"count={desc['count']:.0f}\n"
            f"mean={desc['mean']:.4f}\n"
            f"std={desc['std']:.4f}\n"
            f"min={desc['min']:.4f}\n"
            f"max={desc['max']:.4f}"
        )

    def _add_stats_box(self, ax, text: str, x: float = 0.02, y: float = 0.98):
        ax.text(
            x,
            y,
            text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "#999999"},
        )

    def _choose_density_label_anchor(self, values: np.ndarray) -> tuple[float, float, str]:
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.size == 0:
            return 0.04, 0.92, "left"
        bins = int(np.clip(np.sqrt(arr.size), 20, 90))
        hist, edges = np.histogram(arr, bins=bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        y_candidates = np.array([0.90, 0.78, 0.66, 0.54, 0.42, 0.30, 0.18], dtype=float)
        ymin = float(np.min(arr))
        ymax = float(np.max(arr))
        if ymax <= ymin:
            return 0.04, 0.92, "left"
        data_candidates = ymin + y_candidates * (ymax - ymin)
        local_density = np.interp(data_candidates, centers, hist, left=hist[0], right=hist[-1])
        density_scale = float(np.max(local_density))
        if density_scale <= 0:
            idx = 0
        else:
            norm_density = local_density / density_scale
            edge_penalty = 0.14 * np.abs(y_candidates - 0.5)
            scores = norm_density + edge_penalty
            idx = int(np.argmin(scores))
        y_anchor = float(y_candidates[idx])
        local_ratio = float(local_density[idx] / density_scale) if density_scale > 0 else 0.0
        if local_ratio >= 0.55:
            return 0.96, y_anchor, "right"
        return 0.04, y_anchor, "left"

    def plot_parity(self, y_true: np.ndarray, y_pred: np.ndarray, 
                   label: str, unit: str, title: str = None):
        """
        Plot Parity (Diagonal) plot: Predicted vs Ground Truth.

        Args:
            y_true (np.ndarray): Ground truth values.
            y_pred (np.ndarray): Predicted values.
            label (str): Label for the quantity (e.g. "Energy").
            unit (str): Unit string (e.g. "eV/atom").
            title (str, optional): Custom title. Defaults to None.
        """
        y_true_valid, y_pred_valid = self._filter_finite_pair(y_true, y_pred, label, "Parity plot")
        if y_true_valid is None:
            return

        plt.figure(figsize=(6, 6))
        
        # Determine limits
        vmin = min(y_true_valid.min(), y_pred_valid.min())
        vmax = max(y_true_valid.max(), y_pred_valid.max())
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            plt.close()
            self.logger.warning(f"Parity plot skipped for {label}: axis bounds are non-finite.")
            return
        if vmin == vmax:
            delta = max(abs(vmin) * 0.05, 1e-6)
            vmin -= delta
            vmax += delta
        margin = (vmax - vmin) * 0.05
        vmin -= margin
        vmax += margin
        
        plt.plot([vmin, vmax], [vmin, vmax], 'k--', lw=1.5, alpha=0.7)
        
        # Scatter
        # Use rasterized=True for large datasets to keep file size small
        plt.scatter(y_true_valid, y_pred_valid, alpha=0.3, s=10, c='blue', edgecolors='none', rasterized=True)
        
        plt.xlim(vmin, vmax)
        plt.ylim(vmin, vmax)
        plt.xlabel(f"True {label} ({unit})")
        plt.ylabel(f"Predicted {label} ({unit})")
        
        if title:
            plt.title(title)
        else:
            plt.title(f"{label} Parity Plot")
            
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = f"parity_{label.lower().replace(' ', '_')}.png"
        plt.savefig(os.path.join(self.output_dir, filename), dpi=self.dpi)
        plt.close()

    def plot_parity_enhanced(self, y_true: np.ndarray, y_pred: np.ndarray, label: str, unit: str, title: str = None):
        """Plot parity with marginal distributions for true and predicted values."""
        y_true_valid, y_pred_valid = self._filter_finite_pair(y_true, y_pred, label, "Enhanced parity plot")
        if y_true_valid is None:
            return
        vmin = min(y_true_valid.min(), y_pred_valid.min())
        vmax = max(y_true_valid.max(), y_pred_valid.max())
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            self.logger.warning(f"Enhanced parity plot skipped for {label}: axis bounds are non-finite.")
            return
        if vmin == vmax:
            delta = max(abs(vmin) * 0.05, 1e-6)
            vmin -= delta
            vmax += delta
        margin = (vmax - vmin) * 0.05
        vmin -= margin
        vmax += margin

        fig = plt.figure(figsize=(8.6, 8.6))
        gs = gridspec.GridSpec(
            4,
            4,
            figure=fig,
            width_ratios=[1, 1, 1, 1.0],
            height_ratios=[0.9, 1, 1, 1],
            wspace=0.10,
            hspace=0.14,
        )
        ax_top = fig.add_subplot(gs[0, :3])
        ax_main = fig.add_subplot(gs[1:, :3], sharex=ax_top)
        ax_right = fig.add_subplot(gs[1:, 3], sharey=ax_main)
        ax_err = fig.add_subplot(gs[0, 3])

        ax_main.plot([vmin, vmax], [vmin, vmax], linestyle="--", lw=1.4, color="#374151", alpha=0.9)
        ax_main.scatter(y_true_valid, y_pred_valid, alpha=0.28, s=10, c="#2563eb", edgecolors="none", rasterized=True)
        ax_main.set_xlim(vmin, vmax)
        ax_main.set_ylim(vmin, vmax)
        ax_main.set_xlabel(f"True {label} ({unit})")
        ax_main.set_ylabel(f"Predicted {label} ({unit})")
        ax_main.grid(True, alpha=0.25)

        sns.histplot(y_true_valid, kde=True, ax=ax_top, color="#ef4444", alpha=0.28, stat="density", element="step")
        ax_top.set_ylabel("Density")
        ax_top.set_title("True Density", color="#ef4444", fontsize=10, pad=3)
        ax_top.grid(True, alpha=0.2)
        ax_top.tick_params(axis="x", labelbottom=False)

        sns.histplot(y=y_pred_valid, kde=True, ax=ax_right, color="#2563eb", alpha=0.28, stat="density", element="step")
        ax_right.set_xlabel("Density")
        ax_right.grid(True, alpha=0.2)
        ax_right.tick_params(axis="y", labelleft=False)
        _, y_anchor, _ = self._choose_density_label_anchor(y_pred_valid)
        ax_right.text(0.98, y_anchor, "Predicted Density", transform=ax_right.transAxes, ha="right", va="top", color="#2563eb", fontsize=10)

        err_values = y_pred_valid - y_true_valid
        sns.histplot(err_values, kde=True, stat="density", color="#f59e0b", alpha=0.28, element="step", ax=ax_err)
        ax_err.axvline(0.0, color="#374151", linestyle="--", linewidth=1.1)
        ax_err.set_title("Error Distribution", fontsize=9, pad=2)
        ax_err.set_xlabel("")
        ax_err.set_ylabel("")
        ax_err.grid(True, alpha=0.25)
        ax_err.xaxis.set_major_locator(MaxNLocator(3))
        ax_err.yaxis.set_major_locator(MaxNLocator(3))
        ax_err.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax_err.ticklabel_format(axis="x", style="plain", useOffset=False)
        ax_err.xaxis.tick_top()
        ax_err.tick_params(axis="x", labelsize=7.8, pad=1)
        ax_err.tick_params(axis="y", labelleft=False, length=0)

        if title:
            fig.suptitle(title, y=0.99)
        else:
            fig.suptitle(f"{label} Enhanced Parity Plot", y=0.99)

        filename = f"parity_{label.lower().replace(' ', '_')}_enhanced.png"
        fig.savefig(os.path.join(self.output_dir, filename), dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

    def plot_distribution(self, data: np.ndarray, label: str, unit: str, 
                          color: str = 'blue', title: str = None, 
                          highlight_outliers: bool = False, outlier_mask: Optional[np.ndarray] = None,
                          show_legend: bool = False, show_stats: bool = True):
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
        plt.figure(figsize=(8, 6))
        
        # Main KDE
        sns.histplot(
            filtered_data,
            kde=True,
            stat="density",
            color=color,
            alpha=0.28,
            element="step",
            bins="fd",
            kde_kws={"bw_adjust": 0.9},
        )
        
        if highlight_outliers and outlier_mask is not None:
            outlier_mask_arr = np.asarray(outlier_mask, dtype=bool).reshape(-1)
            if outlier_mask_arr.size == filtered_data.size and np.any(outlier_mask_arr):
                clean_data = filtered_data[~outlier_mask_arr]
                if clean_data.size > 0:
                    sns.kdeplot(clean_data, color='green', linestyle='--', label="Clean Data (No Outliers)")
            elif outlier_mask_arr.size != filtered_data.size:
                self.logger.warning(
                    f"Distribution plot for {label}: outlier mask size mismatch "
                    f"({outlier_mask_arr.size} vs {filtered_data.size}), skip outlier highlighting."
                )
            
        plt.xlabel(f"{label} ({unit})")
        plt.ylabel("Density")
        
        if title:
            plt.title(title)
        else:
            plt.title(f"{label} Distribution")
        if show_stats:
            self._add_stats_box(plt.gca(), self._stats_text(filtered_data, label), x=0.70, y=0.98)
            
        if show_legend:
            plt.legend(loc="upper right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = f"dist_{label.lower().replace(' ', '_')}.png"
        plt.savefig(os.path.join(self.output_dir, filename), dpi=self.dpi)
        plt.close()

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
        pred_values = self._filter_finite_1d(pred_data, f"{pred_label} {label}", "Overlay distribution plot")
        true_values = self._filter_finite_1d(true_data, f"{true_label} {label}", "Overlay distribution plot")
        if pred_values is None or true_values is None:
            return
        fig, ax = plt.subplots(figsize=(8.6, 6.2))
        sns.histplot(pred_values, kde=True, stat="density", color=pred_color, alpha=0.25, label=pred_label, element="step", ax=ax)
        sns.histplot(true_values, kde=True, stat="density", color=true_color, alpha=0.25, label=true_label, element="step", ax=ax)
        ax.set_xlabel(f"{label} ({unit})")
        ax.set_ylabel("Density")
        ax.set_title(f"{label} Distribution Overlay")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, frameon=True)
        if show_stats:
            stats = self._stats_text(pred_values, pred_label) + "\n\n" + self._stats_text(true_values, true_label)
            self._add_stats_box(ax, stats, x=1.02, y=0.70)
        fig.tight_layout(rect=[0.0, 0.0, 0.84, 1.0])
        filename = f"dist_{label.lower().replace(' ', '_')}_overlay.png"
        fig.savefig(os.path.join(self.output_dir, filename), dpi=self.dpi, bbox_inches="tight")
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
        pred_values = self._filter_finite_1d(pred_data, f"{pred_label} {label}", "Distribution-with-error plot")
        true_values = self._filter_finite_1d(true_data, f"{true_label} {label}", "Distribution-with-error plot")
        err_values = self._filter_finite_1d(error_data, f"{label} Error", "Distribution-with-error plot")
        if pred_values is None or true_values is None or err_values is None:
            return

        fig = plt.figure(figsize=(11.2, 5.8))
        gs = gridspec.GridSpec(1, 2, width_ratios=[3.8, 1.5], wspace=0.34)
        ax_main = fig.add_subplot(gs[0, 0])
        ax_err = fig.add_subplot(gs[0, 1])

        sns.histplot(pred_values, kde=True, stat="density", color=pred_color, alpha=0.24, label=pred_label, element="step", ax=ax_main)
        sns.histplot(true_values, kde=True, stat="density", color=true_color, alpha=0.24, label=true_label, element="step", ax=ax_main)
        ax_main.set_xlabel(f"{label} ({unit})")
        ax_main.set_ylabel("Density")
        ax_main.set_title(f"{label} Distribution with Error")
        ax_main.grid(True, alpha=0.25)
        ax_main.legend(loc="upper right", frameon=True, framealpha=0.9)
        if show_stats:
            stats = self._stats_text(pred_values, pred_label) + "\n\n" + self._stats_text(true_values, true_label)
            self._add_stats_box(ax_main, stats, x=0.03, y=0.38)

        sns.histplot(err_values, kde=True, stat="density", color=error_color, alpha=0.28, element="step", ax=ax_err)
        ax_err.axvline(0.0, color="#374151", linestyle="--", linewidth=1.2)
        ax_err.set_xlabel(f"Error ({unit})")
        ax_err.set_ylabel("Density")
        ax_err.set_title("Error Distribution")
        ax_err.grid(True, alpha=0.25)
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 1.0])
        filename = f"dist_{label.lower().replace(' ', '_')}_with_error.png"
        fig.savefig(os.path.join(self.output_dir, filename), dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

    def plot_error_distribution(self, error: np.ndarray, label: str, unit: str):
        """
        Plot error distribution (Predicted - True).

        Args:
            error (np.ndarray): Error values.
            label (str): Label for the quantity.
            unit (str): Unit string.
        """
        filtered_error = self._filter_finite_1d(error, f"{label} Error", "Error distribution plot")
        if filtered_error is None:
            return
        plt.figure(figsize=(8, 6))
        
        sns.histplot(filtered_error, kde=True, stat="density", color='red', alpha=0.3)
        
        plt.axvline(0, color='k', linestyle='--', lw=1)
        plt.xlabel(f"{label} Error ({unit})")
        plt.ylabel("Density")
        plt.title(f"{label} Error Distribution")
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = f"error_dist_{label.lower().replace(' ', '_')}.png"
        plt.savefig(os.path.join(self.output_dir, filename), dpi=self.dpi)
        plt.close()
