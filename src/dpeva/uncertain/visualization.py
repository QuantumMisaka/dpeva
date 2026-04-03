import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import numpy as np
import os
import logging
from dpeva.utils.visual_style import (
    get_collection_pca_scatter_profile,
    get_legend_layout,
    get_publication_font_hierarchy,
    resolve_linked_tick_step,
    scale_font_hierarchy,
    set_visual_style,
)
from dpeva.constants import (
    FILENAME_UQ_FORCE,
    FILENAME_UQ_FORCE_RESCALED,
    FILENAME_UQ_DIFF_UQ_PARITY,
    FILENAME_UQ_DIFF_FDIFF_PARITY,
    FILENAME_UQ_FORCE_QBC_RND_FDIFF_SCATTER,
    FILENAME_UQ_FORCE_QBC_RND_FDIFF_SCATTER_TRUNCATED,
    FILENAME_UQ_FORCE_QBC_RND_IDENTITY_SCATTER,
    FILENAME_UQ_FORCE_QBC_RND_IDENTITY_SCATTER_TRUNCATED,
    FILENAME_UQ_QBC_CANDIDATE_FDIFF_PARITY,
    FILENAME_UQ_RND_CANDIDATE_FDIFF_PARITY,
    FILENAME_EXPLAINED_VARIANCE,
    FILENAME_COVERAGE_SCORE,
    FILENAME_FINAL_SAMPLED_PCAVIEW,
    FILENAME_FINAL_SAMPLED_PCAVIEW_BY_POOL,
)


class UQVisualizer:
    """Handles visualization of Uncertainty Quantification (UQ) and sampling results."""

    def __init__(
        self,
        save_dir,
        dpi=150,
        font_size=12,
        tick_target_count=6,
        legend_max_rows=8,
        legend_max_cols=4,
    ):
        """
        Initializes the UQVisualizer.

        Args:
            save_dir (str): Directory to save plots.
            dpi (int): DPI for saved figures (default 150).
        """
        self.save_dir = save_dir
        self.dpi = dpi
        self.tick_target_count = tick_target_count
        self.legend_max_rows = legend_max_rows
        self.legend_max_cols = legend_max_cols
        self.fonts = get_publication_font_hierarchy(font_size)
        set_visual_style(font_size=font_size)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def _build_local_fonts(
        self,
        title_scale=1.0,
        label_scale=1.0,
        tick_scale=1.0,
        legend_scale=1.0,
        legend_title_scale=1.0,
    ):
        return scale_font_hierarchy(
            self.fonts,
            title_scale=title_scale,
            label_scale=label_scale,
            tick_scale=tick_scale,
            legend_scale=legend_scale,
            legend_title_scale=legend_title_scale,
        )

    def _build_pca_scatter_profile(self):
        return get_collection_pca_scatter_profile(font_size=self.fonts["base"])

    def _build_pca_fonts(self):
        return self._build_pca_scatter_profile()["fonts"]

    def _build_coverage_fonts(self):
        return self._build_pca_scatter_profile()["fonts"]

    def _apply_text(self, title=None, xlabel=None, ylabel=None, fonts=None):
        active_fonts = self.fonts if fonts is None else fonts
        if title is not None:
            plt.title(title, fontsize=active_fonts["title"])
        if xlabel is not None:
            plt.xlabel(xlabel, fontsize=active_fonts["label"])
        if ylabel is not None:
            plt.ylabel(ylabel, fontsize=active_fonts["label"])

    def _apply_tick_fontsize(self, ax, tick_size):
        ax.tick_params(axis="both", which="major", labelsize=tick_size)

    def _apply_fixed_major_ticks(self, ax, step):
        ax.xaxis.set_major_locator(mtick.MultipleLocator(step))
        ax.yaxis.set_major_locator(mtick.MultipleLocator(step))
        return step

    def _apply_linked_major_ticks(self, ax, target_tick_count=None):
        step = resolve_linked_tick_step(
            ax.get_xlim(),
            ax.get_ylim(),
            target_tick_count=(
                self.tick_target_count
                if target_tick_count is None
                else target_tick_count
            ),
        )
        ax.xaxis.set_major_locator(mtick.MultipleLocator(step))
        ax.yaxis.set_major_locator(mtick.MultipleLocator(step))
        return step

    def _apply_pca_axis_layout(self, ax, profile):
        margin_x, margin_y = profile["axis_margins"]
        ax.margins(x=margin_x, y=margin_y)
        return self._apply_linked_major_ticks(
            ax, target_tick_count=profile["tick_target_count"]
        )

    def _set_nonnegative_limits(self, ax, upper=None):
        x_limits = ax.get_xlim()
        y_limits = ax.get_ylim()
        if upper is None:
            upper = max(float(x_limits[1]), float(y_limits[1]))
        ax.set_xlim(0, upper)
        ax.set_ylim(0, upper)

    def _place_standard_legend(self, title=None):
        ax = plt.gca()
        _, labels = ax.get_legend_handles_labels()
        if not labels:
            return
        plt.legend(
            title=title,
            fontsize=self.fonts["legend"],
            title_fontsize=self.fonts["legend_title"],
            frameon=True,
        )

    def _place_pool_legend(self, item_count, legend_size):
        if item_count <= 0:
            return
        if item_count <= self.legend_max_rows:
            plt.legend(
                frameon=True,
                fontsize=legend_size,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                borderaxespad=0.0,
            )
            plt.gcf().subplots_adjust(right=0.76)
            return

        legend_layout = get_legend_layout(
            item_count,
            max_rows=self.legend_max_rows,
            max_cols=self.legend_max_cols,
        )
        plt.legend(
            frameon=True,
            fontsize=legend_size,
            loc=legend_layout["loc"],
            bbox_to_anchor=legend_layout["bbox_to_anchor"],
            ncol=legend_layout["ncol"],
            borderaxespad=0.0,
        )
        plt.gcf().subplots_adjust(bottom=legend_layout["bottom_margin"])

    def _filter_uq(self, data, name="UQ"):
        """
        Filter UQ data to be within [0, 2] and warn if truncation occurs.

        Args:
            data (np.ndarray): The UQ data array to filter.
            name (str, optional): Name of the data for logging. Defaults to "UQ".

        Returns:
            tuple: (filtered_data, valid_mask)
        """
        mask = (data >= 0) & (data <= 2.0)
        truncated_count = len(data) - np.sum(mask)
        if truncated_count > 0:
            logging.getLogger(__name__).warning(
                f"{name}: Truncating {truncated_count} values outside [0, 2] for visualization."
            )
        return data[mask], mask

    def plot_uq_distribution(self, uq_qbc, uq_rnd, uq_rnd_rescaled=None):
        """
        Plots KDE distribution of UQ metrics.

        Args:
            uq_qbc (np.ndarray): QbC uncertainty values.
            uq_rnd (np.ndarray): RND uncertainty values.
            uq_rnd_rescaled (np.ndarray, optional): Rescaled RND uncertainty values. Defaults to None.
        """
        # Filter data
        uq_qbc_viz, _ = self._filter_uq(uq_qbc, "UQ-QbC")
        uq_rnd_viz, _ = self._filter_uq(uq_rnd, "UQ-RND")

        # 1. Raw UQ comparison
        plt.figure(figsize=(8, 6))
        if len(uq_qbc_viz) > 0:
            sns.kdeplot(uq_qbc_viz, color="blue", label="UQ-QbC", bw_adjust=0.5)
        if len(uq_rnd_viz) > 0:
            sns.kdeplot(uq_rnd_viz, color="red", label="UQ-RND", bw_adjust=0.5)
        self._apply_text(
            title="Distribution of UQ-force by KDEplot (Truncated [0, 2])",
            xlabel="UQ Value",
            ylabel="Density",
        )
        self._place_standard_legend(title="Series")
        plt.grid(True, linestyle="-", alpha=0.6)
        plt.savefig(os.path.join(self.save_dir, FILENAME_UQ_FORCE), dpi=self.dpi)
        plt.close()

        # 2. Rescaled comparison
        if uq_rnd_rescaled is not None:
            uq_rnd_rescaled_viz, _ = self._filter_uq(uq_rnd_rescaled, "UQ-RND-rescaled")

            plt.figure(figsize=(8, 6))
            if len(uq_qbc_viz) > 0:
                sns.kdeplot(uq_qbc_viz, color="blue", label="UQ-QbC", bw_adjust=0.5)
            if len(uq_rnd_rescaled_viz) > 0:
                sns.kdeplot(
                    uq_rnd_rescaled_viz,
                    color="red",
                    label="UQ-RND-rescaled",
                    bw_adjust=0.5,
                )
            self._apply_text(
                title="Distribution of UQ-force by KDEplot (Truncated [0, 2])",
                xlabel="UQ Value",
                ylabel="Density",
            )
            self._place_standard_legend(title="Series")
            plt.grid(True, linestyle="-", alpha=0.6)
            plt.savefig(
                os.path.join(self.save_dir, FILENAME_UQ_FORCE_RESCALED), dpi=self.dpi
            )
            plt.close()

    def plot_uq_with_trust_range(self, uq_data, label, filename, trust_lo, trust_hi):
        """
        Plots UQ distribution with trust range highlights.

        Args:
            uq_data (np.ndarray): UQ data to plot.
            label (str): Label for the data.
            filename (str): Output filename.
            trust_lo (float): Lower trust bound.
            trust_hi (float): Upper trust bound.
        """
        uq_data_viz, _ = self._filter_uq(uq_data, label)

        plt.figure(figsize=(8, 6))
        if len(uq_data_viz) > 0:
            sns.kdeplot(uq_data_viz, color="blue", label=label, bw_adjust=0.5)
        self._apply_text(
            title=f"Distribution of {label} by KDEplot (Truncated [0, 2])",
            xlabel=f"{label} Value",
            ylabel="Density",
        )
        plt.grid(True, linestyle="-", alpha=0.6)

        plt.axvline(
            trust_lo,
            color="purple",
            linestyle="--",
            linewidth=1,
            label="Trust Lower",
        )
        plt.axvline(
            trust_hi,
            color="purple",
            linestyle="--",
            linewidth=1,
            label="Trust Upper",
        )

        # Highlight regions
        # Use viz min/max for span
        viz_min = np.min(uq_data_viz) if len(uq_data_viz) > 0 else 0
        viz_max = np.max(uq_data_viz) if len(uq_data_viz) > 0 else 2

        plt.axvspan(viz_min, trust_lo, alpha=0.1, color="green", label="Trusted")
        plt.axvspan(trust_lo, trust_hi, alpha=0.1, color="yellow", label="Candidate")
        plt.axvspan(trust_hi, viz_max, alpha=0.1, color="red", label="Untrusted")
        self._place_standard_legend(title="Regions")

        plt.savefig(os.path.join(self.save_dir, filename), dpi=self.dpi)
        plt.close()

    def plot_uq_vs_error(self, uq_qbc, uq_rnd, diff_maxf, rescaled=False):
        """
        Plots Parity plot of UQ vs True Error.

        Args:
            uq_qbc (np.ndarray): QbC uncertainty values.
            uq_rnd (np.ndarray): RND uncertainty values.
            diff_maxf (np.ndarray): Maximum force difference (true error).
            rescaled (bool, optional): Whether RND is rescaled. Defaults to False.
        """
        label_rnd = "RND-rescaled" if rescaled else "RND"
        filename = (
            "UQ-force-rescaled-fdiff-parity.png"
            if rescaled
            else "UQ-force-fdiff-parity.png"
        )

        # Filter for scatter plots
        uq_qbc_viz, mask_qbc = self._filter_uq(uq_qbc, "UQ-QbC")
        uq_rnd_viz, mask_rnd = self._filter_uq(uq_rnd, f"UQ-{label_rnd}")

        plt.figure(figsize=(8, 6))
        if len(uq_qbc_viz) > 0:
            plt.scatter(
                uq_qbc_viz, diff_maxf[mask_qbc], color="blue", label="QbC", s=20
            )
        if len(uq_rnd_viz) > 0:
            plt.scatter(
                uq_rnd_viz, diff_maxf[mask_rnd], color="red", label=label_rnd, s=20
            )
        self._apply_text(
            title="UQ vs Force Diff (Truncated [0, 2])",
            xlabel="UQ Value",
            ylabel="True Max Force Diff",
        )
        plt.grid(True, linestyle="-", alpha=0.6)
        self._apply_linked_major_ticks(plt.gca())
        self._place_standard_legend(title="Series")
        plt.savefig(os.path.join(self.save_dir, filename), dpi=self.dpi)
        plt.close()

    def plot_uq_diff_parity(self, uq_qbc, uq_rnd_rescaled, diff_maxf=None):
        """
        Plots difference between QbC and RND vs Error.

        Args:
            uq_qbc (np.ndarray): QbC uncertainty values.
            uq_rnd_rescaled (np.ndarray): Rescaled RND uncertainty values.
            diff_maxf (np.ndarray, optional): Maximum force difference (true error). Defaults to None.
        """
        uq_diff = np.abs(uq_rnd_rescaled - uq_qbc)

        # UQ Diff vs UQ
        plt.figure(figsize=(8, 6))
        plt.scatter(uq_diff, uq_qbc, color="blue", label="UQ-qbc-for", s=20)
        plt.scatter(
            uq_diff, uq_rnd_rescaled, color="red", label="UQ-rnd-for-rescaled", s=20
        )
        self._apply_text(
            title="UQ-diff vs UQ",
            xlabel="UQ-diff Value",
            ylabel="UQ Value",
        )
        plt.grid(True, linestyle="-", alpha=0.6)
        self._apply_linked_major_ticks(plt.gca())
        self._place_standard_legend(title="Series")
        plt.savefig(
            os.path.join(self.save_dir, FILENAME_UQ_DIFF_UQ_PARITY), dpi=self.dpi
        )
        plt.close()

        # UQ Diff vs Force Diff
        if diff_maxf is not None:
            plt.figure(figsize=(8, 6))
            plt.scatter(uq_diff, diff_maxf, color="blue", label="UQ-diff-force", s=20)
            self._apply_text(
                title="UQ-diff vs Force Diff",
                xlabel="UQ-diff Value",
                ylabel="True Max Force Diff",
            )
            plt.grid(True, linestyle="-", alpha=0.6)
            self._apply_linked_major_ticks(plt.gca())
            self._place_standard_legend(title="Series")
            plt.savefig(
                os.path.join(self.save_dir, FILENAME_UQ_DIFF_FDIFF_PARITY), dpi=self.dpi
            )
            plt.close()

    def plot_uq_fdiff_scatter(
        self, df_uq, scheme, trust_lo, trust_hi, rnd_trust_lo, rnd_trust_hi
    ):
        """Plots 2D scatter of QbC vs RND with Max Force Diff as hue."""
        if "diff_maxf_0_frame" not in df_uq.columns:
            logging.getLogger(__name__).warning(
                "diff_maxf_0_frame not in dataframe, skipping UQ-force-qbc-rnd-fdiff-scatter.png"
            )
            return

        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=df_uq,
            x="uq_qbc_for",
            y="uq_rnd_for_rescaled",
            hue="diff_maxf_0_frame",
            palette="Reds",
            alpha=0.8,
            s=60,
        )
        self._apply_text(
            title="UQ-QbC and UQ-RND vs Max Force Diff",
            xlabel="UQ-QbC Value",
            ylabel="UQ-RND-rescaled Value",
        )
        self._setup_2d_plot_axes(trust_lo, trust_hi, rnd_trust_lo, rnd_trust_hi)
        self._draw_boundary(scheme, trust_lo, trust_hi, rnd_trust_lo, rnd_trust_hi)
        self._place_standard_legend(title="Max Force Diff")
        plt.savefig(
            os.path.join(self.save_dir, FILENAME_UQ_FORCE_QBC_RND_FDIFF_SCATTER),
            dpi=self.dpi,
        )
        plt.close()

        max_qbc = df_uq["uq_qbc_for"].max()
        max_rnd = df_uq["uq_rnd_for_rescaled"].max()
        if max_qbc > 2.0 or max_rnd > 2.0:
            logging.getLogger(__name__).warning(
                "UQ-fdiff-scatter: Data exceeds [0, 2] range, truncating for visualization."
            )
            df_uq_trunc = df_uq[
                (df_uq["uq_qbc_for"] >= 0)
                & (df_uq["uq_qbc_for"] <= 2)
                & (df_uq["uq_rnd_for_rescaled"] >= 0)
                & (df_uq["uq_rnd_for_rescaled"] <= 2)
            ]
            if len(df_uq_trunc) < len(df_uq):
                logging.getLogger(__name__).warning(
                    f"UQ-fdiff-scatter: Truncating {len(df_uq) - len(df_uq_trunc)} structures outside [0, 2] for visualization."
                )
            plt.figure(figsize=(8, 6))
            sns.scatterplot(
                data=df_uq_trunc,
                x="uq_qbc_for",
                y="uq_rnd_for_rescaled",
                hue="diff_maxf_0_frame",
                palette="Reds",
                alpha=0.8,
                s=60,
            )
            self._apply_text(
                title="UQ-QbC and UQ-RND vs Max Force Diff (Truncated [0, 2])",
                xlabel="UQ-QbC Value",
                ylabel="UQ-RND-rescaled Value",
            )
            self._draw_boundary(scheme, trust_lo, trust_hi, rnd_trust_lo, rnd_trust_hi)
            ax = plt.gca()
            self._set_nonnegative_limits(ax, upper=2.0)
            plt.grid(True)
            self._apply_linked_major_ticks(ax)
            self._place_standard_legend(title="Max Force Diff")
            plt.savefig(
                os.path.join(
                    self.save_dir, FILENAME_UQ_FORCE_QBC_RND_FDIFF_SCATTER_TRUNCATED
                ),
                dpi=self.dpi,
            )
            plt.close()
        else:
            logging.getLogger(__name__).info(
                "UQ-fdiff-scatter data within [0, 2], skipping truncated plot."
            )

    def plot_uq_identity_scatter(
        self, df_uq, scheme, trust_lo, trust_hi, rnd_trust_lo, rnd_trust_hi
    ):
        """Plots 2D scatter of QbC vs RND with Identity as hue."""
        if "uq_identity" not in df_uq.columns:
            logging.getLogger(__name__).warning(
                "uq_identity not in dataframe, skipping UQ-force-qbc-rnd-identity-scatter.png"
            )
            return

        # Full plot
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=df_uq,
            x="uq_qbc_for",
            y="uq_rnd_for_rescaled",
            hue="uq_identity",
            palette={"candidate": "orange", "accurate": "green", "failed": "red"},
            alpha=0.5,
            s=60,
        )
        self._apply_text(
            title="UQ QbC+RND Selection View",
            xlabel="UQ-QbC Value",
            ylabel="UQ-RND-rescaled Value",
        )
        self._setup_2d_plot_axes(trust_lo, trust_hi, rnd_trust_lo, rnd_trust_hi)
        self._draw_boundary(scheme, trust_lo, trust_hi, rnd_trust_lo, rnd_trust_hi)
        self._place_standard_legend(title="Identity")
        plt.savefig(
            os.path.join(self.save_dir, FILENAME_UQ_FORCE_QBC_RND_IDENTITY_SCATTER),
            dpi=self.dpi,
        )
        plt.close()

        # Truncated plot [0, 2]
        # Only generate if data exceeds [0, 2] range
        max_qbc = df_uq["uq_qbc_for"].max()
        max_rnd = df_uq["uq_rnd_for_rescaled"].max()

        if max_qbc > 2.0 or max_rnd > 2.0:
            logging.getLogger(__name__).warning(
                "UQ-identity-scatter: Data exceeds [0, 2] range, truncating for visualization."
            )
            df_uq_trunc = df_uq[
                (df_uq["uq_qbc_for"] >= 0)
                & (df_uq["uq_qbc_for"] <= 2)
                & (df_uq["uq_rnd_for_rescaled"] >= 0)
                & (df_uq["uq_rnd_for_rescaled"] <= 2)
            ]

            if len(df_uq_trunc) < len(df_uq):
                logging.getLogger(__name__).warning(
                    f"UQ-identity-scatter: Truncating {len(df_uq) - len(df_uq_trunc)} structures outside [0, 2] for visualization."
                )

            plt.figure(figsize=(8, 6))
            sns.scatterplot(
                data=df_uq_trunc,
                x="uq_qbc_for",
                y="uq_rnd_for_rescaled",
                hue="uq_identity",
                palette={"candidate": "orange", "accurate": "green", "failed": "red"},
                alpha=0.5,
                s=60,
            )
            self._apply_text(
                title="UQ QbC+RND Selection View (Truncated [0, 2])",
                xlabel="UQ-QbC Value",
                ylabel="UQ-RND-rescaled Value",
            )
            self._draw_boundary(scheme, trust_lo, trust_hi, rnd_trust_lo, rnd_trust_hi)
            ax = plt.gca()
            self._set_nonnegative_limits(ax, upper=2.0)
            plt.grid(True)
            self._apply_fixed_major_ticks(ax, step=0.25)
            self._place_standard_legend(title="Identity")
            plt.savefig(
                os.path.join(
                    self.save_dir, FILENAME_UQ_FORCE_QBC_RND_IDENTITY_SCATTER_TRUNCATED
                ),
                dpi=self.dpi,
            )
            plt.close()
        else:
            logging.getLogger(__name__).info(
                "UQ data within [0, 2], skipping truncated plot."
            )

    def plot_candidate_vs_error(self, df_uq, df_candidate):
        """
        Plots Candidate UQ vs Error.

        Args:
            df_uq (pd.DataFrame): Dataframe containing all UQ metrics.
            df_candidate (pd.DataFrame): Dataframe containing candidate UQ metrics.
        """
        # QbC
        plt.figure(figsize=(8, 6))
        plt.scatter(
            df_uq["uq_qbc_for"],
            df_uq["diff_maxf_0_frame"],
            color="blue",
            label="UQ-QbC",
            s=20,
        )
        plt.scatter(
            df_candidate["uq_qbc_for"],
            df_candidate["diff_maxf_0_frame"],
            color="orange",
            label="Candidate",
            s=20,
        )
        self._apply_text(
            title="UQ vs Force Diff",
            xlabel="UQ Value",
            ylabel="True Max Force Diff",
        )
        plt.grid(True, linestyle="-", alpha=0.6)
        self._apply_linked_major_ticks(plt.gca())
        self._place_standard_legend(title="Series")
        plt.savefig(
            os.path.join(self.save_dir, FILENAME_UQ_QBC_CANDIDATE_FDIFF_PARITY),
            dpi=self.dpi,
        )
        plt.close()

        # RND
        plt.figure(figsize=(8, 6))
        plt.scatter(
            df_uq["uq_rnd_for_rescaled"],
            df_uq["diff_maxf_0_frame"],
            color="blue",
            label="UQ-RND-rescaled",
            s=20,
        )
        plt.scatter(
            df_candidate["uq_rnd_for_rescaled"],
            df_candidate["diff_maxf_0_frame"],
            color="orange",
            label="Candidate",
            s=20,
        )
        self._apply_text(
            title="UQ vs Force Diff",
            xlabel="UQ Value",
            ylabel="True Max Force Diff",
        )
        plt.grid(True, linestyle="-", alpha=0.6)
        self._apply_linked_major_ticks(plt.gca())
        self._place_standard_legend(title="Series")
        plt.savefig(
            os.path.join(self.save_dir, FILENAME_UQ_RND_CANDIDATE_FDIFF_PARITY),
            dpi=self.dpi,
        )
        plt.close()

    def plot_pca_analysis(
        self,
        explained_variance,
        selected_PC_dim,
        all_features,
        direct_indices,
        random_indices,
        scores_direct,
        scores_random,
        df_uq,
        final_indices,
        n_candidates=None,
        full_features=None,
    ):
        """
        Plots all PCA and DIRECT related figures.

        Args:
            all_features (np.ndarray): PCA features for all samples (Joint if joint sampling used).
            direct_indices (list): Indices selected by DIRECT (in all_features).
            random_indices (list): Indices selected by Random (in all_features).
            scores_direct (list): Coverage scores for DIRECT.
            scores_random (list): Coverage scores for Random.
            df_uq (pd.DataFrame): Dataframe of candidates (for Final_sampled_PCAview).
            final_indices (list): Indices of finally selected candidates (relative to df_uq if n_candidates is None).
            n_candidates (int, optional): Number of candidate samples. If provided, assumes all_features
                                          contains [Candidates; Training]. Used to distinguish markers.
            full_features (np.ndarray, optional): PCA features for the entire dataset (including filtered out ones).
                                                  If provided, plotted as background.
        """
        explained_variance = np.asarray(explained_variance, dtype=float)
        if explained_variance.ndim != 1 or explained_variance.size == 0:
            raise ValueError("explained_variance must be a non-empty 1D array")
        if np.nanmax(explained_variance) > 1.0 + 1e-8:
            total = np.nansum(explained_variance)
            if total > 0:
                explained_variance = explained_variance / total
            else:
                raise ValueError("Invalid explained_variance: sum must be positive")

        pca_scatter_profile = self._build_pca_scatter_profile()
        pca_fonts = pca_scatter_profile["fonts"]

        # 1. Explained Variance
        plt.figure(figsize=(8, 6))
        plt.plot(
            range(1, selected_PC_dim + 6 + 1),
            explained_variance[: selected_PC_dim + 6],
            "o-",
            color="#4c72b0",
        )
        self._apply_text(
            xlabel=r"i$^{\mathrm{th}}$ PC",
            ylabel="Explained Variance Ratio",
            fonts=pca_fonts,
        )
        ax = plt.gca()
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        self._apply_tick_fontsize(ax, pca_fonts["tick"])
        plt.grid(True, linestyle="-", alpha=0.6)
        plt.savefig(
            os.path.join(self.save_dir, FILENAME_EXPLAINED_VARIANCE), dpi=self.dpi, bbox_inches="tight"
        )
        plt.close()

        # 2. PCA Feature Coverage (DIRECT vs Random)
        # Use provided all_features directly (already PCA projected)
        self._plot_coverage(
            all_features,
            direct_indices,
            "DIRECT",
            n_candidates,
            profile=pca_scatter_profile,
        )
        self._plot_coverage(
            all_features,
            random_indices,
            "Random",
            n_candidates,
            profile=pca_scatter_profile,
        )

        # 3. Coverage Score Bar Chart
        x = np.arange(len(scores_direct))
        x_ticks = [f"PC {n + 1}" for n in range(len(x))]
        plt.figure(figsize=(15, 5))
        plt.bar(
            x + 0.6,
            scores_direct,
            width=0.3,
            label=rf"DIRECT, $\overline{{\mathrm{{Coverage\ score}}}}$ = {np.mean(scores_direct):.3f}",
            color="#4c72b0",
        )
        plt.bar(
            x + 0.3,
            scores_random,
            width=0.3,
            label=rf"Random, $\overline{{\mathrm{{Coverage\ score}}}}$ = {np.mean(scores_random):.3f}",
            color="#dd8452",
        )
        self._apply_text(
            title="Coverage Score Analysis (Joint Space)",
            ylabel="Coverage score",
            fonts=pca_fonts,
        )
        plt.xticks(x + 0.45, x_ticks, size=pca_fonts["tick"])
        plt.yticks(np.linspace(0, 1.0, 6), size=pca_fonts["tick"])
        plt.grid(True, axis="y", linestyle="-", alpha=0.6)
        plt.legend(shadow=True, loc="best", fontsize=pca_fonts["legend"], framealpha=0.75)
        plt.savefig(os.path.join(self.save_dir, FILENAME_COVERAGE_SCORE), dpi=self.dpi, bbox_inches="tight")
        plt.close()

        # 4. Final Selection in PCA Space
        plt.figure(figsize=pca_scatter_profile["figure_size"])

        # Define consistent styles
        style_all = {
            "color": "#A9A9A9",
            "alpha": 0.4,
            "s": 15,
            "marker": ".",
        }  # Darker Gray background for All Data
        style_cand = {
            "color": "#FFC000",
            "alpha": 0.6,
            "s": 25,
            "marker": "*",
        }  # Orange-Yellow for Candidate
        style_sel_new = {
            "color": "#6A5ACD",
            "edgecolors": "black",
            "linewidth": 0.5,
            "s": 60,
            "marker": "*",
        }  # SlateBlue for Selected

        # Plot Full Background first if available
        if full_features is not None:
            plt.scatter(
                full_features[:, 0],
                full_features[:, 1],
                label=f"All Data in Pool {len(full_features):,}",
                **style_all,
                zorder=1,
            )

        if n_candidates is not None:
            # Joint Mode Visualization: Simplify to show only Candidates and New Selection
            PCs_alldata = all_features

            # Plot Candidates Only (First n_candidates rows)
            pcs_cand = PCs_alldata[:n_candidates]

            if len(pcs_cand) > 0:
                plt.scatter(
                    pcs_cand[:, 0],
                    pcs_cand[:, 1],
                    label=f"Candidate {len(pcs_cand):,}",
                    **style_cand,
                    zorder=2,
                )

            # Identify new candidates selected (exclude training selection)
            new_cand_indices = [idx for idx in direct_indices if idx < n_candidates]

            if len(new_cand_indices) > 0:
                plt.scatter(
                    PCs_alldata[new_cand_indices, 0],
                    PCs_alldata[new_cand_indices, 1],
                    label=f"Final Selected {len(new_cand_indices)}",
                    **style_sel_new,
                    zorder=10,
                )

        else:
            # Normal Mode (Candidate Only)
            PCs_alldata = all_features

            plt.scatter(
                PCs_alldata[:, 0],
                PCs_alldata[:, 1],
                label=f"Candidate {len(df_uq):,}",
                **style_cand,
                zorder=2,
            )

            plt.scatter(
                PCs_alldata[direct_indices, 0],
                PCs_alldata[direct_indices, 1],
                label=f"Final Selected {len(direct_indices)}",
                **style_sel_new,
                zorder=10,
            )

        self._apply_text(
            title="PCA of UQ-DIRECT sampling",
            xlabel="PC1",
            ylabel="PC2",
            fonts=pca_fonts,
        )
        plt.grid(False)
        ax = plt.gca()
        self._apply_pca_axis_layout(ax, pca_scatter_profile)
        self._apply_tick_fontsize(ax, pca_fonts["tick"])
        plt.legend(frameon=True, fontsize=pca_fonts["legend"], loc="best", framealpha=0.75)
        plt.savefig(
            os.path.join(self.save_dir, FILENAME_FINAL_SAMPLED_PCAVIEW), dpi=self.dpi
        )
        plt.close()

        if n_candidates is not None:
            self._plot_joint_multipool_summary(
                all_features=all_features,
                df_uq=df_uq,
                final_indices=final_indices,
                full_features=full_features,
                n_candidates=n_candidates,
            )

        return pd.DataFrame(all_features[:, :2], columns=["PC1", "PC2"])

    def _plot_joint_multipool_summary(
        self, all_features, df_uq, final_indices, full_features=None, n_candidates=None
    ):
        pool_series = df_uq["dataname"].map(self._pool_name_from_dataname)
        if pool_series.nunique() <= 1:
            logging.getLogger(__name__).info(
                "Single data pool detected, skipping Final_sampled_PCAview_by_pool plot."
            )
            return

        pca_fonts = self._build_pca_fonts()
        plt.figure(figsize=(14, 10))
        if full_features is not None:
            plt.scatter(
                full_features[:, 0],
                full_features[:, 1],
                color="#A9A9A9",
                alpha=0.35,
                s=15,
                marker=".",
                label=f"All Data in Pool {len(full_features):,}",
                zorder=1,
            )

        if n_candidates is None:
            n_candidates = len(df_uq)
        candidate_features = all_features[:n_candidates]

        if len(final_indices) > 0:
            selected_positions = np.asarray(final_indices, dtype=int)
            valid_mask = (selected_positions >= 0) & (
                selected_positions < len(candidate_features)
            )
            selected_positions = selected_positions[valid_mask]
            selected_df = df_uq.iloc[selected_positions].copy()
            selected_df = selected_df.reset_index(drop=True)
            selected_df["pc1"] = candidate_features[selected_positions, 0]
            selected_df["pc2"] = candidate_features[selected_positions, 1]
        else:
            selected_df = pd.DataFrame(columns=list(df_uq.columns) + ["pc1", "pc2"])
        if not selected_df.empty:
            selected_df["pool"] = selected_df["dataname"].map(
                self._pool_name_from_dataname
            )
            unique_pools = sorted(selected_df["pool"].unique())
            cmap = plt.get_cmap("tab20", max(len(unique_pools), 1))

            for idx, pool_name in enumerate(unique_pools):
                pool_df = selected_df[selected_df["pool"] == pool_name]
                plt.scatter(
                    pool_df["pc1"].to_numpy(),
                    pool_df["pc2"].to_numpy(),
                    color=cmap(idx),
                    alpha=0.9,
                    s=70,
                    marker="*",
                    edgecolors="black",
                    linewidth=0.4,
                    label=f"{pool_name} ({len(pool_df)})",
                    zorder=10,
                )

        self._apply_text(
            title="PCA of UQ-DIRECT sampling (Sampled by Pool)",
            xlabel="PC1",
            ylabel="PC2",
            fonts=pca_fonts,
        )
        plt.grid(False)
        ax = plt.gca()
        self._apply_pca_axis_layout(ax, self._build_pca_scatter_profile())
        self._apply_tick_fontsize(ax, pca_fonts["tick"])
        n_legends = selected_df["pool"].nunique() if not selected_df.empty else 0
        self._place_pool_legend(n_legends, legend_size=pca_fonts["legend"])
        plt.savefig(
            os.path.join(self.save_dir, FILENAME_FINAL_SAMPLED_PCAVIEW_BY_POOL),
            dpi=self.dpi,
        )
        plt.close()

    def _pool_name_from_dataname(self, dataname):
        sys_name = str(dataname).rsplit("-", 1)[0]
        pool_name = os.path.dirname(sys_name)
        return pool_name if pool_name else "root"

    def _plot_coverage(
        self,
        all_features,
        selected_indices,
        method,
        n_candidates=None,
        profile=None,
    ):
        active_profile = (
            self._build_pca_scatter_profile() if profile is None else profile
        )
        coverage_fonts = active_profile["fonts"]
        plt.figure(figsize=active_profile["figure_size"])

        # Consistent styles
        style_train = {"color": "#C0C0C0", "alpha": 0.4, "marker": "."}
        style_cand = {
            "color": "#FFC000",
            "alpha": 0.5,
            "marker": "*",
        }  # Orange-Yellow for Candidates
        style_sel_train = {"color": "gray", "marker": "x", "s": 30, "linewidth": 1.0}
        style_sel_new = {
            "color": "#6A5ACD",
            "edgecolors": "black",
            "linewidth": 0.5,
            "s": 60,
            "marker": "*",
        }  # SlateBlue for Selected

        if n_candidates is not None:
            # Plot Training background
            training_pcs = all_features[n_candidates:]
            candidate_pcs = all_features[:n_candidates]

            plt.plot(
                training_pcs[:, 0],
                training_pcs[:, 1],
                linestyle="None",
                label=f"Training {len(training_pcs):,}",
                **style_train,
            )
            plt.plot(
                candidate_pcs[:, 0],
                candidate_pcs[:, 1],
                linestyle="None",
                label=f"Candidates {len(candidate_pcs):,}",
                **style_cand,
            )

            # Split selected into Train/Cand
            sel_cand = all_features[
                [idx for idx in selected_indices if idx < n_candidates]
            ]
            sel_train = all_features[
                [idx for idx in selected_indices if idx >= n_candidates]
            ]

            if len(sel_train) > 0:
                plt.scatter(
                    sel_train[:, 0],
                    sel_train[:, 1],
                    label=f"Selected (Train) {len(sel_train)}",
                    **style_sel_train,
                )
            if len(sel_cand) > 0:
                plt.scatter(
                    sel_cand[:, 0],
                    sel_cand[:, 1],
                    label=f"Selected (New) {len(sel_cand)}",
                    **style_sel_new,
                    zorder=10,
                )

        else:
            selected_features = all_features[selected_indices]
            plt.plot(
                all_features[:, 0],
                all_features[:, 1],
                "*",
                color="gray",
                alpha=0.5,
                label=f"All {len(all_features):,} structures",
            )
            plt.plot(
                selected_features[:, 0],
                selected_features[:, 1],
                "*",
                color="#6A5ACD",
                alpha=0.8,
                label=f"{method} sampled {len(selected_features):,}",
            )

        plt.legend(frameon=True, fontsize=coverage_fonts["legend"], loc="best", framealpha=0.75)
        self._apply_text(
            title=f"{method} Coverage Analysis",
            xlabel="PC1",
            ylabel="PC2",
            fonts=coverage_fonts,
        )
        plt.grid(False)
        ax = plt.gca()
        self._apply_pca_axis_layout(ax, active_profile)
        self._apply_tick_fontsize(ax, coverage_fonts["tick"])
        plt.savefig(
            os.path.join(self.save_dir, f"{method}_PCA_feature_coverage.png"),
            dpi=self.dpi,
        )
        plt.close()

    def _setup_2d_plot_axes(self, x_lo, x_hi, y_lo, y_hi):
        plt.grid(True)
        ax = plt.gca()
        self._set_nonnegative_limits(
            ax,
            upper=max(
                float(x_hi),
                float(y_hi),
                float(ax.get_xlim()[1]),
                float(ax.get_ylim()[1]),
            ),
        )
        self._apply_linked_major_ticks(ax)

    def _draw_boundary(self, scheme, uq_x_lo, uq_x_hi, uq_y_lo, uq_y_hi):
        # Draw bounding box
        plt.plot(
            [0, uq_x_hi], [uq_y_hi, uq_y_hi], color="black", linestyle="--", linewidth=2
        )
        plt.plot(
            [uq_x_hi, uq_x_hi], [0, uq_y_hi], color="black", linestyle="--", linewidth=2
        )

        if scheme == "strict":
            plt.plot(
                [uq_x_lo, uq_x_lo],
                [uq_y_lo, uq_y_hi],
                color="purple",
                linestyle="--",
                linewidth=2,
            )
            plt.plot(
                [uq_x_lo, uq_x_hi],
                [uq_y_lo, uq_y_lo],
                color="purple",
                linestyle="--",
                linewidth=2,
            )

        elif scheme == "circle_lo":
            center = (uq_x_hi, uq_y_hi)
            radius = np.sqrt((uq_x_lo - uq_x_hi) ** 2 + (uq_y_lo - uq_y_hi) ** 2)
            theta = np.linspace(np.pi, 1.5 * np.pi, 100)
            x_val = center[0] + radius * np.cos(theta)
            y_val = center[1] + radius * np.sin(theta)
            plt.plot(x_val, y_val, color="purple", linestyle="--", linewidth=2)

        elif scheme == "tangent_lo":
            x_val = np.linspace(0, uq_x_hi, 100)
            y_val = (
                -(uq_y_hi - uq_y_lo) / (uq_x_hi - uq_x_lo) * (x_val - uq_x_lo) + uq_y_lo
            )
            mask = y_val < uq_y_hi
            plt.plot(
                x_val[mask], y_val[mask], color="purple", linestyle="--", linewidth=2
            )

        elif scheme == "crossline_lo":
            # Crossline logic: intersection of two lines
            x_val = np.linspace(0, uq_x_hi, 100)
            delta_y = uq_y_hi - uq_y_lo
            delta_x = uq_x_hi - uq_x_lo
            y1 = (uq_y_hi * uq_x_lo - delta_y * x_val) / uq_x_lo
            y2 = (uq_y_lo * uq_x_hi - uq_y_lo * x_val) / delta_x
            y = np.max((y1, y2), axis=0)
            mask = y < uq_y_hi
            plt.plot(x_val[mask], y[mask], color="purple", linestyle="--", linewidth=2)

        elif scheme == "loose":
            plt.plot(
                [uq_x_lo, uq_x_lo],
                [0, uq_y_lo],
                color="purple",
                linestyle="--",
                linewidth=2,
            )
            plt.plot(
                [0, uq_x_lo],
                [uq_y_lo, uq_y_lo],
                color="purple",
                linestyle="--",
                linewidth=2,
            )
