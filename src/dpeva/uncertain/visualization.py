import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import numpy as np
import os
import logging

class UQVisualizer:
    """Handles visualization of Uncertainty Quantification (UQ) and sampling results."""

    def __init__(self, save_dir, dpi=150):
        """
        Initializes the UQVisualizer.

        Args:
            save_dir (str): Directory to save plots.
            dpi (int): DPI for saved figures (default 150).
        """
        self.save_dir = save_dir
        self.dpi = dpi
        
        # Configure global plot settings
        from dpeva.utils.visual_style import set_visual_style
        set_visual_style()
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def _filter_uq(self, data, name="UQ"):
        """Filter UQ data to be within [0, 2] and warn if truncation occurs."""
        mask = (data >= 0) & (data <= 2.0)
        truncated_count = len(data) - np.sum(mask)
        if truncated_count > 0:
            logging.getLogger(__name__).warning(f"{name}: Truncating {truncated_count} values outside [0, 2] for visualization.")
        return data[mask], mask

    def plot_uq_distribution(self, uq_qbc, uq_rnd, uq_rnd_rescaled=None):
        """Plots KDE distribution of UQ metrics."""
        # Filter data
        uq_qbc_viz, _ = self._filter_uq(uq_qbc, "UQ-QbC")
        uq_rnd_viz, _ = self._filter_uq(uq_rnd, "UQ-RND")
        
        # 1. Raw UQ comparison
        plt.figure(figsize=(8, 6))
        if len(uq_qbc_viz) > 0:
            sns.kdeplot(uq_qbc_viz, color="blue", label="UQ-QbC", bw_adjust=0.5)
        if len(uq_rnd_viz) > 0:
            sns.kdeplot(uq_rnd_viz, color="red", label="UQ-RND", bw_adjust=0.5)
        plt.title("Distribution of UQ-force by KDEplot (Truncated [0, 2])")
        plt.xlabel("UQ Value")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.save_dir}/UQ-force.png", dpi=self.dpi)
        plt.close()

        # 2. Rescaled comparison
        if uq_rnd_rescaled is not None:
            uq_rnd_rescaled_viz, _ = self._filter_uq(uq_rnd_rescaled, "UQ-RND-rescaled")
            
            plt.figure(figsize=(8, 6))
            if len(uq_qbc_viz) > 0:
                sns.kdeplot(uq_qbc_viz, color="blue", label="UQ-QbC", bw_adjust=0.5)
            if len(uq_rnd_rescaled_viz) > 0:
                sns.kdeplot(uq_rnd_rescaled_viz, color="red", label="UQ-RND-rescaled", bw_adjust=0.5)
            plt.title("Distribution of UQ-force by KDEplot (Truncated [0, 2])")
            plt.xlabel("UQ Value")
            plt.ylabel("Density")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{self.save_dir}/UQ-force-rescaled.png", dpi=self.dpi)
            plt.close()

    def plot_uq_with_trust_range(self, uq_data, label, filename, trust_lo, trust_hi):
        """Plots UQ distribution with trust range highlights."""
        uq_data_viz, _ = self._filter_uq(uq_data, label)
        
        plt.figure(figsize=(8, 6))
        if len(uq_data_viz) > 0:
            sns.kdeplot(uq_data_viz, color="blue", bw_adjust=0.5)
        plt.title(f"Distribution of {label} by KDEplot (Truncated [0, 2])")
        plt.xlabel(f"{label} Value")
        plt.ylabel("Density")
        plt.grid(True)
        
        plt.axvline(trust_lo, color='purple', linestyle='--', linewidth=1)
        plt.axvline(trust_hi, color='purple', linestyle='--', linewidth=1)
        
        # Highlight regions
        # Use viz min/max for span
        viz_min = np.min(uq_data_viz) if len(uq_data_viz) > 0 else 0
        viz_max = np.max(uq_data_viz) if len(uq_data_viz) > 0 else 2
        
        plt.axvspan(viz_min, trust_lo, alpha=0.1, color='green')
        plt.axvspan(trust_lo, trust_hi, alpha=0.1, color='yellow')
        plt.axvspan(trust_hi, viz_max, alpha=0.1, color='red')
        
        plt.savefig(f"{self.save_dir}/{filename}", dpi=self.dpi)
        plt.close()

    def plot_uq_vs_error(self, uq_qbc, uq_rnd, diff_maxf, rescaled=False):
        """Plots Parity plot of UQ vs True Error."""
        label_rnd = "RND-rescaled" if rescaled else "RND"
        filename = "UQ-force-rescaled-fdiff-parity.png" if rescaled else "UQ-force-fdiff-parity.png"
        
        # Filter for scatter plots
        uq_qbc_viz, mask_qbc = self._filter_uq(uq_qbc, "UQ-QbC")
        uq_rnd_viz, mask_rnd = self._filter_uq(uq_rnd, f"UQ-{label_rnd}")
        
        plt.figure(figsize=(8, 6))
        if len(uq_qbc_viz) > 0:
            plt.scatter(uq_qbc_viz, diff_maxf[mask_qbc], color="blue", label="QbC", s=20)
        if len(uq_rnd_viz) > 0:
            plt.scatter(uq_rnd_viz, diff_maxf[mask_rnd], color="red", label=label_rnd, s=20)
        
        plt.title("UQ vs Force Diff (Truncated [0, 2])")
        plt.xlabel("UQ Value")
        plt.ylabel("True Max Force Diff")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.save_dir}/{filename}", dpi=self.dpi)
        plt.close()

    def plot_uq_diff_parity(self, uq_qbc, uq_rnd_rescaled, diff_maxf=None):
        """Plots difference between QbC and RND vs Error."""
        uq_diff = np.abs(uq_rnd_rescaled - uq_qbc)
        
        # UQ Diff vs UQ
        plt.figure(figsize=(8, 6))
        plt.scatter(uq_diff, uq_qbc, color="blue", label="UQ-qbc-for", s=20)
        plt.scatter(uq_diff, uq_rnd_rescaled, color="red", label="UQ-rnd-for-rescaled", s=20)
        plt.title("UQ-diff vs UQ")
        plt.xlabel("UQ-diff Value")
        plt.ylabel("UQ Value")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.save_dir}/UQ-diff-UQ-parity.png", dpi=self.dpi)
        plt.close()

        # UQ Diff vs Force Diff
        if diff_maxf is not None:
            plt.figure(figsize=(8, 6))
            plt.scatter(uq_diff, diff_maxf, color="blue", label="UQ-diff-force", s=20)
            plt.title("UQ-diff vs Force Diff")
            plt.xlabel("UQ-diff Value")
            plt.ylabel("True Max Force Diff")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{self.save_dir}/UQ-diff-fdiff-parity.png", dpi=self.dpi)
            plt.close()

    def plot_uq_fdiff_scatter(self, df_uq, scheme, trust_lo, trust_hi, rnd_trust_lo, rnd_trust_hi):
        """Plots 2D scatter of QbC vs RND with Max Force Diff as hue."""
        if "diff_maxf_0_frame" not in df_uq.columns:
            logging.getLogger(__name__).warning("diff_maxf_0_frame not in dataframe, skipping UQ-force-qbc-rnd-fdiff-scatter.png")
            return

        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df_uq, 
                        x="uq_qbc_for", 
                        y="uq_rnd_for_rescaled", 
                        hue="diff_maxf_0_frame", 
                        palette="Reds",
                        alpha=0.8,
                        s=60)
        
        plt.title("UQ-QbC and UQ-RND vs Max Force Diff", fontsize=14)
        self._setup_2d_plot_axes(trust_lo, trust_hi, rnd_trust_lo, rnd_trust_hi)
        self._draw_boundary(scheme, trust_lo, trust_hi, rnd_trust_lo, rnd_trust_hi)
        plt.legend(title="Max Force Diff", fontsize=10)
        plt.savefig(f"{self.save_dir}/UQ-force-qbc-rnd-fdiff-scatter.png", dpi=self.dpi)
        plt.close()

    def plot_uq_identity_scatter(self, df_uq, scheme, trust_lo, trust_hi, rnd_trust_lo, rnd_trust_hi):
        """Plots 2D scatter of QbC vs RND with Identity as hue."""
        if "uq_identity" not in df_uq.columns:
            logging.getLogger(__name__).warning("uq_identity not in dataframe, skipping UQ-force-qbc-rnd-identity-scatter.png")
            return

        # Full plot
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df_uq, 
                        x="uq_qbc_for", 
                        y="uq_rnd_for_rescaled", 
                        hue="uq_identity", 
                        palette={"candidate": "orange", "accurate": "green", "failed": "red"},
                        alpha=0.5,
                        s=60)
        plt.title("UQ QbC+RND Selection View", fontsize=14)
        self._setup_2d_plot_axes(trust_lo, trust_hi, rnd_trust_lo, rnd_trust_hi)
        self._draw_boundary(scheme, trust_lo, trust_hi, rnd_trust_lo, rnd_trust_hi)
        plt.legend(title="Identity", fontsize=10)
        plt.savefig(f"{self.save_dir}/UQ-force-qbc-rnd-identity-scatter.png", dpi=self.dpi)
        plt.close()
        
        # Truncated plot [0, 2]
        df_uq_trunc = df_uq[(df_uq["uq_qbc_for"] >= 0) & (df_uq["uq_qbc_for"] <= 2) & 
                            (df_uq["uq_rnd_for_rescaled"] >= 0) & (df_uq["uq_rnd_for_rescaled"] <= 2)]
        if len(df_uq_trunc) < len(df_uq):
            logging.getLogger(__name__).warning(f"UQ-identity-scatter: Truncating {len(df_uq) - len(df_uq_trunc)} structures outside [0, 2] for visualization.")
        
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df_uq_trunc, 
                        x="uq_qbc_for", 
                        y="uq_rnd_for_rescaled", 
                        hue="uq_identity", 
                        palette={"candidate": "orange", "accurate": "green", "failed": "red"},
                        alpha=0.5,
                        s=60)
        plt.title("UQ QbC+RND Selection View (Truncated [0, 2])", fontsize=14)
        
        plt.xlabel("UQ-QbC Value", fontsize=12)
        plt.ylabel("UQ-RND-rescaled Value", fontsize=12)
        plt.grid(True)
        ax = plt.gca()
        # Adaptive locator for truncated view
        x_major_locator = mtick.MultipleLocator(0.1)
        y_major_locator = mtick.MultipleLocator(0.1)
        ax.xaxis.set_major_locator(x_major_locator)
        ax.yaxis.set_major_locator(y_major_locator)
        
        self._draw_boundary(scheme, trust_lo, trust_hi, rnd_trust_lo, rnd_trust_hi)
        plt.legend(title="Identity", fontsize=10)
        plt.savefig(f"{self.save_dir}/UQ-force-qbc-rnd-identity-scatter-truncated.png", dpi=self.dpi)
        plt.close()

    def plot_2d_uq_scatter(self, df_uq, scheme, trust_lo, trust_hi, rnd_trust_lo, rnd_trust_hi):
        """Deprecated: Use plot_uq_fdiff_scatter and plot_uq_identity_scatter instead."""
        pass

    def plot_candidate_vs_error(self, df_uq, df_candidate):
        """Plots Candidate UQ vs Error."""
        # QbC
        plt.figure(figsize=(8, 6))
        plt.scatter(df_uq["uq_qbc_for"], df_uq["diff_maxf_0_frame"], color="blue", label="UQ-QbC", s=20)
        plt.scatter(df_candidate["uq_qbc_for"], df_candidate["diff_maxf_0_frame"], color="orange", label="Candidate", s=20)
        plt.title("UQ vs Force Diff")
        plt.xlabel("UQ Value")
        plt.ylabel("True Max Force Diff")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.save_dir}/UQ-QbC-Candidate-fdiff-parity.png", dpi=self.dpi)
        plt.close()
        
        # RND
        plt.figure(figsize=(8, 6))
        plt.scatter(df_uq["uq_rnd_for_rescaled"], df_uq["diff_maxf_0_frame"], color="blue", label="UQ-RND-rescaled", s=20)
        plt.scatter(df_candidate["uq_rnd_for_rescaled"], df_candidate["diff_maxf_0_frame"], color="orange", label="Candidate", s=20)
        plt.title("UQ vs Force Diff")
        plt.xlabel("UQ Value")
        plt.ylabel("True Max Force Diff")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.save_dir}/UQ-RND-Candidate-fdiff-parity.png", dpi=self.dpi)
        plt.close()

    def plot_pca_analysis(self, explained_variance, selected_PC_dim, all_features, direct_indices, random_indices,
                          scores_direct, scores_random, df_uq, final_indices, n_candidates=None, full_features=None):
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
        # 1. Explained Variance
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, selected_PC_dim+6+1), explained_variance[:selected_PC_dim+6], "o-", color="#4c72b0")
        plt.xlabel(r"i$^{\mathrm{th}}$ PC", size=14)
        plt.ylabel("Explained variance", size=14)
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
        plt.grid(True, linestyle='-', alpha=0.6)
        plt.title("Explained Variance Ratio", fontsize=16)
        plt.savefig(f"{self.save_dir}/explained_variance.png", dpi=self.dpi)
        plt.close()

        # 2. PCA Feature Coverage (DIRECT vs Random)
        # Use provided all_features directly (already PCA projected)
        self._plot_coverage(all_features, direct_indices, "DIRECT", n_candidates)
        self._plot_coverage(all_features, random_indices, "Random", n_candidates)

        # 3. Coverage Score Bar Chart
        x = np.arange(len(scores_direct))
        x_ticks = [f"PC {n+1}" for n in range(len(x))]
        plt.figure(figsize=(15, 5))
        plt.bar(x + 0.6, scores_direct, width=0.3, label=rf"DIRECT, $\overline{{\mathrm{{Coverage\ score}}}}$ = {np.mean(scores_direct):.3f}", color="#4c72b0")
        plt.bar(x + 0.3, scores_random, width=0.3, label=rf"Random, $\overline{{\mathrm{{Coverage\ score}}}}$ = {np.mean(scores_random):.3f}", color="#dd8452")
        plt.xticks(x + 0.45, x_ticks, size=12)
        plt.yticks(np.linspace(0, 1.0, 6), size=12)
        plt.ylabel("Coverage score", size=14)
        plt.grid(True, axis='y', linestyle='-', alpha=0.6)
        plt.legend(shadow=True, loc="best", fontsize=12)
        plt.title("Coverage Score Analysis (Joint Space)", fontsize=16)
        plt.savefig(f"{self.save_dir}/coverage_score.png", dpi=self.dpi)
        plt.close()

        # 4. Final Selection in PCA Space
        plt.figure(figsize=(12, 10))
        
        # Define consistent styles
        style_all = {"color": "#D3D3D3", "alpha": 0.3, "s": 15, "marker": "."} # Gray background for All Data
        style_cand = {"color": "#4169E1", "alpha": 0.6, "s": 25, "marker": "*"}  # RoyalBlue for Candidate
        style_sel_new = {"color": "red", "edgecolors": "black", "linewidth": 0.8, "s": 100, "marker": "*"}
        
        # Plot Full Background first if available
        if full_features is not None:
             plt.scatter(full_features[:, 0], full_features[:, 1], 
                        label=f"All Data in Pool {len(full_features):,}", **style_all, zorder=0)

        if n_candidates is not None:
            # Joint Mode Visualization: Simplify to show only Candidates and New Selection
            PCs_alldata = all_features
            
            # Plot Candidates Only (First n_candidates rows)
            pcs_cand = PCs_alldata[:n_candidates]
            
            if len(pcs_cand) > 0:
                plt.scatter(pcs_cand[:, 0], pcs_cand[:, 1], 
                           label=f"Candidate {len(pcs_cand):,}", **style_cand, zorder=2)
            
            # Identify new candidates selected (exclude training selection)
            new_cand_indices = [idx for idx in direct_indices if idx < n_candidates]
            
            if len(new_cand_indices) > 0:
                plt.scatter(PCs_alldata[new_cand_indices, 0], PCs_alldata[new_cand_indices, 1], 
                           label=f"Final Selected {len(new_cand_indices)}", **style_sel_new, zorder=10)

        else:
            # Normal Mode (Candidate Only)
            PCs_alldata = all_features
            
            plt.scatter(PCs_alldata[:, 0], PCs_alldata[:, 1], 
                       label=f"Candidate {len(df_uq):,}", **style_cand, zorder=2) 
            
            plt.scatter(PCs_alldata[direct_indices, 0], PCs_alldata[direct_indices, 1], 
                       label=f"Final Selected {len(direct_indices)}", **style_sel_new, zorder=10)

        plt.title(f"PCA of UQ-DIRECT sampling", fontsize=16)
        plt.xlabel("PC1", size=14)
        plt.ylabel("PC2", size=14)
        plt.grid(True, linestyle='-', alpha=0.6)
        plt.legend(frameon=True, fontsize=12, loc='best')
        plt.savefig(f"{self.save_dir}/Final_sampled_PCAview.png", dpi=self.dpi)
        plt.close()
        
        return pd.DataFrame(all_features[:, :2], columns=['PC1', 'PC2'])

    def _plot_coverage(self, all_features, selected_indices, method, n_candidates=None):
        plt.figure(figsize=(10, 8))
        
        # Consistent styles
        style_train = {"color": "#C0C0C0", "alpha": 0.4, "marker": "."} 
        style_cand = {"color": "#6fa8dc", "alpha": 0.5, "marker": "*"} # Lighter blue for background
        style_sel_train = {"color": "gray", "marker": "x", "s": 60, "linewidth": 1.5}
        style_sel_new = {"color": "#FF8C00", "edgecolors": "black", "linewidth": 0.8, "s": 100, "marker": "*"} # DarkOrange for Coverage plots
        
        if n_candidates is not None:
            # Plot Training background
            training_pcs = all_features[n_candidates:]
            candidate_pcs = all_features[:n_candidates]
            
            plt.plot(training_pcs[:, 0], training_pcs[:, 1], linestyle='None', label=f"Training {len(training_pcs):,}", **style_train)
            plt.plot(candidate_pcs[:, 0], candidate_pcs[:, 1], linestyle='None', label=f"Candidates {len(candidate_pcs):,}", **style_cand)
            
            # Split selected into Train/Cand
            sel_cand = all_features[[idx for idx in selected_indices if idx < n_candidates]]
            sel_train = all_features[[idx for idx in selected_indices if idx >= n_candidates]]
            
            if len(sel_train) > 0:
                plt.scatter(sel_train[:, 0], sel_train[:, 1], label=f"Selected (Train) {len(sel_train)}", **style_sel_train)
            if len(sel_cand) > 0:
                plt.scatter(sel_cand[:, 0], sel_cand[:, 1], label=f"Selected (New) {len(sel_cand)}", **style_sel_new, zorder=10)
                
        else:
            selected_features = all_features[selected_indices]
            plt.plot(all_features[:, 0], all_features[:, 1], "*", color="gray", alpha=0.5, label=f"All {len(all_features):,} structures")
            plt.plot(selected_features[:, 0], selected_features[:, 1], "*", color="#FF8C00", alpha=0.8, label=f"{method} sampled {len(selected_features):,}")
            
        plt.legend(frameon=True, fontsize=12, loc='best')
        plt.ylabel("PC 2", size=14)
        plt.xlabel("PC 1", size=14)
        plt.title(f"{method} Coverage Analysis", fontsize=16)
        plt.grid(True, linestyle='-', alpha=0.6)
        plt.savefig(f"{self.save_dir}/{method}_PCA_feature_coverage.png", dpi=self.dpi)
        plt.close()

    def _setup_2d_plot_axes(self, x_lo, x_hi, y_lo, y_hi):
        plt.xlabel("UQ-QbC Value", fontsize=12)
        plt.ylabel("UQ-RND-rescaled Value", fontsize=12)
        plt.grid(True)
        ax = plt.gca()
        
        # Only set fine ticks if range is small enough
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        if (xlim[1] - xlim[0]) < 10:
            ax.xaxis.set_major_locator(mtick.MultipleLocator(0.1))
        if (ylim[1] - ylim[0]) < 10:
            ax.yaxis.set_major_locator(mtick.MultipleLocator(0.1))

    def _draw_boundary(self, scheme, uq_x_lo, uq_x_hi, uq_y_lo, uq_y_hi):
        # Draw bounding box
        plt.plot([0, uq_x_hi], [uq_y_hi, uq_y_hi], color='black', linestyle='--', linewidth=2)
        plt.plot([uq_x_hi, uq_x_hi], [0, uq_y_hi], color='black', linestyle='--', linewidth=2)

        if scheme == "strict":
            plt.plot([uq_x_lo, uq_x_lo], [uq_y_lo, uq_y_hi], color='purple', linestyle='--', linewidth=2)
            plt.plot([uq_x_lo, uq_x_hi], [uq_y_lo, uq_y_lo], color='purple', linestyle='--', linewidth=2)
        
        elif scheme == "circle_lo":
            center = (uq_x_hi, uq_y_hi)
            radius = np.sqrt((uq_x_lo - uq_x_hi)**2 + (uq_y_lo - uq_y_hi)**2)
            theta = np.linspace(np.pi, 1.5*np.pi, 100)
            x_val = center[0] + radius * np.cos(theta)
            y_val = center[1] + radius * np.sin(theta)
            plt.plot(x_val, y_val, color="purple", linestyle="--", linewidth=2)
            
        elif scheme == "tangent_lo":
            x_val = np.linspace(0, uq_x_hi, 100)
            y_val = - (uq_y_hi - uq_y_lo) / (uq_x_hi - uq_x_lo) * (x_val - uq_x_lo) + uq_y_lo
            mask = y_val < uq_y_hi
            plt.plot(x_val[mask], y_val[mask], color="purple", linestyle="--", linewidth=2)
            
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
            plt.plot([uq_x_lo, uq_x_lo], [0, uq_y_lo], color='purple', linestyle='--', linewidth=2)
            plt.plot([0, uq_x_lo], [uq_y_lo, uq_y_lo], color='purple', linestyle='--', linewidth=2)
