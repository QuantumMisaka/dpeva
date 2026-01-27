import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import numpy as np
import os

class UQVisualizer:
    """Handles visualization of Uncertainty Quantification (UQ) and sampling results."""

    def __init__(self, save_dir, dpi=150):
        self.save_dir = save_dir
        self.dpi = dpi
        
        # Configure global plot settings
        from dpeva.utils.visual_style import set_visual_style
        set_visual_style()
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def plot_uq_distribution(self, uq_qbc, uq_rnd, uq_rnd_rescaled=None):
        """Plots KDE distribution of UQ metrics."""
        # 1. Raw UQ comparison
        plt.figure(figsize=(8, 6))
        sns.kdeplot(uq_qbc, color="blue", label="UQ-QbC", bw_adjust=0.5)
        sns.kdeplot(uq_rnd, color="red", label="UQ-RND", bw_adjust=0.5)
        plt.title("Distribution of UQ-force by KDEplot")
        plt.xlabel("UQ Value")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.save_dir}/UQ-force.png", dpi=self.dpi)
        plt.close()

        # 2. Rescaled comparison
        if uq_rnd_rescaled is not None:
            plt.figure(figsize=(8, 6))
            sns.kdeplot(uq_qbc, color="blue", label="UQ-QbC", bw_adjust=0.5)
            sns.kdeplot(uq_rnd_rescaled, color="red", label="UQ-RND-rescaled", bw_adjust=0.5)
            plt.title("Distribution of UQ-force by KDEplot")
            plt.xlabel("UQ Value")
            plt.ylabel("Density")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{self.save_dir}/UQ-force-rescaled.png", dpi=self.dpi)
            plt.close()

    def plot_uq_with_trust_range(self, uq_data, label, filename, trust_lo, trust_hi):
        """Plots UQ distribution with trust range highlights."""
        plt.figure(figsize=(8, 6))
        sns.kdeplot(uq_data, color="blue", bw_adjust=0.5)
        plt.title(f"Distribution of {label} by KDEplot")
        plt.xlabel(f"{label} Value")
        plt.ylabel("Density")
        plt.grid(True)
        
        plt.axvline(trust_lo, color='purple', linestyle='--', linewidth=1)
        plt.axvline(trust_hi, color='purple', linestyle='--', linewidth=1)
        
        # Highlight regions
        plt.axvspan(np.min(uq_data), trust_lo, alpha=0.1, color='green')
        plt.axvspan(trust_lo, trust_hi, alpha=0.1, color='yellow')
        plt.axvspan(trust_hi, np.max(uq_data), alpha=0.1, color='red')
        
        plt.savefig(f"{self.save_dir}/{filename}", dpi=self.dpi)
        plt.close()

    def plot_uq_vs_error(self, uq_qbc, uq_rnd, diff_maxf, rescaled=False):
        """Plots Parity plot of UQ vs True Error."""
        label_rnd = "RND-rescaled" if rescaled else "RND"
        filename = "UQ-force-rescaled-fdiff-parity.png" if rescaled else "UQ-force-fdiff-parity.png"
        
        plt.figure(figsize=(8, 6))
        plt.scatter(uq_qbc, diff_maxf, color="blue", label="QbC", s=20)
        plt.scatter(uq_rnd, diff_maxf, color="red", label=label_rnd, s=20)
        plt.title("UQ vs Force Diff")
        plt.xlabel("UQ Value")
        plt.ylabel("True Max Force Diff")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.save_dir}/{filename}", dpi=self.dpi)
        plt.close()

    def plot_uq_diff_parity(self, uq_qbc, uq_rnd_rescaled, diff_maxf, rescaled=False):
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
        plt.figure(figsize=(8, 6))
        plt.scatter(uq_diff, diff_maxf, color="blue", label="UQ-diff-force", s=20)
        plt.title("UQ-diff vs Force Diff")
        plt.xlabel("UQ-diff Value")
        plt.ylabel("True Max Force Diff")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.save_dir}/UQ-diff-fdiff-parity.png", dpi=self.dpi)
        plt.close()

    def plot_2d_uq_scatter(self, df_uq, scheme, trust_lo, trust_hi, rnd_trust_lo, rnd_trust_hi):
        """Plots 2D scatter of QbC vs RND with filtering boundaries."""
        # 1. Scatter with Max Force Diff as hue
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

        # 2. Scatter with Identity as hue
        if "uq_identity" in df_uq.columns:
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
                          scores_direct, scores_random, df_uq, final_indices):
        """Plots all PCA and DIRECT related figures."""
        # 1. Explained Variance
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, selected_PC_dim+6+1), explained_variance[:selected_PC_dim+6], "o-")
        plt.xlabel(r"i$^{\mathrm{th}}$ PC", size=12)
        plt.ylabel("Explained variance", size=12)
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
        plt.savefig(f"{self.save_dir}/explained_variance.png", dpi=self.dpi)
        plt.close()

        # 2. PCA Feature Coverage (DIRECT vs Random)
        self._plot_coverage(all_features, direct_indices, "DIRECT")
        self._plot_coverage(all_features, random_indices, "Random")

        # 3. Coverage Score Bar Chart
        x = np.arange(len(scores_direct))
        x_ticks = [f"PC {n+1}" for n in range(len(x))]
        plt.figure(figsize=(15, 4))
        plt.bar(x + 0.6, scores_direct, width=0.3, label=rf"DIRECT, $\overline{{\mathrm{{Coverage\ score}}}}$ = {np.mean(scores_direct):.3f}")
        plt.bar(x + 0.3, scores_random, width=0.3, label=rf"Random, $\overline{{\mathrm{{Coverage\ score}}}}$ = {np.mean(scores_random):.3f}")
        plt.xticks(x + 0.45, x_ticks, size=12)
        plt.yticks(np.linspace(0, 1.0, 6), size=12)
        plt.ylabel("Coverage score", size=12)
        plt.legend(shadow=True, loc="lower right", fontsize=12)
        plt.savefig(f"{self.save_dir}/coverage_score.png", dpi=self.dpi)
        plt.close()

        # 4. Final Selection in PCA Space
        # Note: Recalculating PCA on all data for visualization as per original script logic
        from sklearn.decomposition import PCA
        X = df_uq[[col for col in df_uq.columns if col.startswith("desc_stru_")]].values
        pca_vis = PCA(n_components=2)
        PCs_alldata = pca_vis.fit_transform(X)
        
        # Get indices relative to df_uq
        # candidate_indices are subset of all, final_indices are subset of candidate
        # Need to map final_indices (which are indices into candidate df) back to global indices
        # In the original script, final_indices are indices OF the candidate dataframe
        # We need to ensure we use the correct global indices for plotting
        
        # Map back to integer indices for array slicing
        # Assuming df_uq has integer index range(0, N)
        candidate_global_indices = df_uq[df_uq['uq_identity'] == 'candidate'].index
        final_global_indices = final_indices # These should be passed as global indices
        
        plt.figure(figsize=(10, 8))
        plt.scatter(PCs_alldata[:, 0], PCs_alldata[:, 1], marker="*", color="gray", label=f"All {len(df_uq)} structures", alpha=0.7, s=15)
        plt.scatter(PCs_alldata[candidate_global_indices, 0], PCs_alldata[candidate_global_indices, 1], marker="*", color="blue", label=f"UQ sampled {len(candidate_global_indices)}", alpha=0.7, s=30)
        plt.scatter(PCs_alldata[final_global_indices, 0], PCs_alldata[final_global_indices, 1], marker="*", color="red", label=f"UQ-DIRECT sampled {len(final_global_indices)}", s=30)
        plt.title(f"PCA of UQ-DIRECT sampling", fontsize=14)
        plt.xlabel("PC1", size=12)
        plt.ylabel("PC2", size=12)
        plt.legend(frameon=False, fontsize=12, reverse=True)
        plt.savefig(f"{self.save_dir}/Final_sampled_PCAview.png", dpi=self.dpi)
        plt.close()
        
        return pd.DataFrame(PCs_alldata, columns=['PC1', 'PC2'])

    def _plot_coverage(self, all_features, selected_indices, method):
        plt.figure(figsize=(8, 6))
        selected_features = all_features[selected_indices]
        plt.plot(all_features[:, 0], all_features[:, 1], "*", alpha=0.6, label=f"All {len(all_features):,} structures")
        plt.plot(selected_features[:, 0], selected_features[:, 1], "*", alpha=0.6, label=f"{method} sampled {len(selected_features):,}")
        plt.legend(frameon=False, fontsize=10, reverse=True)
        plt.ylabel("PC 2", size=12)
        plt.xlabel("PC 1", size=12)
        plt.savefig(f"{self.save_dir}/{method}_PCA_feature_coverage.png", dpi=self.dpi)
        plt.close()

    def _setup_2d_plot_axes(self, x_lo, x_hi, y_lo, y_hi):
        plt.xlabel("UQ-QbC Value", fontsize=12)
        plt.ylabel("UQ-RND-rescaled Value", fontsize=12)
        plt.grid(True)
        ax = plt.gca()
        ax.xaxis.set_major_locator(mtick.MultipleLocator(0.1))
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
            # Simplified reproduction of the complex crossline logic
            # y1 = (y_hi*x_lo - (y_hi - y_lo)*x)/x_lo
            # y2 = (y_lo*x_hi - y_lo*x)/(x_hi - x_lo)
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
