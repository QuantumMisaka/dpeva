import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional

class InferenceVisualizer:
    """
    Visualization tools for inference results (Parity Plots, Error Distributions).
    """
    
    def __init__(self, output_dir: str, dpi: int = 150):
        self.output_dir = output_dir
        self.dpi = dpi
        
        # Style settings
        from dpeva.utils.visual_style import set_visual_style
        set_visual_style()
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def plot_parity(self, y_true: np.ndarray, y_pred: np.ndarray, 
                   label: str, unit: str, title: str = None):
        """
        Plot Parity (Diagonal) plot: Predicted vs Ground Truth.
        """
        plt.figure(figsize=(6, 6))
        
        # Determine limits
        vmin = min(y_true.min(), y_pred.min())
        vmax = max(y_true.max(), y_pred.max())
        margin = (vmax - vmin) * 0.05
        vmin -= margin
        vmax += margin
        
        plt.plot([vmin, vmax], [vmin, vmax], 'k--', lw=1.5, alpha=0.7)
        
        # Scatter
        # Use rasterized=True for large datasets to keep file size small
        plt.scatter(y_true, y_pred, alpha=0.3, s=10, c='blue', edgecolors='none', rasterized=True)
        
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

    def plot_distribution(self, data: np.ndarray, label: str, unit: str, 
                          color: str = 'blue', title: str = None, 
                          highlight_outliers: bool = False, outlier_mask: Optional[np.ndarray] = None):
        """
        Plot KDE distribution of data.
        """
        plt.figure(figsize=(8, 6))
        
        # Main KDE
        sns.histplot(data, kde=True, stat="density", color=color, alpha=0.3, label="All Data", element="step")
        
        if highlight_outliers and outlier_mask is not None and np.any(outlier_mask):
            # Plot non-outliers separately to show the 'clean' distribution
            clean_data = data[~outlier_mask]
            sns.kdeplot(clean_data, color='green', linestyle='--', label="Clean Data (No Outliers)")
            
        plt.xlabel(f"{label} ({unit})")
        plt.ylabel("Density")
        
        if title:
            plt.title(title)
        else:
            plt.title(f"{label} Distribution")
            
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = f"dist_{label.lower().replace(' ', '_')}.png"
        plt.savefig(os.path.join(self.output_dir, filename), dpi=self.dpi)
        plt.close()

    def plot_error_distribution(self, error: np.ndarray, label: str, unit: str):
        """
        Plot error distribution (Predicted - True).
        """
        plt.figure(figsize=(8, 6))
        
        sns.histplot(error, kde=True, stat="density", color='red', alpha=0.3)
        
        plt.axvline(0, color='k', linestyle='--', lw=1)
        plt.xlabel(f"{label} Error ({unit})")
        plt.ylabel("Density")
        plt.title(f"{label} Error Distribution")
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = f"error_dist_{label.lower().replace(' ', '_')}.png"
        plt.savefig(os.path.join(self.output_dir, filename), dpi=self.dpi)
        plt.close()
