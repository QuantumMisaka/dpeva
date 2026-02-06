from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Any

import numpy as np
from scipy.stats import gaussian_kde
from sklearn.preprocessing import RobustScaler

if TYPE_CHECKING:
    from dpeva.io.types import PredictionData

logger = logging.getLogger(__name__)

class UQCalculator:
    """Calculates Uncertainty Quantification (UQ) metrics from model predictions."""

    def __init__(self) -> None:
        """Initializes the UQCalculator."""
        pass

    def compute_qbc_rnd(
        self,
        predictions: List[PredictionData]
    ) -> Dict[str, np.ndarray]:
        """
        Computes QbC and RND uncertainty metrics for N models (N >= 2).
        
        Model 0 (predictions[0]) is treated as the 'main' model or 'baseline'.
        Models 1..N-1 are treated as the 'committee' or 'ensemble' members.

        Args:
            predictions: List of PredictionData objects. Must contain at least 2 models.
                         predictions[0] is the main model.
                         predictions[1:] are the committee members.

        Returns:
            dict: Dictionary containing:
                - uq_qbc_for: QbC force uncertainty (max per structure).
                - uq_rnd_for: RND force uncertainty (max per structure).
                - diff_maxf_0_frame: Max force difference between model 0 and ground truth (if available).
                - diff_rmsf_0_frame: RMS force difference between model 0 and ground truth (if available).
                - fx_expt, fy_expt, fz_expt: Ensemble mean forces (of models 1..N-1).
        """
        if len(predictions) < 2:
            raise ValueError(f"Need at least 2 models for UQ (got {len(predictions)}).")

        # Unpack predictions
        # Helper to extract force components
        def get_forces(pred: PredictionData):
            if pred.force is not None:
                return pred.force['pred_fx'], pred.force['pred_fy'], pred.force['pred_fz']
            return None, None, None

        # Extract forces for all models
        forces_list = [get_forces(p) for p in predictions]
        
        # Check if forces are present
        if any(fx is None for fx, _, _ in forces_list):
             raise ValueError("Some models do not have force predictions.")

        # Model 0 (Baseline)
        fx_0, fy_0, fz_0 = forces_list[0]
        
        # Committee Models (1..N-1)
        committee_forces = forces_list[1:]
        
        # Stack for vectorized operations
        # shape: (n_committee, n_atoms_total)
        fx_comm = np.vstack([f[0] for f in committee_forces])
        fy_comm = np.vstack([f[1] for f in committee_forces])
        fz_comm = np.vstack([f[2] for f in committee_forces])

        # Calculate ensemble mean (pseudo-experimental) - axis=0 implies averaging over committee members
        fx_expt = np.mean(fx_comm, axis=0)
        fy_expt = np.mean(fy_comm, axis=0)
        fz_expt = np.mean(fz_comm, axis=0)

        # 1. QbC Force UQ (Variance of committee members)
        # Calculate squared differences from mean
        fx_qbc_sq_diff = np.mean((fx_comm - fx_expt)**2, axis=0)
        fy_qbc_sq_diff = np.mean((fy_comm - fy_expt)**2, axis=0)
        fz_qbc_sq_diff = np.mean((fz_comm - fz_expt)**2, axis=0)
        
        # Clamp sum of squares to 0 to prevent negative values from float precision errors
        sum_qbc_sq = np.maximum(fx_qbc_sq_diff + fy_qbc_sq_diff + fz_qbc_sq_diff, 0.0)
        f_qbc_stddiff = np.sqrt(sum_qbc_sq)

        # 2. RND Force UQ (Deviation of committee members from main model 0)
        # Broadcast model 0 forces for subtraction
        fx_rnd_sq_diff = np.mean((fx_comm - fx_0)**2, axis=0)
        fy_rnd_sq_diff = np.mean((fy_comm - fy_0)**2, axis=0)
        fz_rnd_sq_diff = np.mean((fz_comm - fz_0)**2, axis=0)
        
        # Clamp sum of squares to 0
        sum_rnd_sq = np.maximum(fx_rnd_sq_diff + fy_rnd_sq_diff + fz_rnd_sq_diff, 0.0)
        f_rnd_stddiff = np.sqrt(sum_rnd_sq)


        # Aggregate per structure (max atomic UQ)
        uq_qbc_for_list = []
        uq_rnd_for_list = []
        
        # Calculate force difference for model 0 if labels exist (for visualization)
        predictions_0 = predictions[0] # Alias for readability
        diff_f_0 = None
        
        # Calculate diff manually from raw data if present, instead of relying on .diff_fx attribute
        if predictions_0.has_ground_truth and predictions_0.force is not None:
             dfx = predictions_0.force['pred_fx'] - predictions_0.force['data_fx']
             dfy = predictions_0.force['pred_fy'] - predictions_0.force['data_fy']
             dfz = predictions_0.force['pred_fz'] - predictions_0.force['data_fz']
             diff_f_0 = np.sqrt(dfx**2 + dfy**2 + dfz**2)
             
        diff_maxf_0_frame = []
        diff_rmsf_0_frame = []

        index = 0
        for item in predictions_0.dataname_list:
            natom = item[2]
            # QbC
            uq_qbc_for_list.append(np.max(f_qbc_stddiff[index:index + natom]))
            # RND
            uq_rnd_for_list.append(np.max(f_rnd_stddiff[index:index + natom]))
            # Diff 0
            if diff_f_0 is not None:
                diff_item = diff_f_0[index:index + natom]
                diff_maxf_0_frame.append(np.max(diff_item))
                diff_rmsf_0_frame.append(np.sqrt(np.mean(diff_item**2)))
            else:
                diff_maxf_0_frame.append(0.0)
                diff_rmsf_0_frame.append(0.0)
            
            index += natom

        # Convert to numpy arrays and replace NaNs with 0.0 (Clean step)
        uq_qbc_for = np.array(uq_qbc_for_list)
        uq_rnd_for = np.array(uq_rnd_for_list)
        diff_maxf_0 = np.array(diff_maxf_0_frame)
        diff_rmsf_0 = np.array(diff_rmsf_0_frame)
        
        # Check for NaNs and log warning if found
        if np.isnan(uq_qbc_for).any() or np.isnan(uq_rnd_for).any():
             logger.warning("NaNs detected in UQ calculation. Replacing with Infinity (High Uncertainty).")
             uq_qbc_for = np.nan_to_num(uq_qbc_for, nan=np.inf)
             uq_rnd_for = np.nan_to_num(uq_rnd_for, nan=np.inf)
        
        return {
            "uq_qbc_for": uq_qbc_for,
            "uq_rnd_for": uq_rnd_for,
            "diff_maxf_0_frame": diff_maxf_0,
            "diff_rmsf_0_frame": diff_rmsf_0,
            "fx_expt": fx_expt,
            "fy_expt": fy_expt, 
            "fz_expt": fz_expt
        }

    def align_scales(self, uq_qbc: np.ndarray, uq_rnd: np.ndarray) -> np.ndarray:
        """
        Aligns RND UQ scale to QbC UQ scale using RobustScaler logic (Median/IQR).
        Manually implemented to handle Infinity values robustly.
        
        Args:
            uq_qbc: QbC uncertainty values.
            uq_rnd: RND uncertainty values.
            
        Returns:
            uq_rnd_rescaled: RND values rescaled to match QbC distribution.
        """
        def get_robust_stats(data: np.ndarray) -> Tuple[float, float]:
            """Calculates median and IQR, ignoring non-finite values."""
            valid_mask = np.isfinite(data)
            if not np.any(valid_mask):
                return 0.0, 1.0 # Fallback for all-inf/nan data
            
            clean_data = data[valid_mask]
            q25, median, q75 = np.percentile(clean_data, [25, 50, 75])
            iqr = q75 - q25
            
            # Prevent division by zero if IQR is 0 (e.g. constant data)
            if iqr == 0:
                iqr = 1.0
                
            return median, iqr

        med_qbc, iqr_qbc = get_robust_stats(uq_qbc)
        med_rnd, iqr_rnd = get_robust_stats(uq_rnd)
        
        # Transform RND to standard scale: (x - med) / iqr
        # Then inverse transform to QbC scale: y * iqr_new + med_new
        # Combined: result = (rnd - med_rnd) / iqr_rnd * iqr_qbc + med_qbc
        
        # Operations with Inf will correctly result in Inf (or NaN if Inf/Inf, but iqr is finite)
        uq_rnd_rescaled = (uq_rnd - med_rnd) / iqr_rnd * iqr_qbc + med_qbc
        
        return uq_rnd_rescaled

    def calculate_trust_lo(
        self,
        data: np.ndarray,
        ratio: float = 0.5,
        grid_size: int = 1000,
        bound: Tuple[float, float] = (0, 2.0)
    ) -> Optional[float]:
        """
        Automatically determines the lower bound of the trust region based on KDE.
        Finds the x-value on the right side of the peak where density drops to ratio * peak_density.
        
        Args:
            data: The uncertainty data (1D array).
            ratio: The ratio of peak density to define the cutoff (default 0.5).
            grid_size: Number of points for KDE evaluation grid.
            bound: Tuple of (min, max) for the grid range.
            
        Returns:
            float: The calculated uq_trust_lo value, or None if calculation fails.
        """
        # Filter valid data within bounds for KDE
        valid_mask = (data >= bound[0]) & (data <= bound[1])
        clean_data = data[valid_mask]
        
        if len(clean_data) < 2:
            return None # Return None to signal fallback
            
        try:
            kde = gaussian_kde(clean_data)
            x_grid = np.linspace(bound[0], bound[1], grid_size)
            y_grid = kde(x_grid)
            
            # Find peak
            peak_idx = np.argmax(y_grid)
            peak_density = y_grid[peak_idx]
            target_density = peak_density * ratio
            
            # Search to the right of the peak
            right_side = y_grid[peak_idx:]
            
            # Find indices where density is below target
            below_target = np.where(right_side < target_density)[0]
            
            if len(below_target) > 0:
                # The first point on the right
                idx_offset = below_target[0]
                final_idx = peak_idx + idx_offset
                return float(x_grid[final_idx])
            else:
                # If never drops below target within bound, return the bound
                return float(bound[1])
        except Exception as e:
            # Fallback in case of KDE failure (e.g. singular matrix)
            logger.warning(f"KDE calculation failed ({e}).")
            return None
