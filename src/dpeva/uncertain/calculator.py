import numpy as np
from sklearn.preprocessing import StandardScaler

class UQCalculator:
    """Calculates Uncertainty Quantification (UQ) metrics from model predictions."""

    def __init__(self):
        """Initializes the UQCalculator."""
        pass

    def compute_qbc_rnd(self, predictions_0, predictions_1, predictions_2, predictions_3):
        """
        Computes QbC and RND uncertainty metrics.

        Args:
            predictions_0: Predictions from model 0 (main model).
            predictions_1: Predictions from model 1.
            predictions_2: Predictions from model 2.
            predictions_3: Predictions from model 3.

        Returns:
            dict: Dictionary containing:
                - uq_qbc_for: QbC force uncertainty (max per structure).
                - uq_rnd_for: RND force uncertainty (max per structure).
                - diff_maxf_0_frame: Max force difference between model 0 and ground truth (if available).
                - diff_rmsf_0_frame: RMS force difference between model 0 and ground truth (if available).
                - fx_expt, fy_expt, fz_expt: Ensemble mean forces.
        """
        # Unpack predictions
        # Model 0
        fx_0, fy_0, fz_0 = predictions_0.data_f['pred_fx'], predictions_0.data_f['pred_fy'], predictions_0.data_f['pred_fz']
        # Ensemble models
        fx_1, fy_1, fz_1 = predictions_1.data_f['pred_fx'], predictions_1.data_f['pred_fy'], predictions_1.data_f['pred_fz']
        fx_2, fy_2, fz_2 = predictions_2.data_f['pred_fx'], predictions_2.data_f['pred_fy'], predictions_2.data_f['pred_fz']
        fx_3, fy_3, fz_3 = predictions_3.data_f['pred_fx'], predictions_3.data_f['pred_fy'], predictions_3.data_f['pred_fz']

        # Calculate ensemble mean (pseudo-experimental)
        fx_expt = np.mean((fx_1, fx_2, fx_3), axis=0)
        fy_expt = np.mean((fy_1, fy_2, fy_3), axis=0)
        fz_expt = np.mean((fz_1, fz_2, fz_3), axis=0)

        # 1. QbC Force UQ (Variance of committee members 1, 2, 3)
        fx_qbc_sq_diff = np.mean(((fx_1 - fx_expt)**2, (fx_2 - fx_expt)**2, (fx_3 - fx_expt)**2), axis=0)
        fy_qbc_sq_diff = np.mean(((fy_1 - fy_expt)**2, (fy_2 - fy_expt)**2, (fy_3 - fy_expt)**2), axis=0)
        fz_qbc_sq_diff = np.mean(((fz_1 - fz_expt)**2, (fz_2 - fz_expt)**2, (fz_3 - fz_expt)**2), axis=0)
        f_qbc_stddiff = np.sqrt(fx_qbc_sq_diff + fy_qbc_sq_diff + fz_qbc_sq_diff)

        # 2. RND Force UQ (Deviation of committee members from main model 0)
        fx_rnd_sq_diff = np.mean(((fx_1 - fx_0)**2, (fx_2 - fx_0)**2, (fx_3 - fx_0)**2), axis=0)
        fy_rnd_sq_diff = np.mean(((fy_1 - fy_0)**2, (fy_2 - fy_0)**2, (fy_3 - fy_0)**2), axis=0)
        fz_rnd_sq_diff = np.mean(((fz_1 - fz_0)**2, (fz_2 - fz_0)**2, (fz_3 - fz_0)**2), axis=0)
        f_rnd_stddiff = np.sqrt(fx_rnd_sq_diff + fy_rnd_sq_diff + fz_rnd_sq_diff)

        # Aggregate per structure (max atomic UQ)
        uq_qbc_for_list = []
        uq_rnd_for_list = []
        
        # Calculate force difference for model 0 if labels exist (for visualization)
        diff_f_0 = None
        if getattr(predictions_0, 'has_ground_truth', True) and predictions_0.diff_fx is not None:
             diff_f_0 = np.sqrt(predictions_0.diff_fx**2 + predictions_0.diff_fy**2 + predictions_0.diff_fz**2)
             
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

        return {
            "uq_qbc_for": np.array(uq_qbc_for_list),
            "uq_rnd_for": np.array(uq_rnd_for_list),
            "diff_maxf_0_frame": np.array(diff_maxf_0_frame),
            "diff_rmsf_0_frame": np.array(diff_rmsf_0_frame),
            "fx_expt": fx_expt,
            "fy_expt": fy_expt, 
            "fz_expt": fz_expt
        }

    def align_scales(self, uq_qbc, uq_rnd):
        """
        Aligns RND UQ scale to QbC UQ scale using RobustScaler (Median/IQR).
        
        Args:
            uq_qbc: QbC uncertainty values.
            uq_rnd: RND uncertainty values.
            
        Returns:
            uq_rnd_rescaled: RND values rescaled to match QbC distribution.
        """
        from sklearn.preprocessing import RobustScaler
        
        scaler_qbc = RobustScaler()
        scaler_rnd = RobustScaler()
        
        uq_qbc_scaled = scaler_qbc.fit_transform(uq_qbc.reshape(-1, 1)).flatten()
        uq_rnd_scaled = scaler_rnd.fit_transform(uq_rnd.reshape(-1, 1)).flatten()
        
        uq_rnd_rescaled = scaler_qbc.inverse_transform(uq_rnd_scaled.reshape(-1, 1)).flatten()
        return uq_rnd_rescaled

    def calculate_trust_lo(self, data, ratio=0.5, grid_size=1000, bound=(0, 2.0)):
        """
        Automatically determines the lower bound of the trust region based on KDE.
        Finds the x-value on the right side of the peak where density drops to ratio * peak_density.
        
        Args:
            data: The uncertainty data (1D array).
            ratio: The ratio of peak density to define the cutoff (default 0.5).
            grid_size: Number of points for KDE evaluation grid.
            bound: Tuple of (min, max) for the grid range.
            
        Returns:
            float: The calculated uq_trust_lo value.
        """
        from scipy.stats import gaussian_kde
        import logging
        logger = logging.getLogger(__name__)
        
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
