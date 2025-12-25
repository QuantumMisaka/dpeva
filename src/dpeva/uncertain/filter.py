import numpy as np

class UQFilter:
    """Filters data based on Uncertainty Quantification (UQ) metrics."""

    def __init__(self, scheme="tangent_lo", trust_lo=0.12, trust_hi=0.22, 
                 rnd_trust_lo=None, rnd_trust_hi=None):
        self.scheme = scheme
        self.trust_lo = trust_lo
        self.trust_hi = trust_hi
        self.rnd_trust_lo = rnd_trust_lo if rnd_trust_lo is not None else trust_lo
        self.rnd_trust_hi = rnd_trust_hi if rnd_trust_hi is not None else trust_hi

        self.supported_schemes = ["strict", "circle_lo", "crossline_lo", "tangent_lo", "loose"]
        if self.scheme not in self.supported_schemes:
            raise ValueError(f"UQ selection scheme {self.scheme} not supported! Choose from {self.supported_schemes}.")
        
        if self.trust_lo >= self.trust_hi or self.rnd_trust_lo >= self.rnd_trust_hi:
             raise ValueError("Low trust threshold should be lower than High trust threshold!")

    def filter(self, df_uq, qbc_col="uq_qbc_for", rnd_col="uq_rnd_for_rescaled"):
        """
        Applies the filtering scheme to classify data into candidate, accurate, and failed.

        Args:
            df_uq: DataFrame containing UQ metrics.
            qbc_col: Column name for QbC UQ.
            rnd_col: Column name for RND UQ (usually rescaled).

        Returns:
            tuple: (df_candidate, df_accurate, df_failed) DataFrames.
        """
        uq_x = df_uq[qbc_col]
        uq_y = df_uq[rnd_col]
        
        uq_x_lo, uq_x_hi = self.trust_lo, self.trust_hi
        uq_y_lo, uq_y_hi = self.rnd_trust_lo, self.rnd_trust_hi

        mask_candidate = None
        mask_accurate = None
        mask_failed = (uq_x > uq_x_hi) | (uq_y > uq_y_hi)

        if self.scheme == "strict":
            mask_candidate = (
                (uq_x >= uq_x_lo) & (uq_x <= uq_x_hi) & 
                (uq_y >= uq_y_lo) & (uq_y <= uq_y_hi)
            )
            mask_accurate = (
                ((uq_x < uq_x_lo) & (uq_y < uq_y_hi)) |
                ((uq_x < uq_x_hi) & (uq_y < uq_y_lo))
            )

        elif self.scheme == "circle_lo":
            # Inside the bounding box AND inside the circle centered at (hi, hi)
            in_box = (uq_x <= uq_x_hi) & (uq_y <= uq_y_hi)
            dist_sq = (uq_x - uq_x_hi)**2 + (uq_y - uq_y_hi)**2
            radius_sq = (uq_x_lo - uq_x_hi)**2 + (uq_y_lo - uq_y_hi)**2
            
            mask_candidate = in_box & (dist_sq <= radius_sq)
            mask_accurate = (dist_sq > radius_sq) & (uq_x < uq_x_hi) & (uq_y < uq_y_hi)

        elif self.scheme == "tangent_lo":
            # Inside bounding box AND below the tangent line
            in_box = (uq_x <= uq_x_hi) & (uq_y <= uq_y_hi)
            # Tangent line condition: dot product <= 0
            tangent_cond = (uq_x - uq_x_lo)*(uq_x_lo - uq_x_hi) + (uq_y - uq_y_lo)*(uq_y_lo - uq_y_hi) <= 0
            
            mask_candidate = in_box & tangent_cond
            mask_accurate = (~tangent_cond) & (uq_x < uq_x_hi) & (uq_y < uq_y_hi)

        elif self.scheme == "crossline_lo":
             mask_candidate = (
                (uq_x <= uq_x_hi) & 
                (uq_y <= uq_y_hi) &
                (uq_x_lo * uq_y + (uq_y_hi - uq_y_lo) * uq_x >= uq_x_lo * uq_y_hi) &
                (uq_x * uq_y_lo + (uq_x_hi - uq_x_lo) * uq_y >= uq_x_hi * uq_y_lo)
            )
             mask_accurate = (
                (uq_x_lo * uq_y + (uq_y_hi - uq_y_lo) * uq_x < uq_x_lo * uq_y_hi) |
                (uq_x * uq_y_lo + (uq_x_hi - uq_x_lo) * uq_y < uq_x_hi * uq_y_lo)
            )

        elif self.scheme == "loose":
            mask_candidate = (
                ((uq_x >= uq_x_lo) & (uq_x <= uq_x_hi)) | 
                ((uq_y >= uq_y_lo) & (uq_y <= uq_y_hi))
            )
            mask_accurate = (uq_x < uq_x_lo) & (uq_y < uq_y_lo)

        df_candidate = df_uq[mask_candidate].copy()
        df_accurate = df_uq[mask_accurate].copy()
        df_failed = df_uq[mask_failed].copy()

        return df_candidate, df_accurate, df_failed

    def get_identity_labels(self, df_uq, df_candidate, df_accurate):
        """Adds a column 'uq_identity' to the DataFrame."""
        df_uq['uq_identity'] = np.where(df_uq['dataname'].isin(df_candidate['dataname']), 'candidate',
                                      np.where(df_uq['dataname'].isin(df_accurate['dataname']), 'accurate', 'failed'))
        return df_uq
