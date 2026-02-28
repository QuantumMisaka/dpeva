from dataclasses import dataclass
from typing import Optional, List, Dict
import numpy as np

@dataclass
class PredictionData:
    """
    Standardized data structure for model predictions.
    Replaces the legacy DPTestResults object.
    """
    # Raw Data from Parser
    energy: Optional[np.ndarray] = None  # Structured array with 'data_e', 'pred_e'
    force: Optional[np.ndarray] = None   # Structured array with 'data_fx', 'pred_fx', etc.
    virial: Optional[np.ndarray] = None  # Structured array
    
    # Metadata
    has_ground_truth: bool = False
    dataname_list: List = None           # List of [name, index, natom]
    datanames_nframe: Dict = None        # Dict of {name: n_frames}
    
    # Processed/Convenience Accessors (Populated post-init if needed, or properties)
    # Ideally we keep this pure data.
    
    @property
    def data_e(self):
        """Alias for energy to maintain some compatibility during migration."""
        return self.energy

    @property
    def data_f(self):
        """Alias for force."""
        return self.force
