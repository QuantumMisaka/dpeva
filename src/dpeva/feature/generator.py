import os
import logging
import numpy as np
import dpdata

# Optional import for direct inference mode
try:
    from deepmd.infer.deep_pot import DeepPot
    from torch.cuda import empty_cache
    _DEEPMD_AVAILABLE = True
except ImportError as e:
    import logging
    logging.getLogger("dpeva.feature.generator").warning(f"DeepMD import failed: {e}")
    _DEEPMD_AVAILABLE = False
    def empty_cache():
        """Dummy empty_cache function when torch is not available."""
        pass

from dpeva.io.dataset import load_systems

class DescriptorGenerator:
    """
    Core Domain Service for calculating atomic descriptors using DeepPot.
    Pure calculation engine, decoupled from Job Execution/Submission.
    """
    
    def __init__(self, model_path, head="OC20M", batch_size=1000, omp_threads=1):
        """
        Initialize the DescriptorGenerator.
        
        Args:
            model_path (str): Path to the frozen DeepMD model file.
            head (str): Head type for multi-head models (default: "OC20M").
            batch_size (int): Batch size for inference (default: 1000).
            omp_threads (int): Number of OMP threads (default: 1).
        """
        self.model_path = os.path.abspath(model_path)
        self.head = head
        self.batch_size = batch_size
        self.omp_threads = omp_threads
        self.logger = logging.getLogger(__name__)
        
        if not _DEEPMD_AVAILABLE:
            self.logger.warning("DeepMD-kit not available. Python mode calculation will fail.")
            self.model = None
        else:
            # Set OMP threads for python mode
            os.environ['OMP_NUM_THREADS'] = f'{omp_threads}'
            # Load model
            try:
                self.model = DeepPot(self.model_path, head=head)
            except Exception as e:
                self.logger.error(f"Failed to load DeepPot model: {e}")
                raise

    def _descriptor_from_model(self, sys: dpdata.System, nopbc=False) -> np.ndarray:
        """Calculate descriptors for a single system."""
        coords = sys.data["coords"]
        cells = sys.data["cells"]
        if nopbc:
            cells = None
        
        model_type_map = self.model.get_type_map()
        type_trans = np.array([model_type_map.index(i) for i in sys.data['atom_names']])
        atypes = list(type_trans[sys.data['atom_types']])
        
        predict = self.model.eval_descriptor(coords, cells, atypes)
        return predict

    def _get_desc_by_batch(self, sys: dpdata.System, nopbc=False) -> list:
        """Calculate descriptors in batches."""
        desc_list = []
        for i in range(0, len(sys), self.batch_size):
            batch = sys[i:i + self.batch_size]  
            desc_batch = self._descriptor_from_model(batch, nopbc=nopbc)
            desc_list.append(desc_batch)
        return desc_list

    def compute_descriptors(self, data_path, output_mode="atomic"):
        """
        Compute descriptors for a given dataset (single system).
        
        Args:
            data_path (str): Path to the dataset (system directory).
            output_mode (str): "atomic" (per atom) or "structural" (per frame, mean pooled).
            
        Returns:
            np.ndarray: The computed descriptors.
        """
        sys_name = os.path.basename(data_path)
        
        if output_mode != "atomic":
            self.logger.warning(f"Output mode is '{output_mode}'. Note that only 'atomic' mode is consistent with 'dp eval-desc' CLI output.")
            
        # Use dpeva.io.dataset.load_systems for unified data loading
        # Note: fmt is "auto" by default in load_systems if not specified
        systems = load_systems(data_path)
        
        if not systems:
            raise ValueError(f"No valid systems found in {data_path}")
            
        n_frames = sum(len(s) for s in systems)
        self.logger.info(f"Processing system: {sys_name} ({n_frames} frames)")
            
        desc_list = []
        for s in systems:
            nopbc = s.data.get('nopbc', False)
            desc_list.extend(self._get_desc_by_batch(s, nopbc))
        
        if not desc_list:
            return np.array([])
        
        desc = np.concatenate(desc_list, axis=0)
        
        if output_mode == "structural":
            desc = np.mean(desc, axis=1)
            
        # Clear memory
        del systems
        empty_cache()
        
        return desc
