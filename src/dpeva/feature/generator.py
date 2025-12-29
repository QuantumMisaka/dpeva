import os
import logging
import numpy as np
import dpdata
from deepmd.infer.deep_pot import DeepPot
from torch.cuda import empty_cache

class DescriptorGenerator:
    """
    Generates atomic and structural descriptors using a pre-trained DeepPot model.
    """
    
    def __init__(self, model_path, head="OC20M", batch_size=1000, omp_threads=24):
        """
        Initialize the DescriptorGenerator.
        
        Args:
            model_path (str): Path to the frozen DeepMD model file.
            head (str): Head type for multi-head models (default: "OC20M").
            batch_size (int): Batch size for inference (default: 1000).
            omp_threads (int): Number of OMP threads (default: 24).
        """
        self.model_path = model_path
        self.head = head
        self.batch_size = batch_size
        self.omp_threads = omp_threads
        
        # Set OMP threads
        os.environ['OMP_NUM_THREADS'] = f'{omp_threads}'
        
        # Load model
        self.model = DeepPot(model_path, head=head)
        self.logger = logging.getLogger(__name__)

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

    def compute_descriptors(self, data_path, data_format="deepmd/npy", output_mode="atomic"):
        """
        Compute descriptors for a given dataset.
        
        Args:
            data_path (str): Path to the dataset (system directory).
            data_format (str): Format of the dataset (default: "deepmd/npy").
            output_mode (str): "atomic" (per atom) or "structural" (per frame, mean pooled).
            
        Returns:
            np.ndarray: The computed descriptors.
        """
        self.logger.info(f"Loading data from {data_path} with format {data_format}")
        
        if data_format == "deepmd/npy/mixed":
            onedata = dpdata.MultiSystems.from_file(data_path, fmt=data_format)
        else:
            onedata = dpdata.System(data_path, fmt=data_format)
            
        desc_list = []
        if data_format == "deepmd/npy/mixed":
            for onesys in onedata:
                nopbc = onesys.data.get('nopbc', False)
                one_desc_list = self._get_desc_by_batch(onesys, nopbc)
                desc_list.extend(one_desc_list)
        else:
            nopbc = onedata.data.get('nopbc', False)
            desc_list = self._get_desc_by_batch(onedata, nopbc)
            
        desc = np.concatenate(desc_list, axis=0)
        
        if output_mode == "structural":
            desc = np.mean(desc, axis=1)
            
        # Clear memory
        del onedata
        empty_cache()
        
        return desc
