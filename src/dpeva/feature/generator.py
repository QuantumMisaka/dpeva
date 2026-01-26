import os
import logging
import numpy as np
import dpdata
import subprocess
from dpeva.submission import JobManager, JobConfig

# Optional import for direct inference mode
try:
    from deepmd.infer.deep_pot import DeepPot
    from torch.cuda import empty_cache
    _DEEPMD_AVAILABLE = True
except ImportError:
    _DEEPMD_AVAILABLE = False

class DescriptorGenerator:
    """
    Generates atomic and structural descriptors using a pre-trained DeepPot model.
    Supports two modes:
    1. 'cli' (Recommended): Uses `dp --pt eval-desc` command (supports Local/Slurm).
    2. 'direct' (Legacy): Uses `deepmd.infer` Python API directly.
    """
    
    def __init__(self, model_path, head="OC20M", batch_size=1000, omp_threads=24, 
                 mode="cli", backend="local", slurm_config=None, env_setup=None):
        """
        Initialize the DescriptorGenerator.
        
        Args:
            model_path (str): Path to the frozen DeepMD model file.
            head (str): Head type for multi-head models (default: "OC20M").
            batch_size (int): Batch size for inference (default: 1000). Ignored in 'cli' mode.
            omp_threads (int): Number of OMP threads (default: 24).
            mode (str): 'cli' or 'direct'. 'cli' uses `dp eval-desc`.
            backend (str): 'local' or 'slurm'. Only used in 'cli' mode.
            slurm_config (dict): Configuration for Slurm submission.
            env_setup (str): Environment setup script for job execution.
        """
        self.model_path = os.path.abspath(model_path)
        self.head = head
        self.batch_size = batch_size
        self.omp_threads = omp_threads
        self.mode = mode
        self.backend = backend
        self.slurm_config = slurm_config or {}
        self.env_setup = env_setup or ""
        
        self.logger = logging.getLogger(__name__)
        
        if self.mode == "direct":
            if not _DEEPMD_AVAILABLE:
                raise ImportError("DeepMD-kit not found or not importable. Cannot use 'direct' mode.")
            # Set OMP threads for direct mode
            os.environ['OMP_NUM_THREADS'] = f'{omp_threads}'
            # Load model
            self.model = DeepPot(model_path, head=head)
        elif self.mode == "cli":
            # Initialize JobManager
            self.job_manager = JobManager(mode=backend)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'cli' or 'direct'.")

    # ==========================================
    # CLI Mode Methods (New Implementation)
    # ==========================================

    def run_cli_generation(self, data_path, output_dir, blocking=True):
        """
        Run descriptor generation using `dp --pt eval-desc` via JobManager.
        
        Args:
            data_path (str): Path to the dataset.
            output_dir (str): Directory to save descriptors.
            blocking (bool): Whether to wait for the job to complete (local mode only).
        """
        abs_data_path = os.path.abspath(data_path)
        abs_output_dir = os.path.abspath(output_dir)
        
        # Ensure output directory exists (parent)
        # Note: eval-desc creates the final directory itself if it doesn't exist
        os.makedirs(os.path.dirname(abs_output_dir), exist_ok=True)
        
        # Default env setup if not provided
        if not self.env_setup:
            self.env_setup = f"""
export OMP_NUM_THREADS={self.omp_threads}
"""

        # Construct Command
        # dp --pt eval-desc -s data -m model -o output --head head
        cmd = (
            f"dp --pt eval-desc "
            f"-s {abs_data_path} "
            f"-m {self.model_path} "
            f"-o {abs_output_dir} "
            f"--head {self.head} "
            f"2>&1 | tee eval_desc.log"
        )
        
        # Create JobConfig
        job_name = f"dpa_evaldesc_{os.path.basename(abs_data_path)}"
        
        # Filter out generic keys from slurm_config to avoid duplication or errors
        task_slurm_config = self.slurm_config.copy()
        for k in ["job_name", "output_log", "error_log"]:
            task_slurm_config.pop(k, None)
            
        job_config = JobConfig(
            job_name=job_name,
            command=cmd,
            env_setup=self.env_setup,
            output_log="eval_desc.out",
            error_log="eval_desc.err",
            **task_slurm_config
        )
        
        # Generate Script
        script_name = "run_evaldesc.slurm" if self.backend == "slurm" else "run_evaldesc.sh"
        # We place the script in the parent of output_dir to avoid cluttering results
        # Or create a dedicated task directory?
        # Let's put it in output_dir if possible, but output_dir might be overwritten by dp?
        # dp eval-desc -o output_dir -> output_dir/data_name.npy
        # If output_dir is the final destination, it's safe.
        
        # Better: create a 'scripts' dir or put it alongside results.
        # Let's use output_dir as the working directory for the job.
        os.makedirs(abs_output_dir, exist_ok=True)
        script_path = os.path.join(abs_output_dir, script_name)
        
        self.job_manager.generate_script(job_config, script_path)
        
        self.logger.info(f"Submitting eval-desc job for {data_path}")
        self.job_manager.submit(script_path, working_dir=abs_output_dir)
        
        if blocking and self.backend == "local":
            # For local execution, we might want to wait?
            # JobManager.submit for 'local' currently uses subprocess.run which IS blocking unless nohup used.
            # Our JobManager implementation for local is blocking (subprocess.run).
            pass

    # ==========================================
    # Direct Mode Methods (Legacy Implementation)
    # ==========================================

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

    def compute_descriptors_direct(self, data_path, data_format="deepmd/npy", output_mode="atomic"):
        """
        Compute descriptors for a given dataset using direct API.
        
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
