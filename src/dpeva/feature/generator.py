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
    2. 'python' (Native): Uses `deepmd.infer` Python API directly.
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
            mode (str): 'cli' or 'python'. 'cli' uses `dp eval-desc`.
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
        
        if self.mode == "python":
            if self.backend == "local":
                if not _DEEPMD_AVAILABLE:
                    raise ImportError("DeepMD-kit not found or not importable. Cannot use 'python' mode.")
                # Set OMP threads for python mode
                os.environ['OMP_NUM_THREADS'] = f'{omp_threads}'
                # Load model
                self.model = DeepPot(model_path, head=head)
            else:
                # For Slurm backend, we don't load the model here.
                # It will be loaded in the generated worker script.
                self.job_manager = JobManager(mode=backend)
        elif self.mode == "cli":
            # Initialize JobManager
            self.job_manager = JobManager(mode=backend)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'cli' or 'python'.")

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
        )
        
        # Only use tee for local backend to avoid redundant logs in Slurm mode
        # In Slurm mode, output is captured via #SBATCH -o (output_log)
        if self.backend == "local":
            cmd += f"2>&1 | tee eval_desc.log"
        
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
            output_log="eval_desc.log", # Unified log name
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
    # Python Mode Methods (Native Implementation)
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

    def run_python_generation(self, data_path, output_dir, data_format="deepmd/npy", output_mode="atomic"):
        """
        Run descriptor generation in 'python' mode.
        Recursively handles multi-level directory structures.
        """
        abs_data_path = os.path.abspath(data_path)
        abs_output_dir = os.path.abspath(output_dir)
        os.makedirs(abs_output_dir, exist_ok=True)
        
        if self.backend == "local":
            self.logger.info("Running python descriptor generation locally...")
            
            # Recursive function to handle nested structures
            def process_recursive(current_path, current_output_dir):
                # Check if leaf system
                is_leaf = os.path.exists(os.path.join(current_path, "type.raw")) or \
                          os.path.exists(os.path.join(current_path, "set.000"))
                          
                if is_leaf:
                    sys_name = os.path.basename(current_path)
                    try:
                        desc = self.compute_descriptors_python(
                            data_path=current_path,
                            data_format=data_format,
                            output_mode=output_mode
                        )
                        # For leaf, we save as sys_name.npy in the PARENT's output dir?
                        # No, usually output_dir/sys_name.npy
                        # But if we recursed, current_output_dir is already specific?
                        # Let's see how run_cli logic works: process_dir(src, out/dirname)
                        # So for leaf src/A, output is out/A.
                        # dp eval-desc -o out/A -> creates out/A.npy or out/A/desc.npy?
                        # dp eval-desc -o outdir -> outdir/system.npy
                        
                        # So here we should save to current_output_dir/sys_name.npy?
                        # If current_path is passed as 'data_path', we expect output in 'output_dir'.
                        # The recursion passes 'out_dir/dirname'.
                        # So if we are at leaf, we are at 'Dataset/System'.
                        # The output dir is 'Output/Dataset/System'.
                        # But we want 'Output/Dataset/System.npy'.
                        
                        # Let's adjust recursion logic to match CLI script logic
                        pass 
                    except Exception as e:
                        self.logger.error(f"Failed to process {sys_name}: {e}")
                        return

                    # Actually, we should just call compute and save.
                    # Wait, compute_descriptors_python takes path and returns array.
                    # We need to decide where to save.
                    # If we are strictly following "Root -> Dataset -> System"
                    # We want "Output/Dataset/System.npy"
                    
                    # Logic:
                    # If we are at Root. List Datasets.
                    # For each Dataset, create Output/Dataset.
                    # For each System in Dataset, save Output/Dataset/System.npy.
                    
                    # But what if structure is just Root -> System?
                    # Then Output/System.npy.
                    
                    # So:
                    out_file = current_output_dir + ".npy"
                    # But wait, current_output_dir is a directory path.
                    # We want to save a file.
                    np.save(out_file, desc)
                    return

                # If not leaf, iterate
                subdirs = [d for d in os.listdir(current_path) if os.path.isdir(os.path.join(current_path, d))]
                for d in subdirs:
                    process_recursive(os.path.join(current_path, d), os.path.join(current_output_dir, d))

            # Trigger recursion
            # But handle the initial call carefully.
            # If data_path is leaf, save to output_dir/basename.npy?
            # Or output_dir.npy?
            
            if os.path.exists(os.path.join(abs_data_path, "type.raw")) or \
               os.path.exists(os.path.join(abs_data_path, "set.000")):
                # Single system
                desc = self.compute_descriptors_python(abs_data_path, data_format, output_mode)
                np.save(os.path.join(abs_output_dir, os.path.basename(abs_data_path) + ".npy"), desc)
            else:
                # Iterate subdirectories
                subdirs = [d for d in os.listdir(abs_data_path) if os.path.isdir(os.path.join(abs_data_path, d))]
                for d in subdirs:
                    # Create corresponding output dir if it's a folder-of-folders
                    # We don't know yet if 'd' is leaf.
                    # Let process_recursive decide.
                    # If 'd' is leaf (System), we want output_dir/d.npy
                    # If 'd' is Dataset, we want output_dir/d/System.npy
                    
                    process_recursive(os.path.join(abs_data_path, d), os.path.join(abs_output_dir, d))
                    
        elif self.backend == "slurm":
            self.logger.info("Preparing to submit python descriptor generation job via Slurm...")
            
            # 1. Generate Worker Script
            worker_script_content = f"""
import os
import sys
import numpy as np

# Ensure dpeva is in path, only used in develop
sys.path.append("{os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))}")

from dpeva.feature.generator import DescriptorGenerator

def main():
    generator = DescriptorGenerator(
        model_path="{self.model_path}",
        head="{self.head}",
        batch_size={self.batch_size},
        omp_threads={self.omp_threads},
        mode="python",
        backend="local"
    )
    
    generator.run_python_generation(
        data_path="{abs_data_path}",
        output_dir="{abs_output_dir}",
        data_format="{data_format}",
        output_mode="{output_mode}"
    )

if __name__ == "__main__":
    main()
"""
            worker_script_path = os.path.join(abs_output_dir, "run_desc_worker.py")
            with open(worker_script_path, "w") as f:
                f.write(worker_script_content.strip())
            
            # 2. Generate Slurm Script
            job_name = f"dpeva_py_desc_{os.path.basename(abs_data_path)}"
            
            # Filter slurm config
            task_slurm_config = self.slurm_config.copy()
            for k in ["job_name", "output_log", "error_log"]:
                task_slurm_config.pop(k, None)
                
            # Env setup
            if not self.env_setup:
                self.env_setup = f"export OMP_NUM_THREADS={self.omp_threads}"
                
            cmd = f"python3 {worker_script_path}"
            
            job_config = JobConfig(
                job_name=job_name,
                command=cmd,
                env_setup=self.env_setup,
                output_log="eval_desc_py.log",
                error_log="eval_desc_py.err",
                **task_slurm_config
            )
            
            script_name = "run_evaldesc_py.slurm"
            script_path = os.path.join(abs_output_dir, script_name)
            
            self.job_manager.generate_script(job_config, script_path)
            
            # 3. Submit
            self.logger.info(f"Submitting python mode job for {data_path}")
            self.job_manager.submit(script_path, working_dir=abs_output_dir)

    def compute_descriptors_python(self, data_path, data_format="deepmd/npy", output_mode="atomic"):
        """
        Compute descriptors for a given dataset using native Python API.
        
        Args:
            data_path (str): Path to the dataset (system directory).
            data_format (str): Format of the dataset (default: "deepmd/npy").
            output_mode (str): "atomic" (per atom) or "structural" (per frame, mean pooled).
            
        Returns:
            np.ndarray: The computed descriptors.
        """
        sys_name = os.path.basename(data_path)
        
        # Check output mode
        if output_mode != "atomic":
            self.logger.warning(f"Output mode is '{output_mode}'. Note that only 'atomic' mode is consistent with 'dp eval-desc' CLI output.")
            
        # self.logger.info(f"Loading data from {data_path} with format {data_format}")
        
        if data_format == "deepmd/npy/mixed":
            onedata = dpdata.MultiSystems.from_file(data_path, fmt=data_format)
        else:
            onedata = dpdata.System(data_path, fmt=data_format)
            
        n_frames = len(onedata)
        self.logger.info(f"# -----------------------------------")
        self.logger.info(f"# -------output of python mode-------")
        self.logger.info(f"# processing system : {sys_name}")
        self.logger.info(f"# evaluating descriptors for {n_frames} frames")
        self.logger.info(f"# output mode: {output_mode}")
            
        desc_list = []
        if data_format == "deepmd/npy/mixed":
            self.logger.warning(f"mixed format descriptor calculation is wrong in python mode")
            for onesys in onedata:
                nopbc = onesys.data.get('nopbc', False)
                one_desc_list = self._get_desc_by_batch(onesys, nopbc)
                desc_list.extend(one_desc_list)
        elif data_format == "deepmd/npy":
            nopbc = onedata.data.get('nopbc', False)
            desc_list = self._get_desc_by_batch(onedata, nopbc)
        else:
            raise ValueError(f"Unknown data format: {data_format}, supported: 'deepmd/npy', 'deepmd/npy/mixed'")
        
        desc = np.concatenate(desc_list, axis=0)
        
        if output_mode == "structural":
            desc = np.mean(desc, axis=1)
            
        self.logger.info(f"# descriptor shape: {desc.shape}")
        self.logger.info(f"# -----------------------------------")
            
        # Clear memory
        del onedata
        empty_cache()
        
        return desc
