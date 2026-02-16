import os
import glob
import logging
import numpy as np
from typing import List, Optional, Dict, Union

from dpeva.constants import WORKFLOW_FINISHED_TAG
from dpeva.submission import JobManager, JobConfig
from dpeva.utils.command import DPCommandBuilder

logger = logging.getLogger(__name__)

class FeatureIOManager:
    """
    Manages IO operations for Feature Workflow:
    - Path resolution
    - Multi-pool structure detection
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def detect_multi_pool_structure(self, data_path: str) -> List[str]:
        """
        Detect if data_path contains multiple sub-pools.
        A sub-pool is a directory that is NOT a system itself but contains systems.
        Returns a list of sub-pool directory names relative to data_path.
        """
        abs_data_path = os.path.abspath(data_path)
        
        if not os.path.exists(abs_data_path):
            return []
            
        subdirs = [d for d in os.listdir(abs_data_path) 
                   if os.path.isdir(os.path.join(abs_data_path, d))]
        
        sub_pools = []
        for d in subdirs:
            d_path = os.path.join(abs_data_path, d)
            # Check if it is a system (simple check)
            is_system = os.path.exists(os.path.join(d_path, "type.raw")) or \
                        os.path.exists(os.path.join(d_path, "type_map.raw")) or \
                        len(glob.glob(os.path.join(d_path, "set.*"))) > 0
            
            if not is_system:
                sub_pools.append(d)
                
        return sub_pools

    def is_leaf_system(self, path: str) -> bool:
        """Check if a path is a leaf system directory (contains type.raw or set.000)."""
        return os.path.exists(os.path.join(path, "type.raw")) or \
               os.path.exists(os.path.join(path, "set.000")) or \
               len(glob.glob(os.path.join(path, "set.*"))) > 0


class FeatureExecutionManager:
    """
    Manages Execution for Feature Workflow:
    - CLI command generation and submission
    - Python script generation and submission (Slurm)
    - Local Python execution orchestration
    """
    def __init__(self, backend: str, slurm_config: Dict, env_setup: str, dp_backend: str, omp_threads: int):
        self.backend = backend
        self.slurm_config = slurm_config or {}
        self.env_setup = env_setup or ""
        self.dp_backend = dp_backend
        self.omp_threads = omp_threads
        
        DPCommandBuilder.set_backend(self.dp_backend)
        self.job_manager = JobManager(mode=backend)
        self.logger = logging.getLogger(__name__)
        
        # Default env setup
        if not self.env_setup:
            self.env_setup = f"export OMP_NUM_THREADS={self.omp_threads}"

    def submit_cli_job(self, data_path: str, output_dir: str, model_path: str, head: str, 
                      sub_pools: List[str], blocking: bool = True):
        """
        Submit a CLI job (dp eval-desc).
        Handles both Single-Pool and Multi-Pool structures.
        """
        abs_data_path = os.path.abspath(data_path)
        abs_output_dir = os.path.abspath(output_dir)
        os.makedirs(abs_output_dir, exist_ok=True)
        
        log_file = "eval_desc.log" if self.backend == "local" else None
        
        cmd = ""
        if sub_pools:
            self.logger.info(f"Detected multi-pool structure with {len(sub_pools)} pools. Generating iterative script.")
            for pool in sub_pools:
                pool_in = os.path.join(abs_data_path, pool)
                pool_out = os.path.join(abs_output_dir, pool)
                
                cmd += f"mkdir -p {pool_out}\n"
                
                pool_cmd = DPCommandBuilder.eval_desc(
                    model=model_path,
                    system=pool_in,
                    output=pool_out,
                    head=head,
                    log_file=None
                )
                
                cmd += f"echo 'Processing pool: {pool}'\n"
                cmd += f"{pool_cmd}\n"
        else:
            cmd = DPCommandBuilder.eval_desc(
                model=model_path,
                system=abs_data_path,
                output=abs_output_dir,
                head=head,
                log_file=log_file
            )
            
        cmd += f"\necho \"{WORKFLOW_FINISHED_TAG}\""
        
        # Create JobConfig
        job_name = f"dpa_evaldesc_{os.path.basename(abs_data_path)}"
        
        # Filter Slurm config
        task_slurm_config = self.slurm_config.copy()
        for k in ["job_name", "output_log", "error_log"]:
            task_slurm_config.pop(k, None)
            
        job_config = JobConfig(
            job_name=job_name,
            command=cmd,
            env_setup=self.env_setup,
            output_log="eval_desc.log",
            error_log="eval_desc.err",
            **task_slurm_config
        )
        
        script_name = "run_evaldesc.slurm" if self.backend == "slurm" else "run_evaldesc.sh"
        script_path = os.path.join(abs_output_dir, script_name)
        
        self.job_manager.generate_script(job_config, script_path)
        
        self.logger.info(f"Submitting eval-desc job for {data_path}")
        self.job_manager.submit(script_path, working_dir=abs_output_dir)

    def submit_python_slurm_job(self, data_path: str, output_dir: str, model_path: str, head: str, 
                               batch_size: int, output_mode: str):
        """
        Submit a Python script job to Slurm.
        """
        abs_data_path = os.path.abspath(data_path)
        abs_output_dir = os.path.abspath(output_dir)
        os.makedirs(abs_output_dir, exist_ok=True)
        
        # Generate Worker Script Content
        # We need to import FeatureWorkflow to run the local recursion logic inside the worker
        # Or simpler: Instantiate Generator and use a recursion helper function.
        # Ideally, we should reuse the logic in run_local_python_recursion.
        # But for the worker script, we need a standalone entry point.
        # Let's create a dedicated worker function in generator.py or here?
        # To avoid circular imports, we can put the worker script logic to call FeatureWorkflow or similar.
        # Actually, best practice is to make the worker script call a function that does the recursion.
        # Since we are refactoring, let's make the worker script import FeatureExecutionManager and call run_local_python_recursion!
        # But FeatureExecutionManager needs init args.
        
        # Alternative: The worker script instantiates DescriptorGenerator and does the recursion itself.
        # But we moved recursion OUT of DescriptorGenerator.
        # So we need a standalone function for recursion.
        
        worker_script_content = f"""
import os
import sys
import numpy as np

# Ensure dpeva is in path
sys.path.append("{os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))}")

from dpeva.feature.generator import DescriptorGenerator
from dpeva.feature.managers import FeatureIOManager, FeatureExecutionManager

def main():
    # Initialize components
    generator = DescriptorGenerator(
        model_path="{model_path}",
        head="{head}",
        batch_size={batch_size},
        omp_threads={self.omp_threads}
    )
    
    exec_manager = FeatureExecutionManager(
        backend="local", 
        slurm_config={{}}, 
        env_setup="", 
        dp_backend="{self.dp_backend}", 
        omp_threads={self.omp_threads}
    )
    
    print("Starting recursive descriptor generation...")
    exec_manager.run_local_python_recursion(
        generator, 
        "{abs_data_path}", 
        "{abs_output_dir}", 
        output_mode="{output_mode}"
    )
    
    print("{WORKFLOW_FINISHED_TAG}")

if __name__ == "__main__":
    main()
"""
        job_name = f"dpeva_py_desc_{os.path.basename(abs_data_path)}"
        
        task_slurm_config = self.slurm_config.copy()
        for k in ["job_name", "output_log", "error_log"]:
            task_slurm_config.pop(k, None)
            
        job_config = JobConfig(
            job_name=job_name,
            command="", # Set by submit_python_script
            env_setup=self.env_setup,
            output_log="eval_desc_py.log",
            error_log="eval_desc_py.err",
            **task_slurm_config
        )
        
        self.logger.info(f"Submitting python mode job for {data_path}")
        self.job_manager.submit_python_script(
            worker_script_content, 
            "run_desc_worker.py", 
            job_config, 
            working_dir=abs_output_dir
        )

    def run_local_python_recursion(self, generator, data_path: str, output_dir: str, output_mode: str = "atomic"):
        """
        Execute Python descriptor generation recursively in the local process.
        """
        io_manager = FeatureIOManager()
        abs_data_path = os.path.abspath(data_path)
        abs_output_dir = os.path.abspath(output_dir)
        os.makedirs(abs_output_dir, exist_ok=True)
        
        self.logger.info(f"Scanning {abs_data_path} for systems...")
        
        def process_recursive(current_path, current_output_dir):
            # Check if leaf system
            if io_manager.is_leaf_system(current_path):
                sys_name = os.path.basename(current_path)
                try:
                    desc = generator.compute_descriptors(
                        data_path=current_path,
                        output_mode=output_mode
                    )
                    # Logic: If current_path matches data_path (root is system), save as basename.npy
                    # If current_path is subdir, save as subdir.npy in parent output
                    
                    # However, the recursion logic below passes `current_output_dir` as `parent_out/subdir`.
                    # So if we are at `parent_out/subdir`, and it is a system, we want to save `parent_out/subdir.npy`.
                    # But wait, `current_output_dir` is a directory path.
                    # We should append `.npy` to it?
                    
                    # Let's align with the old logic:
                    # Old logic: `out_file = current_output_dir + ".npy"`
                    out_file = current_output_dir + ".npy"
                    
                    # Ensure parent dir of out_file exists
                    os.makedirs(os.path.dirname(out_file), exist_ok=True)
                    
                    np.save(out_file, desc)
                    self.logger.info(f"Saved descriptors to {out_file}")
                    return
                except Exception as e:
                    self.logger.error(f"Failed to process {sys_name}: {e}")
                    return

            # If not leaf, iterate subdirs
            try:
                subdirs = [d for d in os.listdir(current_path) if os.path.isdir(os.path.join(current_path, d))]
            except OSError:
                return

            for d in subdirs:
                process_recursive(os.path.join(current_path, d), os.path.join(current_output_dir, d))

        # Initial call
        if io_manager.is_leaf_system(abs_data_path):
            # Single system
            desc = generator.compute_descriptors(abs_data_path, output_mode)
            out_file = os.path.join(abs_output_dir, os.path.basename(abs_data_path) + ".npy")
            np.save(out_file, desc)
            self.logger.info(f"Saved descriptors to {out_file}")
        else:
            # Recursive scan
            subdirs = [d for d in os.listdir(abs_data_path) if os.path.isdir(os.path.join(abs_data_path, d))]
            for d in subdirs:
                process_recursive(os.path.join(abs_data_path, d), os.path.join(abs_output_dir, d))
