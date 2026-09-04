import os
import glob
import logging
import shlex
import warnings
import numpy as np
from typing import List, Dict

from dpeva.constants import WORKFLOW_FINISHED_TAG
from dpeva.compatibility import DeepMDAdapter
from dpeva.submission import JobManager, JobConfig
from dpeva.submission.guards import guarded_command
from dpeva.utils.exceptions import WorkflowError

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
    def __init__(
        self,
        backend: str,
        slurm_config: Dict,
        env_setup: str,
        dp_backend: str,
        omp_threads: int,
        adapter: DeepMDAdapter | None = None,
    ):
        self.backend = backend
        self.slurm_config = slurm_config or {}
        self.env_setup = env_setup or ""
        self.omp_threads = omp_threads
        
        if adapter is None:
            warnings.warn(
                "FeatureExecutionManager uses the legacy unchecked DeepMD adapter; "
                "provide an authorized DeepMDAdapter",
                DeprecationWarning,
                stacklevel=2,
            )
        self.adapter = adapter or DeepMDAdapter.for_legacy_unchecked(dp_backend)
        self.dp_backend = self.adapter.backend
        self.job_manager = JobManager(mode=backend)
        self.logger = logging.getLogger(__name__)
        
        # Default env setup
        if not self.env_setup:
            self.env_setup = f"export OMP_NUM_THREADS={self.omp_threads}"

    def submit_cli_job(
        self,
        data_path: str,
        output_dir: str,
        model_path: str,
        head: str | None,
        sub_pools: List[str],
        blocking: bool = True,
        feature_exporter: str = "eval_desc",
        feature_kind: str = "descriptor",
        embedding_dtype: str = "fp32",
    ):
        """
        Submit a CLI job (`dp eval-desc` or `dp embed`).
        Handles both Single-Pool and Multi-Pool structures.
        """
        abs_data_path = os.path.abspath(data_path)
        abs_output_dir = os.path.abspath(output_dir)
        os.makedirs(abs_output_dir, exist_ok=True)

        if feature_exporter == "eval_desc":
            if feature_kind != "descriptor":
                raise ValueError("feature_exporter='eval_desc' only supports descriptor features.")
            log_file = "eval_desc.log" if self.backend == "local" else None

            cmd = ""
            if sub_pools:
                self.logger.info(f"Detected multi-pool structure with {len(sub_pools)} pools. Generating iterative script.")
                for pool in sub_pools:
                    pool_in = os.path.join(abs_data_path, pool)
                    pool_out = os.path.join(abs_output_dir, pool)

                    cmd += f"mkdir -p {pool_out}\n"

                    pool_cmd = self.adapter.eval_desc(
                        model=model_path,
                        system=pool_in,
                        output=pool_out,
                        head=head,
                        log_file=None
                    )

                    cmd += f"echo 'Processing pool: {pool}'\n"
                    cmd += f"{pool_cmd}\n"
            else:
                cmd = self.adapter.eval_desc(
                    model=model_path,
                    system=abs_data_path,
                    output=abs_output_dir,
                    head=head,
                    log_file=log_file
                )

            job_name = f"dpa_evaldesc_{os.path.basename(abs_data_path)}"
            output_log = "eval_desc.log"
            error_log = "eval_desc.err"
            script_name = "run_evaldesc.slurm" if self.backend == "slurm" else "run_evaldesc.sh"

        elif feature_exporter == "embed":
            if feature_kind not in {"descriptor", "fitting_last_layer"}:
                raise ValueError(f"Unsupported feature kind for embed: {feature_kind}")
            log_file = "embed.log" if self.backend == "local" else None
            if sub_pools:
                self.logger.info(f"Detected multi-pool structure with {len(sub_pools)} pools. Generating iterative script.")
                cmd = ""
                for pool in sub_pools:
                    pool_in = os.path.join(abs_data_path, pool)
                    pool_out = os.path.join(abs_output_dir, pool)
                    output_hdf5 = os.path.join(pool_out, "embedding.hdf5")

                    cmd += f"mkdir -p {pool_out}\n"
                    pool_cmd = self.adapter.embed(
                        model=model_path,
                        system=pool_in,
                        output=output_hdf5,
                        head=head,
                        dtype=embedding_dtype,
                        log_file=None,
                    )
                    cmd += f"echo 'Processing pool: {pool}'\n"
                    cmd += f"{pool_cmd}\n"
            else:
                output_hdf5 = os.path.join(abs_output_dir, "embedding.hdf5")
                cmd = self.adapter.embed(
                    model=model_path,
                    system=abs_data_path,
                    output=output_hdf5,
                    head=head,
                    dtype=embedding_dtype,
                    log_file=log_file,
                )
            job_name = f"dpa_embed_{os.path.basename(abs_data_path)}"
            output_log = "embed.log"
            error_log = "embed.err"
            script_name = "run_embed.slurm" if self.backend == "slurm" else "run_embed.sh"
        else:
            raise ValueError(f"Unsupported feature exporter: {feature_exporter}")
            
        if feature_exporter == "embed":
            output_paths = [
                os.path.join(abs_output_dir, pool, "embedding.hdf5")
                for pool in sub_pools
            ] if sub_pools else [output_hdf5]
            checks = [f"test -s {shlex.quote(path)}" for path in output_paths]
        elif sub_pools:
            checks = [
                f"find {shlex.quote(os.path.join(abs_output_dir, pool))} "
                "-type f -name '*.npy' -size +0c -print -quit | grep -q ."
                for pool in sub_pools
            ]
        else:
            checks = [
                f"find {shlex.quote(abs_output_dir)} -type f -name '*.npy' "
                "-size +0c -print -quit | grep -q ."
            ]
        cmd = guarded_command(command=cmd, artifact_checks=checks)

        # Filter Slurm config
        task_slurm_config = self.slurm_config.copy()
        for k in ["job_name", "output_log", "error_log"]:
            task_slurm_config.pop(k, None)
            
        job_config = JobConfig(
            job_name=job_name,
            command=cmd,
            env_setup=self.env_setup,
            output_log=output_log,
            error_log=error_log,
            **task_slurm_config
        )
        script_path = os.path.join(abs_output_dir, script_name)
        
        self.job_manager.generate_script(job_config, script_path)
        
        self.logger.info(f"Submitting {feature_exporter} job for {data_path}")
        return self.job_manager.submit(script_path, working_dir=abs_output_dir)

    def submit_python_slurm_job(
        self,
        data_path: str,
        output_dir: str,
        model_path: str,
        head: str | None,
        batch_size: int,
        output_mode: str,
        feature_kind: str = "descriptor",
    ):
        """
        Submit a Python script job to Slurm.
        """
        abs_data_path = os.path.abspath(data_path)
        abs_output_dir = os.path.abspath(output_dir)
        os.makedirs(abs_output_dir, exist_ok=True)
        
        # Generate Worker Script Content
        worker_script_content = f"""
import os
import sys
import numpy as np

# Ensure dpeva is in path
sys.path.append("{os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))}")

from dpeva.feature.generator import DescriptorGenerator
from dpeva.feature.managers import FeatureIOManager, FeatureExecutionManager
from dpeva.utils.exceptions import WorkflowError

def main():
    # Initialize components
    generator = DescriptorGenerator(
        model_path="{model_path}",
        head={head!r},
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
        output_mode="{output_mode}",
        feature_kind="{feature_kind}"
    )

    artifacts = [
        os.path.join(root, filename)
        for root, _, filenames in os.walk("{abs_output_dir}")
        for filename in filenames
        if filename.endswith(".npy")
    ]
    if not any(os.path.isfile(path) and os.path.getsize(path) > 0 for path in artifacts):
        raise WorkflowError("Feature generation produced no non-empty .npy artifacts")
    
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
        return self.job_manager.submit_python_script(
            worker_script_content, 
            "run_desc_worker.py", 
            job_config, 
            working_dir=abs_output_dir
        )

    def run_local_python_recursion(
        self,
        generator,
        data_path: str,
        output_dir: str,
        output_mode: str = "atomic",
        feature_kind: str = "descriptor",
    ):
        """
        Execute Python descriptor generation recursively in the local process.
        """
        io_manager = FeatureIOManager()
        abs_data_path = os.path.abspath(data_path)
        abs_output_dir = os.path.abspath(output_dir)
        os.makedirs(abs_output_dir, exist_ok=True)
        failures: list[str] = []
        
        self.logger.info(f"Scanning {abs_data_path} for systems...")
        
        def process_recursive(current_path, current_output_dir):
            """Recursively processes directories to generate descriptors."""
            # Check if leaf system
            if io_manager.is_leaf_system(current_path):
                try:
                    desc = self._compute_feature(
                        generator,
                        data_path=current_path,
                        output_mode=output_mode,
                        feature_kind=feature_kind,
                    )
                    # Logic: If current_path matches data_path (root is system), save as basename.npy
                    # If current_path is subdir, save as subdir.npy in parent output
                    out_file = current_output_dir + ".npy"
                    
                    # Ensure parent dir of out_file exists
                    os.makedirs(os.path.dirname(out_file), exist_ok=True)
                    
                    np.save(out_file, desc)
                    self.logger.info(f"Saved descriptors to {out_file}")
                    return
                except Exception as e:
                    self.logger.error(f"Failed to process {current_path}: {e}")
                    failures.append(f"{current_path}: {e}")
                    return

            # If not leaf, iterate subdirs
            try:
                subdirs = [d for d in os.listdir(current_path) if os.path.isdir(os.path.join(current_path, d))]
            except OSError as e:
                failures.append(f"{current_path}: {e}")
                return

            for d in sorted(subdirs):
                process_recursive(os.path.join(current_path, d), os.path.join(current_output_dir, d))

        # Initial call
        if io_manager.is_leaf_system(abs_data_path):
            # Single system
            try:
                desc = self._compute_feature(generator, abs_data_path, output_mode, feature_kind)
                out_file = os.path.join(abs_output_dir, os.path.basename(abs_data_path) + ".npy")
                np.save(out_file, desc)
                self.logger.info(f"Saved descriptors to {out_file}")
            except Exception as e:
                failures.append(f"{abs_data_path}: {e}")
        else:
            # Recursive scan
            try:
                subdirs = [d for d in os.listdir(abs_data_path) if os.path.isdir(os.path.join(abs_data_path, d))]
            except OSError as e:
                failures.append(f"{abs_data_path}: {e}")
            else:
                for d in sorted(subdirs):
                    process_recursive(os.path.join(abs_data_path, d), os.path.join(abs_output_dir, d))

        if failures:
            raise WorkflowError(
                f"feature generation failed for {len(failures)} system(s): {failures}"
            )

    def _compute_feature(self, generator, data_path: str, output_mode: str, feature_kind: str):
        if feature_kind == "descriptor":
            return generator.compute_descriptors(data_path, output_mode)
        if feature_kind == "fitting_last_layer":
            return generator.compute_fitting_last_layer(data_path, output_mode)
        raise ValueError(f"Unsupported feature kind: {feature_kind}")
