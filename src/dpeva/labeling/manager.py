"""
Labeling Manager
================

Orchestrates the labeling process: input generation, task packing, and result collection.
"""

import os
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple

from dpeva.config import LabelingConfig
from dpeva.labeling.generator import AbacusGenerator
from dpeva.labeling.packer import TaskPacker
from dpeva.labeling.postprocess import AbacusPostProcessor
from dpeva.labeling.strategy import ResubmissionStrategy

import dpdata

logger = logging.getLogger(__name__)

class LabelingManager:
    """
    Manager for the labeling workflow.
    """

    def __init__(self, config: Union[LabelingConfig, Dict[str, Any]]):
        if isinstance(config, LabelingConfig):
            self.config_obj = config
            self.config = config.model_dump()
        else:
            self.config = config
            self.config_obj = None

        self.generator = AbacusGenerator(self.config)
        self.packer = TaskPacker(self.config.get("tasks_per_job", 50))
        self.postprocessor = AbacusPostProcessor(self.config)
        self.strategy = ResubmissionStrategy(self.config.get("attempt_params"))
        
        self.work_dir = Path(self.config.get("work_dir", "."))
        self.input_dir = self.work_dir / "inputs"
        self.output_dir = self.work_dir / "outputs"
        self.converged_dir = self.work_dir / "CONVERGED"

    def prepare_tasks(self, input_data: dpdata.MultiSystems, task_prefix: str = "task") -> List[Path]:
        """
        Generate input files and pack tasks.
        
        Returns:
            List[Path]: List of packed job directories (e.g., N_50_0).
        """
        logger.info(f"Generating inputs for {len(input_data)} systems...")
        self.input_dir.mkdir(parents=True, exist_ok=True)
        
        generated_count = 0
        for i, system in enumerate(input_data):
            n_frames = system.get_nframes()
            atoms_list = system.to_ase_structure()
            # Use short_name (dataset name) if available, else generic name
            sys_name = getattr(system, "short_name", f"sys_{i}")
            
            for f_idx, atoms in enumerate(atoms_list):
                task_name = f"{sys_name}_{f_idx}"
                
                # Determine stru_type first to construct path: inputs/dataset/stru_type/task_name
                
                try:
                    # Use new analyze method from analyzer
                    pre_atoms, stru_type, vacuum_status = self.generator.analyzer.analyze(atoms)
                    
                    # Construct hierarchical path
                    task_dir = self.input_dir / sys_name / stru_type / task_name
                    
                    # Generate files using pre-analyzed data
                    self.generator.generate(pre_atoms, task_dir, task_name, stru_type, vacuum_status)
                    generated_count += 1
                except Exception as e:
                    logger.error(f"Failed to generate input for {task_name}: {e}")
        
        logger.info(f"Generated {generated_count} tasks.")
        
        # Pack tasks
        # Note: packer needs to scan recursively now because of deeper structure
        packed_job_dirs = self.packer.pack(self.input_dir)
        return packed_job_dirs

    def generate_runner_script(self, job_dir: Path) -> str:
        """
        Generate the content of a Python runner script for a packed job directory.
        This script will iterate over all subdirectories and run ABACUS.
        """
        script_content = f"""
import os
import subprocess
import sys
from pathlib import Path

def run_abacus_tasks():
    root_dir = Path.cwd()
    # Assume we are inside a packed directory (e.g. N_50_X)
    # Iterate over all subdirectories (tasks)
    task_dirs = [d for d in root_dir.iterdir() if d.is_dir()]
    task_dirs.sort()
    
    print(f"Found {{len(task_dirs)}} tasks in {{root_dir}}")
    
    for task_dir in task_dirs:
        # Resolve task_dir to absolute path if it's relative
        # Though task_dirs from iterdir() are usually absolute or relative to cwd.
        # Let's ensure it's a Path object.
        task_dir_path = Path(task_dir)
        print(f"Running task: {{task_dir_path.name}}")
        os.chdir(task_dir_path)
        
        # Command to run ABACUS
        # We assume 'abacus' is in PATH or loaded via modules
        # Check environment variable for ABACUS command
        abacus_cmd = os.environ.get("ABACUS_COMMAND", "abacus")
        
        # Check SLURM_NTASKS for MPI
        slurm_ntasks = os.environ.get("SLURM_NTASKS", "1")
        
        # Construct command
        # If slurm_ntasks > 1, use mpirun
        if int(slurm_ntasks) > 1:
            cmd = f"mpirun -np {{slurm_ntasks}} {{abacus_cmd}}"
        else:
            cmd = abacus_cmd
            
        try:
            with open("abacus.out", "w") as outfile:
                subprocess.run(cmd, shell=True, check=True, stdout=outfile, stderr=subprocess.STDOUT)
            print(f"Task {{task_dir_path.name}} completed.")
        except subprocess.CalledProcessError as e:
            print(f"Task {{task_dir_path.name}} failed: {{e}}")
        except Exception as e:
            print(f"Task {{task_dir_path.name}} error: {{e}}")
        finally:
            os.chdir(root_dir)

if __name__ == "__main__":
    run_abacus_tasks()
"""
        return script_content

    def process_results(self, packed_job_dirs: List[Path]) -> Tuple[List[Path], List[Path]]:
        """
        Process results from packed job directories.
        Iterates over tasks inside packed jobs (recursive).
        """
        logger.info("Processing results...")
        self.converged_dir.mkdir(parents=True, exist_ok=True)
        
        converged_tasks = []
        failed_tasks = []
        
        for job_dir in packed_job_dirs:
            if not job_dir.exists():
                logger.warning(f"Job directory {job_dir} does not exist.")
                continue
            
            # Recursive scan for tasks (contain INPUT)
            # Use same logic as packer to identify tasks
            for input_file in job_dir.rglob("INPUT"):
                task_dir = input_file.parent
                if not task_dir.is_dir():
                    continue
                
                # Check convergence
                if self.postprocessor.check_convergence(task_dir):
                    converged_tasks.append(task_dir)
                    
                    # Move to CONVERGED
                    # Preserve hierarchy relative to inputs/
                    try:
                        # Extract sys_name from task_name (sys_name_fidx)
                        # Warning: sys_name might contain underscores.
                        # We know f_idx is int at end.
                        parts = task_dir.name.rsplit('_', 1)
                        if len(parts) == 2 and parts[1].isdigit():
                            sys_name = parts[0]
                            # Create subdir in CONVERGED
                            target_parent = self.converged_dir / sys_name
                            target_parent.mkdir(parents=True, exist_ok=True)
                            shutil.move(str(task_dir), str(target_parent))
                        else:
                            # Fallback
                            shutil.move(str(task_dir), str(self.converged_dir))
                    except Exception as e:
                        logger.error(f"Failed to move {task_dir} to CONVERGED: {e}")
                else:
                    failed_tasks.append(task_dir)
        
        logger.info(f"Converged: {len(converged_tasks)}, Failed: {len(failed_tasks)}")
        return converged_tasks, failed_tasks

    def apply_attempt_params(self, failed_tasks: List[Path], attempt: int):
        """
        Apply parameters for the next attempt to failed tasks.
        """
        logger.info(f"Applying parameters for attempt {attempt} to {len(failed_tasks)} tasks...")
        fixed_count = 0
        for task_dir in failed_tasks:
            if self.strategy.apply(task_dir, attempt):
                fixed_count += 1
        
        logger.info(f"Parameters applied to {fixed_count} tasks.")
        return fixed_count

    def repack_failed_tasks(self, failed_tasks: List[Path]) -> List[Path]:
        """
        Repack failed tasks into new N_50_X directories for resubmission.
        Since failed tasks are still in their original N_50_X directories (unless moved),
        we might need to consolidate them if they are sparse.
        
        However, simplicity first: If we keep them in place, we just re-submit the original N_50_X jobs?
        No, because converged tasks are moved out.
        So N_50_X only contains failed tasks now.
        We can just return the list of N_50_X directories that still contain tasks.
        """
        job_dirs = set()
        for task_dir in failed_tasks:
            job_dirs.add(task_dir.parent)
        
        return list(job_dirs)

    def collect_and_export(self):
        """
        Collect converged results and export cleaned dataset.
        Also exports anomalies to separate directories.
        """
        logger.info("Collecting converged results...")
        systems = dpdata.MultiSystems()
        
        if not self.converged_dir.exists():
             logger.warning("CONVERGED directory not found.")
             return

        # Scan CONVERGED directory (recursive)
        # CONVERGED structure might be hierarchical now
        # We need to find directories that contain OUT.ABACUS or are valid dpdata systems
        # Since we use postprocessor.load_data, it checks for abacus.out or similar.
        # But we need to traverse.
        
        task_dirs = []
        # Walk through CONVERGED
        for root, dirs, files in os.walk(self.converged_dir):
            # If this dir looks like a task dir (has abacus.out or OUT.ABACUS)
            # load_data checks for these.
            # But we don't want to try loading parent dirs.
            # Heuristic: if it has 'STRU' and 'INPUT', it's likely a task dir.
            if "INPUT" in files and "STRU" in files:
                task_dirs.append(Path(root))
        
        # Sort for consistency
        task_dirs.sort(key=lambda p: str(p))
        
        for task_dir in task_dirs:
            ls = self.postprocessor.load_data(task_dir)
            if ls:
                systems.append(ls)
        
        total_input = len(task_dirs) # Approximation of converged count based on dirs found
        logger.info(f"Loaded {len(systems)} converged systems from {total_input} directories.")
        
        if len(systems) == 0:
            logger.warning("No converged systems found.")
            return

        # Compute metrics
        df = self.postprocessor.compute_metrics(systems)
        
        # Filter
        df_clean = self.postprocessor.filter_data(df)
        
        # Stats logging (Requirement 3)
        # We need to break down by dataset (pool).
        # Let's count
        pool_stats = {}
        
        # Since systems is a new MultiSystems, we need to re-associate with task_dirs to get pool names.
        # But systems order matches append order.
        # Let's rebuild loop to track pool names.
        
        valid_systems = []
        sys_pool_map = [] # Index -> Pool Name
        
        for task_dir in task_dirs:
            ls = self.postprocessor.load_data(task_dir)
            if ls:
                valid_systems.append(ls)
                # Infer pool name from path relative to CONVERGED
                # CONVERGED/pool_name/...
                try:
                    rel_path = task_dir.relative_to(self.converged_dir)
                    pool_name = rel_path.parts[0] if len(rel_path.parts) > 0 else "unknown"
                except:
                    pool_name = "unknown"
                sys_pool_map.append(pool_name)
        
        systems = dpdata.MultiSystems(valid_systems) # Re-wrap
        
        # Re-compute df on valid systems
        df = self.postprocessor.compute_metrics(systems)
        df_clean = self.postprocessor.filter_data(df)
        
        # Aggregate stats
        # Total
        total_converged = len(df)
        total_clean = len(df_clean)
        total_filtered = total_converged - total_clean
        
        logger.info("=== Labeling Statistics ===")
        logger.info(f"Total Converged Structures: {total_converged}")
        logger.info(f"Total Cleaned Structures:   {total_clean}")
        logger.info(f"Total Filtered (Outliers):  {total_filtered}")
        
        # Per Pool Stats
        pools = set(sys_pool_map)
        for pool in sorted(pools):
            # Indices belonging to this pool
            pool_indices = [i for i, p in enumerate(sys_pool_map) if p == pool]
            
            # Filter df for these indices
            # df has 'sys_idx' column
            pool_df = df[df['sys_idx'].isin(pool_indices)]
            pool_clean_df = df_clean[df_clean['sys_idx'].isin(pool_indices)]
            
            p_conv = len(pool_df)
            p_clean = len(pool_clean_df)
            p_filt = p_conv - p_clean
            
            logger.info(f"  Pool '{pool}': Converged={p_conv}, Clean={p_clean}, Filtered={p_filt}")
            
        # Export
        output_format = self.config.get("output_format", "deepmd/npy")
        
        # Requirement 2: Cleaned data in separate dir
        # Proposed: outputs/cleaned/dataset and outputs/anomalies/dataset
        
        cleaned_dir = self.output_dir / "cleaned"
        anomaly_dir = self.output_dir / "anomalies"
        
        self.postprocessor.export_data(systems, df_clean, cleaned_dir, format=output_format)
        
        # Export anomalies if filtering was performed
        if len(df) > len(df_clean):
            df_anomalies = df.drop(df_clean.index)
            self.postprocessor.export_data(systems, df_anomalies, anomaly_dir, format=output_format)
            
            # Export extxyz for quick view (sampled from anomalies)
            extxyz_dir = anomaly_dir / "extxyz"
            extxyz_dir.mkdir(exist_ok=True, parents=True)
            
            # Export top 50 anomalies by energy/force
            top_anomalies = df_anomalies.sort_values("max_force", ascending=False).head(50)
            for idx, row in top_anomalies.iterrows():
                try:
                    si, fi = int(row["sys_idx"]), int(row["frame_idx"])
                    atoms = systems[si][fi].to_ase_structure()[0]
                    # Filename with pool info
                    pool_name = sys_pool_map[si]
                    atoms.write(extxyz_dir / f"anomaly_{pool_name}_{si}_{fi}.extxyz")
                except Exception as e:
                    logger.warning(f"Failed to export anomaly extxyz: {e}")
