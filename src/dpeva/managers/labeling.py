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
            sys_name = getattr(system, "short_name", f"sys_{i}")
            
            for f_idx, atoms in enumerate(atoms_list):
                task_name = f"{sys_name}_{f_idx}"
                task_dir = self.input_dir / task_name
                
                try:
                    self.generator.generate(atoms, task_dir, task_name)
                    generated_count += 1
                except Exception as e:
                    logger.error(f"Failed to generate input for {task_name}: {e}")
        
        logger.info(f"Generated {generated_count} tasks.")
        
        # Pack tasks
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
        print(f"Running task: {task_dir.name}")
        os.chdir(task_dir)
        
        # Command to run ABACUS
        # We assume 'abacus' is in PATH or loaded via modules
        # Check environment variable for ABACUS command
        abacus_cmd = os.environ.get("ABACUS_COMMAND", "abacus")
        
        # Check SLURM_NTASKS for MPI
        slurm_ntasks = os.environ.get("SLURM_NTASKS", "1")
        
        # Construct command
        # If slurm_ntasks > 1, use mpirun
        if int(slurm_ntasks) > 1:
            cmd = f"mpirun -np {slurm_ntasks} {abacus_cmd}"
        else:
            cmd = abacus_cmd
            
        try:
            with open("abacus.out", "w") as outfile:
                subprocess.run(cmd, shell=True, check=True, stdout=outfile, stderr=subprocess.STDOUT)
            print(f"Task {task_dir.name} completed.")
        except subprocess.CalledProcessError as e:
            print(f"Task {task_dir.name} failed: {e}")
        except Exception as e:
            print(f"Task {task_dir.name} error: {e}")
        finally:
            os.chdir(root_dir)

if __name__ == "__main__":
    run_abacus_tasks()
"""
        return script_content

    def process_results(self, packed_job_dirs: List[Path]) -> Tuple[List[Path], List[Path]]:
        """
        Process results from packed job directories.
        Iterates over tasks inside packed jobs.
        """
        logger.info("Processing results...")
        self.converged_dir.mkdir(parents=True, exist_ok=True)
        
        converged_tasks = []
        failed_tasks = []
        
        for job_dir in packed_job_dirs:
            if not job_dir.exists():
                logger.warning(f"Job directory {job_dir} does not exist.")
                continue
            
            # Iterate tasks inside job dir
            for task_dir in job_dir.iterdir():
                if not task_dir.is_dir():
                    continue
                
                # Check convergence
                if self.postprocessor.check_convergence(task_dir):
                    converged_tasks.append(task_dir)
                    # Move to CONVERGED
                    try:
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
        """
        logger.info("Collecting converged results...")
        systems = dpdata.MultiSystems()
        
        if not self.converged_dir.exists():
             logger.warning("CONVERGED directory not found.")
             return

        # Scan CONVERGED directory
        for task_dir in self.converged_dir.iterdir():
            if task_dir.is_dir():
                ls = self.postprocessor.load_data(task_dir)
                if ls:
                    systems.append(ls)
        
        if len(systems) == 0:
            logger.warning("No converged systems found.")
            return

        # Compute metrics
        df = self.postprocessor.compute_metrics(systems)
        
        # Filter
        df_clean = self.postprocessor.filter_data(df)
        
        # Export
        output_format = self.config.get("output_format", "deepmd/npy")
        self.postprocessor.export_data(systems, df_clean, self.output_dir, format=output_format)
        
        # Export anomalies if filtering was performed
        if len(df) > len(df_clean):
            df_anomalies = df.drop(df_clean.index)
            anomaly_dir = self.output_dir / "anomalies"
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
                    atoms.write(extxyz_dir / f"anomaly_{si}_{fi}.extxyz")
                except Exception as e:
                    logger.warning(f"Failed to export anomaly extxyz: {e}")
