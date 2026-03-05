"""
Labeling Workflow
=================

Workflow for First Principles (FP) labeling tasks.
"""

import logging
import time
import subprocess
import re
from pathlib import Path
from typing import List

import dpdata

from dpeva.config import LabelingConfig
from dpeva.managers.labeling import LabelingManager
from dpeva.submission.manager import JobManager
from dpeva.submission.templates import JobConfig
from dpeva.constants import WORKFLOW_FINISHED_TAG
from dpeva.io.dataset import load_systems

logger = logging.getLogger(__name__)

class LabelingWorkflow:
    """
    Workflow for FP labeling.
    Steps:
    1. Load Data
    2. Generate Inputs & Pack
    3. Loop (Submit -> Wait -> Check -> Resubmit)
    4. Export
    """

    def __init__(self, config: LabelingConfig):
        self.config = config
        self.manager = LabelingManager(config)
        self.job_manager = JobManager(
            mode=config.submission.backend,
            # Pass custom template path if provided in slurm_config
            # SubmissionConfig.slurm_config is a dict
            custom_template_path=config.submission.slurm_config.get("template_path")
        )
        
        # Setup workflow logger if needed (though global logger might be configured)
        # self._setup_logging()

    def run(self):
        logger.info("Starting Labeling Workflow...")
        
        # 1. Load Data
        logger.info(f"Loading input data from {self.config.input_data_path}")
        input_data = dpdata.MultiSystems()
        try:
            # Handle nested structure: input_data_path / dataset_name / system
            # Iterate subdirectories
            root_path = Path(self.config.input_data_path)
            if not root_path.exists():
                raise FileNotFoundError(f"Input path not found: {root_path}")
                
            dataset_dirs = [d for d in root_path.iterdir() if d.is_dir()]
            dataset_dirs.sort()
            
            for d_dir in dataset_dirs:
                # Try loading as MultiSystems (deepmd/npy/mixed or deepmd/npy)
                # If d_dir is a system itself, load_systems handles it.
                # If d_dir contains systems, load_systems handles it.
                # But we want to preserve dataset info if possible.
                try:
                    loaded = load_systems(str(d_dir), fmt="auto")
                    for s in loaded:
                        input_data.append(s)
                except Exception as e:
                    logger.warning(f"Failed to load from {d_dir}: {e}")
            
            if len(input_data) == 0:
                raise ValueError(f"No valid systems found in {self.config.input_data_path}")
                
            logger.info(f"Loaded {len(input_data)} systems total.")
            
        except Exception as e:
            logger.critical(f"Failed to load input data: {e}")
            raise

        # 2. Prepare Tasks
        packed_job_dirs = self.manager.prepare_tasks(input_data)
        if not packed_job_dirs:
            logger.warning("No tasks generated.")
            return

        # 3. Execution Loop
        max_attempts = len(self.config.attempt_params)
        
        # Handle case where attempt_params is empty (should not happen with default, but possible)
        if max_attempts == 0:
            logger.warning("No attempt_params defined. Executing single run with default parameters.")
            max_attempts = 1
            
        failed_tasks = []
        
        for attempt in range(max_attempts):
            logger.info(f"=== Execution Attempt {attempt}/{max_attempts - 1} ===")
            
            if attempt > 0:
                if not failed_tasks:
                    logger.info("No failed tasks to retry. Workflow finished.")
                    break
                
                # Ensure attempt index is within bounds of attempt_params if we extended max_attempts
                if attempt < len(self.config.attempt_params):
                    logger.info(f"Preparing retry for {len(failed_tasks)} tasks...")
                    self.manager.apply_attempt_params(failed_tasks, attempt)
                else:
                    logger.warning(f"No parameters for attempt {attempt}. Skipping parameter application.")
            
            # Identify active jobs (directories containing tasks)
            # We iterate over all packed_job_dirs. If they are empty (all moved to CONVERGED), skip.
            active_job_dirs = []
            for job_dir in packed_job_dirs:
                # Check if has subdirectories
                if any(d.is_dir() for d in job_dir.iterdir()):
                    active_job_dirs.append(job_dir)
            
            if not active_job_dirs:
                logger.info("All tasks converged.")
                break
            
            logger.info(f"Submitting {len(active_job_dirs)} job bundles...")
            job_ids = []
            
            for job_dir in active_job_dirs:
                # Generate runner script content
                runner_content = self.manager.generate_runner_script(job_dir)
                runner_name = "run_batch.py"
                
                # Create JobConfig
                # We need to populate it from self.config.submission.slurm_config
                # and override command.
                # JobManager.submit_python_script handles command override.
                
                # Flatten slurm_config to match JobConfig fields
                slurm_conf = self.config.submission.slurm_config
                job_config = JobConfig(
                    command="", # Will be set by submit_python_script
                    job_name=f"fp_{job_dir.name}_att{attempt}",
                    partition=slurm_conf.get("partition"),
                    nodes=slurm_conf.get("nodes", 1),
                    ntasks=slurm_conf.get("ntasks", 1),
                    gpus_per_node=slurm_conf.get("gpus_per_node"),
                    cpus_per_task=slurm_conf.get("cpus_per_task"),
                    walltime=slurm_conf.get("walltime", "24:00:00"),
                    env_setup=self.config.submission.env_setup
                )
                
                try:
                    job_id = self.job_manager.submit_python_script(
                        runner_content, 
                        runner_name, 
                        job_config, 
                        working_dir=str(job_dir)
                    )
                    job_ids.append(job_id)
                except Exception as e:
                    logger.error(f"Failed to submit job for {job_dir}: {e}")
            
            # Wait for jobs
            if self.config.submission.backend == "slurm":
                self._monitor_slurm_jobs(job_ids)
            else:
                # Local backend blocks, so we are done here
                logger.info("Local execution completed.")
            
            # Process Results
            logger.info("Checking convergence...")
            converged, failed_tasks = self.manager.process_results(active_job_dirs)
            
            if not failed_tasks:
                logger.info("All active tasks converged.")
                break
        
        # 4. Final Export
        self.manager.collect_and_export()
        logger.info(WORKFLOW_FINISHED_TAG)

    def _monitor_slurm_jobs(self, job_ids: List[str], interval: int = 60):
        """Monitor Slurm jobs and wait until they finish."""
        if not job_ids:
            return

        logger.info(f"Monitoring {len(job_ids)} Slurm jobs...")
        
        clean_ids = []
        for jid in job_ids:
            match = re.search(r'\d+', jid)
            if match:
                clean_ids.append(match.group())
        
        if not clean_ids:
            logger.warning("No valid job IDs found to monitor.")
            return

        while True:
            # Check queue via squeue
            cmd = ["squeue", "--job", ",".join(clean_ids), "--noheader", "--format=%i"]
            try:
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
                active_ids = result.stdout.strip().split()
                
                if not active_ids:
                    logger.info("All jobs finished.")
                    break
                
                logger.info(f"{len(active_ids)} jobs still running...")
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Failed to query Slurm queue: {e}")
                time.sleep(interval)
