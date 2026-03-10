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
from dpeva.labeling.manager import LabelingManager
from dpeva.labeling.integration import DataIntegrationManager
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
        dataset_map = {}
        try:
            root_path = Path(self.config.input_data_path)
            if not root_path.exists():
                raise FileNotFoundError(f"Input path not found: {root_path}")
            
            # Detect mode: Single-Pool vs Multi-Pool
            # Logic:
            # 1. If root is a system (has type.raw etc) -> Single Pool (Dataset = root.name)
            # 2. If subdirs are systems -> Single Pool (Dataset = root.name)
            # 3. Else -> Multi Pool (Dataset = subdir.name)
            
            is_root_system = (root_path / "type.raw").exists() or (root_path / "type_map.raw").exists() or (root_path / "set.000").exists()
            
            if is_root_system:
                # Case 1: Root is a single system
                logger.info(f"Detected Single-System mode. Dataset: {root_path.name}")
                ms = dpdata.MultiSystems()
                loaded = load_systems(str(root_path), fmt="auto")
                for s in loaded: ms.append(s)
                dataset_map[root_path.name] = ms
            else:
                subdirs = sorted([d for d in root_path.iterdir() if d.is_dir()])
                is_subdir_system = False
                for d in subdirs:
                    if (d / "type.raw").exists() or (d / "type_map.raw").exists() or (d / "set.000").exists():
                        is_subdir_system = True
                        break
                
                if is_subdir_system:
                    # Case 2: Root is a dataset containing systems
                    logger.info(f"Detected Single-Pool mode. Dataset: {root_path.name}")
                    ms = dpdata.MultiSystems()
                    loaded = load_systems(str(root_path), fmt="auto")
                    for s in loaded: ms.append(s)
                    dataset_map[root_path.name] = ms
                else:
                    # Case 3: Root contains datasets
                    logger.info("Detected Multi-Pool mode.")
                    for d in subdirs:
                        # Skip hidden or internal dirs if any
                        if d.name.startswith("."): continue
                        
                        logger.info(f"Loading dataset: {d.name}")
                        loaded = load_systems(str(d), fmt="auto")
                        if loaded:
                            ms = dpdata.MultiSystems()
                            for s in loaded: ms.append(s)
                            dataset_map[d.name] = ms
                        else:
                            logger.warning(f"No systems found in {d.name}, skipping.")

            if not dataset_map:
                raise ValueError(f"No valid systems found in {self.config.input_data_path}")
                
            total_systems = sum(len(ms) for ms in dataset_map.values())
            logger.info(f"Loaded {len(dataset_map)} datasets, {total_systems} systems total.")
            
        except Exception as e:
            logger.critical(f"Failed to load input data: {e}")
            raise

        # 2. Prepare Tasks
        # Note: Manager now accepts dataset_map
        packed_job_dirs = self.manager.prepare_tasks(dataset_map)
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
                    qos=slurm_conf.get("qos"),
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
        if self.config.integration_enabled:
            integration_manager = DataIntegrationManager(deduplicate=self.config.integration_deduplicate)
            cleaned_dir = Path(self.config.work_dir) / "outputs" / "cleaned"
            merged_path = self.config.merged_training_data_path or (Path(self.config.work_dir) / "outputs" / "merged_training_data")
            summary = integration_manager.integrate(
                new_labeled_data_path=cleaned_dir,
                merged_output_path=Path(merged_path),
                existing_training_data_path=self.config.existing_training_data_path,
            )
            logger.info(f"Data integration finished: {summary}")
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

        wait_count = 0
        while True:
            # Check queue via squeue
            cmd = ["squeue", "--job", ",".join(clean_ids), "--noheader", "--format=%i"]
            try:
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
                active_ids = result.stdout.strip().split()
                
                if not active_ids:
                    logger.info("All jobs finished.")
                    break
                
                # Log only every 10th interval (10 mins) to reduce spam
                # But keep polling every minute
                if wait_count % 10 == 0:
                    logger.info(f"{len(active_ids)} jobs still running... (waited {wait_count * interval // 60} mins)")
                
                wait_count += 1
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Failed to query Slurm queue: {e}")
                time.sleep(interval)
