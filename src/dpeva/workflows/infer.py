import os
import sys
import logging
from typing import Union, Dict, Optional

from dpeva.config import InferenceConfig
from dpeva.inference.managers import InferenceIOManager, InferenceExecutionManager, InferenceAnalysisManager
from dpeva.utils.command import DPCommandBuilder
from dpeva.constants import WORKFLOW_FINISHED_TAG
from dpeva.submission import JobManager, JobConfig

class InferenceWorkflow:
    """
    Workflow for running inference using an ensemble of DPA models.
    Supports both Local and Slurm backends via JobManager.
    Refactored using DDD Managers.
    """
    
    def __init__(self, config: Union[Dict, InferenceConfig], config_path: Optional[str] = None):
        """
        Initialize the Inference Workflow.

        Args:
            config (Union[Dict, InferenceConfig]): Configuration object or dictionary.
            config_path (Optional[str]): Path to the configuration file (required for Slurm self-submission).
        """
        if isinstance(config, dict):
            self.config = InferenceConfig(**config)
        else:
            self.config = config

        self.config_path = config_path
        self._setup_logger()
        
        # Core Configurations
        self.data_path = str(self.config.data_path)
        self.work_dir = str(self.config.work_dir)
        self.task_name = self.config.task_name
        self.head = self.config.model_head
        self.results_prefix = self.config.results_prefix
        
        # 1. Initialize IO Manager
        self.io_manager = InferenceIOManager(self.work_dir)
        
        # Auto-infer models_paths from work_dir structure
        self.models_paths = self.io_manager.discover_models()
        
        # 2. Initialize Execution Manager
        self.execution_manager = InferenceExecutionManager(
            backend=self.config.submission.backend,
            slurm_config=self.config.submission.slurm_config,
            env_setup=self.config.submission.env_setup,
            dp_backend=self.config.dp_backend,
            omp_threads=self.config.omp_threads
        )
        
        # 3. Initialize Analysis Manager
        # Try to load ref_energies from config (Optional)
        ref_energies = getattr(self.config, "ref_energies", None)
        self.analysis_manager = InferenceAnalysisManager(ref_energies=ref_energies)
        
    def _setup_logger(self):
        self.logger = logging.getLogger(__name__)

    def run(self):
        # Check backend override
        env_backend = os.environ.get("DPEVA_INTERNAL_BACKEND")
        if env_backend:
            self.logger.info(f"Overriding backend to '{env_backend}'")
            self.execution_manager.backend = env_backend
            
        self.logger.info(f"Initializing Inference Workflow (Backend: {self.execution_manager.backend})")
        
        # Self-submission to Slurm
        if self.execution_manager.backend == "slurm" and not env_backend:
            self._submit_to_slurm()
            return
            
        if not self.data_path or not os.path.exists(self.data_path):
            self.logger.error(f"Test data path not found: {self.data_path}")
            return

        if not self.models_paths:
            self.logger.error("No models provided for inference.")
            return

        # Submit Jobs
        self.execution_manager.submit_jobs(
            models_paths=self.models_paths,
            data_path=self.data_path,
            work_dir=self.work_dir,
            task_name=self.task_name,
            head=self.head,
            results_prefix=self.results_prefix
        )
        
        # Optional: Auto-analyze if local and blocking
        if self.execution_manager.backend == "local":
            self.logger.info("Local execution completed. Starting analysis...")
            self.analyze_results()
        else:
            self.logger.info("Slurm jobs submitted. Analysis must be run manually or wait for jobs to finish.")

    def _submit_to_slurm(self):
        """
        Submit the inference workflow to Slurm using JobManager.
        """
        if not self.config_path: raise ValueError("Config path required for Slurm.")
        
        work_dir_abs = os.path.abspath(self.work_dir)
        if not os.path.exists(work_dir_abs): os.makedirs(work_dir_abs)
        
        cmd = f"{sys.executable} -m dpeva.cli infer {os.path.abspath(self.config_path)}"
        
        # Environment Setup
        user_env_setup = self.config.submission.env_setup
        if isinstance(user_env_setup, list):
            user_env_setup = "\n".join(user_env_setup)
        elif user_env_setup is None:
            user_env_setup = ""
            
        internal_backend_setup = "export DPEVA_INTERNAL_BACKEND=local"
        final_env_setup = f"{user_env_setup}\n{internal_backend_setup}"
        
        slurm_config = self.config.submission.slurm_config or {}
        
        job_conf = JobConfig(
            command=cmd,
            job_name=slurm_config.get("job_name", "dpeva_infer"),
            partition=slurm_config.get("partition", "CPU-MISC"),
            qos=slurm_config.get("qos"),
            ntasks=slurm_config.get("ntasks", 1),
            output_log=os.path.join(work_dir_abs, "infer_slurm.out"),
            error_log=os.path.join(work_dir_abs, "infer_slurm.err"),
            env_setup=final_env_setup
        )
        
        manager = JobManager(mode="slurm")
        script_path = os.path.join(work_dir_abs, "submit_infer.slurm")
        manager.generate_script(job_conf, script_path)
        manager.submit(script_path, working_dir=work_dir_abs)
        self.logger.info(f"InferenceWorkflow submitted successfully to Slurm. Job script: {script_path}")


    def analyze_results(self, output_dir_suffix="analysis"):
        """
        Parse results, compute metrics, and generate plots for all models.
        """
        self.logger.info("Starting result analysis...")
        
        # Pre-load atom counts if possible for cohesive energy calculation
        atom_counts_list, atom_num_list = self.io_manager.load_composition_info(self.data_path)
        
        summary_metrics = []
        
        for i, model_path in enumerate(self.models_paths):
            # Resolve work_dir same as run()
            if self.task_name:
                job_work_dir = os.path.join(self.work_dir, str(i), self.task_name)
            else:
                job_work_dir = os.path.join(self.work_dir, str(i))
                
            if not os.path.exists(job_work_dir):
                self.logger.warning(f"Work dir not found: {job_work_dir}, skipping analysis for model {i}")
                continue
            
            try:
                # 1. Parse Results
                data = self.io_manager.parse_results(job_work_dir, self.results_prefix)
                
                # 2. Analyze
                stats_export, metrics = self.analysis_manager.analyze_model(
                    model_idx=i,
                    job_work_dir=job_work_dir,
                    data=data,
                    atom_counts_list=atom_counts_list,
                    atom_num_list=atom_num_list,
                    output_dir_suffix=output_dir_suffix
                )
                
                if metrics:
                    summary_metrics.append(metrics)
                    
            except Exception as e:
                self.logger.error(f"Analysis failed for model {i}: {e}", exc_info=True)
                
        # Save Global Summary
        self.io_manager.save_summary(summary_metrics)
        self.logger.info(WORKFLOW_FINISHED_TAG)
