import os
import logging
import json
from typing import Union, Dict

from dpeva.config import TrainingConfig
from dpeva.io.training import TrainingIOManager
from dpeva.training.managers import TrainingConfigManager, TrainingExecutionManager
from dpeva.constants import WORKFLOW_FINISHED_TAG, LOG_FILE_TRAIN
from dpeva.utils.logs import setup_workflow_logger

class TrainingWorkflow:
    """
    Workflow for parallel fine-tuning of DeepMD models.
    Supports both local multiprocessing and Slurm submission.
    Refactored using DDD Managers.
    """
    
    def __init__(self, config: Union[Dict, TrainingConfig]):
        """
        Initialize the Training Workflow.

        Args:
            config (Union[Dict, TrainingConfig]): Configuration object or dictionary.
        """
        if isinstance(config, dict):
            self.config = TrainingConfig(**config)
        else:
            self.config = config
            
        self._setup_logger()
        
        self.work_dir = str(self.config.work_dir)
        self.input_json_path = str(self.config.input_json_path)
        self.num_models = self.config.num_models
        self.mode = self.config.training_mode
        
        # 1. Initialize IO Manager
        self.io_manager = TrainingIOManager(self.work_dir)
        
        # 2. Initialize Config Manager
        if not os.path.exists(self.input_json_path):
             raise FileNotFoundError(f"Config file not found: {self.input_json_path}")
             
        with open(self.input_json_path, 'r') as f:
            base_config = json.load(f)
            
        self.config_manager = TrainingConfigManager(base_config, self.input_json_path)
        
        # 3. Initialize Execution Manager
        self.execution_manager = TrainingExecutionManager(
            backend=self.config.submission.backend,
            slurm_config=self.config.submission.slurm_config,
            env_setup=self.config.submission.env_setup,
            dp_backend=self.config.dp_backend,
            template_path=str(self.config.template_path) if self.config.template_path else None
        )
        
        # Base model configuration
        self.base_model_path = str(self.config.base_model_path)
        
        # OMP Settings
        self.omp_threads = self.config.omp_threads
        
        # Finetune head name configuration
        self.finetune_head_name = self.config.model_head
        
        # Override training data path
        self.training_data_path = str(self.config.training_data_path) if self.config.training_data_path else None


    def _setup_logger(self):
        self.logger = logging.getLogger(__name__)

    def run(self):
        """
        Executes the training workflow.

        1. Configures workflow logging.
        2. Generates task-specific configurations and seeds.
        3. Creates task directories and copies base models.
        4. Generates submission scripts (Slurm or Local).
        5. Submits jobs to the execution backend.
        """
        # Configure logging: log to training.log, but DO NOT capture stdout (propagate=True)
        setup_workflow_logger(
            logger_name="dpeva",
            work_dir=self.work_dir,
            filename=LOG_FILE_TRAIN,
            capture_stdout=False
        )
        
        self.logger.info(f"Initializing Training Workflow in {self.work_dir}")
        self.logger.info(f"Mode: {self.mode}, Backend: {self.config.submission.backend}")
        
        # 1. Prepare Configs
        seeds = self.config_manager.generate_seeds(self.num_models, self.config.seeds)
        training_seeds = self.config_manager.generate_seeds(self.num_models, self.config.training_seeds)
        finetune_heads = self.config_manager.get_finetune_heads(self.mode, self.finetune_head_name, self.num_models)
        
        task_configs = self.config_manager.prepare_task_configs(
            self.num_models, seeds, training_seeds, finetune_heads, self.training_data_path
        )
        
        script_paths = []
        task_dirs = []
        
        # 2. Setup Workspace & Tasks
        for i in range(self.num_models):
            task_dir = self.io_manager.create_task_dir(i)
            task_dirs.append(task_dir)
            
            # Save Config
            self.io_manager.save_task_config(task_dir, task_configs[i])
            
            # Copy Base Model
            if not self.base_model_path:
                 raise ValueError("base_model_path must be provided")
            base_model_name = self.io_manager.copy_base_model(self.base_model_path, task_dir)
            
            # Generate Script
            script_path = self.execution_manager.generate_script(
                task_idx=i,
                task_dir=task_dir,
                base_model_name=base_model_name,
                omp_threads=self.omp_threads
            )
            script_paths.append(script_path)
            
        # 3. Submit Jobs
        self.logger.info("Starting parallel training...")
        self.execution_manager.submit_jobs(script_paths, task_dirs, blocking=True)
        self.logger.info("Training Workflow Submission Completed.")
        self.logger.info(WORKFLOW_FINISHED_TAG)
