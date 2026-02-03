import os
import logging
from dpeva.training.trainer import ParallelTrainer

class TrainingWorkflow:
    """
    Workflow for parallel fine-tuning of DeepMD models.
    Supports both local multiprocessing and Slurm submission.
    """
    
from typing import Union, Dict
from dpeva.config import TrainingConfig

class TrainingWorkflow:
    """
    Workflow for parallel fine-tuning of DeepMD models.
    Supports both local multiprocessing and Slurm submission.
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
        
        # Seeds configuration
        default_seeds = [19090, 42, 10032, 2933]
        
        if self.config.seeds:
            self.seeds = self.config.seeds
        else:
             if self.num_models > len(default_seeds):
                 self.logger.warning(f"num_models ({self.num_models}) > default seeds length. Cycling default seeds.")
                 self.seeds = (default_seeds * (self.num_models // len(default_seeds) + 1))[:self.num_models]
             else:
                 self.seeds = default_seeds[:self.num_models]

        if self.config.training_seeds:
            self.training_seeds = self.config.training_seeds
        else:
             if self.num_models > len(default_seeds):
                 self.training_seeds = (default_seeds * (self.num_models // len(default_seeds) + 1))[:self.num_models]
             else:
                 self.training_seeds = default_seeds[:self.num_models]
        
        # Base model configuration
        self.base_model_path = str(self.config.base_model_path)
        
        # OMP Settings
        self.omp_threads = self.config.omp_threads
        
        # Submission Configuration
        self.backend = self.config.submission.backend
        self.slurm_config = self.config.submission.slurm_config
        self.template_path = str(self.config.template_path) if self.config.template_path else None
        
        # Finetune head name configuration
        self.finetune_head_name = self.config.model_head
        
        # Override training data path if provided in config
        self.training_data_path = str(self.config.training_data_path) if self.config.training_data_path else None

    def _setup_logger(self):
        self.logger = logging.getLogger(__name__)

    def _determine_finetune_heads(self):
        """Determine finetune heads based on mode."""
        if self.mode == "init":
            # First model uses configured head name, others use RANDOM
            heads = [self.finetune_head_name]
            if self.num_models > 1:
                heads.extend(["RANDOM"] * (self.num_models - 1))
            return heads
        elif self.mode == "cont":
            return [self.finetune_head_name] * self.num_models
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Must be 'init' or 'cont'.")

    def _resolve_base_models(self):
        """Resolve base model paths based on mode."""
        if not self.base_model_path:
             raise ValueError("base_model_path must be provided")
             
        return [self.base_model_path] * self.num_models

    def run(self):
        self.logger.info(f"Initializing Training Workflow in {self.work_dir}")
        self.logger.info(f"Mode: {self.mode}, Backend: {self.backend}")
        
        trainer = ParallelTrainer(
            base_config_path=self.input_json_path,
            work_dir=self.work_dir,
            num_models=self.num_models,
            backend=self.backend,
            template_path=self.template_path,
            slurm_config=self.slurm_config,
            training_data_path=self.training_data_path # Pass the override path
        )
        
        finetune_heads = self._determine_finetune_heads()
        trainer.prepare_configs(self.seeds, self.training_seeds, finetune_heads)
        
        base_models = self._resolve_base_models()
        trainer.setup_workdirs(base_models, omp_threads=self.omp_threads)
        
        self.logger.info("Starting parallel training...")
        trainer.train(blocking=True)
        self.logger.info("Training Workflow Submission Completed.")
