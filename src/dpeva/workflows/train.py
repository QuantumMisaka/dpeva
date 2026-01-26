import os
import logging
from dpeva.training.trainer import ParallelTrainer

class TrainingWorkflow:
    """
    Workflow for parallel fine-tuning of DeepMD models.
    Supports both local multiprocessing and Slurm submission.
    """
    
    def __init__(self, config):
        self.config = config
        self._setup_logger()
        
        self.work_dir = config.get("work_dir", os.getcwd())
        self.input_json_path = config.get("input_json_path", "input.json")
        self.num_models = config.get("num_models", 4)
        self.mode = config.get("mode", "cont") # 'init' or 'cont'
        
        # Seeds configuration
        default_seeds = [19090, 42, 10032, 2933]
        
        if "seeds" in config:
            self.seeds = config["seeds"]
        else:
             if self.num_models > len(default_seeds):
                 self.logger.warning(f"num_models ({self.num_models}) > default seeds length. Cycling default seeds.")
                 self.seeds = (default_seeds * (self.num_models // len(default_seeds) + 1))[:self.num_models]
             else:
                 self.seeds = default_seeds[:self.num_models]

        if "training_seeds" in config:
            self.training_seeds = config["training_seeds"]
        else:
             if self.num_models > len(default_seeds):
                 self.training_seeds = (default_seeds * (self.num_models // len(default_seeds) + 1))[:self.num_models]
             else:
                 self.training_seeds = default_seeds[:self.num_models]
        
        # Base model configuration
        self.base_model_path = config.get("base_model_path")
        if config.get("base_models_paths"):
             self.logger.warning("base_models_paths is deprecated. Use base_model_path (str) instead.")
             if not self.base_model_path:
                 self.base_model_path = config.get("base_models_paths")[0]
        
        # OMP Settings
        self.omp_threads = config.get("omp_threads", 12)
        
        # Submission Configuration
        self.backend = config.get("backend", "local") # 'local' or 'slurm'
        self.slurm_config = config.get("slurm_config", {})
        self.template_path = config.get("template_path")
        
        # Finetune head name configuration
        self.finetune_head_name = config.get("finetune_head_name", "Hybrid_Perovskite")
        
        # Override training data path if provided in config
        self.training_data_path = config.get("training_data_path")

    def _setup_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
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
             
        if self.mode == "init":
            return [self.base_model_path] * self.num_models
        else:
            # For 'cont' mode, currently using the same base model path.
            # Future extensions may require template string support for distinct models per task.
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
