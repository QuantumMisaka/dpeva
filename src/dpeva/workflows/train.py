import os
import logging
from dpeva.training.trainer import ParallelTrainer

class TrainingWorkflow:
    """
    Workflow for parallel fine-tuning of DeepMD models.
    """
    
    def __init__(self, config):
        self.config = config
        self._setup_logger()
        
        self.work_dir = config.get("work_dir", os.getcwd())
        self.input_json_path = config.get("input_json_path", "input.json")
        self.num_models = config.get("num_models", 4)
        self.mode = config.get("mode", "cont") # 'init' or 'cont'
        
        # Seeds configuration
        self.seeds = config.get("seeds", [19090, 42, 10032, 2933])
        self.training_seeds = config.get("training_seeds", [19090, 42, 10032, 2933])
        
        # Base models configuration
        self.base_models_paths = config.get("base_models_paths", [])
        
        # OMP Settings
        self.omp_threads = config.get("omp_threads", 12)

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
            return ["Target_FTS", "RANDOM", "RANDOM", "RANDOM"]
        elif self.mode == "cont":
            return ["Target_FTS"] * self.num_models
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Must be 'init' or 'cont'.")

    def _resolve_base_models(self):
        """Resolve base model paths based on mode."""
        if self.mode == "init":
            # In init mode, we use the same base model for all (usually index 0 from list)
            if not self.base_models_paths:
                raise ValueError("base_models_paths must be provided even for init mode")
            return [self.base_models_paths[0]] * self.num_models
        else:
            # In cont mode, we expect one base model per task
            if len(self.base_models_paths) != self.num_models:
                raise ValueError(f"In 'cont' mode, provide {self.num_models} base models.")
            return self.base_models_paths

    def run(self):
        self.logger.info(f"Initializing Training Workflow in {self.work_dir}")
        self.logger.info(f"Mode: {self.mode}")
        
        trainer = ParallelTrainer(
            base_config_path=self.input_json_path,
            work_dir=self.work_dir,
            num_models=self.num_models
        )
        
        finetune_heads = self._determine_finetune_heads()
        trainer.prepare_configs(self.seeds, self.training_seeds, finetune_heads)
        
        base_models = self._resolve_base_models()
        trainer.setup_workdirs(base_models, omp_threads=self.omp_threads)
        
        self.logger.info("Starting parallel training...")
        trainer.train(blocking=True)
        self.logger.info("Training Workflow Completed.")
