import os
import json
import shutil
import logging
import subprocess
import multiprocessing
from copy import deepcopy

class ParallelTrainer:
    """
    Manages parallel fine-tuning of multiple DeepMD models.
    """
    
    def __init__(self, base_config_path, work_dir, num_models=4):
        """
        Initialize the ParallelTrainer.
        
        Args:
            base_config_path (str): Path to the base input.json template.
            work_dir (str): Root directory for training tasks.
            num_models (int): Number of models to train in parallel (default: 4).
        """
        self.base_config_path = base_config_path
        self.work_dir = os.path.abspath(work_dir)
        self.num_models = num_models
        self.logger = logging.getLogger(__name__)
        
        with open(base_config_path, 'r') as f:
            self.base_config = json.load(f)

    def prepare_configs(self, seeds, training_seeds, finetune_heads):
        """
        Prepare configuration files for each model.
        
        Args:
            seeds (list): List of seeds for fitting_net.
            training_seeds (list): List of seeds for training.
            finetune_heads (list): List of finetune head names.
        """
        if len(seeds) != self.num_models or \
           len(training_seeds) != self.num_models or \
           len(finetune_heads) != self.num_models:
            raise ValueError(f"Length of seeds/heads must match num_models ({self.num_models})")

        self.configs = []
        for i in range(self.num_models):
            config = deepcopy(self.base_config)
            
            # Update seeds and head
            if "fitting_net" in config["model"]:
                 config["model"]["fitting_net"]["seed"] = seeds[i]
            # Handle DPA-2/3 descriptor seed if needed (usually in model.descriptor)
            # Assuming standard structure per original script:
            
            config["training"]["seed"] = training_seeds[i]
            config["model"]["finetune_head"] = finetune_heads[i]
            
            self.configs.append(config)
            
    def setup_workdirs(self, base_models, omp_threads=12):
        """
        Create working directories and scripts for each model.
        
        Args:
            base_models (list): List of paths to base model checkpoints.
            omp_threads (int): Number of OMP threads per training task.
        """
        if len(base_models) != self.num_models:
             raise ValueError(f"Length of base_models must match num_models ({self.num_models})")

        self.task_dirs = []
        for i in range(self.num_models):
            folder_name = os.path.join(self.work_dir, str(i))
            os.makedirs(folder_name, exist_ok=True)
            self.task_dirs.append(folder_name)
            
            # Save input.json
            with open(os.path.join(folder_name, "input.json"), "w") as f:
                json.dump(self.configs[i], f, indent=4)
            
            # Copy base model
            base_model_path = base_models[i]
            if not os.path.exists(base_model_path):
                raise FileNotFoundError(f"Base model {base_model_path} not found")
            
            # We copy the model to the task dir to ensure isolation or just reference it?
            # Original script copies it.
            shutil.copy(base_model_path, folder_name)
            base_model_name = os.path.basename(base_model_path)
            
            # Generate train.sh
            train_script = f'''#!/bin/bash
export OMP_NUM_THREADS={omp_threads}
export DP_INTER_OP_PARALLELISM_THREADS={omp_threads // 2}
export DP_INTRA_OP_PARALLELISM_THREADS={omp_threads}
dp --pt train input.json --finetune {base_model_name} 2>&1 | tee train.log
'''
            with open(os.path.join(folder_name, "train.sh"), "w") as f:
                f.write(train_script)

    def _run_single_task(self, task_idx):
        """Run training for a single task."""
        task_dir = self.task_dirs[task_idx]
        self.logger.info(f"Starting training for model {task_idx} in {task_dir}")
        
        try:
            # Using subprocess to run the shell script
            # We use 'bash train.sh' instead of './train.sh' to avoid permission issues
            result = subprocess.run(
                ["bash", "train.sh"],
                cwd=task_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if result.returncode == 0:
                self.logger.info(f"Training for model {task_idx} completed successfully.")
            else:
                self.logger.error(f"Training for model {task_idx} failed. Error: {result.stderr}")
                
        except Exception as e:
            self.logger.error(f"Exception during training model {task_idx}: {e}")

    def train(self, blocking=True):
        """
        Start parallel training.
        
        Args:
            blocking (bool): If True, wait for all tasks to complete.
        """
        processes = []
        for i in range(self.num_models):
            p = multiprocessing.Process(target=self._run_single_task, args=(i,))
            processes.append(p)
            p.start()
        
        if blocking:
            for p in processes:
                p.join()
            self.logger.info("All training tasks finished.")
        else:
            self.logger.info("Training tasks started in background.")
            return processes
