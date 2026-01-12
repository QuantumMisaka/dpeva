import os
import json
import shutil
import logging
import multiprocessing
from copy import deepcopy
from dpeva.submission import JobManager, JobConfig

class ParallelTrainer:
    """
    Manages parallel fine-tuning of multiple DeepMD models using a unified JobManager.
    Supports both local execution (multiprocessing) and Slurm submission.
    """
    
    def __init__(self, base_config_path, work_dir, num_models=4, 
                 backend="local", template_path=None, slurm_config=None):
        """
        Initialize the ParallelTrainer.
        
        Args:
            base_config_path (str): Path to the base input.json template.
            work_dir (str): Root directory for training tasks.
            num_models (int): Number of models to train in parallel (default: 4).
            backend (str): 'local' or 'slurm'.
            template_path (str): Path to custom submission template file.
            slurm_config (dict): Additional Slurm configuration (partition, nodes, etc.).
        """
        self.base_config_path = base_config_path
        self.work_dir = os.path.abspath(work_dir)
        self.num_models = num_models
        self.backend = backend
        self.slurm_config = slurm_config or {}
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize JobManager
        self.job_manager = JobManager(mode=backend, custom_template_path=template_path)
        
        with open(base_config_path, 'r') as f:
            self.base_config = json.load(f)

    def prepare_configs(self, seeds, training_seeds, finetune_heads):
        """
        Prepare configuration files for each model.
        """
        if len(seeds) != self.num_models or \
           len(training_seeds) != self.num_models or \
           len(finetune_heads) != self.num_models:
            raise ValueError(f"Length of seeds/heads must match num_models ({self.num_models})")

        self.configs = []
        for i in range(self.num_models):
            config = deepcopy(self.base_config)
            if "fitting_net" in config["model"]:
                 config["model"]["fitting_net"]["seed"] = seeds[i]
            config["training"]["seed"] = training_seeds[i]
            config["model"]["finetune_head"] = finetune_heads[i]
            self.configs.append(config)
            
    def setup_workdirs(self, base_models, omp_threads=12):
        """
        Create working directories and scripts for each model.
        """
        if len(base_models) != self.num_models:
             raise ValueError(f"Length of base_models must match num_models ({self.num_models})")

        self.task_dirs = []
        self.script_paths = []
        
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
            
            shutil.copy(base_model_path, folder_name)
            base_model_name = os.path.basename(base_model_path)
            
            # Construct Command
            cmd = f"""
export OMP_NUM_THREADS={omp_threads}
export DP_INTER_OP_PARALLELISM_THREADS={omp_threads // 2}
export DP_INTRA_OP_PARALLELISM_THREADS={omp_threads}
dp --pt train input.json --finetune {base_model_name} 2>&1 | tee train.log
"""
            # Create JobConfig
            job_config = JobConfig(
                job_name=f"dpeva_train_{i}",
                command=cmd,
                output_log="train.out",
                error_log="train.err",
                **self.slurm_config # Inject partition, nodes, etc.
            )
            
            # Generate Script
            script_name = "train.sh" if self.backend == "local" else "train.slurm"
            script_path = os.path.join(folder_name, script_name)
            self.job_manager.generate_script(job_config, script_path)
            self.script_paths.append(script_path)

    def train(self, blocking=True):
        """
        Start parallel training using JobManager.
        
        Args:
            blocking (bool): If True, wait for all tasks to complete (Only valid for 'local' backend in current design).
        """
        processes = [] # For local backend tracking
        
        # In Slurm mode, 'blocking' usually means polling squeue, which is complex.
        # For now, we enforce blocking behavior only for local backend via multiprocessing pool wrapper if needed,
        # BUT JobManager.submit is synchronous for 'local' (it runs subprocess.run).
        # Wait, if JobManager.submit is blocking for local, we can't run parallel!
        
        # Correction: JobManager.submit for 'local' executes 'bash script.sh'.
        # If we want parallel local execution, we need to wrap JobManager.submit in multiprocessing here.
        
        if self.backend == "local":
            # Use multiprocessing to spawn local jobs in parallel
            for i in range(self.num_models):
                p = multiprocessing.Process(
                    target=self.job_manager.submit,
                    args=(self.script_paths[i], self.task_dirs[i])
                )
                processes.append(p)
                p.start()
            
            if blocking:
                for p in processes:
                    p.join()
                self.logger.info("All local training tasks completed.")
            else:
                self.logger.info("Local training tasks started in background.")
                
        else: # Slurm Backend
            # Slurm submission is fast, we can do it sequentially
            for i in range(self.num_models):
                self.job_manager.submit(self.script_paths[i], self.task_dirs[i])
            
            self.logger.info("All Slurm jobs submitted. Please check queue.")
            # Note: 'blocking' is currently ignored for Slurm mode as we haven't implemented polling yet.
            if blocking:
                self.logger.warning("Blocking wait is not yet implemented for Slurm backend.")
