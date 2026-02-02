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
                 backend="local", template_path=None, slurm_config=None,
                 training_data_path=None):
        """
        Initialize the ParallelTrainer.
        """
        self.base_config_path = base_config_path
        self.work_dir = os.path.abspath(work_dir)
        self.num_models = num_models
        self.backend = backend
        self.slurm_config = slurm_config or {}
        self.training_data_path = training_data_path
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize JobManager
        self.job_manager = JobManager(mode=backend, custom_template_path=template_path)
        
        # Ensure base_config_path is absolute or correct relative to CWD
        if not os.path.exists(base_config_path):
             self.logger.error(f"Config file not found at: {os.path.abspath(base_config_path)}")
             raise FileNotFoundError(f"Config file not found: {base_config_path}")

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
            
            # Auto-resolve relative data path to absolute path
            # This fixes the issue where relative paths in input.json break when run in subdirectories
            if "training" in config and "training_data" in config["training"]:
                 data_config = config["training"]["training_data"]
                 
                 # 0. Override systems path if training_data_path is provided
                 if self.training_data_path:
                     data_config["systems"] = self.training_data_path
                     self.logger.info(f"Task {i}: Overridden data path with '{self.training_data_path}'")
                 
                 if "systems" in data_config:
                     original_path = data_config["systems"]
                     
                     # 1. Resolve relative path to absolute
                     if isinstance(original_path, str) and not os.path.isabs(original_path):
                         # Resolve relative to the base_config_path directory
                         base_dir = os.path.dirname(self.base_config_path)
                         abs_path = os.path.abspath(os.path.join(base_dir, original_path))
                         data_config["systems"] = abs_path
                         self.logger.info(f"Task {i}: Resolved data path '{original_path}' -> '{abs_path}'")
                     
                     # 2. Expand directory if it's a container folder (does not contain type.raw)
                     # DeepMD expects a list of system directories, not a parent directory.
                     current_systems_path = data_config["systems"]
                     if isinstance(current_systems_path, str) and os.path.isdir(current_systems_path):
                         # Check if this directory is ITSELF a system (has type.raw)
                         if not os.path.exists(os.path.join(current_systems_path, "type.raw")):
                             # It's likely a container directory. Scan for subdirectories.
                             sub_dirs = [
                                 os.path.join(current_systems_path, d) 
                                 for d in os.listdir(current_systems_path) 
                                 if os.path.isdir(os.path.join(current_systems_path, d))
                             ]
                             if sub_dirs:
                                 # Sort to ensure deterministic order
                                 sub_dirs.sort()
                                 data_config["systems"] = sub_dirs
                                 self.logger.info(f"Task {i}: Auto-expanded data path '{current_systems_path}' into {len(sub_dirs)} sub-systems")
                             else:
                                 self.logger.warning(f"Task {i}: Path '{current_systems_path}' is a directory but contains no subdirectories and no type.raw.")

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
            gpus_per_node = self.slurm_config.get("gpus_per_node", 0)
            
            if gpus_per_node > 1:
                # Use torchrun for multi-GPU
                cmd = f"""
export OMP_NUM_THREADS={omp_threads}
# torchrun command adapted from gpu_DPAtrain-multigpu.sbatch
torchrun --nproc_per_node=$((SLURM_NTASKS*SLURM_GPUS_ON_NODE)) \\
    --no-python --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \\
    dp --pt train input.json --skip-neighbor-stat --finetune {base_model_name} 2>&1 | tee train.log
dp --pt freeze
echo "DPEVA_TAG: WORKFLOW_FINISHED"
"""
            else:
                # Standard single GPU/CPU command
                cmd = f"""
export OMP_NUM_THREADS={omp_threads}
export DP_INTER_OP_PARALLELISM_THREADS={omp_threads // 2}
export DP_INTRA_OP_PARALLELISM_THREADS={omp_threads}
dp --pt train input.json --finetune {base_model_name} 2>&1 | tee train.log
dp --pt freeze
echo "DPEVA_TAG: WORKFLOW_FINISHED"
"""
            # Create JobConfig
            task_slurm_config = self.slurm_config.copy()
            task_slurm_config.pop("job_name", None)
            task_slurm_config.pop("output_log", None)
            task_slurm_config.pop("error_log", None)
            
            job_config = JobConfig(
                job_name=f"dpeva_train_{i}",
                command=cmd,
                output_log="train.out",
                error_log="train.err",
                **task_slurm_config # Inject partition, nodes, etc.
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
        processes = [] 
        
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
            if blocking:
                self.logger.warning("Blocking wait is not yet implemented for Slurm backend.")
