import os
import json
import shutil
import logging
import multiprocessing
from copy import deepcopy
from dpeva.constants import WORKFLOW_FINISHED_TAG
from dpeva.submission import JobManager, JobConfig
from dpeva.utils.command import DPCommandBuilder

class ParallelTrainer:
    """
    Manages parallel fine-tuning of multiple DeepMD models using a unified JobManager.
    Supports both local execution (multiprocessing) and Slurm submission.
    """
    
    def __init__(self, base_config_path, work_dir, num_models=4, 
                 backend="local", template_path=None, slurm_config=None, env_setup=None,
                 training_data_path=None, dp_backend="pt"):
        """
        Initialize the ParallelTrainer.

        Args:
            base_config_path (str): Path to the base training configuration file (JSON).
            work_dir (str): Directory where training artifacts and logs will be saved.
            num_models (int, optional): Number of models to train in the ensemble. Defaults to 4.
                Must be at least 2 for UQ calculation.
            backend (str, optional): Execution backend ('local' or 'slurm'). Defaults to "local".
            template_path (str, optional): Path to a custom submission script template. Defaults to None.
            slurm_config (dict, optional): Configuration dictionary for Slurm submission (e.g., partition, nodes).
                Defaults to None.
            env_setup (str or list[str], optional): Environment setup commands. Defaults to None.
            training_data_path (str, optional): Override path for training data systems. Defaults to None.
            dp_backend (str, optional): DeepMD-kit backend flag (e.g. 'pt', 'tf'). Defaults to "pt".
        """
        self.base_config_path = base_config_path
        self.work_dir = os.path.abspath(work_dir)
        self.num_models = num_models
        self.backend = backend
        self.slurm_config = slurm_config or {}
        self.env_setup = env_setup
        self.training_data_path = training_data_path
        self.dp_backend = dp_backend
        
        # Set backend for command builder
        DPCommandBuilder.set_backend(self.dp_backend)
        
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

        Args:
            seeds (list[int]): List of random seeds for the model fitting net.
            training_seeds (list[int]): List of random seeds for the training process.
            finetune_heads (list[str]): List of head names to finetune for each model.
        
        Raises:
            ValueError: If the length of seeds/heads does not match num_models.
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
            
    def setup_workdirs(self, base_models, omp_threads=1):
        """
        Create working directories and scripts for each model.

        Args:
            base_models (list[str]): List of paths to the base model files (e.g., *.pt).
            omp_threads (int, optional): Number of OpenMP threads to use for training. Defaults to 1.
                Setting this too high on shared resources may degrade performance.
        
        Raises:
            ValueError: If the length of base_models does not match num_models.
            FileNotFoundError: If a base model file is not found.
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
            
            # Copy base model if provided
            base_model_path = base_models[i]
            base_model_name = None
            
            if base_model_path:
                if not os.path.exists(base_model_path):
                    raise FileNotFoundError(f"Base model {base_model_path} not found")
                
                shutil.copy(base_model_path, folder_name)
                base_model_name = os.path.basename(base_model_path)
            
            # Construct Command
            gpus_per_node = self.slurm_config.get("gpus_per_node", 0)
            
            # Prepare DP Commands
            # Note: finetune path is just base_model_name because we copied it to the folder
            
            dp_freeze_cmd = DPCommandBuilder.freeze()
            
            if gpus_per_node > 1:
                # Multi-GPU Mode (torchrun)
                # Hardcoded skip_neighbor_stat=True for multi-gpu as per original code
                dp_train_cmd = DPCommandBuilder.train(
                    "input.json", 
                    finetune_path=base_model_name,
                    skip_neighbor_stat=True,
                    log_file="train.log"
                )
                
                cmd = f"""
export OMP_NUM_THREADS={omp_threads}
# torchrun command adapted from gpu_DPAtrain-multigpu.sbatch
torchrun --nproc_per_node=$((SLURM_NTASKS*SLURM_GPUS_ON_NODE)) \\
    --no-python --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \\
    {dp_train_cmd}
{dp_freeze_cmd}
echo "{WORKFLOW_FINISHED_TAG}"
"""
            else:
                # Standard single GPU/CPU command
                # Original code used skip_neighbor_stat=False (implied) for single GPU?
                # Actually original code: `dp --pt train input.json {finetune_flag} ...`
                # So skip_neighbor_stat defaults to False.
                
                dp_train_cmd = DPCommandBuilder.train(
                    "input.json", 
                    finetune_path=base_model_name,
                    log_file="train.log"
                )
                
                cmd = f"""
export OMP_NUM_THREADS={omp_threads}
export DP_INTER_OP_PARALLELISM_THREADS={omp_threads // 2}
export DP_INTRA_OP_PARALLELISM_THREADS={omp_threads}
{dp_train_cmd}
{dp_freeze_cmd}
echo "{WORKFLOW_FINISHED_TAG}"
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
                env_setup=self.env_setup,
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
