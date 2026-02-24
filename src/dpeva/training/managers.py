import os
import logging
import multiprocessing
from copy import deepcopy
from typing import List, Dict, Any, Optional

from dpeva.constants import WORKFLOW_FINISHED_TAG
from dpeva.submission import JobManager, JobConfig
from dpeva.utils.command import DPCommandBuilder

class TrainingConfigManager:
    """
    Manages Training Configuration:
    - Path resolution
    - Seed generation
    - Finetune head determination
    - Task-specific config generation
    """
    def __init__(self, base_config: Dict[str, Any], base_config_path: str):
        self.base_config = base_config
        self.base_config_path = base_config_path
        self.logger = logging.getLogger(__name__)

    def resolve_data_path(self, config: Dict[str, Any], task_idx: int, override_path: Optional[str] = None):
        """Resolves absolute paths for training data."""
        if "training" in config and "training_data" in config["training"]:
             data_config = config["training"]["training_data"]
             
             # 0. Override systems path if provided
             if override_path:
                 data_config["systems"] = override_path
                 self.logger.info(f"Task {task_idx}: Overridden data path with '{override_path}'")
             
             if "systems" in data_config:
                 original_path = data_config["systems"]
                 
                 # 1. Resolve relative path to absolute
                 if isinstance(original_path, str) and not os.path.isabs(original_path):
                     # Resolve relative to the base_config_path directory
                     base_dir = os.path.dirname(self.base_config_path)
                     abs_path = os.path.abspath(os.path.join(base_dir, original_path))
                     data_config["systems"] = abs_path
                     self.logger.info(f"Task {task_idx}: Resolved data path '{original_path}' -> '{abs_path}'")
                 
                 # 2. Expand directory if it's a container folder
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
                             sub_dirs.sort()
                             data_config["systems"] = sub_dirs
                             self.logger.info(f"Task {task_idx}: Auto-expanded data path '{current_systems_path}' into {len(sub_dirs)} sub-systems")
                         else:
                             self.logger.warning(f"Task {task_idx}: Path '{current_systems_path}' is a directory but contains no subdirectories and no type.raw.")

    def generate_seeds(self, num_models: int, user_seeds: Optional[List[int]] = None) -> List[int]:
        """Generate seeds for training."""
        default_seeds = [19090, 42, 10032, 2933]
        
        if user_seeds:
            return user_seeds
        
        if num_models > len(default_seeds):
            self.logger.warning(f"num_models ({num_models}) > default seeds length. Cycling default seeds.")
            return (default_seeds * (num_models // len(default_seeds) + 1))[:num_models]
        else:
            return default_seeds[:num_models]

    def get_finetune_heads(self, mode: str, head_name: str, num_models: int) -> List[str]:
        """Determine finetune heads based on mode."""
        if mode == "init":
            # First model uses configured head name, others use RANDOM
            heads = [head_name]
            if num_models > 1:
                heads.extend(["RANDOM"] * (num_models - 1))
            return heads
        elif mode == "cont":
            return [head_name] * num_models
        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'init' or 'cont'.")


    def prepare_task_configs(self, num_models: int, seeds: List[int], training_seeds: List[int], 
                            finetune_heads: List[str], override_data_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Generates a list of config dicts for each task."""
        
        if len(seeds) != num_models or len(training_seeds) != num_models or len(finetune_heads) != num_models:
            raise ValueError(f"Length of seeds/heads must match num_models ({num_models})")

        configs = []
        for i in range(num_models):
            config = deepcopy(self.base_config)
            
            # Resolve paths
            self.resolve_data_path(config, i, override_data_path)

            # Set seeds and heads
            if "fitting_net" in config["model"]:
                 config["model"]["fitting_net"]["seed"] = seeds[i]
            config["training"]["seed"] = training_seeds[i]
            config["model"]["finetune_head"] = finetune_heads[i]
            
            configs.append(config)
            
        return configs


class TrainingExecutionManager:
    """
    Manages Training Execution:
    - Command construction
    - Job script generation
    - Job submission
    """
    def __init__(self, backend: str, slurm_config: Dict, env_setup: str, dp_backend: str, template_path: Optional[str] = None):
        self.backend = backend
        self.slurm_config = slurm_config or {}
        self.env_setup = env_setup
        self.dp_backend = dp_backend
        
        DPCommandBuilder.set_backend(self.dp_backend)
        self.job_manager = JobManager(mode=backend, custom_template_path=template_path)
        self.logger = logging.getLogger(__name__)

    def generate_script(self, task_idx: int, task_dir: str, base_model_name: Optional[str], omp_threads: int) -> str:
        """Generates training script for a task."""
        
        # Construct Command
        gpus_per_node = self.slurm_config.get("gpus_per_node", 0)
        
        dp_freeze_cmd = DPCommandBuilder.freeze()
        
        if gpus_per_node > 1:
            # Multi-GPU Mode
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
            # Single GPU/CPU
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
            job_name=f"dpeva_train_{task_idx}",
            command=cmd,
            output_log="train.out",
            error_log="train.err",
            env_setup=self.env_setup,
            **task_slurm_config
        )
        
        script_name = "train.sh" if self.backend == "local" else "train.slurm"
        script_path = os.path.join(task_dir, script_name)
        
        self.job_manager.generate_script(job_config, script_path)
        return script_path

    def submit_jobs(self, script_paths: List[str], task_dirs: List[str], blocking: bool = True):
        """Submits all jobs."""
        processes = [] 
        
        if self.backend == "local":
            for i in range(len(script_paths)):
                p = multiprocessing.Process(
                    target=self.job_manager.submit,
                    args=(script_paths[i], task_dirs[i])
                )
                processes.append(p)
                p.start()
            
            if blocking:
                for p in processes:
                    p.join()
                self.logger.info("All local training tasks completed.")
            else:
                self.logger.info("Local training tasks started in background.")
                
        else: # Slurm
            for i in range(len(script_paths)):
                self.job_manager.submit(script_paths[i], task_dirs[i])
            
            self.logger.info("All Slurm jobs submitted. Please check queue.")
            if blocking:
                self.logger.warning("Blocking wait is not yet implemented for Slurm backend.")
