import os
import logging
from typing import Union, Dict

from dpeva.config import FeatureConfig
from dpeva.feature.managers import FeatureIOManager, FeatureExecutionManager
from dpeva.feature.generator import DescriptorGenerator
from dpeva.constants import WORKFLOW_FINISHED_TAG

class FeatureWorkflow:
    """
    Workflow for generating atomic/structural descriptors using DeepPot.
    Refactored using DDD Managers.
    """
    
    def __init__(self, config: Union[Dict, FeatureConfig]):
        """
        Initialize the Feature Workflow.

        Args:
            config (Union[Dict, FeatureConfig]): Configuration object or dictionary.
        """
        if isinstance(config, dict):
            self.config = FeatureConfig(**config)
        else:
            self.config = config

        self._setup_logger()
        
        # Core Configurations
        self.data_path = str(self.config.data_path)
        self.model_path = str(self.config.model_path)
        self.head = self.config.model_head
        self.output_mode = self.config.output_mode
        self.batch_size = self.config.batch_size
        self.mode = self.config.mode
        
        # Determine output directory
        if self.config.savedir:
            self.output_dir = str(self.config.savedir)
        else:
            # Auto-generate: desc-{model_stem}-{data_name}
            model_stem = os.path.splitext(os.path.basename(self.model_path))[0]
            data_name = os.path.basename(os.path.normpath(self.data_path))
            self.output_dir = f"desc-{model_stem}-{data_name}"
            
        # 1. Initialize IO Manager
        self.io_manager = FeatureIOManager()
        
        # 2. Initialize Execution Manager
        self.execution_manager = FeatureExecutionManager(
            backend=self.config.submission.backend,
            slurm_config=self.config.submission.slurm_config,
            env_setup=self.config.submission.env_setup,
            dp_backend=self.config.dp_backend,
            omp_threads=self.config.omp_threads
        )
        
    def _setup_logger(self):
        self.logger = logging.getLogger(__name__)

    def run(self):
        self.logger.info(f"Initializing Feature Workflow (Mode: {self.mode}, Backend: {self.execution_manager.backend})")
        
        if not os.path.exists(self.data_path):
            self.logger.error(f"Data path not found: {self.data_path}")
            return

        if self.mode == "cli":
            # CLI Mode: Use dp eval-desc
            # Detect multi-pool structure
            sub_pools = self.io_manager.detect_multi_pool_structure(self.data_path)
            
            self.execution_manager.submit_cli_job(
                data_path=self.data_path,
                output_dir=self.output_dir,
                model_path=self.model_path,
                head=self.head,
                sub_pools=sub_pools
            )
            
        elif self.mode == "python":
            # Python Mode: Use DeepPot API
            
            if self.execution_manager.backend == "local":
                # Local: Initialize Generator and run recursively
                try:
                    generator = DescriptorGenerator(
                        model_path=self.model_path,
                        head=self.head,
                        batch_size=self.batch_size,
                        omp_threads=self.config.omp_threads
                    )
                    
                    self.execution_manager.run_local_python_recursion(
                        generator,
                        data_path=self.data_path,
                        output_dir=self.output_dir,
                        output_mode=self.output_mode
                    )
                    self.logger.info(WORKFLOW_FINISHED_TAG)
                    
                except ImportError:
                    self.logger.error("DeepMD-kit not available for Python mode.")
                except Exception as e:
                    self.logger.error(f"Python mode execution failed: {e}", exc_info=True)
                    
            elif self.execution_manager.backend == "slurm":
                # Slurm: Submit a python script
                self.execution_manager.submit_python_slurm_job(
                    data_path=self.data_path,
                    output_dir=self.output_dir,
                    model_path=self.model_path,
                    head=self.head,
                    batch_size=self.batch_size,
                    output_mode=self.output_mode
                )
        else:
            self.logger.error(f"Unknown mode: {self.mode}")
