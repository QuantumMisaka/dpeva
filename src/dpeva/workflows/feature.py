import os
import glob
import time
import logging
import numpy as np
from typing import Union, Dict
from dpeva.config import FeatureConfig
from dpeva.feature.generator import DescriptorGenerator

class FeatureWorkflow:
    """
    Workflow for generating descriptors for a dataset using a pre-trained model.
    """
    
    def __init__(self, config: Union[Dict, FeatureConfig]):
        """
        Initialize the Feature Generation Workflow.

        Args:
            config (Union[Dict, FeatureConfig]): Configuration object or dictionary.
        """
        if isinstance(config, dict):
            self.config = FeatureConfig(**config)
        else:
            self.config = config
            
        self._setup_logger()
        
        self.data_path = str(self.config.data_path)
        self.modelpath = str(self.config.model_path)
        self.output_mode = self.config.output_mode # 'atomic' or 'structural'
        
        # savedir is auto-populated by Pydantic if None
        self.savedir = str(self.config.savedir)
        
        self.head = self.config.model_head
        self.batch_size = self.config.batch_size
        self.omp_threads = self.config.omp_threads
        
        # New configurations for CLI/Slurm support
        self.mode = self.config.mode 
        self.submission_config = self.config.submission
        self.backend = self.submission_config.backend
        
        # DeepMD Backend
        self.dp_backend = self.config.dp_backend
        
        # Handle env_setup: Pydantic handles validation/conversion to string
        self.env_setup = self.submission_config.env_setup
            
        self.slurm_config = self.submission_config.slurm_config

    def _setup_logger(self):
        self.logger = logging.getLogger(__name__)

    def run(self):
        """Execute the feature generation workflow."""
        self.logger.info("Initializing Feature Generation Workflow")
        
        if not os.path.exists(self.data_path):
            self.logger.error(f"Data directory not found: {self.data_path}")
            return
            
        # Initialize Generator
        # Pass new parameters for CLI mode
        try:
            generator = DescriptorGenerator(
                model_path=self.modelpath, 
                head=self.head,
                batch_size=self.batch_size,
                omp_threads=self.omp_threads,
                mode=self.mode,
                backend=self.backend,
                slurm_config=self.slurm_config,
                env_setup=self.env_setup,
                dp_backend=self.dp_backend
            )
        except (ImportError, ValueError) as e:
            self.logger.error(f"Failed to initialize DescriptorGenerator: {e}")
            return
            
        os.makedirs(self.savedir, exist_ok=True)
        
        # Identify sub-datasets (directories)
        # We assume datadir contains subfolders, each being a dpdata system
        # e.g. datadir/system1, datadir/system2...
        # If datadir itself is a system, logic might differ, but dpdata.MultiSystems handles structure well.
        
        # For CLI mode (dp eval-desc), it can take a single directory or we loop over subdirectories.
        # The prompt mentions:
        # dp --pt eval-desc -s $POOLDATA -m $MODEL -o $SAVEDIR --head $HEAD
        # This implies dp eval-desc can handle a directory of systems recursively or as a set.
        # Let's check if we should loop or submit one job.
        # The sbatch template shows: dp --pt eval-desc -s $POOLDATA ...
        # So we submit ONE job for the entire POOLDATA directory.
        
        if self.mode == "cli":
            self.logger.info(f"Running in CLI mode with backend: {self.backend}")
            # Submit single job for the whole directory
            generator.run_cli_generation(self.data_path, self.savedir)
            self.logger.info(f"Job submitted. Output directory: {self.savedir}")
            
        elif self.mode == "python": # Python Native mode
            self.logger.info(f"Running in Python Native mode (backend: {self.backend})")
            generator.run_python_generation(
                data_path=self.data_path,
                output_dir=self.savedir,
                data_format="auto",
                output_mode=self.output_mode
            )
        else:
            self.logger.error(f"Unknown mode: {self.mode}")

        self.logger.info("Feature Generation Workflow Completed.")
