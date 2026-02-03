import os
import glob
import time
import logging
import numpy as np
from dpeva.feature.generator import DescriptorGenerator

class FeatureWorkflow:
    """
    Workflow for generating descriptors for a dataset using a pre-trained model.
    """
    
    def __init__(self, config):
        """
        Initialize the Feature Generation Workflow.

        Args:
            config (dict): Configuration dictionary containing:
                - data_path (str): Path to dataset (Required).
                - modelpath (str): Path to model file (Required).
                - format (str): Data format (default: "deepmd/npy").
                - output_mode (str): 'atomic' or 'structural' (default: "atomic"). 
                                     Note: Only effective in 'python' mode. 'cli' mode always outputs atomic descriptors.
                - savedir (str): Output directory (default: auto-generated).
                - head (str): Model head (default: "OC20M").
                - batch_size (int): Batch size (default: 1000).
                - omp_threads (int): OpenMP threads (default: 1).
                - mode (str): 'cli' or 'python' (default: "cli").
                - submission (dict): Submission config.
        """
        self.config = config
        self._setup_logger()
        
        self.data_path = config.get("data_path")
        self.modelpath = config.get("modelpath")
        self.format = config.get("format", "deepmd/npy")
        self.output_mode = config.get("output_mode", "atomic") # 'atomic' or 'structural'
        
        self.savedir = config.get("savedir", f"desc-{os.path.basename(self.modelpath).split('.')[0]}-{os.path.basename(self.data_path)}")
        
        self.head = config.get("head", "OC20M")
        self.batch_size = config.get("batch_size", 1000)
        self.omp_threads = config.get("omp_threads", 1)
        
        # New configurations for CLI/Slurm support
        self.mode = config.get("mode", "cli") # Default to CLI mode as requested
        self.submission_config = config.get("submission", {})
        self.backend = self.submission_config.get("backend", "local")
        
        # Handle env_setup: support string or list of strings
        raw_env_setup = self.submission_config.get("env_setup", "")
        if isinstance(raw_env_setup, list):
            self.env_setup = "\n".join(raw_env_setup)
        else:
            self.env_setup = raw_env_setup
            
        self.slurm_config = self.submission_config.get("slurm_config", {})

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
                env_setup=self.env_setup
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
                data_format=self.format,
                output_mode=self.output_mode
            )
        else:
            self.logger.error(f"Unknown mode: {self.mode}")

        self.logger.info("Feature Generation Workflow Completed.")
