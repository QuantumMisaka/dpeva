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
        self.config = config
        self._setup_logger()
        
        self.datadir = config.get("datadir")
        self.modelpath = config.get("modelpath")
        self.format = config.get("format", "deepmd/npy")
        self.output_mode = config.get("output_mode", "structural") # 'atomic' or 'structural'
        
        self.savedir = config.get("savedir", f"desc-{os.path.basename(self.modelpath).split('.')[0]}-{os.path.basename(self.datadir)}")
        
        self.head = config.get("head", "OC20M")
        self.batch_size = config.get("batch_size", 1000)
        self.omp_threads = config.get("omp_threads", 24)
        
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
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)

    def run(self):
        """Execute the feature generation workflow."""
        self.logger.info("Initializing Feature Generation Workflow")
        
        if not os.path.exists(self.datadir):
            self.logger.error(f"Data directory not found: {self.datadir}")
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
            generator.run_cli_generation(self.datadir, self.savedir)
            self.logger.info(f"Job submitted. Output directory: {self.savedir}")
            
        else: # direct mode
            self.logger.info("Running in Direct (Python API) mode")
            # Logic for direct mode (iterating over subdirs manually if needed)
            # Original logic seemed to assume datadir was one system or handled by dpdata
            
            # Let's try to iterate over subdirectories to match typical dpdata usage if datadir is a pool
            subdirs = [os.path.join(self.datadir, d) for d in os.listdir(self.datadir) if os.path.isdir(os.path.join(self.datadir, d))]
            if not subdirs:
                # Maybe datadir is the system itself
                subdirs = [self.datadir]
                
            total_systems = len(subdirs)
            self.logger.info(f"Found {total_systems} potential systems to process")
            
            for i, system_path in enumerate(subdirs):
                system_name = os.path.basename(system_path)
                self.logger.info(f"Processing system {i+1}/{total_systems}: {system_name}")
                
                try:
                    desc = generator.compute_descriptors_direct(
                        data_path=system_path,
                        data_format=self.format,
                        output_mode=self.output_mode
                    )
                    
                    # Save descriptors in flat format: savedir/system_name.npy
                    output_file = os.path.join(self.savedir, f"{system_name}.npy")
                    np.save(output_file, desc)
                    self.logger.info(f"Saved descriptors to {output_file}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to process {system_name}: {e}")

        self.logger.info("Feature Generation Workflow Completed.")
