import os
import glob
import time
import logging
import json
import numpy as np
import pandas as pd
import dpdata
from typing import Union, Dict
from collections import Counter
from dpeva.submission import JobManager, JobConfig
from dpeva.io.dataproc import DPTestResultParser
from dpeva.inference.stats import StatsCalculator
from dpeva.inference.visualizer import InferenceVisualizer
from dpeva.config import InferenceConfig

class InferenceWorkflow:
    """
    Workflow for running inference using an ensemble of DPA models.
    Supports both Local and Slurm backends via JobManager.
    Also handles result parsing and visualization.
    """
    
    def __init__(self, config: Union[Dict, InferenceConfig]):
        """
        Initialize the Inference Workflow.

        Args:
            config (Union[Dict, InferenceConfig]): Configuration object or dictionary.
        """
        if isinstance(config, dict):
            self.config = InferenceConfig(**config)
        else:
            self.config = config

        self._setup_logger()
        
        # Data and Model Configuration
        self.data_path = str(self.config.data_path)
        self.work_dir = str(self.config.work_dir)
        
        # Auto-infer models_paths from work_dir structure
        # Assumes structure: work_dir/[0,1,2,3...]/model.ckpt.pt
        self.models_paths = []
        if os.path.exists(self.work_dir):
            i = 0
            while True:
                possible_model = os.path.join(self.work_dir, str(i), "model.ckpt.pt")
                if os.path.exists(possible_model):
                    self.models_paths.append(possible_model)
                    i += 1
                else:
                    break
        
        self.task_name = self.config.task_name
        self.head = self.config.model_head
        self.results_prefix = self.config.results_prefix
        
        # Submission Configuration
        self.backend = self.config.submission.backend
        self.slurm_config = self.config.submission.slurm_config
        self.env_setup = self.config.submission.env_setup
        
        # Parallelism (for OMP settings if not in env_setup)
        self.omp_threads = self.config.omp_threads
        
    def _setup_logger(self):
        self.logger = logging.getLogger(__name__)

    def _get_default_env_setup(self):
        """Provide default environment variables if user didn't specify any."""
        return f"""
export OMP_NUM_THREADS={self.omp_threads}
"""

    def run(self):
        self.logger.info(f"Initializing Inference Workflow (Backend: {self.backend})")
        
        if not self.data_path or not os.path.exists(self.data_path):
            self.logger.error(f"Test data path not found: {self.data_path}")
            return

        if not self.models_paths:
            self.logger.error("No models provided for inference.")
            return

        # Initialize Job Manager
        try:
            manager = JobManager(mode=self.backend)
        except ValueError as e:
            self.logger.error(str(e))
            return

        # Determine Environment Setup
        final_env_setup = self.env_setup if self.env_setup.strip() else self._get_default_env_setup()

        self.logger.info(f"Submitting {len(self.models_paths)} inference jobs...")
        
        for i, model_path in enumerate(self.models_paths):
            if not os.path.exists(model_path):
                self.logger.warning(f"Model file not found: {model_path}, skipping.")
                continue
                
            # Define output directory structure: work_dir/i/task_name
            if self.task_name:
                job_work_dir = os.path.join(self.work_dir, str(i), self.task_name)
            else:
                job_work_dir = os.path.join(self.work_dir, str(i))
                
            os.makedirs(job_work_dir, exist_ok=True)
            
            # Construct Command
            abs_data_path = os.path.abspath(self.data_path)
            abs_model_path = os.path.abspath(model_path)
            
            # Command: dp --pt test ...
            # -d specifies output prefix
            cmd = (
                f"dp --pt test "
                f"-s {abs_data_path} "
                f"-m {abs_model_path} "
                f"-d {self.results_prefix} "
            )
            
            if self.head:
                cmd += f"--head {self.head} "
            
            if self.backend == "local":
                cmd += f"2>&1 | tee test.log"
            
            # Append completion marker
            cmd += f"\necho \"DPEVA_TAG: WORKFLOW_FINISHED\""
            
            # Create JobConfig
            job_name = f"dp_test_{i}"
            job_config = JobConfig(
                job_name=job_name,
                command=cmd,
                env_setup=final_env_setup,
                output_log="test_job.log",
                error_log="test_job.err",
                # Slurm specific params from config
                partition=self.slurm_config.get("partition", "partition"),
                nodes=self.slurm_config.get("nodes", 1),
                ntasks=self.slurm_config.get("ntasks", 1),
                gpus_per_node=self.slurm_config.get("gpus_per_node", 0),
                qos=self.slurm_config.get("qos"),
                nodelist=self.slurm_config.get("nodelist"),
                walltime=self.slurm_config.get("walltime", "24:00:00")
            )
            
            # Generate Script
            script_name = "run_test.slurm" if self.backend == "slurm" else "run_test.sh"
            script_path = os.path.join(job_work_dir, script_name)
            
            manager.generate_script(job_config, script_path)
            
            # Submit Job
            manager.submit(script_path, working_dir=job_work_dir)
            
        self.logger.info("Inference Workflow Submission Completed.")
        
        # Optional: Auto-analyze if local and blocking (conceptually, JobManager local is blocking)
        if self.backend == "local":
            self.logger.info("Local execution completed. Starting analysis...")
            self.analyze_results()

    def analyze_results(self, output_dir_suffix="analysis"):
        """
        Parse results, compute metrics, and generate plots for all models.
        """
        self.logger.info("Starting result analysis...")
        
        # NOTE: The current analysis logic, especially for cohesive energy, heavily relies on 
        # the 'dp test' output containing system names that reflect the composition (e.g., 'H2O1').
        # This implicitly assumes the test dataset is organized in 'deepmd/npy' format where 
        # directory names are system names. 'deepmd/npy/mixed' or other formats where system 
        # names do not contain composition info might cause the relative energy calculation to fail 
        # or require fallback to mean-subtraction.
        self.logger.warning("Note: This analysis workflow assumes the test dataset is in 'deepmd/npy' format "
                            "(directory names = system names) for correct composition parsing from 'dp test' results. "
                            "'deepmd/npy/mixed' format is not fully supported for cohesive energy analysis.")
        
        # Pre-load atom counts if possible for cohesive energy calculation
        atom_counts_list = None
        atom_num_list = None
        
        if self.data_path and os.path.exists(self.data_path):
            try:
                self.logger.info(f"Loading system composition from {self.data_path} using dpdata...")
                loaded_systems = load_systems(self.data_path)
                    
                atom_counts_list = []
                atom_num_list = []
                
                # Iterate in the same order as dp test usually does (assumed alphabetical/system order)
                for s in loaded_systems:
                    atom_names = s["atom_names"]
                    atom_types = s["atom_types"]
                    elements = [atom_names[t] for t in atom_types]
                    counts = Counter(elements)
                    n_atoms = len(elements)
                    
                    # Replicate for each frame in the system
                    for _ in range(s.get_nframes()):
                        atom_counts_list.append(counts)
                        atom_num_list.append(n_atoms)
                        
                self.logger.info(f"Loaded composition info for {len(atom_counts_list)} frames.")
            except Exception as e:
                self.logger.warning(f"Failed to load composition info: {e}. Relative energy will use mean subtraction.")
        else:
            self.logger.warning("dpdata not available or test_data_path invalid. Skipping composition loading.")

        summary_metrics = []
        
        for i, model_path in enumerate(self.models_paths):
            # Resolve work_dir same as run()
            if self.task_name:
                job_work_dir = os.path.join(self.work_dir, str(i), self.task_name)
            else:
                job_work_dir = os.path.join(self.work_dir, str(i))
                
            if not os.path.exists(job_work_dir):
                self.logger.warning(f"Work dir not found: {job_work_dir}, skipping analysis for model {i}")
                continue
                
            # Create analysis output dir inside work_dir
            analysis_dir = os.path.join(job_work_dir, output_dir_suffix)
            os.makedirs(analysis_dir, exist_ok=True)
            
            try:
                # 1. Parse Results
                parser = DPTestResultParser(job_work_dir, head=self.results_prefix)
                data = parser.parse()
                
                # Check consistency between loaded composition and parsed data
                current_atom_counts = atom_counts_list
                current_atom_nums = atom_num_list
                
                if current_atom_counts is not None:
                    if len(current_atom_counts) != len(data["energy"]["pred_e"]):
                        self.logger.warning(f"Frame count mismatch: dpdata loaded {len(current_atom_counts)}, "
                                            f"dp test has {len(data['energy']['pred_e'])}. Disabling composition info.")
                        current_atom_counts = None
                        current_atom_nums = None
                
                # 2. Prepare Data for Calculator
                f_pred = None
                f_true = None
                
                if data["force"] is not None:
                    f_pred = np.column_stack((
                        data["force"]["pred_fx"], 
                        data["force"]["pred_fy"], 
                        data["force"]["pred_fz"]
                    )).flatten()
                    
                    if data["has_ground_truth"]:
                        f_true = np.column_stack((
                            data["force"]["data_fx"], 
                            data["force"]["data_fy"], 
                            data["force"]["data_fz"]
                        )).flatten()
                
                # Try to load ref_energies from config (Optional, not in schema yet)
                # We can access config dict via model_dump() or getattr if added to schema
                # For now assume None as it wasn't in Pydantic schema
                ref_energies = getattr(self.config, "ref_energies", None)

                stats_calc = StatsCalculator(
                    energy_per_atom=data["energy"]["pred_e"],
                    force_flat=f_pred,
                    energy_true=data["energy"]["data_e"] if data["has_ground_truth"] else None,
                    force_true=f_true,
                    atom_counts_list=current_atom_counts,
                    atom_num_list=current_atom_nums,
                    ref_energies=ref_energies
                )
                
                viz = InferenceVisualizer(analysis_dir)
                
                # 3. Metrics & Parity Plots
                if data["has_ground_truth"]:
                    metrics = stats_calc.compute_metrics()
                    metrics["model_idx"] = i
                    summary_metrics.append(metrics)
                    
                    viz.plot_parity(stats_calc.e_true, stats_calc.e_pred, "Energy", "eV/atom")
                    if stats_calc.f_true is not None:
                        viz.plot_parity(stats_calc.f_true, stats_calc.f_pred, "Force", "eV/A")
                        
                    viz.plot_error_distribution(stats_calc.e_pred - stats_calc.e_true, "Energy", "eV/atom")
                    if stats_calc.f_true is not None:
                        viz.plot_error_distribution(stats_calc.f_pred - stats_calc.f_true, "Force", "eV/A")

                # 4. Distributions Analysis (No Outliers)
                # Energy
                viz.plot_distribution(stats_calc.e_pred, "Predicted Energy", "eV/atom")
                if stats_calc.e_true is not None:
                    viz.plot_distribution(stats_calc.e_true, "True Energy", "eV/atom", color="green")
                
                # Relative Energy Distribution (Only if composition available)
                e_rel = stats_calc.compute_relative_energy(stats_calc.e_pred)
                if e_rel is not None:
                    label_rel = "Cohesive Energy"
                    viz.plot_distribution(e_rel, label_rel, "eV/atom", color="purple")
                    
                    # If we have ground truth, plot parity for Cohesive Energy
                    if stats_calc.e_true is not None:
                        e_rel_true = stats_calc.compute_relative_energy(stats_calc.e_true)
                        if e_rel_true is not None:
                            viz.plot_parity(e_rel_true, e_rel, "Cohesive Energy", "eV/atom")
                            viz.plot_error_distribution(e_rel - e_rel_true, "Cohesive Energy", "eV/atom")
                            
                            if stats_calc.e_true is not None:
                                viz.plot_distribution(e_rel_true, "True Cohesive Energy", "eV/atom", color="magenta")
                
                # Force Magnitude
                if stats_calc.f_pred is not None:
                    f_pred_norm = stats_calc.compute_force_magnitude(stats_calc.f_pred)
                    viz.plot_distribution(f_pred_norm, "Predicted Force Magnitude", "eV/A", color="orange")
                    
                    if stats_calc.f_true is not None:
                        f_true_norm = stats_calc.compute_force_magnitude(stats_calc.f_true)
                        viz.plot_distribution(f_true_norm, "True Force Magnitude", "eV/A", color="teal")

                # Virial
                if stats_calc.v_pred is not None:
                    # Plot flattened virial components
                    viz.plot_distribution(stats_calc.v_pred.flatten(), "Predicted Virial", "eV", color="red")
                    
                    if stats_calc.v_true is not None:
                        viz.plot_distribution(stats_calc.v_true.flatten(), "True Virial", "eV", color="cyan")

                # Save Statistics JSON
                stats_export = {
                    "energy": stats_calc.get_distribution_stats(stats_calc.e_pred, "energy"),
                }
                if stats_calc.e_true is not None:
                    stats_export["energy_true"] = stats_calc.get_distribution_stats(stats_calc.e_true, "energy_true")
                
                if e_rel is not None:
                    stats_export["relative_energy"] = stats_calc.get_distribution_stats(e_rel, "relative_energy")
                    
                if stats_calc.f_pred is not None:
                    stats_export["force_magnitude"] = stats_calc.get_distribution_stats(f_pred_norm, "force_magnitude")
                    if stats_calc.f_true is not None:
                        stats_export["force_magnitude_true"] = stats_calc.get_distribution_stats(f_true_norm, "force_magnitude_true")

                if stats_calc.v_pred is not None:
                    stats_export["virial"] = stats_calc.get_distribution_stats(stats_calc.v_pred.flatten(), "virial")
                    if stats_calc.v_true is not None:
                        stats_export["virial_true"] = stats_calc.get_distribution_stats(stats_calc.v_true.flatten(), "virial_true")


                def default(o):
                    if isinstance(o, (np.integer, int)): return int(o)
                    if isinstance(o, (np.floating, float)): return float(o)
                    if isinstance(o, np.ndarray): return o.tolist()
                    return str(o)
                    
                with open(os.path.join(analysis_dir, "statistics.json"), "w") as f:
                    json.dump(stats_export, f, indent=4, default=default)

            except Exception as e:
                self.logger.error(f"Analysis failed for model {i}: {e}", exc_info=True)
                
        # Save Global Summary
        if summary_metrics:
            summary_path = os.path.join(self.work_dir, "inference_summary.csv")
            pd.DataFrame(summary_metrics).to_csv(summary_path, index=False)
            self.logger.info(f"Analysis completed. Summary saved to {summary_path}")
