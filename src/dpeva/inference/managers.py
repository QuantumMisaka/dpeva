import os
import json
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Tuple
from collections import Counter

from dpeva.constants import WORKFLOW_FINISHED_TAG, FILENAME_STATS_JSON
from dpeva.submission import JobManager, JobConfig
from dpeva.io.dataproc import DPTestResultParser
from dpeva.inference.stats import StatsCalculator
from dpeva.inference.visualizer import InferenceVisualizer
from dpeva.io.dataset import load_systems
from dpeva.utils.command import DPCommandBuilder

logger = logging.getLogger(__name__)

class InferenceIOManager:
    """
    Manages IO operations for Inference Workflow:
    - Model discovery
    - Data loading (composition info)
    - Result parsing
    - Statistics saving
    """
    def __init__(self, work_dir: str):
        self.work_dir = work_dir
        self.logger = logging.getLogger(__name__)

    def discover_models(self) -> List[str]:
        """Discover models in work_dir subdirectories (0/, 1/, ...)."""
        models_paths = []
        if os.path.exists(self.work_dir):
            i = 0
            while True:
                possible_model = os.path.join(self.work_dir, str(i), "model.ckpt.pt")
                if os.path.exists(possible_model):
                    models_paths.append(possible_model)
                    i += 1
                else:
                    break
        return models_paths

    def load_composition_info(self, data_path: str) -> Tuple[Optional[List[Dict]], Optional[List[int]]]:
        """Load composition info using dpdata."""
        if not data_path or not os.path.exists(data_path):
            self.logger.warning("dpdata not available or test_data_path invalid. Skipping composition loading.")
            return None, None

        try:
            self.logger.info(f"Loading system composition from {data_path} using dpdata...")
            loaded_systems = load_systems(data_path)
                
            atom_counts_list = []
            atom_num_list = []
            
            # Iterate in the same order as dp test usually does (assumed alphabetical/system order)
            # load_systems returns a list, usually sorted by how it discovered them.
            # Ideally we trust load_systems order matches dp test order if directory traversal is consistent.
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
            return atom_counts_list, atom_num_list
        except Exception as e:
            self.logger.warning(f"Failed to load composition info: {e}. Relative energy will use mean subtraction.")
            return None, None

    def parse_results(self, job_work_dir: str, prefix: str) -> Dict:
        """Parse dp test results."""
        parser = DPTestResultParser(job_work_dir, head=prefix)
        return parser.parse()

    def save_statistics(self, analysis_dir: str, stats_data: Dict):
        """Save statistics to JSON."""
        def default(o):
            if isinstance(o, (np.integer, int)): return int(o)
            if isinstance(o, (np.floating, float)): return float(o)
            if isinstance(o, np.ndarray): return o.tolist()
            return str(o)
            
        with open(os.path.join(analysis_dir, FILENAME_STATS_JSON), "w") as f:
            json.dump(stats_data, f, indent=4, default=default)

    def save_summary(self, summary_metrics: List[Dict]):
        """Save summary CSV."""
        if summary_metrics:
            summary_path = os.path.join(self.work_dir, "inference_summary.csv")
            pd.DataFrame(summary_metrics).to_csv(summary_path, index=False)
            self.logger.info(f"Analysis completed. Summary saved to {summary_path}")


class InferenceExecutionManager:
    """
    Manages Execution for Inference Workflow:
    - Command construction
    - Job submission
    """
    def __init__(self, backend: str, slurm_config: Dict, env_setup: str, dp_backend: str, omp_threads: int):
        self.backend = backend
        self.slurm_config = slurm_config or {}
        self.env_setup = env_setup
        self.dp_backend = dp_backend
        self.omp_threads = omp_threads
        
        DPCommandBuilder.set_backend(self.dp_backend)
        self.job_manager = JobManager(mode=backend)
        self.logger = logging.getLogger(__name__)

    def _get_default_env_setup(self):
        """Provide default environment variables if user didn't specify any."""
        return f"export OMP_NUM_THREADS={self.omp_threads}"

    def submit_jobs(self, models_paths: List[str], data_path: str, work_dir: str, task_name: str, 
                   head: str, results_prefix: str):
        """Submit inference jobs for all models."""
        final_env_setup = self.env_setup if self.env_setup.strip() else self._get_default_env_setup()
        
        self.logger.info(f"Submitting {len(models_paths)} inference jobs...")
        
        for i, model_path in enumerate(models_paths):
            if not os.path.exists(model_path):
                self.logger.warning(f"Model file not found: {model_path}, skipping.")
                continue
                
            # Define output directory structure: work_dir/i/task_name
            if task_name:
                job_work_dir = os.path.join(work_dir, str(i), task_name)
            else:
                job_work_dir = os.path.join(work_dir, str(i))
                
            os.makedirs(job_work_dir, exist_ok=True)
            
            # Construct Command
            abs_data_path = os.path.abspath(data_path)
            abs_model_path = os.path.abspath(model_path)
            
            log_file = "test.log" if self.backend == "local" else None
            
            cmd = DPCommandBuilder.test(
                model=abs_model_path,
                system=abs_data_path,
                prefix=results_prefix,
                head=head,
                log_file=log_file
            )
            
            # Append completion marker
            cmd += f"\necho \"{WORKFLOW_FINISHED_TAG}\""
            
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
            
            self.job_manager.generate_script(job_config, script_path)
            
            # Submit Job
            self.job_manager.submit(script_path, working_dir=job_work_dir)
            
        self.logger.info("Inference Workflow Submission Completed.")


class InferenceAnalysisManager:
    """
    Manages Analysis for Inference Workflow:
    - Stats calculation
    - Visualization
    """
    def __init__(self, ref_energies: Optional[Dict[str, float]] = None):
        self.ref_energies = ref_energies
        self.logger = logging.getLogger(__name__)

    def analyze_model(self, model_idx: int, job_work_dir: str, data: Dict, 
                     atom_counts_list: Optional[List], atom_num_list: Optional[List],
                     output_dir_suffix: str = "analysis") -> Tuple[Dict, Optional[Dict]]:
        """
        Analyze results for a single model.
        Returns: (stats_export_dict, metrics_dict_or_None)
        """
        
        analysis_dir = os.path.join(job_work_dir, output_dir_suffix)
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Check frame count consistency
        current_atom_counts = atom_counts_list
        current_atom_nums = atom_num_list
        
        if current_atom_counts is not None:
            if len(current_atom_counts) != len(data["energy"]["pred_e"]):
                self.logger.warning(f"Frame count mismatch: dpdata loaded {len(current_atom_counts)}, "
                                    f"dp test has {len(data['energy']['pred_e'])}. Disabling composition info.")
                current_atom_counts = None
                current_atom_nums = None
        
        # Prepare Data for Calculator
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

        stats_calc = StatsCalculator(
            energy_per_atom=data["energy"]["pred_e"],
            force_flat=f_pred,
            energy_true=data["energy"]["data_e"] if data["has_ground_truth"] else None,
            force_true=f_true,
            atom_counts_list=current_atom_counts,
            atom_num_list=current_atom_nums,
            ref_energies=self.ref_energies
        )
        
        viz = InferenceVisualizer(analysis_dir)
        
        # Metrics & Parity Plots
        metrics = None
        if data["has_ground_truth"]:
            metrics = stats_calc.compute_metrics()
            metrics["model_idx"] = model_idx
            
            viz.plot_parity(stats_calc.e_true, stats_calc.e_pred, "Energy", "eV/atom")
            if stats_calc.f_true is not None:
                viz.plot_parity(stats_calc.f_true, stats_calc.f_pred, "Force", "eV/A")
                
            viz.plot_error_distribution(stats_calc.e_pred - stats_calc.e_true, "Energy", "eV/atom")
            if stats_calc.f_true is not None:
                viz.plot_error_distribution(stats_calc.f_pred - stats_calc.f_true, "Force", "eV/A")

        # Distributions Analysis (No Outliers)
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
        f_pred_norm = None
        f_true_norm = None
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
        
        self.save_statistics(analysis_dir, stats_export)
        
        return stats_export, metrics

    def save_statistics(self, analysis_dir: str, stats_data: Dict):
        """Save statistics to JSON."""
        def default(o):
            if isinstance(o, (np.integer, int)): return int(o)
            if isinstance(o, (np.floating, float)): return float(o)
            if isinstance(o, np.ndarray): return o.tolist()
            return str(o)
            
        with open(os.path.join(analysis_dir, FILENAME_STATS_JSON), "w") as f:
            json.dump(stats_data, f, indent=4, default=default)
