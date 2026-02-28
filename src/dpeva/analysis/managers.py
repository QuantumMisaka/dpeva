import os
import json
import logging
import pandas as pd
import numpy as np
import shutil
from typing import Dict, List, Tuple, Optional, Any

from dpeva.inference import DPTestResultParser, StatsCalculator, InferenceVisualizer
from dpeva.config import AnalysisConfig
from dpeva.constants import FILENAME_STATS_JSON, UNIT_ENERGY, UNIT_ENERGY_PER_ATOM, UNIT_FORCE
from dpeva.io.dataset import load_systems
from collections import Counter

class AnalysisIOManager:
    """
    Manages IO operations for Analysis Workflow:
    - Logging configuration
    - Data loading
    - Result saving
    """
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)

    def load_data(self, result_dir: str, type_map: Optional[List[str]] = None) -> Dict[str, Any]:
        """Parse results using DPTestResultParser."""
        self.logger.info(f"Reading results from {result_dir}")
        parser = DPTestResultParser(result_dir=result_dir, head="results", type_map=type_map)
        data = parser.parse()
        
        return data, parser

    def discover_models(self, work_dir: str, task_name: Optional[str] = None) -> List[str]:
        """
        Discover model directories for ensemble analysis.
        Expects structure: work_dir/{model_idx}/{task_name}/results*
        """
        self.logger.info(f"Discovering model results in {work_dir}")
        model_dirs = []
        
        # Assume directories named "0", "1", "2"... are model directories
        for entry in os.listdir(work_dir):
            if entry.isdigit() and os.path.isdir(os.path.join(work_dir, entry)):
                model_path = os.path.join(work_dir, entry)
                if task_name:
                    model_path = os.path.join(model_path, task_name)
                
                # Verify if results exist
                # DPTestResultParser usually looks for prefix.e_peratom.out or .out
                # We do a loose check here
                if os.path.exists(model_path):
                     model_dirs.append(model_path)
        
        # Sort by model index
        try:
            model_dirs.sort(key=lambda x: int(os.path.basename(os.path.dirname(x) if task_name else x)))
        except:
            model_dirs.sort()
            
        self.logger.info(f"Found {len(model_dirs)} valid model result directories.")
        return model_dirs

    def load_composition_info(self, data_path: str) -> Tuple[Optional[List[Dict]], Optional[List[int]]]:
        """Load composition info using dpdata."""
        if not data_path or not os.path.exists(data_path):
            self.logger.warning("dpdata not available or data_path invalid. Skipping composition loading.")
            return None, None

        try:
            self.logger.info(f"Loading system composition from {data_path} using dpdata...")
            loaded_systems = load_systems(data_path)
                
            atom_counts_list = []
            atom_num_list = []
            
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

    def save_metrics(self, metrics: Dict[str, float], filename: str = "metrics.json"):
        """Save metrics to JSON."""
        with open(os.path.join(self.output_dir, filename), "w") as f:
            json.dump(metrics, f, indent=4)
            
    def save_summary_csv(self, metrics: Dict[str, float], filename: str = "metrics_summary.csv"):
        """Save metrics summary to CSV."""
        pd.DataFrame([metrics]).to_csv(os.path.join(self.output_dir, filename), index=False)
        
    def save_stats_desc(self, stats: Dict, filename: str):
        """Save statistics description to JSON."""
        def default(o):
            """JSON serializer for NumPy types."""
            if isinstance(o, (np.integer, int)): return int(o)
            if isinstance(o, (np.floating, float)): return float(o)
            if isinstance(o, np.ndarray): return o.tolist()
            return str(o)
            
        with open(os.path.join(self.output_dir, filename), "w") as f:
            json.dump(stats, f, indent=4, default=default)


class UnifiedAnalysisManager:
    """
    Unified Manager for Analysis Logic.
    Replaces AnalysisManager and InferenceAnalysisManager.
    Supports both single-model analysis (with or without composition info) and extended stats.
    """
    def __init__(self, ref_energies: Optional[Dict[str, float]] = None):
        self.ref_energies = ref_energies
        self.logger = logging.getLogger(__name__)

    def analyze_model(self, data: Dict, output_dir: str,
                     atom_counts_list: Optional[List] = None, 
                     atom_num_list: Optional[List] = None,
                     model_idx: Optional[int] = None) -> Tuple[Dict, Optional[Dict], StatsCalculator, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Analyze results for a single model.
        
        Args:
            data: Dictionary containing energy/force/virial data (from DPTestResultParser).
            output_dir: Directory to save plots and stats.
            atom_counts_list: List of atom counts dicts (optional, for cohesive energy).
            atom_num_list: List of atom numbers (optional).
            model_idx: Optional model index for labeling.
            
        Returns:
            Tuple containing:
            - stats_export (Dict): Detailed distribution statistics.
            - metrics (Dict): MAE/RMSE metrics.
            - stats_calc (StatsCalculator): Calculator object.
            - e_rel_pred (np.ndarray): Predicted cohesive energy.
            - e_rel_true (np.ndarray): True cohesive energy.
        """
        os.makedirs(output_dir, exist_ok=True)
        
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

        v_pred = None
        v_true = None
        if data["virial"] is not None:
             v_pred = np.column_stack([data["virial"][f"pred_v{i}"] for i in range(9)])
             if data["has_ground_truth"]:
                 v_true = np.column_stack([data["virial"][f"data_v{i}"] for i in range(9)])

        stats_calc = StatsCalculator(
            energy_per_atom=data["energy"]["pred_e"],
            force_flat=f_pred,
            virial_per_atom=v_pred,
            energy_true=data["energy"]["data_e"] if data["has_ground_truth"] else None,
            force_true=f_true,
            virial_true=v_true,
            atom_counts_list=current_atom_counts,
            atom_num_list=current_atom_nums,
            ref_energies=self.ref_energies
        )
        
        viz = InferenceVisualizer(output_dir)
        
        # Metrics & Parity Plots
        metrics = {}
        if data["has_ground_truth"]:
            metrics = stats_calc.compute_metrics()
            if model_idx is not None:
                metrics["model_idx"] = model_idx
            
            # Log metrics
            formatted_metrics = {k: float(f"{v:.6f}") for k, v in metrics.items() if isinstance(v, (int, float))}
            self.logger.info(f"Computed Metrics:\n{json.dumps(formatted_metrics, indent=4)}")

            viz.plot_parity(stats_calc.e_true, stats_calc.e_pred, "Energy", UNIT_ENERGY_PER_ATOM)
            if stats_calc.f_true is not None:
                viz.plot_parity(stats_calc.f_true, stats_calc.f_pred, "Force", UNIT_FORCE)
                
            viz.plot_error_distribution(stats_calc.e_pred - stats_calc.e_true, "Energy", UNIT_ENERGY_PER_ATOM)
            if stats_calc.f_true is not None:
                viz.plot_error_distribution(stats_calc.f_pred - stats_calc.f_true, "Force", UNIT_FORCE)

        # Distributions Analysis (No Outliers)
        # Energy
        viz.plot_distribution(stats_calc.e_pred, "Predicted Energy", UNIT_ENERGY_PER_ATOM)
        if stats_calc.e_true is not None:
            viz.plot_distribution(stats_calc.e_true, "True Energy", UNIT_ENERGY_PER_ATOM, color="green")
        
        # Relative Energy Distribution (Only if composition available)
        e_rel_pred = stats_calc.compute_relative_energy(stats_calc.e_pred)
        e_rel_true = None
        
        if e_rel_pred is not None:
            label_rel = "Cohesive Energy"
            viz.plot_distribution(e_rel_pred, label_rel, UNIT_ENERGY_PER_ATOM, color="purple")
            
            # If we have ground truth, plot parity for Cohesive Energy
            if stats_calc.e_true is not None:
                e_rel_true = stats_calc.compute_relative_energy(stats_calc.e_true)
                if e_rel_true is not None:
                    viz.plot_parity(e_rel_true, e_rel_pred, "Cohesive Energy", UNIT_ENERGY_PER_ATOM)
                    viz.plot_error_distribution(e_rel_pred - e_rel_true, "Cohesive Energy", UNIT_ENERGY_PER_ATOM)
                    
                    if stats_calc.e_true is not None:
                        viz.plot_distribution(e_rel_true, "True Cohesive Energy", UNIT_ENERGY_PER_ATOM, color="magenta")
        
        # Force Magnitude
        f_pred_norm = None
        f_true_norm = None
        if stats_calc.f_pred is not None:
            f_pred_norm = stats_calc.compute_force_magnitude(stats_calc.f_pred)
            viz.plot_distribution(f_pred_norm, "Predicted Force Magnitude", UNIT_FORCE, color="orange")
            
            if stats_calc.f_true is not None:
                f_true_norm = stats_calc.compute_force_magnitude(stats_calc.f_true)
                viz.plot_distribution(f_true_norm, "True Force Magnitude", UNIT_FORCE, color="teal")

        # Virial
        if stats_calc.v_pred is not None:
            # Plot flattened virial components
            viz.plot_distribution(stats_calc.v_pred.flatten(), "Predicted Virial", UNIT_ENERGY, color="red")
            
            if stats_calc.v_true is not None:
                viz.plot_distribution(stats_calc.v_true.flatten(), "True Virial", UNIT_ENERGY, color="cyan")

        # Save Statistics JSON
        stats_export = {
            "energy": stats_calc.get_distribution_stats(stats_calc.e_pred, "energy"),
        }
        if stats_calc.e_true is not None:
            stats_export["energy_true"] = stats_calc.get_distribution_stats(stats_calc.e_true, "energy_true")
        
        if e_rel_pred is not None:
            stats_export["relative_energy"] = stats_calc.get_distribution_stats(e_rel_pred, "relative_energy")
            
        if stats_calc.f_pred is not None:
            stats_export["force_magnitude"] = stats_calc.get_distribution_stats(f_pred_norm, "force_magnitude")
            if stats_calc.f_true is not None:
                stats_export["force_magnitude_true"] = stats_calc.get_distribution_stats(f_true_norm, "force_magnitude_true")

        if stats_calc.v_pred is not None:
            stats_export["virial"] = stats_calc.get_distribution_stats(stats_calc.v_pred.flatten(), "virial")
            if stats_calc.v_true is not None:
                stats_export["virial_true"] = stats_calc.get_distribution_stats(stats_calc.v_true.flatten(), "virial_true")
        
        self.save_statistics(output_dir, stats_export)
        
        return stats_export, metrics, stats_calc, e_rel_pred, e_rel_true

    def save_statistics(self, analysis_dir: str, stats_data: Dict):
        """Save statistics to JSON."""
        def default(o):
            """JSON serializer for NumPy types."""
            if isinstance(o, (np.integer, int)): return int(o)
            if isinstance(o, (np.floating, float)): return float(o)
            if isinstance(o, np.ndarray): return o.tolist()
            return str(o)
            
        with open(os.path.join(analysis_dir, FILENAME_STATS_JSON), "w") as f:
            json.dump(stats_data, f, indent=4, default=default)
