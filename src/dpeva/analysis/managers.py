import os
import json
import logging
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from dpeva.inference import DPTestResultParser
from dpeva.postprocess import StatsCalculator, InferenceVisualizer
from dpeva.constants import (
    FILENAME_METRICS_JSON,
    FILENAME_METRICS_SUMMARY_CSV,
    FILENAME_STATS_JSON,
    UNIT_ENERGY,
    UNIT_ENERGY_PER_ATOM,
    UNIT_FORCE,
)
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

    def load_data(
        self,
        result_dir: str,
        type_map: Optional[List[str]] = None,
        results_prefix: str = "results",
    ) -> Tuple[Dict[str, Any], DPTestResultParser]:
        """Parse results and return parsed data with parser instance."""
        self.logger.info(f"Reading results from {result_dir}")
        parser = DPTestResultParser(result_dir=result_dir, head=results_prefix, type_map=type_map)
        data = parser.parse()
        
        return data, parser

    def discover_models(self, work_dir: str, task_name: Optional[str] = None) -> List[str]:
        """
        Discover model directories for ensemble analysis.
        Expects structure: work_dir/{model_idx}/{task_name}/results*
        """
        self.logger.info(f"Discovering model results in {work_dir}")
        model_dirs = []

        for entry in os.listdir(work_dir):
            if entry.isdigit() and os.path.isdir(os.path.join(work_dir, entry)):
                model_path = os.path.join(work_dir, entry)
                if task_name:
                    model_path = os.path.join(model_path, task_name)

                if os.path.exists(model_path):
                     model_dirs.append(model_path)

        try:
            model_dirs.sort(key=lambda x: int(os.path.basename(os.path.dirname(x) if task_name else x)))
        except (ValueError, TypeError):
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

                for _ in range(s.get_nframes()):
                    atom_counts_list.append(counts)
                    atom_num_list.append(n_atoms)
                    
            self.logger.info(f"Loaded composition info for {len(atom_counts_list)} frames.")
            return atom_counts_list, atom_num_list
        except Exception as e:
            self.logger.warning(f"Failed to load composition info: {e}. Relative energy will use mean subtraction.")
            return None, None

    def save_metrics(self, metrics: Dict[str, float], filename: str = FILENAME_METRICS_JSON):
        """Save metrics to JSON."""
        with open(os.path.join(self.output_dir, filename), "w") as f:
            json.dump(metrics, f, indent=4)
            
    def save_summary_csv(self, metrics: Dict[str, float], filename: str = FILENAME_METRICS_SUMMARY_CSV):
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
    def __init__(
        self,
        ref_energies: Optional[Dict[str, float]] = None,
        enable_cohesive_energy: bool = True,
        allow_ref_energy_lstsq_completion: bool = False,
        slow_plot_threshold_seconds: float = 60.0,
    ):
        self.ref_energies = ref_energies
        self.enable_cohesive_energy = enable_cohesive_energy
        self.allow_ref_energy_lstsq_completion = allow_ref_energy_lstsq_completion
        self.slow_plot_threshold_seconds = slow_plot_threshold_seconds
        self.logger = logging.getLogger(__name__)

    def analyze_model(self, data: Dict, output_dir: str,
                     atom_counts_list: Optional[List] = None, 
                     atom_num_list: Optional[List] = None,
                     model_idx: Optional[int] = None,
                     plot_level: str = "full") -> Tuple[Dict, Optional[Dict], StatsCalculator, Optional[np.ndarray], Optional[np.ndarray]]:
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
            ref_energies=self.ref_energies,
            enable_cohesive_energy=self.enable_cohesive_energy,
            allow_ref_energy_lstsq_completion=self.allow_ref_energy_lstsq_completion
        )
        
        viz = InferenceVisualizer(output_dir)
        full_plot_enabled = plot_level == "full"

        def safe_plot(plot_name: str, fn, *args, **kwargs):
            """Run plotting call safely and downgrade plotting errors to warnings."""
            start = time.perf_counter()
            try:
                fn(*args, **kwargs)
            except Exception as e:
                self.logger.warning(f"Plot '{plot_name}' failed and was skipped: {e}")
            finally:
                elapsed = time.perf_counter() - start
                if elapsed > self.slow_plot_threshold_seconds:
                    self.logger.warning(
                        f"Plot '{plot_name}' is slow: elapsed={elapsed:.3f}s, "
                        f"threshold={self.slow_plot_threshold_seconds:.3f}s. "
                        "Consider setting analysis plot_level='basic' to reduce plotting overhead."
                    )

        def plot_parity_family(quantity: str, unit: str, y_true: np.ndarray, y_pred: np.ndarray):
            """Generate standard and enhanced parity plots for one quantity."""
            safe_plot(f"parity_{quantity}", viz.plot_parity, y_true, y_pred, quantity, unit)
            if full_plot_enabled:
                safe_plot(f"parity_{quantity}_enhanced", viz.plot_parity_enhanced, y_true, y_pred, quantity, unit)

        def plot_distribution_family(
            quantity: str,
            unit: str,
            pred_data: np.ndarray,
            true_data: Optional[np.ndarray],
            pred_single_label: str,
            true_single_label: str,
            pred_overlay_label: str,
            true_overlay_label: str,
            pred_color: str,
            true_color: str,
            error_data: Optional[np.ndarray] = None,
            overlay_show_stats: bool = True,
            with_error_show_stats: bool = False,
        ):
            """Generate single, overlay and with-error distribution plots for one quantity."""
            safe_plot(f"dist_{pred_single_label.lower().replace(' ', '_')}", viz.plot_distribution, pred_data, pred_single_label, unit, color=pred_color)
            if true_data is not None:
                safe_plot(f"dist_{true_single_label.lower().replace(' ', '_')}", viz.plot_distribution, true_data, true_single_label, unit, color=true_color)
                if full_plot_enabled:
                    safe_plot(
                        f"dist_{quantity.lower().replace(' ', '_')}_overlay",
                        viz.plot_distribution_overlay,
                        pred_data,
                        true_data,
                        quantity,
                        unit,
                        pred_label=pred_overlay_label,
                        true_label=true_overlay_label,
                        pred_color=pred_color,
                        true_color=true_color,
                        show_stats=overlay_show_stats,
                    )
                    if error_data is not None:
                        safe_plot(
                            f"dist_{quantity.lower().replace(' ', '_')}_with_error",
                            viz.plot_distribution_with_error,
                            pred_data,
                            true_data,
                            error_data,
                            quantity,
                            unit,
                            pred_label=pred_overlay_label,
                            true_label=true_overlay_label,
                            pred_color=pred_color,
                            true_color=true_color,
                            show_stats=with_error_show_stats,
                        )
        
        metrics = {}
        if data["has_ground_truth"]:
            metrics = stats_calc.compute_metrics()
            if model_idx is not None:
                metrics["model_idx"] = model_idx

            formatted_metrics = {k: float(f"{v:.6f}") for k, v in metrics.items() if isinstance(v, (int, float))}
            self.logger.info(f"Computed Metrics:\n{json.dumps(formatted_metrics, indent=4)}")

            plot_parity_family("Energy", UNIT_ENERGY_PER_ATOM, stats_calc.e_true, stats_calc.e_pred)
            if stats_calc.f_true is not None:
                plot_parity_family("Force", UNIT_FORCE, stats_calc.f_true, stats_calc.f_pred)
            if stats_calc.v_true is not None:
                plot_parity_family("Virial", UNIT_ENERGY, stats_calc.v_true.flatten(), stats_calc.v_pred.flatten())
                
            safe_plot("error_dist_energy", viz.plot_error_distribution, stats_calc.e_pred - stats_calc.e_true, "Energy", UNIT_ENERGY_PER_ATOM)
            if stats_calc.f_true is not None:
                safe_plot("error_dist_force", viz.plot_error_distribution, stats_calc.f_pred - stats_calc.f_true, "Force", UNIT_FORCE)
            if stats_calc.v_true is not None:
                safe_plot("error_dist_virial", viz.plot_error_distribution, stats_calc.v_pred.flatten() - stats_calc.v_true.flatten(), "Virial", UNIT_ENERGY)

        plot_distribution_family(
            quantity="Energy",
            unit=UNIT_ENERGY_PER_ATOM,
            pred_data=stats_calc.e_pred,
            true_data=stats_calc.e_true,
            pred_single_label="Predicted Energy",
            true_single_label="True Energy",
            pred_overlay_label="Predicted",
            true_overlay_label="True",
            pred_color="#2563eb",
            true_color="#ef4444",
            error_data=(stats_calc.e_pred - stats_calc.e_true) if stats_calc.e_true is not None else None,
        )
        
        e_rel_pred = stats_calc.compute_relative_energy(stats_calc.e_pred)
        e_rel_true = None
        
        if e_rel_pred is not None:
            safe_plot(
                "dist_predicted_cohesive_energy",
                viz.plot_distribution,
                e_rel_pred,
                "Predicted Cohesive Energy",
                UNIT_ENERGY_PER_ATOM,
                color="#7e22ce",
            )
            
            if stats_calc.e_true is not None:
                e_rel_true = stats_calc.compute_relative_energy(stats_calc.e_true)
                if e_rel_true is not None:
                    plot_parity_family("Cohesive Energy", UNIT_ENERGY_PER_ATOM, e_rel_true, e_rel_pred)
                    safe_plot("error_dist_cohesive_energy", viz.plot_error_distribution, e_rel_pred - e_rel_true, "Cohesive Energy", UNIT_ENERGY_PER_ATOM)
                    plot_distribution_family(
                        quantity="Cohesive Energy",
                        unit=UNIT_ENERGY_PER_ATOM,
                        pred_data=e_rel_pred,
                        true_data=e_rel_true,
                        pred_single_label="Predicted Cohesive Energy",
                        true_single_label="True Cohesive Energy",
                        pred_overlay_label="Predicted",
                        true_overlay_label="True",
                        pred_color="#7e22ce",
                        true_color="#ec4899",
                        error_data=e_rel_pred - e_rel_true,
                    )
        
        f_pred_norm = None
        f_true_norm = None
        if stats_calc.f_pred is not None:
            f_pred_norm = stats_calc.compute_force_magnitude(stats_calc.f_pred)
            safe_plot(
                "dist_predicted_force_magnitude",
                viz.plot_distribution,
                f_pred_norm,
                "Predicted Force Magnitude",
                UNIT_FORCE,
                color="#f59e0b",
            )
            
            if stats_calc.f_true is not None:
                f_true_norm = stats_calc.compute_force_magnitude(stats_calc.f_true)
                plot_distribution_family(
                    quantity="Force Magnitude",
                    unit=UNIT_FORCE,
                    pred_data=f_pred_norm,
                    true_data=f_true_norm,
                    pred_single_label="Predicted Force Magnitude",
                    true_single_label="True Force Magnitude",
                    pred_overlay_label="Predicted",
                    true_overlay_label="True",
                    pred_color="#f59e0b",
                    true_color="#0f766e",
                    error_data=f_pred_norm - f_true_norm,
                )

        if stats_calc.v_pred is not None:
            v_pred_flat = stats_calc.v_pred.flatten()
            v_true_flat = stats_calc.v_true.flatten() if stats_calc.v_true is not None else None
            if stats_calc.v_true is not None:
                plot_distribution_family(
                    quantity="Virial",
                    unit=UNIT_ENERGY,
                    pred_data=v_pred_flat,
                    true_data=v_true_flat,
                    pred_single_label="Predicted Virial",
                    true_single_label="True Virial",
                    pred_overlay_label="Predicted",
                    true_overlay_label="True",
                    pred_color="#dc2626",
                    true_color="#0891b2",
                    error_data=v_pred_flat - v_true_flat,
                )
            else:
                safe_plot("dist_predicted_virial", viz.plot_distribution, v_pred_flat, "Predicted Virial", UNIT_ENERGY, color="#dc2626")

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
