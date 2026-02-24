import os
import json
import logging
import pandas as pd
import numpy as np
import shutil
from typing import Dict, List, Tuple, Optional, Any

from dpeva.inference import DPTestResultParser, StatsCalculator, InferenceVisualizer
from dpeva.config import AnalysisConfig
from dpeva.constants import FILENAME_STATS_JSON

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

    def configure_logging(self):
        """Configure file logging for the analysis run."""
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        
        fh = logging.FileHandler(os.path.join(self.output_dir, "analysis.log"))
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        
        # Attach to root logger or specific logger
        # Here we attach to the module logger's parent to capture everything if needed, 
        # or just the workflow logger. 
        # The original code attached to self.logger (AnalysisWorkflow logger).
        # We'll attach to the root 'dpeva' logger or similar to capture all events.
        logging.getLogger("dpeva").addHandler(fh)
        self.fh = fh # Keep reference to remove later

    def close_logging(self):
        """Remove file handler."""
        if hasattr(self, 'fh'):
            logging.getLogger("dpeva").removeHandler(self.fh)

    def load_data(self, result_dir: str, type_map: Optional[List[str]] = None) -> Dict[str, Any]:
        """Parse results using DPTestResultParser."""
        self.logger.info(f"Reading results from {result_dir}")
        parser = DPTestResultParser(result_dir=result_dir, head="results", type_map=type_map)
        data = parser.parse()
        
        # Also get composition info from parser if available
        # Note: DPTestResultParser.get_composition_list() is a method we need to check if it exists or if we need to call it separately.
        # In the original code:
        # atom_counts_list, atom_num_list = parser.get_composition_list()
        
        return data, parser

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
            if isinstance(o, (np.integer, int)): return int(o)
            if isinstance(o, (np.floating, float)): return float(o)
            if isinstance(o, np.ndarray): return o.tolist()
            return str(o)
            
        with open(os.path.join(self.output_dir, filename), "w") as f:
            json.dump(stats, f, indent=4, default=default)


class AnalysisManager:
    """
    Manages the core analysis logic:
    - Stats calculation
    - Visualization
    """
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def compute_metrics(self, data: Dict, parser: DPTestResultParser) -> Tuple[Dict, StatsCalculator, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Compute statistics and metrics.
        Returns: (metrics, stats_calculator, e_rel_pred, e_rel_true)
        """
        self.logger.info("Extracting composition info...")
        atom_counts_list, atom_num_list = parser.get_composition_list()
        
        self.logger.info("Initializing StatsCalculator...")
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

        ref_energies = self.config.ref_energies
        
        stats_calc = StatsCalculator(
            energy_per_atom=data["energy"]["pred_e"],
            force_flat=f_pred,
            virial_per_atom=v_pred,
            energy_true=data["energy"]["data_e"] if data["has_ground_truth"] else None,
            force_true=f_true,
            virial_true=v_true,
            atom_counts_list=atom_counts_list,
            atom_num_list=atom_num_list,
            ref_energies=ref_energies
        )

        metrics = {}
        if data["has_ground_truth"]:
            metrics = stats_calc.compute_metrics()
            # Format metrics for nice display
            formatted_metrics = {k: float(f"{v:.6f}") for k, v in metrics.items()}
            self.logger.info(f"Computed Metrics:\n{json.dumps(formatted_metrics, indent=4)}")

        # Cohesive Energy
        e_rel_pred = stats_calc.compute_relative_energy(stats_calc.e_pred)
        e_rel_true = None
        if stats_calc.e_true is not None:
            e_rel_true = stats_calc.compute_relative_energy(stats_calc.e_true)
            
        return metrics, stats_calc, e_rel_pred, e_rel_true

    def visualize(self, stats_calc: StatsCalculator, e_rel_pred, e_rel_true, output_dir: str):
        """Generate visualizations."""
        self.logger.info("Generating visualizations...")
        viz = InferenceVisualizer(output_dir)

        # Energy
        viz.plot_distribution(stats_calc.e_pred, "Predicted Energy", "eV/atom")
        if stats_calc.e_true is not None:
            viz.plot_parity(stats_calc.e_true, stats_calc.e_pred, "Energy", "eV/atom")
            viz.plot_error_distribution(stats_calc.e_pred - stats_calc.e_true, "Energy Error", "eV/atom")

        # Cohesive Energy
        if e_rel_pred is not None:
            viz.plot_distribution(e_rel_pred, "Predicted Cohesive Energy", "eV/atom", color="purple")
            
        if e_rel_true is not None:
            viz.plot_parity(e_rel_true, e_rel_pred, "Cohesive Energy", "eV/atom")
            viz.plot_error_distribution(e_rel_pred - e_rel_true, "Cohesive Energy Error", "eV/atom")

        # Force
        if stats_calc.f_pred is not None:
            f_pred_norm = stats_calc.compute_force_magnitude(stats_calc.f_pred)
            viz.plot_distribution(f_pred_norm, "Predicted Force Magnitude", "eV/A", color="orange")

            if stats_calc.f_true is not None:
                viz.plot_parity(stats_calc.f_true, stats_calc.f_pred, "Force", "eV/A")
                viz.plot_error_distribution(stats_calc.f_pred - stats_calc.f_true, "Force Error", "eV/A")

        # Virial
        if stats_calc.v_pred is not None:
            viz.plot_distribution(stats_calc.v_pred.flatten(), "Predicted Virial", "eV", color="red")
            
            if stats_calc.v_true is not None:
                viz.plot_parity(stats_calc.v_true.flatten(), stats_calc.v_pred.flatten(), "Virial", "eV")
                viz.plot_error_distribution(stats_calc.v_pred.flatten() - stats_calc.v_true.flatten(), "Virial Error", "eV")
