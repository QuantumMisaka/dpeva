import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import shutil
from pathlib import Path
from typing import Optional, Dict, List, Any, Union

from dpeva.inference import DPTestResultParser, StatsCalculator, InferenceVisualizer
from dpeva.config import AnalysisConfig

class AnalysisWorkflow:
    """
    Workflow for analyzing DeepMD test results.
    """
    
    def __init__(self, config: Union[AnalysisConfig, Dict[str, Any]]):
        """
        Initialize AnalysisWorkflow.
        
        Args:
            config: AnalysisConfig object or dictionary.
        """
        if isinstance(config, dict):
            self.config = AnalysisConfig(**config)
        else:
            self.config = config
            
        self.logger = logging.getLogger(__name__)
        
    def run(self):
        """Execute the analysis workflow."""
        result_dir = str(self.config.result_dir)
        output_dir = str(self.config.output_dir)
        
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup file logging for this run
        fh = logging.FileHandler(os.path.join(output_dir, "analysis.log"))
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(fh)
        
        self.logger.info(f"Starting Analysis in {output_dir}")
        self.logger.info(f"Reading results from {result_dir}")
        
        try:
            self._run_analysis(result_dir, output_dir)
            self.logger.info("Analysis completed successfully.")
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}", exc_info=True)
            raise e
        finally:
            self.logger.removeHandler(fh)

    def _run_analysis(self, result_dir: str, output_dir: str):
        # 1. Parse Results
        type_map = self.config.type_map
        parser = DPTestResultParser(result_dir=result_dir, head="results", type_map=type_map)
        data = parser.parse()
        
        # 2. Get Composition Info
        self.logger.info("Extracting composition info...")
        atom_counts_list, atom_num_list = parser.get_composition_list()
        
        # 3. Prepare Stats Calculator
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

        # 4. Compute Metrics
        summary_metrics = {}
        if data["has_ground_truth"]:
            metrics = stats_calc.compute_metrics()
            # Format metrics for nice display
            formatted_metrics = {k: float(f"{v:.6f}") for k, v in metrics.items()}
            self.logger.info(f"Computed Metrics:\n{json.dumps(formatted_metrics, indent=4)}")
            summary_metrics = metrics

        # Cohesive Energy
        e_rel_pred = stats_calc.compute_relative_energy(stats_calc.e_pred)
        e_rel_true = None
        if stats_calc.e_true is not None:
            e_rel_true = stats_calc.compute_relative_energy(stats_calc.e_true)

        # 5. Visualization
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
            
            # Save Stats
            stats_desc = pd.Series(e_rel_pred).describe().to_dict()
            with open(os.path.join(output_dir, "cohesive_energy_pred_stats.json"), "w") as f:
                json.dump(stats_desc, f, indent=4)
                
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
                
        # Save Summary
        if summary_metrics:
            pd.DataFrame([summary_metrics]).to_csv(os.path.join(output_dir, "metrics_summary.csv"), index=False)
            with open(os.path.join(output_dir, "metrics.json"), "w") as f:
                json.dump(summary_metrics, f, indent=4)
