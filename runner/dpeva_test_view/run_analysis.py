import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import shutil

# Try importing dpeva
try:
    import dpeva
    from dpeva.inference import DPTestResultParser, StatsCalculator, InferenceVisualizer
except ImportError:
    print("Error: The 'dpeva' package is not installed in the current Python environment.")
    print("Please install it using: pip install -e .")
    sys.exit(1)

def setup_logger(output_dir):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "analysis.log")),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("DPEVATestView")

def main():
    # Load config
    config_path = os.path.abspath(os.path.join(current_dir, "config.json"))
    
    # Allow overriding config via command line arg if provided
    if len(sys.argv) > 1:
        config_path = os.path.abspath(sys.argv[1])

    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Resolve paths relative to config file location
    config_dir = os.path.dirname(config_path)
    
    # Helper to resolve paths
    def resolve(path, default):
        if not path:
            path = default
        if os.path.isabs(path):
            return path
        return os.path.abspath(os.path.join(config_dir, path))

    result_dir = resolve(config.get("result_dir"), ".")
    output_dir = resolve(config.get("output_dir"), "analysis")
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    logger = setup_logger(output_dir)
    logger.info(f"Starting Analysis in {output_dir}")
    logger.info(f"Reading results from {result_dir}")

    try:
        # 1. Parse Results
        type_map = config.get("type_map", ["Fe", "C", "O", "H"])
        parser = DPTestResultParser(result_dir=result_dir, head="results", type_map=type_map)
        data = parser.parse()
        
        # 2. Get Composition Info
        logger.info("Extracting composition info...")
        atom_counts_list, atom_num_list = parser.get_composition_list()
        
        # 3. Prepare Stats Calculator
        logger.info("Initializing StatsCalculator...")
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

        ref_energies = config.get("ref_energies")
        
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
            logger.info(f"Computed Metrics:\n{json.dumps(formatted_metrics, indent=4)}")
            summary_metrics = metrics

        # Cohesive Energy
        e_rel_pred = stats_calc.compute_relative_energy(stats_calc.e_pred)
        e_rel_true = None
        if stats_calc.e_true is not None:
            e_rel_true = stats_calc.compute_relative_energy(stats_calc.e_true)

        # 5. Visualization
        logger.info("Generating visualizations...")
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

        logger.info("Analysis completed successfully.")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()
