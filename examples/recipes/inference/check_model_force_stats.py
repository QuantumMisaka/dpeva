
import numpy as np
import os
import pandas as pd
import argparse

def get_force_stats(data):
    """Calculate statistics for a force array."""
    desc = pd.Series(data).describe(percentiles=[0.25, 0.5, 0.75])
    return desc

def check_model_stats(project=".", testing_dir="test_val", testing_head="results", num_models=4):
    print(f"Checking force statistics for {num_models} models...")
    print(f"Project: {project}, Dir: {testing_dir}, Head: {testing_head}\n")

    stats_list = []
    stats_norm_list = []
    
    for i in range(num_models):
        fname = f"./{project}/{i}/{testing_dir}/{testing_head}.f.out"
        if not os.path.exists(fname):
            print(f"Warning: File not found: {fname}")
            continue
        
        try:
            # Load data
            # Format: data_fx data_fy data_fz pred_fx pred_fy pred_fz
            data = np.genfromtxt(fname, names=["data_fx", "data_fy", "data_fz", "pred_fx", "pred_fy", "pred_fz"])
            
            # Extract predicted forces (flattened components)
            pred_f_comps = np.column_stack((data['pred_fx'], data['pred_fy'], data['pred_fz'])).flatten()
            
            # Calculate stats for components
            stats = get_force_stats(pred_f_comps)
            stats_dict = {
                "Model": i,
                "Count": int(stats["count"]),
                "Mean": stats["mean"],
                "Std": stats["std"],
                "Min": stats["min"],
                "25%": stats["25%"],
                "50%": stats["50%"],
                "75%": stats["75%"],
                "Max": stats["max"]
            }
            stats_list.append(stats_dict)

            # Extract predicted forces (magnitudes/norms)
            # F_norm = sqrt(fx^2 + fy^2 + fz^2)
            pred_f_vecs = np.column_stack((data['pred_fx'], data['pred_fy'], data['pred_fz']))
            pred_f_norms = np.linalg.norm(pred_f_vecs, axis=1)

            # Calculate stats for magnitudes
            stats_norm = get_force_stats(pred_f_norms)
            stats_norm_dict = {
                "Model": i,
                "Count": int(stats_norm["count"]),
                "Mean": stats_norm["mean"],
                "Std": stats_norm["std"],
                "Min": stats_norm["min"],
                "25%": stats_norm["25%"],
                "50%": stats_norm["50%"],
                "75%": stats_norm["75%"],
                "Max": stats_norm["max"]
            }
            stats_norm_list.append(stats_norm_dict)
            
        except Exception as e:
            print(f"Error analyzing model {i}: {e}")

    if stats_list:
        df_stats = pd.DataFrame(stats_list)
        df_stats_norm = pd.DataFrame(stats_norm_list)
        
        # Format for display
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.float_format', '{:.4f}'.format)
        
        print("Atomic Force Component Statistics (Predicted):")
        print(df_stats.to_string(index=False))
        print("\n" + "="*50 + "\n")
        print("Atomic Force Magnitude (Absolute Value) Statistics (Predicted):")
        print(df_stats_norm.to_string(index=False))
        
        # Check consistency
        print("\nConsistency Check (Components):")
        means = df_stats["Mean"]
        mean_diff = means.max() - means.min()
        print(f"Max difference in Mean (Components): {mean_diff:.4f}")
        
        if mean_diff > 1.0: 
             print("WARNING: Significant difference in mean force components detected!")
        else:
             print("Mean force components appear consistent across models.")

        print("\nConsistency Check (Magnitudes):")
        means_norm = df_stats_norm["Mean"]
        mean_diff_norm = means_norm.max() - means_norm.min()
        print(f"Max difference in Mean (Magnitudes): {mean_diff_norm:.4f}")

        if mean_diff_norm > 1.0:
             print("WARNING: Significant difference in mean force magnitudes detected!")
        else:
             print("Mean force magnitudes appear consistent across models.")
             
    else:
        print("No data found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check force statistics for DP models.")
    parser.add_argument("--project", type=str, default=".", help="Project directory")
    parser.add_argument("--dir", type=str, default="test_val", help="Testing directory name")
    parser.add_argument("--head", type=str, default="results", help="Results file prefix")
    parser.add_argument("--models", type=int, default=4, help="Number of models")
    
    args = parser.parse_args()
    
    check_model_stats(args.project, args.dir, args.head, args.models)
