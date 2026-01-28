import os
import sys
import argparse
import json
import logging

# Add src to path if not installed
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from dpeva.workflows.collect import CollectionWorkflow

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="DP-EVA UQ Collection Workflow Runner")
    parser.add_argument("--config", type=str, help="Path to JSON configuration file")
    parser.add_argument("--project", type=str, default=".", help="Project root directory")
    args = parser.parse_args()

    # Default Configuration Template (matches verification test defaults)
    config = {
        "project": args.project,
        "uq_select_scheme": "tangent_lo",
        "testing_dir": "test_val",
        "testing_head": "results",
        "desc_dir": "./desc_pool",
        "testdata_dir": "./other_dpdata_test",
        "testdata_fmt": "deepmd/npy",
        
        "training_data_dir": "sampled_dpdata",
        "fig_dpi": 300,
        "root_savedir": "dpeva_uq_post",
        "uq_trust_mode": "auto",
        "uq_trust_ratio": 0.33,
        "uq_trust_width": 0.25,
        
        # Sampling
        "num_selection": 100,
        "direct_k": 1,
        "direct_thr_init": 0.5
    }

    if args.config:
        if os.path.exists(args.config):
            user_config = load_config(args.config)
            config.update(user_config)
            print(f"Loaded configuration from {args.config}")
        else:
            print(f"Error: Config file {args.config} not found.")
            sys.exit(1)

    # Initialize and run workflow
    try:
        workflow = CollectionWorkflow(config)
        workflow.run()
    except Exception as e:
        logging.error(f"Workflow execution failed: {e}")
        raise

if __name__ == "__main__":
    main()
