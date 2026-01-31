import os
import sys
import argparse
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add src to path if not installed
current_dir = os.path.dirname(os.path.abspath(__file__))
# dpeva/runner/dpeva_collect -> dpeva/runner -> dpeva
project_root = os.path.dirname(os.path.dirname(current_dir))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from dpeva.workflows.collect import CollectionWorkflow

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="DP-EVA UQ Collection Workflow Runner")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON configuration file")
    args = parser.parse_args()

    # Explicit Configuration: Rely solely on user config
    # Removed default config template to ensure explicit behavior
    
    config = {}

    if args.config:
        if os.path.exists(args.config):
            user_config = load_config(args.config)
            config.update(user_config)
            print(f"Loaded configuration from {args.config}")
        else:
            print(f"Error: Config file {args.config} not found.")
            sys.exit(1)
    else:
        print("Error: No configuration file provided. Use --config <path>")
        sys.exit(1)

    # Resolve relative paths in config based on config file location
    config_dir = os.path.dirname(os.path.abspath(args.config))
    
    # List of keys that are paths and need resolution
    path_keys = [
        "project", "desc_dir", "testdata_dir", "training_data_dir", "training_desc_dir", "root_savedir"
    ]
    
    for key in path_keys:
        if key in config and isinstance(config[key], str):
             # If path is not absolute, make it absolute relative to config file
            if not os.path.isabs(config[key]):
                config[key] = os.path.abspath(os.path.join(config_dir, config[key]))

    # Initialize and run workflow
    try:
        workflow = CollectionWorkflow(config)
        workflow.run()
    except Exception as e:
        logging.error(f"Workflow execution failed: {e}")
        raise

if __name__ == "__main__":
    main()
