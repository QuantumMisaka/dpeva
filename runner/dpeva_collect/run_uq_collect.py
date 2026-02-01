import os
import sys
import argparse
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("dpeva.runner.collect")

# Try importing dpeva
try:
    from dpeva.workflows.collect import CollectionWorkflow
except ImportError:
    logger.error("The 'dpeva' package is not installed in the current Python environment.")
    logger.error("Please install it using: pip install -e .")
    sys.exit(1)

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="DP-EVA UQ Collection Workflow Runner")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON configuration file")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        logger.error(f"Config file {args.config} not found.")
        sys.exit(1)

    try:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON config: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error loading config: {e}")
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
        logger.info("✅ Collection Workflow Completed.")
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
