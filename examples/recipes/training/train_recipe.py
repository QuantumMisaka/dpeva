"""
DP-EVA Training Workflow Recipe

This script demonstrates how to programmatically invoke the Training Workflow.
Usage:
    Adjust the 'config_path' variable below and run:
    python train_recipe.py
"""

import json
import logging
import os
import sys

# Ensure dpeva is importable
try:
    import dpeva
except ImportError:
    print("Please install dpeva first: pip install -e .")
    sys.exit(1)

from dpeva.utils.config import resolve_config_paths
from dpeva.workflows.train import TrainingWorkflow

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def main():
    # 1. Configuration
    config_path = "config_train.json"  # <--- Change this to your config file
    
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        print("Please create a config file or update the 'config_path' variable in this script.")
        return

    # 2. Load and Resolve Config
    print(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Resolve relative paths in config based on config file location
    config_dict = resolve_config_paths(config_dict, config_path)
    
    # 3. Initialize Workflow
    workflow = TrainingWorkflow(config_dict)
    
    # 4. Run
    workflow.run()

if __name__ == "__main__":
    main()
