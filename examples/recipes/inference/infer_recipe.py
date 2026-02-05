"""
DP-EVA Inference Workflow Recipe

This script demonstrates how to programmatically invoke the Inference Workflow.
Usage:
    Adjust the 'config_path' variable below and run:
    python infer_recipe.py
"""

import json
import logging
import os
import sys

try:
    import dpeva
except ImportError:
    print("Please install dpeva first: pip install -e .")
    sys.exit(1)

from dpeva.utils.config import resolve_config_paths
from dpeva.workflows.infer import InferenceWorkflow

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def main():
    # 1. Configuration
    config_path = "config_infer.json" # <--- Change this
    
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return

    # 2. Load Config
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    config_dict = resolve_config_paths(config_dict, config_path)
    
    # 3. Initialize & Run
    workflow = InferenceWorkflow(config_dict)
    workflow.run()

if __name__ == "__main__":
    main()
