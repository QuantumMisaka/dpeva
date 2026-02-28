"""
Standard DIRECT Sampling Recipe
===============================

This recipe demonstrates how to use the standard DIRECT sampling strategy.
It loads descriptors, performs (optional) UQ filtering, and then samples 
representative structures using structural clustering.

Usage:
    python run_direct.py config.json
"""

import sys
import json
import os
from dpeva.workflows.collect import CollectionWorkflow

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_direct.py <config.json>")
        sys.exit(1)
        
    config_path = sys.argv[1]
    
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    # Ensure paths are absolute or relative to execution dir
    # Here we assume user runs from examples/recipes/sampling_direct or provides correct relative paths
    
    print(f"Initializing Collection Workflow with config: {config_path}")
    wf = CollectionWorkflow(config, config_path=os.path.abspath(config_path))
    wf.run()
    print("DIRECT Sampling Workflow Completed.")

if __name__ == "__main__":
    main()