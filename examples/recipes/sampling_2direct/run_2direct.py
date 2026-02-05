"""
2-DIRECT Sampling Recipe
========================

This recipe demonstrates how to use the 2-Step DIRECT sampling strategy.
It performs:
1. Structural clustering (Step 1) to group similar configurations.
2. Atomic clustering (Step 2) within each structural cluster to find representative atomic environments.
3. Selection based on atomic properties (e.g., 'smallest' number of atoms in cluster).

Usage:
    python run_2direct.py config.json
"""

import sys
import json
import os
from dpeva.workflows.collect import CollectionWorkflow

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_2direct.py <config.json>")
        sys.exit(1)
        
    config_path = sys.argv[1]
    
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    print(f"Initializing Collection Workflow with config: {config_path}")
    wf = CollectionWorkflow(config, config_path=os.path.abspath(config_path))
    wf.run()
    print("2-DIRECT Sampling Workflow Completed.")

if __name__ == "__main__":
    main()