"""
2-DIRECT Sampling Recipe
========================

Usage:
    python run_2direct.py [config_path]
"""

import json
import os
import sys
from pathlib import Path

from dpeva.workflows.collect import CollectionWorkflow


def main():
    default_config = Path(__file__).resolve().parents[2] / "recipes" / "sampling_2direct" / "config.json"
    config_path = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else default_config

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    print(f"Initializing Collection Workflow with config: {config_path}")
    wf = CollectionWorkflow(config, config_path=os.path.abspath(str(config_path)))
    wf.run()
    print("2-DIRECT Sampling Workflow Completed.")


if __name__ == "__main__":
    main()
