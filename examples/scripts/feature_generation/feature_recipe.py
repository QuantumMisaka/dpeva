"""
DP-EVA Feature Generation Workflow Recipe

This script demonstrates how to programmatically invoke the Feature Generation Workflow.
Usage:
    python feature_recipe.py [config_path]
"""

import json
import importlib.util
import logging
import sys
from pathlib import Path

if importlib.util.find_spec("dpeva") is None:
    print("Please install dpeva first: pip install -e .")
    sys.exit(1)

from dpeva.utils.config import resolve_config_paths
from dpeva.workflows.feature import FeatureWorkflow

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def main():
    default_config = Path(__file__).resolve().parents[2] / "recipes" / "feature_generation" / "config_feature.json"
    config_path = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else default_config

    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    config_dict = resolve_config_paths(config_dict, str(config_path))

    workflow = FeatureWorkflow(config_dict)
    workflow.run()


if __name__ == "__main__":
    main()
