"""
DP-EVA Collection (Active Learning) Workflow Recipe

This script demonstrates how to programmatically invoke the Collection Workflow.
Usage:
    python collect_recipe.py [config_path]
"""

import argparse
import json
import importlib.util
import logging
import os
import sys
from pathlib import Path

if importlib.util.find_spec("dpeva") is None:
    print("Please install dpeva first: pip install -e .")
    sys.exit(1)

from dpeva.utils.config import resolve_config_paths
from dpeva.workflows.collect import CollectionWorkflow

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def main():
    default_config = Path(__file__).resolve().parents[2] / "recipes" / "collection" / "config_collect_normal.json"
    parser = argparse.ArgumentParser(description="Run DP-EVA Collection Workflow")
    parser.add_argument(
        "config",
        nargs="?",
        default=str(default_config),
        help=f"Path to the configuration JSON file (default: {default_config})",
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser()
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        recipe_dir = Path(__file__).resolve().parents[2] / "recipes" / "collection"
        print("Available configs:")
        for f in sorted(recipe_dir.glob("config*.json")):
            print(f"  - {f}")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    config_dict = resolve_config_paths(config_dict, str(config_path))

    workflow = CollectionWorkflow(config_dict, config_path=os.path.abspath(str(config_path)))
    workflow.run()


if __name__ == "__main__":
    main()
