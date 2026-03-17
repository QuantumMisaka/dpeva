"""
DP-EVA Analysis Workflow Recipe

This script demonstrates how to programmatically invoke the Analysis Workflow.
Usage:
    python analysis_recipe.py [config_path]
"""

import json
import logging
import os
import sys
from pathlib import Path

try:
    import dpeva
except ImportError:
    print("Please install dpeva first: pip install -e .")
    sys.exit(1)

from dpeva.utils.config import resolve_config_paths
from dpeva.workflows.analysis import AnalysisWorkflow

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def main():
    default_config = Path(__file__).resolve().parents[2] / "recipes" / "analysis" / "config_analysis.json"
    config_path = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else default_config

    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    config_dict = resolve_config_paths(config_dict, str(config_path))

    workflow = AnalysisWorkflow(config_dict)
    workflow.run()


if __name__ == "__main__":
    main()
