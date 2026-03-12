"""
DP-EVA Training Workflow Recipe

Usage:
    python train_recipe.py [config_path]
"""

import json
import logging
import sys
from pathlib import Path

try:
    import dpeva
except ImportError:
    print("Please install dpeva first: pip install -e .")
    sys.exit(1)

from dpeva.utils.config import resolve_config_paths
from dpeva.workflows.train import TrainingWorkflow

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def main():
    default_config = Path(__file__).resolve().parents[2] / "recipes" / "training" / "config_train.json"
    config_path = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else default_config

    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        return

    print(f"Loading configuration from {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    config_dict = resolve_config_paths(config_dict, str(config_path))

    workflow = TrainingWorkflow(config_dict)
    workflow.run()


if __name__ == "__main__":
    main()
