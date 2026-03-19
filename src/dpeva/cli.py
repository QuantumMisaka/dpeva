"""
DP-EVA Command Line Interface (CLI).

This module serves as the main entry point for the DP-EVA application.
It parses command-line arguments and dispatches control to the appropriate
workflow handlers (Training, Inference, Collection, Feature Generation, Analysis).

Usage:
    dpeva <command> <config_path> [options]
"""
import argparse
import sys
import json
import logging
import os
from dpeva.utils.config import resolve_config_paths
from dpeva.utils.banner import show_banner

# Lazy imports for workflows to improve CLI startup time
# Workflows are imported inside handler functions

LABEL_STAGE_TOKENS = {"prepare", "execute", "extract", "postprocess"}


class CLIUserInputError(ValueError):
    pass


def validate_config_path(config_path: str) -> str:
    normalized = os.path.abspath(os.path.expanduser(config_path))
    token = config_path.strip().lower()
    if token in LABEL_STAGE_TOKENS:
        raise argparse.ArgumentTypeError(
            f"Config file not found: {config_path}. "
            f"If you want labeling stage control, use '--stage {token}'."
        )
    if not os.path.exists(normalized):
        raise argparse.ArgumentTypeError(f"Config file not found: {config_path}")
    if not os.path.isfile(normalized):
        raise argparse.ArgumentTypeError(f"Config path is not a file: {config_path}")
    if not os.access(normalized, os.R_OK):
        raise argparse.ArgumentTypeError(f"Config file is not readable: {config_path}")
    if not normalized.lower().endswith(".json"):
        raise argparse.ArgumentTypeError(
            f"Config file should be a JSON file: {config_path}"
        )
    return normalized


def setup_global_logging():
    """Configures the global logging format and level."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - DPEVA - %(levelname)s - %(message)s')

def load_and_resolve_config(config_path):
    """
    Loads a JSON configuration file and resolves relative paths.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: The configuration dictionary with resolved paths.
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise CLIUserInputError(
            f"Invalid JSON in config file: {config_path} (line {e.lineno}, column {e.colno})"
        ) from e
    except PermissionError as e:
        raise CLIUserInputError(f"Config file is not readable: {config_path}") from e
    except OSError as e:
        raise CLIUserInputError(f"Failed to read config file: {config_path} ({e})") from e
    return resolve_config_paths(config, config_path)

def handle_train(args):
    """
    Handles the 'train' command.
    Initializes and runs the TrainingWorkflow.

    Args:
        args (argparse.Namespace): Command-line arguments containing 'config'.
    """
    from dpeva.workflows.train import TrainingWorkflow
    config = load_and_resolve_config(args.config)
    workflow = TrainingWorkflow(config)
    workflow.run()

def handle_infer(args):
    """
    Handles the 'infer' command.
    Initializes and runs the InferenceWorkflow.

    Args:
        args (argparse.Namespace): Command-line arguments containing 'config'.
    """
    from dpeva.workflows.infer import InferenceWorkflow
    config = load_and_resolve_config(args.config)
    workflow = InferenceWorkflow(config, config_path=os.path.abspath(args.config))
    workflow.run()

def handle_feature(args):
    """
    Handles the 'feature' command.
    Initializes and runs the FeatureWorkflow.

    Args:
        args (argparse.Namespace): Command-line arguments containing 'config'.
    """
    from dpeva.workflows.feature import FeatureWorkflow
    config = load_and_resolve_config(args.config)
    workflow = FeatureWorkflow(config)
    workflow.run()

def handle_collect(args):
    """
    Handles the 'collect' command.
    Initializes and runs the CollectionWorkflow.

    Args:
        args (argparse.Namespace): Command-line arguments containing 'config'.
    """
    from dpeva.workflows.collect import CollectionWorkflow
    config = load_and_resolve_config(args.config)
    # CollectionWorkflow needs config_path for self-submission
    workflow = CollectionWorkflow(config, config_path=os.path.abspath(args.config))
    workflow.run()

def handle_analysis(args):
    """
    Handles the 'analysis' command.
    Initializes and runs the AnalysisWorkflow.

    Args:
        args (argparse.Namespace): Command-line arguments containing 'config'.
    """
    from dpeva.workflows.analysis import AnalysisWorkflow
    config = load_and_resolve_config(args.config)
    workflow = AnalysisWorkflow(config, config_path=os.path.abspath(args.config))
    workflow.run()

def handle_label(args):
    """
    Handles the 'label' command.
    Initializes and runs the LabelingWorkflow (FP Calculation).
    
    Args:
        args (argparse.Namespace): Command-line arguments containing 'config'.
    """
    from dpeva.workflows.labeling import LabelingWorkflow
    from dpeva.config import LabelingConfig
    
    config_dict = load_and_resolve_config(args.config)
    # Validate and parse config using Pydantic model
    config = LabelingConfig(**config_dict)
    workflow = LabelingWorkflow(config)
    stage = getattr(args, "stage", "all")
    if stage == "prepare":
        workflow.run_prepare()
        return
    if stage == "execute":
        workflow.run_execute()
        return
    if stage == "postprocess":
        workflow.run_postprocess()
        return
    if stage == "extract":
        workflow.run_extract()
        return
    workflow.run()

def main():
    """
    Main entry point for the CLI.
    Parses arguments, displays banner, and executes the selected command.
    """
    setup_global_logging()
    parser = argparse.ArgumentParser(prog="dpeva", description="DP-EVA: Deep Potential Evolution Accelerator")
    parser.add_argument("--no-banner", action="store_true", help="Skip the welcome banner")
    
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available Workflows")

    # Training Sub-command
    p_train = subparsers.add_parser("train", help="Run Training (Parallel Fine-tuning) Workflow")
    p_train.add_argument("config", type=validate_config_path, help="Path to configuration JSON")
    p_train.set_defaults(func=handle_train)

    # Inference Sub-command
    p_infer = subparsers.add_parser("infer", help="Run Inference (Parallel Evaluation) Workflow")
    p_infer.add_argument("config", type=validate_config_path, help="Path to configuration JSON")
    p_infer.set_defaults(func=handle_infer)

    # Feature Sub-command
    p_feature = subparsers.add_parser("feature", help="Run Feature Generation Workflow")
    p_feature.add_argument("config", type=validate_config_path, help="Path to configuration JSON")
    p_feature.set_defaults(func=handle_feature)

    # Collection Sub-command
    p_collect = subparsers.add_parser("collect", help="Run Data Collection Workflow")
    p_collect.add_argument("config", type=validate_config_path, help="Path to configuration JSON")
    p_collect.set_defaults(func=handle_collect)
    
    # Analysis Sub-command
    p_analysis = subparsers.add_parser("analysis", help="Run Inference Analysis Workflow")
    p_analysis.add_argument("config", type=validate_config_path, help="Path to configuration JSON")
    p_analysis.set_defaults(func=handle_analysis)

    # Labeling Sub-command
    p_label = subparsers.add_parser(
        "label",
        help="Run FP Labeling Workflow",
        description="Run FP Labeling Workflow with stage control.",
        epilog=(
            "Examples:\n"
            "  dpeva label config.json\n"
            "  dpeva label config.json --stage prepare\n"
            "  dpeva label config.json --stage execute\n"
            "  dpeva label config.json --stage extract\n"
            "  dpeva label config.json --stage postprocess"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p_label.add_argument("config", type=validate_config_path, help="Path to configuration JSON")
    p_label.add_argument(
        "--stage",
        choices=["all", "prepare", "execute", "extract", "postprocess"],
        default="all",
        help="Run labeling by stage: all|prepare|execute|extract|postprocess (default: all).",
    )
    p_label.set_defaults(func=handle_label)

    args = parser.parse_args()
    
    if not args.no_banner:
        show_banner()
        
    try:
        args.func(args)
    except CLIUserInputError as e:
        logging.error(f"Execution failed: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Execution failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
