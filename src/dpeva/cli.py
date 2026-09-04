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
from dpeva.config_migration import MigrationResult, migrate_legacy_config
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

def load_json_config(config_path):
    """Load source JSON without changing the user's file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise CLIUserInputError(
            f"Invalid JSON in config file: {config_path} (line {e.lineno}, column {e.colno})"
        ) from e
    except PermissionError as e:
        raise CLIUserInputError(f"Config file is not readable: {config_path}") from e
    except OSError as e:
        raise CLIUserInputError(f"Failed to read config file: {config_path} ({e})") from e


def _log_migration_warnings(result: MigrationResult) -> None:
    for warning in result.warnings:
        logging.warning(
            "legacy config field %s; use %s; removal target %s",
            warning.field,
            warning.replacement,
            warning.removal_version,
        )


def _normalized_config(result):
    """Return the mapping for compatibility with injected CLI test handlers."""
    if isinstance(result, MigrationResult):
        return result.normalized
    return result


def load_and_resolve_config(config_path) -> MigrationResult:
    """
    Loads a JSON configuration file and resolves relative paths.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        MigrationResult: The migrated configuration with resolved paths and
            compatibility warnings.
    """
    migrated = migrate_legacy_config(load_json_config(config_path))
    resolved = resolve_config_paths(migrated.normalized, config_path)
    result = MigrationResult(normalized=resolved, warnings=migrated.warnings)
    _log_migration_warnings(result)
    return result

def handle_train(args):
    """
    Handles the 'train' command.
    Initializes and runs the TrainingWorkflow.

    Args:
        args (argparse.Namespace): Command-line arguments containing 'config'.
    """
    from dpeva.workflows.train import TrainingWorkflow
    config = _normalized_config(load_and_resolve_config(args.config))
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
    config = _normalized_config(load_and_resolve_config(args.config))
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
    config = _normalized_config(load_and_resolve_config(args.config))
    workflow = FeatureWorkflow(config)
    workflow.run()

def handle_explore(args):
    """
    Handles the 'explore' command.
    Runs an optional trajectory exploration backend.
    """
    from ase.io import read

    from dpeva.config import ExplorationConfig
    from dpeva.exploration.base import ExplorationRequest
    from dpeva.exploration.manager import run_exploration

    config_dict = _normalized_config(load_and_resolve_config(args.config))
    config = ExplorationConfig(**config_dict)
    input_structures = []
    for path in config.input_structure_paths:
        loaded = read(path)
        if isinstance(loaded, list):
            input_structures.extend(loaded)
        else:
            input_structures.append(loaded)

    metadata = dict(config.metadata)
    metadata["result_structure_paths"] = list(config.result_structure_paths)
    request = ExplorationRequest(
        backend=config.backend,
        workflow_type=config.workflow_type,
        work_dir=config.work_dir,
        config_path=config.backend_config_path,
        input_structures=input_structures,
        metadata=metadata,
    )
    result = run_exploration(request)
    if result.status == "failed":
        raise CLIUserInputError(result.error_message or "Exploration failed")

def handle_collect(args):
    """
    Handles the 'collect' command.
    Initializes and runs the CollectionWorkflow.

    Args:
        args (argparse.Namespace): Command-line arguments containing 'config'.
    """
    from dpeva.workflows.collect import CollectionWorkflow
    config = _normalized_config(load_and_resolve_config(args.config))
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
    config = _normalized_config(load_and_resolve_config(args.config))
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
    
    config_dict = _normalized_config(load_and_resolve_config(args.config))
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

def handle_clean(args):
    """
    Handles the 'clean' command.
    Initializes and runs the DataCleaningWorkflow.

    Args:
        args (argparse.Namespace): Command-line arguments containing 'config'.
    """
    from dpeva.workflows.data_cleaning import DataCleaningWorkflow
    config = _normalized_config(load_and_resolve_config(args.config))
    workflow = DataCleaningWorkflow(config)
    workflow.run()


def handle_doctor(args):
    """Report runtime capability checks in human or JSON form."""
    from dpeva.run.doctor import build_doctor_report

    report = build_doctor_report()
    if args.json:
        print(report.model_dump_json(indent=2))
    else:
        for check in report.checks:
            print(f"{check.name}: {check.status} - {check.detail}")
    if report.status != "ok":
        raise SystemExit(1)

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

    # Exploration Sub-command
    p_explore = subparsers.add_parser("explore", help="Run optional trajectory exploration backend")
    p_explore.add_argument("config", type=validate_config_path, help="Path to configuration JSON")
    p_explore.set_defaults(func=handle_explore)

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

    p_clean = subparsers.add_parser("clean", help="Run dataset cleaning by inference error thresholds")
    p_clean.add_argument("config", type=validate_config_path, help="Path to configuration JSON")
    p_clean.set_defaults(func=handle_clean)

    p_doctor = subparsers.add_parser("doctor", help="Report runtime capabilities")
    p_doctor.add_argument("--json", action="store_true", help="Emit a stable JSON report")
    p_doctor.set_defaults(func=handle_doctor)

    args = parser.parse_args()
    
    # A JSON report is a machine-readable stdout contract, so it must not be
    # preceded by the human-facing banner even when --no-banner is omitted.
    if not args.no_banner and not (args.command == "doctor" and args.json):
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
