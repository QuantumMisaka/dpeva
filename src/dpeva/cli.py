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
import tempfile
from dpeva.config_migration import MigrationResult, migrate_legacy_config
from dpeva.utils.config import resolve_config_paths
from dpeva.utils.banner import show_banner
from dpeva.run.context import RunOptions

# Lazy imports for workflows to improve CLI startup time
# Workflows are imported inside handler functions

LABEL_STAGE_TOKENS = {"prepare", "execute", "extract", "postprocess"}


class CLIUserInputError(ValueError):
    pass


class EvaluationCardPublicationError(RuntimeError):
    """A card publication failed, with explicit publication state."""

    def __init__(self, message: str, path: str, *, published: bool) -> None:
        super().__init__(message)
        self.path = path
        self.published = published


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


def _original_config(result):
    if isinstance(result, MigrationResult) and result.original is not None:
        return result.original
    return _normalized_config(result)


def _config_metadata(result):
    if not isinstance(result, MigrationResult):
        return None
    return {
        "schema_version": "1.0",
        "input_schema_version": result.input_schema_version,
        "migration_warnings": [
            {
                "field": warning.field,
                "replacement": warning.replacement,
                "removal_version": warning.removal_version,
            }
            for warning in result.warnings
        ],
    }


def add_run_options(parser: argparse.ArgumentParser) -> None:
    """Add immutable run allocation controls to the pilot workflows."""
    parser.add_argument("--run-id", help="Stable run identity for manifest evidence")
    parser.add_argument("--resume", action="store_true", help="Resume an incomplete run")
    parser.add_argument("--force", action="store_true", help="Start a new attempt for an existing run")
    parser.add_argument("--reason", help="Required audit reason for --force")


def _run_options(args) -> RunOptions:
    return RunOptions(
        run_id=getattr(args, "run_id", None),
        resume=getattr(args, "resume", False),
        force=getattr(args, "force", False),
        reason=getattr(args, "reason", None),
    )


def load_config_with_metadata(config_path) -> MigrationResult:
    """
    Loads a JSON configuration file and resolves relative paths.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        MigrationResult: The migrated configuration with resolved paths,
            source metadata, and compatibility warnings.
    """
    raw = load_json_config(config_path)
    migrated = migrate_legacy_config(raw)
    resolved = resolve_config_paths(migrated.normalized, config_path)
    result = MigrationResult(
        normalized=resolved,
        warnings=migrated.warnings,
        original=raw,
        input_schema_version=migrated.input_schema_version,
    )
    _log_migration_warnings(result)
    return result


def load_and_resolve_config(config_path) -> dict:
    """Load a config using the legacy dict-returning helper contract."""
    return load_config_with_metadata(config_path).normalized

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
    loaded = load_config_with_metadata(args.config)
    original_config = _original_config(loaded)
    config = _normalized_config(loaded)
    workflow = InferenceWorkflow(
        config,
        config_path=os.path.abspath(args.config),
        original_config=original_config,
        config_metadata=_config_metadata(loaded),
        run_options=_run_options(args),
    )
    workflow.run()

def handle_feature(args):
    """
    Handles the 'feature' command.
    Initializes and runs the FeatureWorkflow.

    Args:
        args (argparse.Namespace): Command-line arguments containing 'config'.
    """
    from dpeva.workflows.feature import FeatureWorkflow
    loaded = load_config_with_metadata(args.config)
    original_config = _original_config(loaded)
    config = _normalized_config(loaded)
    workflow = FeatureWorkflow(
        config,
        original_config=original_config,
        config_metadata=_config_metadata(loaded),
        run_options=_run_options(args),
    )
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


def _open_fsyncable_directory(path: str, output_path: str) -> int:
    """Open and preflight a POSIX directory before creating evidence."""
    if os.name != "posix" or not hasattr(os, "link"):
        raise EvaluationCardPublicationError(
            "evaluation-card publication requires POSIX directory fsync and hard links; "
            "card not published",
            output_path,
            published=False,
        )
    directory_fd = None
    try:
        directory_fd = os.open(path, os.O_RDONLY)
        os.fsync(directory_fd)
    except OSError as exc:
        if directory_fd is not None:
            os.close(directory_fd)
        raise EvaluationCardPublicationError(
            f"parent directory durability preflight failed; card not published: {exc}",
            output_path,
            published=False,
        ) from exc
    return directory_fd


def _publish_evaluation_card(path, card) -> None:
    """Publish one valid card atomically without replacing existing evidence."""
    output_path = os.path.abspath(os.path.expanduser(os.fspath(path)))
    output_parent = os.path.dirname(output_path)
    os.makedirs(output_parent, exist_ok=True)
    payload = card.model_dump_json(indent=2) + "\n"
    temporary_path = None
    directory_fd = _open_fsyncable_directory(output_parent, output_path)
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_parent,
            prefix=f".{os.path.basename(output_path)}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

        # A hard link is an atomic no-replace publication primitive on the
        # local filesystems used for evidence.  It closes the race between a
        # preflight existence check and publication, preserving immutable
        # candidate evidence if another process wins the destination.
        os.link(temporary_path, output_path)
        os.unlink(temporary_path)
        temporary_path = None
        try:
            os.fsync(directory_fd)
        except OSError as exc:
            raise EvaluationCardPublicationError(
                f"evaluation card published but parent-directory durability is not confirmed: {exc}",
                output_path,
                published=True,
            ) from exc
    finally:
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass
        os.close(directory_fd)


def handle_eval_card(args) -> None:
    """Assemble and atomically publish one candidate evaluation card."""
    from dpeva.config import EvaluationCardConfig
    from dpeva.evaluation.card import build_evaluation_card

    migrated = load_and_resolve_config(args.config)
    config = EvaluationCardConfig.model_validate(_normalized_config(migrated))
    card = build_evaluation_card(config)
    _publish_evaluation_card(config.output_path, card)
    logging.info("Evaluation card written: %s", config.output_path)

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
    add_run_options(p_infer)
    p_infer.set_defaults(func=handle_infer)

    # Feature Sub-command
    p_feature = subparsers.add_parser("feature", help="Run Feature Generation Workflow")
    p_feature.add_argument("config", type=validate_config_path, help="Path to configuration JSON")
    add_run_options(p_feature)
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

    p_eval_card = subparsers.add_parser(
        "eval-card", help="Assemble one candidate evaluation card from evidence references"
    )
    p_eval_card.add_argument("config", type=validate_config_path, help="Path to configuration JSON")
    p_eval_card.set_defaults(func=handle_eval_card)

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
