import argparse
import sys
import json
import logging
from dpeva.utils.config import resolve_config_paths
from dpeva.utils.banner import show_banner

# Lazy imports for workflows to improve CLI startup time
# Workflows are imported inside handler functions

def setup_global_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - DPEVA - %(levelname)s - %(message)s')

def load_and_resolve_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return resolve_config_paths(config, config_path)

def handle_train(args):
    from dpeva.workflows.train import TrainingWorkflow
    config = load_and_resolve_config(args.config)
    workflow = TrainingWorkflow(config)
    workflow.run()

def handle_infer(args):
    from dpeva.workflows.infer import InferenceWorkflow
    import os
    config = load_and_resolve_config(args.config)
    workflow = InferenceWorkflow(config, config_path=os.path.abspath(args.config))
    workflow.run()

def handle_feature(args):
    from dpeva.workflows.feature import FeatureWorkflow
    config = load_and_resolve_config(args.config)
    workflow = FeatureWorkflow(config)
    workflow.run()

def handle_collect(args):
    from dpeva.workflows.collect import CollectionWorkflow
    import os
    config = load_and_resolve_config(args.config)
    # CollectionWorkflow needs config_path for self-submission
    workflow = CollectionWorkflow(config, config_path=os.path.abspath(args.config))
    workflow.run()

def handle_analysis(args):
    from dpeva.workflows.analysis import AnalysisWorkflow
    config = load_and_resolve_config(args.config)
    workflow = AnalysisWorkflow(config)
    workflow.run()

def main():
    setup_global_logging()
    parser = argparse.ArgumentParser(prog="dpeva", description="DP-EVA: Deep Potential Evolution Accelerator")
    parser.add_argument("--no-banner", action="store_true", help="Skip the welcome banner")
    
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available Workflows")

    # Training Sub-command
    p_train = subparsers.add_parser("train", help="Run Training (Parallel Fine-tuning) Workflow")
    p_train.add_argument("config", help="Path to configuration JSON")
    p_train.set_defaults(func=handle_train)

    # Inference Sub-command
    p_infer = subparsers.add_parser("infer", help="Run Inference (Parallel Evaluation) Workflow")
    p_infer.add_argument("config", help="Path to configuration JSON")
    p_infer.set_defaults(func=handle_infer)

    # Feature Sub-command
    p_feature = subparsers.add_parser("feature", help="Run Feature Generation Workflow")
    p_feature.add_argument("config", help="Path to configuration JSON")
    p_feature.set_defaults(func=handle_feature)

    # Collection Sub-command
    p_collect = subparsers.add_parser("collect", help="Run Data Collection Workflow")
    p_collect.add_argument("config", help="Path to configuration JSON")
    p_collect.set_defaults(func=handle_collect)
    
    # Analysis Sub-command
    p_analysis = subparsers.add_parser("analysis", help="Run Inference Analysis Workflow")
    p_analysis.add_argument("config", help="Path to configuration JSON")
    p_analysis.set_defaults(func=handle_analysis)

    args = parser.parse_args()
    
    if not args.no_banner:
        show_banner()
        
    try:
        args.func(args)
    except Exception as e:
        logging.error(f"Execution failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
