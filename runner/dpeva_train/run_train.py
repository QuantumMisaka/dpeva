import json
import argparse
import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("dpeva.runner.train")

# Try importing dpeva, fallback to src injection if not installed
try:
    import dpeva
except ImportError:
    # Add src to sys.path so we can import dpeva modules even if not installed
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir)) # Up from runner/dpeva_train to dpeva root
    src_path = os.path.join(project_root, "src")
    if src_path not in sys.path:
        sys.path.append(src_path)
    logger.warning(f"dpeva package not found in python path. Added {src_path} to sys.path. Please consider installing via 'pip install -e .'")

try:
    from dpeva.workflows.train import TrainingWorkflow
except ImportError as e:
    logger.error(f"Failed to import TrainingWorkflow: {e}")
    sys.exit(1)

def resolve_path(path, base_dir):
    """Resolves a path relative to a base directory."""
    if not path or not isinstance(path, str):
        return path
    return os.path.abspath(os.path.join(base_dir, path))

def main():
    parser = argparse.ArgumentParser(description="Run DPEVA Training Workflow from Config")
    parser.add_argument("config", help="Path to the JSON configuration file")
    args = parser.parse_args()
    
    config_path = args.config
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found at {config_path}")
        sys.exit(1)

    try:
        with open(config_path, "r") as f:
            config = json.load(f)
            
        # Resolve relative paths relative to the config file location
        config_dir = os.path.dirname(os.path.abspath(config_path))
        
        # Resolve paths in config
        for key in ["work_dir", "input_json_path", "base_model_path", "training_data_path"]:
            if key in config:
                config[key] = resolve_path(config[key], config_dir)
            
        # Handle env_setup in slurm_config: support list of strings -> join to string
        if "slurm_config" in config and "env_setup" in config["slurm_config"]:
            env_setup = config["slurm_config"]["env_setup"]
            if isinstance(env_setup, list):
                config["slurm_config"]["env_setup"] = "\n".join(env_setup)

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON config: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during config loading: {e}")
        sys.exit(1)

    logger.info("🚀 Starting DPEVA Training Workflow...")
    logger.info(f"📄 Configuration: {config_path}")
    
    try:
        workflow = TrainingWorkflow(config)
        workflow.run()
        logger.info("✅ Workflow Completed.")
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
