import json
import argparse
import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("dpeva.runner.train")

# Try importing dpeva
try:
    from dpeva.workflows.train import TrainingWorkflow
except ImportError:
    logger.error("The 'dpeva' package is not installed in the current Python environment.")
    logger.error("Please install it using: pip install -e .")
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
