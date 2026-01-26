import json
import argparse
import os
import sys

# Add src to sys.path so we can import dpeva modules even if not installed
# This assumes the script is running from dpeva/runner/dpeva_train/
# Adjust relative path as needed if directory structure changes
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir)) # Up from runner/dpeva_train to dpeva root
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from dpeva.workflows.train import TrainingWorkflow

def main():
    parser = argparse.ArgumentParser(description="Run DPEVA Training Workflow from Config")
    parser.add_argument("config", help="Path to the JSON configuration file")
    args = parser.parse_args()
    
    config_path = args.config
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at {config_path}")
        return

    try:
        with open(config_path, "r") as f:
            config = json.load(f)
            
        # Resolve relative paths relative to the config file location
        config_dir = os.path.dirname(os.path.abspath(config_path))
        
        def resolve_path(path):
            if not path or not isinstance(path, str):
                return path
            # If path is absolute, join(base, path) returns path
            # If path is relative, join(base, path) returns base/path
            return os.path.abspath(os.path.join(config_dir, path))

        # Resolve paths in config
        if "work_dir" in config:
            config["work_dir"] = resolve_path(config["work_dir"])
            
        if "input_json_path" in config:
            config["input_json_path"] = resolve_path(config["input_json_path"])
            
        if "base_model_path" in config:
            config["base_model_path"] = resolve_path(config["base_model_path"])
            
        if "training_data_path" in config:
            config["training_data_path"] = resolve_path(config["training_data_path"])
            
        # Handle env_setup in slurm_config: support list of strings -> join to string
        if "slurm_config" in config and "env_setup" in config["slurm_config"]:
            env_setup = config["slurm_config"]["env_setup"]
            if isinstance(env_setup, list):
                config["slurm_config"]["env_setup"] = "\n".join(env_setup)

    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON config: {e}")
        return

    print("🚀 Starting DPEVA Training Workflow...")
    print(f"📄 Configuration: {config_path}")
    
    workflow = TrainingWorkflow(config)
    workflow.run()
    
    print("✅ Workflow Completed.")

if __name__ == "__main__":
    main()
