import json
import argparse
import os
import sys

# Add src to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir)) # Up from runner/dpeva_evaldesc to dpeva root
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from dpeva.workflows.feature import FeatureWorkflow

def main():
    parser = argparse.ArgumentParser(description="Run DPEVA Feature Generation Workflow from Config")
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
            return os.path.abspath(os.path.join(config_dir, path))

        # Resolve paths in config
        if "datadir" in config:
            config["datadir"] = resolve_path(config["datadir"])
            
        if "modelpath" in config:
            config["modelpath"] = resolve_path(config["modelpath"])
            
        if "savedir" in config:
            config["savedir"] = resolve_path(config["savedir"])
            
        # Handle env_setup in slurm_config: support list of strings -> join to string
        # Actually logic is handled in Workflow class now, but good to be consistent
        pass

    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON config: {e}")
        return

    print("🚀 Starting DPEVA Feature Generation Workflow...")
    print(f"📄 Configuration: {config_path}")
    
    workflow = FeatureWorkflow(config)
    workflow.run()
    
    print("✅ Workflow Completed.")

if __name__ == "__main__":
    main()
