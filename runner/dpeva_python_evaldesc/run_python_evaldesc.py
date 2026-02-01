import json
import argparse
import os
import sys

# Try importing dpeva
try:
    import dpeva
    from dpeva.workflows.feature import FeatureWorkflow
except ImportError:
    print("Error: The 'dpeva' package is not installed in the current Python environment.")
    print("Please install it using: pip install -e .")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run DPEVA Feature Generation Workflow (Python Mode)")
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
            # If path starts with /, it's absolute, os.path.join handles it correctly (ignores config_dir)
            # If path is relative, it joins with config_dir
            if os.path.isabs(path):
                return path
            return os.path.abspath(os.path.join(config_dir, path))

        # Resolve paths in config
        if "datadir" in config:
            config["datadir"] = resolve_path(config["datadir"])
            
        if "modelpath" in config:
            config["modelpath"] = resolve_path(config["modelpath"])
            
        if "savedir" in config:
            config["savedir"] = resolve_path(config["savedir"])
            
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON config: {e}")
        return

    print("🚀 Starting DPEVA Feature Generation Workflow (Python Mode)...")
    print(f"📄 Configuration: {config_path}")
    print(f"🔧 Mode: {config.get('mode', 'python')}")
    
    workflow = FeatureWorkflow(config)
    workflow.run()
    
    print("✅ Workflow Completed.")

if __name__ == "__main__":
    main()
