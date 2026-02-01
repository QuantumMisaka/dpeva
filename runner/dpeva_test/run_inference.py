import json
import argparse
import os
import sys

try:
    from dpeva.workflows.infer import InferenceWorkflow
except ImportError:
    print("Error: The 'dpeva' package is not installed in the current Python environment.")
    print("Please install it using: pip install -e .")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run DPEVA Inference Workflow from Config")
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
            if not path:
                return path
            # If path is absolute, join(base, path) returns path
            # If path is relative, join(base, path) returns base/path
            return os.path.abspath(os.path.join(config_dir, path))

        # Resolve paths in config
        if "test_data_path" in config:
            config["test_data_path"] = resolve_path(config["test_data_path"])
            
        if "output_basedir" in config:
            config["output_basedir"] = resolve_path(config["output_basedir"])
            
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON config: {e}")
        return

    print("🚀 Starting DPEVA Inference Workflow...")
    print(f"📄 Configuration: {config_path}")
    
    workflow = InferenceWorkflow(config)
    workflow.run()
    
    print("✅ Workflow Completed.")

if __name__ == "__main__":
    main()
