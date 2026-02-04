import json
import argparse
import os
import sys

# Try importing dpeva
try:
    from dpeva.workflows.feature import FeatureWorkflow
    from dpeva.utils.config import resolve_config_paths
except ImportError:
    print("Error: The 'dpeva' package is not installed in the current Python environment.")
    print("Please install it using: pip install -e .")
    sys.exit(1)

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
            
        # Resolve paths using utility
        config = resolve_config_paths(config, config_path)
            
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON config: {e}")
        return

    mode = config.get("mode", "cli")
    print(f"🚀 Starting DPEVA Feature Generation Workflow (Mode: {mode})...")
    print(f"📄 Configuration: {config_path}")
    
    workflow = FeatureWorkflow(config)
    workflow.run()
    
    print("✅ Workflow Completed.")

if __name__ == "__main__":
    main()
