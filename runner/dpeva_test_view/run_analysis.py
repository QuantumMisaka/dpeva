import os
import sys
import json
import logging
import argparse

# Try importing dpeva
try:
    from dpeva.workflows.analysis import AnalysisWorkflow
    from dpeva.utils.config import resolve_config_paths
except ImportError:
    print("Error: The 'dpeva' package is not installed in the current Python environment.")
    print("Please install it using: pip install -e .")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run DPEVA Analysis Workflow from Config")
    parser.add_argument("config", nargs="?", help="Path to the JSON configuration file")
    args = parser.parse_args()
    
    # Locate default config if not provided
    if not args.config:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config.json")
    else:
        config_path = args.config

    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Resolve paths using utility
        config = resolve_config_paths(config, config_path)
            
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON config: {e}")
        return
    except Exception as e:
        print(f"Error: {e}")
        return

    print("🚀 Starting DPEVA Analysis Workflow...")
    print(f"📄 Configuration: {config_path}")
    
    try:
        workflow = AnalysisWorkflow(config)
        workflow.run()
        print("✅ Analysis Workflow Completed.")
    except Exception as e:
        print(f"❌ Analysis Workflow Failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
