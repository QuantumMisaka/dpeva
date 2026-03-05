#!/bin/bash
# Example script to run DP-EVA Labeling Workflow

# 1. Activate Environment （Should have ase-abacus and dpeva)
# source /path/to/your/conda/activate dpeva-env

# 2. Run Labeling
# Ensure your config file (labeling_recipe.json) is properly configured with valid paths.
# Especially check 'pp_dir', 'orb_dir', and 'input_data_path'.

# Run with GPU config
dpeva label config_gpu.json

# Run with CPU config
# dpeva label config_cpu.json

# If you want to see help
# dpeva label --help
