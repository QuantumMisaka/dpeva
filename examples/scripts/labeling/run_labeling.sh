#!/bin/bash

# Example script to run DP-EVA Labeling Workflow
# Ensure your environment has ase-abacus and dpeva.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_DIR="$(cd "${SCRIPT_DIR}/../../recipes/labeling" && pwd)"

# Run with GPU config
dpeva label "${RECIPE_DIR}/config_gpu.json"

# Run with CPU config
# dpeva label "${RECIPE_DIR}/config_cpu.json"

# If you want to see help
# dpeva label --help
