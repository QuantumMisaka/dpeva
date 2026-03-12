# DP-EVA Examples

This directory contains minimal reproducible examples (recipes) for DP-EVA workflows.

## ⚠️ Important Prerequisites

Before running any examples, especially with `dpeva` CLI or Slurm submission:

1.  **Check Environment Paths**:
    Open the configuration files (e.g., `config_collect_normal.json`) and verify the `env_setup` section.
    You **MUST** update paths like `/opt/envs/deepmd3.1.2.env` to match your actual cluster environment.

2.  **Data Preparation**:
    The examples assume certain data directories exist relative to the config file.
    -   `other_dpdata/`: Candidate pool data (for inference/collection).
    -   `sampled_dpdata/`: Training data (for training/joint collection).
    -   `desc_pool/`: Pre-calculated descriptors.

## Directory Structure

-   **`recipes/`**: The core collection of configuration templates for all DP-EVA workflows.
    -   See [recipes/README.md](recipes/README.md) for detailed usage instructions.
-   **`scripts/`**: Executable helper scripts for recipe configurations.
    -   See [scripts/README.md](scripts/README.md) for available entry points.

## Quick Start

Most examples are designed to be run via the DP-EVA CLI.

**Example: Active Learning Collection**

```bash
# Run standard collection workflow
dpeva collect examples/recipes/collection/config_collect_normal.json
```

**Example: Feature Generation**

```bash
# Generate descriptors for a dataset
dpeva feature examples/recipes/feature_generation/config_feature.json
```

**Example: First Principles Labeling**

```bash
# Run ABACUS labeling workflow
dpeva label examples/recipes/labeling/config_cpu.json
```
