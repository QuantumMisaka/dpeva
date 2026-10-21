# DP-EVA Examples

This directory contains interactive tutorials, workflow configuration recipes, and helper scripts for DP-EVA.

## Interactive teaching notebooks (CPU)

Start with the [DP-EVA paper case tutorial](tutorials/paper_case/README.md)
for a 20–30 minute demonstration using real paper data. Four notebooks cover
representation, DIRECT, uncertainty, joint sampling, and workflow handoff.
They include precomputed data and need only the tutorial's CPU dependencies.

## ⚠️ Important Prerequisites

Before running the workflow recipes with `dpeva` CLI or Slurm submission:

1.  **Check Environment Paths**:
    Open the configuration files (e.g., `config_collect_normal.json`) and verify the `env_setup` section.
    You **MUST** update paths like `/opt/envs/deepmd3.1.2.env` to match your actual cluster environment.

2.  **Data Preparation**:
    The examples assume certain data directories exist relative to the config file.
    -   `other_dpdata/`: Candidate pool data (for inference/collection).
    -   `sampled_dpdata/`: Training data (for training/joint collection).
    -   `desc_pool/`: Pre-calculated descriptors.

## Directory Structure

-   **`tutorials/`**: CPU notebooks with bundled paper data and teaching instructions.
    -   See [tutorials/README.md](tutorials/README.md) to start.
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
