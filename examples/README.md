# DP-EVA Examples

This directory contains usage examples and recipes for DP-EVA workflows.

## Directory Structure

- `recipes/`: Contains configuration templates for different workflows.
  - `feature_generation/`: Generating descriptors from data.
  - `training/`: Training DeepMD models.
  - `inference/`: Running inference and calculating errors.
  - `collection/`: Active learning collection (UQ + Sampling).
  - `analysis/`: Post-processing analysis.
  - `sampling_direct/`: Standalone sampling using Standard DIRECT.
  - `sampling_2direct/`: Standalone sampling using 2-Step DIRECT.

## Prerequisites

1. **DeepMD-kit Environment**: Ensure `dp` command is available or the environment path in `env_setup` is correct.
   - Example configs use `/opt/envs/deepmd3.1.2.env`. Update this path if your environment differs.
2. **Slurm (Optional)**: If using `backend: "slurm"`, ensure you have access to `sbatch`.
3. **Data Preparation**:
   - The examples assume certain data directories exist relative to the config file or work directory.
   - Common expected data:
     - `other_dpdata/`: Candidate pool data.
     - `sampled_dpdata/`: Training data.
     - `desc_pool/`: Descriptors for candidate pool.
     - `DPA-3.1-3M.pt`: Pre-trained model (if using transfer learning).

## How to Run

Most examples are designed to be run via the CLI:

```bash
# Example: Running feature generation
dpeva feature recipes/feature_generation/config_feature.json

# Example: Running collection workflow
dpeva collect recipes/collection/config_single_normal.json
```

**Note**: You should copy the `recipes` or specific config files to your working directory and adjust paths (like `data_path`, `project`) to point to your actual data.
