# DP-EVA Recipes

This directory contains configuration templates for the core workflows in DP-EVA.

## 1. Active Learning Collection (`collection/`)

The `collect` workflow is the heart of DP-EVA, handling Uncertainty Quantification (UQ), Filtering, and Sampling.

| Configuration File | Mode | Description |
| :--- | :--- | :--- |
| **`config_normal.json`** | **Normal Sampling** | Selects new data purely based on the candidate pool's descriptor distribution. Suitable for initial exploration or when no training set is available/relevant. |
| **`config_joint.json`** | **Joint Sampling** | Considers both the candidate pool and an existing training set (`training_data_dir`). Ensures new samples cover the "blind spots" of the current model relative to the training data. |

**Note**:
-   **Data Pool Structure**: DP-EVA automatically detects whether your data is a single pool (one system) or multi-pool (multiple systems/trajectories) based on the directory structure. You do not need separate config files for this.
-   **Clustering**: Default `direct_n_clusters` is set to 1000 for robust production use. For very small test sets (<1000 frames), you may reduce this value.

**Usage:**
```bash
dpeva collect examples/recipes/collection/config_normal.json
```

## 2. Model Training (`training/`)

Located in `training/`.
Standard configuration for fine-tuning DeepMD models using DP-EVA's parallel training infrastructure.

-   **Input**: `input.json` (DeepMD config), training data.
-   **Output**: Trained models in `work_dir/0..N-1/`.

**Usage:**
```bash
dpeva train examples/recipes/training/config_train.json
```

## 3. Inference (`inference/`)

Located in `inference/`.
Evaluates trained models on a test set (candidate pool) to calculate errors (RMSE) and generate parity plots.

**Usage:**
```bash
dpeva infer examples/recipes/inference/config_infer.json
```

## 4. Feature Generation (`feature_generation/`)

Located in `feature_generation/`.
Pre-calculates descriptors (e.g., using `dp eval-desc`) for use in the Collection workflow. This step is often required before running `dpeva collect` if descriptors are not generated on-the-fly.

**Usage:**
```bash
dpeva feature examples/recipes/feature_generation/config_feature.json
```

## 5. Standalone Sampling Tools

For users who want to use the sampling algorithms directly without the full DP-EVA workflow.

-   **Standard DIRECT (`sampling_direct/`)**:
    -   Uses structural clustering to select representative frames.
    -   Run with: `python sampling_direct/run_direct.py ...`

-   **2-Step DIRECT (`sampling_2direct/`)**:
    -   Two-step clustering (Structural -> Atomic) to optimize for labeling costs.
    -   Run with: `python sampling_2direct/run_2direct.py ...`
