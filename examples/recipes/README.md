# DP-EVA Recipes

This directory contains configuration templates for the core workflows in DP-EVA.
Executable example scripts are maintained in `examples/scripts/`.

## 1. Active Learning Collection (`collection/`)

The `collect` workflow is the heart of DP-EVA, handling Uncertainty Quantification (UQ), Filtering, and Sampling.

| Configuration File | Mode | Description |
| :--- | :--- | :--- |
| **`config_collect_normal.json`** | **Normal Sampling** | Selects new data purely based on the candidate pool's descriptor distribution. Suitable for initial exploration or when no training set is available/relevant. |
| **`config_collect_joint.json`** | **Joint Sampling** | Considers both the candidate pool and an existing training set (`training_data_dir`). Ensures new samples cover the "blind spots" of the current model relative to the training data. |

**Note**:
-   **Data Pool Structure**: DP-EVA automatically detects whether your data is a single pool (one system) or multi-pool (multiple systems/trajectories) based on the directory structure. You do not need separate config files for this.
-   **Clustering**: Default `direct_n_clusters` is set to 1000 for robust production use. For very small test sets (<1000 frames), you may reduce this value.

**Usage:**
```bash
dpeva collect examples/recipes/collection/config_collect_normal.json
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
Evaluates trained models on a test set (candidate pool) and writes raw `dp test` outputs.
Set `auto_analysis=true` only for local backend if you want chained analysis.
For Slurm, run `dpeva analysis` after jobs finish.

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
    -   Run with `dpeva collect examples/recipes/sampling_direct/config.json`
    -   Reference: `python examples/scripts/sampling_direct/run_direct.py ...`

-   **2-Step DIRECT (`sampling_2direct/`)**:
    -   Two-step clustering (Structural -> Atomic) to optimize for labeling costs.
    -   Run with `dpeva collect examples/recipes/sampling_2direct/config.json`
    -   Reference: `python examples/scripts/sampling_2direct/run_2direct.py ...`

## 6. Analysis Recipes

- **Dataset + Inference Results (`analysis/config_analysis.json`)**
  - Inputs: `result_dir`, `results_prefix`, and optional `data_path` for composition-aware cohesive analysis.
  - Run with `dpeva analysis examples/recipes/analysis/config_analysis.json`

- **Dataset-only Analysis (`analysis/config_analysis_dataset.json`)**
  - Inputs: `mode=dataset` + `dataset_dir`.
  - Run with `dpeva analysis examples/recipes/analysis/config_analysis_dataset.json`

## 7. Script Entry Points

For programmatic workflow demos and helper scripts, use:

- `examples/scripts/training/train_recipe.py`
- `examples/scripts/inference/infer_recipe.py`
- `examples/scripts/inference/check_model_force_stats.py`
- `examples/scripts/feature_generation/feature_recipe.py`
- `examples/scripts/collection/collect_recipe.py`
- `examples/scripts/analysis/analysis_recipe.py`
- `examples/scripts/labeling/run_labeling.sh`

## 8. Data Cleaning Recipes (`data_cleaning/`)

- **All thresholds enabled (`data_cleaning/config_clean_all_thresholds.json`)**
  - Inputs: labeled `dataset_dir` + inference `result_dir`, with energy/force/stress thresholds.
  - Run with `dpeva clean examples/recipes/data_cleaning/config_clean_all_thresholds.json`

- **Force-only threshold (`data_cleaning/config_clean_force_only.json`)**
  - Inputs: same dataset/result pair, only `force_max_diff_threshold` enabled.
  - Run with `dpeva clean examples/recipes/data_cleaning/config_clean_force_only.json`

- **Passthrough mode (`data_cleaning/config_clean_passthrough.json`)**
  - Inputs: no thresholds enabled; keeps all structures and only exports reports.
  - Run with `dpeva clean examples/recipes/data_cleaning/config_clean_passthrough.json`
