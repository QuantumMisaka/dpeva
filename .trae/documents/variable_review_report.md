# Variable Review and Optimization Report

## 1. Overview
This report documents the systematic review of input variables in the `runner` and `src/dpeva/workflows` directories. The goal is to identify unused, redundant, or inconsistently named variables and optimize the codebase for clarity and maintainability.

## 2. Workflow Analysis

### 2.1 FeatureWorkflow (`src/dpeva/workflows/feature.py`)
- **Input Source**: `runner/dpeva_evaldesc/config.json`
- **Variables Analyzed**:
  - `datadir`: Path to input data. **[Inconsistent Naming]**
  - `modelpath`: Path to model.
  - `savedir`: Output directory.
  - `head`: Model head.
  - `mode`: Execution mode (`cli`, `python`).
  - `submission`: Backend config.
  - `batch_size`: Batch size (default 1000).
  - `omp_threads`: OMP threads (default 24). **[Redundant with env_setup]**
  - `format`: Data format (default `deepmd/npy`).
  - `output_mode`: Output mode (default `atomic`).

### 2.2 InferenceWorkflow (`src/dpeva/workflows/infer.py`)
- **Input Source**: `runner/dpeva_test/config.json`
- **Variables Analyzed**:
  - `test_data_path`: Path to input data. **[Inconsistent Naming]**
  - `output_basedir`: Base output directory.
  - `task_name`: Task identifier.
  - `head`: Model head.
  - `submission`: Backend config.
  - `omp_threads`: OMP threads (default 2). **[Redundant with env_setup]**
  - `ref_energies`: Reference energies for cohesive energy calculation.

### 2.3 TrainingWorkflow (`src/dpeva/workflows/train.py`)
- **Input Source**: `runner/dpeva_train/config.json`
- **Variables Analyzed**:
  - `work_dir`: Working directory.
  - `input_json_path`: Base input JSON.
  - `num_models`: Number of models.
  - `mode`: Training mode (`init` or `cont`).
  - `seeds`: Model initialization seeds.
  - `training_seeds`: Data shuffling seeds. **[Potentially Redundant if identical to seeds]**
  - `base_model_path`: Path to base model.
  - `training_data_path`: Override data path.
  - `finetune_head_name`: Head name.
  - `backend`: Submission backend.
  - `slurm_config`: Slurm options.
  - `omp_threads`: OMP threads.

### 2.4 CollectWorkflow (`src/dpeva/workflows/collect.py`)
- **Input Source**: `runner/dpeva_collect/config.json`
- **Status**: Previously optimized. `testdata_string` was identified and removed.
- **Variables**:
  - `testdata_dir`: Candidate data path.
  - `training_data_dir`: Training data path (Optional).
  - `desc_filename`: Deprecated but kept for backward compatibility.

## 3. Identified Issues & Optimization Plan

### 3.1 Naming Inconsistency: Data Paths
Different workflows use different names for the primary input data directory:
- `FeatureWorkflow`: `datadir`
- `InferenceWorkflow`: `test_data_path`
- `TrainingWorkflow`: `training_data_path`
- `CollectWorkflow`: `testdata_dir`

**Optimization**: Standardize `FeatureWorkflow` and `InferenceWorkflow` to use `data_path`. This is a clear, generic term suitable for these workflows. `TrainingWorkflow` and `CollectWorkflow` retain specific prefixes (`training_`, `testdata_`) to distinguish between multiple data sources (training vs candidate) where applicable.

### 3.2 Redundancy: OMP Threads
`omp_threads` is defined as an explicit argument but is often also set via `env_setup` in the configuration.
- **Action**: Retain `omp_threads` as a convenience parameter for Python-based execution where `env_setup` might not be sourced, but ensure it doesn't conflict. (No code change required, just awareness).

### 3.3 Unused Variables
- `testdata_string` was previously identified and removed from `CollectWorkflow`.
- No other strictly "unused" variables were found in the active `run()` methods of the analyzed workflows.

## 4. Applied Changes

1.  **Refactor `FeatureWorkflow`**:
    -   Renamed `datadir` -> `data_path`.
    -   Updated `runner/dpeva_evaldesc/config.json`.

2.  **Refactor `InferenceWorkflow`**:
    -   Renamed `test_data_path` -> `data_path`.
    -   Updated `runner/dpeva_test/config.json`.

3.  **Code Cleanup**:
    -   Ensured variable access is consistent.

These changes improve code consistency and reduce cognitive load when configuring different workflows.
