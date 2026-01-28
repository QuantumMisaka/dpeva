# Robust Support for Multi-Pool Data in DP-EVA

## Problem
The current architecture assumes a single data pool with unique system names. The introduction of "multi-pool" data (Dataset/System structure) creates two issues:
1.  **Name Collision**: `dp test` output parsing (`dataproc.py`) extracts only the basename (System), losing the Dataset context (e.g., `PoolA/Sys1` becomes `Sys1`), which can cause collisions if `PoolB` also has `Sys1`.
2.  **Ordering Mismatch**: `CollectWorkflow` loads descriptors via `glob` (filesystem order) but relies on `dp test` output order for alignment. These orders are not guaranteed to match across multiple pools.
3.  **Feature Generation**: Python mode and CLI mode in `generator.py` do not fully support the recursive 3-level structure (`Root/Dataset/System`) required for multi-pool descriptors.

## Solution Plan

### 1. Enhance `dataproc.py` (Parsing)
- **Objective**: Preserve Dataset context in system names.
- **Change**: Modify `TestResultParser._get_dataname_info` to extract the `Dataset/System` path (last two directory components) instead of just `System` (basename) when a nested structure is detected in the `dp test` output comments.
- **Impact**: `dataname` becomes unique (e.g., `mptrj-FeCOH/C0Fe4H0O8`), enabling precise matching.

### 2. Upgrade `collect.py` (Alignment)
- **Objective**: Align Data and Descriptors strictly with Inference Results.
- **Change**:
    - **Ordered Loading**: Modify `run()` to use the list of unique `dataname`s extracted from `dp test` results as the "Ground Truth" for ordering.
    - **Descriptor Loading**: Update `_load_descriptors` to accept this ordered list and load specific files (`desc_dir/Dataset/System.npy`) instead of `glob`ing wildcards.
    - **Test Data Loading**: Update data loading logic to find and load `testdata_dir/Dataset/System` based on the ordered names, rather than iterating directories.

### 3. Upgrade `generator.py` (Generation)
- **Objective**: Support 3-level structure (`Root/Dataset/System`).
- **Change**:
    - **CLI Mode**: Update `run_cli_generation` to detect if `data_path` contains Datasets (sub-directories). If so, generate a script that iterates over Datasets and runs `dp eval-desc` for each, ensuring `desc_pool` mirrors the `Dataset/System` structure.
    - **Python Mode**: Update logic to recurse into Datasets if detected, ensuring robust `dpdata` loading for the 3-level structure.

## Verification
- Run `CollectWorkflow` with `test-for-multiple-datapool`.
- Verify that `df_uq` contains correct `Dataset/System` names.
- Verify that descriptors and test data are loaded in correct correspondence.
