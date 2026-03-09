---
title: Archived Document
status: archived
audience: Historians
last-updated: 2026-03-09
---

# Labeling Workflow Implementation Review

## Overview
This document details the implementation of the Labeling Workflow in DP-EVA, specifically addressing the requirements for structure classification, hierarchical directory generation, output organization, and logging.

## 1. Directory Structure Implementation (Requirement 1)
The original script `FeCHO_fp_set.py` logic has been fully ported to `LabelingManager.prepare_tasks`.

*   **Structure Classification**:
    *   Implemented in `AbacusGenerator._judge_vacuum` (for dimensionality) and `LabelingManager.prepare_tasks` (for cubic cluster check).
    *   Classifies structures into: `cluster`, `cubic_cluster`, `layer`, `string`, `bulk`.
*   **Hierarchical Generation**:
    *   Input files are generated at: `inputs/[dataset_name]/[stru_type]/[task_name]`.
    *   This preserves the logical organization of tasks before packing.
*   **Task Packing**:
    *   `TaskPacker.pack` has been enhanced to recursively scan for `INPUT` files using `rglob`.
    *   It moves tasks from the hierarchical structure into flat job bundles (`N_50_0`, `N_50_1`, etc.) for efficient submission.
    *   **Note**: While packing flattens the *submission* directories, the task names (`dataset_idx`) preserve traceability.

## 2. Output Organization (Requirement 2)
The output directory structure has been reorganized to separate cleaned data from anomalies.

*   **Cleaned Data**: Exported to `outputs/cleaned/[dataset_name]`.
*   **Anomalies**: Exported to `outputs/anomalies/[dataset_name]`.
*   **Visualization**: `extxyz` files for anomalies are stored in `outputs/anomalies/extxyz`.

This ensures that the main `outputs` directory remains clean and structured.

## 3. Statistics Logging (Requirement 3)
Enhanced logging provides a detailed breakdown of the labeling process.

*   **Global Stats**: Total converged, cleaned, and filtered structures.
*   **Per-Pool Stats**: For each dataset (pool), the system logs:
    *   Converged count
    *   Cleaned count (passed quality checks)
    *   Filtered count (outliers removed)
*   **Implementation**: `collect_and_export` reconstructs the pool mapping by tracing back the task directory paths in `CONVERGED`.

## 4. Log Noise Reduction (Requirement 4)
The job monitoring loop has been optimized to reduce log volume.

*   **Polling Frequency**: Remains at 60 seconds to ensure timely response to job completion.
*   **Logging Frequency**: Reduced to every 10 minutes (every 10th poll).
*   **Implementation**: Added a `wait_count` counter in `_monitor_slurm_jobs`.

## 5. Code Quality & Standards
*   **Refactoring**: Logic from `FeCHO_fp_set.py` (magmom, kpoints, vacuum check) has been modularized into `AbacusGenerator`.
*   **Documentation**: All methods have Google-style docstrings.
*   **Cleanliness**: Redundant comments and debug code have been removed.

## 6. Verification
The implementation has been verified against the `fp-setting-2` test case, confirming:
1.  Correct generation of `cluster`/`bulk` subdirectories in `inputs`.
2.  Correct packing of tasks into `N_50_X` bundles.
3.  Successful submission and monitoring (with reduced logging).
4.  Correct export of data to `outputs/cleaned` and `outputs/anomalies`.
