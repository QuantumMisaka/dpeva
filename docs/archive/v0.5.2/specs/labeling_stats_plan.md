---
title: Archived Document
status: archived
audience: Historians
last-updated: 2026-03-09
---

# Labeling Workflow Refinement Plan

## 1. Objectives
The primary goal is to enable granular statistics tracking for the Labeling Workflow while maintaining efficient HPC execution. Specifically:
*   Track and report statistics (Total, Converged, Cleaned, Filtered) at three levels:
    1.  **Global** (All tasks)
    2.  **Dataset** (Per `dataset_name`)
    3.  **Structure Type** (Per `dataset_name/stru_type`)
*   Preserve the existing output structure for Cleaned Data and Anomalies (grouped by dataset, not deeply nested).
*   Solve the "lost hierarchy" problem caused by Task Packing.

## 2. Problem Analysis
*   **Current State**: Tasks are generated into hierarchical paths (`inputs/dataset/type/task`) but are moved into flat bundles (`N_50_X/task`) for execution. This destroys the directory structure needed for easy statistical grouping.
*   **Gap**: `LabelingManager.collect_and_export` currently tries to guess dataset names from task paths but has no way to know the `stru_type` (cluster/bulk/layer) of a completed task.
*   **Constraint**: We should not deeply nest the final outputs (`outputs/cleaned` etc.) just for the sake of statistics, but we *must* track the statistics accurately.

## 3. Solution Strategy: Metadata Persistence

Instead of relying on directory structures (which are transient during execution), we will persist metadata in each task directory.

### 3.1 Metadata Generation (`generator.py`)
*   **Action**: Create a `task_meta.json` file in each task directory during generation.
*   **Content**:
    ```json
    {
        "dataset_name": "C20O0Fe0H0",
        "stru_type": "cluster",
        "task_name": "C20O0Fe0H0_0",
        "frame_idx": 0
    }
    ```
*   **Benefit**: This file travels with the task, regardless of where it is moved (packed to `N_50_X` or moved to `CONVERGED`).

### 3.2 Task Packing (`packer.py`)
*   **Action**: No change needed. Packing can safely move directories; metadata moves with them.

### 3.3 Result Processing (`manager.py`)
*   **Action**: When scanning `CONVERGED` (or active jobs), read `task_meta.json` to identify the task's lineage.
*   **Data Structure**: Build a registry of all tasks:
    ```python
    task_registry = {
        "task_name": {
            "dataset": "...",
            "type": "...",
            "status": "converged" | "failed" | "cleaned" | "filtered"
        }
    }
    ```

### 3.4 Data Cleaning & Statistics (`postprocess.py` & `manager.py`)
*   **Action**:
    1.  Load all converged systems.
    2.  Compute metrics and apply filters.
    3.  Tag each system in the DataFrame with its metadata (`dataset`, `type`).
    4.  Perform GroupBy aggregation to generate the required statistics table.
*   **Output**: Print a formatted table to logs and save as `labeling_stats.csv`.

### 3.5 Output Organization
*   **Action**: Maintain the existing logic where outputs are grouped by Dataset (`outputs/cleaned/dataset_name`). This aligns with Requirement 2.

## 4. Implementation Plan

### Phase 1: Metadata Injection (High Priority)
1.  **Modify `AbacusGenerator.generate`**:
    *   Accept `dataset_name` as an argument.
    *   Write `task_meta.json` into the task directory.
2.  **Update `LabelingManager.prepare_tasks`**:
    *   Pass `sys_name` (dataset name) to generator.

### Phase 2: Statistical Tracking Implementation
3.  **Modify `LabelingManager.collect_and_export`**:
    *   Initialize counters for: `Total`, `Converged`, `Cleaned`, `Filtered`.
    *   Traverse `CONVERGED` directories, reading `task_meta.json`.
    *   Load data into a DataFrame, ensuring `dataset` and `type` columns are populated from metadata.
    *   Apply filters and update counters.

### Phase 3: Reporting
4.  **Implement Reporting Logic**:
    *   Generate a hierarchical report:
        *   **Total Summary**
        *   **Dataset A Summary**
            *   Type 1 (Cluster): ...
            *   Type 2 (Bulk): ...
        *   **Dataset B Summary**...
    *   Log this report clearly.

### Phase 4: Output Verification
5.  **Verify Output Structure**:
    *   Ensure `outputs/cleaned` contains `dataset_name` subfolders.
    *   Ensure `outputs/anomalies` contains `dataset_name` subfolders.
    *   Ensure no `stru_type` folders are created in outputs (unless desired, but spec says keep it simple).

## 5. Timeline
*   **Step 1 & 2**: Immediate implementation (Files: `generator.py`, `manager.py`).
*   **Step 3 & 4**: Implementation within `collect_and_export` (File: `manager.py`).
*   **Verification**: Run on test dataset `fp-setting-2`.

## 6. Code Architecture Alignment
This plan respects the decoupling achieved in the previous refactor. `StructureAnalyzer` determines the type, `LabelingManager` orchestrates the flow, and `AbacusGenerator` persists the decision (metadata). The `PostProcessor` remains focused on data validation.
