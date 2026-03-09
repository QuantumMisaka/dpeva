# Labeling Workflow Refactoring & Optimization Plan (Test-4 Analysis)

## 1. Issue Analysis

### 1.1 Double Counting of Failed Tasks
*   **Symptom**: `Failed: 28` when `Total Tasks: 26`.
*   **Root Cause**: `process_results` uses `job_dir.rglob("INPUT")` to identify tasks. ABACUS writes a backup `INPUT` file into its output directory (`OUT.ABACUS/INPUT`). Thus, each failed task (which retains its `OUT.ABACUS` folder) is counted twice: once for the root `INPUT` and once for the output backup.
*   **Fix Strategy**: Modify the recursive scan to exclude `INPUT` files located inside `OUT.*` directories.

### 1.2 Result Collection Failure (`RuntimeError`)
*   **Symptom**: `RuntimeError: Object must be System or MultiSystems!` during `collect_and_export`.
*   **Root Cause**: `dpdata.MultiSystems` constructor does not accept a list of systems.
*   **Fix Strategy**: Initialize an empty `MultiSystems` and loop `append()` each loaded system. (Note: This was partially addressed in previous turn but needs confirmation in final code).

### 1.3 Potential Directory Conflicts
*   **Symptom**: Potential naming collisions if different datasets contain systems with identical names (e.g., `C20O0Fe0H0_0` in both Dataset A and Dataset B).
*   **Root Cause**: Currently, tasks are flattened into `inputs/N_50_X/task_name`. If `task_name` is just `sysname_index`, collisions are possible if `sysname` is not unique across datasets.
*   **Analysis**:
    *   `prepare_tasks` generates `task_name = f"{sys_name}_{f_idx}"`.
    *   `sys_name` comes from `system.short_name` or `sys_{i}`.
    *   In a multi-pool scenario, if two pools have the same system name, they will collide.
    *   Also, `CONVERGED` directory flattens or semi-flattens tasks.
*   **Fix Strategy**:
    *   Enforce `task_name` uniqueness by prefixing with `dataset_name` if available, or ensuring `sys_name` is unique globally.
    *   In `CONVERGED`, maintain `dataset/type/task` hierarchy (already planned via metadata restoration).

### 1.4 Resubmission Optimization (Async Retry)
*   **Proposal**: Resubmit failed tasks immediately within their sub-job (`N_X_Y`) instead of waiting for *all* jobs to finish.
*   **Analysis**:
    *   **Pros**: Faster turnaround for heterogeneous clusters where some jobs finish early.
    *   **Cons**:
        *   Drastically complicates the `LabelingWorkflow` loop. Currently, it's synchronized: Submit All -> Wait All -> Check All.
        *   Moving to async requires tracking state per-job-bundle.
        *   Slurm dependencies or job arrays could handle this, but our `JobManager` is generic.
    *   **Verdict**: **Deferred**. The complexity risk outweighs the benefit for the current phase. We will stick to the synchronized "Epoch" approach (Attempt 0 -> Attempt 1) for stability. Optimization can be a future feature.

### 1.5 Statistics & Hierarchy (Dataset/Type)
*   **Requirement**: Accurate counts per Dataset and Structure Type.
*   **Strategy**:
    *   The **Metadata Injection** plan (implemented in previous turn) is the correct approach.
    *   `task_meta.json` travels with the task.
    *   `collect_and_export` reads this metadata to aggregate stats.
    *   **Refinement**: Ensure `collect_and_export` also restores the `CONVERGED` directory structure to `CONVERGED/dataset/type/task` for clarity and collision avoidance.

## 2. Implementation Plan

### 2.1 Fix Task Counting (`manager.py`)
Modify `process_results` loop:
```python
for input_file in job_dir.rglob("INPUT"):
    # Fix: Skip if parent is an output directory
    if "OUT." in input_file.parent.name:
        continue
    task_dir = input_file.parent
    # ...
```

### 2.2 Fix Collection Logic (`manager.py`)
Ensure `collect_and_export` uses the loop-append pattern for `dpdata.MultiSystems`.

### 2.3 Collision Prevention & Structure Restoration
1.  **Unique Naming**: In `prepare_tasks`, ensure `task_name` includes dataset info if not already present. (Current: `sys_name_idx`, where `sys_name` is usually dataset name).
2.  **CONVERGED Restoration**:
    *   In `collect_and_export` (or `process_results`), when moving to `CONVERGED`, use the metadata (`dataset`, `type`) to create the target path: `CONVERGED/{dataset}/{type}/{task_name}`.
    *   This physically isolates potentially conflicting names (assuming `dataset` names are unique).

### 2.4 Final Output Structure
*   `outputs/cleaned/{dataset}/...`
*   `outputs/anomalies/{dataset}/...`
*   No change needed here, aligns with requirements.

## 3. Verification Steps
1.  **Clean Run**: Delete `labeling_test4` and `labeling_test4_output`.
2.  **Execute**: Run `dpeva label config_gpu.json`.
3.  **Check 1**: Log shows correct `Total Tasks` and `Failed` counts (no double counting).
4.  **Check 2**: `CONVERGED` directory has `dataset/type/task` hierarchy.
5.  **Check 3**: Final report in log shows correct breakdown by Dataset and Type.
6.  **Check 4**: Output data is exported without `RuntimeError`.

## 4. Documentation
Update `docs/specs/labeling_test4_review.md` with these findings and the executed plan.
