# Implementation Plan - Refine Collection Workflow Visualization

This plan addresses the user's request to improve the aesthetics of PCA plots and correct the logic regarding training set selection visibility in the Collection Workflow.

## User Requirements
1.  **Color Refinement**:
    - Candidates: Orange (already changed to `#FFC000`, will verify/fine-tune).
    - Selected: Blue-Purple (previously Royal Blue `#4169E1`, request to shift towards "Blue-Purple" and "Make it more beautiful").
2.  **Logic Correction**:
    - `DIRECT_PCA_feature_coverage.png`: Currently missing the gray "X" markers for selected training data.
    - `Random_PCA_feature_coverage.png`: Annotation is confusing/incorrect regarding counts.
    - Cause: `sampling/manager.py` currently filters out training indices *before* returning them, preventing visualization of the full selection context.

## Proposed Changes

### 1. `src/dpeva/sampling/manager.py`
-   **Modify `_run_direct`**:
    -   Remove the code block that filters `selected_indices` based on `n_candidates`.
    -   Allow `selected_indices` to contain indices $\ge$ `n_candidates` (i.e., training data).
    -   This ensures `_calculate_sampling_stats` uses the full selection count for the Random baseline, making the comparison fair and the counts consistent with the total effort.
    -   Add a log message to inform how many training samples were selected (for observability).

### 2. `src/dpeva/uncertain/visualization.py`
-   **Update Color Palette**:
    -   `style_cand`: Keep `#FFC000` (Orange) but adjust alpha/size if needed for better "beauty".
    -   `style_sel_new`: Change from `#4169E1` (RoyalBlue) to `#6A5ACD` (SlateBlue) or `#7B68EE` (MediumSlateBlue) to better match "Blue-Purple". Let's use **`#6A5ACD` (SlateBlue)**.
-   **Verify Plotting Logic**:
    -   The existing `_plot_coverage` method already has logic to plot `Selected (Train)` with gray "X" markers. Once `sampling/manager.py` returns full indices, this will work automatically.

### 3. Verification
-   Run the `collection` workflow (or a mocked test) to verify that:
    -   `DIRECT_PCA_feature_coverage.png` now shows gray "X"s.
    -   `Random_PCA_feature_coverage.png` shows a mix of candidates and training data consistent with the total count.
    -   `Final_sampled_PCAview.png` uses the new Blue-Purple color.
    -   The `final_df.csv` (export list) still *only* contains candidates (handled by `workflows/collect.py` logic which re-filters for export).

## Execution Steps

1.  **Edit `src/dpeva/sampling/manager.py`**:
    -   Locate `_run_direct` method.
    -   Comment out or remove the filtering logic: `selected_indices = [idx for idx in selected_indices if idx < self.n_candidates]`.
    -   Add logging to report `len(selected_indices)` vs `n_candidates`.

2.  **Edit `src/dpeva/uncertain/visualization.py`**:
    -   Update `style_sel_new` color to `#6A5ACD`.
    -   Ensure `style_cand` is `#FFC000`.

3.  **Review `src/dpeva/workflows/collect.py`**:
    -   Confirm that `final_indices = [idx for idx in selected_indices if idx < n_candidates]` is present and correct (it is, based on read).
