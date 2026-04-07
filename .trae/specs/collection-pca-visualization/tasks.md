# Tasks
- [x] Task 1: Update PCA typography scales in visual_style.py
  - [x] SubTask 1.1: Change `title_scale` to 1.73 in `get_collection_pca_scatter_profile`
  - [x] SubTask 1.2: Change `label_scale` to 1.65
  - [x] SubTask 1.3: Change `tick_scale` to 1.57
  - [x] SubTask 1.4: Change `legend_scale` and `legend_title_scale` to 1.90
- [x] Task 2: Implement single-pool check in visualization.py
  - [x] SubTask 2.1: In `_plot_joint_multipool_summary`, check `df_uq["dataname"]` for unique pools. If `<= 1`, return early without generating the plot.
- [x] Task 3: Update logging logic in collect.py
  - [x] SubTask 3.1: In `_run_sampling_phase`, calculate the number of unique pools from `df_candidate["dataname"]`.
  - [x] SubTask 3.2: Update the `if use_joint:` block to `if use_joint and is_multipool:` and log `Final_sampled_PCAview_by_pool` as generated.
  - [x] SubTask 3.3: Update the `else` block to log the plot as skipped, providing the correct reason (`joint_mode_disabled` or `single_pool_detected`).

# Task Dependencies
- [Task 3] depends on [Task 2]
