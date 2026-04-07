# Collection PCA Visualization Spec

## Why
Collection Workflow 中的 PCA 出图 (`Final_sampled_PCAview_by_pool.png` 和 `DIRECT_PCA_feature_coverage.png`) 字号偏小，达不到直接用于论文发表的要求。此外，在单数据池 (Single Pool) 场景下，并不需要按池来源来区分展示数据的来源，因此不应当生成 `Final_sampled_PCAview_by_pool.png` 这张图。

## What Changes
- 修改 PCA 图的字体配置 profile，将图表标题、轴标题和轴刻度数字的字号各增大约 2pt，同时将图例 (legend label) 的字号与轴标题的字号对齐。
- 在 `UQVisualizer._plot_joint_multipool_summary` 和 `CollectionWorkflow._run_sampling_phase` 中增加逻辑判断：仅当候选数据集 (`df_candidate`) 中存在多个数据池来源时，才生成并记录 `Final_sampled_PCAview_by_pool.png`。单池模式下跳过该图。

## Impact
- Affected specs: Collection Workflow 可视化及日志审计模块
- Affected code:
  - `src/dpeva/utils/visual_style.py`
  - `src/dpeva/uncertain/visualization.py`
  - `src/dpeva/workflows/collect.py`

## ADDED Requirements
### Requirement: Conditional Multi-pool Summary Plot
The system SHALL only generate `Final_sampled_PCAview_by_pool.png` when the input candidate data originates from more than one distinct data pool.
#### Scenario: Single pool sampling
- **WHEN** user runs collection workflow with candidates from a single pool
- **THEN** `Final_sampled_PCAview_by_pool.png` is not generated and is logged as skipped with reason `single_pool_detected`.

## MODIFIED Requirements
### Requirement: PCA Plot Font Scaling
The system SHALL use an updated typography scale for PCA scatter plots:
- Title scale: ~1.73 (increases base 15.6 by ~2pt)
- Label scale: ~1.65 (increases base 12.96 by ~2pt)
- Tick scale: ~1.57 (increases base 11.52 by ~2pt)
- Legend scale: ~1.90 (aligns legend font size with label font size)
