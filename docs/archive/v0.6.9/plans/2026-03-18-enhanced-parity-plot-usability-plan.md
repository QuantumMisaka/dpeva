***

title: Enhanced Parity Plot Usability Plan
status: active
audience: developers
last-updated: 2026-03-18
owner: DP-EVA Maintainers
-------------------------

# Enhanced Parity Plot 可读性改进计划

- Status: active
- Audience: developers
- Last-Updated: 2026-03-18

## Summary

- 改进对象：
  - `test/s-head-huber/test_analysis/parity_cohesive_energy_enhanced.png`
  - `test/s-head-huber/test_analysis/parity_energy_enhanced.png`
  - `test/s-head-huber/test_analysis/parity_force_enhanced.png`
  - `test/s-head-huber/test_analysis/parity_virial_enhanced.png`
- 目标：提升 enhanced parity 图可读性，明确边缘密度分布的语义，并利用右上空白区新增误差分布缩略图。
- 约束：仅调整可视化表达与布局，不改变误差定义、统计口径与分析工作流输入输出契约。

## Current State Analysis

- 现有实现位于 `src/dpeva/inference/visualizer.py` 的 `plot_parity_enhanced(...)`。
- 当前布局使用三块区域：
  - 顶部 density：来源于 `y_true`（未显式标注 True）；
  - 右侧 density：来源于 `y_pred`（未显式标注 Predicted）；
  - 主区 parity scatter + 对角虚线。
- 当前问题：
  - 用户不能直观看出顶部/右侧 density 分别对应哪个标签；
  - 右上角区域闲置，信息承载不足。
- 参考一致性要求：
  - 新增误差子图的作图逻辑需与 `dist_*_with_error.png` 右侧 error panel 一致（error = predicted - true、密度直方图+KDE、x=0 虚线）。

## Proposed Changes

### 1) 明确边缘 density 标签语义

- 文件：`src/dpeva/inference/visualizer.py`
- 函数：`plot_parity_enhanced(...)`
- 方案：
  - 顶部 density 区域添加显式标签 `True Density`（配色与 true 数据一致）；
  - 右侧 density 区域添加显式标签 `Predicted Density`（配色与 predicted 数据一致）；
  - 保持 parity 主图坐标标签语义不变（x=True，y=Predicted）。

### 2) 在右上角增加缩小版 Error Distribution

- 文件：`src/dpeva/inference/visualizer.py`
- 函数：`plot_parity_enhanced(...)`
- 方案：
  - 启用右上子图轴（当前空白区域）；
  - 误差定义：`error = y_pred_valid - y_true_valid`；
  - 采用与 `plot_distribution_with_error(...)` 右侧 error 图一致的视觉逻辑：
    - `sns.histplot(..., kde=True, stat="density", element="step", color=#f59e0b, alpha≈0.28)`；
    - `axvline(0.0, linestyle='--', color=#374151)`；
    - 网格与坐标风格保持统一。

### 3) 布局与可读性微调

- 文件：`src/dpeva/inference/visualizer.py`
- 函数：`plot_parity_enhanced(...)`
- 方案：
  - 适度微调 `GridSpec` 的 `wspace/hspace/ratios`，确保新增子图后主图不拥挤；
  - 右上误差子图使用较小字号与简化刻度，避免视觉抢占；
  - 维持主图为优先信息载体。

### 4) 回归测试与实图验证

- 文件：`tests/unit/inference/test_visualizer.py`
- 方案：
  - 保持既有 enhanced parity 生成测试；
  - 增补最小回归断言（函数可执行、目标文件存在）。
- 回归运行：
  - `pytest tests/unit/inference/test_visualizer.py`
  - `dpeva analysis test/s-head-huber/config_analysis.json`
- 验收标准：
  - 四张 `parity_*_enhanced.png` 中，顶部/右侧 density 能直观区分 True 与 Predicted；
  - 右上角新增缩小版 Error Distribution，且绘图逻辑与 `dist_*_with_error` error panel 一致；
  - parity 主图仍保持清晰，不出现明显遮挡或拥挤。

## Assumptions & Decisions

- 假设 `test/s-head-huber/config_analysis.json` 指向的数据与 result\_dir 在当前环境可直接回归。
- 决策：本轮先采用固定增强行为，不新增配置项，优先达成一致可视化规范。
- 决策：误差子图基本逻辑（除图尺寸以外）与现有 `with_error` 系列严格对齐，降低跨图认知成本。

