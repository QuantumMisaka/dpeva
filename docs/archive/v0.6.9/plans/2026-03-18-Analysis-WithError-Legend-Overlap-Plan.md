# Analysis With-Error 图例重叠修复计划

## Summary

- 目标：修复 Analysis 模块 `dist_energy_with_error.png`、`dist_force_magnitude_with_error.png`、`dist_cohesive_energy_with_error.png` 三张图中“右侧 Error Distribution 区域与左侧分布图图例重叠”的问题。
- 约束：仅通过参数与布局调整完成修复，不改动统计语义与数据处理逻辑。
- 验证：使用 `test/s-head-huber/config_analysis.json` 重新运行 analysis，确认目标图重新生成且布局正常。

## Current State Analysis

### 1) 复现输入与目标产物

- 配置文件存在且可用：`test/s-head-huber/config_analysis.json`。
- 目标产物已存在：`test/s-head-huber/test_analysis/dist_*_with_error.png`。

### 2) 代码定位

- 问题核心在 `src/dpeva/inference/visualizer.py` 的 `plot_distribution_with_error(...)`：
  - 当前左图 legend 使用 `loc="upper left", bbox_to_anchor=(1.02, 1.0)`，将图例锚定到主图坐标轴右外侧；
  - 两栏布局间距 `wspace=0.25` 偏小，导致 legend 进入右侧 error 子图区域，产生可见重叠。
- 相关调用由 `src/dpeva/analysis/managers.py` 的 `plot_distribution_family(...)` 触发，无需改调用链即可修复。

### 3) 风险评估

- 改动仅限绘图参数，风险低；
- 可能影响其他 with-error 图（如 virial），属于正向一致性优化。

## Proposed Changes

### A. 调整 with-error 图布局参数

- 文件：`src/dpeva/inference/visualizer.py`
- 函数：`plot_distribution_with_error(...)`
- 变更内容：
  - 将左图 legend 从“轴外右侧”改为“轴内右上角”（避免进入右侧子图区域）；
  - 增加 `ax_main` 与 `ax_err` 间距（`GridSpec wspace` 上调）；
  - 适度收紧 `tight_layout` 的 `rect`，确保右侧 error 图有稳定留白；
  - 保持统计框位置在左下区域，不与主峰与图例冲突。
- 结果预期：
  - `dist_*_with_error.png` 的 legend 与 Error Distribution 面板彻底分离；
  - 不改变统计项、不改变曲线/直方图数据。

### B. 保持现有调用接口与行为兼容

- 文件：`src/dpeva/analysis/managers.py`
- 说明：不改参数签名与调用方式，仅复用改进后的 visualizer 布局参数。

### C. 执行回归分析并核对产物

- 使用命令：`dpeva analysis test/s-head-huber/config_analysis.json`
- 核对项：
  - 命令退出码为 0；
  - 目标三张图更新成功（文件时间戳/存在性）；
  - 视觉检查确认 legend 不再与右侧 Error Distribution 区域重叠。

## Assumptions & Decisions

- 假设当前 `config_analysis.json` 指向的数据与结果目录可直接复用，无需额外准备。
- 决策：优先采用“legend 内收 + 子图间距增大”的稳定方案，而不是仅微调 bbox 偏移量，避免不同数据分布下再次重叠。
- 决策：本次不改动 `plot_distribution_overlay` 与单图分布函数，聚焦用户指定问题范围。

## Verification Steps

1. 运行 `pytest tests/unit/inference/test_visualizer.py tests/unit/workflows/test_analysis_workflow.py`，确认绘图与 analysis 工作流关键测试通过。  
2. 运行 `dpeva analysis test/s-head-huber/config_analysis.json` 完整回归。  
3. 检查产物：
   - `test/s-head-huber/test_analysis/dist_energy_with_error.png`
   - `test/s-head-huber/test_analysis/dist_force_magnitude_with_error.png`
   - `test/s-head-huber/test_analysis/dist_cohesive_energy_with_error.png`
4. 记录执行日志与结果，确认无新增异常、无布局重叠。
