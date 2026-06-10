---
title: Enhanced Parity Plot Layout Polish Plan
status: active
audience: developers
last-updated: 2026-03-18
owner: DP-EVA Maintainers
---

# Enhanced Parity Plot 布局与标注优化计划

- Status: active
- Audience: developers
- Last-Updated: 2026-03-18

## Summary

- 目标图：
  - `test/s-head-huber/test_analysis/parity_cohesive_energy_enhanced.png`
  - `test/s-head-huber/test_analysis/parity_energy_enhanced.png`
  - `test/s-head-huber/test_analysis/parity_force_enhanced.png`
  - `test/s-head-huber/test_analysis/parity_virial_enhanced.png`
- 核心问题：
  - 右上角 Error Distribution 子图与相邻子图坐标数字发生视觉冲突，标题高度也与主标题竞争，层级不合理。
  - `True Density`/`Predicted Density` 的文本框样式突兀，位置不稳定，造成“标签像悬浮贴纸”的视觉噪声。
- 约束：只优化信息设计与布局，不更改误差定义、统计口径、分析流程契约。

## Current State Analysis

- 代码入口：`src/dpeva/inference/visualizer.py::plot_parity_enhanced(...)`。
- 当前四区布局：
  - 上左：True 边缘分布（红色）
  - 下左：Parity 主图
  - 下右：Predicted 边缘分布（蓝色）
  - 上右：Error Distribution（橙色）
- 现状问题细化：
  - 上右 error 子图的 x 轴刻度与标题和右列下方子图顶部文本区发生紧邻/重叠感；
  - 右列上下两图都显示较多轴元素，导致“信息密度集中在窄列”；
  - True/Predicted 标签使用彩色边框文本框，与数据层视觉风格不一致，且在不同量纲下位置显得不自然。

## Proposed Changes

### 1) 重构标签表达方式（去贴纸化）

- 文件：`src/dpeva/inference/visualizer.py`
- 函数：`plot_parity_enhanced(...)`
- 方案：
  - 去除 `True Density`/`Predicted Density` 的大文本框叠加。
  - 改用“轴标题语义化”：
    - 顶部子图标题：`True Density`
    - 右侧子图标题：`Predicted Density`
  - 保持颜色映射（红=True，蓝=Predicted），让“文字 + 颜色 + 轴位置”三重信息一致。
- 预期收益：
  - 读者无需解释框即可理解语义；
  - 标签不再遮挡数据，不再产生突兀视觉块。

### 2) 重新定义 Error Distribution 子图信息层级

- 文件：`src/dpeva/inference/visualizer.py`
- 函数：`plot_parity_enhanced(...)`
- 方案：
  - Error 子图标题降级为较小字号，并减小 `pad`，避免与主标题竞争。
  - Error 子图只保留必要坐标信息：
    - x 轴保留 `Error (unit)`，但减少刻度数量（建议 `MaxNLocator`）；
    - y 轴不显示 label，仅保留稀疏刻度或直接隐藏刻度文本，避免与左侧密度图的 y 轴信息抢夺注意力。
  - 上右子图与下右子图之间增加纵向留白（适度增大 `hspace`），解除上下轴文本碰撞。
  - 保留与 `dist_*_with_error` 一致的核心逻辑元素：`hist + kde`、`error = pred - true`、`x=0` 虚线。
- 预期收益：
  - Error 子图成为“补充诊断信息”，而不是主视觉冲突源；
  - 数值表达保留可读性，同时降低窄列的标注拥挤。

### 3) 统一右列数值显示策略（适配多量纲）

- 文件：`src/dpeva/inference/visualizer.py`
- 函数：`plot_parity_enhanced(...)`
- 方案：
  - 对右列两个子图启用一致的刻度压缩策略（少量主刻度 + 合理字号）。
  - 对跨度极大的量（如 energy）允许科学计数格式；对常规量（force/virial）保持普通格式。
  - 明确“优先显示趋势，不追求每个子图全量刻度”。
- 预期收益：
  - 同一模板可稳定适配 cohesive/energy/force/virial 四类尺度，不再在个别图上爆发重叠。

### 4) 回归验证

- 单测：
  - `pytest tests/unit/inference/test_visualizer.py`
- 实图回归：
  - `dpeva analysis test/s-head-huber/config_analysis.json`
- 验收标准：
  - Error 子图坐标数字与其他子图无重叠/挤压感；
  - Error 标题显著低于主标题层级；
  - True/Predicted 语义传达自然、稳定，不再出现突兀文本框；
  - 四张 `parity_*_enhanced.png` 在不同量纲下均保持一致风格与可读性。

## Assumptions & Decisions

- 假设 `test/s-head-huber/config_analysis.json` 可直接复用做回归。
- 决策：采用“轴语义 + 轻量刻度”的最小干预方案，优先解决视觉冲突。
- 决策：Error 子图在“逻辑一致”前提下允许做视觉层级压缩（标题/刻度精简），以服务整体可读性。
