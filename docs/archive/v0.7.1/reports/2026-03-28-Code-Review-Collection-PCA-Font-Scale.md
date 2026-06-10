---
title: Collection PCA Font Scale Review
status: active
audience: developers
last-updated: 2026-03-28
owner: Docs Owner
---

# Collection PCA 图字号呈现差异分析

## 1. 背景

用户在对比 `Final_sampled_PCAview.png` 与 `DIRECT_PCA_feature_coverage.png` 时观察到：尽管两图中的 PCA 点云分布看起来相近，但前者的标题、轴标题与刻度数字视觉上更小。

本报告聚焦确认这一观感差异是否真实来自“字号设置不同”，并给出可复核的代码与输出证据。

## 2. 结论摘要

- 结论一：两张图**并不是因为 Final_sampled_PCAview 的字体配置更小**而显得更小。
- 结论二：`Final_sampled_PCAview` 的标题、轴标题和刻度字号在代码中实际上**略大于** `DIRECT_PCA_feature_coverage`。
- 结论三：用户感知到的“字号更小”，主要来自**更大的画布尺寸、未压缩的默认边距、以及图内元素相对占比下降**。
- 结论四：这是一个**相对版式密度问题**，不是 DPI、像素分辨率或字体参数退化问题。

## 3. 代码路径确认

### 3.1 共用入口

- 两张图都由 `src/dpeva/workflows/collect.py` 中的 `CollectionWorkflow.run` 创建同一个 `UQVisualizer`。
- `UQVisualizer` 在 `src/dpeva/workflows/collect.py` 中注入统一的 `fig_dpi`、`fig_base_font_size`、`fig_tick_target_count`、`fig_legend_max_rows`、`fig_legend_max_cols`。
- 默认配置来源于 `src/dpeva/config.py` 中的 `CollectionConfig`。

### 3.2 各自生成位置

- `Final_sampled_PCAview.png` 在 `src/dpeva/uncertain/visualization.py` 的 `plot_pca_analysis` 中直接绘制。
- `DIRECT_PCA_feature_coverage.png` 在 `src/dpeva/uncertain/visualization.py` 的 `plot_pca_analysis` 调用 `_plot_coverage` 后生成。

## 4. 字号配置对比

### 4.1 统一基线

- 全局样式入口为 `src/dpeva/utils/visual_style.py` 中的 `set_visual_style`。
- 基础字体层级由 `src/dpeva/utils/visual_style.py` 中的 `get_publication_font_hierarchy` 给出。
- 当 `fig_base_font_size=12` 时，基础层级为：
  - title = 15.60
  - label = 12.96
  - tick = 11.52
  - legend = 11.28

### 4.2 Final_sampled_PCAview 使用的局部字体

- `Final_sampled_PCAview` 使用 `src/dpeva/uncertain/visualization.py` 中的 `_build_pca_fonts`。
- 其倍率为：
  - title × 1.24
  - label × 1.20
  - tick × 1.20
- 换算后约为：
  - title = 19.34
  - label = 15.55
  - tick = 13.82

### 4.3 DIRECT_PCA_feature_coverage 使用的局部字体

- `DIRECT_PCA_feature_coverage` 使用 `src/dpeva/uncertain/visualization.py` 中的 `_build_coverage_fonts`。
- 其倍率为：
  - title × 1.20
  - label × 1.16
  - tick × 1.16
- 换算后约为：
  - title = 18.72
  - label = 15.03
  - tick = 13.36

### 4.4 字号层面的直接结论

- `Final_sampled_PCAview` 的 title / label / tick 实际值都**略大于** `DIRECT_PCA_feature_coverage`。
- 因此，“Final_sampled_PCAview 看起来更小”的直接原因**不是字号设置更小**。

## 5. 画布与输出尺寸对比

### 5.1 figsize

- `Final_sampled_PCAview` 使用 `figsize=(12, 10)`，定义位于 `src/dpeva/uncertain/visualization.py`。
- `DIRECT_PCA_feature_coverage` 使用 `figsize=(10, 8)`，定义位于 `src/dpeva/uncertain/visualization.py`。

### 5.2 实际像素

基于本次验收读取 PNG 元数据，300 DPI 下两图输出分别为：

- `Final_sampled_PCAview.png` = `3600 × 3000`
- `DIRECT_PCA_feature_coverage.png` = `3000 × 2400`

这说明：

- `Final_sampled_PCAview` 的像素并不更小，反而更大。
- 问题不在分辨率，而在**元素相对于更大画布的占比变小**。

## 6. 为何会产生“看起来更小”的观感

### 6.1 画布放大幅度大于字体放大幅度

- `Final_sampled_PCAview` 的画布面积为 `12 × 10 = 120 in²`
- `DIRECT_PCA_feature_coverage` 的画布面积为 `10 × 8 = 80 in²`
- 前者比后者面积大 **50%**

但字体增幅只有小幅提升：

- title：约从 `18.72` 提高到 `19.34`
- label：约从 `15.03` 提高到 `15.55`
- tick：约从 `13.36` 提高到 `13.82`

因此，**字体没有按画布扩张比例同步变大**，导致内容在整张图中的相对占比下降。

### 6.2 默认边距在更大画布上被同步放大

两张图保存时都没有额外使用：

- `tight_layout()`
- `bbox_inches="tight"`
- 定制 `subplots_adjust(...)`

可见于：

- `Final_sampled_PCAview` 的保存逻辑位于 `src/dpeva/uncertain/visualization.py`
- `DIRECT_PCA_feature_coverage` 的保存逻辑位于 `src/dpeva/uncertain/visualization.py`

这意味着：

- 更大的 `12×10` 画布会保留更多默认留白
- 标题、坐标轴、刻度与点云在画面中的“占屏比例”进一步下降

### 6.3 图内内容密度不同

- `DIRECT_PCA_feature_coverage` 使用训练集背景、候选集、训练侧选中点、新选中点四层叠加，相关实现位于 `src/dpeva/uncertain/visualization.py` 的 `_plot_coverage`。
- `Final_sampled_PCAview` 使用全体背景、候选、最终采样三层，相关实现位于 `src/dpeva/uncertain/visualization.py` 的 `plot_pca_analysis`。

虽然两图的主分布骨架接近，但 coverage 图：

- 图层更多
- marker 形状差异更强
- 主图填充度更高

这会进一步强化“coverage 图更满、更大”的视觉印象。

### 6.4 图例位置对主观感受有次级影响

- 两图当前都使用内嵌图例 `loc="best"`。
- `Final_sampled_PCAview` 的图例位于右上，覆盖的是较为空的区域，但同时也让读者更注意整张图的留白感。
- `DIRECT_PCA_feature_coverage` 图例位于右下，其周围本身已有更密集的点层，主观上更容易感受到“图内容铺得更满”。

这不是主因，但会放大已有的版式差异。

## 7. 最终判定

`Final_sampled_PCAview.png` 之所以看起来比 `DIRECT_PCA_feature_coverage.png` 的轴标题和刻度更小，**根本原因不是字号参数更小，而是版式密度更低**。更具体地说：

1. 它的画布更大；
2. 字号虽然略大，但没有按画布面积同步放大；
3. 默认边距没有压缩；
4. 图内图层更少、视觉填充度更低；
5. 最终导致同样级别的文字在整图中的相对占比下降。

## 8. 建议

- 若目标是让 `Final_sampled_PCAview` 与 `DIRECT_PCA_feature_coverage` 在论文中呈现更一致的字号观感，优先建议：
  - 进一步增大 `Final_sampled_PCAview` 的局部 title / label / tick 比例；
  - 或适度减小 `Final_sampled_PCAview` 的 `figsize`；
  - 或引入 `tight_layout()` / 局部 `subplots_adjust()` 压缩默认留白。
- 若目标是严格保持两图同等视觉密度，则需要把“字体策略”和“画布策略”联动，而不是单独只调字号。
