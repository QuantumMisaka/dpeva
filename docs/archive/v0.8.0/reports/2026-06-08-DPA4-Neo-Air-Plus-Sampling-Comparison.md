---
title: DPA4 Neo / Air / Plus Sampling Comparison
status: archived
audience: Developers / Researchers
last-updated: 2026-06-25
owner: DPEVA Maintainers
series: Sampling Comparison
related:
  - 2026-06-07-DPA4-Mini-MACE-Sampling-Comparison.md
---

# DPA4 Neo / Air / Plus 采样对比报告

- Date: 2026-06-08
- Status: archived
- Owner: DPEVA Maintainers
- Series: Sampling Comparison
- Related Reports: [2026-06-07-DPA4-Mini-MACE-Sampling-Comparison.md](2026-06-07-DPA4-Mini-MACE-Sampling-Comparison.md)
- Result Artifact: [DPA4_NEO_AIR_PLUS_SAMPLING_COMPARISON.html](../../../reports/assets/2026-06-08-dpa4-neo-air-plus/DPA4_NEO_AIR_PLUS_SAMPLING_COMPARISON.html)

## 1. 结论摘要

本报告归档 DPA4-Neo、DPA4-Air、DPA4-Plus 在同一 DP-EVA 采样框架下的互相比对结果。三者都能完成全链路采样，但行为差异很明显：

- Neo 采样最紧，`354` 帧、`176` 个 system，descriptor 维数 `32`，覆盖最保守。
- Air 采样更宽，`440` 帧、`192` 个 system，覆盖与 novelty 都高于 Neo。
- Plus 选得最多，`523` 帧、`224` 个 system，`64` 维 descriptor 下 retained variance 高达 `97.60%`，但训练 descriptor 提取存在 `177 -> 173` 的可用数下降，说明它的训练侧更容易触发资源边界。

从采样结果看，Plus 的覆盖抬升最明显，Neo 最保守，Air 介于两者之间。

## 2. 目的与范围

本报告用于长期归档以下内容：

1. DPA4 Neo / Air / Plus 的特征提取、direct sampling、collect 汇总结果；
2. 三者在同一候选池上的 sample rate、PCA 主导性、coverage、novelty、diversity、selection overlap；
3. 供后续 DPA 模型选择时参考。

## 3. 方法与数据

统一输入：

- training set: `test/dpa4-dpeva-test/sampled_dpdata`
- candidate pool: `test/dpa4-dpeva-test/other_dpdata`
- UQ candidate: 各自模型对应的已完成结果

统一采样口径：

- `batch_size filter:128`
- 其他参数与 dpeva 内置模板保持一致
- 对比的均为 DPA4 系列模型在同一任务上的采样行为

描述符维数：

- DPA4 Neo: `32`
- DPA4 Air: `64`
- DPA4 Plus: `64`

## 4. 结果与分析

### 4.1 核心指标

| 模型 | descriptor dim | uq candidate frames | selected frames | sampled systems | important PCs | retained variance | PC1/PC2 ratio | coverage selected mean | coverage random mean | novelty to train mean | selected nn diversity mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Neo | 32 | 21254 | 354 | 176 | 6 | 87.74% | 2.05 | 0.8832 | 0.7564 | 1.1889 | 1.6200 |
| Air | 64 | 19728 | 440 | 192 | 8 | 91.57% | 3.82 | 0.7801 | 0.7536 | 1.9681 | 2.0515 |
| Plus | 64 | 17852 | 523 | 224 | 4 | 97.60% | 3.67 | 0.8000 | 0.7276 | 1.1689 | 0.8991 |

### 4.2 采样行为差异

- Neo 的采样数量最少，说明其 descriptor 更紧凑，直接导致 direct 侧更早收敛。
- Air 的 novelty 和 diversity 最高，说明它选出的样本更“远”，更偏向探索。
- Plus 的 selected frames 最多，说明它在当前任务下更愿意保留更多候选点，但其 selected nn diversity 明显更低，说明新增样本之间彼此更接近。
- Plus 的 retained variance 最高，说明前三四个主成分几乎就能解释大多数 descriptor 变化，结构空间更集中。

### 4.3 选择重叠

| 对比 | intersection | union | Jaccard |
|---|---:|---:|---:|
| Neo vs Plus | 16 | 861 | 0.018583 |
| Air vs Neo | 24 | 770 | 0.031169 |
| Air vs Plus | 29 | 934 | 0.031049 |

重叠度都很低，说明三者不是同一批样本的微调排序，而是直接采样决策不同。

## 5. 附件与可追溯结果

- HTML 报告: [DPA4_NEO_AIR_PLUS_SAMPLING_COMPARISON.html](../../../reports/assets/2026-06-08-dpa4-neo-air-plus/DPA4_NEO_AIR_PLUS_SAMPLING_COMPARISON.html)
- 指标汇总: [neo_air_plus_sampling_metrics.csv](../../../reports/assets/2026-06-08-dpa4-neo-air-plus/neo_air_plus_sampling_metrics.csv)
- 选择重叠: [neo_air_plus_selection_overlap.csv](../../../reports/assets/2026-06-08-dpa4-neo-air-plus/neo_air_plus_selection_overlap.csv)
- PCA 表: [neo_air_plus_pca_table.csv](../../../reports/assets/2026-06-08-dpa4-neo-air-plus/neo_air_plus_pca_table.csv)
- pool 计数: [neo_air_plus_pool_counts.csv](../../../reports/assets/2026-06-08-dpa4-neo-air-plus/neo_air_plus_pool_counts.csv)
- UQ 汇总: [neo_air_plus_uq_summary.csv](../../../reports/assets/2026-06-08-dpa4-neo-air-plus/neo_air_plus_uq_summary.csv)

## 6. 建议

如果后续要在 DPA4 系列里选代表性数据筛选模型，可以先按这个顺序看：

1. 追求更强探索性，优先看 Air；
2. 追求更高覆盖和更大采样量，优先看 Plus；
3. 追求更保守、更紧凑的筛选，Neo 更合适。
