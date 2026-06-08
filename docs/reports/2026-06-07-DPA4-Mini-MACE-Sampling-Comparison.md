# DPA4 Mini / MACE small / MACE medium 采样对比报告

- Date: 2026-06-07
- Status: active
- Owners: DPEVA Maintainers
- Related: [MACE_DPA4_MINI_SAMPLING_COMPARISON.html](assets/2026-06-07-dpa4-mini-mace/MACE_DPA4_MINI_SAMPLING_COMPARISON.html)

## 1. 结论摘要

本报告整理了基于同一 DPA4 Mini UQ candidate 集合、同一 DP-EVA direct sampling 参数、同一训练/候选数据标尺下的三组采样结果：DPA4 Mini、MACE small、MACE medium。

结论很直接：

- DPA4 Mini 选得最多，`359` 帧、`181` 个 system，覆盖更宽，但在其自身 descriptor 空间里 retained variance 只有 `84.69%`。
- MACE small 采样最紧，`87` 帧、`76` 个 system，但 DIRECT coverage 反而略高于 Mini，说明它在更高维 descriptor 表征下更快形成了更紧的选择边界。
- MACE medium 比 small 更宽松，`142` 帧、`113` 个 system，覆盖和 novelty 都略高于 small，但仍明显少于 DPA4 Mini。

同一任务下，三者的采样行为差异主要来自 descriptor 表征维度和几何结构分布方式，而不是候选池大小差异。

## 2. 目的与范围

本报告用于记录：

1. 在相同 DPA4 Mini UQ 候选集上，DPA4 Mini、MACE small、MACE medium 的 DP-EVA feature + collect 行为对比；
2. 三个模型在同一采样任务下的 sample rate、PC 主导性、coverage、novelty、diversity、selection overlap；
3. 相关 HTML 与 CSV 结果的可追溯归档。

## 3. 方法与数据

统一输入数据：

- training set: `test/dpa4-dpeva-test/sampled_dpdata`
- candidate pool: `test/dpa4-dpeva-test/other_dpdata`
- UQ candidate: 复用 DPA4 Mini 已完成结果

统一采样约束：

- `batch_size filter:128`
- 其他参数与 dpeva 内置模板保持一致
- direct 侧使用同一标尺下的候选集比较

描述符维数：

- DPA4 Mini: `32`
- MACE small: `256`
- MACE medium: `256`

## 4. 结果与分析

### 4.1 核心指标

| 模型 | descriptor dim | candidate count | selected frames | selected systems | sample rate | important PCs | retained variance | PC1 dominance | Top3 dominance | DIRECT coverage | Random coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DPA4 Mini | 32 | 22605 | 359 | 181 | 1.59% | 6 | 84.69% | 39.55% | 71.22% | 0.8927 | 0.7087 |
| MACE small | 256 | 22605 | 87 | 76 | 0.38% | 18 | 96.33% | 29.94% | 65.86% | 0.9070 | 0.6692 |
| MACE medium | 256 | 22605 | 142 | 113 | 0.63% | 16 | 96.31% | 25.52% | 62.49% | 0.9031 | 0.6677 |

### 4.2 采样行为差异

- DPA4 Mini 的选样数量最多，说明它在当前任务下对候选池的切分更细，保留了更多边界样本。
- MACE small/medium 的 retained variance 明显更高，说明它们的 descriptor 空间更“可压缩”，PCA 前几个主成分就能解释更多变化。
- MACE medium 相比 small 选得更多，说明更大的 MACE 表征并没有把采样进一步压缩到极端，而是提供了略更宽的覆盖。
- 三者的 DIRECT coverage 差异不大，但 MACE small/medium 在更少样本下达到接近甚至略高的 coverage，说明其选择更偏向“高信息密度样本”。

### 4.3 选择重叠

| 对比 | intersection | union | Jaccard |
|---|---:|---:|---:|
| DPA4 Mini vs MACE small | 7 | 439 | 0.015945 |
| DPA4 Mini vs MACE medium | 11 | 490 | 0.022449 |
| MACE small vs MACE medium | 14 | 215 | 0.065116 |

重叠度很低，说明三者不是“同一批样本只是在排序上有细小差异”，而是采样决策本身就明显不同。

## 5. 附件与可追溯结果

- HTML 报告: [MACE_DPA4_MINI_SAMPLING_COMPARISON.html](assets/2026-06-07-dpa4-mini-mace/MACE_DPA4_MINI_SAMPLING_COMPARISON.html)
- 指标汇总: [mace_mini_sampling_metrics.csv](assets/2026-06-07-dpa4-mini-mace/mace_mini_sampling_metrics.csv)
- 选择重叠: [mace_mini_selection_overlap.csv](assets/2026-06-07-dpa4-mini-mace/mace_mini_selection_overlap.csv)
- MACE small summary: [sampling_summary_mace_small.json](assets/2026-06-07-dpa4-mini-mace/sampling_summary_mace_small.json)
- MACE medium summary: [sampling_summary_mace_medium.json](assets/2026-06-07-dpa4-mini-mace/sampling_summary_mace_medium.json)

## 6. 建议

如果后续要用 DP-EVA 做代表性数据筛选，当前结果支持如下判断：

1. 若优先要更高覆盖和更强探索，DPA4 Mini 更合适；
2. 若优先要更少采样点、但仍保持较高 coverage，MACE small/medium 更合适；
3. 若需要和既有 DPA4 Mini 结果保持高度一致，这三类模型不应视为可直接互换。
