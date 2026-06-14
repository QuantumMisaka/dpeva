# DP-EVA UQ Method Deep Dive

## tangent_lo 双维度划分

本实验按 DP-EVA `UQFilter(scheme="tangent_lo")` 重新划分候选池，输入维度为 `uq_qbc_for` 与 `uq_rnd_rescaled`。阈值来源为 `auto-derived`：`qbc_lo=0.1201`、`qbc_hi=0.3701`、`rnd_lo=0.1201`、`rnd_hi=0.3701`。当前实验没有人工阈值配置，因此 auto-derived 口径使用项目默认 `ratio=0.33`、`width=0.25`；若 KDE 无法给出 lo，则使用 `1-ratio` 分位数 fallback。

| uq_identity | error_metric | n_frames | frame_fraction | high_error_rate | high_error_recall | high_error_enrichment |
| --- | --- | --- | --- | --- | --- | --- |
| accurate | force_error_max | 2.295e+04 | 0.8299 | 0.005185 | 0.08604 | 0.1037 |
| candidate | force_error_max | 4471 | 0.1617 | 0.2451 | 0.7925 | 4.902 |
| failed | force_error_max | 232 | 0.008389 | 0.7241 | 0.1215 | 14.48 |
| accurate | force_error_rms | 2.295e+04 | 0.8299 | 0.003834 | 0.06363 | 0.07667 |
| candidate | force_error_rms | 4471 | 0.1617 | 0.2521 | 0.8149 | 5.041 |
| failed | force_error_rms | 232 | 0.008389 | 0.7241 | 0.1215 | 14.48 |

解释：`tangent_lo` 不是单一 UQ 排序，而是在 QbC 与 RND-rescaled 平面上把低不确定度、候选区和超高不确定度失败区分开。因此它更适合回答“DP-EVA collect 会把哪些结构送入候选集”，而不替代单指标 Spearman/Pearson 排序。本实验中 force 主比较的最佳 rank correlation 是 `uq_rnd_for` vs `force_error_max`，Spearman = `0.7801`；tangent_lo 汇总则检验这些几何分区是否富集真实 high-force-error frame。

## LLPR/DPOSE 偏差是否合理

现有结果显示，QbC/RND force UQ 与真实 force error 强正相关，而 `uq_llpr_energy_per_atom` / `uq_dpose_energy_ensemble_std_per_atom` 与 `energy_error_per_atom_abs` 只有弱正相关；最佳 energy 主比较是 `uq_llpr_energy_per_atom`，Spearman = `0.1253`。这在当前实现下是合理的：

- DP-EVA QbC/RND 直接从 force 输出差异构造 frame-level force UQ，因此目标与 `force_error_max` / `force_error_rms` 对齐。
- 当前 DP-EVA LLPR/DPOSE 使用 detached fitting-last-layer features，输出 energy-level analytic uncertainty 与 energy ensemble std；它不是 force-aware DPOSE。
- force DPOSE 需要可微 energy ensemble 对坐标求导或等价 Jacobian 路径。detached feature 表无法恢复该路径，所以 LLPR/DPOSE vs force error 只能作为 exploratory。

## 外部资料依据

- Atomistic Cookbook PET-MAD UQ 教程展示 PET-MAD 内置 LLPR uncertainty，并把 ensemble/LLPR uncertainty 用于数据集、能量差、MD 平均量等 derived quantities：https://atomistic-cookbook.org/examples/pet-mad-uq/pet-mad-uq.html
- metatrain LLPR 文档把 LLPR 定义为 wrapper architecture，可输出 `{target}_uncertainty` standard deviation 和 `{target}_ensemble`，用于 cheap uncertainty quantification：https://docs.metatensor.org/metatrain/latest/architectures/generated/llpr.html
- DPOSE/浅层系综资料强调 shallow ensemble 通过共享 backbone、只 ensemble last layer 来降低 full ensemble 成本，并指出 force uncertainty 的可靠校准需要 force-aware 目标或可微传播：https://arxiv.org/html/2602.15747v1

## 方法比较

| 方法 | 本实验信号 | 浅层含义 | 主要成本 | 主要限制 |
| --- | --- | --- | --- | --- |
| QbC force ensemble | force error 强相关 | 多个完整模型的 force 输出差异 | 训练/推理多个模型 | 成本高于单模型，依赖成员多样性 |
| RND force deviation | force error 强相关，本实验最高 | baseline 与 committee force 偏离 | 复用 ensemble 推理 | 仍是 force-output ensemble，不是 last-layer posterior |
| DP-EVA fitting-last-layer 后处理 | energy risk 补充 | frozen representation + last-layer feature covariance/weights | feature 导出 + 线性代数 | 当前 detached feature 只支持 energy-level |
| LLPR/DPOSE shallow ensemble | energy error 弱正相关 | last-layer posterior / last-layer sampled ensemble | 比 full ensemble 低，需 feature 和权重 | force UQ 需可微 graph adapter |

## 结论

在这个 DPA4 Mini 实验上，主动学习若以 force error 控制为目标，应优先使用 QbC/RND 及其 tangent_lo 双维度筛选；LLPR/DPOSE 更适合补充 energy-level 风险、组合量不确定度或低成本后验分析。若要把 DPOSE 用作 force-level 主采样信号，需要实现可微的 DeepMD PyTorch graph adapter 或显式 force-aware shallow ensemble。
