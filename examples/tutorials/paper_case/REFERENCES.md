# 教程参考资料与阅读路线

现场用四章 notebook 完成数据选择；本页供课前备课和课后深入阅读。
教学叙述围绕“预训练表示 → 模型信息 → 代表性采样 → 任务评估”展开。

## 先读什么

| 资料 | 建议阅读位置 | 帮助回答的问题 |
|---|---|---|
| [DP-EVA 论文](https://doi.org/10.1093/ce/zkag029) | Figure 1、Methods、Table 1–3；应用部分再看 Figure 5 与 SI Table S1 | 方法怎样利用预训练知识，实验如何支持数据效率结论？ |
| 作者确认的最终中文供稿 `DP-EVA-CleanEnergy-微信推文供稿-0908.docx` | 背景、方法两条线、案例结果；来源位置见下文 | 怎样用中文串起研究问题与方法？ |
| [分享会微信推送](https://mp.weixin.qq.com/s/WhFq-Qy_uMPCZ2nzGUqbLg) | 本次分享会的活动推送 | 了解分享会背景；与论文中文供稿分别记录 |
| [DIRECT 原始论文](https://doi.org/10.1038/s41524-024-01227-4) | 降维、聚类与分层选样的方法部分 | 为什么代表性采样能改善复杂结构空间的覆盖？ |
| [FT²DP 原始论文](https://doi.org/10.20517/jmi.2024.105) | 数据构建、Fe₅C₂ 表面反应与重构 | 这些 FeCHO 数据来自什么研究任务，模型最终用来做什么？ |

DP-EVA：Liu Z, Deng Z, Zhao H, Wang H, Chen M, Jiang H.
*DP-EVA: data-efficient fine-tuning framework via maximizing pretrained knowledge of large atomistic models for developing domain-specific machine-learning interatomic potentials*.
Clean Energy, 2026, 10(4), 103–116. DOI: 10.1093/ce/zkag029。

DIRECT：Qi J, Ko T-W, Wood BC, Pham TA, Ong SP.
*Robust training of machine learning interatomic potentials with dimensionality reduction and stratified sampling*.
npj Computational Materials, 2024, 10, 43. DOI: 10.1038/s41524-024-01227-4。

FT²DP：Liu Z-Q, Deng Z, Zhao H, Wang H, Chen M, Jiang H.
*FT2DP: large atomic model fine-tuned machine learning potential for accelerating atomistic simulation of iron-based Fischer-Tropsch synthesis*.
Journal of Materials Informatics, 2025, 5, 27. DOI: 10.20517/jmi.2024.105。

## 每章的来源与实现

| 章节 | 原文依据 | 教程与当前代码的对应 |
|---|---|---|
| 01 数据与表示 | DP-EVA Figure 1、Dataset and computational details；FT²DP 数据与应用背景 | 30,656 帧、128 维 DPA3 描述符；[数据来源记录](data/provenance.json) |
| 02 冷启动 | DP-EVA DIRECT 方法段、Table 1；DIRECT 原始方法 | [PCA](../../../src/dpeva/sampling/pca.py)、[BIRCH](../../../src/dpeva/sampling/clustering.py)、[分层选点](../../../src/dpeva/sampling/stratified_sampling.py)；[论文指标表](data/paper_metrics.csv) |
| 03 UQ 与联合采样 | DP-EVA 浅层系综方法、Figure 4；SI 自动阈值说明 | [UQ 计算](../../../src/dpeva/uncertain/calculator.py)、[UQ 分类](../../../src/dpeva/uncertain/filter.py)、[采样管理](../../../src/dpeva/sampling/manager.py)；[案例参数](configs/paper_sampling.json) |
| 04 导出与科学评估 | DP-EVA Table 2–3、SI Table S1、Figure 5；FT²DP 应用方法 | 真实结构导出；原论文训练误差与应用结果解读；[工作流配置入口](configs/README.md) |

补充覆盖对照和新增信息距离属于教程核查实验，结果见
[conclusion_audit.json](data/conclusion_audit.json)，不作为论文原始数值引用。
现场重点解释指标和观察到的趋势；公式推导置于第 03 章末尾，供课后阅读。

## 本轮使用的材料

2026-10-21 核对使用作者提供的 `DPEVA-v7.docx`、`DPEVA-v7-SI.docx`、
实验归档及当前仓库代码。中文叙述已按作者确认的最终供稿
`DP-EVA-CleanEnergy-微信推文供稿-0908.docx` 核对，源文件位于
`D:\wsl-web\`（WSL 对应 `/mnt/d/wsl-web/`）。此前参考的 v10 Markdown 为历史供稿，
最终中文表达以此 Word 为准。核对涉及预训练知识的两条利用路径、浅层委员会与最终模型的区别、
DIRECT 与 joint-DIRECT 的归属，以及数据量和精度的有效数字；与教程当前讲解一致。

上述微信链接由作者确认为**分享会的活动推送**，与论文中文供稿分别记录。
本轮浏览工具仍未能直接读取活动推送正文，未据此核实活动时间、议程等细节。
方法和科学数值以论文及可追溯的实验、代码为依据；软件能力以当前仓库文档为准。

两张论文图片随目录和 notebook 附件一起保存，来源及许可见
[figures/README.md](figures/README.md)。教程运行无需访问上述个人归档或在线文献。

## 接入自己的工作

软件操作以当前仓库的 [CLI 指南](../../../docs/guides/cli.md)、
[配置指南](../../../docs/guides/configuration.md) 和
[四阶段 quickstart](../../recipes/quickstart/README.md) 为准。
论文中的历史训练设置用于理解案例，当前接口按仓库配置执行。
换用 DPA4 或 FT²DP-v2 时，应重新生成匹配的表示与委员会预测，并为目标反应和表面结构建立应用评估集。
