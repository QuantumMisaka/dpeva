# 工作流配置：分析导出结构与后续标注

`paper_sampling.json` 保存论文案例的采样参数，供 notebook 辅助代码读取：自动阈值比例 0.50、
宽度 0.25、冷启动/joint 预算 3,000、normal 预算 1,000。它是本案例的参数文件，
不是 `dpeva collect` 的完整 CLI 配置。

本目录用于从[交互教程](../README.md)衔接到完整 DP-EVA 环境；运行四章 notebook
不需要手动修改这里的配置。

教学主线调用当前 `UQCalculator`、`UQFilter` 和 `SamplingManager`，不加载模型。
现场每次导出都会新建 `work/selection-*/`，包含无标签 `label_input/`（DeepMD npy）、
`selected.extxyz` 与 `selected.csv` 来源映射。

[`analysis_export.json`](analysis_export.json) 是数据集分析模板，其中
`replace-with-selection-directory` 是占位路径，不能直接运行。
把 `dataset_dir` 替换成一次真实导出的 `label_input` 路径后，使用
`dpeva analysis configs/analysis_export.json`（从教学目录运行）。
配置相对路径以 JSON 所在目录为基准。
[第 04 章](../04_export_and_paper_results.ipynb)会自动生成绑定本次导出路径的配置。
执行 `dpeva` CLI 需另按仓库[安装指南](../../../../docs/guides/installation.md)准备完整环境；
教程的轻量环境只负责 notebook 中的交互与导出。

需要模型与外部程序的步骤使用现有配置入口：

- [四阶段 quickstart](../../../recipes/quickstart/README.md)：独立的 1 帧训练 / 5 帧候选示例，
  不是论文 30,656 帧实验。运行它需要 DeepMD PyTorch 环境。
- [采集 recipes](../../../recipes/README.md)：实际生产 UQ 与采样。
- [ABACUS 标注模板](../../../recipes/labeling/config_gpu.json)：将 `input_data_path` 改为本次
  `label_input/`，配置自己的赝势、轨道与运行环境。`dpeva label CONFIG --stage prepare`
  仅准备计算输入；教学 notebook 不会提交作业。

论文环境为 DeePMD-kit 3.1.2、DPA-3.1-3M / OC22。归档中四个浅层委员会的
`model.ckpt.pt` 是零字节占位，教学包使用真实保存的预测值。不能把此模板或旧配置直接
当成已验证的论文重训练入口。DPA4 迁移需要重新生成匹配的特征、委员会预测与评估。
