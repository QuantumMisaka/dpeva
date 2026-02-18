# test-for-multiple-datapool 生产目录输入输出分析

目标目录：[test-for-multiple-datapool](file:///home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/test/test-for-multiple-datapool)

本文档用于把生产级运行产物拆解为“输入资产 / 输出资产”，并将其映射到 DP-EVA 各 Workflow 的配置字段（对照 [examples/recipes](file:///home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/examples/recipes)）。

## 1. 目录结构概览（按语义分组）

- `other_dpdata_all/`
  - 多数据池候选集合（Multi Pool），目录结构为 `Pool/System/set.000/*`。
  - 典型候选 system 可能只包含 `coord.npy/box.npy/type.raw/type_map.raw`（无能量/力标注），用于推理与不确定性评估。
- `sampled_dpdata/`
  - 训练用标注数据（Single Pool 的平铺 system 目录），目录结构为 `System/set.000/*`。
  - 典型训练 system 包含 `energy.npy/force.npy/virial.npy` 等标注数据。
- `desc_pool/`
  - 候选集合（`other_dpdata_all`）对应的描述符输出（Feature Workflow 的产物）。
  - 目录结构为 `Pool/System.npy`，system 名称与候选 dpdata 中的 `System` 目录名一致。
- `desc_train/`
  - 训练集合（`sampled_dpdata`）对应的描述符输出（Feature Workflow 的产物）。
  - 目录结构为 `System.npy`（平铺）。
- `0/ 1/ 2/ 3/`
  - Ensemble 训练输出目录（Training Workflow 的产物），每个目录代表一个 seed / 一个模型实例。
  - 典型文件：
    - `input.json`：DeepMD-kit `dp train` 输入（训练超参、数据路径等）。
    - `model.ckpt.pt` / `model.ckpt-*.pt`：训练权重（推理输入）。
    - `lcurve.out`：训练曲线。
    - `test_val/`：推理输出目录（Inference Workflow 的产物），包含 `results.e.out/results.f.out/...` 与 `test.log`。
- `dpeva_uq_result/`
  - 数据采集与筛选（Collection Workflow）的输出目录（以 `root_savedir` 命名）。
  - 典型产物：
    - `dataframe/*.csv`：UQ/描述符/采样过程的中间与最终表格。
    - `view/*.png`：UQ 分布、PCA 覆盖度等可视化。
    - `dpdata/sampled_dpdata/`：筛选出的样本（保持 Multi Pool 结构）。
    - `dpdata/other_dpdata/`：未被选中的候选样本集合（保持 Multi Pool 结构）。

## 2. Workflow ↔ 输入/输出资产映射

### 2.1 FeatureWorkflow（描述符生成）

输入：

- `FeatureConfig.data_path`
  - 候选描述符：指向 `other_dpdata_all/`
  - 训练描述符：指向 `sampled_dpdata/`
- `FeatureConfig.model_path`
  - 用于 `dp eval-desc` 的模型（生产目录中可见 `DPA-3.1-3M.pt` / 或任意可用模型）。
- `FeatureConfig.model_head`
- `submission.backend="slurm"`

输出：

- `FeatureConfig.savedir`
  - 候选：`desc_pool/`（Multi Pool 输出，`Pool/System.npy`）
  - 训练：`desc_train/`（Single Pool 输出，`System.npy`）

参考配置：

- [config_feature.json](file:///home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/examples/recipes/feature_generation/config_feature.json)

### 2.2 TrainingWorkflow（Ensemble 训练）

输入：

- `TrainingConfig.training_data_path`：指向 `sampled_dpdata/`（需要包含标注能量/力等）。
- `TrainingConfig.input_json_path`：DeepMD-kit 训练输入，例如生产目录内各模型下的 `input.json`。
  - 注意：生产 `input.json` 内 `training.training_data.systems` 指向 `../sampled_dpdata`。
- `TrainingConfig.base_model_path`：预训练模型（例如 `DPA-3.1-3M.pt`）。
- `TrainingConfig.num_models`：对应输出子目录数量（生产为 4，对应 `0..3/`）。
- `TrainingConfig.model_head`
- `submission.backend="slurm"`

输出：

- `work_dir/0..N-1/`
  - `model.ckpt.pt`、`checkpoint`、`lcurve.out` 等。

参考配置：

- [config_train.json](file:///home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/examples/recipes/training/config_train.json)

### 2.3 InferenceWorkflow（候选集合推理）

输入：

- `InferenceConfig.data_path`：指向 `other_dpdata_all/`（候选集合，可无标注）。
- `InferenceConfig.task_name`：生产目录中为 `test_val/`。
- `work_dir`：应为训练输出所在目录，推理会扫描 `0..N-1/` 模型目录并提交 `dp test`。
- `InferenceConfig.model_head`（如使用 Frozen/Head 模式）
- `submission.backend="slurm"`

输出：

- `work_dir/<i>/<task_name>/`
  - `results.e.out/results.f.out/results.v.out` 等。
  - `test.log`：`dp test` 过程日志（可用于定位测试数据扫描情况）。

参考配置：

- [config_infer.json](file:///home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/examples/recipes/inference/config_infer.json)

### 2.4 CollectionWorkflow（UQ + Filtering + Sampling + Export）

输入（核心）：

- `CollectionConfig.desc_dir`：候选描述符目录 `desc_pool/`。
- `CollectionConfig.testdata_dir`：候选 dpdata 目录 `other_dpdata_all/`。
- `CollectionConfig.testing_dir`：推理输出子目录名 `test_val/`。
- `CollectionConfig.results_prefix`：推理结果前缀 `results`。
- `CollectionConfig.root_savedir`：输出目录名，例如生产目录 `dpeva_uq_result/`。
- 采样参数（例如 `direct_n_clusters/direct_k` 或 `sampler_type="2-direct"` 等）。
- `submission.backend="slurm"`（生产配置中用 CPU 队列执行 collect）。

输入（联合采样 Joint Sampling，可选）：

- `CollectionConfig.training_data_dir`：训练 dpdata 目录 `sampled_dpdata/`。
- `CollectionConfig.training_desc_dir`：训练描述符目录 `desc_train/`。

输出：

- `<root_savedir>/dataframe/*.csv`
- `<root_savedir>/view/*.png`
- `<root_savedir>/dpdata/sampled_dpdata/<Pool>/<System>/...`
- `<root_savedir>/dpdata/other_dpdata/<Pool>/<System>/...`

参考配置：

- 普通模式：[config_multi_normal.json](file:///home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/examples/recipes/collection/config_multi_normal.json)
- 联合模式：[config_multi_joint.json](file:///home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/examples/recipes/collection/config_multi_joint.json)

### 2.5 AnalysisWorkflow（结果统计与可视化）

输入：

- `AnalysisConfig.result_dir`：指向某一个模型的推理输出目录，例如 `0/test_val/`。
- `AnalysisConfig.type_map`：元素列表（与体系一致）。

输出：

- `AnalysisConfig.output_dir`：分析产物目录（图表/统计汇总）。

参考配置：

- [analysis/config.json](file:///home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/examples/recipes/analysis/config.json)

## 3. 用于集成测试编排的关键观察

- 候选集合（`other_dpdata_all`）与训练集合（`sampled_dpdata`）在“是否包含标注”上语义不同：
  - 训练数据需要 `energy/force/virial` 等，用于 `dp train`。
  - 候选数据可只包含结构信息，用于 `dp test` 生成预测并供 UQ 计算。
- 描述符输出的命名规则与输入目录结构一致：
  - Multi Pool：输出 `Pool/System.npy`。
  - Single Pool：输出 `System.npy`。
- 集成测试链路应以统一完成标记 `DPEVA_TAG: WORKFLOW_FINISHED` 作为每一步结束的可靠锚点；测试编排器通过监控 Slurm 输出日志定位完成节点再启动下一步。

