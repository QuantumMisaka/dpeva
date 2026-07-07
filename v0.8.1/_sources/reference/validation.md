---
title: Document
status: active
audience: Developers
last-updated: 2026-07-04
owner: Docs Owner
---

# 参数校验与约束（Validation Rules）

- Status: active
- Audience: Developers
- Last-Updated: 2026-07-04

本文件用于作为参数校验规则的单一权威来源。

本文档说明了 DP-EVA 系统中各参数的校验逻辑和约束条件，这些规则由 Pydantic 验证器在运行时强制执行。

## 1. 基础类型校验
所有参数必须符合定义的 Python 类型。
*   **Path**: 必须是字符串或 Path 对象，且部分路径必须在文件系统中实际存在（参见具体参数说明）。
*   **Int/Float**: 会自动尝试进行类型转换（如字符串 "1" 转为整数 1），转换失败则报错。

## 2. 数值范围约束
| 参数 | 约束条件 | 错误示例 |
| :--- | :--- | :--- |
| `omp_threads` | `>= 1` (若为数字) | `0`, `-1` |
| `num_models` | `>= 3` | `2` (QbC UQ 至少需要 3 个模型) |
| `uq_trust_ratio` | `0.0 <= x <= 1.0` | `1.5`, `-0.1` |
| `uq_trust_width` | `> 0.0` | `0.0`, `-0.5` |
| `direct_n_clusters` | `> 0` (若设置) | `0` |
| `direct_k` | `>= 1` | `0` |

## 3. 逻辑依赖校验 (Cross-Field Validation)

### 3.1 采集工作流 (CollectionConfig)

#### 3.1.1 UQ Trust Mode 依赖
*   **规则**: 当 `uq_trust_mode` 为 `"manual"` 时，必须提供手动边界参数。
*   **规则**: 当 `uq_trust_mode` 为 `"no_filter"` 时，不启用信任区筛选，手动边界参数将被忽略。
*   **QbC 检查**:
    *   必须提供 `uq_qbc_trust_lo`。
    *   必须提供 `uq_qbc_trust_hi` **或者** `uq_qbc_trust_width` (此时 `hi = lo + width`)。
*   **RND 检查**:
    *   必须提供 `uq_rnd_rescaled_trust_lo`。
    *   必须提供 `uq_rnd_rescaled_trust_hi` **或者** `uq_rnd_rescaled_trust_width`。
*   **报错信息**: `ValueError: In 'manual' trust mode, uq_qbc_trust_lo must be provided.`

#### 3.1.2 采样参数依赖
*   **规则**: 采样策略由 `sampler_type` 控制：
    *   `sampler_type="direct"`：使用 `direct_n_clusters/direct_k/direct_thr_init`。
    *   `sampler_type="2-direct"`：使用 `step1_*` 与 `step2_*` 参数组。
*   **规则**: 若 `direct_n_clusters` 显式给定，必须 `> 0`。

#### 3.1.3 出图分层开关依赖
*   **规则**: `enable_diagnostic_plots` 默认 `False`，仅输出 Core 层图像。
*   **规则**: 当 `enable_diagnostic_plots=True` 时，仅在满足真值和数据前置条件时输出 Diagnostic 层 parity 图像。
*   **可观测性**: 跳过图像必须输出统一 `reason` 字段，且生成/跳过清单在日志中可审计。

### 3.2 特征工作流 (FeatureConfig)

#### 3.2.1 自动保存目录 (Auto Savedir)
*   **规则**: 如果用户未指定 `savedir`，系统会自动根据 `model_path` 和 `data_path` 生成默认保存路径。
*   **逻辑**: `savedir = "desc-{model_stem}-{data_name}"`

#### 3.2.2 Feature exporter
*   **默认**: `feature_exporter="eval_desc"`，CLI 模式调用 `dp eval-desc`，输出旧 `.npy` descriptor。
*   **Embed**: `feature_exporter="embed"` 调用 `dp embed`，输出 `savedir/embedding.hdf5`。
*   **特征映射**: `feature_kind="descriptor"` 对应 HDF5 `descriptor`；`feature_kind="fitting_last_layer"` 对应 HDF5 `atomic_feature`。
*   **精度**: `embedding_dtype` 允许 `fp32`、`fp64`、`native`，默认 `fp32`。

#### 3.2.3 Collect feature input
*   **规则**: `CollectionConfig.desc_dir` 与 `training_desc_dir` 可指向 `.npy` descriptor 目录、单个 HDF5 文件，或包含 `embedding.hdf5` 的目录。
*   **特征映射**: `CollectionConfig.desc_feature_kind="descriptor"` 默认读取 HDF5 `descriptor`；`"fitting_last_layer"` 读取 HDF5 `atomic_feature`。
*   **多池命名**: 递归 HDF5 目录会把 `embedding.hdf5` 相对 `desc_dir` 的父目录作为 system 前缀，例如 `desc/poolA/embedding.hdf5` 生成 `poolA/sys1-0`。
*   **LLPR**: `llpr_train_feature_dir` 与 `llpr_candidate_feature_dir` 可直接读取 HDF5 `atomic_feature`。

### 3.3 提交配置 (SubmissionConfig)

#### 3.3.1 环境变量格式化
*   **规则**: `env_setup` 字段支持字符串或字符串列表。
*   **转换**: 如果输入为列表 `["cmd1", "cmd2"]`，会自动转换为多行字符串 `"cmd1
cmd2"`。

## 4. 路径存在性校验
以下参数在初始化时会检查文件/目录是否存在，若不存在将抛出 `ValidationError` (或后续运行时错误)：

*   `CollectionConfig.desc_dir`
*   `CollectionConfig.testdata_dir`
*   `FeatureConfig.data_path`
*   `FeatureConfig.model_path`
*   `InferenceConfig.data_path`
*   `TrainingConfig.base_model_path`

## 5. 标注工作流校验 (LabelingConfig)

### 5.1 阈值与参数
*   `cleaning_thresholds`: 支持设置为 `null` 以跳过对特定物理量（如 `cohesive_energy`）的检查。
*   `kpt_criteria`: 必须 `> 0`。用于自动计算 K 点网格密度。
*   `labeling_task_classes`: 若启用 task class，`resource_mode="single_gpu"` 必须搭配 `launcher_mode="abacus"`；`resource_mode="multi_gpu_mpi"` 必须搭配 `launcher_mode="mpi_abacus"`。
*   普通 FP11 单卡 class 应设置 `ntasks=1`、`gpus_per_node=1`，且不加载 rank-map 或引用 `MAP_OPT`。多卡 MPI ABACUS class 才加载 SAI rank-map 并通过 `MAP_OPT` 组装 `mpirun`。

### 5.2 路径校验
*   `input_data_path`: 必须存在。
*   `pp_dir`: 必须存在。
*   `orb_dir`: 必须存在。
