# 参数变更日志 (Parameter Changelog)

记录从 v1.0 (Dict-based) 到 v2.0 (Pydantic-based) 的参数变更历史。

## v2.0.1 (Review Updates)

根据用户审查反馈进行的修正和优化。

### 1. 逻辑与约束变更
*   **Training**: `num_models` 最小值约束从 `2` 提升至 `3` (为了满足 QbC UQ 的统计要求)。
*   **Training**: `base_model_path` 现在是**可选**的 (`init` 模式下)，仅在 `cont` 模式下必填。
*   **Training**: `init` 模式下不再强制添加 `--finetune` 参数。
*   **Inference**: `model_head` 现在是**可选**的，默认值为 `None` (内部自动解析为 `"results"`)，以支持 Frozen Model 的直接推理。
*   **Inference**: `output_basedir` 被移除，统一使用 `work_dir` 作为工作目录。

### 2. 默认值变更
*   `omp_threads`: 默认值从 `1` 更改为 `4`。新增 `"auto"` 选项 (自动使用 `os.cpu_count()`)。
*   `batch_size` (Feature): 默认值从 `1000` (Doc: 1) 更改为 `32`。
*   `fig_dpi`: 默认值从 `150` 提升至 `300` (出版级质量)。

## v2.0.0 (Refactoring)

### 1. 命名标准化 (Renaming)
统一了分散在各个模块中的异构参数名，强制使用标准化命名。

| 模块 | 旧参数名 (Legacy) | 新参数名 (Standard) | 说明 |
| :--- | :--- | :--- | :--- |
| **Training** | `finetune_head_name` | `model_head` | 统一模型 Head 参数名 |
| **Training** | `training_mode` | `mode` | 简化参数名 |
| **Collection** | `testing_head` | `results_prefix` | 明确语义为结果文件前缀，而非模型 Head |
| **Feature** | `modelpath` | `model_path` | 遵循下划线命名法 |
| **Feature** | `head` | `model_head` | 统一模型 Head 参数名 |
| **Inference** | `head` | `model_head` | 统一模型 Head 参数名 |
| **Inference** | `datadir` / `test_data_path` | `data_path` | 统一数据路径参数名 |

### 2. 默认值变更 (Defaults)

#### 移除不合理默认值 (Make Required)
为了提高系统的安全性和明确性，移除了以下参数的硬编码默认值，现在用户必须显式提供这些参数：

*   `FeatureConfig.model_head` (原默认: `"OC20M"`)
*   `InferenceConfig.model_head` (原默认: `"Hybrid_Perovskite"`)
*   `TrainingConfig.model_head` (原默认: `"Hybrid_Perovskite"`)

#### 修改不合理默认值 (Change Defaults)
*   `CollectionConfig.project`: `"stage9-2"` -> `"./"` (默认为当前目录)
*   `CollectionConfig.testing_dir`: `"test-val-npy"` -> `"test_results"` (使用更通用的目录名)
*   `CollectionConfig.results_prefix` (原 `model_head`): 默认值 `"results"` (符合 DeepMD `dp test` 默认输出行为)

#### 其他调整
| 参数名 | 旧默认值 | 新默认值 | 原因 |
| :--- | :--- | :--- | :--- |
| `omp_threads` | `24` (or undefined) | `1` | 避免默认高并发导致资源争抢，需用户显式指定高性能参数 |
| `uq_trust_mode` | `manual` (implicit) | `"auto"` | 默认启用更智能的自动边界计算，减少人工配置错误 |

### 3. 结构变更 (Structural Changes)
*   **Submission Config**: 所有的任务提交相关参数 (`backend`, `slurm_config`, `env_setup`) 现在统一收敛到 `submission` 嵌套对象中。
    *   旧: `{"backend": "slurm", "slurm_config": {...}}`
    *   新: `{"submission": {"backend": "slurm", "slurm_config": {...}}}`
    *   *(注: 为了兼容性，Pydantic 模型仍支持从顶层扁平结构自动解析，但推荐使用嵌套结构)*

### 4. 废弃参数 (Deprecated)
*   所有未在 `INPUT_PARAMETERS.md` 中定义的参数将被直接忽略（`extra='ignore'`），不再支持隐式传递未知参数。
