# 参数校验与约束（Validation Rules）

- Status: active
- Audience: Developers
- Last-Updated: 2026-02-18

本文件用于作为参数校验规则的单一权威来源。

---

# 参数验证规则 (Validation Rules)

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

### 3.2 特征工作流 (FeatureConfig)

#### 3.2.1 自动保存目录 (Auto Savedir)
*   **规则**: 如果用户未指定 `savedir`，系统会自动根据 `model_path` 和 `data_path` 生成默认保存路径。
*   **逻辑**: `savedir = "desc-{model_stem}-{data_name}"`

### 3.3 提交配置 (SubmissionConfig)

#### 3.3.1 环境变量格式化
*   **规则**: `env_setup` 字段支持字符串或字符串列表。
*   **转换**: 如果输入为列表 `["cmd1", "cmd2"]`，会自动转换为多行字符串 `"cmd1\ncmd2"`。

## 4. 路径存在性校验
以下参数在初始化时会检查文件/目录是否存在，若不存在将抛出 `ValidationError` (或后续运行时错误)：

*   `CollectionConfig.desc_dir`
*   `CollectionConfig.testdata_dir`
*   `FeatureConfig.data_path`
*   `FeatureConfig.model_path`
*   `InferenceConfig.data_path`
*   `TrainingConfig.base_model_path`
