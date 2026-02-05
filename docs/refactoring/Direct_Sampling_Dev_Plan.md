# DIRECT 采样独立调用与功能增强开发方案

## 1. 需求分析
用户希望能够通过 `dpeva cli` 独立调用 DIRECT 采样功能，具体要求如下：
1.  **支持普通采样与联合采样**：即支持仅基于目标池采样，或基于目标池+训练集（辅助）进行采样。
2.  **全量数据集采样**：支持跳过不确定度（UQ）筛选步骤，直接对全量数据进行采样。
3.  **完整出图**：在跳过 UQ 的情况下，仍需输出所有与 DIRECT 采样相关的可视化图表（如 PCA 覆盖率图）。
4.  **无缝集成**：复用现有 `CollectionWorkflow`，避免代码冗余。

## 2. 现有架构调研结论
经过对 `src/dpeva/workflows/collect.py` 和 `src/dpeva/config.py` 的审计：
-   **现状**：`CollectionWorkflow` 强制执行 UQ 计算与筛选，无法直接跳过。
-   **联合采样**：已通过 `training_desc_dir` 参数支持，逻辑位于 `_prepare_features_for_direct` 中，且依赖于 `df_candidate`（候选集）。
-   **可视化**：DIRECT 相关的可视化（PCA 覆盖率）独立于 UQ 可视化，但在当前流程中位于 UQ 之后。

## 3. 开发方案

### 3.1 配置层变更 (`src/dpeva/config.py`)
在 `CollectionConfig` 中扩展 `uq_trust_mode` 的定义，增加 `"no_filter"` 模式。

```python
# 修改前
uq_trust_mode: Literal["auto", "manual"] = "auto"

# 修改后
uq_trust_mode: Literal["auto", "manual", "no_filter"] = "auto"
```

### 3.2 业务逻辑变更 (`src/dpeva/workflows/collect.py`)
修改 `CollectionWorkflow.run` 方法，根据 `uq_trust_mode` 分流逻辑：

1.  **当 `uq_trust_mode == "no_filter"` 时**：
    *   **跳过**：UQ 计算（QbC/RND）、UQ 统计、UQ 相关的可视化（分布图、Trust Range 图）。
    *   **执行**：
        *   加载描述符 (`_load_descriptors`)。
        *   构造 `df_candidate`，使其包含所有加载的描述符数据（即 `df_candidate = df_desc`）。
        *   设置 `df_accurate` 和 `df_failed` 为空 DataFrame。
        *   **直接进入** DIRECT 采样阶段（复用现有逻辑）。
        *   **执行** DIRECT 相关的可视化 (`vis.plot_pca_analysis`)。
2.  **当 `uq_trust_mode != "no_filter"` 时**：
    *   保持原有逻辑不变（计算 UQ -> 筛选 -> 采样）。

### 3.3 联合采样支持
由于联合采样逻辑 (`_prepare_features_for_direct`) 是基于 `df_candidate` 进行的，上述修改（将全量数据赋值给 `df_candidate`）天然支持联合采样。只要配置文件中提供了 `training_desc_dir`，联合采样逻辑将自动生效。

## 4. 实施步骤

1.  **修改 Config**：更新 `dpeva/config.py` 中的 `CollectionConfig`。
2.  **重构 Workflow**：在 `dpeva/workflows/collect.py` 中引入条件分支，处理 `"no_filter"` 模式。
3.  **单元测试**：
    *   新增 `tests/unit/workflows/test_collect_nofilter.py`。
    *   测试用例 1：**Standalone Direct**（仅目标池，无 UQ，检查输出文件和图表）。
    *   测试用例 2：**Joint Direct**（目标池 + 训练集，无 UQ，检查采样逻辑是否包含训练集信息）。

## 5. 预期交付物
-   更新后的 `src/dpeva/config.py`
-   更新后的 `src/dpeva/workflows/collect.py`
-   新增的测试文件 `tests/unit/workflows/test_collect_nofilter.py`

此方案最大程度复用了现有代码，仅通过配置项控制流程分支，符合“无冗余、完美融入”的要求。
