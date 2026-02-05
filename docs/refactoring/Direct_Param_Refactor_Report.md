# DIRECT 采样参数激进重构开发报告

## 1. 核心变更概述
根据您的指示，我对 DIRECT 采样的参数控制逻辑进行了激进的重构，移除了隐式的默认值，并引入了“动态聚类”的高级模式。此次重构使得标准 DIRECT 采样的控制逻辑更加原生（直接控制簇数量），同时与 2-DIRECT 的参数体系形成了清晰的区分。

## 2. 详细变更内容

### 2.1 配置层 (`config.py`)
-   **新增 `direct_n_clusters`**: 类型为 `Optional[int]`，默认 `None`。用于显式控制标准 DIRECT 的聚类数量。
-   **弃用 `num_selection`**: 类型为 `Optional[int]`，默认 `None`。仅保留作为遗留参数以支持旧版配置，使用时会触发 Warning。
-   **移除默认值**: 移除了 `DEFAULT_NUM_SELECTION` 在 Config 中的应用，强制用户显式指定参数，或明确选择进入动态模式。

### 2.2 业务逻辑层 (`collect.py`)
在 `CollectionWorkflow.run()` 中实现了三级决策逻辑：

1.  **显式控制模式 (Level 1 - Explicit)**:
    -   当 `direct_n_clusters` 被设置时，系统直接将其用作 `Birch` 聚类的 `n_clusters` 参数。
    -   这是推荐的用法。

2.  **兼容模式 (Level 2 - Compatibility)**:
    -   当 `direct_n_clusters` 未设置但 `num_selection` 被设置时，系统计算 `n_clusters = num_selection // direct_k`。
    -   触发 DeprecationWarning。

3.  **动态聚类模式 (Level 3 - Dynamic)**:
    -   当两者均未设置时，系统将 `n_clusters` 设为 `None`。
    -   `Birch` 算法将根据 `direct_thr_init` (阈值) 自动决定聚类数量。
    -   触发 Warning 提示用户当前处于动态模式。

### 2.3 常量层 (`constants.py`)
-   移除了 `DEFAULT_NUM_SELECTION`，彻底切断了对“默认 100 个样本”的依赖。

## 3. 验证结果
新增的单元测试 `tests/unit/workflows/test_collect_nofilter.py` 覆盖了所有三种模式：
-   `test_direct_mode_explicit`: 验证 `direct_n_clusters` 正确传递。
-   `test_direct_mode_compat`: 验证 `num_selection` 的转换逻辑。
-   `test_direct_mode_dynamic`: 验证全空配置下 `n=None` 的行为及 Warning 输出。

所有测试（包括 2-DIRECT 和联合采样回归测试）均已通过。

## 4. 使用建议
**推荐配置 (显式控制):**
```yaml
sampler_type: "direct"
direct_n_clusters: 50
direct_k: 2
# 结果: 50 * 2 = 100 个样本
```

**高级配置 (动态模式):**
```yaml
sampler_type: "direct"
direct_thr_init: 0.5
direct_k: 1
# 结果: 聚类数由数据分布和 0.5 阈值决定
```
