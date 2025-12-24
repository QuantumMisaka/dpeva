# 重构方案：DPEVA 主动学习工作流 (Revision 3 - Final)

## 1. 核心架构决策
我们明确区分 **核心功能模块 (Core Mechanisms)** 与 **业务工作流 (Business Workflows)**。

*   **功能模块**: 存放纯粹的计算、数据处理和可视化逻辑，不包含流程控制。
    *   `src/dpeva/uncertain`: 不确定度计算与分析。
    *   `src/dpeva/sampling`: 数据采样算法 (DIRECT 等)。
*   **工作流模块**: 存放将功能模块串联起来的业务逻辑。
    *   **`src/dpeva/workflows`** (新增): 专门用于定义高层流水线。

## 2. 模块迁移与重构规划

### 2.1 填充 `src/dpeva/uncertain` (功能模块)
我们将从 `uq-post-view.py` 提取以下功能类：

*   **`calculator.py`**:
    *   实现 `UQCalculator` 类。
    *   职责：加载模型预测，计算 QbC/RND 指标，执行 Z-Score 对齐。
*   **`filter.py`**:
    *   实现 `UQFilter` 类。
    *   职责：封装 `strict`, `tangent_lo` 等几何筛选策略，返回分类结果。
*   **`visualization.py`**:
    *   实现 `UQVisualizer` 类。
    *   职责：封装所有 matplotlib/seaborn 绘图逻辑，确保复现 `ref_results` 中的图表样式。

### 2.2 完善 `src/dpeva/sampling` (功能模块)
*   保持现有 `direct.py`, `clustering.py` 不变。
*   如有必要，将原脚本中 DIRECT 相关的 PCA 可视化逻辑封装进 `src/dpeva/sampling/visualization.py`。

### 2.3 创建 `src/dpeva/workflows` (工作流模块)
*   **`uq.py`**:
    *   定义 `UQActiveLearningWorkflow` 类。
    *   **职责**:
        1.  **Orchestration**: 调用 `io`, `uncertain`, `sampling` 模块。
        2.  **State Management**: 管理 DataFrame 的流动（从 raw data -> UQ data -> candidate data -> selected data）。
        3.  **Logging**: 统一管理日志输出，确保复现 `UQ-DIRECT-selection.log`。

## 3. 脚本层重构
原脚本 `utils/uq/uq-post-view.py` 将被极度简化，仅作为配置入口。

**`utils/uq/run_uq_workflow.py` (新脚本):**
```python
from dpeva.workflows.uq import UQActiveLearningWorkflow

# 配置字典 (显式优于隐式)
config = {
    "paths": {"work_dir": ".", "desc_dir": "..."},
    "uq": {"scheme": "tangent_lo", "trust_lo": 0.05, ...},
    "sampling": {"method": "direct", ...}
}

# 一键运行
workflow = UQActiveLearningWorkflow(config)
workflow.run()
```

## 4. 实施步骤

1.  **基础设施**: 创建 `src/dpeva/workflows` 和 `src/dpeva/uncertain` 内的文件结构。
2.  **逻辑拆解 (Refactoring)**:
    *   **Step 1**: 将 UQ 计算逻辑移入 `uncertain/calculator.py`。
    *   **Step 2**: 将筛选逻辑移入 `uncertain/filter.py`。
    *   **Step 3**: 将绘图逻辑移入 `uncertain/visualization.py`。
    *   **Step 4**: 将主流程控制移入 `workflows/uq.py`。
3.  **脚本重写**: 编写新的 `run_uq_workflow.py`。
4.  **验证 (Testing)**:
    *   在 `test/UQ-post-view` 下运行新脚本。
    *   **Strict Check**: 比对 `selected_indices.npy`, `*.png` 和日志内容，确保与 `ref_results` 一致。

## 5. 预期成果
*   代码结构清晰：算法在底层，流程在上层。
*   复用性强：`src/dpeva` 成为一个真正的库，可通过 pip 安装使用。
*   结果一致：完全复现参考结果。

请确认此最终方案，我们将开始执行。