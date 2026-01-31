# DPTestResults 类废弃与重构方案文档

## 1. 背景与现状分析

### 1.1 组件概述
*   **类名**: `DPTestResults`
*   **位置**: `src/dpeva/io/dataproc.py` (L255)
*   **状态**: 已标记为 `Deprecated` (遗留包装类)。
*   **功能**: 作为一个适配器 (Adapter)，它封装了 `DPTestResultParser` 的解析逻辑，并将返回的字典数据映射为特定的对象属性（如 `data_f`, `diff_fx`），同时执行了一些字段重命名（`pred_e` -> `pred_Energy`）。

### 1.2 依赖关系图谱
通过静态代码分析，确认该类是当前核心业务流程的关键节点：

*   **生产者 (Instantiation)**:
    *   `src/dpeva/workflows/collect.py`: 在主动学习的数据采集阶段，实例化该类以加载多模型的预测结果。
*   **消费者 (Usage)**:
    *   `src/dpeva/uncertain/calculator.py`: `UQCalculator` 深度依赖该类的对象属性结构（如 `.data_f['pred_fx']`, `.diff_fx`）来进行不确定度（QbC/RND）计算。
*   **测试 (Testing)**:
    *   `tests/unit/test_parser.py`: 包含专门针对该类的兼容性测试 `test_dptestresults_compatibility`。

### 1.3 存在的问题
1.  **功能冗余**: 差值计算逻辑与 `src/dpeva/inference/stats.py` 中的 `StatsCalculator` 高度重叠。
2.  **违反单一职责 (SRP)**: 该类混合了数据解析调用和业务逻辑处理（差值计算），导致维护困难。
3.  **接口不一致**: 新的推理模块 (`infer.py`) 使用“Parser + StatsCalculator”模式，而旧的采集模块使用 `DPTestResults`，造成项目架构割裂。

---

## 2. 重构决策

**结论**: **废弃并移除 `DPTestResults` 类。**

**核心策略**: 采用 **"Replace Wrapper with Parser + Calculator"** 模式。
*   **数据层**: 统一使用 `DPTestResultParser` 返回的标准字典（或引入轻量级 `Dataclass`）。
*   **逻辑层**: 将差值计算和统计逻辑迁移至 `StatsCalculator` 或 `UQCalculator` 内部。

---

## 3. 详细重构方案

### 阶段一：基础设施准备 (Infrastructure)
目标：确保替代组件具备完整功能。

1.  **定义标准数据结构**:
    *   在 `src/dpeva/io/types.py` (新建或现有) 中定义 `PredictionData` (Dataclass)，包含 `energy`, `force`, `virial` 等字段，规范化数据传递。
2.  **增强 `StatsCalculator`**:
    *   位置: `src/dpeva/inference/stats.py`
    *   任务: 确保其能方便地输出 `UQCalculator` 所需的原始差值数据（如 `f_diff`），不仅仅是 RMSE 指标。

### 阶段二：业务逻辑迁移 (Migration)
目标：切断核心业务对 `DPTestResults` 的依赖。

1.  **重构 `UQCalculator`**:
    *   位置: `src/dpeva/uncertain/calculator.py`
    *   修改 `compute_qbc_rnd` 方法签名，不再接收 `DPTestResults` 对象，改为接收 `PredictionData` 或标准字典。
    *   内部逻辑修改：使用传入的数据直接计算（或调用 `StatsCalculator` 计算）所需的差值，不再依赖 `.diff_fx` 等预计算属性。
2.  **重构 `CollectionWorkflow`**:
    *   位置: `src/dpeva/workflows/collect.py`
    *   移除 `DPTestResults` 的实例化。
    *   改为：直接调用 `DPTestResultParser.parse()` 获取数据，组装成 `PredictionData`，然后传递给重构后的 `UQCalculator`。

### 阶段三：清理与收尾 (Cleanup)
目标：移除遗留代码。

1.  **删除代码**:
    *   删除 `src/dpeva/io/dataproc.py` 中的 `DPTestResults` 类定义。
2.  **移除过时测试**:
    *   删除 `tests/unit/test_parser.py` 中的 `test_dptestresults_compatibility`。
3.  **验证**:
    *   运行所有单元测试，特别是 `test_filter_uq.py` 和 `test_collect_workflow_routing.py`，确保业务逻辑未受影响。

---

## 4. 实施时间表建议

*   **Step 1 (Immediate)**: 保持现状，但在 `DPTestResults` 中添加 `DeprecationWarning`（已完成）。
*   **Step 2 (Short-term)**: 完成 `UQCalculator` 的重构，使其支持字典输入，打破强耦合。
*   **Step 3 (Mid-term)**: 更新 `collect.py`，彻底移除 `DPTestResults` 引用。
