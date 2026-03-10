---
title: Archived Document
status: archived
audience: Historians
last-updated: 2026-03-09
---

# Labeling Workflow Refactoring Specification (v0.5.3)

## 1. 概述 (Overview)
本设计文档旨在解决 Labeling Workflow 在多数据池（Multi-Systems）场景下对 Dataset 概念理解偏差的问题。通过重构数据加载、任务元数据管理及统计逻辑，实现对 Dataset -> System -> Task 的精确追溯与分层统计。

## 2. 核心概念定义 (Definitions)

| 概念 | 定义 | 示例 | 来源 |
| :--- | :--- | :--- | :--- |
| **Dataset** | 数据的最高层级组织单元，通常对应一个物理性质相近的数据池。在文件系统中体现为包含多个 System 的父目录（多数据池模式）或 System 目录本身（单数据池模式）。 | `DeepCNT`, `oc22-FeCOH` | `sampled_dpdata` 下的一级子目录名 |
| **System** | 具有相同原子类型和数量的结构集合。对应 `dpdata.System` 对象。 | `C0Fe38`, `C200Fe54` | Dataset 目录下的子目录名 |
| **Task** | 单个第一性原理计算任务，对应 System 中的某一帧。 | `C0Fe38_0` | `[SystemName]_[FrameIdx]` |
| **Structure Type** | 结构的几何分类（如团簇、层状、块体）。 | `cluster`, `layer` | `StructureAnalyzer` 自动分析 |

## 3. 详细设计 (Detailed Design)

### 3.1 数据加载逻辑重构 (Data Loading)
**涉及模块**: `dpeva.workflows.labeling.LabelingWorkflow`

**变更点**:
- 废弃将所有数据扁平化加载到单一 `dpdata.MultiSystems` 的做法。
- 引入 `DatasetRegistry` 结构，显式维护 `Dataset Name` 到 `MultiSystems` 的映射。

**逻辑流程**:
1. 扫描 `input_data_path`。
2. **多数据池判定**: 检查子目录是否包含 `type.raw` 或 `type_map.raw` (System特征) 或仅包含子目录 (Dataset特征)。
    - **Case A (多数据池)**: 输入目录包含多个子目录（如 `DeepCNT`, `oc22`），且这些子目录本身是 MultiSystems。
        - `Dataset Name` = 子目录名 (e.g., `DeepCNT`)
        - 加载该子目录下的所有 Systems。
    - **Case B (单数据池)**: 输入目录本身就是一个 MultiSystems（包含多个 System 子目录或直接是 System）。
        - `Dataset Name` = 输入目录名 (e.g., `DeepCNT`)
        - 加载输入目录下的所有 Systems。
3. 将加载结果组织为 `Dict[str, dpdata.MultiSystems]` 传递给 Manager。

### 3.2 目录结构与元数据 (Directory & Metadata)
**涉及模块**: `dpeva.labeling.manager.LabelingManager`, `dpeva.labeling.generator.AbacusGenerator`

**目录结构变更**:
- **旧**: `inputs/[SystemName]/[Type]/[TaskName]` (错误地将 SystemName 当作 Dataset)
- **新**: `inputs/[DatasetName]/[Type]/[TaskName]`

**`task_meta.json` Schema 变更**:
```json
{
  "dataset_name": "DeepCNT",       // 新增: 对应上层数据池名称
  "system_name": "C0Fe38",         // 新增: 对应 dpdata System 名称
  "stru_type": "cluster",
  "task_name": "C0Fe38_0",         // 保持: [SystemName]_[FrameIdx]
  "frame_idx": 0
}
```

### 3.3 统计与导出 (Statistics & Export)
**涉及模块**: `dpeva.labeling.manager.LabelingManager`

**统计维度**:
1. **按 Dataset 统计**:
    - **Total Submitted**: 总提交数
    - **Converged**: 收敛数 (存在于 `CONVERGED` 目录)
    - **Failed**: 失败数 (仍残留于 `inputs` 目录或标记为失败)
    - **Cleaned (Valid)**: 清洗后保留数 (满足 Energy/Force Criteria)
    - **Filtered (Invalid)**: 清洗被剔除数
2. **Global 汇总**: 所有 Dataset 的加和。

**实现策略**:
- 在 `collect_and_export` 阶段：
    - 扫描 `CONVERGED` 目录获取 `Converged` 任务。
    - 扫描 `inputs` 目录（递归寻找 `task_meta.json` 或 `INPUT` 文件）获取 `Failed` 任务（未被移动到 CONVERGED 的即视为 Failed）。
    - 读取每个任务的 `task_meta.json` 以归类到对应的 Dataset。
    - 结合 `AbacusPostProcessor` 的清洗结果计算 `Cleaned` 和 `Filtered`。

## 4. 开发计划 (Development Plan)

1. **Step 1: Interface Update**
   - 修改 `LabelingManager.prepare_tasks` 接口，使其接受 `Dict[str, dpdata.MultiSystems]`。
   - 修改 `AbacusGenerator.generate` 接口，增加 `system_name` 参数，并更新 `task_meta.json` 生成逻辑。

2. **Step 2: Workflow Logic**
   - 重写 `LabelingWorkflow.run` 中的数据加载部分，实现单/多数据池的智能识别与 Dataset 字典构建。

3. **Step 3: Path & Packing**
   - 更新 `LabelingManager` 中的路径构建逻辑，使用 `dataset_name` 作为第一级目录。
   - 确保 `TaskPacker` 能兼容新的目录层级（目前 Packer 是递归扫描，应能自动适配，需验证）。

4. **Step 4: Statistics Implementation**
   - 重构 `LabelingManager.collect_and_export`。
   - 增加对 `inputs` 目录的扫描以统计 `Failed` 任务。
   - 实现按 Dataset 分组的详细统计日志输出。

5. **Step 5: Verification**
   - 使用 `test/fp-setting-2` 数据集进行验证。
   - 检查 `task_meta.json` 内容是否正确。
   - 检查最终输出的统计日志是否包含 Dataset 维度的详细数据。

## 5. 验收标准 (Acceptance Criteria)
1. `inputs` 目录下的第一级子目录必须是 `DeepCNT`, `oc22-FeCOH` 等 Dataset 名称。
2. `task_meta.json` 中必须包含正确的 `dataset_name` 和 `system_name`。
3. Labeling 结束后，日志中必须打印如下格式的统计表：
   ```text
   Dataset: DeepCNT
     Total: 100, Converged: 95, Failed: 5
     Cleaned: 90, Filtered: 5
   Dataset: oc22-FeCOH
     ...
   Global Summary:
     ...
   ```
