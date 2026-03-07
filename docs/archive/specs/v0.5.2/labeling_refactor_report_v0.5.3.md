# Labeling Workflow Refactoring Report (v0.5.3)

## 1. 概述 (Overview)
本次开发针对 Labeling Workflow 在多数据池（Multi-Systems）场景下对 Dataset 概念的理解偏差进行了修复与重构。通过重新定义 Dataset 加载逻辑、增强任务元数据（Metadata）及实现分层统计，确保了工作流能精确追溯每个任务的数据来源，并提供细粒度的进度报告。

## 2. 核心变更 (Core Changes)

### 2.1 Dataset 加载逻辑重构
- **模块**: `dpeva.workflows.labeling.LabelingWorkflow`
- **变更**:
  - 实现了智能的数据加载策略，自动识别 **单数据池 (Single-Pool)** 与 **多数据池 (Multi-Pool)** 模式。
  - **Single-Pool**: 若输入目录本身包含 `type.raw` 等系统文件，则将其视为单一 Dataset（名称为目录名）。
  - **Multi-Pool**: 若输入目录包含多个子目录，则每个子目录视为一个独立 Dataset（名称为子目录名）。
  - **验证**: 在 `test/fp-setting-2` 测试中，正确识别了 `DeepCNT`, `oc22-FeCOH`, `omat24-FeCOH` 三个数据集。

### 2.2 任务元数据增强 (Metadata Injection)
- **模块**: `dpeva.labeling.generator.AbacusGenerator`
- **变更**:
  - `task_meta.json` 新增 `system_name` 字段，配合原有的 `dataset_name`，实现了完整的数据溯源。
  - **Schema**:
    ```json
    {
      "dataset_name": "DeepCNT",
      "system_name": "C0Fe38",
      "stru_type": "cluster",
      "task_name": "C0Fe38_0",
      "frame_idx": 0
    }
    ```
  - **验证**: 检查生成的 `inputs/N_2_0/C0Fe38_0/task_meta.json`，内容正确无误。

### 2.3 分层统计实现 (Hierarchical Statistics)
- **模块**: `dpeva.labeling.manager.LabelingManager`
- **变更**:
  - 重构了 `collect_and_export` 方法。
  - **Failed 任务统计**: 增加了对 `inputs` 目录的递归扫描，通过读取残留任务的 `task_meta.json` 统计失败任务。
  - **Converged 任务统计**: 基于 `CONVERGED` 目录和 `AbacusPostProcessor` 的清洗结果。
  - **分层报告**: 日志输出支持 Global -> Dataset -> Type 三级统计。

## 3. 测试验证 (Verification)

### 3.1 环境与配置
- **环境**: `dpeva-dev`
- **配置**: `test/fp-setting-2/config_gpu.json` (Slurm backend)
- **输入数据**: `test/fp-setting-2/dpdata/sampled_dpdata` (含 3 个 Dataset, 18 Systems)

### 3.2 验证结果
1. **加载阶段**:
   ```text
   INFO - Loading input data from .../sampled_dpdata
   INFO - Detected Multi-Pool mode.
   INFO - Loading dataset: DeepCNT
   INFO - Loading dataset: oc22-FeCOH
   INFO - Loading dataset: omat24-FeCOH
   INFO - Loaded 3 datasets, 18 systems total.
   ```
2. **生成阶段**:
   ```text
   INFO - Generated 26 tasks.
   INFO - Packed tasks into 13 job directories.
   ```
3. **统计功能 (Mock 验证)**:
   使用 `verify_stats.py` 模拟了包含 Failed 和 Converged 任务的场景，输出如下：
   ```text
   INFO:dpeva.labeling.manager:=== Labeling Statistics Report ===
   INFO:dpeva.labeling.manager:Global: Total=2, Converged=1, Failed=1, Cleaned=1, Filtered=0
   INFO:dpeva.labeling.manager:  Dataset: DS1                  (Total=2, Conv=1, Fail=1, Clean=1, Filt=0)
   INFO:dpeva.labeling.manager:    Type: cluster         -> Total=2, Conv=1, Fail=1, Clean=1, Filt=0
   ```

## 4. 结论 (Conclusion)
本次重构成功解决了 Dataset 概念混淆问题，并按需求实现了详细的统计报告功能。代码已通过集成测试（生成与提交）及单元验证（统计逻辑），具备发布条件。
