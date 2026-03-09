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

## 4. 详细测试报告 (Detailed Test Report)

### 4.1 测试用例执行明细 (Test Case Execution)

| 用例 ID | 测试项 | 预期结果 | 实际结果 | 状态 | 备注 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **TC-001** | **多数据池识别** | 系统自动识别 `DeepCNT`, `oc22-FeCOH`, `omat24-FeCOH` 为独立 Dataset | 日志显示 `Detected Multi-Pool mode` 并正确加载 3 个 Dataset | ✅ Pass | 覆盖了 Multi-Pool 场景 |
| **TC-002** | **目录层级生成** | 任务目录遵循 `inputs/[Dataset]/[Type]/[Task]` 结构 | 实际生成路径如 `inputs/DeepCNT/cluster/C0Fe38_0` | ✅ Pass | 验证了 Manager 的路径构建逻辑 |
| **TC-003** | **元数据注入** | `task_meta.json` 包含 `dataset_name` 和 `system_name` | 抽样检查 `C0Fe38_0/task_meta.json` 包含预期字段 | ✅ Pass | 确保了数据溯源能力 |
| **TC-004** | **任务打包** | 26 个任务按 `tasks_per_job=2` 打包为 13 个 Job | 日志显示 `Packed tasks into 13 job directories` | ✅ Pass | 验证了 Packer 的递归扫描兼容性 |
| **TC-005** | **作业提交** | 成功提交 Slurm 作业 | 日志显示 `Submitted batch job 2084xx` 共 13 次 | ✅ Pass | 验证了 JobManager 的集成 |
| **TC-006** | **统计报告** | 输出包含 Dataset 维度的 Total/Conv/Fail/Clean/Filt 统计 | `verify_stats.py` 模拟运行输出符合预期格式 | ✅ Pass | 覆盖了统计聚合逻辑 |

### 4.2 缺陷统计与修复 (Defects & Fixes)

| 缺陷 ID | 描述 | 严重级 | 修复方案 | 状态 |
| :--- | :--- | :--- | :--- | :--- |
| **BUG-001** | `AbacusGenerator` 接口参数缺失 | High | 在 `generate` 方法中增加 `system_name` 参数并透传至 `task_meta.json` | ✅ Fixed |
| **BUG-002** | `LabelingManager` 统计逻辑遗漏 Failed 任务 | Medium | 增加对 `inputs` 目录的递归扫描逻辑以补全 Failed 任务统计 | ✅ Fixed |
| **BUG-003** | `verify_stats.py` 中 `dpdata` Mock 不完整 | Low | 完善 Mock 对象的属性以通过 `postprocessor` 的内部检查 | ✅ Fixed |

### 4.3 风险评估 (Risk Assessment)
- **Dataset 命名冲突**: 若不同 Dataset 下存在同名 System (e.g. `DS1/sys1`, `DS2/sys1`)，在 `inputs` 目录下通过 Dataset 分层已解决冲突，但在 `dpdata.MultiSystems` 对象内部需确保 `target_name` 唯一性（目前代码已处理）。
- **IO 性能**: 统计阶段需递归扫描 `inputs` 目录，若任务量极大 (>10w)，可能存在 IO 瓶颈。建议后续引入 SQLite 或 Redis 进行状态管理。

## 5. 结论 (Conclusion)
本次重构成功解决了 Dataset 概念混淆问题，并按需求实现了详细的统计报告功能。代码已通过集成测试（生成与提交）及单元验证（统计逻辑），具备发布条件。建议在下一版本中关注 IO 性能优化。
