---
title: Archived Document
status: archived
audience: Historians
last-updated: 2026-03-09
---

# Labeling Workflow 单元测试设计方案 (v1.0)

## 1. 测试目标与范围
本方案旨在为 `dpeva.labeling` 模块及 `LabelingWorkflow` 建立完整的单元测试体系，确保重构后的多数据池（Multi-Pool）支持、元数据管理及分层统计逻辑的正确性与鲁棒性。

### 1.1 测试范围
- **核心逻辑**: `AbacusGenerator` (生成器), `StructureAnalyzer` (结构分析), `ResubmissionStrategy` (重算策略), `AbacusPostProcessor` (后处理).
- **业务流程**: `LabelingManager` (路径管理、任务打包、统计), `LabelingWorkflow` (数据加载路由).
- **数据结构**: `task_meta.json` 生成与解析, `inputs/[Dataset]/[Type]/[Task]` 目录结构.

### 1.2 覆盖率目标
- **行覆盖率 (Line Coverage)**: > 90%
- **分支覆盖率 (Branch Coverage)**: > 85%
- **核心算法 (Generator/Strategy)**: 100%

---

## 2. 测试策略与规范

### 2.1 测试分层
1. **Unit Tests (`tests/unit/labeling/`)**:
   - 针对 Generator, Packer, Strategy, PostProcessor 的独立测试。
   - 使用 Mock 隔离文件系统与外部依赖（dpdata, ase）。
2. **Workflow Tests (`tests/unit/workflows/test_labeling_workflow.py`)**:
   - 验证 `LabelingWorkflow` 的数据加载路由（Single vs Multi Pool）。
   - Mock `LabelingManager` 以验证调用参数。

### 2.2 Mock 策略
- **文件系统**: 使用 `pytest` 的 `tmp_path` fixture 创建临时目录，避免污染环境。
- **外部库**:
  - `dpdata`: Mock `MultiSystems` 和 `System` 对象，避免真实加载大文件。
  - `ase`: 使用 `ase.build.bulk` 创建合成原子结构。
  - `ase-abacus`: 显式 Mock `ase.io.abacus.write_input` 和 `ase.io.abacus.write_abacus`，避免对外部插件的强依赖及真实文件 I/O。
  - `subprocess`: Mock `run` 方法，避免执行真实命令。

### 2.3 数据构造
- **Dataset**: 构造包含 `dataset_name` 和 `system_name` 的虚拟数据集结构。
- **Atoms**: 构造 0D (Cluster), 1D (Wire), 2D (Layer), 3D (Bulk) 四种典型结构用于 Generator 测试。

---

## 3. 测试用例详细设计

### 3.1 `AbacusGenerator` 测试 (`tests/unit/labeling/test_generator.py`)

| 用例ID | 测试场景 | 输入 | 预期结果 |
| :--- | :--- | :--- | :--- |
| **GEN-001** | 结构类型识别 (Cluster) | 孤立原子结构 (大真空) | `stru_type`="cluster", KPT=[1,1,1] |
| **GEN-002** | 结构类型识别 (Layer) | XY周期性，Z方向真空 | `stru_type`="layer", `dip_cor_flag`=1 |
| **GEN-003** | 元数据注入 | `system_name`="sys1" | 生成 `task_meta.json` 且包含 `system_name` |
| **GEN-004** | 磁矩设置 | `mag_map`={"Fe": 5} | `INPUT` 或 `STRU` 中包含磁矩设置 |

### 3.2 `LabelingManager` 测试 (`tests/unit/labeling/test_manager.py`)

| 用例ID | 测试场景 | 输入 | 预期结果 |
| :--- | :--- | :--- | :--- |
| **MGR-001** | 路径生成 (Multi-Pool) | Dataset="DS1", Type="cluster" | 路径为 `inputs/DS1/cluster/task_0` |
| **MGR-002** | 任务打包 | 100个任务, `per_job`=50 | 生成 2 个打包目录 (`N_50_0`, `N_50_1`) |
| **MGR-003** | 统计逻辑 (Mixed) | 5 Converged, 2 Failed | 统计报告正确归类到对应 Dataset |
| **MGR-004** | 失败任务扫描 | `inputs` 中残留 INPUT 文件 | 被计入 Failed 统计 |

### 3.3 `LabelingWorkflow` 测试 (`tests/unit/workflows/test_labeling_workflow.py`)

| 用例ID | 测试场景 | 输入 | 预期结果 |
| :--- | :--- | :--- | :--- |
| **WF-001** | 单数据池加载 | 目录含 `type.raw` | 识别为 Single-Pool, Dataset=DirName |
| **WF-002** | 多数据池加载 | 目录含子目录 `DS1`, `DS2` | 识别为 Multi-Pool, 加载 2 个 Dataset |
| **WF-003** | 空目录处理 | 空目录 | 抛出 `ValueError` |

---

## 4. 执行计划

1. **环境准备**: 确保 `dpeva-dev` 环境就绪，安装 `pytest`, `pytest-cov`, `pytest-mock`。
2. **代码实现**:
   - 创建 `tests/unit/labeling/` 目录。
   - 编写 `test_generator.py`, `test_manager.py`, `test_workflow.py`。
3. **执行与回归**:
   - 运行 `pytest tests/unit/labeling`。
   - 运行 `pytest tests/unit/workflows/test_labeling_workflow.py`。
   - 检查覆盖率报告。

## 5. 交付物
- 测试代码文件。
- 测试执行报告（HTML/XML）。
- 更新后的 `developer-guide.md`（包含测试规范更新）。
