### 变量管理体系审查报告 (Variable Management System Review)

#### 1. 审计概览
本次审查覆盖了 `config.py` (变量入口)、`constants.py` (常量定义)、`INPUT_PARAMETERS.md` (用户文档) 以及核心业务逻辑代码。整体上，项目采用了 Pydantic V2 进行强类型的配置管理，代码结构清晰，大部分默认值已通过 `constants.py` 集中管理。

**主要发现：**
- **严重缺陷 (CRITICAL)**：采集工作流 (`CollectionWorkflow`) 与 UQ 计算器 (`UQCalculator`) 存在严重的**硬编码耦合**，强制依赖 4 个模型，破坏了 `TrainingConfig` 中 `num_models` 的可配置性。
- **重要缺陷 (MAJOR)**：部分高级配置参数（如手动 Trust Overrides、自动 UQ 边界字典）在代码中存在但在用户文档中缺失。
- **次要缺陷 (MINOR)**：存在少量字符串字面量未提取为常量；部分配置项未在文档中对齐。

---

#### 2. 问题清单与修复建议

##### 🔴 CRITICAL (严重)

**1. 模型数量硬编码 (Hardcoded Model Count)**
- **位置**:
  - `src/dpeva/workflows/collect.py`: Line 513 (`for i in range(4):`), Line 542 (`calculator.compute_qbc_rnd(preds[0], ..., preds[3])`)
  - `src/dpeva/uncertain/calculator.py`: Line 22 (`def compute_qbc_rnd(..., predictions_3)`)
- **问题描述**: 代码显式硬编码了 `4` 个模型。尽管 `TrainingConfig.num_models` 允许用户设置模型数量（如 3 或 5），但在采集与 UQ 阶段，如果模型数量不等于 4，代码将直接崩溃（`IndexError` 或参数缺失）。
- **修复建议**:
  - 重构 `UQCalculator.compute_qbc_rnd` 接收 `List[PredictionData]` 而非固定参数。
  - 在 `CollectionConfig` 中增加 `num_models` 参数（或从元数据中自动推断），并使用该值替代 `range(4)`。

##### 🟠 MAJOR (重要)

**2. 文档与代码不一致 (Documentation Gap)**
- **位置**: `src/dpeva/config.py` vs `docs/parameters/INPUT_PARAMETERS.md`
- **问题描述**: 以下代码中存在的用户可配置参数未在文档中说明：
  - `CollectionConfig.config_path`: 用于 Slurm 自提交的路径参数。
  - `CollectionConfig.uq_qbc_trust_ratio` / `width`: 手动覆盖 QbC 特定参数。
  - `CollectionConfig.uq_rnd_rescaled_trust_ratio` / `width`: 手动覆盖 RND 特定参数。
  - `CollectionConfig.uq_auto_bounds`: 用于限制自动 UQ 边界的字典配置。
  - `FeatureConfig.mode`: (`cli` vs `python`) 缺失说明。
- **修复建议**: 更新 `INPUT_PARAMETERS.md`，补全上述参数的说明表格。

##### 🟡 MINOR (次要)

**3. 字符串字面量 (String Literals)**
- **位置**: `src/dpeva/config.py`
- **问题描述**:
  - `AnalysisConfig.output_dir` 默认为 `Path("analysis")`。
  - `TrainingConfig.input_json_path` 使用常量 `DEFAULT_INPUT_JSON` (Good)，但部分逻辑中仍有 `"input.json"` 字面量出现的风险（如 grep 结果所示）。
- **修复建议**: 将 `"analysis"` 提取到 `constants.py` 中作为 `DEFAULT_ANALYSIS_OUTPUT_DIR`。

**4. 冗余变量/导入 (Redundant Variables)**
- **位置**: `src/dpeva/workflows/collect.py`
- **已修复**: 已移除未使用的 `import dpdata`。
- **状态**: 静态分析显示目前无明显未使用的局部变量。

---

#### 3. 变量追踪矩阵 (Variable Traceability Matrix)

下表展示了核心用户变量从定义、配置到使用的流向：

| 变量类别 | 变量名 | 定义位置 (`config.py`) | 默认值源 (`constants.py`) | 核心使用位置 | 文档状态 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Common** | `work_dir` | `BaseWorkflowConfig` | `Path.cwd` (内置) | `dpeva.utils.config` | ✅ |
| **Common** | `omp_threads` | `BaseWorkflowConfig` | `DEFAULT_OMP_THREADS` | `dpeva.env` (隐式) | ✅ |
| **Submit** | `backend` | `SubmissionConfig` | `DEFAULT_BACKEND` | `JobManager` | ✅ |
| **Train** | `num_models` | `TrainingConfig` | `DEFAULT_NUM_MODELS` | `Trainer` | ✅ |
| **Collect** | `uq_trust_ratio` | `CollectionConfig` | `DEFAULT_UQ_TRUST_RATIO` | `CollectionWorkflow` | ✅ |
| **Collect** | `uq_qbc_trust_lo`| `CollectionConfig` | `None` | `CollectionWorkflow` -> `UQFilter` | ✅ |
| **Collect** | `num_models` | **MISSING** | **N/A** | `CollectionWorkflow` (Hardcoded `4`) | ❌ |

---

#### 4. 下一步行动建议

根据本次审查，建议您执行以下操作以完善变量管理体系：

1.  **重构 UQ 模块** (Priority: High): 修复 `UQCalculator` 和 `CollectionWorkflow` 以支持动态模型数量（N models）。
2.  **更新开发者文档** (Priority: Medium): 将缺失的参数补充到 `INPUT_PARAMETERS.md`。
3.  **常量提取** (Priority: Low): 将 `"analysis"` 等剩余字面量移入 `constants.py`。

您是否希望我先**着手修复 Critical 级别的“模型数量硬编码”问题**，或者**先完成文档的补全工作**？