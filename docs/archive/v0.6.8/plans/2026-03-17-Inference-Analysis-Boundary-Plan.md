# Inference 与 Analysis 边界重构计划（调研版）

## Summary

- 结论判断：当前“`backend=local` 自动分析、`backend=slurm` 手动分析”的设计在短期是**可用且务实**的，但在长期存在职责混杂与契约不一致风险，不是最优工程边界。
- 核心建议：将“作图与统计”收敛为 Analysis Workflow 的单一职责；Inference Workflow 仅负责批量执行 `dp test` 与产物落盘。Inference 中保留“可选链式触发 Analysis”的编排能力，但不内嵌分析实现。
- 本计划目标：先消除高风险耦合（尤其 `results_prefix`/`head` 契约），再渐进完成职责解耦，保证向后兼容与用户体验稳定。

## Current State Analysis

### 1) 已确认的调用链与现状行为

- CLI `dpeva infer` 直接进入 `InferenceWorkflow.run()`：`src/dpeva/cli.py`。
- Inference 在构造阶段即依赖分析管理器：`src/dpeva/workflows/infer.py` 中 `UnifiedAnalysisManager(...)`。
- 自动分析触发条件仅为 `backend == "local"`：`src/dpeva/workflows/infer.py`。
- `slurm` 模式只提交任务，不触发分析：`src/dpeva/workflows/infer.py`。
- 分析绘图实际实现位于 `UnifiedAnalysisManager -> InferenceVisualizer`：`src/dpeva/analysis/managers.py`。

### 2) 工程化评估（结合项目代码风格要求）

- 合理点：
  - 本地调试路径短，用户一条命令即可看到图与指标，实用性强。
  - 复用同一分析组件，减少重复逻辑。
- 风险点：
  - **职责混杂**：Inference 同时承担“任务执行 + 后处理分析”，偏离单一职责。
  - **契约不一致（高优先级）**：Inference 解析结果可用 `results_prefix`，而 AnalysisIOManager 里 parser `head="results"` 写死，独立 analysis 在非默认前缀下有失败风险。
  - **模块语义反向依赖**：analysis 层依赖 inference 命名空间中的可视化与统计实现（语义上应是共享后处理层）。
  - **运行语义分叉**：同一 infer 命令因 backend 不同而改变是否分析，容易造成用户心智负担。

### 3) 架构判断（回答“是否要完全独立”）

- 不建议“一步到位硬切”直接移除 Inference 中分析入口；这会损害本地快速迭代体验。
- 建议采用“**分析能力独立 + 编排触发可选**”：
  - 能力与实现完全归 Analysis Workflow；
  - Inference 仅在本地模式下可选触发 analysis 命令（作为 orchestration），不直接调用分析实现。

## Proposed Changes

### Phase A（先稳态，消风险，不改用户习惯）

#### A1. 统一结果文件契约，消除前缀不一致
- 文件：
  - `src/dpeva/config.py`
  - `src/dpeva/workflows/analysis.py`
  - `src/dpeva/analysis/managers.py`
  - `examples/recipes/analysis/config_analysis.json`
  - `docs/source/guides/configuration.md`
- 变更：
  - 在 `AnalysisConfig` 新增 `results_prefix`（默认 `results`，与 Inference 对齐）。
  - `AnalysisIOManager.load_data(...)` 不再写死 `head="results"`，改为从配置透传。
  - 更新 analysis recipe 与文档示例，明确“若 infer 自定义前缀，analysis 必须同前缀”。
- 目的：
  - 修复最关键的跨工作流隐式耦合错误源。

#### A2. 明确“执行/分析”边界开关（保持兼容）
- 文件：
  - `src/dpeva/config.py`
  - `src/dpeva/workflows/infer.py`
  - `examples/recipes/inference/config_infer.json`
  - `docs/source/guides/cli.md`
- 变更：
  - 新增 `InferenceConfig.auto_analysis: bool = False`。
  - 行为规则：
    - `backend=local & auto_analysis=true`：触发分析编排；
    - 其他情况：仅执行推理并记录提示。
  - 不再以 backend 作为唯一隐式开关，改为“显式用户意图优先”。
- 目的：
  - 降低行为分叉的隐式性，符合 Explicit is better than implicit。

### Phase B（职责解耦，保持链式体验）

#### B1. 将分析编排迁移为独立入口调用
- 文件：
  - `src/dpeva/workflows/infer.py`
  - `src/dpeva/workflows/analysis.py`
  - `src/dpeva/cli.py`（若需补充内部调用辅助）
- 变更：
  - 从 InferenceWorkflow 移除 `UnifiedAnalysisManager` 直接依赖。
  - Inference 仅产出一个标准 analysis 配置（或调用内部公共函数），触发 AnalysisWorkflow 运行。
  - 失败处理保持模型级隔离 + 全局可感知失败码。
- 目的：
  - 让 Inference 真正只做执行，Analysis 真正只做统计绘图。

#### B2. 共享后处理能力下沉（命名空间解耦）
- 文件（建议新建/迁移）：
  - `src/dpeva/postprocess/stats.py`（由 `inference/stats.py` 迁移）
  - `src/dpeva/postprocess/visualizer.py`（由 `inference/visualizer.py` 迁移）
  - `src/dpeva/analysis/managers.py`
  - `src/dpeva/inference/managers.py`（仅保留执行相关）
- 变更：
  - 统计与绘图能力从 inference 语义域迁移到共享后处理域。
  - 提供向后兼容 import 过渡层（短期保留旧路径 re-export）。
- 目的：
  - 语义一致、依赖方向清晰、后续维护成本更低。

### Phase C（文档与示例收敛）

#### C1. 用户入口双路径清晰化
- 文件：
  - `examples/recipes/inference/config_infer.json`
  - `examples/recipes/analysis/config_analysis.json`
  - `examples/recipes/README.md`
  - `docs/source/guides/cli.md`
  - `docs/source/guides/configuration.md`
- 变更：
  - 明确两种推荐用法：
    - 快速本地链式：infer（可选 auto_analysis）。
    - 生产/HPC标准：infer（slurm）→ analysis（独立命令）。
  - 给出最小配置示例与常见错误（prefix 不一致、result_dir 指向不对）。

## Assumptions & Decisions

- 决策 1：不做激进破坏性变更，优先兼容现有用户入口。
- 决策 2：以“显式开关 + 独立工作流”替代“backend 隐式行为”。
- 决策 3：短期允许 Inference 触发 Analysis 编排，但不持有分析核心实现。
- 假设：现有 `dpeva analysis` 命令在用户环境可独立运行，且结果目录结构稳定。

## Verification Steps

### 1) 单元测试

- 更新/新增：
  - `tests/unit/workflows/test_infer_workflow_exec.py`
  - `tests/unit/workflows/test_analysis_workflow.py`
  - `tests/unit/analysis/test_analysis_io_manager.py`（新增）
- 覆盖点：
  - `auto_analysis` 显式开关行为。
  - `results_prefix` 在 infer 与 analysis 的一致透传。
  - backend 差异下的行为可预测性（不再靠隐式分支）。

### 2) 回归验证

- `pytest tests/unit/workflows/test_infer_workflow_exec.py`
- `pytest tests/unit/workflows/test_analysis_workflow.py`
- `pytest tests/unit/analysis`

### 3) 端到端验收（最小）

- local 链式：`dpeva infer <local-config-with-auto_analysis>`
  - 验证推理产物与 analysis 产物均生成。
- slurm 解耦：`dpeva infer <slurm-config>` 后执行 `dpeva analysis <analysis-config>`
  - 验证两阶段均成功，且前缀可自定义并一致读取。

### 4) 文档一致性

- 核查 `docs/source/guides/*.md` 与 `examples/recipes/*.json` 参数一致。
- 防止出现“文档说自动分析，配置却无开关”的漂移。

## 执行顺序（实施时）

1. A1 契约统一（必须先做）  
2. A2 显式开关（保持兼容）  
3. B1 编排迁移（消除直接实现耦合）  
4. B2 能力下沉（语义解耦）  
5. C1 文档与示例收敛（收尾）
