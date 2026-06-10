---
title: Workflow Entry Mapping Review
status: active
audience: developers
last-updated: 2026-03-25
owner: DP-EVA Maintainers
---

# 2026-03-25 Workflow 入口到实现映射审查报告

## 1. 审查范围

- 命令范围：`train / infer / analysis / feature / collect / label / clean`
- 审查目标：
  - 建立“CLI 入口 -> 配置模型 -> Workflow -> 核心实现模块”的可追溯映射
  - 识别“可能未接入主链路”的模块候选，并给出证据

## 关联报告

- [Collection Workflow 全量出图审计报告.md](Collection%20Workflow%20全量出图审计报告.md)：Collection 全量出图索引、可生成性状态与缺口跟踪。
- [2026-03-25-Code-Review-Unwired-Modules-Roadmap.md](2026-03-25-Code-Review-Unwired-Modules-Roadmap.md)：未接入模块治理优先级路线图与阶段进展。

## 2. 入口到实现映射

| CLI 子命令 | CLI 入口函数 | 配置模型 | Workflow 实现 | 核心实现模块（Workflow 内调用） | 关键证据 |
|---|---|---|---|---|---|
| `train` | `handle_train` | `TrainingConfig` | `TrainingWorkflow` | `TrainingIOManager`、`TrainingConfigManager`、`TrainingExecutionManager` | `src/dpeva/cli.py` L77-89；`src/dpeva/workflows/train.py` L6-9, L39-57, L75-134 |
| `infer` | `handle_infer` | `InferenceConfig` | `InferenceWorkflow` | `InferenceIOManager`、`InferenceExecutionManager`；可选链式 `AnalysisWorkflow` | `src/dpeva/cli.py` L90-102；`src/dpeva/workflows/infer.py` L6-8, L41-48, L60-111, L112-171 |
| `analysis` | `handle_analysis` | `AnalysisConfig` | `AnalysisWorkflow` | `AnalysisIOManager`、`UnifiedAnalysisManager`、`DatasetAnalysisManager` | `src/dpeva/cli.py` L130-141；`src/dpeva/workflows/analysis.py` L8-11, L50-61, L63-90 |
| `feature` | `handle_feature` | `FeatureConfig` | `FeatureWorkflow` | `FeatureIOManager`、`FeatureExecutionManager`、`DescriptorGenerator` | `src/dpeva/cli.py` L103-115；`src/dpeva/workflows/feature.py` L5-8, L49-59, L64-140 |
| `collect` | `handle_collect` | `CollectionConfig` | `CollectionWorkflow` | `CollectionIOManager`、`UQManager`、`SamplingManager` | `src/dpeva/cli.py` L116-129；`src/dpeva/workflows/collect.py` L8, L24-27, L76-102, L142-150 |
| `label` | `handle_label` | `LabelingConfig` | `LabelingWorkflow` | `LabelingManager`、`JobManager`、`DataIntegrationManager` | `src/dpeva/cli.py` L143-171；`src/dpeva/workflows/labeling.py` L18-23, L39-48, L50-60, L117-135 |
| `clean` | `handle_clean` | `DataCleaningConfig` | `DataCleaningWorkflow` | `DPTestResultParser`、`load_systems`、帧级过滤与导出逻辑 | `src/dpeva/cli.py` L173-184；`src/dpeva/workflows/data_cleaning.py` L10-15, L38-57 |

补充入口证据：

- 控制台入口绑定到 `dpeva.cli:main`，见 `pyproject.toml` L59-60。
- 子命令注册覆盖 7 个命令并绑定对应 handler，见 `src/dpeva/cli.py` L197-248。

## 3. 未接入模块候选（静态审查）

说明：以下为“候选”，表示其未在当前 `src/` 内主执行链路中出现直接导入或仅承担包导出占位职责，不等价于“必须删除”。

### 候选 A：`src/dpeva/workflows/__init__.py`

- 现状：仅做 Workflow 类 re-export（`__all__`），无业务逻辑。
- 证据：
  - 文件内容仅导入/导出，见 `src/dpeva/workflows/__init__.py` L1-15。
  - CLI 直接导入具体子模块（`dpeva.workflows.train` 等）而非包级导出，见 `src/dpeva/cli.py` L85, L98, L111, L124, L138, L151, L181。
- 结论：候选“未被主 CLI 调用链直接接入”的包级聚合模块。

### 候选 B：`src/dpeva/analysis/__init__.py`

- 现状：空文件。
- 证据：文件无有效内容，见 `src/dpeva/analysis/__init__.py`。
- 结论：候选“占位包初始化模块”。

### 候选 C：包级聚合/占位 `__init__`（`io/labeling/sampling/uncertain`）

- 现状：
  - `src/dpeva/io/__init__.py` 仅导出 `DPTestResultParser`（L1-4）
  - `src/dpeva/labeling/__init__.py` 仅导出若干类（L14-18）
  - `src/dpeva/sampling/__init__.py`、`src/dpeva/uncertain/__init__.py` 为描述/占位（无实际链路逻辑）
- 证据：上述文件内容显示其主要职责为包导出或占位；而主链路普遍按子模块路径直接导入（如 `dpeva.io.collection`、`dpeva.sampling.manager`、`dpeva.uncertain.manager`）。
- 结论：候选“未被当前主命令链路直接消费的包级入口”。

## 4. 结论

- 七个工作流命令均已在 CLI 中完成注册并接入具体 Workflow 实现，主链路可追溯。
- `infer` 存在可选内联分析路径（`auto_analysis && backend=local`），与独立 `analysis` 命令并行存在。
- 未接入候选主要集中在包级 `__init__` 聚合/占位模块，风险等级低，但建议在后续治理中明确其“对外 API”或“仅内部占位”定位，避免接口认知漂移。

## 5. 结构化审计清单（可复核）

| 模块位置 | 预期所属工作流 | 当前接入状态 | 证据摘要 | 处置建议 | 优先级 |
|---|---|---|---|---|---|
| `src/dpeva/workflows/__init__.py` | train/infer/analysis/feature/collect/label/clean | 保留占位（包级聚合） | 仅 `__all__` 导出，CLI 直接导入具体 workflow 子模块 | 保留；在开发文档补充“包级 API”定位 | P3 |
| `src/dpeva/analysis/__init__.py` | analysis | 保留占位（空模块） | 文件为空，不参与运行时逻辑 | 保留；后续若无计划可补充模块用途说明 | P3 |
| `src/dpeva/io/__init__.py` | clean/collect/label | 保留聚合 | 仅导出 `DPTestResultParser`，主链路按子模块导入 | 保留；继续限制其职责为统一导出层 | P3 |
| `src/dpeva/labeling/__init__.py` | label | 保留聚合 | 仅导出管理器类，主链路按 `workflows/labeling.py` 调用 | 保留；新增公共 API 时同步更新导出契约 | P3 |
| `src/dpeva/sampling/__init__.py` | collect | 保留占位 | 未承载 workflow 入口，主链路走 `sampling/manager.py` | 保留；避免在该文件叠加业务逻辑 | P3 |
| `src/dpeva/uncertain/__init__.py` | collect | 保留占位 | 未承载 workflow 入口，主链路走 `uncertain/manager.py` 与 `uncertain/visualization.py` | 保留；约束为命名空间占位 | P3 |

## 6. 与 Task 5 对齐的同步项

- 已补齐本审查报告的结构化审计清单，满足“模块位置/接入状态/证据/建议/优先级”可复核要求。
- 已同步更新 `docs/reports/Collection Workflow 全量出图审计报告.md` 中 Candidate parity 两图的可生成性状态为“可原生生成”。
- 已在 Collection 审计报告中移除 Candidate parity 两图的“无法直生图像缺口”条目，避免与当前实现状态冲突。
- 已补齐候选包级模块 `__init__` 的职责说明，进一步降低“聚合导出层与主链路入口混淆”的风险。
