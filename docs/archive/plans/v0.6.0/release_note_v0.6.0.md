# DP-EVA v0.6.0 Release Note

- Status: active
- Audience: Users / Developers / Maintainers
- Release-Date: 2026-03-09
- Previous-Version: v0.5.3
- Related:
  - /docs/plans/iteration_plan_v0.6.md
  - /docs/plans/v0.6-acceptance-record.md

## 1. 发布摘要

v0.6.0 聚焦“数据闭环 + 分析能力 + 文档治理”三项主线，完成从候选样本到下一轮训练集的最小闭环能力，并补齐对应测试与验收资产。

## 2. 主要变更

### 2.1 Analysis 双模式

- 新增 `dataset` 模式：直接面向数据集做统计分析与可视化输出。
- 保持 `model_test` 模式兼容：原 `result_dir` 路径用法不变。
- 输出增强：
  - `dataset_stats.json`
  - `dataset_frame_summary.csv`
  - 能量/力/virial/压力分布图

### 2.2 Labeling 自动整合（Integration）

- 在 `LabelingWorkflow.collect_and_export` 后接入 Data Integration。
- 支持将 `outputs/cleaned` 与可选 `existing_training_data_path` 合并成下一代训练集。
- 新增一致性校验与统计摘要：
  - `integration_summary.json`
  - `existing/new/merged/filtered` 计数
  - `atom_names` 兼容性检查

### 2.3 文档与技能治理

- 新增 `.trae/skills/analysis.md`，覆盖 `model_test/dataset` 双模式示例与排障。
- 已完成 active/archive/spec 边界治理，避免 archive 反向成为现行规范源。
- CLI 与开发者指南已与实现对齐。

## 3. 配置与兼容性

### 3.1 AnalysisConfig

- 新增字段：
  - `mode`: `model_test`（默认） / `dataset`
  - `dataset_dir`: dataset 模式输入目录
- 兼容策略：
  - 旧配置（仅 `result_dir`）继续可用
  - `mode=model_test` 时 `result_dir` 必填
  - `mode=dataset` 时 `dataset_dir` 必填

### 3.2 LabelingConfig

- 新增字段：
  - `integration_enabled`
  - `existing_training_data_path`
  - `merged_training_data_path`
  - `integration_deduplicate`
- 默认行为：
  - 默认不启用自动整合（保持旧流程）
  - 默认不去重（优先保证正确性）

## 4. 验证与验收证据

执行：

```bash
pytest tests/unit/analysis/test_dataset_manager.py tests/unit/labeling/test_integration.py tests/unit/workflows/test_analysis_workflow.py tests/unit/workflows/test_labeling_workflow.py tests/integration/test_e2e_cycle.py -q
```

结果：`13 passed`。

## 5. 文档治理边界复核结论

- 结论：当前文档集不需要做目录级重组。  
- 理由：
  - active 规范位于 `docs/guides`、`docs/reference`、`docs/policy`、`docs/governance`，边界清晰。
  - `docs/archive` 当前仅承载历史文档，未发现 active 状态文档混入。
  - 执行型规格已统一落在 `docs/plans`，符合既定策略。
- 后续建议：
  - 持续做链接与结构漂移巡检；不做一次性大迁移。

## 6. 已知限制

- Integration 当前一致性校验以 `atom_names` 为最小规则，后续可扩展更多物理/拓扑一致性检查。
- E2E 仍以本地后端最小闭环为主，Slurm 场景建议在环境可用时补充发布后回归。
