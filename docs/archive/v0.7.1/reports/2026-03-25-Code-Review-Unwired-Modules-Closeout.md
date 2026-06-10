---
title: Unwired Modules Governance Closeout
status: active
audience: developers
last-updated: 2026-03-25
owner: DP-EVA Maintainers
---

# 2026-03-25 未接入模块治理收官总结

## 1. 收官范围

- Collection 出图链路治理（含 Candidate parity 接线、门控、日志可观测性）。
- Collection/Workflow 级“未接入模块”审查与结构化结论沉淀。
- 文档治理能力补齐（互链、审查模板、开发规范边界约定）。

## 2. 已完成项

### 2.1 Collection 主链路修复
- 已将以下图像接入标准 `dpeva collect` 出图链路（在 GT 条件满足时自动触发）：
  - `UQ-QbC-Candidate-fdiff-parity.png`
  - `UQ-RND-Candidate-fdiff-parity.png`
- 已补齐条件门控：
  - `has_gt=False` 或 `diff_maxf_0_frame` 无效时，跳过 force-error 依赖图并记录原因。
  - `uq_rnd_rescaled` 缺失时，回退 raw RND 仅用于过滤，并跳过 rescaled 依赖图。
- 已标准化日志标签：
  - `COLLECT_UQ_FALLBACK`
  - `COLLECT_PLOT_SKIPPED`

### 2.2 未接入模块治理
- 已完成 `train/infer/analysis/feature/collect/label/clean` 入口到实现映射审查。
- 已输出未接入候选的结构化清单（模块位置、状态、证据、建议、优先级）。
- 已落地优先级路线图（P0-P3）并推进到 P3-2。

### 2.3 文档治理固化
- 三份核心报告已完成互链，形成证据闭环。
- 已新增“未接入模块审查模板”，用于后续变更复用。
- 已在开发指南写入“包级导出层 vs 主链路入口”边界约定。

## 3. 关键产物索引

- 出图全量审计：`Collection Workflow 全量出图审计报告.md`
- 入口映射审查：`2026-03-25-Code-Review-Workflow-Entry-Mapping.md`
- 治理路线图：`2026-03-25-Code-Review-Unwired-Modules-Roadmap.md`
- 审查模板：`docs/governance/tools/unwired-module-audit-template.md`
- 开发规范边界：`docs/guides/developer-guide.md`（2.3 节）

## 4. 回归与门禁结果

- 单测回归：`tests/unit/workflows/test_collect_refactor.py` 与 `tests/unit/uncertain/test_visualization.py` 通过。
- 质量检查：`ruff check .` 通过。
- 文档治理：`scripts/doc_check.py`、`scripts/check_docs_freshness.py --days 90` 通过。
- 文档构建：`make -C docs html SPHINXOPTS="-W --keep-going"` 通过。

## 5. 剩余建议（非阻塞）

- 持续提升 active docs 的 owner 覆盖率（当前为 non-blocking 告警）。
- 在后续 release 归档时，将本收官报告迁入 `docs/archive/vX.Y.Z/reports/` 并更新归档索引。
