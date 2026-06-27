---
title: Technical Reports
status: active
audience: Developers
last-updated: 2026-06-27
owner: Docs Owner
---

# 技术报告（Technical Reports）

- Status: active
- Audience: All
- Last-Updated: 2026-06-27

本文档目录用于存放“项目级、可共享、可追溯”的技术结论，不承载执行过程草稿。

## 1. 报告类型

- **Summary Report**: 汇总多份审查或多条证据链的主入口报告。
- **Focused Report**: 针对 Summary Report 中某个主题的专题展开，不作为全局总结入口。
- **Experiment Series**: 围绕同一实验范式持续产出的对比报告，按系列组织。
- **Archived Report**: 已闭环、过时或强过程性的历史记录，迁入 `docs/archive/vX.Y.Z/reports/`。

## 2. 维护策略

- **Active**: 当前版本仍需被团队频繁引用的结论文档。
- **Archived**: 发布后或过时结论，归档至 `docs/archive/vX.Y.Z/reports/`。
- **Naming**: `YYYY-MM-DD-<topic>.md`
- **Ingestion Rule**: 过程记录先放 `.trae/documents/`；仅“收敛结论”才迁入本目录。
- **Index Rule**: 总报告、专题报告、实验系列和归档项必须分组展示，避免不同生命周期文档平铺并列。

## 3. AI 执行文档边界

- `.trae/documents/`：调试记录、迭代草稿、阶段性分析片段。
- `docs/reports/`：可对外复核、可长期引用、可纳入版本发布说明的最终结论。
- 迁入 `docs/reports/` 的前提：
  1. 结论可复核（含依据与验证结果）；
  2. 结论不再依赖即时上下文；
  3. 对后续开发或治理具有复用价值。

## 4. Active Summary Reports

- 当前无 active summary report。v0.8.0 release 总结已归档至 [docs/archive/v0.8.0/reports/2026-06-17-DP-EVA-v0.8.0-Release.md](../archive/v0.8.0/reports/2026-06-17-DP-EVA-v0.8.0-Release.md)。

## 5. Active Focused Reports

- 当前无 active focused report。已闭环专题报告请查看 [v0.8.0 归档报告索引](../archive/v0.8.0/reports/README.md)。

## 6. Active Experiment Series

### Sampling Comparison Series

- 当前无 active experiment series report。v0.8.0 sampling comparison 系列已归档至 [v0.8.0 归档报告索引](../archive/v0.8.0/reports/README.md)。

## 7. Recently Archived

- [docs/archive/v0.8.0/reports/2026-06-27-HDF5-Last-Layer-Collect-Routing.md](../archive/v0.8.0/reports/2026-06-27-HDF5-Last-Layer-Collect-Routing.md)
  - Type: Archived Focused Report
  - Status: archived
  - Reason: HDF5 `atomic_feature` last-layer collect routing, multi-pool embed output layout, and related config/recipe contract have been reviewed and documented
  - Related Plan: [docs/archive/v0.8.0/plans/2026-06-24-dp-embed-hdf5-plan.md](../archive/v0.8.0/plans/2026-06-24-dp-embed-hdf5-plan.md)
- [docs/archive/v0.8.0/reports/2026-06-24-DP-Embed-vs-Eval-Desc-Comparison.md](../archive/v0.8.0/reports/2026-06-24-DP-Embed-vs-Eval-Desc-Comparison.md)
  - Type: Archived Focused Report
  - Status: archived
  - Reason: DeepMD `dp embed` HDF5 适配、`eval-desc` 数值/资源对比和 DPA4 OOM 复现验证已闭环
  - Related Plan: [docs/archive/v0.8.0/plans/2026-06-24-dp-embed-hdf5-plan.md](../archive/v0.8.0/plans/2026-06-24-dp-embed-hdf5-plan.md)
- [docs/archive/v0.8.0/reports/2026-06-17-DP-EVA-v0.8.0-Release.md](../archive/v0.8.0/reports/2026-06-17-DP-EVA-v0.8.0-Release.md)
  - Type: Archived Summary Report
  - Status: archived
  - Reason: v0.8.0 release 总结已随本轮文档归档闭环
- [docs/archive/v0.8.0/reports/2026-06-08-DPA4-Neo-Air-Plus-Sampling-Comparison.md](../archive/v0.8.0/reports/2026-06-08-DPA4-Neo-Air-Plus-Sampling-Comparison.md)
  - Type: Archived Experiment Series Report
  - Status: archived
  - Reason: DPA4 Neo/Air/Plus 采样对比已闭环并纳入 v0.8.0 归档
- [docs/archive/v0.8.0/reports/2026-06-07-DPA4-Mini-MACE-Sampling-Comparison.md](../archive/v0.8.0/reports/2026-06-07-DPA4-Mini-MACE-Sampling-Comparison.md)
  - Type: Archived Experiment Series Report
  - Status: archived
  - Reason: DPA4 Mini/MACE 采样对比已闭环并纳入 v0.8.0 归档
- [docs/archive/v0.8.0/reports/2026-06-12-dpose-llpr-dashboard.html](../archive/v0.8.0/reports/2026-06-12-dpose-llpr-dashboard.html)
  - Type: Archived Experiment Dashboard
  - Status: archived
  - Reason: LLPR/DPOSE 与 DPA4 Mini UQ correlation 实验看板已闭环，并由 v0.8.0 归档索引承接
  - Related Plans: [docs/archive/v0.8.0/plans/2026-06-11-dpeva-native-dpose-llpr-plan.md](../archive/v0.8.0/plans/2026-06-11-dpeva-native-dpose-llpr-plan.md), [docs/archive/v0.8.0/plans/2026-06-13-dpa4-mini-uq-correlation.md](../archive/v0.8.0/plans/2026-06-13-dpa4-mini-uq-correlation.md)
- [docs/archive/v0.8.0/reports/2026-06-11-v0.8.0-atst-integration-progress-audit.md](../archive/v0.8.0/reports/2026-06-11-v0.8.0-atst-integration-progress-audit.md)
  - Type: Archived Focused Report
  - Status: archived
  - Reason: v0.8.0 ATST integration 最小验收闭环完成
  - Related Plan: [docs/archive/v0.8.0/plans/2026-06-10-v0.8.0-atst-integration-plan.md](../archive/v0.8.0/plans/2026-06-10-v0.8.0-atst-integration-plan.md)
- [docs/archive/v0.8.0/reports/2026-04-01-Code-Review-Repository-Audit.md](../archive/v0.8.0/reports/2026-04-01-Code-Review-Repository-Audit.md)
  - Type: Archived Summary Report
  - Status: archived
  - Reason: 仓库审查结论已被后续治理与 v0.8.0 门禁收口承接
- [docs/archive/v0.8.0/reports/2026-04-02-Code-Review-Unit-Test-Audit.md](../archive/v0.8.0/reports/2026-04-02-Code-Review-Unit-Test-Audit.md)
  - Type: Archived Focused Report
  - Status: archived
  - Reason: 单元测试审查结论已被后续测试与 CI 门禁收口承接
- [docs/archive/v0.7.1/reports/2026-04-05-Code-Review-Agents-Docs-Governance.md](../archive/v0.7.1/reports/2026-04-05-Code-Review-Agents-Docs-Governance.md)
  - Type: Archived Report
  - Status: archived
  - Reason: 已闭环的文档治理阶段性收口记录
  - Current Sources: [docs/guides/developer-guide.md](../guides/developer-guide.md), [docs/guides/docs-governance-quickstart.md](../guides/docs-governance-quickstart.md), [docs/policy/maintenance.md](../policy/maintenance.md)

## 8. 归档与索引要求

- 报告归档时，必须同步更新：
  - 当前目录 `README.md` 活跃报告列表
  - 目标归档版本 `docs/archive/vX.Y.Z/README.md`
  - 若导航策略发生变化，更新 `docs/source/index.rst` 或对应子索引说明
- 提交前必须通过：
  - `python3 scripts/doc_check.py`
  - `python3 scripts/check_docs_freshness.py --days 90`
  - `make -C docs html SPHINXOPTS="-W --keep-going"`
