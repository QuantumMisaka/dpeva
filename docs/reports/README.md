---
title: Technical Reports
status: active
audience: Developers
last-updated: 2026-06-10
owner: Docs Owner
---

# 技术报告（Technical Reports）

- Status: active
- Audience: All
- Last-Updated: 2026-06-10

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

- [2026-04-01-Code-Review-Repository-Audit.md](2026-04-01-Code-Review-Repository-Audit.md)
  - Type: Summary Report
  - Status: active
  - Related Focused Report: [2026-04-02-Code-Review-Unit-Test-Audit.md](2026-04-02-Code-Review-Unit-Test-Audit.md)
  - Archive Policy: 闭环后随版本发布归档到 `docs/archive/vX.Y.Z/reports/`

## 5. Active Focused Reports

- [2026-04-02-Code-Review-Unit-Test-Audit.md](2026-04-02-Code-Review-Unit-Test-Audit.md)
  - Type: Focused Report
  - Status: active
  - Parent Report: [2026-04-01-Code-Review-Repository-Audit.md](2026-04-01-Code-Review-Repository-Audit.md)
  - Archive Policy: 随父报告闭环或失效时同步归档

## 6. Active Experiment Series

### Sampling Comparison Series

- [2026-06-08-DPA4-Neo-Air-Plus-Sampling-Comparison.md](2026-06-08-DPA4-Neo-Air-Plus-Sampling-Comparison.md)
  - Type: Experiment Series Report
  - Status: active
  - Series: Sampling Comparison
  - Archive Policy: 结论被后续系列报告取代或发布闭环后归档
- [2026-06-07-DPA4-Mini-MACE-Sampling-Comparison.md](2026-06-07-DPA4-Mini-MACE-Sampling-Comparison.md)
  - Type: Experiment Series Report
  - Status: active
  - Series: Sampling Comparison
  - Archive Policy: 结论被后续系列报告取代或发布闭环后归档

## 7. Recently Archived

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
