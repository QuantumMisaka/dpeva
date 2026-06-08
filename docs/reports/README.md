---
title: Document
status: active
audience: Developers
last-updated: 2026-04-05
owner: Docs Owner
---

# 技术报告（Technical Reports）

- Status: active
- Audience: All
- Last-Updated: 2026-04-05

本文档目录用于存放“项目级、可共享、可追溯”的技术结论，不承载执行过程草稿。

## 1. 报告类型

- **Technical Analysis**: 算法性能分析、设计权衡、架构结论。
- **Experiment Reports**: 关键实验结果、参数评估与对比结论。
- **Release Reports**: 版本发布总结与变更影响说明。

## 2. 维护策略

- **Active**: 当前版本仍需被团队频繁引用的结论文档。
- **Archived**: 发布后或过时结论，归档至 `docs/archive/vX.Y.Z/reports/`。
- **Naming**: `YYYY-MM-DD-<topic>.md`
- **Ingestion Rule**: 过程记录先放 `.trae/documents/`；仅“收敛结论”才迁入本目录。

## 3. AI 执行文档边界

- `.trae/documents/`：调试记录、迭代草稿、阶段性分析片段。
- `docs/reports/`：可对外复核、可长期引用、可纳入版本发布说明的最终结论。
- 迁入 `docs/reports/` 的前提：
  1. 结论可复核（含依据与验证结果）；
  2. 结论不再依赖即时上下文；
  3. 对后续开发或治理具有复用价值。

## 4. 活跃报告 (Active Reports)

- [2026-06-08-DPA4-Neo-Air-Plus-Sampling-Comparison.md](2026-06-08-DPA4-Neo-Air-Plus-Sampling-Comparison.md)：归档 DPA4 Neo / Air / Plus 的 HTML 互比报告与配套 CSV，给出采样量、PCA、coverage、novelty、重叠度的可复核汇总。
- [2026-06-07-DPA4-Mini-MACE-Sampling-Comparison.md](2026-06-07-DPA4-Mini-MACE-Sampling-Comparison.md)：基于同一 DPA4 Mini UQ candidate 集合，对 DPA4 Mini、MACE small、MACE medium 的 DP-EVA feature + collect 结果做定量对比，并附带 HTML/CSV 结果归档。
- [2026-04-05-Code-Review-Agents-Docs-Governance.md](2026-04-05-Code-Review-Agents-Docs-Governance.md)：交叉审阅 `docs/`、`examples/recipes/` 与 `src/`，核查工程规范落实度、修复 active 文档偏差，并沉淀 `AGENTS.md` 目录化改造的证据基线。
- [2026-04-01-Code-Review-Repository-Audit.md](2026-04-01-Code-Review-Repository-Audit.md)：基于本轮真实仓库审查结果重写，覆盖八维度评估、严重级别、优先级、行动项、证据与验证记录。
- [2026-04-02-Code-Review-Unit-Test-Audit.md](2026-04-02-Code-Review-Unit-Test-Audit.md)：基于 `docs/`、`src/` 与 `tests/` 的单元测试深度审查结果，覆盖覆盖率、Mock、命名、性能、Lint、类型检查与回归步骤。

## 5. 归档与索引要求

- 报告归档时，必须同步更新：
  - 当前目录 `README.md` 活跃报告列表
  - 目标归档版本 `docs/archive/vX.Y.Z/README.md`
  - 若导航策略发生变化，更新 `docs/source/index.rst` 或对应子索引说明
- 提交前必须通过：
  - `python3 scripts/doc_check.py`
  - `python3 scripts/check_docs_freshness.py --days 90`
  - `make -C docs html SPHINXOPTS="-W --keep-going"`
