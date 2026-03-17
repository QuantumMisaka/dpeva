---
title: Document
status: active
audience: Developers
last-updated: 2026-03-17
owner: Docs Owner
---

# 技术报告 (Technical Reports)

- Status: active
- Audience: All
- Last-Updated: 2026-03-17

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

- 暂无活跃报告，最新报告已归档至 `docs/archive/v0.6.8/reports/`。

## 5. 归档与索引要求

- 报告归档时，必须同步更新：
  - 当前目录 `toctree`
  - 目标归档版本 `docs/archive/vX.Y.Z/README.md`
  - 若涉及导航入口，更新 `docs/source/index.rst` 或对应子索引
- 提交前必须通过：
  - `python3 scripts/doc_check.py`
  - `python3 scripts/check_docs_freshness.py --days 90`
  - `make -C docs html SPHINXOPTS="-W --keep-going"`
