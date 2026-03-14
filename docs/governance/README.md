---
title: Governance（文档治理总览）
status: active
audience: Maintainers / Developers
last-updated: 2026-03-10
owner: Docs Owner
---

# Governance（文档治理）

本目录用于固化 DP-EVA 文档治理方法，保障人类开发者与 AI 开发者在同一规则下协作。

## 1. 子目录速览

| 目录 | 用途 |
|---|---|
| `/docs/governance/audits/` | 治理审计输入与检查记录。 |
| `/docs/governance/reviews/` | 合规性评审报告与复核结论。 |
| `/docs/governance/traceability/` | 功能-文档-测试追踪矩阵。 |
| `/docs/governance/inventory/` | 文档库存与 Owner 责任矩阵。 |
| `/docs/governance/tools/` | 脚本与自动化治理能力说明。 |

## 2. 快速开始

- 文档总入口：`../README.md`
- 开发者治理上手：`../guides/docs-governance-quickstart.md`
- 最新执行闭环方案：`../archive/v0.6.6/plans/2026-03-14-collection-uq-fdiff-scatter-plan.md`
- Owner 责任矩阵：`inventory/owners-matrix.md`

## 3. 稳态治理基线

- 提交前必须可执行：
  - `python3 scripts/doc_check.py`
  - `python3 scripts/check_docs_freshness.py --days 90`
  - `make -C docs html SPHINXOPTS="-W --keep-going"`
- PR 必须使用 `.github/PULL_REQUEST_TEMPLATE.md`
- 治理关键路径评审由 `.github/CODEOWNERS` 执行

## 4. 相关制度

- 贡献规范：`../policy/contributing.md`
- 维护机制：`../policy/maintenance.md`
- 质量标准：`../policy/quality.md`
