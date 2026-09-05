---
title: Governance（文档治理总览）
status: active
audience: Maintainers / Developers
last-updated: 2026-09-05
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
- 历史执行闭环示例：`../archive/v0.7.1/plans/README.md`
- Owner 责任矩阵：`inventory/owners-matrix.md`

## 3. 稳态治理基线

- 文档变更执行 `python scripts/run_gate.py docs_pr`；完整发布执行
  `python scripts/run_gate.py release`。
- 门禁命令的唯一来源是 [`scripts/gates.toml`](../../scripts/gates.toml)，由
  [`scripts/run_gate.py`](../../scripts/run_gate.py) 分发；本页不复制命令参数。
- 季度治理审计默认为 report-only：`python scripts/audit_governance_rules.py
  --format json`。仅在显式发布评审时使用 `--strict`。
- PR 必须使用 `.github/PULL_REQUEST_TEMPLATE.md`
- 治理关键路径评审由 `.github/CODEOWNERS` 执行

## 4. 相关制度

- 贡献规范：`../policy/contributing.md`
- 维护机制：`../policy/maintenance.md`
- 质量标准：`../policy/quality.md`
