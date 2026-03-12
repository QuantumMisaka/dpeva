---
title: 过程资产生命周期（AI 草案到项目文档）
status: active
audience: Developers / Maintainers
last-updated: 2026-03-12
owner: Docs Owner
---

# 过程资产生命周期（AI 草案到项目文档）

本文定义 AI 开发场景下，计划与报告类文档从“执行草案”到“项目资产”的标准迁移路径，确保文档系统高可读、低噪音、可追溯。

## 1. 目标与原则

- 降噪：将高频中间记录与稳定结论文档分层管理。
- 可追溯：每份进入 `docs/` 的文档都可被团队复核并长期引用。
- 可归档：发布后形成版本化历史，不覆盖旧结论。

## 2. 四层落点模型

| 阶段 | 目录 | 内容类型 | 是否默认进入 Sphinx 主导航 |
|---|---|---|---|
| 执行草案 | `.trae/documents/` | 临时计划、调试记录、阶段草稿 | 否 |
| 收敛计划 | `docs/plans/` | 团队共享计划、验收标准、迁移策略 | 是（精选入口） |
| 收敛结论 | `docs/reports/` | 迭代总结、评审结论、发布总结 | 是（精选入口） |
| 历史归档 | `docs/archive/vX.Y.Z/{plans,reports}/` | 已发布的冻结历史 | 否（默认） |

## 3. 准入标准（迁入 docs 的前提）

从 `.trae/documents/` 迁入 `docs/plans/` 或 `docs/reports/` 前，必须满足：

1. 可理解：非当前执行者可独立阅读并复述核心结论。
2. 可复核：包含测试、构建、评审或实验依据。
3. 可复用：对后续开发、治理或发布仍有参考价值。

## 4. 标准迁移流程

1. **草案阶段**：在 `.trae/documents/` 高频迭代。
2. **收敛阶段**：筛选稳定内容迁入 `docs/plans/` 或 `docs/reports/`。
3. **索引阶段**：更新对应 README 与 `toctree`，避免孤立文档。
4. **归档阶段**：版本发布或任务闭环后迁入 `docs/archive/vX.Y.Z/`。

## 5. Sphinx 展示策略

- 主导航仅展示“稳定入口与收敛文档”，不直接展示执行草案。
- `docs/assets/` 为静态资源目录，不作为独立页面。
- `docs/archive/` 默认不进主导航，按需通过归档索引访问。

## 6. 提交前检查

```bash
python3 scripts/doc_check.py
python3 scripts/check_docs_freshness.py --days 90
make -C docs html SPHINXOPTS="-W --keep-going"
```
