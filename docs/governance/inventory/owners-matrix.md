---
title: 文档 Owner 覆盖矩阵
status: active
audience: Maintainers
last-updated: 2026-03-10
owner: Docs Owner
---

# 文档 Owner 覆盖矩阵

本矩阵用于稳态化阶段的责任归属治理，作为 `status: active` 文档 Owner 覆盖率核对基线。

## 1. 模块 Owner 映射

| 文档域 | 负责范围 | Owner 角色 | 审查角色 |
|---|---|---|---|
| `docs/guides/*` | 使用与操作指南 | Workflow Owner | Tech Lead |
| `docs/reference/*` | 配置字段与规则权威来源 | Config Model Owner | Architect |
| `docs/architecture/*` | 架构说明与决策记录 | Architect | Project Lead |
| `docs/policy/*` | 治理制度与标准 | Docs Owner | Project Lead |
| `docs/governance/*` | 审计、追踪、库存 | Docs Owner | Maintainers |
| `docs/plans/*` | 执行方案与里程碑 | Feature Owner | Tech Lead |

## 2. 稳态化验收要求

- 所有 `status: active` 文档必须包含 `owner` 或 `owners`
- PR 合并前必须完成对应 Owner 或 Code Owner 审查
- 目录级 Owner 映射与文档 front matter 声明不一致时，按本矩阵与 CODEOWNERS 修正

## 3. 维护策略

- 每周增量更新：新增 active 文档时同步补 Owner 字段
- 每月全量复核：对照 `scripts/doc_check.py` 与 CODEOWNERS 校验覆盖率
