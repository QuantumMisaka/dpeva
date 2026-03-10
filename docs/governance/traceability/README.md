---
title: Traceability（追踪矩阵）
status: active
audience: Maintainers / Developers
last-updated: 2026-03-10
owner: Docs Owner
---

# Traceability（追踪矩阵）

本目录用于维护“功能-文档-测试”双向追踪关系，降低接口变更遗漏文档更新的风险。

## 1. 核心矩阵

- 功能-文档矩阵：`feature-doc-matrix.md`
- 工作流契约-测试矩阵：`workflow-contract-test-matrix.md`

## 2. 使用原则

- 发布前检查新增/变更功能是否有对应文档与测试条目
- PR 评审时将矩阵作为变更影响范围核对清单
- 发现缺口时优先补齐矩阵，再补齐文档与测试
