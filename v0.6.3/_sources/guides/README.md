---
title: Guides（操作指南导航）
status: active
audience: Developers / Users
last-updated: 2026-03-10
owner: Docs Owner
---

# Guides（操作指南）

本目录提供“怎么做”的高频文档入口，并作为项目文档治理策略在开发侧的主入口。

## 1. 开发者快速上手

- 开发总入口：`developer-guide.md`
- 文档治理上手：`docs-governance-quickstart.md`
- 配置文档维护：`developer/config-docs-maintenance.md`

## 2. 使用与运维指南

- 环境安装：`installation.md`
- 快速跑通：`quickstart.md`
- 命令行指南：`cli.md`
- 配置编写：`configuration.md`
- Slurm 指南：`slurm.md`
- 排障指南：`troubleshooting.md`
- 测试指南：`testing/README.md`

## 3. 文档治理开发约定

- 接口/配置/输出契约变更必须同 PR 更新文档
- 字段定义以 `docs/reference/*` 为单一权威来源
- 提交前建议执行：
  - `python3 scripts/doc_check.py`
  - `python3 scripts/check_docs_freshness.py --days 90`
  - `make -C docs html SPHINXOPTS="-W --keep-going"`

治理制度详见：`../policy/{contributing,maintenance,quality}.md`  
治理操作详见：`../governance/README.md`
