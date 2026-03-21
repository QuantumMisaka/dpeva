---
title: 文档贡献指南 (Contributing)
status: active
audience: Developers / Maintainers
last-updated: 2026-03-10
owner: Docs Owner
---

# 文档贡献指南（Contributing）

## 1. 目的与范围

本页定义 DP-EVA 文档系统的贡献流程与基本规范，确保文档长期可维护、可导航、与代码一致。

范围：

- `docs/guides/*`、`docs/reference/*`、`docs/architecture/*`、`docs/policy/*`、`docs/plans/*`、`docs/reports/*`

## 2. 相关方

- 开发者：更新使用说明、配置示例、行为变更说明
- 维护者：把控质量标准、避免重复与冲突、治理归档

## 3. 基本原则

- 单一权威来源：字段字典与校验规则只在 `docs/reference/*` 维护，其他文档只引用不复制。
- 相对路径链接：项目内资源链接统一使用相对路径，避免与环境绑定。
- 与代码一致：用户接口（CLI/配置/输出目录/完成标记）变更必须在同一 PR 内同步更新文档。
- 禁止文件系统绝对路径链接：禁止 `/home/...`、`C:\...`；引用仓库代码文件请使用仓库 URL。

## 4. 文档类型与落点
- 分类与落点以 [docs-structure.md](https://github.com/QuantumMisaka/dpeva/blob/main/docs/policy/docs-structure.md) 为准。

常见落点：

- 使用说明：`docs/guides/`
- 权威查表：`docs/reference/`
- 架构与决策：`docs/architecture/` 与 `docs/architecture/decisions/`
- 一次性结论：`docs/reports/`
- 项目级计划：`docs/plans/`
- 制度与规范：`docs/policy/`
- 历史归档：`docs/archive/`
- 执行期草案：`.trae/documents/`（不直接作为项目文档入口）

### 4.1 草案与项目文档迁移规则

- AI/开发执行期文档先放 `.trae/documents/`。
- 当内容进入“团队共享、可复盘、可发布”的状态时，迁移到 `docs/plans/` 或 `docs/reports/`。
- 发布后统一归档至 `docs/archive/vX.Y.Z/{plans,reports}/`，并更新对应索引。

## 5. 必备元信息

所有标记为 `Status: active` 的文档必须包含：

- Status / Audience / Last-Updated / Owner(s)

建议补充：

- Applies-To / Related

## 6. Review 与验收

验收标准以 [quality.md](https://github.com/QuantumMisaka/dpeva/blob/main/docs/policy/quality.md) 为准。

建议的 PR 自检：

- 链接检查：确保无断链、无绝对路径链接
- 示例一致性：示例配置字段与 Pydantic 模型一致
- 风险说明：对破坏性变更提供迁移说明与旧路径跳转入口（如确需重命名/重排）
- 责任归属：确认本次变更文档具备 `owner` 或 `owners`

## 7. 稳态化提交流程（标准）

1. 在提交前执行：
   - `python3 scripts/doc_check.py`
   - `python3 scripts/check_docs_freshness.py --days 90`
   - `make html SPHINXOPTS="-W --keep-going"`
2. 在 PR 描述中回答：
   - 是否涉及 CLI/配置/输出契约变更
   - 若涉及，已同步更新哪些文档与示例
3. 由对应 Owner 或 Code Owner 完成审查后合并。

## 8. 模板

- 页面模板：/docs/_templates/page.md
- ADR 模板：/docs/_templates/adr.md
- 报告模板：/docs/_templates/report.md
