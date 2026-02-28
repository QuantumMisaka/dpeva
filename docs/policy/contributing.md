# 文档贡献指南（Contributing）

- Status: active
- Audience: Developers / Maintainers
- Last-Updated: 2026-02-19

## 1. 目的与范围

本页定义 DP-EVA 文档系统的贡献流程与基本规范，确保文档长期可维护、可导航、与代码一致。

范围：

- `docs/guides/*`、`docs/reference/*`、`docs/architecture/*`、`docs/policy/*`、`docs/reports/*`

## 2. 相关方

- 开发者：更新使用说明、配置示例、行为变更说明
- 维护者：把控质量标准、避免重复与冲突、治理归档

## 3. 基本原则

- 单一权威来源：字段字典与校验规则只在 `docs/reference/*` 维护，其他文档只引用不复制。
- 相对路径链接：项目内资源链接统一使用相对路径，避免与环境绑定。
- 与代码一致：用户接口（CLI/配置/输出目录/完成标记）变更必须在同一 PR 内同步更新文档。

## 4. 文档类型与落点
- 分类与落点以 [DOCS_CLASSIFICATION.md](/docs/DOCS_CLASSIFICATION.md) 为准。

常见落点：

- 使用说明：`docs/guides/`
- 权威查表：`docs/reference/`
- 架构与决策：`docs/architecture/` 与 `docs/architecture/decisions/`
- 一次性结论：`docs/reports/`
- 制度与规范：`docs/policy/`
- 历史归档：`docs/archive/`

## 5. 必备元信息

所有标记为 `Status: active` 的文档必须包含：

- Status / Audience / Last-Updated

建议补充：

- Applies-To / Owners / Related

## 6. Review 与验收

验收标准以 [quality.md](/docs/policy/quality.md) 为准。

建议的 PR 自检：

- 链接检查：确保无断链、无绝对路径链接
- 示例一致性：示例配置字段与 Pydantic 模型一致
- 风险说明：对破坏性变更提供迁移说明与旧路径跳转入口（如确需重命名/重排）

## 7. 模板

- 页面模板：/docs/_templates/page.md
- ADR 模板：/docs/_templates/adr.md
- 报告模板：/docs/_templates/report.md
