---
title: 文档版本管理与维护机制 (Maintenance)
status: active
audience: Maintainers
last-updated: 2026-03-10
---

# 文档版本管理与维护机制（Maintenance）

## 1. 文档版本策略

- 面向用户的 Guide/Reference 默认随 `main` 分支演进。
- 重大接口变更（CLI/配置字段/目录结构）必须在同一 PR 内：
  - 更新相关文档
  - 更新示例（`examples/recipes`）
  - 更新/新增回归测试（unit 或 integration）
- 报告与归档文档默认冻结：新增通过“新文件”形式，不在旧报告中“覆盖式修改结论”。

## 2. Ownership（责任到人/模块）

建议按模块分配文档 Owner：

- CLI/配置总览：接口维护人（CLI Owner）
- Train/Infer/Feature/Collect：各 Workflow 维护人
- Slurm/集群适配：平台/运维协作负责人
- API Reference：配置模型维护人（Pydantic Model Owner）

Owner 可以是角色而非具体姓名；但每篇 `active` 文档必须有 Owner。

### 2.1 RACI 职责矩阵

| 文档类型 (Type) | 谁负责写 (Responsible) | 谁负责审 (Accountable) | 谁提供咨询 (Consulted) | 谁接收通知 (Informed) |
| :--- | :--- | :--- | :--- | :--- |
| **Guides (User/Dev)** | Feature Developer | Tech Lead | QA / Users | All Developers |
| **Reference (API/Config)** | Code Owner | Architect | - | All Developers |
| **Architecture (Design)** | Architect | Project Lead | Tech Lead | All Developers |
| **Policy (Governance)** | Docs Owner | Project Lead | Team Members | All Developers |
| **Reports (Audit/Exp)** | Auditor / Researcher | Tech Lead | - | Stakeholders |


## 3. 变更触发规则（何时必须更新文档）

- 新增/修改 Pydantic 配置字段：必须更新 `docs/reference/*`
- 变更输出目录命名、日志文件名、完成标记：必须更新 Slurm 与集成测试相关文档
- 新增 CLI 子命令或更改参数：必须更新 CLI Guide 与 Quickstart

## 4. Review 流程（建议）

- 所有文档变更走 PR Review
- 文档 PR 至少 1 名 Owner Review
- 若变更涉及用户接口或配置字段，建议在 PR 描述中附“迁移说明”与“最小示例”

## 5. 文档过期治理（建议）

- 每季度（或每 2 个 release）做一次 docs 体检：
  - 断链扫描（链接可达性）
  - 过期字段扫描（例如 `num_selection` 这类已废弃字段）
  - 示例可运行性抽检（优先 Quickstart 与集成测试）

## 6. 稳态化运行基线（必须满足）

- 基线门禁：
  - `python3 scripts/doc_check.py` 必须通过
  - `python3 scripts/check_docs_freshness.py --days 90` 必须通过
  - `make html SPHINXOPTS="-W --keep-going"` 必须通过
- 责任归属：
  - 所有 `active` 文档必须声明 `owner` 或 `owners`
  - Owner 角色映射与覆盖追踪统一维护在 `docs/governance/inventory/owners-matrix.md`
- 变更流程：
  - 涉及 CLI、配置字段、输出契约、示例目录结构的变更，必须在同一 PR 完成文档更新
  - PR 必须使用文档检查清单并由对应 Code Owner 审查
- 复核节奏：
  - 每周一次增量复核（断链、元信息、新鲜度）
  - 每月一次全量复核（结构、导航、重复内容、Owner 覆盖率）

## 7. 过程资产管理规范 (Process Asset Management)

为确保项目知识沉淀有序，所有过程性文档必须遵循以下生命周期管理规范：

### 7.1 代码审查报告 (Code Review Reports)
- **落点**: `docs/reports/`
- **命名**: `YYYY-MM-DD-Code-Review-<Feature/Module>.md`
- **内容**: 审查对象、发现的问题、修复建议、修复状态。
- **归档**: 审查结束并修复闭环后，随版本发布归档至 `docs/archive/vX.Y.Z/reports/`。

### 7.2 功能开发细则 (Feature Plans/Specs)
- **落点**: `docs/plans/`
- **命名**: `YYYY-MM-DD-<Feature>-Plan.md` 或 `YYYY-MM-DD-<Feature>-Spec.md`
- **内容**: 目标、范围、设计方案、测试计划、任务分解。
- **归档**: 功能开发完成并通过验收后，归档至 `docs/archive/vX.Y.Z/plans/`。

### 7.3 文档归档 (Archiving)
- **触发时机**:
  - 版本发布 (Release) 前。
  - 重大功能开发闭环 (Feature Completed) 后。
- **操作步骤**:
  1. 确认文档状态已更新为 `completed` 或 `archived`。
  2. 移动文件至对应版本的 `docs/archive/vX.Y.Z/{plans,reports}/` 目录。
  3. 更新 `docs/archive/vX.Y.Z/README.md` 索引。

### 7.4 索引文件维护 (Index Maintenance)
- **规则**: 文档的物理移动（重命名/归档）必须伴随逻辑索引（Sphinx `.rst`）的更新。
- **Sphinx 映射**:
  - 核心索引文件位于 `docs/source/` 目录。
  - `docs/source/index.rst` 映射整个文档站点的导航结构。
  - 子目录索引（如 `docs/source/reference/index.rst`）负责该板块的文件列表。
- **强制检查**:
  - 任何 Markdown 文件的增删改，必须检查 `docs/source/**/*.rst` 是否有对应的 `toctree` 引用需要更新。
  - 运行 `make html` 确保无 `WARNING: toctree contains reference to nonexisting document` 报错。


