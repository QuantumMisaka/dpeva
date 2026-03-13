---
title: DP-EVA 文档导航 (Docs Index)
status: active
audience: Users / Developers
last-updated: 2026-03-12
owner: Quantum Misaka
---

# DP-EVA 文档导航（Docs Index）


本文档是 `docs/` 的统一入口：提供读者分流、权威信息源定位、以及文档维护规则。

## 1. 我应该读哪一类文档？

### 1.1 使用者（想把流程跑起来）

- 项目概览与快速开始：[README.md](README.md)
- 安装与环境准备：[installation.md](guides/installation.md)
- Quickstart（最短路径跑通）：[quickstart.md](guides/quickstart.md)
- CLI 使用指南：[cli.md](guides/cli.md)
- 核心上游软件与职责：[upstream-software.md](reference/upstream-software.md)

### 1.2 开发者（要改代码/加功能/修Bug）

- 开发流程标准与架构说明：[developer-guide.md](guides/developer-guide.md)
- 配置字段权威参考（查表）：[reference/index.rst](reference/index.rst)
- 参数验证与约束补充：[validation.md](reference/validation.md)
- 核心上游软件与职责：[upstream-software.md](reference/upstream-software.md)
- Slurm 使用与排障：[slurm.md](guides/slurm.md)
- 文档贡献指南：[contributing.md](policy/contributing.md)

### 1.3 研究者（关注算法假设/设计权衡/实验结论）

- 系统设计与模式分析：[design-report.md](architecture/design-report.md)
- DeepMD 依赖决策记录：[2026-02-04-deepmd-dependency.md](architecture/decisions/2026-02-04-deepmd-dependency.md)
- 描述符模长假设与实验报告：[modulo-hypothesis.md](archive/v0.5.2/reports/modulo-hypothesis.md)

## 2. 文档分层与权威来源（避免重复/冲突）

- `docs/reference/`：权威“查表类”参考（字段列表、校验规则）。只在这里维护全量字段说明。
- `docs/guides/`：主线操作指南（Quickstart/CLI/配置/Slurm/测试专题等）。遇到字段解释只链接到 `docs/reference/`。
- `docs/architecture/`：架构与关键技术决策（ADR）以及设计报告。
- `docs/governance/`：文档治理交付物（规划、审计、追踪矩阵、工具配置）。
- `docs/plans/`：项目级“已收敛计划”（可复盘、可归档、面向团队共享）。
- `docs/reports/`：项目级“已收敛结论”（里程碑总结、评审结论、发布总结）。
- `docs/logo_design/`：项目 logo 设计与相关文档。
- `docs/archive/`：历史与弃用文档（只读）。必须在文件头显式标注适用版本与是否已落地。

### 2.1 AI 执行文档与项目文档边界

为减少噪音并提升可维护性，区分“过程草案”与“项目资产”：

| 位置 | 文档类型 | 可见性 | 生命周期 |
|---|---|---|---|
| `.trae/documents/` | AI 执行期草案、临时计划、阶段记录 | 工作空间内部 | 任务期内可变，不要求 Sphinx 暴露 |
| `docs/plans/` | 当前版本执行期草案、临时计划、阶段记录的集中存放点 | 项目文档系统 | 完成后归档至 `docs/archive/vX.Y.Z/plans/` |
| `docs/reports/` | 当前版本代码审查报告和其他类型报告存放位置（评审/实验/发布总结） | 项目文档系统 | 发布后归档至 `docs/archive/vX.Y.Z/reports/` |
| `docs/archive/` | 历史只读资产 | 默认不进主导航 | 长期保留，禁止覆盖式重写 |

### 2.2 Sphinx 展示策略

- `docs/source/index.rst` 只绑定“面向读者的稳定入口”，避免把过程性草稿直接暴露到主导航。
- `docs/plans/`、`docs/reports/` 仅收录“可复盘、可共享”的文档；草案先放 `.trae/documents/`。
- `docs/assets/` 作为静态资源目录，不作为独立页面展示。
- `docs/archive/` 默认不进入主导航，按需通过归档索引访问。

### 2.1 现行文档与归档文档判别规则

- 现行规范只在 `docs/guides/`、`docs/reference/`、`docs/policy/`、`docs/governance/` 维护。
- `docs/archive/` 仅保留历史上下文，不作为现行规范来源。
- 同一主题若 active 与 archive 并存，读取顺序固定为：active 文档优先，archive 仅作背景参考。
- 当前版本的执行型规格统一落在 `docs/plans/`，`docs/archive/specs/` 仅保留历史规格。

## 3. 文档维护规则（建议作为团队约定）

- 变更 `src/` 的用户接口、配置字段、关键目录结构时，PR 必须同步更新：
  - `docs/reference/*`（若涉及字段/校验）
  - `docs/guides/*`（若涉及使用方式/流程）
- 文档必须以“单一权威来源”为原则：同一份字段字典不允许在多处复制粘贴维护。
- `docs/archive/` 内容默认不回写，除非修正事实性错误（需要保留修订说明）。

### 3.1 过程资产迁移规则（AI 与人类开发统一）

1. 草案阶段：文档放置 `.trae/documents/`，允许高频迭代。
2. 收敛阶段：当计划/结论稳定且需团队共享时，迁移至 `docs/plans/` 或 `docs/reports/`。
3. 发布阶段：版本发布或任务闭环后，归档至 `docs/archive/vX.Y.Z/{plans,reports}/` 并更新索引。
4. 展示阶段：仅“收敛文档”进入 Sphinx 导航，避免将过程噪音暴露给最终读者。

## 4. 文档审查记录 (Audit Logs)

定期对文档系统的健康状况进行体检，包括结构完整性、治理合规性与内容有效性。

| 审查日期 | 版本 | 审查人 | 状态 | 快速链接 |
| :--- | :--- | :--- | :--- | :--- |
| 2026-03-09 | v1.0 | Trae AI | Completed | [DocReview_20260309_v1.0.md](governance/reviews/DocReview_20260309_v1.0.md) |
