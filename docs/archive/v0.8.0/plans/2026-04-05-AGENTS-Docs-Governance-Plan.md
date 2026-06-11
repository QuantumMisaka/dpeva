---
title: AGENTS and Docs Governance Plan
status: active
audience: developers
last-updated: 2026-04-05
owner: Docs Owner
---

# AGENTS 与文档系统联动改造方案

## 1. 目标

本方案的目标不是把 `AGENTS.md` 继续写长，而是把它改造成一份**高密度、低重复、可跳转**的目录型文档：

- 用最短路径说明 DP-EVA 的项目目标与核心工程契约；
- 把详细规则导向 `docs/` 中已存在的权威文档；
- 避免与 `.trae/rules/project_rules.md`、`developer-guide.md`、`policy/*`、`governance/*` 形成重复正文。

## 2. 改造前提

`AGENTS.md` 改造必须建立在本轮文档治理之后：

- active 文档中明显错误的 recipe 名称、错链、过时示例已先修复；
- `docs/plans/`、`docs/reports/`、`.trae/documents/` 的职责边界已重新对齐；
- 文档入口页已能较稳定地承担“权威导航”职责；
- 已确认 `docs/source/reference` 是指向 `../reference` 的符号链接，因此 Reference 正文本体天然单源，无需额外拆分包装层；
- `docs/build/` 已确认只是本地构建产物，不是版本库残留。

否则，`AGENTS.md` 只会把旧错误重新索引一遍。

## 3. 设计原则

### 3.1 只保留“必须先知道”的内容

`AGENTS.md` 只放：

- 项目使命与边界；
- 核心工程约束；
- 变更时必须同步关注的文档入口；
- 计划/报告/归档的最小协议。

不放：

- 大段开发哲学正文；
- 重复的安装、测试、构建细节；
- 大段配置字段字典；
- 已在 `docs/` 中稳定存在的完整流程说明。

### 3.2 目录化而非手册化

推荐写法是“摘要 + 链接”：

- 用一句话说明主题；
- 紧接着给出权威文档入口；
- 将细节下沉到 `docs/`。

### 3.3 只表达项目约束，不重复 AI 行为规则

`AGENTS.md` 不应重复 `.trae/rules/project_rules.md` 中的内容，例如：

- Zen of Python 角色设定；
- AI 注释风格；
- AI 代码审查口径。

这些属于工作区规则，不属于项目文档系统正文。

## 4. AGENTS 建议结构

推荐将 `AGENTS.md` 收敛为以下 6 个模块。

### 4.1 Project Purpose

只保留一段摘要，说明：

- DP-EVA 是面向 DPA 大模型高效微调的主动学习框架；
- 主链路由训练、推理、特征、采集、标注、分析、清洗等独立工作流构成；
- 项目目标是以更低标注成本提升模型性能。

详细背景统一导向：

- `docs/README.md`
- `docs/guides/developer-guide.md`
- `docs/architecture/design-report.md`

### 4.2 Read First

为首次进入仓库的人或 AI 提供最小阅读路径：

- 文档总入口：`docs/README.md`
- 开发主入口：`docs/guides/developer-guide.md`
- CLI 契约：`docs/guides/cli.md`
- 配置编写：`docs/guides/configuration.md`
- 文档治理快速上手：`docs/guides/docs-governance-quickstart.md`

约束：

- 若提到配置字段权威入口，面向读者统一写为 “API Reference”；
- 不把 `docs/source/api/config.rst` 作为 AGENTS 首层读者入口直接暴露。

### 4.3 Core Engineering Contracts

这里是 `AGENTS.md` 的真正核心，只保留项目级摘要，不展开细节。

建议保留以下 5 条：

1. **配置中心化**  
   配置、默认值和校验以 `src/dpeva/config.py` 及 Reference 文档为准。

2. **路径显式解析**  
   配置相对路径以配置文件所在目录为基准，避免依赖当前 shell 工作目录。

3. **工作流独立可执行**  
   各工作流可通过 CLI 单独启动，但共享底层模块，不应在 workflow 层复制逻辑。

4. **执行入口与导出入口分离**  
   以 `cli.py -> workflow` 为主链路，`__init__.py` 只承担导出面。

5. **对外契约变更必须联动文档**  
   任何 CLI、配置、输出目录、日志锚点、recipe 变化，必须同 PR 更新文档。

每条后面仅保留 1 个文档跳转，例如：

- `docs/guides/developer-guide.md`
- `docs/guides/cli.md`
- `docs/guides/configuration.md`
- `docs/reference/validation.md`

建议精确落点：

- 配置中心化 -> `docs/guides/developer-guide.md`
- 路径显式解析 -> `docs/guides/configuration.md`
- 工作流独立可执行 -> `docs/guides/cli.md`
- 执行入口与导出入口分离 -> `docs/guides/developer-guide.md`
- 对外契约变更必须联动文档 -> `docs/guides/docs-governance-quickstart.md`

### 4.4 Documentation Lifecycle

这里保留最小协议，不再重复治理正文：

- 审阅结论进入 `docs/reports/`
- 收敛计划进入 `docs/plans/`
- 完成后归档到 `docs/archive/vX.Y.Z/{plans,reports}/`
- 涉及 `.md` 增删改时检查 `docs/source/` 引用

详细规则导向：

- `docs/guides/process-asset-lifecycle.md`
- `docs/policy/contributing.md`
- `docs/policy/maintenance.md`

### 4.5 Quality Gates

只保留“有哪几类门禁”，不重复命令细节：

- 代码质量
- 单元测试
- 文档治理检查
- Sphinx 构建

详细执行步骤导向：

- `docs/guides/developer-guide.md`
- `docs/policy/quality.md`

### 4.6 Do Not Duplicate

在 AGENTS 末尾明确三条“不重复”：

- 不重复 `.trae/rules/project_rules.md` 的 AI 行为规则
- 不重复 `docs/reference/*` 的字段字典
- 不重复 `docs/guides/*` / `docs/policy/*` / `docs/governance/*` 的完整流程正文

## 5. 当前 AGENTS 内容处理建议

### 5.1 建议保留

- 项目一句话定位
- 对 `docs/reports/`、`docs/plans/`、`docs/archive/` 的最小生命周期说明
- “改文档要检查 `docs/source/` 引用”的提醒

### 5.2 建议压缩

- 安装、测试、lint、docs build 命令  
  建议压缩为“快速验证入口 + 文档链接”，不要在 AGENTS 内占据大量篇幅。

- “Workflow Scenarios”  
  改为直接引导到 `docs/guides/cli.md` 和 `examples/recipes/README.md`。

- “Critical Notes”  
  改成更强的项目契约表述，例如“配置以 `config.py` + Reference 为准”。

### 5.3 建议删除

- 与 AI 角色和行为规范重复的内容
- 与 `developer-guide.md` 已重复的长段流程说明
- 容易因版本演进而频繁过期的细节例子

### 5.4 建议新增

- 项目核心工程契约摘要
- “先读哪里”的文档导航块
- “什么变更必须同步文档”的联动提醒

## 6. 与文档系统的联动方案

为保证 `AGENTS.md` 足够简洁，建议形成以下分工：

| 主题 | AGENTS.md | 权威文档 |
|---|---|---|
| 项目目标 | 一段摘要 | `docs/README.md`、`docs/guides/developer-guide.md` |
| CLI 与工作流入口 | 一句提示 + 跳转 | `docs/guides/cli.md` |
| 配置与默认值 | 一句契约 + 跳转 | `docs/guides/configuration.md`、`docs/reference/*` |
| 架构与模块边界 | 一句摘要 + 跳转 | `docs/guides/developer-guide.md`、`docs/architecture/*` |
| 文档治理流程 | 最小协议 | `docs/guides/process-asset-lifecycle.md`、`docs/policy/*` |
| 质量门禁 | 仅列门类 | `docs/guides/developer-guide.md`、`docs/policy/quality.md` |

补充约束：

- `AGENTS.md` 只链接稳定读者入口，不直接链接 `docs/source/**/*.rst`；
- 若需要提示 Sphinx 构建层文件位置，应放在维护文档而不是 `AGENTS.md` 首层结构中。

## 7. 建议的精简骨架

下面是建议中的 `AGENTS.md` 骨架，不是最终正文，只是结构草案。

```md
# DP-EVA

## Purpose
- 一句话说明项目目标

## Read First
- docs/README.md
- docs/guides/developer-guide.md
- docs/guides/cli.md
- docs/guides/configuration.md
- docs/guides/docs-governance-quickstart.md

## Core Engineering Contracts
- 配置中心化
- 路径显式解析
- 工作流独立可执行、共享底层模块
- CLI 主链路优先，__init__ 仅导出
- 对外契约变更必须同步文档与 recipes

## Documentation Lifecycle
- reports / plans / archive 的最小协议
- 改动 .md 时检查 docs/source/

## Quality Gates
- 代码质量、单测、文档治理、Sphinx

## Do Not Duplicate
- 不重复 AI 规则
- 不重复 reference 字段字典
- 不重复 guides/policy/governance 正文
```

## 8. 实施顺序

建议按以下顺序落地：

1. 先合并本轮文档治理修复；
2. 再根据本方案重写 `AGENTS.md`；
3. 重写后复查 `docs/README.md`、`developer-guide.md`、`docs-governance-quickstart.md` 是否仍有重复；
4. 最后运行文档治理检查与 Sphinx 构建。

实施前确认：

- 当前文档治理前提已满足，可直接进入 `AGENTS.md` 改写；
- 若改写后发现入口文档仍有明显重复，应优先瘦身入口文档，而不是回退 AGENTS 的目录化设计。

## 9. 验收标准

`AGENTS.md` 改造完成后，应满足：

- 首屏即可看出项目目标和核心约束；
- 文档长度显著短于当前版本；
- 不再重复 AI 开发规则；
- 不再复制字段字典与完整开发流程；
- 能把读者稳定引导到 `docs/` 中正确的权威文档。
