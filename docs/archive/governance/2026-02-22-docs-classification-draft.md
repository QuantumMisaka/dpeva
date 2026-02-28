# 文档分类维度与命名规范（Docs Taxonomy）

本文档定义 DP-EVA 文档的统一分类维度与命名规范，用于长期维护与扩展。

## 1. 分类维度

### 1.1 Intent（文档意图/功能）

- Guide：操作指南（怎么用/怎么做）
- Reference：权威参考（查表/规范/协议）
- Architecture：系统结构（与代码一致的架构概览）
- Decision（ADR）：关键技术决策（一次性结论，可追溯）
- Report：报告（实验/评审/分析，一次性结论）
- Policy：流程与制度（维护机制、质量标准）
- Template：模板（文档写作骨架）
- Archive：归档（历史/弃用，只读）
- Asset：资源（图片/图表）

### 1.2 Audience（目标受众）

- Users：使用者（想把流程跑起来）
- Developers：开发者（要改代码/排障/扩展）
- Researchers：研究者（算法/实验/假设）
- Maintainers：维护者（版本/发布/治理）

### 1.3 Cadence（更新频率）

- High：随接口/流程频繁更新
- Medium：随配置/结构变更更新
- Low：只追加或很少更新

### 1.4 Status（状态）

- draft：草稿/待补全
- active：现行有效
- deprecated：已弃用但保留参考
- archived：归档冻结（只读）

## 2. 目录归档规则（归档到哪里）

为保证“同类文档归入同一子目录”，推荐使用下列目录作为权威落点：

- `docs/guides/`：Guides（操作指南）
- `docs/reference/`：Reference（字段表/约束/术语）
- `docs/architecture/`：Architecture（现状一致的结构说明）
- `docs/architecture/decisions/`：Decision（ADR）
- `docs/reports/`：Report（分析/评审/实验）
- `docs/policy/`：Policy（制度/流程/质量标准）
- `docs/_templates/`：Template（写作模板）
- `docs/archive/`：Archive（历史/弃用）
- `docs/assets/`：Asset（图片/图表等）

迁移期允许保留旧路径的“redirect stub”（短文件跳转到新路径），但权威内容必须落在上述目录中。

## 3. 命名规范

- 文件名：`kebab-case.md`，避免长前缀与版本号堆叠。
- ADR：`YYYY-MM-DD-<topic>.md`（例如 `2026-02-04-deepmd-dependency.md`）。
- 图片：`docs/assets/img/<topic>.png`。
- 标题：中文为主（必要时附英文括注），避免同义重复。

## 4. 元信息规范（每篇 active 文档建议包含）

- Status / Applies-To / Owners / Last-Updated / Related Links

参考模板：

- `docs/_templates/page.md`
- `docs/_templates/adr.md`
- `docs/_templates/report.md`

