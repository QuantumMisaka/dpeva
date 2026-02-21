# DP-EVA 文档目录清单（结构化摘要与标签）

本清单用于逐份记录 `docs/` 下文档的主题、功能、受众与关键信息点，并作为后续分类归档与补全的依据。

字段说明：

- Category：Guide / Reference / Architecture / Decision(ADR) / Report / Archive / Policy / Template / Asset
- Audience：Users / Developers / Researchers / Maintainers
- Cadence：High(高频) / Medium(中频) / Low(低频)
- Status：draft / active / archived

## 1. Docs（入口与规范）

| Document | Category | Audience | Cadence | Status | Tags | 摘要（关键信息点） |
|---|---|---|---|---|---|---|
| [docs/README.md](/docs/README.md) | Policy | All | High | active | index\|navigation\|single-source-of-truth | 统一入口与读者分流；明确 docs 分层与“权威来源”；定义文档维护规则。 |
| [policy/maintenance.md](/docs/policy/maintenance.md) | Policy | Maintainers | High | active | ownership\|review\|governance | 文档版本策略、Owner 分配、变更触发规则、Review 机制、过期治理建议。 |
| [policy/quality.md](/docs/policy/quality.md) | Policy | Maintainers | Medium | active | quality\|checklist\|acceptance | 文档质量维度评分、统一元信息字段、不同类型文档验收清单。 |
| [policy/contributing.md](/docs/policy/contributing.md) | Policy | Developers/Maintainers | Medium | active | contributing\|review | 文档贡献流程与规范：相对链接、权威来源、元信息与验收标准。 |
| [docs/_templates/page.md](/docs/_templates/page.md) | Template | Writers | Medium | active | template\|page | 通用页面结构模板（背景/范围/核心内容/示例/FAQ/链接）。 |
| [docs/_templates/adr.md](/docs/_templates/adr.md) | Template | Maintainers | Medium | active | template\|adr\|decision | 技术决策记录模板（Context/Decision/Consequences/Alternatives）。 |
| [docs/_templates/report.md](/docs/_templates/report.md) | Template | Researchers | Low | active | template\|report | 报告模板（结论摘要/方法与数据/结果分析/建议）。 |

## 2. API / Reference（配置与校验）

| Document | Category | Audience | Cadence | Status | Tags | 摘要（关键信息点） |
|---|---|---|---|---|---|---|
| [config-schema.md](/docs/reference/config-schema.md) | Reference | Developers | Medium | active | config\|pydantic\|schema | Workflow 配置字段字典（通用字段、Train/Infer/Feature/Collect/Analysis），包括类型/默认值/说明/约束。 |
| [validation.md](/docs/reference/validation.md) | Reference | Developers | Medium | active | validation\|constraints | Pydantic 参数校验逻辑补充：范围约束、跨字段依赖、env_setup 格式化、路径存在性校验。 |
| [reference/README.md](/docs/reference/README.md) | Policy | Developers | Medium | active | reference\|single-source | Reference 分层说明与迁移建议入口。 |

## 3. Guides（用户/开发者操作指南）

| Document | Category | Audience | Cadence | Status | Tags | 摘要（关键信息点） |
|---|---|---|---|---|---|---|
| [guides/README.md](/docs/guides/README.md) | Policy | Users/Developers | Medium | active | guides\|index | Guides 目录定位与现有权威入口链接。 |
| [guides/installation.md](/docs/guides/installation.md) | Guide | Users/Developers | High | active | install\|env | 安装与环境准备：Python 依赖、可编辑安装、DeepMD 外部依赖与验证方法。 |
| [guides/quickstart.md](/docs/guides/quickstart.md) | Guide | Users | High | active | quickstart\|e2e | 最短路径跑通一次 Feature/Train/Infer/Collect/Analysis，并给出输出验证与排障入口。 |
| [guides/cli.md](/docs/guides/cli.md) | Guide | Users/Developers | High | active | cli\|subcommands | CLI 子命令职责、输入输出与退出码；完成标记与排障入口。 |
| [guides/configuration.md](/docs/guides/configuration.md) | Guide | Users/Developers | High | active | config\|paths | 路径解析规则、Submission 结构与各 Workflow 最小配置示例；字段查表链接到 Reference。 |
| [guides/slurm.md](/docs/guides/slurm.md) | Guide | Users/Infra | High | active | slurm\|monitoring | Slurm submission 配置、日志命名、完成标记监控与常见故障排查。 |
| [guides/troubleshooting.md](/docs/guides/troubleshooting.md) | Guide | Users/Developers | High | active | troubleshooting\|faq | 结构化排查顺序与环境/数据/作业/数值四类常见问题处理建议。 |

## 4. Governance（文档治理）

| Document | Category | Audience | Cadence | Status | Tags | 摘要（关键信息点） |
|---|---|---|---|---|---|---|
| [doc-system-planning.md](/docs/governance/plans/2026-02-18-doc-system-planning.md) | Plan | Maintainers | Medium | active | docs\|planning\|raci | 文档体系规划：分类体系、缺失补充、模板、维护与质量标准、实施排期与责任分配。 |
| [feature-doc-matrix.md](/docs/governance/traceability/feature-doc-matrix.md) | Policy | Maintainers | High | active | traceability\|release | 功能-文档双向追踪矩阵（发布前核对）。 |
| [doc-accuracy-audit.md](/docs/governance/audits/2026-02-19-doc-accuracy-audit.md) | Report | Maintainers | Medium | active | audit\|accuracy | 文档准确性审计：对照代码与对外接口，形成差异与风险项。 |
| [docs-review-report.md](/docs/governance/audits/2026-02-19-docs-review-report.md) | Report | Maintainers | Medium | active | audit\|quality | 文档质量审查结论与问题清单。 |
| [link-normalization.json](/docs/governance/tools/link-normalization.json) | Asset | Maintainers | Medium | active | tooling\|links | 链接规范化机器可读规则（用于工具/CI）。 |

## 5. Design（技术设计与科研分析）

| Document | Category | Audience | Cadence | Status | Tags | 摘要（关键信息点） |
|---|---|---|---|---|---|---|
| [DP-EVA_Design_Report.md](/docs/design/DP-EVA_Design_Report.md) | Architecture | Developers | Medium | active | patterns\|ddd\|refactor | 设计模式识别、耦合分析与解耦策略。 |
| [2026-02-04-deepmd-dependency.md](/docs/architecture/decisions/2026-02-04-deepmd-dependency.md) | Decision | Maintainers | Low | active | adr\|deepmd\|dependency | DeepMD-kit 依赖管理决策。 |
| [modulo-hypothesis.md](/docs/reports/modulo-hypothesis.md) | Report | Researchers | Low | active | descriptor\|modulo\|evidence | 结构描述符模长的物理意义假设与实证分析。 |

## 6. Archive（历史与弃用）

| Document | Category | Audience | Cadence | Status | Tags | 摘要（关键信息点） |
|---|---|---|---|---|---|---|
| [archive/README.md](/docs/archive/README.md) | Policy | All | Low | active | archive\|policy | 归档策略：适用版本、落地状态、修订说明约定。 |
| [Code_Review_Report_v2.7.1.md](/docs/archive/Code_Review_Report_v2.7.1.md) | Archive | Developers | Low | archived | code-review | 历史代码审查报告（性能/并发/测试等）。 |
| [Variable_Review_Report.md](/docs/archive/Variable_Review_Report.md) | Archive | Developers | Low | archived | config\|variables | 历史变量/配置体系审查。 |

