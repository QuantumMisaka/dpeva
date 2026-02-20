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
| [reference/README.md](/docs/reference/README.md) | Policy | Developers | Medium | active | reference\|single-source | Reference 分层说明与迁移建议入口（未来承接 api/ 的权威查表）。 |

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

## 4. Main（主线交付与专题）

| Document | Category | Audience | Cadence | Status | Tags | 摘要（关键信息点） |
|---|---|---|---|---|---|---|
| [developer-guide.md](/docs/guides/developer-guide.md) | Guide | Developers | High | active | architecture\|workflow\|cli\|testing | 项目概览、目录结构、模块详解（Train/Infer/UQ/Sampling/Feature/Submission）、CLI 使用、测试与工作流监控标记规范。 |
| [integration-slurm-plan.md](/docs/guides/testing/integration-slurm-plan.md) | Guide | Developers | Medium | active | integration-test\|slurm | 集成测试里程碑与 DoD：基于生产目录反推 I/O，构建 Slurm 链式编排与最小化数据集。 |
| [integration-slurm.md](/docs/guides/testing/integration-slurm.md) | Guide | Developers | Medium | active | integration-test\|deliverable | 集成测试交付索引：指向 tests/integration 编排器、裁剪脚本、模板与完成标记机制。 |
| [integration-config-templates.md](/docs/guides/testing/integration-config-templates.md) | Guide | Developers | Medium | active | configs\|slurm | 集成测试最小配置模板说明：Feature/Train/Infer/Collect/Analysis 的输入输出约定与模板索引。 |
| [multi-datapool-artifacts.md](/docs/guides/testing/multi-datapool-artifacts.md) | Report | Developers | Low | active | i/o-analysis\|multi-pool | 生产目录 I/O 拆解与 Workflow 配置映射（多数据池候选/训练集/描述符/模型与采样结果）。 |
| [DP-EVA_Documentation_System_Planning_Report.md](/docs/main/DP-EVA_Documentation_System_Planning_Report.md) | Plan | Maintainers | Medium | active | docs\|planning\|raci | 文档体系规划：分类体系、缺失补充、模板、维护与质量标准、实施排期与责任分配。 |

## 5. Design（技术设计与科研分析）

| Document | Category | Audience | Cadence | Status | Tags | 摘要（关键信息点） |
|---|---|---|---|---|---|---|
| [DP-EVA_Design_Report.md](/docs/design/DP-EVA_Design_Report.md) | Architecture | Developers | Medium | active | patterns\|ddd\|refactor | 设计模式识别、耦合分析与解耦策略；包含“已实施重构”的历史说明，需要通过状态/适用版本进一步澄清。 |
| [2026-02-04-deepmd-dependency.md](/docs/architecture/decisions/2026-02-04-deepmd-dependency.md) | Decision | Maintainers | Low | active | adr\|deepmd\|dependency | DeepMD-kit 依赖管理决策：拒绝 submodule，采用外部环境依赖 + 版本约束 + 运行时检查。 |
| [modulo-hypothesis.md](/docs/reports/modulo-hypothesis.md) | Report | Researchers | Low | active | descriptor\|modulo\|evidence | 结构描述符模长的物理意义假设与实证分析，包含统计与对比表格。 |

## 6. Archive（历史与弃用）

| Document | Category | Audience | Cadence | Status | Tags | 摘要（关键信息点） |
|---|---|---|---|---|---|---|
| [archive/README.md](/docs/archive/README.md) | Policy | All | Low | active | archive\|policy | 归档策略：适用版本、落地状态、修订说明约定。 |
| [Code_Review_Report_v2.7.1.md](/docs/archive/Code_Review_Report_v2.7.1.md) | Archive | Developers | Low | archived | code-review | 历史代码审查报告（性能/并发/测试等），仅供参考。 |
| [Variable_Review_Report.md](/docs/archive/Variable_Review_Report.md) | Archive | Developers | Low | archived | config\|variables | 历史变量/配置体系审查，记录过往不一致与修复建议。 |
| [SLURM_WORKFLOW_DESIGN.md](/docs/archive/refactoring/SLURM_WORKFLOW_DESIGN.md) | Archive | Developers | Low | archived | slurm\|workflow-chaining | 历史 Slurm 链式编排设计文档（当时缺少监控/等待机制的方案）。 |
| [Training_Module_Refactoring_Report_v2.9.0.md](/docs/archive/refactoring/Training_Module_Refactoring_Report_v2.9.0.md) | Archive | Developers | Low | active | refactoring\|training | 训练模块重构报告（版本化历史记录）。 |
| [Runner_Interface_Refractor_Plan.md](/docs/archive/refactoring/Runner_Interface_Refractor_Plan.md) | Archive | Developers | Low | active | refactoring\|cli | Runner 接口重构计划（历史）。 |
| [DP-EVA_Labeling_Module_Plan.md](/docs/archive/refactoring/DP-EVA_Labeling_Module_Plan.md) | Archive | Developers | Low | active | plan\|labeling | 标注模块规划（历史/待实现方向）。 |

## 7. Assets（图像）

| Asset | Category | Audience | Cadence | Status | Tags | 摘要（关键信息点） |
|---|---|---|---|---|---|---|
| [dpeva-workflow.png](/docs/img/dpeva-workflow.png) | Asset | All | Low | active | diagram\|workflow | DP-EVA 工作流示意图（用于架构与指南插图）。 |
| [modulo_distribution.png](/docs/design/modulo_distribution.png) | Asset | Researchers | Low | active | plot\|descriptor | 模长分布图（配套科研报告）。 |
