# DP-EVA 文档系统规划与重构报告

面向目录：[docs/](/docs)

本报告目标：基于现有文档盘点与代码现状，给出一套可执行的文档体系规划方案，包括分类体系、缺失文档补充、优化建议、模板标准、版本维护机制与质量标准，并给出实施步骤、优先级、时间安排与责任分配。

## 1. 现状盘点与质量评估

### 1.1 当前文档分布（按目录）

- `docs/guides/`（主线指南）
  - [developer-guide.md](/docs/guides/developer-guide.md)
  - [quickstart.md](/docs/guides/quickstart.md)
  - `guides/testing/*`（Slurm 集成测试专题）
- `docs/reference/`（字段字典与校验）
  - [config-schema.md](/docs/reference/config-schema.md)
  - [validation.md](/docs/reference/validation.md)
  - 字段/校验的单一权威来源为 `docs/reference/*`
- `docs/architecture/`（架构与决策）
  - [DP-EVA_Design_Report.md](/docs/design/DP-EVA_Design_Report.md)
  - [2026-02-04-deepmd-dependency.md](/docs/architecture/decisions/2026-02-04-deepmd-dependency.md)
- `docs/reports/`（一次性分析/实验）
  - [modulo-hypothesis.md](/docs/reports/modulo-hypothesis.md)
- `docs/archive/`（历史/弃用）
  - [Code_Review_Report_v2.7.1.md](/docs/archive/Code_Review_Report_v2.7.1.md)
  - [Variable_Review_Report.md](/docs/archive/Variable_Review_Report.md)
  - `archive/refactoring/*`
- 资产图片：`docs/img/`、`docs/design/*.png`

### 1.2 主要问题（质量与一致性）

- 缺少 `docs/` 总入口与导航，读者难以快速定位“权威来源”。
- 字段/规则存在过期风险：例如历史文档与 README 示例曾出现已废弃字段（已先行修正部分示例与校验文档）。
- 同一主题在多处重复叙述：配置说明、集成测试专题存在多份文档交叉覆盖，后续极易不同步。
- “现状 vs 历史/计划”边界不清：部分设计报告混合历史实施与建议，缺少适用版本与状态标识。

## 2. 文档分类体系设计（功能/受众/更新频率三维）

### 2.1 按受众（Audience）

- Users：安装、快速跑通、CLI 使用、常见问题
- Developers：架构、编码规范、测试策略、贡献指南
- Researchers：算法假设、实验结论、方法学讨论
- Maintainers：发布/版本兼容、迁移指南、维护机制

### 2.2 按功能（Intent）

- Guides：怎么用/怎么做（流程性、示例性）
- Reference：查表/权威定义（字段、约束、术语）
- Architecture：系统结构（与代码一致的稳定描述）
- ADR/Decisions：关键技术决策（一次性结论，可追溯）
- Reports：分析/实验/评审报告（只追加）
- Archive：弃用/历史（只读）

### 2.3 按更新频率（Cadence）

- 高频（每次接口变更必须同步）：Guides、部分 Architecture
- 中频（配置字段变更同步）：Reference、Slurm 指南
- 低频（只追加）：ADR、Reports、Archive

## 3. 缺失文档识别与补充计划

### 3.1 缺失清单（P0/P1/P2）

- P0（影响上手与可用性）
  - `docs/README.md`：统一入口与导航（已补齐）
  - Quickstart：最短路径跑通（已建立骨架：[quickstart.md](/docs/guides/quickstart.md)）
  - CLI 指南（已建立骨架：[cli.md](/docs/guides/cli.md)）
  - Troubleshooting（已建立骨架：[troubleshooting.md](/docs/guides/troubleshooting.md)）
- P1（影响长期维护与一致性）
  - 配置编写指南（已建立骨架：[configuration.md](/docs/guides/configuration.md)）
  - Slurm 指南（已建立骨架：[slurm.md](/docs/guides/slurm.md)）
  - Glossary（术语表，避免 DataPool/System/Descriptor 等歧义）
  - 兼容性矩阵（DeepMD/dpdata/Python/后端）
- P2（工程化与对外发布）
  - CHANGELOG / Release Notes（当前缺失）
  - 迁移指南（字段重命名、弃用字段说明）

## 4. 现有文档优化建议（结构/完整性/示例/图表）

### 4.1 README（仓库根）

- 问题：示例字段容易随版本变更过期；测试命令路径历史不一致。
- 建议：
  - Quickstart 只保留“最小配置 + 链接到 docs/guides/quickstart.md”
  - 示例配置字段与 `docs/reference/config-schema.md` 建立强链接

### 4.2 Developer Guide（主线）

- 建议拆分职责：
  - “架构概览/模块边界”拆到 `docs/architecture/overview.md`
  - “配置字段全量表”只保留链接到 `docs/reference/*`
  - “测试规范”与“集成测试专题”通过索引页串联，避免四处复制

### 4.3 reference（字段字典与校验）

- 建议将 `config-schema.md` 定义为单一权威来源，并逐步自动化生成：
  - 由 Pydantic 模型提取字段、类型、默认值、约束
  - 文档仅补充“语义解释/示例/迁移说明”

### 4.4 design/ 与 archive/

- `design/` 建议引入 ADR（Decision）子体系，将一次性决策从“叙述性报告”中抽离。
- `archive/` 建议增加入口索引与命名规范，避免读者误读历史文档为现行规范。

## 5. 文档模板标准化方案

- 模板已落地（可作为新文档基线）：
  - 页面模板：[docs/_templates/page.md](/docs/_templates/page.md)
  - ADR 模板：[docs/_templates/adr.md](/docs/_templates/adr.md)
  - 报告模板：[docs/_templates/report.md](/docs/_templates/report.md)

统一元信息建议（适用于 `active` 文档）：Status / Applies-To / Owners / Last-Updated / Related Links。

## 6. 文档版本管理与维护机制

权威规则与建议流程见：

- 文档维护机制：[docs/policy/maintenance.md](/docs/policy/maintenance.md)

关键落点：

- 接口/配置变更必须同步更新文档与示例
- 报告/归档文档默认冻结，只追加不回写
- 每季度做一次 docs 体检（断链/过期字段/示例抽检）

## 7. 文档质量评估标准

权威标准见：

- 文档质量标准：[docs/policy/quality.md](/docs/policy/quality.md)

建议将“质量评分/验收清单”纳入 PR Review 流程，作为 `active` 文档准入条件。

## 8. 实施步骤、优先级、时间安排与责任人分配

### 8.1 实施步骤（按优先级）

- P0（立刻提升可用性）
  - 建立 docs 入口与导航（已完成）
  - 建立 Quickstart/CLI/排障骨架（已完成）
  - 修正明显过期示例（已完成部分，持续治理）
- P1（形成稳定的“单一权威来源”）
  - 将配置字段解释从叙述性文档中剥离，所有字段解释统一链接到 `docs/reference`
  - 建立术语表与兼容性矩阵
  - 将“架构现状”从设计报告中剥离成可维护的 `architecture/overview`
- P2（工程化）
  - 增加 CHANGELOG / Release Notes
  - 引入 ADR 目录与决策索引
  - 评估是否引入 MkDocs/Sphinx 生成站点（当前 repo 仅提供 Sphinx 依赖但未成体系）

### 8.2 时间安排（建议以两周为一个迭代）

- 第 1–2 周（Iteration 1：可用性）
  - 完成 Quickstart/CLI/Configuration/Slurm/Troubleshooting 的可运行示例与截图/图表
  - 将 README 的 Quickstart 收敛为“最小示例 + 链接到 docs/guides”
- 第 3–4 周（Iteration 2：一致性）
  - 完成“单一权威来源”治理：字段表与指南去重
  - 增补 Glossary 与 Compatibility Matrix
  - 梳理 archive 与 design 的入口索引
- 第 5–6 周（Iteration 3：工程化）
  - 增加 CHANGELOG / Migration Guide
  - 引入 ADR 体系并迁移关键决策文档
  - 评估并落地文档站点工具链（可选）

### 8.3 责任人分配（RACI 建议）

建议按“角色/模块”分配，避免依赖单一个人：

- 文档体系 Owner（A）：项目维护负责人（Maintainer）
- Users Guides Owner（R）：CLI/用户接口维护人
- Reference Owner（R）：配置模型维护人（Pydantic Model Owner）
- Slurm Guide Owner（R）：平台/集群适配负责人（HPC/Infra Owner）
- Workflow Owners（R）：Train/Infer/Feature/Collect 各模块维护人
- Reviewers（C）：至少 1 名模块 Owner + 1 名文档体系 Owner
- 执行支持（I）：全体贡献者（按 PR 通知）

推荐的落地动作：为每个 `active` 文档在头部填写 `Owners`，并在 PR 模板中增加“文档是否需要更新”的检查项。
