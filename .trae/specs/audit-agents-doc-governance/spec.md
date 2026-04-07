# DP-EVA 文档治理审阅与 AGENTS 目录化改造 Spec

## Why
当前项目已经具备较丰富的 `docs/` 文档体系、配置中心化机制与工作流解耦实现，但 `AGENTS.md` 仍主要停留在轻量操作提示层，尚未准确承担“项目核心目标与规范目录”的角色。同时，文档系统本身仍可能存在与核心代码不一致、冗余、错漏或过时的内容。需要先基于 `docs/`、`examples/recipes/` 与 `src/` 的交叉审阅形成证据化结论，先治理文档系统，再提出一套精炼且不重复现有文档的 `AGENTS.md` 改造方案。

## What Changes
- 深读并审阅 `docs/`、`examples/recipes/`、`src/`，形成一份详细的项目目标意义、治理规范落实度与工程化实践缺口审阅文档
- 审核文档系统中的工程规范是否已被代码与示例配置真实落实，并明确“文档已声明但落实不足”与“代码已有实践但文档未体现”两类问题
- 在交叉审阅基础上，先对项目文档系统做细致治理，修复与核心代码不一致的内容，并识别可删减的冗余、错漏或过时文档
- 形成一套针对 `AGENTS.md` 的细致改进方案，使其成为项目开发目标、核心工程约束与文档入口的精简目录
- 约束 `AGENTS.md` 内容边界：不重复 `.trae/rules/project_rules.md` 的 AI 开发指引，不重复现有开发规范文档中的细节正文，仅保留高价值摘要与跳转
- 视审阅结论补充必要的文档系统改进建议，确保 `AGENTS.md` 与 `docs/` 形成联动而不是新的信息孤岛

## Impact
- Affected specs: 文档治理审计、工程规范追踪、AGENTS 导航设计、开发者入口优化
- Affected code: `AGENTS.md`、`docs/`、`examples/recipes/`、`src/dpeva/cli.py`、`src/dpeva/config.py`、`src/dpeva/utils/config.py`、`src/dpeva/workflows/*.py`

## ADDED Requirements
### Requirement: 交叉证据化审阅
系统 SHALL 基于文档、示例配置与核心代码三类证据，对 DP-EVA 的目标意义、治理规范与工程化实践进行交叉审阅，而非仅依据单一文档做结论。

#### Scenario: 审阅覆盖三类证据源
- **WHEN** 审阅任务开始执行
- **THEN** 必须覆盖 `docs/`、`examples/recipes/`、`src/` 三个范围
- **THEN** 审阅文档必须显式区分“文档中的声明”“代码中的实现”“示例中的对外契约”

### Requirement: 治理规范落实度核查
系统 SHALL 核查项目文档中声明的工程化规范是否已在核心代码与示例配置中得到妥善落实，并记录偏差。

#### Scenario: 文档到实现的闭环核查
- **WHEN** 审阅文档输出治理结论
- **THEN** 必须覆盖至少以下主题：配置与默认值统一管理、CLI 到 workflow 的主链路入口、工作流解耦、文档单一权威来源、文档归档与索引规则
- **THEN** 对每项主题必须判断其状态为“已落实”“部分落实”“未落实”或“实现存在但文档缺失”

### Requirement: 隐性工程实践补全文档化
系统 SHALL 识别核心代码中已经形成但尚未被当前文档系统清晰表达的约定俗成工程实践，并将其纳入审阅结论与改进建议。

#### Scenario: 发现代码先行的实践
- **WHEN** 代码中存在稳定且可复用的工程实践
- **THEN** 审阅文档必须标明其代码证据、实践价值、当前文档缺口与建议落点
- **THEN** 这类实践至少应考虑 CLI 前置校验、路径相对配置解析、`config.py` 配置中心化、包级导出层与主链路入口边界、工作流完成标记、阶段化工作流入口等内容

### Requirement: 文档系统先行治理
系统 SHALL 在形成审阅结论之后、制定 `AGENTS.md` 改造方案之前，先对项目文档系统执行一次面向一致性与精简性的治理。

#### Scenario: 先修文档再改 AGENTS
- **WHEN** 交叉审阅已识别出文档与代码、文档与示例配置之间的偏差
- **THEN** 必须先修复 `docs/` 中与核心代码不一致的内容
- **THEN** 必须检查是否存在冗余、错漏、过时或可删除的文档文件，并给出处理动作
- **THEN** 文档治理完成后，才能进入 `AGENTS.md` 目录化改造方案设计

### Requirement: AGENTS 目录化改造方案
系统 SHALL 为 `AGENTS.md` 产出一套面向项目的精简改造方案，使其作为“核心目的与规范目录”而非重复性手册。

#### Scenario: AGENTS 改造方案可直接落地
- **WHEN** 改造方案形成
- **THEN** 必须说明 `AGENTS.md` 建议保留、删减、新增与改写的内容模块
- **THEN** 必须明确每个模块应链接到 `docs/` 中的哪个权威文档
- **THEN** 必须避免重复 `.trae/rules/project_rules.md` 中的 AI 行为准则
- **THEN** 必须避免复写 `docs/guides/*`、`docs/policy/*`、`docs/governance/*` 中已存在的规范细节

### Requirement: 文档系统联动改进建议
系统 SHALL 在完成文档系统先行治理后，针对剩余的导航缺口、职责重叠或契约漂移提出与 `AGENTS.md` 配套的文档联动改进建议。

#### Scenario: AGENTS 改造触发文档联动
- **WHEN** 审阅发现 `AGENTS.md` 无法仅通过瘦身完成改进
- **THEN** 必须补充对应的文档侧改进建议
- **THEN** 改进建议需指明目标文档类别与预期职责，例如 Guides、Policy、Governance、Architecture 或 Reference

## MODIFIED Requirements
### Requirement: AGENTS 文档职责边界
`AGENTS.md` SHALL 从“泛化开发操作说明”调整为“项目开发目标、核心工程契约与文档导航入口”的目录型文档。

#### Scenario: AGENTS 成为高密度索引
- **WHEN** 开发者或 AI 首次打开 `AGENTS.md`
- **THEN** 应先看到项目目标、核心设计约束、必知工程实践与文档入口
- **THEN** 详细流程、质量门禁、治理规则与字段字典应通过链接导向 `docs/` 对应权威文档
- **THEN** 文档本体应保持简洁，避免成为与 `developer-guide.md` 重叠的第二份长文档

## REMOVED Requirements
### Requirement: AGENTS 承担细节性规范正文
**Reason**: 细节性规范正文已经分散并沉淀在 `docs/guides/`、`docs/policy/`、`docs/governance/` 与 `docs/reference/` 中，继续在 `AGENTS.md` 重复维护会削弱单一权威来源。
**Migration**: `AGENTS.md` 仅保留必要摘要与跳转，细节规范统一引用现有文档系统中的权威入口。
