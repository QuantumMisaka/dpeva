# Analysis Workflow功能拆解与审查 Spec

## Why
当前需要对DP-EVA的Analysis Workflow形成可验证的职责边界认知，而不仅是阅读代码。通过“功能拆解 + 实际运行 + 代码审查”的闭环，可识别能力缺口、耦合风险和潜在故障点。

## What Changes
- 梳理Analysis Workflow的功能点与执行阶段，形成结构化功能拆解结果
- 基于两个指定配置分别执行analysis命令并采集关键运行证据（日志、产物、统计信息）
- 对照运行证据与代码实现，明确Analysis Workflow在项目中的功能职责与边界
- 对Analysis相关代码做细致审查，覆盖功能完备性、模块耦合度、可维护性、鲁棒性与风险点
- 输出带优先级的审查结论与可落地改进建议（短期修复与中期重构）

## Impact
- Affected specs: analysis workflow capability, observability, reliability, maintainability
- Affected code: `src/dpeva/analysis/`, `src/dpeva/workflows/`中analysis相关入口与调度代码，`src/dpeva/cli.py` analysis命令分发，可能涉及公共I/O与日志模块

## ADDED Requirements
### Requirement: Analysis Workflow功能拆解
系统 SHALL 提供可追溯的Analysis Workflow功能拆解，包含输入、处理阶段、输出产物与失败处理路径。

#### Scenario: 功能点完整识别
- **WHEN** 对Analysis模块及workflow入口进行静态分析
- **THEN** 产出按阶段组织的功能点清单，并标注每个功能点的代码入口与依赖关系

### Requirement: 双配置运行证据采集
系统 SHALL 使用指定的两个配置文件分别触发analysis workflow，并记录运行日志与输出结果用于职责判定。

#### Scenario: 配置一执行并记录
- **WHEN** 使用 `test/s-head-huber/config_analysis_dataset.json` 启动analysis
- **THEN** 记录命令输出、关键日志片段、生成文件与失败/告警信息

#### Scenario: 配置二执行并记录
- **WHEN** 使用 `test/s-head-huber/config_analysis.json` 启动analysis
- **THEN** 记录命令输出、关键日志片段、生成文件与失败/告警信息

### Requirement: 职责边界与代码审查结论
系统 SHALL 基于“代码 + 运行证据”给出Analysis Workflow职责边界与审查结论。

#### Scenario: 职责与风险评估
- **WHEN** 完成双配置执行与代码审读
- **THEN** 输出功能职责说明，并按“功能完备性、耦合度、风险点”给出分项审查结论与改进建议

## MODIFIED Requirements
### Requirement: 无
本次为新增审查型需求，不修改既有功能规范。

## REMOVED Requirements
### Requirement: 无
**Reason**: 本次需求不涉及功能下线。
**Migration**: 无需迁移。
