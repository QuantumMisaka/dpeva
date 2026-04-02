# 仓库系统性代码审查 Spec

## Why
当前仓库覆盖源码、测试、文档、维护脚本与 CI/CD 配置，缺少一次面向整体质量基线的统一审查输出。需要形成可直接进入后续迭代的审查报告，识别高风险问题并明确修复优先级。

## What Changes
- 对 `src`、`tests`、`docs`、`.trae/skills`、`scripts`、`.github` 以及 `AGENTS.md`、`README.md`、`pyproject.toml` 执行系统性代码审查
- 按代码风格与规范一致性、架构设计、安全、性能、测试、文档、脚本可维护性、CI/CD 配置八个维度输出问题与建议
- 按严重级别分类问题，并给出可执行的修复建议、改进优先级与后续行动项
- 生成一份详细审查报告，满足项目文档协议并可直接用于后续迭代

## Impact
- Affected specs: 仓库质量审计、审查报告交付、后续整改规划
- Affected code: `src/`、`tests/`、`docs/`、`.trae/skills/`、`scripts/`、`.github/`、`AGENTS.md`、`README.md`、`pyproject.toml`

## ADDED Requirements
### Requirement: 全仓库质量审查覆盖
系统 SHALL 对用户指定的核心源码、测试、文档、脚本与 CI/CD 范围执行一次完整且可追溯的系统性审查。

#### Scenario: 审查范围完整
- **WHEN** 审查任务开始执行
- **THEN** 审查必须覆盖 `src`、`tests`、`docs`、`.trae/skills`、`scripts`、`.github` 以及 `AGENTS.md`、`README.md`、`pyproject.toml`
- **THEN** 审查结论必须显式说明每个范围已检查的重点与发现

### Requirement: 多维质量评估
系统 SHALL 依据统一维度对仓库进行质量评估，并在报告中分别给出发现与建议。

#### Scenario: 八个维度均有结论
- **WHEN** 审查完成
- **THEN** 报告必须覆盖代码风格与规范一致性、架构设计合理性、安全漏洞与风险、性能瓶颈、测试完整性与覆盖率、文档准确性与完整性、脚本可维护性与健壮性、CI/CD 流水线配置正确性及效率
- **THEN** 每个维度必须至少包含审查结论，若存在问题则附带定位依据与修复建议

### Requirement: 问题分级与整改优先级
系统 SHALL 将发现的问题按严重级别分类，并给出面向迭代的整改排序。

#### Scenario: 问题可直接进入迭代
- **WHEN** 报告输出问题清单
- **THEN** 每个问题必须包含严重级别、影响范围、问题描述、建议修复方向
- **THEN** 报告必须提供改进优先级与后续行动项，以支持后续排期

### Requirement: 审查报告符合项目文档协议
系统 SHALL 按项目文档协议生成审查报告，并确保文档引用完整。

#### Scenario: 报告可归档可索引
- **WHEN** 审查报告被创建
- **THEN** 报告必须写入 `docs/reports/`，并使用 `YYYY-MM-DD-Code-Review-<Topic>.md` 命名
- **THEN** 若新增或移动了 `.md` 文档，必须同步检查并更新 `docs/source/` 下对应索引，避免 Sphinx 引用失效

## MODIFIED Requirements
### Requirement: 质量门禁证据收集
系统 SHALL 在给出最终结论前收集必要的静态分析、测试、文档与配置证据，以避免仅凭主观判断输出审查结论。

#### Scenario: 结论有可验证依据
- **WHEN** 报告形成最终结论
- **THEN** 审查过程必须结合代码阅读、静态检查、测试或配置核查结果
- **THEN** 报告必须区分事实性发现与建议性改进，避免混淆

## REMOVED Requirements
