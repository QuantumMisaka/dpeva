# CollectionWorkflow 出图精简与分层策略 Spec

## Why
实战反馈显示 CollectionWorkflow 当前部分 parity 图在多数场景下决策价值有限，导致默认输出噪声偏高、人工解读成本增加。需要明确“默认必需图”与“可选诊断图”边界，并将出图行为配置化、可审计化。

## What Changes
- 确认并固化“低使用价值图像”判定依据（触发频率、决策贡献、运维成本）
- 新增 CollectionWorkflow 出图分层策略：Core（默认开启）与 Diagnostic（默认关闭）
- 为 Diagnostic 图提供显式配置开关与日志说明，确保按需启用
- 统一输出清单与跳过原因记录，便于复盘和审计
- 补充单元测试与文档，覆盖默认/开启/关闭三类路径

## Impact
- Affected specs: Collection 可视化输出契约、Collection 运行日志契约、配置文档契约
- Affected code: `src/dpeva/workflows/collect.py`、`src/dpeva/config.py`、`src/dpeva/uncertain/visualization.py`、`tests/unit/workflows/test_collect_refactor.py`、相关文档

## ADDED Requirements
### Requirement: Collection 出图分层控制
系统 SHALL 提供 Collection 出图分层控制能力，允许将低使用价值图像从默认产物中移除，并支持按需启用。

#### Scenario: 默认运行仅输出 Core 图
- **WHEN** 用户以默认配置运行 `dpeva collect`
- **THEN** 系统仅生成 Core 层图像，不生成 Diagnostic 层图像
- **AND** 日志记录被跳过的 Diagnostic 图像与原因

#### Scenario: 显式开启 Diagnostic 图
- **WHEN** 用户在配置中开启 Diagnostic 图选项
- **THEN** 系统在满足数据前置条件时生成对应 Diagnostic 图像
- **AND** 输出目录与命名规则保持兼容

### Requirement: 出图价值审计记录
系统 SHALL 在运行日志或摘要中输出本次出图清单、跳过清单与触发条件，支持回溯判断“图是否真正被使用”。

#### Scenario: 生成后审计
- **WHEN** Collection 任务完成
- **THEN** 用户可从日志/摘要读取每类图像的生成状态（generated/skipped）与原因

## MODIFIED Requirements
### Requirement: Collection 默认出图策略
Collection 默认出图策略从“尽可能多出图”调整为“以决策最小闭环为目标的精简出图”，优先保留对样本筛选决策直接有贡献的图像。

## REMOVED Requirements
### Requirement: 默认生成全部 parity 类诊断图
**Reason**: 多数 parity 诊断图在常规迭代中使用率低、噪声高，且不影响筛选主流程闭环。  
**Migration**: 将此类图迁移到 Diagnostic 层，通过显式开关按需启用；默认关闭。
