# Analysis Workflow Slurm Backend Spec

## Why
当前 Analysis Workflow 仅支持本地同步执行，无法与项目现有的 Slurm 提交流程对齐，限制了集群场景下的大规模分析能力。为保持工作流一致性与可维护性，需要复用 Collection 的成熟 Slurm backend 设计，为 Analysis 增加标准化 Slurm 支持。

## What Changes
- 为 Analysis Workflow 增加 `submission.backend=slurm` 执行路径，采用与 Collection 一致的“自提交 + worker 本地执行”模式。
- 在 Analysis 配置中引入与 Collection 对齐的 Slurm 参数入口（`submission.slurm_config`），并保持本地 backend 行为不变。
- 为 Analysis 增加 Slurm 提交脚本生成与提交流程，脚本内注入内部 backend 覆盖，避免递归提交。
- 更新 `examples/recipes/analysis` 下案例为 Slurm backend 版本，并使 Slurm 参数结构与 `examples/recipes/collection` 对齐。
- 新增并完善单元测试：覆盖 Analysis Slurm 提交分支、关键命令构造、脚本参数透传与本地分支回归。

## Impact
- Affected specs: Analysis workflow execution, Submission backend integration, Recipe examples, Unit test coverage
- Affected code: `src/dpeva/config.py`, `src/dpeva/cli.py`, `src/dpeva/workflows/analysis.py`, `examples/recipes/analysis/*`, `tests/unit/workflows/test_analysis_*`

## ADDED Requirements
### Requirement: Analysis 支持 Slurm Backend
系统 SHALL 支持在 Analysis Workflow 中使用 `submission.backend=slurm` 进行作业提交，并通过 Slurm 脚本在计算节点执行 Analysis 任务。

#### Scenario: Slurm 提交成功
- **WHEN** 用户使用包含 `submission.backend=slurm` 的 analysis 配置运行命令
- **THEN** 系统生成 Analysis 提交脚本并提交到 Slurm 队列
- **AND** 脚本中的执行命令使用配置文件路径启动 analysis worker
- **AND** worker 进程通过内部 backend 覆盖走本地执行路径

#### Scenario: 本地行为保持兼容
- **WHEN** 用户使用 `submission.backend=local` 或未显式配置 backend
- **THEN** Analysis Workflow 按现有本地路径直接执行分析逻辑
- **AND** 不生成 Slurm 提交脚本

### Requirement: Analysis Slurm 示例配置对齐 Collection
系统 SHALL 在 analysis recipes 中提供 Slurm backend 示例，并与 collection recipes 保持一致的 Slurm 参数结构与字段语义。

#### Scenario: 示例可直接复用 Slurm 参数模板
- **WHEN** 用户查看或复制 `examples/recipes/analysis` 中配置
- **THEN** 能看到与 collection 示例一致的 `submission.slurm_config` 字段布局
- **AND** 参数命名与层级保持一致，减少跨 workflow 使用成本

### Requirement: Slurm 路径单元测试覆盖
系统 SHALL 为 Analysis Slurm backend 增加单元测试，验证提交流程、命令构造与参数透传正确性。

#### Scenario: 测试验证关键行为
- **WHEN** 运行 analysis workflow 相关单元测试
- **THEN** 测试可断言 Slurm 脚本生成与提交调用被触发
- **AND** 可断言脚本命令包含正确配置路径与内部 backend 覆盖
- **AND** 现有本地分支测试继续通过

## MODIFIED Requirements
### Requirement: Analysis Workflow 执行入口
系统 SHALL 将 Analysis Workflow 执行入口从“仅本地执行”扩展为“支持 local 与 slurm 双 backend”，并复用项目统一的 submission 组件完成脚本渲染与提交。

## REMOVED Requirements
### Requirement: Analysis 仅支持本地同步执行
**Reason**: 该限制与项目多 workflow 的 Slurm 一致性目标冲突，无法满足集群化执行需求。  
**Migration**: 现有本地配置无需迁移；需要集群执行时仅需在 analysis 配置中补充 `submission.backend=slurm` 及 `slurm_config`。
