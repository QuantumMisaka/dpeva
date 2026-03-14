# Collection 导出路径与采样流程统一修复 Spec

## Why
当前 Collection Workflow 在启用 UQ 筛选后导出 sampled/other 数据时出现额外目录嵌套，导致工程级输出路径错误。与此同时，启用 UQ 与纯 DIRECT 的采样导出逻辑出现分叉，增加回归风险并降低可维护性。

## What Changes
- 修复单池普通采样与联合采样在启用 UQ 后的导出目录重复嵌套问题，确保 `sampled_dpdata` 与 `other_dpdata` 导出层级一致且稳定。
- 统一“UQ 筛选后 Candidates 进入 DIRECT”与“无 UQ 直接 DIRECT”的导出路径构建逻辑，消除冗余分支。
- 增强 CollectionWorkflow 导出日志，显式记录 sampled 与 other 两类数据的最终导出位置及导出统计。
- 为路径解析、导出行为与日志内容补齐单元测试/集成回归测试，覆盖单池与联合采样关键场景。
- 引入工程化防回归机制：以统一导出策略函数为唯一入口，并在代码审查中加入“路径重复拼接与分支分叉”检查项。

## Impact
- Affected specs: Collection 导出行为、Collection 可观测性日志、Collection 采样流程一致性
- Affected code: `src/dpeva/workflows/collect*`、`src/dpeva/io/*collection*`（或等价导出实现文件）、`tests/unit/workflows/collect*`、`tests/integration/*collect*`

## ADDED Requirements
### Requirement: 导出目录层级一致性保障
系统 SHALL 在 Collection 导出 sampled/other 数据时保证目录层级不重复、不额外嵌套。

#### Scenario: 启用 UQ 的单池普通采样导出
- **WHEN** 用户执行 `config_collect_normal.json` 并完成 UQ + DIRECT 采样
- **THEN** sampled 数据导出到 `.../dpdata/sampled_dpdata/*`
- **AND** other 数据导出到 `.../dpdata/other_dpdata/*`
- **AND** 不出现 `other_dpdata/other_dpdata` 或等价重复层级

#### Scenario: 启用 UQ 的联合采样导出
- **WHEN** 用户执行 `config_collect_joint.json` 并完成 UQ + DIRECT 采样
- **THEN** sampled 与 other 导出目录遵循统一层级约束
- **AND** 不因 pool/system 命名导致重复目录拼接

### Requirement: UQ+DIRECT 与 DIRECT-only 导出流程统一
系统 SHALL 通过单一导出路径构建入口处理两类 Candidates（UQ 后 Candidates 与原始 Candidates），避免分叉实现。

#### Scenario: DIRECT-only 路径回归不变
- **WHEN** 用户执行 `config_direct_normal.json` 或 `config_direct_joint.json`
- **THEN** 导出目录结构保持现有正确行为
- **AND** 与 UQ+DIRECT 共用同一导出路径构建策略

### Requirement: 导出日志可观测性增强
系统 SHALL 在 CollectionWorkflow 日志中输出 sampled 与 other 的导出位置和导出数量摘要。

#### Scenario: 导出日志完整记录
- **WHEN** CollectionWorkflow 完成采样导出
- **THEN** `collection.log` 中包含 sampled 数据导出路径
- **AND** 包含 other 数据导出路径
- **AND** 包含与路径对应的导出结构/帧数统计信息

## MODIFIED Requirements
### Requirement: Collection 导出行为
Collection 导出模块必须以“规范化系统名 + 目标根目录 + 导出类型子目录”的顺序构建路径，并在写出前执行重复段检测与规整，确保任何输入命名下最终目录唯一且可预测。

## REMOVED Requirements
### Requirement: 导出路径按候选来源分支独立实现
**Reason**: 候选来源（UQ 后或非 UQ）不应影响导出路径策略，分支独立实现会引入行为漂移与重复拼接风险。  
**Migration**: 将分支实现迁移到统一导出策略函数，原分支仅负责传入候选集合与上下文元数据。
