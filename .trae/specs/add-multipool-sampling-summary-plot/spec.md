# Collection 多数据池采样总结图 Spec

## Why
在多数据池（joint sampling）场景下，当前 `Final_sampled_PCAview` 无法直观看出最终样本来自哪些数据池，影响采样结果解释与复盘效率。需要以最小改动新增一张“按数据池可区分”的采样总结图，并仅在多数据池模式触发。

## What Changes
- 在 CollectionWorkflow 多数据池采样路径新增一张“按池区分”的采样总结图
- 复用现有 PCA 结果与绘图链路，避免重复计算与大范围重构
- 新图仅在 `use_joint=true`（多数据池）时生成，单池模式不触发
- 全集背景点保持现有灰色风格，不改变基础分布参照层
- 被采样点按数据池来源使用不同颜色或形状进行区分
- 面向十几个到几十个数据池场景，图注放置在图外侧并支持紧凑布局
- 统一命名、输出路径与日志记录，保证可审计
- 补充单元测试与文档说明，覆盖触发条件与兼容性

## Impact
- Affected specs: Collection 可视化输出契约、Joint Sampling 结果可解释性契约
- Affected code: `src/dpeva/workflows/collect.py`、`src/dpeva/uncertain/visualization.py`、`tests/unit/workflows/test_collect_refactor.py`、Collection 相关文档

## ADDED Requirements
### Requirement: 多数据池采样来源可视化
系统 SHALL 在多数据池采样模式下额外生成一张可区分不同数据池来源的采样总结图。

#### Scenario: 多数据池模式触发新图
- **WHEN** Collection 运行在 joint sampling（多数据池）模式
- **THEN** 系统在现有 `Final_sampled_PCAview` 基础上额外生成“按池区分”的总结图
- **AND** 全集背景点保持灰色
- **AND** sampled 点按不同数据池来源具有可区分视觉编码（颜色或形状）

### Requirement: 多数据池图注外置与可读性
系统 SHALL 在多数据池总结图中采用图外图注布局，以保证大量数据池标签可读。

#### Scenario: 数据池数量较多时图注布局
- **WHEN** 数据池数量达到十几个或更多
- **THEN** 图注显示在绘图区外侧（不覆盖图内散点）
- **AND** 图注布局支持紧凑显示（如多列）以降低遮挡和拥挤

#### Scenario: 单数据池模式不触发
- **WHEN** Collection 运行在单数据池模式
- **THEN** 系统不生成该“按池区分”总结图
- **AND** 不影响现有默认出图行为

## MODIFIED Requirements
### Requirement: Collection 采样总结图输出
Collection 采样总结图输出能力扩展为“基础总结图 +（多数据池时）按池区分总结图”，并保持原有结果与命名兼容，不引入破坏性变更。

## REMOVED Requirements
### Requirement: 无
**Reason**: 本次为增量可视化增强，不移除既有能力。  
**Migration**: 无需迁移，历史配置可直接沿用。
