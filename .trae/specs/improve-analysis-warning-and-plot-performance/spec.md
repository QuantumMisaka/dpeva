# Analysis 可视化告警与性能优化 Spec

## Why
在 `config_analysis_val.json` 的 model_test 分析流程中，日志显示明显性能长尾（约 7 分钟级绘图阶段）与重复 warning，影响分析效率与日志可读性。需要在不破坏现有统计结果的前提下，提升大规模数据可视化性能并消除可预期的布局告警。

## What Changes
- 新增 Analysis 可视化信息量配置 `plot_level`，可选 `basic` / `full`，默认 `full`。
- 优化 `distribution_with_error` 布局策略，移除/替换不兼容 `tight_layout` 的调用路径，避免重复 UserWarning。
- 在 analysis 执行日志中输出阶段级耗时统计，明确解析、组分加载、绘图等关键阶段耗时。
- 新增“单图耗时告警”机制：当任一图绘制耗时超过阈值（默认 60s）时，输出 WARNING 并提示切换到 `basic` 模式。
- 收敛配置复杂度：不引入“用户显式控制图簇”这类细粒度开关，避免配置膨胀与逻辑分叉。
- 补充回归测试与基准验证，确保指标一致性与 warning 消失。

## Impact
- Affected specs: Analysis model_test 可视化能力、Analysis 可观测性能力、配置可控性能力
- Affected code: `src/dpeva/config.py`、`src/dpeva/workflows/analysis.py`、`src/dpeva/analysis/managers.py`、`src/dpeva/inference/visualizer.py`、相关 tests 与示例配置

## ADDED Requirements
### Requirement: Analysis 出图信息量分级
系统 SHALL 提供 `plot_level` 配置项，支持 `basic` 与 `full` 两档，默认 `full`，用于控制 analysis 输出图表信息量。

#### Scenario: 默认全量模式
- **WHEN** 用户未显式设置 `plot_level`
- **THEN** 系统采用 `full` 模式，保持当前完整图表输出行为

#### Scenario: 精简模式
- **WHEN** 用户设置 `plot_level=basic`
- **THEN** 系统仅生成基础诊断图，跳过高成本增强图，并在日志中输出模式说明

### Requirement: 单图耗时告警
系统 SHALL 记录每张图的绘制耗时；当耗时超过阈值（默认 60 秒）时输出 WARNING，并给出切换 `basic` 模式建议。

#### Scenario: 慢图提醒
- **WHEN** 任一图绘制耗时超过阈值
- **THEN** 日志包含图名、耗时、阈值与“建议切换到 basic 模式”的提示信息

### Requirement: Analysis 阶段耗时可观测
系统 SHALL 输出结构化阶段耗时日志，至少覆盖结果解析、组分加载、核心统计与绘图阶段。

#### Scenario: 定位性能瓶颈
- **WHEN** 用户运行 analysis 并查看日志
- **THEN** 能直接看到各阶段耗时与总耗时，支持快速定位慢点

## MODIFIED Requirements
### Requirement: Distribution-with-error 绘图布局稳定性
系统 SHALL 使用与当前子图结构兼容的布局方式，避免 `Axes not compatible with tight_layout` warning，并保持图像输出完整可读。

#### Scenario: with_error 图生成
- **WHEN** 生成 `dist_*_with_error.png` 图
- **THEN** 日志中不出现 tight_layout 兼容性 warning，且图片正常保存

## REMOVED Requirements
### Requirement: 用户显式控制图簇开关
**Reason**: 细粒度图簇开关会显著增加配置与分支复杂度，不利于维护与用户理解。  
**Migration**: 统一使用 `plot_level` 两档策略；需要完整信息时用 `full`，性能优先时用 `basic`。
