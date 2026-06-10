# CollectionWorkflow 作图一致性审查与修复 Spec

## Why
当前 CollectionWorkflow 的 PCA 结果图中，“All Data in Pool”灰色背景与 Candidate 点云分布明显不一致，影响采样结果可解释性和质量判断。需要先完成可追溯审查，再给出修复与验证方案，避免误导后续主动学习决策。

## What Changes
- 审查 CollectionWorkflow 到 UQVisualizer 的 PCA 作图数据流，定位灰色背景错位根因。
- 产出代码审查报告，落盘到 `docs/reports/`（按日期命名）。
- 产出可回溯实施计划，落盘到 `docs/plans/`（含状态与受众元信息）。
- 修复背景点 PCA 变换链路，确保与 Candidate 使用同一标准化与 PCA 投影坐标系。
- 增加针对作图链路的一致性校验（至少覆盖标准 DIRECT 路径；若适用同步覆盖 2-DIRECT）。
- 更新相关文档索引，确保 Sphinx toctree 无悬挂引用。

## Impact
- Affected specs: Collection 可视化一致性、采样结果可解释性、文档治理与归档流程
- Affected code: `src/dpeva/sampling/manager.py`、`src/dpeva/uncertain/visualization.py`、`src/dpeva/workflows/collect.py`、`tests/`、`docs/reports/`、`docs/plans/`、`docs/source/`

## ADDED Requirements
### Requirement: 背景点与候选点必须处于同一 PCA 坐标系
系统 SHALL 在绘制 `Final_sampled_PCAview.png` 时，保证背景点（All Data in Pool）与候选点（Candidate）使用同一条变换链路（同一标准化器与同一 PCA 模型）。

#### Scenario: Joint/Normal 模式下背景坐标一致
- **WHEN** CollectionWorkflow 生成 `full_features` 与 `all_features` 并调用 `plot_pca_analysis`
- **THEN** 背景点不应异常塌缩到 `(0,0)` 附近
- **AND** 背景点云尺度与候选点云在同一坐标量纲下可比较

### Requirement: 审查与计划文档可追溯
系统 SHALL 在修复实施前输出可审计文档：审查报告与执行计划，分别落盘到 `docs/reports/` 与 `docs/plans/`，并满足项目命名与索引规范。

#### Scenario: 文档交付与定位
- **WHEN** 本次问题被确认与归因
- **THEN** 审查报告包含现象、证据链、根因、风险评估、修复建议
- **AND** 计划文档包含任务拆解、验收标准、回滚与验证策略

## MODIFIED Requirements
### Requirement: SamplingManager 背景投影流程
现有背景特征投影流程 SHALL 从“仅调用 PCA.transform”调整为“先通过与采样一致的标准化步骤，再通过同一 PCA 模型投影”，并在 DIRECT 与 2-DIRECT 对应路径保持一致性。

## REMOVED Requirements
### Requirement: 无
**Reason**: 本次不删除既有能力，仅修正实现与补齐流程约束。
**Migration**: 无需迁移，保持对外接口兼容。
