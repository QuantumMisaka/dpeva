# Collection 出图接线与未接入模块审查 Spec

## Why
当前 `CollectionWorkflow` 中存在已实现但未接入标准主链路的出图逻辑（如 Candidate parity 图），同时项目范围内可能存在更多“代码模块存在但未接入工作流”的隐性技术债，需要一次系统化梳理并落地修复。

## What Changes
- 将 `UQ-QbC-Candidate-fdiff-parity.png` 与 `UQ-RND-Candidate-fdiff-parity.png` 接入 Collection 在“含 DFT label 数据”条件下的标准出图流程。
- 对 Collection 可视化模块进行同类“未接线”逻辑排查，并明确处置（接入/保留但标注为手动/删除）。
- 对全项目工作流（`train/infer/analysis/feature/collect/label/clean`）执行未接入模块审查，输出结构化审计结论与处置建议。
- 增补回归测试，防止后续再次出现“模块定义存在但主链路失联”。

## Impact
- Affected specs:
  - Collection 含 label 场景下的 UQ/误差可视化能力
  - 工作流模块接入完整性治理（跨 train/infer/analysis/feature/collect/label/clean）
- Affected code:
  - `src/dpeva/workflows/collect.py`
  - `src/dpeva/uncertain/visualization.py`
  - `tests/unit/uncertain/test_visualization.py`
  - `tests/unit/workflows/test_collect.py`（或同等 workflow 测试文件）
  - 与工作流入口相关文件（`src/dpeva/cli.py`、`src/dpeva/workflows/*`、`src/dpeva/*`）
  - 审计结果文档（位于 `docs/reports/`，遵循现有命名规范）

## ADDED Requirements
### Requirement: Collection 在含 DFT label 条件下必须输出 Candidate parity 图
系统 SHALL 在 Collection 工作流满足真值误差可用条件时，自动生成 Candidate parity 两张图并保存到标准 `view` 输出目录。

#### Scenario: 含 DFT label 的正常执行
- **WHEN** 用户执行 `dpeva collect <config>`，且 `has_gt=True` 且 `diff_maxf_0_frame` 非空并全有限
- **THEN** 流程自动生成：
  - `UQ-QbC-Candidate-fdiff-parity.png`
  - `UQ-RND-Candidate-fdiff-parity.png`
- **AND** 图像输出路径符合 `<project>/<root_savedir>/view/`

#### Scenario: 无 DFT label 或误差数据不可用
- **WHEN** `has_gt=False` 或 `diff_maxf_0_frame` 缺失/空/包含非有限值
- **THEN** Candidate parity 图不生成
- **AND** 日志明确说明跳过原因

### Requirement: Collection 模块未接线逻辑需被显式治理
系统 SHALL 对 Collection 相关可视化能力进行未接线审查，并将每一项分类为“已接入 / 保留手动 / 废弃移除”。

#### Scenario: 发现未接线函数
- **WHEN** 审查发现函数在生产代码中仅定义无主链路调用
- **THEN** 必须给出处置动作与理由
- **AND** 结果纳入审计报告

### Requirement: 全工作流未接入模块审查必须可复核
系统 SHALL 对 `train/infer/analysis/feature/collect/label/clean` 进行全量“模块定义-入口接线”一致性审查，形成可复核的结构化结果。

#### Scenario: 输出跨工作流审查结论
- **WHEN** 完成全量扫描
- **THEN** 每条结论至少包含：模块位置、预期所属工作流、当前接入状态、证据链、处置建议、优先级

## MODIFIED Requirements
### Requirement: Collection 含 label 出图链路
Collection 的“含真值误差”可视化链路从“仅身份/误差散点与 parity”扩展为“包含 Candidate parity 两图”的完整链路，并保持现有条件门控一致（仅在真值误差可用时触发）。

## REMOVED Requirements
### Requirement: Candidate parity 仅可手动调用
**Reason**: 与用户对标准工作流可复现出图的一致性要求冲突。  
**Migration**: 保留 `plot_candidate_vs_error` 作为底层实现，改由 `CollectionWorkflow` 在满足条件时自动调用；单测同步调整为“主链路可达 + 函数行为正确”。
