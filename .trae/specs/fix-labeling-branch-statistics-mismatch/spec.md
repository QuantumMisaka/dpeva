# Labeling 分支统计错配修复 Spec

## Why
当前 Labeling 后处理日志中，全局总帧数正确，但按 Dataset/Type 的分支统计与采样输入不一致，影响数据质量判断与后续回溯。需要修复统计映射根因，并提供对历史已完成工作目录的批量修复能力。

## What Changes
- 修复 Labeling 统计阶段对 Dataset/Type 的来源策略，避免 metadata 缺失时从打包目录结构错误回推。
- 在结果提取阶段增加稳定的分支身份信息保留机制，使后续统计不依赖脆弱路径推断。
- 增加统计一致性校验：分支求和必须与全局统计一致，且可与采样输入统计对齐核验。
- 新增离线修复脚本，用于对历史已完成 Labeling 工作目录重建并重导出统计报告。
- 补齐单元测试与回归测试，覆盖 metadata 缺失、损坏、跨 stage 运行等场景。

## Impact
- Affected specs: Labeling workflow staged execution, Labeling statistics reporting, Postprocess observability
- Affected code: `src/dpeva/labeling/manager.py`, `src/dpeva/workflows/labeling.py`, `src/dpeva/cli.py`, `tests/unit/labeling/*`, `scripts/*`, `docs/guides/*`

## ADDED Requirements
### Requirement: 历史工作目录统计修复工具
系统 SHALL 提供离线修复能力，对已完成的 Labeling 工作目录生成可信的 Dataset/Type 统计结果，而不触发重新计算任务。

#### Scenario: 对单个历史目录执行修复
- **WHEN** 用户提供一个已完成的 `labeling_workdir` 路径执行修复命令
- **THEN** 系统基于可用元数据与目录信息重建分支统计并输出修复报告
- **THEN** 报告中的全局统计与分支统计求和一致

#### Scenario: 批量修复多个历史目录
- **WHEN** 用户提供多个工作目录或根目录批量执行修复
- **THEN** 系统逐目录输出结果与失败原因摘要
- **THEN** 单个目录失败不会中断其他目录处理

## MODIFIED Requirements
### Requirement: Labeling 统计分支归属准确性
系统 SHALL 在 Labeling postprocess 统计中准确归属每个任务到其真实 Dataset/Type，并保证日志中的 Dataset/Type 统计可追溯、可复核。

#### Scenario: 正常 metadata 可读取
- **WHEN** 任务目录包含有效 `task_meta.json`
- **THEN** 统计必须优先使用 metadata 中的 `dataset_name` 和 `stru_type`

#### Scenario: metadata 缺失或损坏
- **WHEN** `task_meta.json` 缺失、损坏或字段不完整
- **THEN** 系统不得使用打包目录层级（如 `N_50_x`）作为 Dataset/Type
- **THEN** 系统使用稳定回退机制（如提取阶段保留的身份映射）并记录告警
- **THEN** 若仍无法识别，任务归类到受控 `unknown` 分支且可被审计

#### Scenario: 统计一致性校验
- **WHEN** 统计报告生成完成
- **THEN** 系统校验 `sum(Dataset/Type)` 与 Global 各指标完全一致
- **THEN** 校验失败时输出显式错误信息并标记报告不可信

## REMOVED Requirements
### Requirement: 基于打包目录结构推断分支归属
**Reason**: 打包目录（`N_50_x/task_*`）不包含真实 Dataset/Type 语义，会导致分支统计错配。  
**Migration**: 迁移到“metadata 优先 + 稳定身份映射回退 + unknown 受控兜底”的统一策略。
