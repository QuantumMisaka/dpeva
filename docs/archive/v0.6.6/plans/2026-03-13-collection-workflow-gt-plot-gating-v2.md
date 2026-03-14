---
status: active
audience: developers
last-updated: 2026-03-13
owner: DP-EVA Developers
---

# CollectionWorkflow 有无真值作图门控改进计划（V2）

## 问题重述

当前在 data pool 无参考真值场景，出现了两类并存风险：

- 误绘（false positive）：依赖 force 真值误差的图被错误绘制，例如 `UQ-diff-fdiff-parity.png`、`UQ-diff-UQ-parity.png`、`UQ-force-rescaled-fdiff-parity.png`。
- 漏绘/错屏蔽（false negative）：不依赖真值误差的图可能被错误屏蔽，用户重点反馈 `UQ-QbC-force.png` 正常，但怀疑 `UQ-RND-force.png` 被错误屏蔽。

本次将以“全量图依赖矩阵 + 统一门控 + 双向测试（误绘/漏绘）”方式修复，确保行为与项目既有设计一致。

## 目标

1. 完整梳理 CollectionWorkflow 里“真值判定、UQ分析、筛选、作图”的分支与触发点。
2. 明确每张图是否依赖真值误差，形成可执行的输出矩阵。
3. 修复无真值场景下的真值图误绘问题。
4. 修复或澄清非真值图错屏蔽问题（含 `UQ-RND-force` 专项核查）。
5. 通过单测和流程级回归同时覆盖误绘与漏绘。

## 实施步骤

### 步骤 1：深挖真值判定与传播链

- 梳理 `cli.handle_collect -> CollectionWorkflow.run -> _run_filtered_uq_phase`。
- 梳理 `UQManager.load_predictions` 的 `has_gt` 来源与传播。
- 核查 `DPTestResultParser._check_ground_truth` 的判定鲁棒性（全零启发式误判风险）。
- 核查多模型情形下是否仅取 model0 `has_gt`，避免单模型状态污染全局。

### 步骤 2：建立全量“图输出依赖矩阵”

- 盘点 `CollectionWorkflow` 当前实际调用的所有绘图函数与文件名。
- 盘点 `UQVisualizer` 中存在但 workflow 可能未调用的绘图函数。
- 将图分为三类：A类（不依赖真值误差）、B类（依赖真值误差）、C类（混合/需澄清）。
- 形成矩阵字段：`图名 | 函数 | 依赖字段 | 无真值期望 | 有真值期望 | 当前行为`。
- 对 `UQ-RND-force.png` 做专项结论：应输出则补齐调用；不应输出则明确设计边界并固化测试预期。

### 步骤 3：Workflow 层统一门控与调用编排

- 在 `_run_filtered_uq_phase` 建立集中化作图编排：B类统一受 `has_gt` 门控，A类显式放在非门控路径。
- 对 `UQ-RND-force` 按步骤2结论执行：补齐对称调用或明确不输出策略。
- 保持目录结构、命名、日志风格与现有实现一致。

### 步骤 4：Visualizer 层防御式校验（最小侵入）

- 在 B类函数中增加最小必要输入校验，误调用时显式跳过并 warning。
- A类函数避免引入“无真值即跳过”的过度校验，仅做必要数据完整性校验。
- 保持函数签名兼容，不引入新配置项。

### 步骤 5：测试策略（双向覆盖）

- 无真值：断言 B类图不生成；A类图按期望生成（含 `UQ-RND-force` 结论路径）。
- 有真值：断言 A/B类图都按期望生成。
- 防御式校验：误调用 B类函数不产出伪图，且有日志信号。
- 优先复用 `tests/unit/workflows` 与 `tests/unit/uncertain` 现有测试基建。

### 步骤 6：回归与验收

- 先跑定向单测，再跑 `pytest tests/unit` 做回归兜底（资源允许时）。
- 对比产物清单与依赖矩阵，验证“误绘消失 + 漏绘修复/澄清”。
- 核对日志语义：无真值应明确说明跳过真值依赖图。

## 验收标准

1. 无真值时，依赖真值误差图不再误绘（包括用户指出的三张图）。
2. 无真值时，不依赖真值误差图不被误屏蔽。
3. `UQ-RND-force` 行为与项目既有设计一致（补齐或明确不输出），并有测试约束。
4. 有真值时，相关图输出不回归，整体单测通过。
