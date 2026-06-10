# Collection Workflow 目标池无 Label 判据调整计划

## Summary
- 目标：将 Collection workflow 中“目标池是否有 label”的判据从“全部为 0 才无 label”调整为“只要存在任意一帧 label 为 0（按 `<1e-4` 判定）即视为整个目标池无 label”。
- 行为要求：保持 `label 为 0` 的阈值不变（`<1e-4`），并在判定无 label 后触发现有无 label 的作图与数据分析分支（沿用已有控制变量，不新增分支开关）。
- 可观测性：一旦检测到该无 label 条件，在 Collection Workflow 日志输出 `WARNING` 提醒用户。

## Current State Analysis
- 当前 ground truth 判据入口在 `src/dpeva/io/dataproc.py` 的 `DPTestResultParser._check_ground_truth()`：
  - 使用 `zero_tol = 1e-4`；
  - 现逻辑是能量列和力列“全体近似零”才认为无 ground truth；
  - 或 data 列与 pred 列整体近似相同也认为无 ground truth。
- Collection 主流程通过 `UQManager.load_predictions()` 聚合 `has_ground_truth = all(pred.has_ground_truth for pred in preds)`，并据此控制是否注入 `diff_maxf_0_frame` 与是否执行误差相关作图。
- Collection 的无 GT 分支已经存在：
  - 在 `_run_filtered_uq_phase()` 中 `has_gt=False` 时不写 `diff_maxf_0_frame`；
  - `_should_plot_force_error()` 返回 `False` 时跳过 force-error 相关绘图。
- 现有测试基础：
  - `tests/unit/io/test_dataproc.py` 已覆盖 “all-zero / data==pred” 判据；
  - `tests/unit/workflows/test_collect_refactor.py` 已覆盖 Collection 在 `has_gt=False` 下的无 GT 作图分支。

## Proposed Changes

### 1) 更新 ground truth 判据实现（核心）
- 文件：`src/dpeva/io/dataproc.py`
- 修改点：`DPTestResultParser._check_ground_truth()`
- 改法：
  - 保留 `zero_tol = 1e-4` 不变；
  - 将“全体近似零”逻辑扩展为“存在任意一帧近似零即触发无 GT”；
  - 保留“data 与 pred 近似相同 => 无 GT”逻辑作为补充判据；
  - 新增/调整日志文案，明确是“检测到至少一帧 label 近似 0，因此按无 GT 处理”。
- 实现细节（决策）：
  - 能量按帧天然一一对应，基于 `data_e` 检测“是否存在近似 0 帧”；
  - 力数据按原子展开，基于已解析帧边界（`dataname_list` / `datanames_nframe`）重建帧切片，并按帧判断三分量是否均近似 0；
  - 任一来源（能量帧或力帧）命中“近似 0 帧”即将 `has_ground_truth=False`。

### 2) 在 Collection Workflow 增加 WARNING 可观测性
- 文件：`src/dpeva/uncertain/manager.py`（`load_predictions`）和/或 `src/dpeva/workflows/collect.py`（UQ阶段入口）
- 修改点：
  - 当 `has_ground_truth=False`（由上述新判据触发）时，除了现有 info 日志，新增面向用户的 `WARNING` 日志；
  - WARNING 文案需明确：
    - 判据：检测到至少一帧 label 近似 0（阈值 `<1e-4`）；
    - 影响：将按无 label 处理，禁用/跳过依赖 ground truth 的作图与误差分析。
- 说明：
  - 不新增布尔变量，不改变分支结构，仅增强日志等级和可读性；
  - 继续复用现有 `has_gt` 控制作图/分析分支。

### 3) 补齐/更新单元测试
- 文件：`tests/unit/io/test_dataproc.py`
  - 新增用例：存在“部分帧 label 近似 0”时应判定 `has_ground_truth=False`；
  - 覆盖能量维度与（可行时）力维度的“任意帧触发”场景；
  - 断言阈值边界仍以 `<1e-4` 生效。
- 文件：`tests/unit/workflows/test_collect_refactor.py`（必要时）
  - 补充/强化用例：当上游 `has_gt=False` 时，Collection 仍按现有无 GT 分支执行（不画误差依赖图）。
- 文件：`tests/unit/uncertain/test_manager.py` 或对应 manager 测试文件（如存在）
  - 增加 `WARNING` 日志断言（`caplog` 或 mock logger）。

## Assumptions & Decisions
- 假设 1：用户需求仅针对 Collection 场景，但 ground truth 判据位于通用 parser；本次按“修改通用判据 + 通过 Collection 入口产生期望行为”执行。
- 假设 2：目标“触发无 label 作图分支和数据分析分支”指复用当前 `has_gt` 相关分支控制，不引入新配置项。
- 决策 1：`label 为 0` 判据严格保持 `<1e-4`，不改为 `<=`，不改阈值。
- 决策 2：WARNING 日志至少在 Collection 路径可见（`collection.log`），文案面向最终用户。

## Verification Steps
- 运行与本改动直接相关的单元测试：
  - `pytest tests/unit/io/test_dataproc.py`
  - `pytest tests/unit/workflows/test_collect_refactor.py`
  - `pytest tests/unit/uncertain/test_manager.py`（若实际新增在该文件）
- 目标验证：
  - 任意一帧 `label <1e-4` 时，`has_ground_truth=False`；
  - Collection 流程进入无 GT 分支（跳过 force-error 相关图/分析）；
  - `collection.log` 中出现 WARNING，且包含判据与分支影响说明。
