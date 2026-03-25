# 计划：确认 Stage8 两张 Candidate-Parity 图是否可被 CollectionWorkflow 原生生成

## Summary
- 目标：确认以下两张图在**当前** `CollectionWorkflow` 中是否可原生生成，并确认项目内是否存在对应作图代码模块。  
  - `UQ-QbC-Candidate-fdiff-parity.png`
  - `UQ-RND-Candidate-fdiff-parity.png`
- 交付：给出“是否原生生成”的结论 + 代码证据（定义、调用、流程入口）+ 可复核路径。

## Current State Analysis
- 图名常量在当前代码中仍存在定义：  
  - `src/dpeva/constants.py` 中 `FILENAME_UQ_QBC_CANDIDATE_FDIFF_PARITY`、`FILENAME_UQ_RND_CANDIDATE_FDIFF_PARITY`。
- 对应作图函数在当前代码中存在：  
  - `src/dpeva/uncertain/visualization.py` 的 `plot_candidate_vs_error(self, df_uq, df_candidate)`，内部会保存上述两张图。
- `CollectionWorkflow` 当前主流程调用链（`src/dpeva/workflows/collect.py`）未调用 `plot_candidate_vs_error`：  
  - 仅调用 `plot_uq_identity_scatter`、`plot_uq_fdiff_scatter`、`plot_uq_vs_error`、`plot_uq_diff_parity` 等。
- 额外发现：`plot_candidate_vs_error` 在测试中有调用（`tests/unit/uncertain/test_visualization.py`），说明模块可用但未接入标准工作流。

## Assumptions & Decisions
- 假设“原生生成”定义为：执行标准 `dpeva collect <config>` 主流程即可由 `CollectionWorkflow` 自动产出，不依赖人工额外脚本或手工调用函数。
- 决策：本次仅做“结论确认 + 证据归档式说明”，不修改任何业务代码与文档。
- 范围外：不做功能接入改造（不把 `plot_candidate_vs_error` 接入 `collect.py`）。

## Proposed Changes (Execution After Approval)
- 不进行代码改动。
- 执行步骤为只读验证并输出结果说明：
  1. 复核两个文件名常量定义位置。
  2. 复核 `plot_candidate_vs_error` 函数是否存在及其 `savefig` 目标文件名。
  3. 复核 `CollectionWorkflow.run/_run_filtered_uq_phase` 的可视化调用列表，确认无该函数调用。
  4. 复核单元测试中是否存在独立调用，作为“模块存在但未接入主流程”的补充证据。
  5. 输出最终判定：
     - 在当前 CollectionWorkflow 中：**不能原生生成**；
     - 项目中：**存在该作图模块代码**（可被手动/测试调用）。

## Verification Steps
- 证据核对文件（只读）：
  - `src/dpeva/constants.py`
  - `src/dpeva/uncertain/visualization.py`
  - `src/dpeva/workflows/collect.py`
  - `tests/unit/uncertain/test_visualization.py`
- 一致性检查：
  - 常量名 ↔ 函数 `savefig` 文件名一致；
  - 函数定义存在但在 `collect.py` 调用链缺失；
  - 测试调用存在（证明模块可执行）。

## Expected Final Output Format
- 简明结论（每张图一行）：`原生生成：否` / `模块存在：是`。
- 附 3 组证据：
  - 定义证据（常量+函数）
  - 调用证据（Workflow 调用链）
  - 旁证（单测调用）
