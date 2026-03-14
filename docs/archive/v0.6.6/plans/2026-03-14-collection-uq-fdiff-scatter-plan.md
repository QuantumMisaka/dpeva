# Collection Workflow 缺失 UQ-QbC/RND-ForceDiff 二维散点图修复计划

## Summary
- 目标：修复 `CollectionWorkflow` 在有 Ground Truth 的真实测试数据场景下，未输出 `UQ-force-qbc-rnd-fdiff-scatter.png`（横纵轴为两维 UQ，颜色为 Force Diff）的问题，并补齐单元测试覆盖。
- 成功标准：运行 `collect` 后在 `view/` 目录稳定生成该图；无 Ground Truth 或 `diff_maxf_0_frame` 无效时保持不作图；新增/更新测试覆盖调用门控与输出行为。

## 参考图像对齐要求（view_ref 基准）
- 参考图：`test/dpeva-iter1/view_ref/UQ-force-qbc-rnd-fdiff-scatter.png`
- 图像语义：
  - 横轴：`UQ-QbC Value`
  - 纵轴：`UQ-RND-rescaled Value`
  - 颜色：`diff_maxf_0_frame`（Force Diff，红色系深浅表示大小）
- 视觉要素（作为验收标准）：
  - 标题为 `UQ-QbC and UQ-RND vs Max Force Diff`
  - 图例标题为 `Max Force Diff`
  - 保留 scheme 边界线（黑色虚线框与紫色边界线）
  - 开启网格，且坐标下界从 0 开始（不做 [0,2] 截断子图）

## Current State Analysis
- `UQVisualizer` 中已存在目标作图函数 `plot_uq_fdiff_scatter`，输出文件名常量为 `FILENAME_UQ_FORCE_QBC_RND_FDIFF_SCATTER`：
  - `src/dpeva/uncertain/visualization.py`
  - `src/dpeva/constants.py`
- `CollectionWorkflow._run_filtered_uq_phase` 当前调用了：
  - `plot_uq_identity_scatter`
  - `plot_uq_vs_error`（含 rescaled 分支）
  - `plot_uq_diff_parity`
- 但当前流程未调用 `plot_uq_fdiff_scatter`，因此即使 `has_gt=True` 且 `diff_maxf_0_frame` 有效，也不会输出该文件。
- 现有测试现状：
  - `tests/unit/workflows/test_collect_refactor.py` 已覆盖 force-error 图的 GT/有效性门控，但未覆盖 `plot_uq_fdiff_scatter` 调用。
  - `tests/unit/uncertain/test_visualization.py` 已有 `plot_uq_fdiff_scatter` 函数级测试，但仅断言 `savefig` 被调用，未校验输出文件名。
  - `tests/integration/test_slurm_multidatapool_e2e.py` 的 `view` 断言列表未包含该图。

## Proposed Changes

### 1) 修复 Workflow 调用链（核心修复）
- 文件：`src/dpeva/workflows/collect.py`
- 修改点：
  - 在 `if self._should_plot_force_error(has_gt, uq_results):` 分支中，补充调用 `vis.plot_uq_fdiff_scatter(...)`。
  - 使用与 identity-scatter 一致的参数来源：`df_uq`、`uq_select_scheme`、QbC/RND trust bounds。
- 设计理由：
  - 保持 GT 相关图统一受 `_should_plot_force_error` 门控。
  - 避免在无真值/无效 diff 时误生成颜色语义错误的图。

### 2) 校验并统一门控语义（最小且稳健）
- 文件：`src/dpeva/workflows/collect.py`
- 修改点：
  - 复用现有 `_should_plot_force_error` 作为唯一入口门控，不新增并行门控函数。
  - 保持 `plot_uq_fdiff_scatter` 内部 `diff_maxf_0_frame` 列检查作为防御式校验（双保险）。
- 设计理由：
  - 符合“一个明显方式（One Obvious Way）”：流程入口只在一个位置做业务门控。
  - 保持现有实现风格与错误显式处理策略。

### 2.1) 参考图一致性校验与最小样式修正
- 文件：`src/dpeva/uncertain/visualization.py`
- 修改策略：
  - 先保持现有 `plot_uq_fdiff_scatter` 主体实现不重构（当前已具备 `hue=diff_maxf_0_frame`、`palette="Reds"`、`title/xlabel/ylabel`、`_setup_2d_plot_axes`、`_draw_boundary`）。
  - 仅在与参考图不一致时做最小修正，优先顺序：
    1. 文案一致性（标题/轴标签/图例标题）。
    2. 边界绘制顺序与可见性（确保散点与边界均清晰）。
    3. 点样式（`alpha`、`s`）仅在影响可读性时微调。
- 设计理由：
  - 避免为“仅缺调用”问题引入不必要视觉回归。
  - 将变更面压缩在可验证的最小范围内，符合项目稳态维护策略。

### 3) 补齐单元测试（workflow 门控 + visualizer 输出）
- 文件：`tests/unit/workflows/test_collect_refactor.py`
- 新增测试：
  - `has_gt=True 且 diff 有效` 时，断言 `vis.plot_uq_fdiff_scatter` 被调用 1 次。
  - `has_gt=False` 或 `diff 非有限` 时，断言不调用 `vis.plot_uq_fdiff_scatter`。
- 文件：`tests/unit/uncertain/test_visualization.py`
- 增强测试：
  - 对 `plot_uq_fdiff_scatter` 增加对 `savefig` 目标文件名的断言，确保输出为 `UQ-force-qbc-rnd-fdiff-scatter.png`。
  - 增加对 `sns.scatterplot` 关键参数断言：`x/y/hue/palette` 与参考图语义一致。
  - 增加对标题、坐标轴标签、图例标题调用的断言，锁定参考图的核心视觉契约。
- 设计理由：
  - 让“函数存在但调用链断开”这类回归被单元测试直接拦截。
  - 保持测试粒度：workflow 关注调用/门控，visualizer 关注输出契约。

### 4) 可选集成测试增强（若当前测试环境稳定）
- 文件：`tests/integration/test_slurm_multidatapool_e2e.py`
- 变更策略：
  - 在 Ground Truth 可用的集成场景下，将 `UQ-force-qbc-rnd-fdiff-scatter.png` 加入 `expected_plots`。
  - 若场景存在数据条件波动，则维持单元测试为主，集成仅做非阻塞增强。

## Assumptions & Decisions
- 假设 A：用户给出的测试数据为有标签数据；已能生成 `UQ-force-rescaled-fdiff-parity.png`，说明 GT 门控已通过且 diff 字段有效。
- 决策 1：缺失根因以“workflow 未调用 `plot_uq_fdiff_scatter`”为主因处理，不引入配置开关，遵循现有最小改动原则。
- 决策 2：以 `view_ref` 参考图为 fdiff-scatter 验收基准，优先保证语义与可读性一致，不追求像素级拟合。
- 决策 3：优先补齐单元测试作为回归防线，集成测试按稳定性选择性增强。

## Verification
- 单元测试：
  - 运行 `pytest tests/unit/workflows/test_collect_refactor.py`
  - 运行 `pytest tests/unit/uncertain/test_visualization.py`
- 回归检查：
  - 运行 `pytest tests/unit`
  - 检查新增断言是否覆盖：
    - 调用链：`plot_uq_fdiff_scatter` 在应调用场景被调用
    - 门控：无 GT/无效 diff 场景不调用
    - 输出文件名：`UQ-force-qbc-rnd-fdiff-scatter.png`
- 结果验收：
  - 在用户测试目录（`dpeva-iter1`）复现实验后，`view/` 下出现目标散点图；原有图不回退。
