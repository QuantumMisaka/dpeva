# Analysis Workflow 作图增强实施计划

## 摘要

本计划用于增强 DP-EVA Analysis Workflow 的可视化表达与统计可解释性，覆盖 `model_test` 与 `dataset` 两种模式。目标包括：新增 Pred/True 叠加分布图、将 Error 分布与主分布合图、补充 virial parity、增加带边缘分布的增强 parity、统一配色与标注规范，并将统计信息（完整 describe）尽可能体现在图中。  
关键问题修复方面，将以 `data_path` 驱动真实组分加载，避免 Legacy 文件名解析导致 Cohesive 统计不可用。

## 当前状态分析

### 1) 现有能力

- 绘图入口集中在 [visualizer.py](file:///home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/dpeva/src/dpeva/inference/visualizer.py)：
  - 已有单图 `plot_distribution`、`plot_error_distribution`、`plot_parity`
  - 已有 NaN/Inf 防护与 WARNING 日志
- 业务编排在 [analysis/managers.py](file:///home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/dpeva/src/dpeva/analysis/managers.py#L194-L300)：
  - model_test 输出 energy/force/cohesive/virial 的分布与部分 parity/error
  - 目前无 Pred/True 叠加分布图、无增强 parity、无 virial parity
- dataset 分析在 [analysis/dataset.py](file:///home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/dpeva/src/dpeva/analysis/dataset.py)：
  - 已有 energy/force/virial/pressure 分布与 `dataset_stats.json`
  - 已有元素类别与占比统计（JSON），但缺少对应图
  - 尚未做 cohesive energy 图

### 2) 已定位问题来源（Cohesive L20）

- 日志告警位于 [analysis.log:L20](file:///home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter8-longtrain/s-head-huber/test_analysis/analysis.log#L20)：`non-positive atom count found`
- 根因：Legacy 模式从 `results.e_peratom.out` 注释头解析 system 名（如 `sampled_dpdata/11`），在 [dataproc.py:get_composition_list](file:///home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/dpeva/src/dpeva/io/dataproc.py#L137-L172) 做元素正则失败，得到 0 原子。
- 已验证解决方向：配置 `data_path` 后可走真实组分读取，Cohesive 恢复（同日志后续运行已验证）。

## 方案决策（已确认）

- 分布+误差合图：采用**2列主次布局**（左主分布，右窄栏 error 分布）
- Parity 增强输出策略：**保留原图 + 新增增强图**
- Dataset cohesive 参考能策略：**优先 ref + 可选 lstsq 补全**
- 统计标注密度：**完整 describe 上图**

## 拟议改动（文件级）

### A. 可视化能力扩展

#### A1. 扩展 InferenceVisualizer 统一图元接口
- 文件：`src/dpeva/inference/visualizer.py`
- 改动：
  - 新增叠加分布接口（Pred/True 双分布同图）
  - 新增“分布+误差”组合图接口（2列主次布局）
  - 新增增强 parity 接口（中心 scatter parity + 两轴边缘分布）
  - 新增图内统计注释接口（支持 describe 全量键）
  - 新增统一配色与 label 规范映射（pred/true/error/cohesive/virial）
- 为什么：
  - 避免在 manager 层重复拼图逻辑
  - 将美学、版式、鲁棒性集中治理
- 如何：
  - 复用当前 finite 过滤
  - 使用 `matplotlib.gridspec` 构建主次布局与增强 parity 布局
  - 统一文件命名：`*_overlay.png`、`*_with_error.png`、`*_enhanced.png`

### B. model_test 图谱增强

#### B1. 在 UnifiedAnalysisManager 中新增绘图编排
- 文件：`src/dpeva/analysis/managers.py`
- 改动：
  - 新增 `virial` parity 基础图
  - 保留原 parity 并新增增强 parity（energy/force/virial/cohesive）
  - 对 energy/force/virial/cohesive 输出：
    - 原有单图分布（pred、true）
    - 新增叠加分布图（pred vs true）
    - 新增分布+误差合图（右侧小 error）
  - 将统计 describe（count/mean/std/min/25/50/75/max）写入图内注释
- 为什么：
  - 满足“对比性与科学性”要求，并显式承接 error 信息
- 如何：
  - 以现有 `safe_plot` 为外层保护
  - 对每个物理量使用统一 helper 生成完整图谱族

### C. dataset 模式增强

#### C1. 在 DatasetAnalysisManager 中增加 cohesive 与元素分析图
- 文件：`src/dpeva/analysis/dataset.py`
- 改动：
  - 基于系统组分 + `ref_energies`（及配置）计算 dataset cohesive per-atom
  - 输出 cohesive 单图、叠加图（若可对比项存在）与组合图
  - 对 `dataset_stats.json` 的元素统计新增图：
    - 元素占比条形图（`element_ratio_by_atom`）
    - 元素系统覆盖率条形图（`system_element_presence`）
  - 在能量/力/virial/pressure/cohesive图中加入完整 describe 标注
- 为什么：
  - 让 JSON 统计可视化，不再“只存不看”
- 如何：
  - 复用 visualizer 新接口，新增 dataset 专用 title/label
  - 若 cohesive 不可计算则 warning + 跳过，不中断

### D. 配置与参数透传

#### D1. 配置对象补齐 dataset cohesive 所需参数透传
- 文件：`src/dpeva/config.py`, `src/dpeva/workflows/analysis.py`
- 改动：
  - 复核并保持以下参数透传到 dataset 分析路径：
    - `enable_cohesive_energy`
    - `allow_ref_energy_lstsq_completion`
    - `ref_energies`
  - 若 dataset 模式缺失必要条件（如组分/参考能）给出清晰 warning
- 为什么：
  - 保持 model_test 与 dataset 的 cohesive 行为一致

### E. 示例配置与用户指南

#### E1. 示例配置
- 文件：
  - `examples/recipes/analysis/config_analysis.json`
  - `test/s-head-huber/config_analysis.json`
  - `test/s-head-huber/config_analysis_dataset.json`
- 改动：
  - 明确 `data_path` 的必要性说明场景
  - 明确 `enable_cohesive_energy` 与 `allow_ref_energy_lstsq_completion` 推荐值

#### E2. 用户指南
- 文件：
  - `docs/source/guides/configuration.md`
  - `docs/source/guides/cli.md`
- 改动：
  - 增加“图谱输出清单”与“新文件命名规则”
  - 增加“Cohesive 失败诊断”与“data_path 绑定策略”
  - 说明增强 parity 与组合分布图的解读方式

## 输出命名约定（新增）

- 分布叠加图：`dist_<quantity>_overlay.png`
- 分布+误差组合图：`dist_<quantity>_with_error.png`
- 增强 parity：`parity_<quantity>_enhanced.png`
- 元素占比图：`dataset_element_ratio.png`
- 元素覆盖图：`dataset_element_presence.png`

## 假设与约束

- 假设 `dpdata` 可正常读取 `dataset_dir/data_path`（当前样例可读）
- 约束：保持现有文件兼容，**不删除**旧图；只新增增强图
- 约束：图像生成失败不得中断核心统计导出（沿用 safe_plot）

## 验证计划

### 1) 单元测试

- 新增/更新：
  - `tests/unit/inference/test_visualizer.py`
  - `tests/unit/analysis/test_dataset_manager.py`
  - `tests/unit/workflows/test_analysis_workflow.py`
- 覆盖点：
  - overlay/combo/enhanced 图是否生成
  - virial parity 是否生成
  - 统计注释数据是否可用
  - cohesive 在 dataset/model_test 下的可计算与降级路径

### 2) 集成验证（指定目录）

- 执行：
  - `dpeva analysis test/s-head-huber/config_analysis.json`
  - `dpeva analysis test/s-head-huber/config_analysis_dataset.json`
- 核验：
  - `test_analysis/` 中新增 overlay/combo/enhanced 图
  - `analysis_dataset/` 中新增 cohesive 与元素统计图
  - 日志中无崩溃，必要 warning 可读可定位

### 3) 文档治理验证

- `python3 scripts/doc_check.py`
- 如需，补跑 `make -C docs html SPHINXOPTS="-W --keep-going"`

---

## 增量计划（本轮 /plan：作图二次精修）

### 一、目标与验收

- 目标：解决你指出的图例/统计框与主峰重叠、标签冗余、元素比例图类型不匹配、统计信息过载等问题，进一步提升科学性与可读性。
- 验收标准：
  - `analysis_dataset/dist_dataset_force_magnitude.png` 图中不出现与峰位重叠的图例/统计框，且无 `All Data` 冗余标签。
  - `dataset_element_ratio.png` 与 `dataset_element_presence.png` 改为多色饼图（非同色条形图）。
  - 所有统计框仅包含 `count/mean/std/min/max`。
  - 所有 Error Distribution 图不显示统计框。
  - 所有 enhanced parity 图不显示统计框。
  - `dist_force_magnitude_overlay.png`、`dist_force_magnitude_with_error.png` 信息区与主峰不重叠。

### 二、现状核查（已完成）

- `src/dpeva/inference/visualizer.py`
  - `_stats_text` 当前仍输出 `25%/50%/75%`。
  - `plot_distribution` 固定 `label="All Data"` 且默认展示 legend。
  - `plot_distribution_overlay` / `plot_distribution_with_error` 统计框固定在左上角，易压住 force 主峰。
  - `plot_error_distribution` 仍有统计框。
  - `plot_parity_enhanced` 仍有统计框。
- `src/dpeva/analysis/dataset.py`
  - `_plot_element_statistics` 仍使用条形图。
- `src/dpeva/utils/visual_style.py`
  - 项目已全局启用 seaborn 主题，可直接用于本轮美化（满足你的第3条约束）。
- 文档双路径并存：
  - `docs/guides/*.md` 与 `docs/source/guides/*.md` 同时存在，需同步更新避免漂移。

### 三、实施改动（文件级、决策完备）

#### A. 通用绘图策略精修（核心）

**文件**：`src/dpeva/inference/visualizer.py`

1) 统计内容收敛
- 修改 `_stats_text`：仅输出 `count, mean, std, min, max`。
- 移除分位数输出（25/50/75）。

2) 移除冗余 `All Data`
- 调整 `plot_distribution`：
  - 单序列分布默认 `show_legend=False`；
  - 不再写入 `label="All Data"`（只有对比图才保留 legend）。
- 结果：dataset 与单变量结果图不再出现 `All Data`。

3) 解决信息遮挡（force 为重点）
- 为 `plot_distribution_overlay` / `plot_distribution_with_error` 增加可控参数：
  - `stats_loc`、`legend_loc`、`legend_outside`（或等价实现）。
- 默认策略：
  - legend 外置（右上外侧）；
  - stats 置于左下或右下，避开近零峰。
- 对 force 系列调用时强制使用“峰值避让布局”。

4) Error Distribution 去统计框
- 删除 `plot_error_distribution` 中统计框渲染。
- 删除 `plot_distribution_with_error` 右侧 error 子图统计框渲染。

5) Enhanced Parity 去统计框
- 删除 `plot_parity_enhanced` 中 stats box。
- 保留中心 parity + 边缘分布与必要坐标标签。

6) 主动可视化优化（不增负担）
- 统一对比色盘（色盲友好高对比）与线宽、透明度。
- 对长尾变量（force）：
  - 使用更稳健的 hist/kde 参数；
  - 必要时限制 KDE 过冲，保证主峰可辨。

#### B. Dataset 元素比例图改饼图

**文件**：`src/dpeva/analysis/dataset.py`

1) `dataset_element_ratio.png`
- 条形图改为饼图；
- 使用 `sns.color_palette(...)` 生成多色扇区；
- 扇区标签展示元素与比例（必要时 legend 辅助）。

2) `dataset_element_presence.png`
- 条形图改为饼图；
- 标签展示元素、比例、系统计数（例如 `C: 74.6% (983)`）。

3) 饼图可读性
- 扇区过小的标签可转移到 legend，防止重叠。
- 统一起始角与边框，提高对比度。

#### C. 编排层最小改动

**文件**：`src/dpeva/analysis/managers.py`

- 在 force/energy/virial/cohesive 调用 overlay/with_error 时，传入推荐布局参数（避免遮挡）。
- 不变更现有文件命名约定，确保下游兼容。

#### D. 文档同步（按你的当前改动基线）

**文件**：
- `docs/guides/configuration.md`
- `docs/guides/cli.md`
- `docs/source/guides/configuration.md`
- `docs/source/guides/cli.md`

同步更新要点：
- 统计框内容简化（无分位数）；
- Error 图与 enhanced parity 无统计框；
- 单分布图无 `All Data`；
- dataset 元素图采用饼图与多色。

### 四、假设与约束

- 不新增依赖，仅使用现有 matplotlib + seaborn。
- 不改变核心统计口径与输出 JSON 语义，仅优化图形表达与布局。
- 保持产物主文件名稳定（在现有命名上调整图内容与样式）。

### 五、验证步骤

1) 单测
- `tests/unit/inference/test_visualizer.py`
  - 校验统计文本字段变化；
  - 校验 error/enhanced parity 不再调用统计框；
  - 校验图例开关与布局参数生效。
- `tests/unit/analysis/test_dataset_manager.py`
  - 校验元素图生成路径不变，图类型改动后仍可落盘。

2) 集成复跑（你的目标目录）
- `dpeva analysis test/s-head-huber/config_analysis.json`
- `dpeva analysis test/s-head-huber/config_analysis_dataset.json`
- 重点人工核查：
  - `analysis_dataset/dist_dataset_force_magnitude.png`
  - `analysis_dataset/dataset_element_ratio.png`
  - `analysis_dataset/dataset_element_presence.png`
  - `test_analysis/dist_force_magnitude_overlay.png`
  - `test_analysis/dist_force_magnitude_with_error.png`
  - `test_analysis/parity_*_enhanced.png`

3) 文档一致性核查
- 校验 `docs/guides` 与 `docs/source/guides` 同步一致。
