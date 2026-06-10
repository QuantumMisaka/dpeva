---
title: Analysis Module Repair and Enhancement Plan
status: archived
audience: developers
last-updated: 2026-03-16
owner: DP-EVA Maintainers
---

# 2026-03-16 Analysis 模块修复与功能增补计划

## 1. 目标

- 修复 Analysis 在 Cohesive Energy 分析链路中的稳定性缺陷，确保 model_test 模式可稳定完成。
- 增补 Cohesive Energy 的参考能量混合补全能力，满足“部分参考值 + 最小二乘补全”场景。
- 增补 Dataset 分析中的元素类别与占比统计，提升数据画像能力。
- 校正配置路径解析一致性与异常隔离能力，降低运行环境敏感性。
- 明确 `InferenceVisualizer` 在 Inference Workflow 的应用质量并补齐其鲁棒性短板。

## 2. 范围与非范围

### 2.1 范围内

- `src/dpeva/inference/visualizer.py`
- `src/dpeva/inference/stats.py`
- `src/dpeva/analysis/managers.py`
- `src/dpeva/analysis/dataset.py`
- `src/dpeva/utils/config.py`
- 对应单元测试与文档更新

### 2.2 范围外

- 不改动训练/推理主执行命令行为。
- 不引入新的第三方绘图库或重做图形风格系统。

## 3. 设计原则

- 错误显式处理：图表失败不得静默吞错，也不得中断核心统计导出。
- 保持向后兼容：现有配置默认行为不破坏。
- 用户可控：Cohesive Energy 混合补全必须由配置显式启用。
- 可观测：关键降级路径输出结构化日志。

## 4. 任务分解

## Task A：修复 `InferenceVisualizer` 鲁棒性并核对 Inference 应用质量

### A1. 输入有限值防护

- 在 parity 绘图前统一过滤 `y_true/y_pred` 的非有限值。
- 有效点数不足时跳过绘图并记录 warning（包含原始点数、过滤后点数）。

### A2. 轴限安全计算

- 若 `vmin/vmax` 非有限值，直接降级跳过该图。
- 若 `vmin == vmax`，使用最小窗宽扩展避免 Matplotlib 轴限异常。

### A3. Inference Workflow 适配性确认

- 复核 `InferenceWorkflow -> UnifiedAnalysisManager -> InferenceVisualizer` 调用链。
- 增补测试验证修复后不会破坏 Inference 的现有分析输出。

## Task B：修复 Cohesive Energy 计算逻辑并补全 Parity Plot

### B1. 引入“部分参考值锁定 + 剩余元素最小二乘补全”

- 新增配置开关建议：
  - `enable_cohesive_energy: bool`（默认按现有行为兼容）
  - `allow_ref_energy_lstsq_completion: bool`（默认 `false`）
- 计算策略：
  - 若用户提供元素参考能且覆盖完整：直接使用；
  - 若覆盖不完整且用户启用补全：固定已知元素，未知元素通过 `lstsq` 回归；
  - 若矩阵秩不足或条件恶劣：输出告警并降级仅做非 Cohesive 分析。

### B2. Cohesive Energy Parity 与误差图补全

- 当存在各元素参考值时，针对pred结果和true值，以相同元素参考值为基准，输出 parity 与 error distribution。
- 当不可用时写明原因并跳过对应图，不中断主流程。

### B3. 结果导出一致性

- 在 metrics 计算成功后优先持久化，图像失败不影响 `metrics.json/summary.csv`。

## Task C：增强 Dataset 统计（元素类别与占比）

### C1. 新增统计字段

- 在 `dataset_stats.json` 新增：
  - `element_categories`
  - `element_ratio_by_atom`
  - `element_count_by_atom`
  - `system_element_presence`

### C2. 统计口径

- 默认采用“全帧全原子加权”统计元素占比。
- 同步输出“出现该元素的系统数/系统占比”，用于补充结构覆盖视角。

### C3. 输出与兼容

- 仅新增字段，不删除现有统计字段，保持下游兼容。

## Task D：路径解析一致性与异常隔离

### D1. 配置路径解析修复

- 将 `dataset_dir` 纳入 `resolve_config_paths` 的路径键白名单。
- 增加相对路径解析单测，保证与 `result_dir/output_dir` 一致。

### D2. 异常隔离改造

- 分离“核心统计导出”与“可视化导出”执行边界。
- 图像失败记录在日志与导出摘要中，不阻断流程完成标记。

## Task E：测试与验证

### E1. 单元测试

- `visualizer.py`：NaN/Inf、空数组、常量数组、退化轴限。
- `stats.py`：部分参考值补全的矩阵可解/不可解场景。
- `dataset.py`：元素类别和占比统计准确性。
- `utils/config.py`：`dataset_dir` 相对路径解析。

### E2. 集成验证

- 使用以下配置回归：
  - `test/s-head-huber/config_analysis_dataset.json`
  - `test/s-head-huber/config_analysis.json`
- 验证点：
  - dataset 模式输出新增元素统计字段
  - model_test 模式可完成 Cohesive Energy 图表或按策略降级且流程不崩溃
  - 指标文件稳定落盘

## 5. 交付物

- 修复代码与单元测试。
- 更新后的审查报告与计划文档。
- 运行验证证据（日志片段、输出文件清单、关键统计字段截图或摘要）。

## 6. 验收标准

- `config_analysis.json` 不再因 `Axis limits cannot be NaN or Inf` 导致流程失败。
    - 若出现NaN/Inf的情况，需要有显式的WARNING信息在日志中输出。
- Cohesive Energy 在“完整参考值”和“部分参考值+补全开关开启”下均可执行。
- `dataset_stats.json` 出现元素类别与占比相关新字段，数据口径一致可复核。
- `InferenceVisualizer` 修复后在 Inference Workflow 分析链路中保持兼容并通过回归测试。
- 文档索引更新完成，无失效 toctree 引用。

## 7. 风险与回滚策略

- 风险：混合最小二乘补全在病态样本下不稳定。
  - 缓解：增加秩与条件数检查，必要时禁用 Cohesive 图并告警。
- 风险：异常隔离可能掩盖严重数据问题。
  - 缓解：保留错误计数与失败摘要，并在最终状态中显式提示“部分图失败”。
- 回滚：保留开关，若出现不可控副作用可暂时关闭补全逻辑与增强图表路径。
