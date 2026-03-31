# Enhanced Parity Plots 细节精调总结 (第三版)

根据您的最新反馈，本版调整将专注于解决标题重叠与加粗，以及微调散点模式下的间距：

## 1. 解决散点模式间距过小导致的标签重叠
**问题**：由于之前极度压缩了 `aligned_sidebar_gap` 至 0.008，导致右侧 Error Density 标签过于贴近左侧 True Density，造成了局部视觉重叠。
**方案**：
- 在 `src/dpeva/utils/visual_style.py` 中，将全局 `enhanced` 字典里的 `aligned_sidebar_gap` 从 `0.008` 略微放宽回调至 `0.020`。
- 这一微调能在保证 2x2 布局紧凑性的同时，给垂直的标签（如 True Density 和 Error Density）留出安全的呼吸空间。

## 2. 大标题 (Suptitle) 垂直位置上移
**问题**：目前大标题的纵向位置距离子图太近，甚至与刻度标签产生了重叠。同时由于缺乏与其他分布图一致的粗体样式，导致学术阅读体验不佳。
**方案**：
- **位置上调**：在 `src/dpeva/utils/visual_style.py` 中，全面上调 `suptitle_y` 参数：
  - `enhanced` 基础配置中：将 `suptitle_y` 从 `0.962` 提高至 `0.985`。
  - `energy` override 中：将 `suptitle_y` 从 `0.958` 提高至 `0.985`。
  - `cohesive_energy` override 中：将 `suptitle_y` 从 `0.955` 提高至 `0.985`。
- **字体加粗**：在 `src/dpeva/inference/visualizer.py` 的 `fig.suptitle` 调用中，新增 `fontweight="bold"` 参数，使得无论是否传入自定义 title，均能保持加粗的醒目样式。

## 3. 验证步骤
1. 运行 `export DPEVA_INTERNAL_BACKEND=local && dpeva analysis config_analysis_val.json` 生成最新图表。
2. 检查 `parity_cohesive_energy_enhanced.png`：
   - 确认左右面板间距适中，Error Density 标题不与 True Density 面板重叠。
   - 确认顶部大标题 "Cohesive Energy Enhanced Parity" 已经明显上移，与顶部图表脱开足够距离（方便论文裁剪），且文字变为了加粗体。
3. 检查 `parity_force_enhanced.png`：
   - 确认大标题 "Force Enhanced Parity" 同样上移并加粗。