---
status: active
audience: developers
---

# Parity Plots Readability Enhancement Plan

## 1. 总结 (Summary)
用户要求在之前统一定宽的基础上，进一步优化 Parity Plots（包括普通与增强型）的学术阅读体验。具体需求包括：将所有 parity plots 的基础字号整体增大2，略微增大 scatter 散点和 hexbin 六边形的尺寸，以及将 Scatter 模式下 Error Density 子图上方的数字（刻度标签）向上微调以防止与刻度线和曲线重叠。此外，还需要确保各处字体不被边缘化，图片重点突出，子图协调一致。

## 2. 当前状态分析 (Current State Analysis)
1. **基础字号**：在 `src/dpeva/utils/visual_style.py` 中，`get_analysis_parity_profile` 的默认 `font_size` 仍然为 `12`。
2. **统计框字号脱节**：在 `src/dpeva/inference/visualizer.py` 的 `plot_parity` 和 `plot_parity_enhanced` 中，调用 `self._add_stats_box` 时未传入 `fontsize` 参数，导致统计框一直使用硬编码的默认字号 `12`，未能与配置的 `fonts["label"]` 同步缩放。
3. **Scatter与Hexbin尺寸**：目前的 `scatter_size` 约在 `14.5`~`17.0` 之间；Hexbin 的 `main_density_gridsize` 为 `60`。
4. **数字重叠**：增强型散点图的 Error Density 顶部刻度标签的 `error_tick_pad` 目前设定为 `-3.0`。负值导致标签向图内收缩，从而与刻度线及核密度曲线发生重叠。

## 3. 拟议的修改 (Proposed Changes)

### A. 优化视觉配置文件 (`src/dpeva/utils/visual_style.py`)
1. **全局字号增大**：将 `get_analysis_parity_profile` 的函数签名默认参数由 `font_size: int | float = 12` 修改为 `16`（原定为14，根据用户反馈进一步增大）。这会使所有基础字号大幅提升。
2. **同步增大标题字号**：为了让主标题在大字号下更具视觉层级，将 `enhanced_fonts` 中的 `title_scale`由 `1.14` 略微提升至 `1.20`，配合基础字号 `16`，使大标题更加醒目、重点突出。
3. **防重叠 (Tick Pad)**：将 `enhanced` 配置中的 `error_tick_pad` 从 `-3.0` 修改为 `1.0`。正向的 pad 值会将顶部坐标轴的刻度标签向上推，从而完全避开刻度线与图表内容。
4. **增大散点与六边形 (Marker Size & Grid Size)**：
   - 将基础 `scatter_size` 由 15.0 增至 20.0，`main_scatter_size` 由 15.5 增至 20.0。
   - 对 `quantity_overrides` 中的散点尺寸进行等比例放大（如 cohesive_energy 增大至 22.0，force 增大至 24.0 等）。
   - 将 `main_density_gridsize` 由 `60` 减小至 `50`（包括基础配置、`force` 以及 `virial`）。在 hexbin 中，较小的网格数意味着将画布划分为更少、更大的六边形，从而实现“增大 markersize”的视觉效果。

### B. 修复统计框字号传参 (`src/dpeva/inference/visualizer.py`)
1. 在 `plot_parity` 中，将 `self._add_stats_box(ax, stats_text)` 修改为 `self._add_stats_box(ax, stats_text, fontsize=fonts["label"])`。
2. 在 `plot_parity_enhanced` 中，将 `self._add_stats_box(ax_main, stats_text)` 修改为 `self._add_stats_box(ax_main, stats_text, fontsize=fonts["label"])`。
*原因*：落实用户此前“将统计模块的字号提高到和轴标题一致”的诉求，保证在全局字号提升时，左上角的统计模块文字足够大且重点突出，不被边缘化。

## 4. 假设与决策 (Assumptions & Decisions)
- 将 `font_size` 默认值增加至 16 (12 -> 16) 且提升标题字号比例，配合现有的 `layout_padding` 是安全的。因为去除了 `tight_layout` 裁剪，画幅较大，能够容纳更大的文字。
- 将 hexbin 的 `gridsize` 设为 `50` 是一个折中选择，既能让六边形看起来更清晰饱满，又不会损失过多误差密度的空间分布细节。
- `error_tick_pad` 设为 `1.0` 足以将数字提至边框外侧的上方，呈现干净的学术排版。

## 5. 验证步骤 (Verification Steps)
1. 在 `test/init` 目录下执行 `dpeva analysis config_analysis_val.json`。
2. 观察生成的 `parity_cohesive_energy_enhanced.png`：
   - 确认右上角 Error Density 图表上方的数字是否已上移且不与刻度重叠。
   - 确认主图散点大小明显增加，主图左上角统计框内的字号是否已与轴标题大小一致。
3. 观察生成的 `parity_force_enhanced.png`：
   - 确认六边形 (hexbin) 颗粒是否变大，图表整体字号是否更加清晰易读，整体留白与排版是否和谐。
