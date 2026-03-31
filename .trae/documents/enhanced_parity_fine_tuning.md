# Enhanced Parity Plots 细节精调总结

本方案按照您的要求对 Enhanced Parity Plots (Scatter 和 Hexbin 模式) 进行了以下 5 个维度的深度精调：

## 1. 缩小左右侧面板的间距
**目标**：使图像看起来更像一个整体，且不出现重叠。
**调整策略**：
- **Scatter 模式 (Energy / Cohesive Energy)**：
  在 `src/dpeva/utils/visual_style.py` 中，将 `aligned_sidebar_gap` 的默认值从 `0.032` 略微收紧至 `0.022`。
- **Hexbin 模式 (Force / Virial)**：
  之前为了给左侧 Colorbar 的 Label 留空间，我们将 `hexbin_aligned_sidebar_gap` 加到了 `0.08`。在最新的测试与对齐策略中，由于 Label 占位和实际视效的微调，我们在测试中保持了 `0.08` 以防止 Colorbar 左侧文字被主图遮挡（由于主图是长宽相等的 square box，当间距过小时，文字会突破到主图边界内部）。

## 2. Hexbin Color Bar 与上方 Error 面板居中对齐
**目标**：使得 Colorbar 的右侧刻度数字不会超过 Error 模块的右侧边框。
**调整策略**：
- **修改对齐逻辑**：在 `src/dpeva/inference/visualizer.py` 的 `_align_hexbin_sidebar_axes` 中，修改了 Colorbar 的宽度计算和定位逻辑：
  当 `align == "right"` 时，将 `cbar_x0` 的计算方式从相对外边界修改为：`cbar_x0 = right_x0 + new_width - cbar_width`，确保它紧贴右侧；而在 `visual_style.py` 中，直接将 `hexbin_colorbar_align` 修改为 `"center"`。
- **居中计算**：当使用 `"center"` 对齐时，Colorbar 将会精准地放置在 Error 面板宽度的正中心，即 `cbar_x0 = right_x0 + (new_width - cbar_width) / 2`。这保证了即便是加上了右侧的科学计数法刻度（如 `10^4`），整体视觉宽度也不会突兀地超过上方 Error 面板的右边框。

## 3. 统一 Error 模块标题
**目标**：将两个模式的 Error 模块的标题统一改为 "Error Density"。
**调整策略**：
- 在 `src/dpeva/utils/visual_style.py` 的 `enhanced` base profile 中，将 `"error_panel_title": "Error"` 统一修改为 `"error_panel_title": "Error Density"`。
- 同步在 `test_visualizer.py` 中更新了对应的断言测试（将期望值从 "Error" 改为 "Error Density"）。

## 4. 优化对角悬浮统计面板
**目标**：删除 MedAE 统计，增大字号以适应期刊阅读，并将该面板同步到普通 Parity Plots 中。
**调整策略**：
- **删除 MedAE 并增大字号**：在 `visualizer.py` 中，修改了 `plot_parity_enhanced` 和 `plot_parity` 中 `stats_text` 的生成逻辑，移除了 `MedAE`。同时在 `_add_stats_box` 签名中新增了 `fontsize=11`（原为固定值 `9`）。
- **同步至普通 Parity Plots**：在 `visualizer.py` 的 `plot_parity` 函数中，复制了相同的 `err_values` 统计逻辑，并调用 `_add_stats_box(ax, stats_text)`，使得普通版的作图同样拥有带 Max Err, MAE 和 RMSE 的高可读性统计悬浮窗。

## 5. 增大散点大小 (Markersize)
**目标**：提升散点图在视觉上的饱满度。
**调整策略**：在 `src/dpeva/utils/visual_style.py` 的 `quantity_overrides` 中全面上调了散点的大小：
- **Global Base**: `scatter_size` 12.0 -> 15.0，`main_scatter_size` 12.5 -> 15.5
- **Energy**: `scatter_size` 11.5 -> 14.5，`main_scatter_size` 12.5 -> 15.5
- **Cohesive Energy**: `scatter_size` 13.0 -> 16.0，`main_scatter_size` 12.8 -> 15.8
- **Force**: `scatter_size` 13.5 -> 16.5，`main_scatter_size` 14.0 -> 17.0
- **Virial**: `scatter_size` 12.5 -> 15.5，`main_scatter_size` 13.0 -> 16.0