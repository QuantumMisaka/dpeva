# Hexbin Mode Enhanced Parity Plots 改进方案 (修订版)

根据您的最新反馈，本修订版计划对先前的方案进行如下调整：

## 1. 彻底移除冗余的散点图逻辑 (Remove Overlay Scatter Completely)
- **目标**：不只是在配置中禁用，而是从代码层面上彻底删除 `main_overlay_scatter` 相关的逻辑和测试，保持代码整洁。
- **实施文件**：
  - `src/dpeva/inference/visualizer.py`：删除 `_plot_parity_main_layer` 中关于 `ax.scatter` 叠加层的整块代码，同时移除 `_plot_violin_distribution` 相关的未使用函数。
  - `src/dpeva/utils/visual_style.py`：在 `force` 和 `virial` 的配置字典中彻底删除所有 `main_overlay_scatter_*` 键值对，并将 `main_density_mincnt` 设为 `1`。
  - `tests/unit/inference/test_visualizer.py`：删除针对叠加散点图的测试用例（如 `test_plot_parity_enhanced_force_overlays_faint_scatter_on_hexbin`），并将对 Violin Plot 的测试修改为测试 Histogram。

## 2. Error 面板 X 轴截断与主图统计面板 (X-Axis Truncation & Stats Panel)
- **目标**：采用 Histogram+KDE 替代极细的小提琴图，引入 X 轴截断以展示真实分布，同时在主图补充统计信息以弥补截断导致的极差信息丢失。
- **实施文件**：`src/dpeva/inference/visualizer.py`
  - 在 `plot_parity_enhanced` 中，为 Error 面板计算 99 分位数 `p_high`，应用稳健的截断：`ax_err.set_xlim(-p_high * 1.5, p_high * 1.5)`。
  - 计算预测误差的极差（Max Error）、中位差（MedAE）、MAE 和 RMSE。
  - 调用现有的 `_add_stats_box`，将这些统计信息作为一个半透明文本面板添加到主对角线图（`ax_main`）的左上角，不仅优雅地保留了极差信息，还能让人一目了然地看到模型表现。

## 3. 标签与标题精准布局控制 (Label & Title Layout Control)
- **目标**：保持 Error KDE 标题在底部且不与下方的 Colorbar 冲突；Colorbar 标题移至左侧且不与左侧的主图冲突；确保 Error 面板顶部没有文字凸出，与主图完美平齐。
- **实施文件**：`src/dpeva/inference/visualizer.py` & `src/dpeva/utils/visual_style.py`
  - **Error KDE 底部标签**：在 `visualizer.py` 中，继续将 `Error (eV/A)` 等字样作为 `ax_err` 的 `xlabel` 放置在底部。
  - **避免上下重叠**：在 `visual_style.py` 中，将 `force` 和 `virial` 的 `hexbin_sidebar_hspace` 从 `0.28` 调大至 `0.45`（或更大），为 Error 面板底部的标签与下方的 Colorbar 之间预留充足的安全距离。
  - **Colorbar 左侧标题**：在 `visualizer.py` 中设置 `colorbar.ax.yaxis.set_label_position("left")`，将 "Counts Per Hexbin" 移至色条左侧。
  - **避免左右重叠**：在 `visual_style.py` 中，将 `hexbin_aligned_sidebar_gap` 增大至 `0.08`，确保色条左侧的标题有足够的空间，不与左侧的 `ax_main` 发生挤压重叠。

## 4. 验证步骤
1. 运行 `pytest tests/unit/inference/test_visualizer.py` 确保所有冗余测试已安全删除且其余测试通过。
2. 运行 `export DPEVA_INTERNAL_BACKEND=local && dpeva analysis config_analysis_val.json`。
3. 检查生成的 `parity_force_enhanced.png`：
   - 确认叠加散点已彻底移除。
   - 确认主图左上角有包含 Max, MedAE, MAE, RMSE 的半透明统计面板。
   - 确认右上角 Error 面板呈蓝色直方图+KDE 且分布饱满，无顶部凸出文字，底部带有清晰的 `Error (eV/A)`。
   - 确认下方的 Colorbar 与上方 Error 面板互不重叠，且左侧带有垂直的 `Counts Per Hexbin` 标题，未与主图冲突。