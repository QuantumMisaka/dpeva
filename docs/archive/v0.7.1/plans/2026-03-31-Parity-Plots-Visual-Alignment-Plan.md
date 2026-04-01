---
status: active
audience: developers
---

# Parity Plots Visual Alignment Plan

## 1. 背景与核心问题
用户发现 `parity_cohesive_energy_enhanced.png` (Scatter模式) 与 `parity_force_enhanced.png` (Hexbin模式) 在代码中设置了完全相同的字体大小（pt），但在最终生成的图片中，字体的视觉大小差异显著。

**根本原因分析**：
- **`aspect="equal"` 约束**：Parity 主图为了保证 X 轴和 Y 轴的物理比例一致（1:1），使用了 `ax.set_aspect("equal", adjustable="box")`。这会强制主图为正方形。
- **不同宽度的侧边栏**：Hexbin 模式右侧包含误差分布图和 Colorbar，较宽；而 Scatter 模式右侧仅有误差分布图，较窄。
- **`bbox_inches="tight"` 裁剪机制**：在保存图片时，Matplotlib 默认启用了 `bbox_inches="tight"`。该机制会自动裁剪掉图片边缘的空白区域。由于 Scatter 模式右侧较窄，且主图为正方形，导致其左右两侧有大量留白被裁剪。裁剪后，Scatter 模式的实际图片宽度远小于 Hexbin 模式。
- **文档渲染缩放**：当这两张物理宽度不同的图片被插入到 Markdown 或 Word 中并以相同的显示宽度（例如 100% 页面宽度）渲染时，原本较窄的 Scatter 图片被放大了更多倍，导致其字体在视觉上显得比 Hexbin 图片大得多。

## 2. 迄今为止的尝试总结

为了解决上述问题，我们进行了以下探索和尝试：

### 尝试一：全局调整 `pt` 字号大小
- **操作**：在 `visual_style.py` 中提高了所有字体的基础字号（加2），并对齐了 Stats Box 的字号。
- **结果**：字体确实变大了，但 Scatter 和 Hexbin 之间的**相对大小差异**依然存在。因为只要裁剪机制存在，物理宽度的差异就会导致缩放比例不同。

### 尝试二：移除 `bbox_inches="tight"` 强制固定物理尺寸
- **操作**：在 `visualizer.py` 的 `savefig` 中去除了 `bbox_inches="tight"` 参数，试图让所有图片严格按照 `figure_size` 的设定输出，保证画布物理尺寸完全一致。
- **结果**：引发了 Matplotlib 布局引擎死循环（进程挂起）。
- **原因**：在去除紧凑裁剪后，内部自定义的对齐函数 `_align_scatter_enhanced_axes` 在计算右侧边栏宽度时，由于主图占据了太多空间，计算出了负数宽度。Matplotlib 无法处理负数宽度的 Axes，导致死循环。

### 尝试三：修复负宽度 Bug 并准备统筹 `figure_size`
- **操作**：在 `_align_scatter_enhanced_axes` 中引入了宽度截断（`right_width = max(0.01, min(...))`），成功修复了死循环 Bug。
- **当前状态**：系统已经可以生成未被 `tight_layout` 裁剪的定尺寸图片，但尚未在 `visual_style.py` 中为两类图找到一个完美的、能兼容两者的统一 `figure_size` 和边距配置。

## 3. 作图要求是否可以以较小代价达成？

**结论：可以以较小代价达成，但需要接受视觉上的小妥协（留白）。**

要让两张图在相同 `figure_size` 下具有相同的字体视觉大小，且不改变现有排布方式、不发生重叠，**唯一且代价最小的方案**是：

1. **放弃 `bbox_inches="tight"`**：严格遵循代码中设定的 `figure_size`。这样生成的图片物理尺寸完全一致，在文档中同宽显示时，缩放比例也完全一致，字体视觉大小自然完美统一。
2. **设立统一的、宽容度高的 `figure_size`**：例如设定为 `[12.5, 10.0]`，并在 `visualizer.py` 中手动调整 `subplots_adjust`（如 `left=0.08, right=0.92, bottom=0.1, top=0.9`），确保在最宽的情况（Hexbin + Colorbar）下，标签也不会被切掉。
3. **视觉妥协（留白）**：由于 Scatter 模式没有 Colorbar，且主图必须是正方形，在统一的定宽画布下，Scatter 模式的右侧或图形边缘必然会比 Hexbin 模式多出一些**空白区域（Whitespace）**。这是几何学上的必然结果（正方形主图 + 窄侧栏 放在 宽画布 中）。如果不接受留白，就必须裁剪；一旦裁剪，等宽插入文档时字体大小就不可能一致。

## 4. 后续执行计划 (Action Items)

如果您确认接受上述分析和“Scatter图会有稍多留白”的妥协，我们将立即执行以下步骤完成最终交付：

1. **统一画布尺寸**：在 `src/dpeva/utils/visual_style.py` 中，清理所有 quantity_overrides 中的 `figure_size`，将所有 enhanced parity 的 `figure_size` 统一为一个较大且安全的尺寸（如 `[12.5, 10.0]`）。
2. **全局边距保护**：在 `plot_parity_enhanced` 中通过 `fig.subplots_adjust` 显式声明安全边距，保护长标题或轴标签不会因超出画布物理边界而被截断。
3. **测试验证**：运行 `config_analysis_val.json` 进行端到端作图测试，确保 cohesive energy 和 forces 的图像生成顺利，物理尺寸完美对齐，字号在视觉上完全一致。
