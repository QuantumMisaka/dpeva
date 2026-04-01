# Plan: 修复 Enhanced Parity Plot 中 Error Density 子图的倒三角伪影

## 1. 问题分析 (Current State Analysis)
在 `parity_force_enhanced.png` 的 hexbin 模式图中，右上角的 Error Density 子图出现了一个奇特的“倒三角线”（平滑且呈尖角的折线）。
经过排查，该伪影产生的原因如下：
1. **数据范围极大**：Force 的误差数据 `err_values` 中存在极少数的离群点，导致数据的整体数值范围非常大（例如 `[-80, 80]`）。
2. **KDE 网格稀疏**：在调用 `sns.histplot(kde=True)` 时，seaborn 会基于整个数据的最大/最小值生成一个固定大小（默认 `gridsize=200`）的网格来计算核密度估计 (KDE)。由于数据跨度极大，网格点之间的间距变得非常宽。
3. **强制缩放引发失真**：为了只展示核心误差分布，代码在绘制完成后使用了 `ax_err.set_xlim(-p_high * 1.5, p_high * 1.5)` 强制将 X 轴缩放到了 99 分位数附近（例如 `[-0.5, 0.5]`）。在这一极小的可视范围内，KDE 的评估点仅剩下 1 到 3 个。Matplotlib 将这几个稀疏的点用直线连接起来，从而在视觉上形成了一个粗糙的、带尖角的“倒三角”。
4. **副遗留问题**：极大的数据范围同时导致 Freedman-Diaconis 规则（`bins="fd"`）计算出数千个极细的直方图分箱，造成绘制负担和潜在的渲染过密问题。

## 2. 解决方案 (Proposed Changes)
为解决上述问题，我们需要在调用 `_plot_histogram_with_kde` 绘制分布之前，提前过滤掉位于可视范围之外的极端离群误差值，使得 KDE 的网格评估点和直方图的分箱计算都能够集中在目标展示区域内。

**修改文件**: `src/dpeva/inference/visualizer.py`
**目标函数**: `plot_parity_enhanced`
**具体修改**:
在 `ax_err` 相关的绘制代码块中（约在 L724-L734），在调用 `_plot_histogram_with_kde` 之前增加对 `err_values` 的过滤逻辑：

```python
        if ax_err is not None:
            error_xlabel = self._format_error_axis_label(unit, profile)
            is_hexbin_sidebar = ax_top is None and ax_right is None
            
            p_high = np.percentile(np.abs(err_values), 99)
            if p_high > 0:
                err_limit = p_high * 1.5
                visible_err_values = err_values[(err_values >= -err_limit) & (err_values <= err_limit)]
            else:
                visible_err_values = err_values

            self._plot_histogram_with_kde(
                ax_err,
                visible_err_values,
                color=colors["error"],
                orientation="vertical",
                alpha=profile["error_hist_alpha"],
            )
            
            if p_high > 0:
                ax_err.set_xlim(-p_high * 1.5, p_high * 1.5)
```

## 3. 假设与决策 (Assumptions & Decisions)
- **假设**：`p_high` 对应于 `np.abs(err_values)` 的 99 分位数，因此 `visible_err_values` 至少包含 99% 的原始数据。截断极少量的离群点不会影响核心分布的核密度形状，反而能极大提升局部 KDE 曲线的平滑度和分辨率。
- **决策**：直接对数据进行截断过滤，而不是尝试向 `sns.histplot` 传递 `kde_kws={"clip": ...}` 或者增大 `gridsize`。因为过滤数据不仅能彻底解决 KDE 曲线的低分辨率问题，还能一并解决直方图由于全量范围计算导致的过密分箱（`bins="fd"` 计算过拟合）问题，是最优解。

## 4. 验证步骤 (Verification steps)
1. 在 `src/dpeva/inference/visualizer.py` 中应用上述修改。
2. 运行相关的测试脚本 `pytest tests/unit/inference/test_visualizer.py` 确保不破坏现有功能。
3. （可选）如果用户愿意，可以在本地触发一次生成 `parity_force_enhanced.png` 的命令，观察右上角的 Error Density 图。预期的结果是：倒三角伪影消失，取而代之的是一条平滑、高分辨率的橘色核密度曲线。
