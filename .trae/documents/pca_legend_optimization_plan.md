# Collection PCA 可视化图例优化方案

## 1. 现状与问题分析
在最近的改动中，为了统一视觉层级，将 PCA 散点图 (`DIRECT_PCA_feature_coverage.png` 和 `Final_sampled_PCAview.png`) 的图例字号（`legend_scale=1.90`）与坐标轴标题对齐。但在实际出图中发现，由于数据点分布广泛，过大的图例在 `loc="best"` 策略下不可避免地遮挡了部分关键数据点。

用户核心约束：
1. 图例（label块）**必须保持在图内**，不能移到图外。
2. 字体大小**不能过度回调**，至少不能比修改前（`1.30`）小，需保证论文图表的可读性。

## 2. 拟修改方案

我们采用**综合优化策略（字号 + 边距 + 透明度）**，具体实施步骤如下：

### 2.1 调整 PCA 视觉配置 Profile
文件路径：`src/dpeva/utils/visual_style.py`
修改函数：`get_collection_pca_scatter_profile`
- **调整图例字号缩放比例**：将 `legend_scale` 和 `legend_title_scale` 从 `1.90` 下调至 `1.50`。这确保了图例字号适度缩小以释放空间，但仍然大于此前的原始配置（`1.30`），保障了清晰度。
- **增加坐标轴内边距**：将 `axis_margins` 从 `(0.02, 0.02)` 增大至 `(0.08, 0.08)`。这会在数据散点和坐标轴边缘之间留出 8% 的空白缓冲地带，极大增加了 `loc="best"` 算法寻找不遮挡数据点位置的成功率。

### 2.2 调低图例背景透明度
文件路径：`src/dpeva/uncertain/visualization.py`
涉及函数：`plot_pca_analysis` (Coverage Score Bar Chart 及其 Final Selection 图) 与 `_plot_coverage` (PCA feature coverage 图)
- **修改 `plt.legend` 的 `framealpha` 参数**：将 `framealpha` 从 `0.9` 调低至 `0.75`。在图例不可避免地覆盖到少许边缘数据点时，这一修改能让下方的数据点透过图例背景半透明显示，缓解“遮挡死角”的问题。

## 3. 预期收益
通过这套组合拳：
- **内边距增大** 提供了物理空间让图例“有处安放”；
- **字号合理回调** 减小了图例占据的绝对面积；
- **透明度降低** 作为最后一道防线，保证数据分布的整体可见性；
- **不把图例移出图外**，保持了论文单图结构的紧凑性和画幅比例的稳定。