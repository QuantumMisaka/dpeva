# Parity Plots 排版统筹调整调研与测试总结报告

## 1. 问题回顾与现状分析 (Current State Analysis)

用户提出需求：**“统筹调整两类 Enhanced Parity Plots（Cohesive Energy 与 Force），使其在同样的 `figure_size` 和同样的字号大小下，呈现出类似的字体大小，且不改变已有排布方式、不产生文字图片重叠。”**

目前我们在 `visual_style.py` 中为不同的物理量设置了不同的 `figure_size`：
- **Cohesive Energy**: `(9.1, 7.7)`，右侧为概率密度子图。
- **Force**: `(9.8, 7.4)`，右侧为误差密度子图和长数字标签的 Colorbar。

虽然两者底层引用的 `font_size` (例如 14pt) 是完全一致的，但在文档中（或缩放到相同宽度显示时），Force 的字号看起来明显比 Cohesive Energy 小。

## 2. 调研与排查过程 (Investigations)

我进行了一系列脚本测试，深入剖析了导致该现象的根本原因。

### **尝试 1：探究 `bbox_inches="tight"` 的裁剪机制**
我通过脚本捕获了 `savefig` 最终输出的像素尺寸：
- **Force** 最终图片尺寸：`2782 x 2227` 像素。
- **Cohesive Energy** 最终图片尺寸：`2272 x 2308` 像素。

**结论**：`bbox_inches="tight"` 会根据图中**实际可见的墨水区域**（Visible Artists）裁剪掉所有的边缘留白。因为 Force 右侧带有带有极宽数字标签的 Colorbar，所以它保留了更多的右侧宽度；而 Cohesive Energy 右侧较空，因此被裁剪掉了大量空余宽度。
**影响**：当这两张被裁剪到不同宽度（2272px vs 2782px）的图片在论文排版中被强行缩放对齐到同一物理宽度时，原始较宽的 Force 图会被压缩得更厉害，导致其字体在视觉上缩小。

### **尝试 2：直接禁用 `bbox_inches="tight"`**
我尝试在代码中移除了 `bbox_inches="tight"`，直接使用预设的 `figure_size` 输出：
- **结果**：此时图片尺寸严格等于 `figure_size * dpi`（即 2730x2310 和 2940x2220）。因为 `visualizer.py` 中有硬编码的 `layout_padding`（比如 `right=0.955` 等），**所有文本均未被切断**，且没有重叠。
- **代价**：虽然解决了相对缩放比例的问题，但由于 `ax_main.set_aspect("equal")` 强制主图为正方形，这使得在长宽比不匹配时，图表内部会产生较大的多余留白（White Space），视觉上不够紧凑，可能不符合“无切断且无过度留白”的论文发表标准。

### **尝试 3：强制修改 `figure_size` 使其宽度统一**
为了在保留 `bbox_inches="tight"`（保证紧凑）的同时让两者的最终裁切宽度一致，我计算并测试了扩大 Cohesive Energy 的 `figure_size` 宽度（例如从 9.1 增加到 11.4）。
- **结果**：由于 `aspect="equal"` 约束，仅增加宽度而不等比增加高度会导致多余的宽度纯粹变成空白区域，而这部分空白区域在 `savefig` 时依然会被 `tight` 裁剪掉。更严重的是，过大的宽高比触发了 `_align_scatter_enhanced_axes` 对齐计算时的负宽度 Bug，导致 Matplotlib 布局引擎陷入无限循环死锁。

### **尝试 4：等比例缩放 `figure_size` 以骗过相对缩放**
由于裁剪的核心是由“有内容的宽度”决定的，我等比例放大了 Cohesive Energy 的 `figure_size`（从 `(9.1, 7.7)` 放大到 `(11.16, 9.45)`）：
- **结果**：成功输出了宽度为 2666px 的紧凑图片，与 Force 的 2747px 宽度仅相差 3%。此时两者在文档中被缩放时，其物理字体大小差异将被缩小至肉眼不可见的 3% 以内。
- **问题**：这种做法违背了“在同样的 `figure_size` 下”的约束前提，属于治标不治本的 Hack。

## 3. 核心矛盾点总结

- `plot_parity_enhanced` 中由于必须保持物理量对角线为 **1:1 的正方形 (`set_aspect("equal")`)**，使得它的布局对 `figure_size` 的长宽比极其敏感。
- Force（带 Colorbar）与 Cohesive Energy（无 Colorbar）在右侧的**内容占地宽度存在天然的不一致**。
- `bbox_inches="tight"` 会根据内容占地无差别地切除所有辅助留白。
- 只要右侧内容宽度不一致，且启用了 `tight` 裁剪，任何相同 `figure_size` 的图最终输出的相对长宽比就必然不同。

## 4. 结论：是否可以较小代价达成？

**可以，但需要做出明确的取舍与策略转变。**

要完全满足“同 `figure_size` + 同 `font_size` + 字体视觉一致”，**代价最小且唯一合理的方案是放弃依赖 `bbox_inches="tight"` 的自动裁剪**。

具体方案建议如下：
1. **统一尺寸配置**：在 `visual_style.py` 中为所有的 Parity 设定一个通用且宽容度高的 `figure_size`（如 `(9.6, 7.5)`）。
2. **移除 `tight` 裁剪**：在 `visualizer.py` 中去掉 `bbox_inches="tight"`。
3. **修复对齐算法 Bug**：由于移除 `tight` 后需要精准控制边距，需要修复 `_align_scatter_enhanced_axes` 中因为过度宽泛的边距计算导致 `right_width` 小于 0 从而引发挂起的隐藏 Bug（已在测试脚本中验证该修复仅需一行代码 `max(0.01, min(...))`）。

**该方案的效果**：
所有出图将具有**完全相同、精确到像素的最终图片尺寸**（例如 2880 x 2250）。当您将它们排版到论文中时，缩放比例将100%相等，字体大小看起来绝对一致。
**小瑕疵**：由于 Force 有 Colorbar 而 Cohesive 没有，Cohesive 图的右侧会比 Force 图留有稍微多一点点的白色边缘，但这是保持物理比例一致所必须付出的排版代价。

如果您认可这种“取消紧凑裁剪，保证外框绝对尺寸一致”的思路，请回复同意，我将立即通过 Spec 将其落地并执行修改。
