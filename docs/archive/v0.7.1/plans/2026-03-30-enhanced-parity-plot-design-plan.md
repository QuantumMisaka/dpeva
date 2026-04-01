---
title: Enhanced Parity Plot 设计规划
status: active
audience: developers
last-updated: 2026-03-30
owner: Visualization Maintainers
---

# Enhanced Parity Plot 设计规划

- Status: active
- Audience: Developers
- Last-Updated: 2026-03-30

## 1. 目标

本规划文档用于解释当前 `enhanced parity plot` 中不同物理量图形出现显著差异的来源，并将这些差异上升为可复用、可审计的设计契约。重点对象是：

- `parity_cohesive_energy_enhanced.png`
- `parity_force_enhanced.png`

目标不是让所有物理量“看起来完全一样”，而是在统一母体之上建立 **quantity-aware** 设计：

- 小动态范围、贴线精度高的物理量，优先突出细腻的贴线关系与边缘分布；
- 高动态范围、高密点云物理量，优先保证主图区可读性，弱化容易喧宾夺主的辅助面板。

## 2. 两张图为何明显不同

### 2.1 直接原因

两张图出现明显不同，并不是随机结果，而是当前代码已按物理量类型应用了不同的 profile 覆盖：

1. **主图区渲染方式不同**
   - Cohesive Energy 默认仍使用 `scatter`
   - Force 已切换为 `hexbin` 密度主图区

2. **辅助面板策略不同**
   - Cohesive Energy 保留 top / right / error 三个辅助面板
   - Force 关闭 top / right，仅保留 error 面板

3. **坐标格式策略不同**
   - Cohesive Energy 关闭 scientific offset 干扰
   - Force 继续允许 scientific formatter，以兼容更宽的动态范围

4. **布局比例不同**
   - Cohesive Energy 使用更完整的四象限增强模板
   - Force 使用更偏“主图区优先”的压缩辅助布局

### 2.2 设计动机

#### Cohesive Energy

Cohesive Energy 的数据有几个典型特征：

- 数值范围相对紧凑；
- 主点云大多贴近对角线；
- 主结论通常是“整体拟合是否均匀、局部峰群是否存在系统偏移”；
- 边缘分布本身也有解释价值，因为它反映了样本能量分布结构。

因此当前设计选择：

- 主图区继续保留散点表达，强调逐点贴线关系；
- 保留 top/right 边缘分布，用于帮助读者理解数据群峰结构；
- 对 error panel 保持较轻权重，只作为辅助说明；
- 关闭 scientific offset，避免小范围能量图出现多余的指数偏移文本。

#### Force

Force 的数据有不同特征：

- 动态范围大；
- 零附近点云极度密集；
- 少量远离零点的样本会把全局坐标范围拉宽；
- 如果继续用普通 scatter + 全量边缘面板，主图区容易发白、辅助面板反而占据注意力。

因此当前设计选择：

- 将主图区切到 `hexbin`，把“局部密度”而不是“单点透明叠加”作为核心表达；
- 关闭 top/right 面板，避免高密直方分布挤占版面；
- 仅保留 error panel，继续提供误差中心化信息；
- 保留 scientific formatter，以避免大范围坐标下刻度表达失真或过长。

## 3. 代码层来源映射

### 3.1 样式源头

`src/dpeva/utils/visual_style.py` 是当前 enhanced parity quantity-aware 设计的单一事实源：

- 基线 enhanced profile 负责统一母体参数；
- `quantity_overrides` 负责按物理量覆写；
- Force / Virial 在这里启用 density 主图区；
- Cohesive Energy / Energy 在这里保留 scatter 并关闭 scientific offset。

当前关键参数包括：

- `renderer_policy`
- `renderer_default`
- `side_panels_enabled`
- `error_inset_enabled`
- `main_density_mode`
- `main_density_gridsize`
- `main_density_mincnt`
- `main_density_norm`
- `main_density_norm_gamma`
- `colorbar_enabled`
- `colorbar_title`
- `scientific_enabled`
- `width_ratios`
- `height_ratios`

### 3.2 渲染分发

`src/dpeva/inference/visualizer.py` 中，enhanced parity 的差异化主要经过两层分发：

1. `_get_parity_profile(...)`
   - 将 base profile 与 quantity override 合并；
   - 将 `enhanced_parity_renderer` 配置与 profile 的 `renderer_policy / renderer_default` 合并，生成最终渲染配置。

2. `_plot_parity_main_layer(...)`
   - 当 `main_density_mode="scatter"` 时走散点；
   - 当 `main_density_mode="hexbin"` 时走密度主图区，并按 `main_density_norm` / `main_density_norm_gamma` 应用可审计的 density normalization。

3. `plot_parity_enhanced(...)`
   - 根据 `side_panels_enabled` 决定是否创建 top/right；
   - 在 hexbin 模式下为 Force / Virial 构建“底部横轴承载标题”的 `Error + colorbar` 右侧信息栏；
   - 根据 `error_inset_enabled` 决定是否保留 error panel；
   - 根据 `scientific_enabled` 控制 formatter 行为。

## 4. 当前 quantity-aware 设计矩阵

| 物理量 | 主图区 | 辅助面板 | formatter | 当前用途定位 |
|---|---|---|---|---|
| Cohesive Energy | scatter | top + right + error | 非 scientific 优先 | 主文候选 / 高质量补充图 |
| Energy | scatter | top + right + error | 非 scientific 优先 | 补充图 |
| Force | hexbin | error + colorbar | scientific 保留 | 高密诊断增强图 / 补充图 |
| Virial | hexbin | error + colorbar | scientific 保留 | 高动态范围诊断增强图 |

## 5. 设计原则

### 5.1 统一母体，不追求表面一致

Enhanced parity 应共享同一套语义骨架：

- 主图区永远优先；
- identity line 永远存在；
- error panel 仍是增强图的重要辅助信息；
- 样式入口统一收敛在 profile，而不是散落到绘图函数中。

但不同物理量不应被强制塞进同一视觉模板。真正的一致性应来自：

- 同一职责边界；
- 同一配置结构；
- 同一视觉层级原则；
- 而不是同一张“长得一样”的图。

### 5.2 高密点云优先读密度，不优先读单点

Force / Virial 的核心挑战不是“看见每一个点”，而是：

- 看见密度主团；
- 看见尾部偏移；
- 看见误差是否围绕零集中；
- 看见是否存在远离主团的异常结构。

因此 density-aware 主图区是合理扩展，而不是风格漂移。

### 5.3 小动态范围图优先保留边缘分布

Cohesive Energy 的 top/right 面板仍然有价值，因为：

- 峰群结构本身具有解释意义；
- 点云不至于拥挤到必须使用 density 替代；
- 主图区与边缘分布之间能形成互补，而不是竞争。

## 6. 后续规划

### 6.1 必须维持

- `visual_style.py` 继续作为 parity 样式唯一入口；
- quantity override 只能覆写必要字段，不得复制整套 profile；
- Force / Virial 的 density 模式必须保持显式、可审计、可测试。

### 6.2 建议推进

- 已实现的本轮收敛项：
  - Force 默认提高 `main_density_mincnt` 与 `gridsize`
  - Virial 单独保留更紧凑的 `gridsize`
  - Force / Virial 使用顺序色图并显式补齐 colorbar
  - 新增 `enhanced_parity_renderer` 配置项，允许用户统一切换 `auto / scatter / hexbin`
- 后续仍可继续优化：
  - 细化 error panel 的窄峰/宽尾刻度策略
  - 视真实样例继续微调 colorbar 底部横轴标题与 tick 密度
- 为 error panel 建立更明确的“窄峰/宽尾”刻度策略。

### 6.3 可选扩展

- 增加 `main_density_mode="hist2d"` 或 `contourf` 的实验入口；
- 将主图区 colorbar 作为显式可选项，而不是默认开启；
- 为投稿版和诊断版分别建立更清晰的 profile 别名。

## 7. 验收标准

- 开发者能够从 profile 直接解释为何 Cohesive Energy 与 Force 图形不同；
- 图形差异可追溯到具体字段，而不是隐式硬编码；
- 新增物理量时，可通过 quantity override 决定其属于 scatter 类还是 density 类；
- 单元测试能够覆盖：
  - density 模式是否启用；
  - scatter 模式是否保持；
  - renderer 显式覆盖是否生效；
  - colorbar 是否按 hexbin 策略出现；
  - panel policy 是否按预期生效；
  - formatter 是否按物理量策略切换。
