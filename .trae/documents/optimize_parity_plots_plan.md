# Plan: 进一步优化 Enhanced Parity Plots

## 1. 现状分析 (Current State Analysis)

根据用户的要求，需要对 `parity_cohesive_energy_enhanced.png` 和 `parity_force_enhanced.png` 两类图进行以下细节优化：

1. **Error Density 数字字号与位置**：
   - 目前 Error Density 面板（右上角子图）的数字（坐标轴刻度 tick labels）字号稍小，并且在scatter parity plots中位置偏低。
2. **Scatter Parity Plots 的 Marker Size**：
   - 目前的 `scatter_size` 对于普通的 Parity Plots 和 Enhanced Parity Plots（以及对应的 Quantity Overrides）都在 `visual_style.py` 中有定义。需要适度增大这些 marker 的尺寸，让散点看起来更加饱满。
3. **密度图的标题表达**：
   - 当前存在 "True Density", "Predicted Density", 和 "Error Density" 的标签，占用空间较多。
   - 用户的要求是将它们调整为“更简单紧凑”的方式。可以将它们分别简化为 "True", "Predicted" 和 "Error"，或者通过配置项直接使用已经定义好的面板标题 (`top_panel_title` 等)。在 `visual_style.py` 中，它们分别被设定为 `top_panel_ylabel`, `right_panel_xlabel`, 和 `error_panel_ylabel`。

## 2. 解决方案 (Proposed Changes)

**文件 1:** **`src/dpeva/utils/visual_style.py`**

- **字号调整**：在 `get_analysis_parity_profile` 中，提升 `panel_fonts` 的 `tick_scale`，使得边侧面板（包括 Error Density）的数字字号变大。
- **标题紧凑化**：将 Density 图的 ylabel/xlabel 简化：
  - `top_panel_ylabel`: "True Density" -> "True"
  - `right_panel_xlabel`: "Predicted Density" -> "Predicted"
  - `error_panel_ylabel`: "Error Density" -> "Error"
  - 连同 `error_panel_title`: "Error Density" -> "Error"
- **Marker Size 放大**：
  - `base_profile` 的 `scatter_size`: 15.0 -> 17.0
  - `enhanced` 的 `main_scatter_size`: 15.5 -> 18.0
  - 针对 `quantity_overrides`（energy, cohesive\_energy, force, virial）同步提高基础 `scatter_size` 和 `enhanced.main_scatter_size`（增加 2-3 个点）。

**文件 2:** **`src/dpeva/inference/visualizer.py`**

- **确保 Error Density 坐标轴数字生效**：Error Density 的 tick\_params 已经是基于 `panel_fonts["tick"]` 的，只需修改 `visual_style.py` 即可生效。

## 3. 假设与决策 (Assumptions & Decisions)

- **决策 1**：将 Density 子图的标签简化为 "True", "Predicted", "Error" 是最紧凑的表达方式，并且配合整体的 "Enhanced Parity" 主标题，语意依然明确。
- **决策 2**：散点大小（markersize）统一在 `visual_style.py` 的配置文件中调整，以确保所有依赖该配置的普通及增强对角图都能同步放大。
- **决策 3**：Stats Box 的 Y 位置从 0.98 调整到 0.985（接近上限，但留有 1.5% 边距，不至于压线）。

## 4. 验证步骤 (Verification steps)

1. 在 `visual_style.py` 和 `visualizer.py` 中实施上述更改。
2. 运行 `pytest /home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/dpeva/tests/unit/inference/test_visualizer.py` 确保修改不破坏已有代码逻辑。
3. （可选）提示用户在本地运行测试用例以查看图表的实际输出效果。

