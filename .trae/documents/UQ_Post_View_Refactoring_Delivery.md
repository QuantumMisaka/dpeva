# uq-post-view 数据处理功能点重构交付文档

## 1. 代码现状分析
本项目（DP-EVA）旨在构建高效的 DPA3 主动学习框架。在本次重构前，数据处理逻辑主要集中在 `utils/uq/uq-post-view.py` 等独立脚本中，存在逻辑耦合、缺乏模块化、配置硬编码等问题。`utils/uq/uq-post-view-2.py` 作为演进后的脚本，引入了多项关键改进，但仍未集成到 `src/dpeva` 核心库中。

## 2. 重构目标
本次重构的核心目标是将 `uq-post-view-2.py` 中的最佳实践和算法改进完全迁移并封装到 `src/dpeva` 核心库中，构建可配置、模块化、健壮的 `CollectionWorkflow`，并确保与原脚本功能的一致性。

## 3. 关键代码改进点说明
本次重构将 `uq-post-view-2.py` 相比旧版脚本的以下改进点成功集成到核心库：

| 功能点 | 旧版实现 (Pre-Refactor) | 重构后实现 (Post-Refactor) | 优势 |
| :--- | :--- | :--- | :--- |
| **UQ对齐** | `StandardScaler` (Z-Score) | **`RobustScaler` (Median/IQR)** | 对异常值（Outliers）更鲁棒，避免少数高UQ值扭曲整体分布 |
| **数据加载** | 简单的 `from_dir` | **鲁棒的迭代加载机制** | 支持部分加载失败的容错，兼容无标签数据（Unlabeled Data） |
| **描述符处理** | 仅均值池化 | **L2 归一化 (L2 Norm)** | 提升 DIRECT 聚类和采样的有效性，消除量纲影响 |
| **可视化** | 直接绘图 | **截断过滤 ([0, 2]) + 异常检测** | 避免极值压缩图表有效区间，提升可视化可读性 |
| **空值处理** | 强依赖 Ground Truth | **兼容无 Ground Truth 场景** | 支持纯推理场景（仅有结构无标签），增强泛化能力 |
| **异常防护** | 无 | **空候选集检查** | 避免因筛选结果为空导致的程序崩溃 |

## 4. 重构后的代码结构
重构后的逻辑分布在 `src/dpeva` 的各个模块中，职责清晰：

*   **`src/dpeva/workflows/collect.py`**:
    *   **`CollectionWorkflow` 类**: 编排整体流程（加载 -> UQ计算 -> 筛选 -> 采样 -> 导出）。
    *   集成数据加载的增强逻辑和配置管理。
*   **`src/dpeva/uncertain/calculator.py`**:
    *   **`UQCalculator` 类**: 更新 `align_scales` 使用 `RobustScaler`。
    *   增强 `compute_qbc_rnd` 以兼容无 Ground Truth 的情况。
*   **`src/dpeva/uncertain/visualization.py`**:
    *   **`UQVisualizer` 类**: 新增 `_filter_uq` 截断逻辑。
    *   增强绘图方法（`plot_2d_uq_scatter` 等）以处理缺失的误差标签（Hue）。
    *   修复 Matplotlib Locator 在大数据范围下的性能问题。

## 5. 验证测试结果
我们在 `test/verification_test_run` 目录下构建了验证环境：
*   **配置文件**: 提取自 `uq-post-view-2.py` 的关键参数。
*   **入口脚本**: `run_refactored_workflow.py`。

**验证结果**:
*   脚本成功运行完成。
*   日志显示 UQ 统计信息正常输出。
*   可视化模块正确触发截断警告（如 `Truncating ... values outside [0, 2]`），证明截断逻辑生效。
*   DIRECT 采样成功执行（Birch 聚类自动调整阈值）。
*   最终输出了 `sampled_dpdata` 和 `other_dpdata`，验证了数据导出功能的完整性。

## 6. 总结与后续建议
本次重构成功将散乱的脚本逻辑收敛至标准化的库结构中，不仅提升了代码质量，还增强了系统的鲁棒性和可维护性。

**后续迭代建议**:
1.  **单元测试**: 为 `UQCalculator` 和 `UQVisualizer` 补充独立的单元测试，覆盖更多边缘情况（如全空数据）。
2.  **配置系统增强**: 考虑引入 `Hydra` 或 `pydantic` 进行更严格的配置校验。
3.  **文档自动生成**: 基于 docstrings 自动生成 API 文档。
