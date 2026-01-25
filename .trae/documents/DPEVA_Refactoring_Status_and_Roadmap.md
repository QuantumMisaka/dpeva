# DPEVA 重构状态分析与开发规划文档

**文档版本**: 1.0
**日期**: 2026-01-25
**状态**: 草稿/征求意见

---

## 1. 项目概述与重构愿景

### 1.1 背景
DP-EVA (Deep Potential EVolution Accelerator) 是一个面向 DPA3 高效微调的主动学习框架。原项目代码主要由分散的 Shell 和 Python 脚本组成（位于 `utils/` 目录），虽然功能实现了科研需求，但在可维护性、扩展性和自动化程度上存在局限。

### 1.2 重构愿景
本次重构旨在将 DPEVA 从“脚本集合”进化为“工程化 Python 包”。遵循 **The Zen of Python** 哲学，我们致力于实现：
*   **显式优于隐式**：用明确的配置字典替代环境变量和硬编码。
*   **模块化设计**：将单体大脚本拆解为职责单一的原子模块。
*   **配置驱动**：实现代码逻辑与实验参数的完全分离。

---

## 2. 现状分析 (Current Status)

截止当前，核心业务逻辑的迁移工作已完成约 **85%**。

### 2.1 模块完成度评估

| 模块 | 功能描述 | 状态 | 评价 |
| :--- | :--- | :--- | :--- |
| **Training** | 并行模型训练 | ✅ 完成 | `ParallelTrainer` 封装了目录管理，支持 `local` (Multiprocessing) 和 `slurm` (Job Submission) 双模式。解决了原脚本无法监控任务状态的问题。 |
| **Feature** | 描述符/特征生成 | ✅ 完成 | `DescriptorGenerator` 统一了原子级和结构级特征生成接口，优化了内存使用，消除了重复代码。 |
| **Uncertainty** | 不确定度计算与筛选 | ✅ 完成 | 实现了 `Calculator` (计算), `Filter` (筛选), `Visualization` (绘图) 的彻底解耦。支持多种筛选边界策略（Strict, Circle, Tangent 等）。 |
| **Visualization** | 科研绘图 | ✅ 完成 | `UQVisualizer` 完美复刻了原 `uq-post-view.py` 的所有关键图表（KDE, Parity, PCA, Scatter），保留了论文级的绘图细节。 |
| **Inference** | 模型推理 | ⚠️ 部分完成 | `ModelEvaluator` 已实现基于 `dp test` 的并行推理，但缺乏对推理结果（如 RMSE, Force Error）的解析和后处理功能。 |
| **Workflow** | 业务流程编排 | ✅ 完成 | `CollectWorkflow` 等类成功串联了底层模块，提供了清晰的顶层接口。 |

### 2.2 架构改进亮点

1.  **可视化与逻辑分离**：
    *   **Before**: `uq-post-view.py` (800+ lines) 混杂了数据处理和 matplotlib 代码。
    *   **After**: `src/dpeva/uncertain/visualization.py` 专注于绘图，`workflow` 专注于调用。修改绘图样式不再影响数据逻辑。

2.  **任务提交抽象层 (`submission`)**：
    *   引入 `JobManager` 和模板引擎，使得代码可以在本地工作站和 Slurm 集群之间无缝切换，无需修改业务逻辑。

3.  **统一配置管理**：
    *   所有 Workflow 均接受 `config` 字典作为输入，为未来支持 YAML/JSON 配置文件和 CLI 工具打下基础。

---

## 3. 待办事项与技术债务 (Gap Analysis)

尽管重构进展顺利，但仍存在以下缺口需要填补：

### 3.1 关键功能缺口
*   **推理后处理 (Inference Post-processing)**：
    *   当前 `ModelEvaluator` 仅运行 `dp test` 并保存日志。
    *   **缺失**：解析 `dp test` 的文本输出或 `.npy` 结果，提取 RMSE (Energy/Force/Virial) 指标，生成“预测值 vs 真实值”的对比图。这是原 `dptest` 模块的重要功能。
*   **全流程集成测试**：
    *   目前缺乏一个端到端的测试脚本，用于验证从“训练 -> 推理 -> 筛选 -> 迭代”的数据流是否通畅。

### 3.2 代码质量与体验
*   **文档注释 (Docstrings)**：部分新模块的类和方法缺乏详细的参数说明（Docstrings）。
*   **错误处理**：部分文件操作未处理异常（如文件不存在、权限错误等），需要增强鲁棒性。

---

## 4. 迭代目标与路线图 (Roadmap)

### 阶段一：稳固基础 (Foundation) - 预计耗时：1-2 周
**目标**：补齐功能短板，确保新框架能完全替代旧脚本进行完整的科研循环。

1.  **完善 Inference 模块** (优先级：高)
    *   实现 `ResultParser` 类，解析 `dp test` 输出。
    *   添加 `InferenceVisualizer`，绘制 Parity Plot (Energy/Force)。
    *   *验收标准*：能够输出与原 `dptest` 一致的误差统计报表和图表。

2.  **建立集成测试 (Integration Test)** (优先级：高)
    *   编写 `tests/run_integration.py`，使用极小数据集跑通全流程。
    *   *验收标准*：一条命令即可无报错完成“训练-推理-筛选”闭环。

### 阶段二：扩展与优化 (Extension) - 预计耗时：2-3 周
**目标**：提升易用性，支持更多算法。

1.  **CLI 命令行工具** (优先级：中)
    *   基于 `argparse` 或 `click` 封装 `dpeva` 命令。
    *   支持 `dpeva train config.yaml`, `dpeva collect config.yaml` 等子命令。

2.  **支持更多采样策略** (优先级：低)
    *   在 `src/dpeva/sampling` 中扩展除 PCA/DIRECT 之外的聚类或采样算法（如 FPS）。

---

## 5. 架构说明与开发指南

### 5.1 目录结构
```text
src/dpeva/
├── training/       # 训练模块 (ParallelTrainer)
├── inference/      # 推理模块 (ModelEvaluator)
├── uncertain/      # 不确定度 (Calculator, Filter, Visualization)
├── feature/        # 特征生成 (DescriptorGenerator)
├── submission/     # 任务提交抽象 (Local/Slurm)
├── workflows/      # 业务流程层 (CollectWorkflow, TrainWorkflow)
└── io/             # 数据读写辅助
```

### 5.2 开发原则
1.  **新增功能**：
    *   首先思考该功能属于哪个原子模块（Training/Inference/Uncertain）。
    *   如果涉及多个模块交互，请在 `workflows` 层实现编排逻辑。
2.  **修改绘图**：
    *   严禁在计算逻辑中混入绘图代码。所有绘图逻辑必须在 `visualization.py` 中实现。
3.  **配置参数**：
    *   不要使用 `sys.argv` 或硬编码。所有参数应通过 `config` 字典传入类构造函数。

### 5.3 示例：如何调用采集流程
```python
from dpeva.workflows.collect import CollectWorkflow

config = {
    "data_path": "/path/to/data",
    "model_paths": ["m1.pb", "m2.pb"],
    "filter_strategy": "strict",
    "trust_range": [0.05, 0.15],
    "output_dir": "./iteration_1"
}

workflow = CollectWorkflow(config)
workflow.run()
```
