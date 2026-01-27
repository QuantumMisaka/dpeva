# DPEVA Inference 模块实现文档

## 1. 概述

本模块 (`src/dpeva/inference`) 旨在完善 DPEVA 框架的推理评估能力。它填补了原工作流中仅能运行 `dp test` 但缺乏后续数据分析的空白。模块实现了对推理结果的自动解析、误差指标计算（RMSE/MAE）、分布统计分析（含离群值检测）以及科研级可视化绘图。

## 2. 模块结构

```text
src/dpeva/inference/
├── __init__.py       # 导出核心类
├── parser.py         # TestResultParser: 解析 dp test 输出结果
├── stats.py          # StatsCalculator: 统计指标计算与离群值分析
└── visualizer.py     # InferenceVisualizer: 绘图工具
```

同时，`src/dpeva/workflows/infer.py` 中的 `InferenceWorkflow` 已升级，集成了上述组件，支持一键式推理与分析。

## 3. 核心功能

### 3.1 结果解析 (`parser.py`)
*   **功能**: 自动读取 `dp test` 生成的 `*.out` 文件。
*   **Virial 支持**: 优先支持读取 `results.v_peratom.out`（原子归一化维里），兼容 `results.v.out`。
*   **智能检测**: 自动判断数据列是否全为零，从而识别是否存在 Ground Truth（真值）。如果不存在真值（例如仅用于 UQ 筛选的无标签数据），将自动跳过误差计算步骤。
*   **鲁棒的组成解析**: 新增 `get_composition_list` 方法，直接从系统名称（如 `O0H46Fe40C2`）中解析原子组成，解决了 `dpdata` 加载顺序与测试结果不匹配的问题。

### 3.2 统计分析 (`stats.py`)
*   **误差指标**: 当存在真值时，计算 Energy, Force, Virial 的 **MAE** (Mean Absolute Error) 和 **RMSE** (Root Mean Square Error)。
*   **结合能 (Cohesive Energy) 分析**:
    *   **自定义参考能量**: 支持通过配置传入单原子参考能量 (`ref_energies`)，用于精确计算结合能 ($E_{coh} = E_{pred} - \sum N_i E_{ref,i}$)。
    *   **自动回退机制**: 若未提供参考能量或存在元素缺失，自动回退到最小二乘法 (Least Squares) 拟合计算参考能量。
    *   **异常处理**: 自动检测训练集与测试集的数据长度不匹配问题，并优雅跳过相关计算，防止程序崩溃。
*   **Virial 分析**: 支持对应力的统计分析与误差计算（单位：eV）。
*   **离群值检测**: 采用 **IQR (Interquartile Range)** 方法自动识别离群值（Outliers）。

### 3.3 可视化 (`visualizer.py`)
*   **Parity Plot (对角线图)**:
    *   展示 Predicted vs True 的对比。
    *   涵盖 Energy, Force, Virial 以及 Cohesive Energy。
    *   单位自动适配（Energy: eV/atom, Force: eV/Å, Virial: eV）。
    *   支持大量数据点的光栅化（Rasterized）渲染，防止矢量图过大。
*   **Distribution Plot (分布图)**:
    *   基于 KDE (Kernel Density Estimation) 的概率密度分布。
    *   **双重分布展示**: 同时绘制“原始数据分布”（含离群值）和“清洗后分布”（剔除离群值，绿色虚线）。
*   **Error Distribution**: 绘制预测误差（Pred - True）的直方图与 KDE 曲线。

## 4. 使用指南

### 4.1 通过 Workflow 自动运行
在运行 `InferenceWorkflow` 时，可以在配置中指定 `ref_energies` 以启用高精度的结合能分析。

```python
from dpeva.workflows.infer import InferenceWorkflow

config = {
    "test_data_path": "/path/to/data",
    "output_basedir": "/path/to/output",
    "task_name": "test_01",
    "submission": {"backend": "local"},
    # 可选：指定单原子参考能量（单位：eV）
    "ref_energies": {
        "Fe": -3215.2791,
        "C": -156.0795,
        "O": -444.6670,
        "H": -13.5410
    }
}

workflow = InferenceWorkflow(config)
workflow.run()
# 运行结束后，分析结果将自动生成在 /path/to/output/0/test_01/analysis/ 目录下
```

### 4.2 手动触发分析
对于异步提交（如 Slurm 模式），可以在任务完成后手动调用分析接口：

```python
workflow.analyze_results()
```

### 4.3 输出文件说明
分析结果将保存在工作目录下的 `analysis/` 子目录中：
*   `statistics.json`: 包含详细的统计数据（均值、方差、分位数、离群值统计等）。
*   `parity_*.png`: 对角线图（Energy, Force, Virial, Cohesive Energy）。
*   `dist_*.png`: 分布图（Predicted Energy, True Energy, Force Magnitude, Virial 等）。
*   `error_dist_*.png`: 误差分布图。
*   `inference_summary.csv` (位于根目录): 汇总所有模型的 RMSE/MAE 指标。

## 5. 实现细节与亮点
*   **健壮性**: 针对无真值数据（Unlabeled Data），代码会自动降级，仅输出预测值的分布统计；针对数据长度不匹配情况，会输出 Warning 并跳过相关计算。
*   **灵活性**: 结合能计算支持 "Explicit Reference" 和 "Least Squares Fitting" 两种模式的自动切换。
*   **美观性**: 绘图采用了科研论文常用的风格（`seaborn` + `matplotlib`），字体清晰，配色区分度高。
