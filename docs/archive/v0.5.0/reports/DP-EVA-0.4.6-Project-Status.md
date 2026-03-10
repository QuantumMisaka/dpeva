---
title: Archived Document
status: archived
audience: Historians
last-updated: 2026-03-09
---

# DP-EVA v0.4.6 项目状态总结报告

**生成日期**: 2026-03-02
**当前版本**: v0.4.6
**基准分支/提交**: Main

## 1. 项目概述

DP-EVA (Deep Potential EVolution Accelerator) 是面向 DPA3/Deep Potential 生态的主动学习框架。其核心目标是通过 **多模型不确定度 (UQ)** 与 **结构代表性采样 (DIRECT/2-DIRECT)**，从海量候选结构中高效筛选出高价值样本，以降低标注成本并提升模型泛化能力。

项目采用 **"Unified CLI + DDD Architecture"** 设计，对外提供统一的 `dpeva` 命令行入口，对内通过分层架构实现流程编排与执行解耦。

## 2. 核心架构与设计理念

### 2.1 设计哲学
- **Explicit & Simple**: 优先清晰直观，避免隐晦技巧（遵循 Zen of Python）。
- **Orchestration vs Execution**: `Workflows` 只负责流程编排与状态流转，具体业务逻辑下沉至 `Managers` (IO, UQ, Sampling, Job)。
- **Configuration as Contract**: 使用 Pydantic v2 定义严格的配置模型（Schema），实现跨字段校验与默认值管理。

### 2.2 关键抽象
- **Dual-Mode Execution**: 支持 `Local`（直接执行）与 `Slurm`（生成脚本并提交）双模式，通过 `JobManager` 屏蔽调度差异。
- **Pipeline Abstraction**: 采样过程管线化（PCA -> Clustering -> Selection），支持 Standard DIRECT 与 2-Step DIRECT 策略。
- **Unified I/O**: 推理结果封装为 `PredictionData`，分析结果封装为 `Statistics`，确保模块间数据流转的标准化。

## 3. 功能完成度矩阵

| 模块/工作流 | 状态 | Local | Slurm | 关键特性 | 备注 |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **Feature** | ✅ | ✅ | ✅ | 支持 `dp eval-desc` (CLI) 与 Python API 双模式；自动批处理 | |
| **Train** | ✅ | N/A | ✅ | 多模型并行微调；自动生成 `0..N-1` 子目录与独立脚本 | 训练通常依赖 GPU 资源，Local 模式较少使用 |
| **Inference** | ✅ | ✅ | ✅ | 自动扫描模型目录；支持多 GPU 并行推理；集成基础误差分析 | Slurm 模式需外部编排等待 |
| **Collection** | ✅ | ✅ | ✅ | **核心模块**。Phase1: UQ/阈值/筛选; Phase2: DIRECT/2-DIRECT 采样; Phase3: 导出 | 支持 Joint Sampling (DIRECT only) |
| **Analysis** | ✅ | ✅ | N/A | 综合统计 (Parity/Error Dist); Cohesive Energy 分析; 自动生成 JSON/CSV 报表 | 通常在 Local 运行 |
| **CLI/Config** | ✅ | - | - | 统一入口 `dpeva`；支持 JSON 配置；相对路径解析 | |

## 4. 质量保障状态 (QA Status)

### 4.1 测试覆盖
- **Unit Tests (`tests/unit`)**: 覆盖率较高，包含 Feature, Inference, IO, Sampling, Submission, Training, UQ, Workflows 等核心模块。
- **Integration Tests (`tests/integration`)**:
  - 重点覆盖 Slurm 场景 (`slurm_multidatapool`) 与端到端流程 (`test_slurm_multidatapool_e2e.py`)。
  - 包含 Mock 数据 (`data/`) 用于模拟真实计算结果。
  - **风险**: 集成测试严重依赖 Slurm 环境，可能在无调度器的 CI 环境中被跳过或失败（"Fragile"）。

### 4.2 代码审计现状
- **审计报告滞后**: `docs/reports/2026-02-28-Code-Audit-Report.md` 中指出的“硬编码路径/单位”问题在 v0.4.6 代码中**已修复**（如 `visualization.py` 已使用 `os.path.join` 和常量）。报告内容已部分失效。

## 5. 已知问题与风险 (Verified)

经过对 v0.4.6 代码的深度核查，确认以下潜在风险与问题：

### 5.1 错误处理静默风险 (High)
- **现象**: `src/dpeva/__init__.py` 在导入时执行环境检查，但捕获了所有 `Exception` 并直接 `pass`。
- **影响**: 若 `dp` 命令缺失或版本不兼容，用户在 `import dpeva` 时不会收到任何警告或错误，直到运行时才崩溃。这违背了 "Errors should never pass silently" 的准则。
- **代码位置**: `src/dpeva/__init__.py:13-16`

### 5.2 默认值一致性风险 (Medium)
- **现象**: `SamplingManager` 内部对 `direct_thr_init` 的兜底默认值为 `0.1`，而全局常量 `DEFAULT_DIRECT_THR_INIT` 与 `CollectionConfig` 默认值为 `0.5`。
- **影响**: 若绕过 `CollectionConfig` 直接初始化 `SamplingManager`（如在脚本或测试中），可能导致采样行为与 CLI 运行不一致（簇数量差异巨大）。
- **代码位置**: `src/dpeva/sampling/manager.py:25` vs `src/dpeva/constants.py:57`

### 5.3 Slurm 后端覆写机制 (Architectural)
- **现象**: `CollectionWorkflow` 在生成 Slurm 脚本时，强制注入 `export DPEVA_INTERNAL_BACKEND=local`。
- **影响**: 这是一种工程妥协。Worker 节点实际上运行的是 "Local" 模式的代码，但通过 Slurm 脚本包裹。这增加了调试复杂度：用户在 Slurm 日志中看到的行为是 Local 模式的行为，且依赖环境变量隐式传递状态。
- **代码位置**: `src/dpeva/workflows/collect.py:95-99` & `361`

### 5.4 文档与代码版本脱节 (Low)
- **现象**: 审计报告指出的问题已修复但未更新报告状态；部分示例配置 (`examples/`) 仍包含特定集群的硬编码路径 (`/opt/envs/...`)。
- **影响**: 新手用户可能会被过期的审计报告误导，或在运行示例时因路径错误受阻。

## 6. 建议后续行动

1.  **Hotfix**: 修正 `__init__.py` 中的异常捕获逻辑，至少应打印 `UserWarning` 而非静默吞掉。
2.  **Refactor**: 统一 `SamplingManager` 的默认值来源，强制要求从 `constants.py` 获取，消除 `0.1` vs `0.5` 的歧义。
3.  **Docs**: 更新审计报告状态，标记已修复项；在 `examples/README.md` 中更显眼地提示用户修改环境路径。
4.  **CI**: 增强本地集成测试能力，减少对 Slurm 环境的强依赖，或引入 MockSlurm 机制。
