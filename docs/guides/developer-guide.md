---
title: Document
status: active
audience: Developers
last-updated: 2026-03-25
---

# DP-EVA 项目开发文档

- Status: active
- Audience: Developers
- Last-Updated: 2026-03-21
- Related:
  - 配置字段字典：/docs/source/api/config.rst
  - 校验规则补充：/docs/reference/validation.md
  - 上游软件与职责：/docs/reference/upstream-software.md

* **版本**: 0.7.1
* **生成日期**: 2026-04-01
* **作者**: Quantum Misaka with Trae SOLO

---

## 1. 项目概述 (Overview)

### 1.1 项目简介
DP-EVA (Deep Potential EVolution Accelerator, 深度势能演化加速器) 是一个面向 DPA 通用机器学习原子间势（Universal Machine Learning Interatomic Potential） 高效微调的自动化主动学习框架。该项目旨在通过智能化的数据筛选策略（合适不确定度 + 高结构代表性），从海量无标签数据中识别出最具价值的样本（”EVA适格者“），从而以最小的数据标注成本实现模型性能的最大化提升。

### 1.2 核心哲学 (The Zen of DP-EVA)
本项目遵循 Python 工程化最佳实践及 **Zen of Python** 哲学进行重构与维护：

#### 1.2.1 核心原则
*   **明确与简洁 (Explicit & Simple)**：优先选择清晰直观的实现，拒绝隐晦的技巧与魔法（如环境变量隐式控制）。
*   **可读性至上 (Readability Counts)**：代码首先是写给人看的。命名、缩进、结构需一目了然。
*   **实用与优雅 (Practicality Beats Purity)**：在理想设计与现实需求冲突时，选择实用方案；优雅是指恰到好处的平衡。
*   **一种最好 (There Should Be One Obvious Way to Do It)**：面对问题应有唯一明显的解决方案，避免提供多个功能重叠的接口。
*   **宽容但不纵容 (Errors Should Never Pass Silently)**：错误必须被显式捕获和处理。允许失败但必须给出清晰的异常信息。严禁使用裸 `except:` 吞掉异常，也不要在关键失败路径上仅记录日志而静默返回（Silent Return）。

#### 1.2.2 架构实践
*   **模块解耦 (Modular Design)**：将复杂的科研脚本拆解为职责单一的原子模块 (Training, Inference, Uncertainty, Sampling)。
*   **领域驱动设计 (Domain-Driven Design)**：v2.8.0+ 引入了领域驱动架构，将核心工作流拆分为 IO、Config、Execution 等独立服务层，大幅降低耦合度。
*   **数据标准化 (Data Standardization)**：引入标准化的 `PredictionData` 接口，替代不透明的遗留对象。
*   **双模调度 (Dual-Mode Scheduling)**：底层统一封装 `JobManager`，无缝支持 Local (Multiprocessing) 和 Slurm 集群环境。
*   **日志规范 (Logging Discipline)**：库代码不干预全局日志配置，确保日志输出清晰、无冗余且易于追踪。
*   **职责边界 (Boundary of Responsibility)**：上层业务逻辑（如 Joint Sampling 的数据切分）应收敛于 Manager/Service 层，严禁侵入式修改底层通用算法模块（如 `DIRECTSampler`）。保持底层模块的纯洁性与通用性（Ref: v0.4.2 n_candidates fix）。

### 1.3 优化方向与路线图 (Roadmap)
基于 Code Review 的建议，项目后续将重点关注以下方向：
1.  **全链路 DDD 重构**：目前 Training, Collection, Inference 模块已完成 DDD 改造，后续需将 Feature 模块迁移至相同架构 (IO/Config/Execution Managers)。
2.  **统一配置管理**：进一步强化 Pydantic 在所有配置类中的应用，消除字典传递，确保类型安全。
3.  **测试覆盖率**：提升集成测试的覆盖范围，特别是针对 Slurm 提交和异常处理的边界测试。
4.  **遗留代码清理**：逐步移除标记为 Deprecated 的单体类 (`ParallelTrainer` 等)，保持代码库轻量化。

### 1.4 开发流程标准 (Development Process Standard)
为确保代码质量与知识沉淀，所有开发活动必须严格遵循以下标准流程：

1.  **计划 (Plan)**: 
    *   在开始编码前，必须基于现状进行需求分析与技术方案设计。
    *   对于重大重构或新功能，需编写简要的 RFC (Request for Comments) 或 Design Doc。
2.  **执行 (Execute)**:
    *   代码编写应遵循 **Zen of Python** 原则。
    *   优先使用领域驱动设计 (DDD) 模式，避免创建上帝类。
    *   保持提交粒度适中，Commit Message 清晰。
3.  **验证 (Verify)**:
    *   **必须** 编写或更新单元测试，确保覆盖核心逻辑。
    *   在提交前运行所有相关测试，确保无回归 (Regression)。
4.  **文档化 (Document)**:
    *   **强制要求**: 开发完成后，必须同步更新本文档 (`developer-guide.md`) 的相关章节（如架构图、接口变更）。
    *   **技术细节**: 将详细的实现细节、配置参数字典、算法推导等内容沉淀至 `docs/reference/` 或 `docs/guides/` 下的专项文档中。
    *   **废弃清理**: 及时标记并清理过时的文档与代码。
    *   **治理入口**: 首次参与贡献请先阅读 `docs/guides/docs-governance-quickstart.md`。
    *   **提交前检查**: 执行 `python3 scripts/doc_check.py`、`python3 scripts/check_docs_freshness.py --days 90`、`make -C docs html SPHINXOPTS="-W --keep-going"`。


---

## 2. 系统架构 (Architecture)

### 2.1 目录结构
项目代码组织在 `src/dpeva` 包下，结构清晰：

```text
dpeva/
├── examples/recipes/       # [API范例] Python 调用示例 (Recipes)
│   ├── training/           # 训练脚本范例
│   ├── inference/          # 推理脚本范例
│   ├── collection/         # 采集脚本范例
│   └── ...
├── scripts/                # [项目维护] 自动化与 CI/CD 脚本 (CI/CD, Release, Audit)
│   ├── gate.sh             # 质量门禁入口
│   ├── audit.py            # 代码静态分析工具
│   ├── check_docs.py       # 文档一致性检查
│   └── release_helper.py   # 版本发布助手
├── tools/                  # [业务工具] DeepMD 数据处理与校验实用程序
│   ├── count_deepmd_system_frames.py
│   ├── verify_desc_consistency.py
│   └── ...
├── src/dpeva/
│   ├── cli.py              # [用户接口] 统一命令行入口 (dpeva)
│   ├── workflows/          # [核心] 业务流程编排层
│   │   ├── train.py        # 训练工作流 (TrainingWorkflow)
│   │   ├── infer.py        # 推理与分析工作流 (InferenceWorkflow)
│   │   ├── collect.py      # 数据采集工作流 (CollectionWorkflow) - 现已重构为编排器
│   │   └── feature.py      # 特征生成工作流 (FeatureWorkflow)
│   ├── services/           # [服务层] (概念上) 领域逻辑服务
│   │   ├── io.collection   # 数据IO服务 (CollectionIOManager)
│   │   ├── uncertain.manager # UQ编排服务 (UQManager)
│   │   └── sampling.manager  # 采样编排服务 (SamplingManager)
│   ├── training/           # 训练模块 (ParallelTrainer)
│   ├── inference/          # 推理模块 (StatsCalculator, Visualizer)
│   ├── uncertain/          # 不确定度模块 (UQCalculator, UQFilter, Visualizer)
│   ├── sampling/           # 采样模块 (DIRECT, PCA, Clustering)
│   ├── feature/            # 特征生成模块 (DescriptorGenerator)
│   ├── submission/         # 任务提交抽象层 (JobManager, JobConfig, Templates)
│   ├── io/                 # 数据读写辅助 (DPTestResultParser, PredictionData, types)
│   └── utils/              # 通用工具
└── test/                   # [开发专用] 单元测试与回归测试脚本
```

### 2.2 数据流图 (Data Flow)
```{mermaid}
graph TD
    %% Initial State
    InitTrain[Existing Training Set]
    TargetPool[Target Data Pool]
    BaseModel[Base Model]

    %% Parallel: Training & Inference & Feature
    subgraph Parallel_Execution
        direction TB
        BaseModel -->|Fine-tune| Ensemble[Ensemble Models]
        InitTrain -->|Fine-tune| Ensemble
        
        Ensemble -->|Inference| Preds[Target Pool Predictions]
        TargetPool -->|Inference| Preds
        
        TargetPool -->|Calc Feature| TargetFeat[Target Features]
        InitTrain -->|Calc Feature (Joint Only)| TrainFeat[Training Features]
    end

    %% Collection
    subgraph Collection_Workflow
        Preds -->|Uncertainty| UQ[UQ Metrics]
        UQ -->|Filter| Candidates[Candidate Structures]
        TargetFeat -->|Descriptor| Candidates
        TrainFeat -->|Descriptor| Candidates
        
        Candidates -->|Sampling (DIRECT)| Sampled[Sampled Samples]
        Candidates -->|Remaining| OtherPool[Other Data Pool]
    end
    
    %% Labeling
    subgraph Labeling_Workflow
        Sampled -->|FP Calc| Labeled[New Labeled Data]
    end

    %% Data Integration (New Feature)
    subgraph Data_Integration
        Labeled -->|Merge| NewTrain[Next Gen Training Set]
        InitTrain -->|Merge| NewTrain
    end
    
    %% Analysis (Standalone)
    subgraph Analysis_Module
        NewTrain -.->|Stats & Viz| Report[Dataset Report]
        Preds -.->|Stats & Viz| ModelReport[Model Performance Report]
    end

    NewTrain -->|Next Cycle| BaseModel

    %% Optional if no another TargetPool
    OtherPool -->|Next Cycle| TargetPool
```

### 2.3 包级导出层与主链路入口边界

为避免“模块存在但未接入主链路”的认知偏差，开发时需区分两类入口：

- **主链路入口（Execution Entry）**：由 `dpeva.cli` 子命令分发到具体 workflow 模块（如 `dpeva.workflows.collect`）。
- **包级导出层（Export Surface）**：`__init__.py` 仅用于命名空间聚合、对外导出和职责声明，不作为工作流执行入口。

实践约定：

- 新增业务功能时，先接入 workflow/manager 主链路，再考虑是否在包级导出层暴露。
- 包级 `__init__.py` 中不放业务编排逻辑与隐式副作用。
- 审查“是否接线”时，证据以 CLI -> workflow -> manager -> module 的可追溯调用链为准，而非仅凭 `__init__.py` 导出关系。

---

## 3. 核心模块详解 (Modules)

### 3.1 Training 模块 (`dpeva.training`)
负责管理 DeepMD 模型的并行训练任务。
*   **`TrainingWorkflow`**: 核心编排类 (Orchestrator)。基于 DDD 模式重构，协调以下 Manager 完成任务：
    *   **`TrainingIOManager`**: 负责工作空间管理与文件操作。
    *   **`TrainingConfigManager`**: 负责配置解析与随机种子管理。
    *   **`TrainingExecutionManager`**: 负责命令构建与作业提交。
*   **`ParallelTrainer`**: (已过时/Deprecated) 遗留类，保留以维持向后兼容性。
*   **特性**:
    *   自动工作目录隔离 (`0/`, `1/`, `2/`, `3/`)。
    *   支持 `OMP_NUM_THREADS` 自动配置。
    *   内置随机种子循环机制，确保多模型多样性。

### 3.2 Inference 模块 (`dpeva.inference`)
负责模型批量推理及后续的误差分析。
*   **`InferenceWorkflow`**: 
    *   自动扫描模型目录并提交 `dp test` 任务。
    *   **高级分析**: 内置 `StatsCalculator`，自动计算 RMSE/MAE。
    *   **相对能量分析**: 支持通过最小二乘法 (Least Squares) 拟合原子能量，计算 Cohesive Energy，从而在不同组分体系间进行公平比较。
    *   **可视化**: 自动生成 Parity Plot (能量/力) 和误差分布图。

### 3.3 Uncertainty & Sampling 模块 (`dpeva.uncertain`, `dpeva.sampling`)
这是主动学习的大脑，负责从海量数据中“淘金”。
*   **数据标准化 (`io.types.PredictionData`)**: 
    *   取代了旧版的 `DPTestResults` 遗留类。
    *   统一使用 `PredictionData` (Dataclass) 作为数据容器，包含 `energy`, `force`, `virial` 等标准字段。
*   **UQ 计算 (`UQCalculator`)**: 
    *   **QbC (Query by Committee)**: 计算多模型预测方差。公式：$\sigma_{QbC} = \sqrt{\sum_{i=x,y,z} Var(F_i)}$。
    *   **RND (Random Network Distillation)**: 计算当前模型与参考模型的偏差。公式：$\sigma_{RND} = \sqrt{\sum_{i=x,y,z} Mean((F_i^{pred} - F_i^{base})^2)}$。
    *   **数值稳定性 (Robustness)**: 实现了 **"Clamp-and-Clean"** 策略：
        *   **Clamp**: 强制方差计算结果非负 (`np.maximum(var, 0.0)`)，消除浮点误差导致的 RuntimeWarning。
        *   **Clean**: 自动检测 `NaN` 输出并将其替换为 `Infinity`（最大不确定度），确保异常模型预测会被标记为 `Failed` 而非被忽略。
        *   **Robust Scaling**: 手动实现了抗 Inf 的 Robust Scaling 算法，仅基于有限值计算统计量，保留 `Inf` 的极端属性。
    *   **自动阈值 (Auto-Threshold)**: 基于 KDE (核密度估计) 自动识别不确定度分布峰值，自适应确定 `trust_lo`。
*   **筛选策略 (`UQFilter`)**: 支持 `strict`, `tangent`, `circle` 等多种 2D 边界筛选算法。
*   **DIRECT 采样 (`DIRECTSampler`)**: 
    *   **联合采样 (Joint Sampling)**: 支持同时加载训练集和候选集，在联合特征空间中进行覆盖度最大化采样，避免新样本与旧样本重复。
    *   **基于聚类**: 使用 BIRCH 聚类算法在 PCA 降维后的空间中寻找最具代表性的样本点。
    *   **归一化策略**: 对结构描述符（原子描述符均值）进行 **L2 归一化**，随后进行 Z-score 标准化。
*   **2-DIRECT 采样 (`TwoStepDIRECTSampler`)**:
    *   **两步策略**: 先基于结构描述符进行粗粒度聚类 (Step 1)，再对每个结构簇内的原子环境进行细粒度聚类 (Step 2)。
    *   **原子级筛选**: 在 Step 2 中支持基于原子数量 (`smallest`) 等策略选择最具代表性的结构，有效降低标注成本。
    *   **归一化策略 (Normalization)**:
        *   **Step 1 (Structure)**: 采用 L2 归一化 + Z-score 标准化。结构描述符通过对原子描述符取平均并进行 L2 归一化生成，保留了反映原子环境一致性的模长特征。
        *   **Step 2 (Atomic)**: 仅采用 Z-score 标准化，**不进行 L2 归一化**。实测表明，保留原子描述符的原始模长（蕴含局部环境复杂度信息）能显著提升聚类效果和最终模型性能。

### 3.4 Feature 模块 (`dpeva.feature`)
负责生成原子结构的描述符。
*   **`DescriptorGenerator`**:
    *   **CLI 模式**: 调用 `dp eval-desc` 命令，支持 Slurm 提交。
    *   **Python 模式**: 直接调用 `deepmd.infer` API，适合小规模或调试使用。
    *   **数据一致性**: 统一使用 `dpeva.io.dataset.load_systems` 加载数据，确保在 `mixed` 格式下与 CLI 模式行为一致。
    *   **单数据池兼容**: 智能识别 `desc_dir/System.npy` 格式的描述符文件，兼容单数据池模式下的平铺结构。
    *   **多数据池支持**: 递归支持 `Dataset/System` 3层结构，通过增强的路径解析逻辑确保描述符与数据一一对应。

### 3.5 Submission 模块 (`dpeva.submission`)
统一的任务提交抽象层。
*   **`JobManager`**: 屏蔽 Local/Slurm 差异。
*   **`JobConfig`**: 强类型的作业配置类，支持 Partition, QoS, GPUs 等 Slurm 高级参数。
*   **`TemplateEngine`**: 基于模板生成作业脚本，易于扩展和定制。

### 3.6 Slurm Backend 设计 (Slurm Architecture)

DP-EVA 专为高性能计算 (HPC) 环境设计，其 Slurm 后端支持以下关键特性：

*   **双模调度 (Dual-Mode)**: 通过 `JobManager` 统一封装，代码逻辑无缝切换 Local/Slurm 模式。
*   **并行投作业 (Parallel Submission)**: 
    *   **Training**: 训练阶段，每个模型（如 4 个 Ensemble 模型）会被分配独立的 Slurm 作业 (`train.slurm`)，从而在集群中并行训练，极大缩短总耗时。
    *   **Inference**: 推理阶段 (v0.4.5+)，每个模型的测试任务 (`dp test`) 同样被封装为独立的 Slurm 作业 (`run_test.slurm`)，实现多模型并行推理。
*   **一任务一作业 (One-Task-One-Job)**: 摒弃了将所有任务打包进单一作业的串行模式，确保每个子任务都能独占申请到的计算资源（如 GPU），避免资源争抢和效率瓶颈。
*   **状态监控**: 所有 Slurm 作业在完成后会输出 `DPEVA_TAG: WORKFLOW_FINISHED` 标记，便于自动化工具监控任务状态。

---

## 4. 接口使用指南 (Interface Guide)

本项目提供两种使用方式：**统一命令行 (CLI)**（推荐）和 **Python API**（高级定制）。

### 4.1 统一命令行 (CLI)
自 v2.6.0 起，项目引入统一的 `dpeva` 命令，支持子命令模式。

**安装**:
```bash
pip install -e .
# 验证安装
dpeva --help
```

**通用用法**:
```bash
dpeva <subcommand> <config_path>
```

#### 4.1.1 训练 (Train)
```bash
dpeva train config_train.json
```

#### 4.1.2 推理与分析 (Infer)
```bash
dpeva infer config_test.json
```

#### 4.1.3 数据采集 (Collect)
```bash
dpeva collect config_collect.json
```

#### 4.1.4 特征生成 (Feature)
```bash
dpeva feature config_feature.json
```

#### 4.1.5 分析 (Analysis)
```bash
dpeva analysis config_analysis.json
```

`analysis` 支持两种模式：

- `model_test`：读取 `result_dir` 分析推理结果（默认）。
- `dataset`：读取 `dataset_dir` 直接统计数据集并输出分布图与统计表。

#### 4.1.6 标注后自动整合 (Label Integration)

在 `label` 配置中启用 `integration_enabled=true` 后，工作流会在 `collect_and_export` 结束后自动执行数据整合：

- 输入：`outputs/cleaned` + 可选 `existing_training_data_path`
- 输出：`merged_training_data_path`（未指定时默认 `outputs/merged_training_data`）
- 统计：输出 `integration_summary.json`

### 4.2 Python API (Recipes)
对于需要动态生成配置或集成到复杂 Python 流程中的场景，可以直接调用 `dpeva.workflows` 中的 Workflow 类。
示例脚本位于 `examples/recipes/` 目录下。

**示例 (Training)**:
```python
from dpeva.workflows.train import TrainingWorkflow
from dpeva.utils.config import resolve_config_paths
import json

# 1. 加载配置
with open("config.json") as f:
    config = json.load(f)
config = resolve_config_paths(config, "config.json")

# 2. 初始化并运行
workflow = TrainingWorkflow(config)
workflow.run()
```

### 4.3 配置参数说明
详细的输入参数定义、类型约束及验证规则，请参阅 API 文档：

*   **参数列表**: [config.rst](https://github.com/QuantumMisaka/dpeva/blob/main/docs/source/api/config.rst)
*   **验证规则**: [validation.md](https://github.com/QuantumMisaka/dpeva/blob/main/docs/reference/validation.md)

以下仅展示标准 JSON 配置文件的基本结构概览。

#### 4.3.1 训练配置 (Train Config)
**参考文件**: `examples/recipes/training/config_train.json`
```json
{
    "project": "./training_task",
    "base_model_path": "frozen_model.pb",
    "num_models": 4,
    "training_mode": "cont",
    "model_head": "ferro",
    "input_json_path": "input.json",
    "training_data_path": "data",
    "submission": {
        "backend": "local",
        "omp_threads": 4
    }
}
```

#### 4.3.2 其他配置
请参考 `examples/recipes/` 目录下的示例文件以及上述 API 文档。

---

### 4.4 UQ 与采样配置要点

本节补充 UQ 与采样相关的关键配置语义；完整字段列表与约束以 `/docs/reference/*` 为准。

#### 4.4.1 `uq_trust_mode` 配置说明

- `auto`：自动计算信任区边界（推荐默认）
- `manual`：手动指定信任区边界（需提供对应阈值参数）
- `no_filter`：不启用信任区筛选

参考：

- `/docs/source/api/config.rst`
- `/docs/reference/validation.md`

#### 4.4.2 Auto-UQ 边界控制 (`uq_auto_bounds`)

Auto-UQ 用于根据数据分布自动确定筛选边界；具体的字段与约束以 Reference 文档为准，并在后续迭代中建议沉淀为独立章节以避免与算法实现脱节。

参考：

- `/docs/source/api/config.rst`
- `/docs/reference/validation.md`

#### 4.4.3 采样参数说明 (Sampling)

采样相关参数（DIRECT/2-direct/joint 等）属于收集工作流的核心可调维度，建议：

- 只在 `/docs/source/api/config.rst` 维护字段字典与默认值
- 在指南中仅描述“如何选择参数组”的经验规则，并引用对应字段

参考：

- `/docs/source/api/config.rst`

#### 4.4.4 Collection 出图分层开关 (`enable_diagnostic_plots`)

- `false`（默认）：仅输出 Core 层图，Diagnostic 层 parity 图默认跳过
- `true`：在满足 `has_gt`、`uq_rnd_rescaled` 等前置条件时输出 Diagnostic 层图像
- 运行日志会输出统一标签：
  - `[COLLECT_PLOT_GENERATED]`：记录已生成图像、层级与触发条件
  - `[COLLECT_PLOT_SKIPPED]`：记录被跳过图像及 reason
  - `[COLLECT_PLOT_AUDIT]`：汇总 generated/skipped 清单，便于回溯审计

## 5. 开发与测试 (Development)

### 5.1 代码规范
*   **日志**: 禁止在 `src/dpeva` 库文件中调用 `logging.basicConfig()`。仅在 `runner` 脚本中配置全局日志。
*   **路径**: 所有文件操作应使用绝对路径 (`os.path.abspath`)。
*   **异常**: 显式捕获并记录异常，避免静默失败。
*   **数据接口**: 使用 `dpeva.io.types.PredictionData` 传递预测结果，禁止传递裸字典。
*   **变量命名**: 统一输入数据路径变量名为 `data_path` (Feature/Inference Workflow)。

### 5.2 验证测试

在开发阶段，你可以通过`conda activate dpeva-dev` 加载所需环境，并在项目目录下运行 `pip install --upgrade .` 更新开发依赖，确保测试环境与开发环境保持一致。

用户自行开展单元测试时，需要自行配置好 Python 环境，确保 `dpeva` 命令在环境内并处于最新状态，且 `pytest` 已安装。

*   **运行单元测试 (Unit Tests)**:
    ```bash
    # 基础运行
    pytest tests/unit
    
    # 带覆盖率报告的运行 (推荐)
    pytest tests/unit --cov=dpeva --cov-report=term-missing
    ```
    *   **规范**:
        *   **Mock 外部依赖**: 所有对 `dp`, `dpdata`, `slurm` 的调用必须被 Mock，严禁在单元测试中产生实际的文件 I/O 或进程提交。
        *   **日志验证**: 涉及日志输出的逻辑，需验证 `setup_workflow_logger` 是否被正确调用。
        *   **异常覆盖**: 必须覆盖所有 `try-except` 分支，确保异常被正确捕获和记录。
    *   **覆盖范围**: 核心算法 (UQCalculator, UQFilter, DIRECTSampler) 的逻辑验证。
    *   **测试策略**: 
        *   **Golden Value**: 与 NumPy 手算结果比对，误差容忍度 < 1e-5。
        *   **边界测试**: 覆盖 NaN, Inf, 空数据, 单点数据等极端场景。
        *   **覆盖率要求**: 核心模块行覆盖率需达到 100%。

*   **运行兼容性测试 (Compatibility Test)**:
    ```bash
    cd test
    python run_compat_test.py
    ```
    此脚本会自动验证单数据池 (Single Pool) 和多数据池 (Multi Pool) 在普通模式与联合模式下的运行正确性。
    *   **描述符一致性**: 已通过 `verify_desc_consistency.py` 验证 CLI 模式下 NPY/Mixed 格式生成结果的一致性 (Diff < 1e-5)。

*   **运行 Auto-UQ 测试**:
    ```bash
    cd test/verification_test_run
    python run_auto_uq_test.py
    ```
    验证 KDE 阈值计算逻辑及可视化图表。

### 5.3 常见问题 (FAQ)

**Q: 单数据池模式下提示 "Found descriptor via basename fallback" 是什么意思？**
A: 旧版本会产生此 Warning，**新版本 (v2.2.0)** 已将其优化为 Info 级别的兼容性提示。这表示系统自动通过 `System.npy` 文件名匹配到了嵌套在 `Dataset/System` 路径下的结构数据，属于正常行为。

**Q: 为什么联合采样导出的样本数少于 `direct_n_clusters`？**
A: 这是预期行为。在联合模式下，`direct_n_clusters` 定义的是特征空间的总覆盖目标。如果某些区域已经被现有训练集覆盖，DIRECT 算法就不会再重复采样，从而节省标注成本。

### 5.4 工作流监控标准 (Workflow Monitoring)

为了支持自动化工作流编排（如使用 Airflow 或自定义 Scheduler），本项目统一了任务完成的日志标记。所有核心 Workflow 在成功执行完毕后，均会在标准输出或日志文件末尾打印以下 Tag：

```text
DPEVA_TAG: WORKFLOW_FINISHED
```

**监控建议**:
*   外部调度系统应通过 `grep` 或正则表达式持续监控任务的 Log 文件（如 `train.log`, `collection.log`, `eval_desc.log` 或 Slurm `.out` 文件）。
*   一旦检测到该 Tag，即可判定当前步骤已从应用层逻辑上成功结束，可以安全触发后续任务。

---

## 6. 版本修订记录 (Revision History)

### 6.1 维护策略 (Maintenance Policy)
为确保文档的可读性与追溯性，本项目的版本记录遵循以下原则：

1.  **历史归档 (Historical Archive)**：
    *   对于重大版本重置（如 v2.x -> v0.4.x）之前的历史记录，不再保留详细条目，仅进行概要性总结。
2.  **当前纪元 (Current Era)**：
    *   当前主版本系列（v0.4.x）必须保持**完整**的记录链条。
    *   **追加式记录 (Append-only)**：新版本记录应始终添加在列表顶部，严禁覆盖或修改旧版本的历史条目。
    *   **中间版本找回**：若因误操作导致记录丢失，必须基于 git log 进行回溯与补全。

### 6.2 版本历史

#### **Current Era (v0.7.x)**

*   **v0.7.1** (2026-04-01):
    *   **[特性]** 提升 Parity Plots 可读性：全局基础字号由 12 增大至 16，同步调大统计框 (Stats Box) 字号。
    *   **[特性]** 优化图像元素尺寸：增大 Scatter 散点和 Hexbin 六边形大小，Hexbin `gridsize` 由 60 降至 50 以突出重点。
    *   **[修复]** 解决布局重叠：将 Error Density 顶部刻度标签向上微调 (`error_tick_pad` 1.0)，防止与曲线重叠。
    *   **[修复]** 统一出图尺寸：移除 `bbox_inches="tight"`，严格遵循预设 `figsize`，确保跨图视觉一致性。
    *   **[修复]** 消除 KDE 伪影：在绘制误差分布前对离群点进行过滤，修复由于网格稀疏导致的“倒三角”尖角伪影。
    *   **[特性]** 新增多数据池采样总结图 (`Final_sampled_PCAview_by_pool`)，支持多 Pool 场景下的特征覆盖可视化。
    *   **[特性]** 引入出图分层控制 (`enable_diagnostic_plots`)，默认仅生成核心图表，减少冗余产出。
    *   **[特性]** 接入 Candidate Parity 图，支持在采样阶段对候选集进行误差预审。
    *   **[治理]** 完成 `v0.7.1` 周期文档归档，同步更新归档索引与包级模块文档。
    *   **[发布]** 版本升级至 `0.7.1`，同步 `__init__`、`README`、Sphinx `conf.py` 与 Banner 版本标识。

*   **v0.7.0** (2026-03-21):
    *   **[特性]** 新增基于推理误差阈值的数据清洗工作流，支持灵活的数据质量控制。
    *   **[特性]** 为分析工作流添加 Slurm 后端支持，增强分布式计算能力。
    *   **[特性]** 新增可视化性能分级与慢图告警机制，提升分析效率。
    *   **[治理]** 按照 Ruff 规范完成全项目代码质量治理，修复 F401/E701/E702/E402 等 lint 问题，提升代码整洁度与可维护性。
    *   **[测试]** 单元测试 276 个全部通过，Ruff 检查全部通过。
    *   **[发布]** 版本升级至 `0.7.0`，同步 `__init__`、`README` 与 Banner 版本标识。

*   **v0.6.9** (2026-03-18):
    *   **[特性]** 新增收敛数据查看与清洗工具，提升数据预处理效率。
    *   **[特性]** 重构分析工作流并增强可视化能力，支持更丰富的统计图表。
    *   **[修复]** 修复分布图图例重叠问题，并针对 Parity 图进行可读性增强（调整 Alpha 值与 Marker 大小）。
    *   **[修复]** 增强 `Collection` 模块，支持多数据池描述符目录的递归加载与匹配。
    *   **[治理]** 按照文档治理规范完成 v0.6.9 版本文档归档，同步更新索引。
    *   **[发布]** 版本升级至 `0.6.9`，同步元数据与版本标识。

*   **v0.6.8** (2026-03-17):
    *   **[架构]** 重构 Inference 与 Analysis 边界：`InferenceConfig` 新增 `auto_analysis` 显式开关，local 场景可链式触发 analysis，slurm 场景保持解耦执行。
    *   **[修复]** Analysis 增加 `results_prefix` 契约并贯通读取链路，消除 `head="results"` 写死导致的前缀不一致风险。
    *   **[重构]** 新增 `dpeva.postprocess` 共享后处理入口，统一导出统计与可视化能力，降低 analysis/inference 语义耦合。
    *   **[治理]** 完成 Analysis 相关计划/报告/规格文档归档至 `docs/archive/v0.6.8/`，同步更新 active 索引与 archive 索引。
    *   **[测试]** 回归通过 `tests/unit/workflows/test_infer_workflow_exec.py`、`tests/unit/workflows/test_analysis_workflow.py` 及 analysis/inference 单测集。

*   **v0.6.7** (2026-03-14):
    *   **[修复]** 调整 Collection 目标池无标签判据：在 `<1e-4` 阈值下，只要任意帧能量标签近零即按无 GT 处理，并在流程日志中增加 WARNING 提示。
    *   **[修复]** 修复 Collection 导出路径重复嵌套问题，统一 UQ+DIRECT 与 DIRECT-only 导出链路，避免 `other_dpdata/other_dpdata` 与错位目录结构。
    *   **[修复]** 修复 Labeling 分支统计错配，增强 metadata 缺失/损坏场景下的归属回退与一致性校验，确保分支统计与全局统计可对齐。
    *   **[特性]** Labeling 新增 extract 阶段能力并完善 BAD_CONVERGED 分流，支持缺失 `TOTAL-FORCE` 的坏收敛任务隔离与可追溯分类。

*   **v0.6.6** (2026-03-14):
    *   **[修复]** Collection 新增 `UQ-force-qbc-rnd-fdiff-scatter` 工作流调用，修复有真值场景下该图未输出的问题。
    *   **[可视化]** 为 `UQ-force-qbc-rnd-fdiff-scatter` 增加 Truncated `[0,2]` 出图分支，补齐超界数据截断逻辑与独立产物。
    *   **[测试]** 增强 `test_visualization.py` 对 fdiff 散点图输出契约断言，并补充 truncated 分支测试覆盖。
    *   **[修复]** 改进 `SamplingManager._transform_background_features` 对 mock 属性维度判断的鲁棒性，修复单测中的维度误判失败。
    *   **[修复]** Labeling 后处理新增“坏 converged”防护：电子收敛但缺失 `TOTAL-FORCE` 的任务会被识别并隔离，避免进入 postprocess 触发 `forces[0]` 越界。
    *   **[特性]** Labeling 新增 `--stage extract` 阶段，支持从 `inputs/N_*` 独立提取结果并分流到 `CONVERGED`/`BAD_CONVERGED`，实现与 execute/postprocess 解耦。
    *   **[稳定性]** `AbacusPostProcessor` 新增任务状态分类与数据完整性校验，`compute_metrics` 增强系统级/帧级容错，单坏样本不再中断整批流程。
    *   **[测试]** 补齐 CLI、Workflow、Manager、Postprocess 回归测试，`pytest tests/unit` 全量通过（231 passed）。
    *   **[治理]** 将 `docs/plans/governance` 下已完成计划归档至 `docs/archive/v0.6.6/plans/`，同步更新归档索引与治理入口链接。
    *   **[发布]** 版本升级至 `0.6.6`，同步 `__init__`、`README`、Sphinx `conf.py` 与 Banner 版本标识。

*   **v0.6.5** (2026-03-12):
    *   **[修复]** 修复 Labeling integration 中 `atom_names` 顺序敏感误报，支持在元素集合一致时自动归一化顺序并完成合并。
    *   **[重构]** 完成 Labeling workflow 三阶段解耦，提供 `run_prepare/run_execute/run_postprocess` 显式阶段入口并保留默认全流程编排。
    *   **[增强]** `dpeva label` 新增 `--stage all|prepare|execute|postprocess` 阶段化执行能力，支持不重算 ABACUS 的后处理复用。
    *   **[稳定性]** 增强 prepare 幂等性，执行前自动重置 `inputs` 工作区，消除重复执行时历史残留导致的任务打包冲突。
    *   **[可观测性]** 三阶段日志独立落盘为 `labeling_prepare.log`、`labeling_execute.log`、`labeling_postprocess.log`，提升排障效率。
    *   **[测试]** 新增并补强 integration、workflow、CLI、manager 相关单测，`pytest tests/unit` 全量通过。
    *   **[特性]** 为 CLI 所有子命令实现配置文件的 Schema 前置校验，在工作流启动前即拦截无效参数。
    *   **[特性]** Labeling 集成阶段新增 `output_format` 参数，支持自定义合并后的数据集格式（默认 `deepmd/npy`）。
    *   **[文档]** 重构 `examples` 目录结构，移除冗余层级，并更新对应的 `README` 与用户指南。
    *   **[可视化]** 更新品牌视觉资产（Logo），优化 PCA 绘图配色（#FFC000/#6A5ACD）及采样可视化逻辑。
    *   **[文档]** 更新 README 展示新标识，新增可视化改进实施计划。


*   **v0.6.4** (2026-03-11):
    *   **[修复]** 完成 R01-R37 全量闭环，修复安全、稳定性、测试与文档一致性问题。
    *   **[重构]** 拆分 Labeling/Collect/Analysis 工作流关键长函数，降低职责耦合并增强可测试性。
    *   **[测试]** 新增并补强 strategy、visualization、analysis、collect 相关回归测试，保持 unit/integration 全绿。
    *   **[治理]** 完成计划文档与评审报告归档至 `docs/archive/v0.6.4/`，同步更新活动索引与归档索引。
    *   **[发布]** 版本号升级至 0.6.4，并同步 README 与 Sphinx 版本标识。

*   **v0.6.3** (2026-03-10):
    *   **[文档]** 重构归档目录结构并完善文档生命周期规范，确保历史文档有序归档。
    *   **[治理]** 完善文档治理结构并新增快速上手指南，优化用户体验。
    *   **[修复]** 修复文档链接策略并完成治理稳态化收尾，消除构建警告。
    *   **[特性]** 文档支持 Mermaid 图表渲染，增强技术架构的可视化表达能力。
    *   **[构建]** 更新文档构建依赖和配置，提升构建稳定性和效率。
    *   **[依赖]** 将 `ase-abacus` 插件纳入核心依赖，并添加导入检查，优化 Labeling 模块的用户体验与稳定性。
    *   **[CI/CD]** Warning修复，添加 `docs/source/_static` 目录下的 `.gitkeep` 文件，避免 Sphinx 构建警告。并在 `doc-lint.yml` 中添加对 `ase-abacus` 的安装。


*   **v0.6.2** (2026-03-10):
    *   **[质量]** 全面验证并修复了单元测试和集成测试，确保所有用例 100% 通过（167/167 Unit, 7/7 Integration）。
    *   **[文档]** 对变量说明文档进行了全面审查与修复，补充了 `config.py` 中大量字段的 docstring，使其满足自动化文档生成的质量要求。
    *   **[CI/CD]** 建立了严格的文档质量门禁 (`docs-check.yml`) 和结构检查流程 (`doc-lint.yml`)，确保文档零警告构建。
    *   **[治理]** 更新了文档技能与维护规范，确立了 `config.py` 作为配置说明单一可信源 (SSOT) 的地位。

*   **v0.6.1** (2026-03-09):
    *   **[部署]** 新增 GitHub Actions 自动部署流程，支持将文档发布至 GitHub Pages，并实现多版本管理。
    *   **[配置]** 修正 Sphinx 语言配置为中文 (`zh_CN`)，优化文档搜索体验。
    *   **[治理]** 修复了文档系统中的绝对路径链接问题，消除了构建过程中的 Cross-reference 警告。
    *   **[文档]** 新增 `docs/guides/deployment.md` 部署指南，详细说明了 CI/CD 架构与配置策略。

*   **v0.6.0** (2026-03-09):
    *   **[治理]** 全面实施文档系统治理 (Docs Governance)，重构了 `docs/` 目录结构，建立 Guides/Reference/Plans 分层体系。
    *   **[自动化]** 引入 Sphinx 文档构建系统，支持从源码自动生成 API 文档，实现 "Code as Source of Truth"。
    *   **[规范]** 废弃静态 `config_schema.md`，转为动态生成的 API Reference，消除文档脱节风险。
    *   **[质量]** 引入 `check_docs_freshness.py` 并集成至 CI，强制监控文档时效性。
    *   **[重构]** `AGENTS.md` 转为面向 Agent 的开发指南，`README.md` 瘦身聚焦项目概览。

*   **v0.5.3** (2026-03-08):
    *   **[增强]** Labeling Workflow 实现了智能的单/多数据池（Single/Multi-Pool）识别逻辑，自动适配 `inputs` 目录结构。
    *   **[特性]** `task_meta.json` 新增 `system_name` 字段，实现了 `Dataset -> System -> Task` 的全链路数据追溯。
    *   **[统计]** 实现了分层统计报告（Global -> Dataset -> Type），涵盖 Total, Converged, Failed 及 Cleaned 任务的详细计数。
    *   **[测试]** 建立了 Labeling 模块的完整单元测试体系，覆盖核心算法与业务逻辑，通过率 100%。
    *   **[文档]** 完成了 Labeling 模块的功能清单与任务追踪文档，并已归档至 `docs/archive/specs/v0.6.0/`。

*   **v0.5.2** (2026-03-07):
    *   **[增强]** Labeling Workflow 新增分层统计（Global/Dataset/Type）和元数据注入机制，确保任务在打包和移动后仍能追溯其数据来源和结构类型。
    *   **[修复]** 解决了 Labeling 任务计数重复（Double Counting）问题，修复了 `dpdata.MultiSystems` 在结果收集时的初始化错误。
    *   **[优化]** `AbacusGenerator` 实现了结构分析逻辑 (`StructureAnalyzer`) 的彻底解耦，提升了代码的可测试性和模块化程度。
    *   **[配置]** 修复了 `output_dir` 配置被硬编码覆盖的缺陷，现在能正确尊重用户配置。

*   **v0.5.1** (2026-03-05):
    *   **[增强]** 在 `CollectionWorkflow` 中新增了 UQ 统计日志输出。现在 `UQManager` 会在不确定度分析阶段自动计算并打印 QbC、RND 及 Rescaled RND 的详细统计量（Count, Mean, Std, Percentiles），大幅提升了筛选过程的可观测性。
    *   **[文档]** 全面补全了 Labeling 模块的文档，包括 CLI 指南、配置 Schema 及校验规则。
    *   **[修复]** 修正了 `LabelingConfig` 中 `cleaning_thresholds` 字段的类型定义，允许使用 `null` 跳过特定物理量的检查，修复了 Pydantic v2 下的验证错误。
    *   **[治理]** 清理并归档了过时的项目状态报告和设计规范，更新了文档治理矩阵。

*   **v0.5.0** (2026-03-05):
    *   **[特性]** 全新发布 Labeling 模块 (`src/dpeva/labeling`)，支持从 `dpdata` 结构数据到 DFT 输入文件的全自动生成、任务打包与提交。
    *   **[重构]** 引入 `TaskPacker` 机制，支持将数万个微小 DFT 任务打包为少量 Slurm 作业，显著降低调度器压力。
    *   **[增强]** 实现了智能的 `ResubmissionStrategy`，支持基于参数梯度（如 `mixing_beta` 衰减）的自动收敛修正。
    *   **[数据]** 新增 `AbacusPostProcessor`，支持内聚能（Cohesive Energy）计算、自动参考能量拟合及多维度数据清洗。
    *   **[配置]** 在 `config.json` 中显式暴露 `ref_energies` 和 `cleaning_thresholds`，遵循“显式优于隐式”原则。
    *   **[架构]** 完成 Labeling 模块的 DDD 架构落地，通过 `LabelingManager` 和 `LabelingWorkflow` 实现全链路编排。

#### **Legacy Era (v0.4.x)**
*(2026-02-14 ~ 2026-03-04)*

在此阶段，项目完成了核心架构的稳固与工具链的完善。主要成就包括：
*   **v0.4.10 - v0.4.12**: 修复了安全漏洞，增强了 `CollectionWorkflow` 的异常检测，并引入了 `audit.py` 代码审计工具。
*   **v0.4.1 - v0.4.9**: 确立了领域驱动设计 (DDD) 架构，完成了 Inference、Collection 和 Training 工作流的重构，并实现了 Slurm 后端的双模调度与并行优化。

#### **Legacy Era (v2.x)**
     *   **[修复]** 增强了 `CollectionWorkflow` 的异常检测机制。在所有候选系统导出失败（Zero-Export）的极端情况下，显式抛出 `RuntimeError` 并输出 Critical 日志，防止任务“静默成功”导致下游流水线异常。
     *   **[文档]** 优化了 `release-helper` 技能文档，明确了基于 Git Log 的变更回溯与追加式文档更新流程，废弃了过时的 CHANGELOG 维护方式。
 
 *   **v0.4.11** (2026-03-03):
     *   **[优化]** 增强了 `dpeva.io.dataset.load_systems` 的智能加载逻辑。现在能优先识别单系统目录，避免了将 `set.000` 等内部数据文件夹误判为独立系统，消除了大量虚假的 Warning 日志。
     *   **[重构]** `CollectionWorkflow` 引入了基于文件结构的原子解析逻辑（File Structure-Based Atom Parsing），通过计算力文件与能量文件的行数比例精确推导原子数，彻底解决了非化学式命名系统的解析难题。
     *   **[增强]** 实现了基于 `testdata_dir` 的双重验证机制，在力文件解析失败时可自动回退到原始数据集查找原子数，显著提升了系统的鲁棒性。
 
 *   **v0.4.10** (2026-03-04):
     *   **[安全]** 修复了 P0/P1 级安全漏洞，包括路径穿越、命令注入和异常吞没问题。
     *   **[质量]** 引入 `scripts/gate.sh` 质量门禁和 `audit.py` 代码审计工具。
     *   **[规范]** 重构目录结构，明确区分 `scripts/` (项目维护) 与 `tools/` (业务工具)。
     *   **[修复]** 修正了 CLI 退出码契约和完成标记语义。
 
 *   **v0.4.9** (2026-03-04):
     *   **[修复]** 修复了日志重复输出到根记录器 (Root Logger) 的问题，确保日志流清晰。
 
 *   **v0.4.8** (2026-03-03):
     *   **[工具]** 重构 `verify_desc_consistency.py`，增强描述符一致性校验能力。
     *   **[发布]** `release-helper` 支持自动更新 README 中的版本徽章。
 
 *   **v0.4.7** (2026-02-28):
    *   **[测试]** 新增 Slurm Backend 的端到端集成测试，在真实 Slurm 环境中验证了作业提交、运行和日志生成的完整闭环。
    *   **[验证]** 强化了单元测试覆盖率，增加了防退化测试。
    *   **[修复]** 解决了集成测试在 Slurm 节点间的文件系统隔离问题。

*   **v0.4.5** (2026-02-27):
    *   **[并行]** 修复了 `InferenceWorkflow` 在 Slurm 后端下的并行提交逻辑。
    *   **[重构]** 移除了 `infer.py` 中的“自提交”机制，改为由 Manager 直接生成并行任务。
    *   **[性能]** 显著提升了多模型推理在集群环境下的吞吐量。

*   **v0.4.4** (2026-02-27):
    *   **[日志]** 全面重构日志系统，引入标准化的 `setup_workflow_logger`。
    *   **[修复]** 修复了 `TrainingWorkflow` 日志错乱且屏蔽 stdout 的严重缺陷。
    *   **[规范]** 为所有 Workflow 定义了专属的日志文件名常量。

*   **v0.4.3** (2026-02-25):
    *   **[架构]** 重构 `AnalysisWorkflow` 和 `CollectionWorkflow`，统一了底层分析逻辑。
    *   **[重构]** 引入 `UnifiedAnalysisManager`，消除代码重复。
    *   **[功能]** `AnalysisWorkflow` 支持稳健的原子成分加载。

*   **v0.4.2** (2026-02-23):
    *   **[修复]** 修复了 `DIRECTSampler` 在 Joint Sampling 模式下的参数错误。
    *   **[重构]** 采用 Post-Filtering 策略，将过滤逻辑移回 Manager 层。
    *   **[测试]** 全面优化测试套件，移除对临时目录的硬编码依赖。

*   **v0.4.1** (2026-02-14):
    *   **[版本]** 版本号重置并统一为 0.4.1，与 PyPI/Package 版本保持一致。
    *   **[架构]** 完成了 `InferenceWorkflow` 的 DDD 重构。
    *   **[文档]** 更新开发者文档，补充了基于 Zen of Python 的项目哲学。

#### **Legacy Era (v2.x)**
*(2026-01-28 ~ 2026-02-14)*

在此阶段，项目完成了从单一脚本到模块化架构的蜕变。主要成就包括：
*   **v2.1 - v2.7**: 引入了 Auto-UQ、联合采样 (Joint Sampling)、2-DIRECT 两步聚类等核心算法特性，并建立了统一的 `dpeva` CLI 接口。
*   **v2.8 - v2.9**: 确立了领域驱动设计 (DDD) 架构，完成了 Collection 和 Training 工作流的彻底解耦与重构，为 v0.4.x 的稳定迭代奠定了基础。
