# DP-EVA 项目设计模式与架构分析报告

## 1. 设计模式识别与分类 (Design Patterns Identification)

本项目在核心架构中应用了多种经典设计模式，有效支撑了主动学习流程的复杂性和灵活性。

### 1.1 创建型模式 (Creational Patterns)

#### **1. 建造者模式 (Builder Pattern)**
*   **实现位置**: [`src/dpeva/utils/command.py`](/src/dpeva/utils/command.py) - `DPCommandBuilder` 类
*   **代码示例**:
    ```python
    class DPCommandBuilder:
        @classmethod
        def train(cls, input_file: str, finetune_path: Optional[str] = None, ...) -> str:
            cmd = f"{cls._get_base_cmd()} train {input_file}"
            if finetune_path: cmd += f" --finetune {finetune_path}"
            return cmd
    ```
*   **解决场景**: DeepMD-kit 的命令行参数繁多且组合复杂（如 Backend 切换、Init Model、Freeze 选项）。Builder 模式将复杂的命令字符串构建过程封装，避免了在业务逻辑中到处拼接字符串，降低了拼写错误风险。

#### **2. 配置对象/DTO 模式 (Configuration Object / DTO)**
*   **实现位置**: [`src/dpeva/config.py`](/src/dpeva/config.py) - `BaseWorkflowConfig` 及其子类
*   **解决场景**: 使用 Pydantic 模型作为数据传输对象（DTO），集中管理配置项的定义、默认值和类型校验。这避免了使用裸字典（Dict）传递参数导致的“参数黑洞”问题。

### 1.2 结构型模式 (Structural Patterns)

#### **3. 管道模式 (Pipeline Pattern)**
*   **实现位置**: [`src/dpeva/sampling/direct.py`](/src/dpeva/sampling/direct.py) - `DIRECTSampler` 继承自 `sklearn.pipeline.Pipeline`
*   **代码示例**:
    ```python
    class DIRECTSampler(Pipeline):
        def __init__(self, ...):
            steps = [('scaler', scaler), ('pca', pca), ('clustering', clustering), ...]
            super().__init__(steps)
    ```
*   **解决场景**: 采样流程包含数据标准化、降维、聚类、筛选等多个连续步骤。Pipeline 模式将这些步骤串联成一个单一的逻辑单元，使得调用者只需调用 `fit_transform` 即可完成全流程，且易于替换其中任意环节（如更换聚类算法）。

#### **4. 外观模式 (Facade Pattern)**
*   **实现位置**: `src/dpeva/submission/job_manager.py` (推断) - `JobManager`
*   **解决场景**: 提交任务到 Slurm 集群涉及生成脚本、环境配置、调用 `sbatch` 等复杂操作。`JobManager` 为上层工作流提供了一个统一的简易接口（如 `submit()`），屏蔽了底层调度系统的复杂性。

### 1.3 行为型模式 (Behavioral Patterns)

#### **5. 策略模式 (Strategy Pattern)**
*   **实现位置**: [`src/dpeva/workflows/collect.py`](/src/dpeva/workflows/collect.py) - `UQFilter` 和 `DIRECTSampler` 的选择
*   **解决场景**:
    *   **UQ 策略**: 根据 `uq_select_scheme` ("tangent_lo", "strict") 动态选择不同的过滤算法。
    *   **采样策略**: 根据 `sampler_type` 选择 `DIRECTSampler` 或 `TwoStepDIRECTSampler`。这允许在运行时根据配置动态切换算法实现。

#### **6. 命令/工作流模式 (Command / Workflow Pattern)**
*   **实现位置**: `src/dpeva/workflows/*.py` - `CollectionWorkflow`, `TrainingWorkflow`
*   **解决场景**: 将一个完整的主动学习任务（如“采集”、“训练”）封装为一个对象。该对象包含执行任务所需的所有上下文（配置、路径、状态），使得任务的执行（`run()`）与调用解耦。

---

## 2. 编程思想总结 (Programming Paradigms)

### **2.1 面向对象编程 (OOP) - 核心骨架**
*   **体现**: 项目通过类（Classes）来组织逻辑，如 `Workflow` 类管理状态，`Config` 类管理数据。继承机制被用于复用通用配置（`BaseWorkflowConfig`）。
*   **影响**: 提供了良好的模块化边界，使得代码易于理解和导航。

### **2.2 数据驱动/向量化编程 (Data-Driven / Vectorization) - 计算核心**
*   **体现**: 在 `UQCalculator` 和 `CollectionWorkflow` 中，大量使用 NumPy 和 Pandas 进行向量化运算（如 `np.mean(axis=0)`, `df.groupby`）。
*   **影响**: 相比 Python 原生循环，性能提升了数个数量级，且代码更简洁（`Explicit & Simple`）。

### **2.3 声明式配置 (Declarative Configuration)**
*   **体现**: 利用 Pydantic 的 Field 定义，声明“我们需要什么参数”及其约束，而不是编写命令式的解析代码。
*   **影响**: 极大提升了配置的可维护性和健壮性，减少了防御性编程的代码量。

---

## 3. 耦合度分析与解耦策略 (Coupling Analysis)

### **当前存在的耦合问题**

1.  **Workflows 类的“上帝对象”倾向 (High Coupling)**
    *   **问题**: `CollectionWorkflow` 职责过重，同时负责了：配置加载、路径管理、日志记录、数据加载、UQ 计算调用、绘图调用、结果导出、Slurm 提交逻辑。
    *   **后果**: 难以单独测试某一环节（如只测试“结果导出”），且代码文件庞大（~1300行）。
    *   **依赖图**: `Workflow` -> `JobManager`, `UQCalculator`, `UQVisualizer`, `DIRECTSampler`, `DPTestResultParser`, `logging`, `os`...

2.  **硬编码的绘图逻辑**
    *   **问题**: `CollectionWorkflow` 中包含大量关于绘图的控制逻辑（调用 `vis.plot_xxx`），导致业务逻辑与展示逻辑耦合。

### **解耦与重构策略**

#### **Strategy 1: 拆分 Workflow 职责**
*   **方案**: 引入 `Context` 对象和 `Task` 接口。
*   **实施**:
    1.  建立 `CollectionContext` 仅存储路径和数据。
    2.  将 Workflow 拆分为独立的 Task 类：`DataLoaderTask`, `UQAnalysisTask`, `SamplingTask`, `ExportTask`。
    3.  `CollectionWorkflow` 仅作为 Orchestrator（指挥者）按顺序执行这些 Task。

#### **Strategy 2: 依赖注入 (Dependency Injection)**
*   **方案**: 不要在 Workflow 内部实例化 `UQCalculator` 或 `UQVisualizer`。
*   **实施**:
    ```python
    # Before
    class CollectionWorkflow:
        def run(self):
            calculator = UQCalculator() # Hard dependency

    # After
    class CollectionWorkflow:
        def __init__(self, calculator: UQCalculator, ...):
            self.calculator = calculator
    ```
    这将极大提升单元测试的可测性（易于 Mock）。

---

## 4. 设计模式使用评估 (Evaluation)

| 模式 | 优势 (Pros) | 劣势 (Cons) | 评分 |
| :--- | :--- | :--- | :--- |
| **Builder** | 消除魔法字符串，易于扩展新参数 | 增加了一个额外的类文件 | ⭐⭐⭐⭐⭐ (必选) |
| **Pipeline** | 逻辑清晰，符合机器学习标准范式 | 调试中间步骤（如查看 PCA 后的数据）稍显麻烦 | ⭐⭐⭐⭐⭐ (推荐) |
| **Config/DTO** | 强类型，自动校验，IDE 提示友好 | 引入了 Pydantic 依赖 | ⭐⭐⭐⭐⭐ (必选) |
| **Facade (JobManager)** | 简化了上层调用 | 如果底层差异过大（如 Slurm vs PBS），外观类会变得臃肿 | ⭐⭐⭐⭐ (良好) |
| **Workflow (God Class)** | 初主要逻辑集中，易于编写原型 | 违反单一职责原则，维护成本随时间指数增长 | ⭐⭐ (需重构) |

---

## 5. 最佳实践建议 (Best Practices)

1.  **坚持“配置即代码”**: 继续保持使用 Pydantic 管理所有新增配置，杜绝 `kwargs` 满天飞。
2.  **重构“上帝工作流”**: 下一阶段的重构重点应是将 `CollectionWorkflow` 拆解为多个单一职责的 Service 类（如 `DataService`, `UQService`, `PlottingService`）。
3.  **强化接口定义**: 在 `src/dpeva/interfaces.py` 中定义抽象基类（Abstract Base Classes），如 `AbstractSampler`, `AbstractUQCalculator`，强制实现类遵循统一接口，便于扩展。
4.  **建立工厂模式**: 对于 `Sampler` 和 `UQCalculator` 的创建，可以引入简单的 Factory 函数，将对象的创建逻辑从业务逻辑中剥离。

### **实施路线图 (Roadmap)**

*   **短期 (Short-term)**: 完成 `CollectionWorkflow` 的解耦重构，将数据、UQ、采样逻辑拆分为独立的 Manager 类 (`CollectionIOManager`, `UQManager`, `SamplingManager`)，并归位到 `io`, `uncertain`, `sampling` 模块中。
*   **中期 (Mid-term)**: 引入 `Dependency Injection` 容器或手动注入模式，提升测试覆盖率。
*   **长期 (Long-term)**: 抽象出通用的 `ActiveLearningLoop` 框架，支持插件式的 UQ 和采样算法，使 DP-EVA 成为一个开放平台。

## 6. 架构重构实施 (Architecture Refactoring Implemented)

基于上述分析，`CollectionWorkflow` 已于 `v0.2.0` 版本完成重构，实现了**领域驱动的职责分离**：

1.  **数据层 (`dpeva.io.collection.CollectionIOManager`)**: 
    *   负责所有磁盘 I/O 操作（描述符加载、原子特征读取、结果导出）。
    *   封装了路径管理和日志配置。
2.  **UQ 层 (`dpeva.uncertain.manager.UQManager`)**:
    *   编排 UQ 计算流程（加载预测、计算方差、对齐尺度）。
    *   封装了自动阈值推导 (`auto_threshold`) 和过滤逻辑。
3.  **采样层 (`dpeva.sampling.manager.SamplingManager`)**:
    *   统一管理 DIRECT 与 2-DIRECT 策略。
    *   处理 Joint Sampling 的特征拼接逻辑。
4.  **工作流层 (`dpeva.workflows.collect.CollectionWorkflow`)**:
    *   仅保留配置注入和高层流程控制。
    *   通过组合（Composition）方式调用上述 Manager，代码量减少约 60%，逻辑清晰度显著提升。

---

## 7. Slurm 后端架构与并行设计 (Slurm Backend Architecture)

DP-EVA 专为高性能计算 (HPC) 环境设计，其核心设计目标之一是能够无缝地在本地单机 (Local) 和 Slurm 集群 (Slurm) 之间切换，且最大化集群资源的并行利用率。

### 7.1 双模调度架构 (Dual-Mode Scheduling)

系统通过 `JobManager` 实现了对底层计算资源的抽象：

*   **Local Backend**: 
    *   利用 `multiprocessing` 模块在本地并发执行任务。
    *   适用于调试、小规模测试或单节点工作站。
*   **Slurm Backend**:
    *   利用 `sbatch` 命令提交作业到集群调度器。
    *   支持 `partition`, `qos`, `nodes`, `gpus_per_node` 等高级 Slurm 参数。
    *   **关键特性**: 生成独立的 `.slurm` 脚本文件，确保作业可追溯、可复现。

### 7.2 并行投作业设计 (Parallel Submission Strategy)

为了解决大规模主动学习任务中的吞吐量瓶颈，**TrainingWorkflow** 和 **InferenceWorkflow** (v0.4.5+) 均采用了 **"One-Task-One-Job" (一任务一作业)** 的并行策略，而非将所有任务打包进单一作业。

#### **设计原则与验证 (v0.4.6+)**
*   **原则**: 系统必须保证 $N$ 个模型任务生成 $N$ 个独立的 Slurm 作业 ID。任何形式的串行化（如循环 subprocess 调用）或任务合并（Job Array 除外）在当前阶段均被视为反模式。
*   **验证**: 
    *   **单元测试**: 引入了专门的 Mock 测试 (`test_training_managers.py`, `test_inference_execution_manager.py`)，断言 `JobManager.submit` 方法的调用次数严格等于模型数量，防止逻辑退化。
    *   **集成测试**: 在真实 Slurm 环境中，Orchestrator 通过轮询文件系统，确认每个模型目录下的日志文件（如 `work/0/test_job.out`）均独立生成并包含 `WORKFLOW_FINISHED` 标记，从而在物理层面上验证了并行执行的真实性。

#### **TrainingWorkflow 的并行设计**
训练阶段通常涉及同时训练 4 个（或更多）模型以构建系综 (Ensemble)。
*   **并行模式**: `TrainingExecutionManager` 会遍历所有模型任务，为每个模型生成一个独立的 `train.slurm` 脚本。
*   **资源隔离**: 每个作业独立申请 GPU 资源（例如 4 个模型申请 4 个 GPU 卡），互不干扰。
*   **提交逻辑**:
    ```python
    # 伪代码逻辑
    for i in range(num_models):
        script_path = generate_slurm_script(task_id=i, ...)
        JobManager.submit(script_path) 
    ```
*   **优势**: 极大缩短了总训练时间（从 $T \times N$ 缩减为 $T$），且单个模型失败不会影响其他模型。

#### **InferenceWorkflow 的并行设计 (v0.4.5 重构)**
推理阶段需要对每个模型在测试集上进行预测。
*   **旧版问题 (v0.4.4及之前)**: 采用 "Self-Submission" 模式，即提交一个 Slurm 作业，该作业内部再串行调用 `subprocess` 运行所有模型的推理。这导致 N 个模型的推理只能串行执行，无法利用集群并行能力。
*   **新版设计 (v0.4.5+)**: 
    *   **并行化**: `InferenceExecutionManager` 现已对齐 Training 的设计，直接为每个模型生成独立的 `run_test.slurm` 脚本并并行提交。
    *   **命令构建**: 脚本内部直接调用 `dp test` (通过 `DPCommandBuilder`)，去除了中间层的 Python 包装，降低了开销。
    *   **状态管理**: 由于是异步提交，分析步骤 (`Analysis`) 需在作业完成后手动触发或由外部工作流编排器（如 Airflow）管理。

### 7.3 作业状态监控 (Job Monitoring)

为了支持自动化工作流，所有 Slurm 作业在成功执行完核心逻辑后，都会向 stdout/日志输出标准化的结束标记：

```text
DPEVA_TAG: WORKFLOW_FINISHED
```

外部系统（或未来的 `MonitorWorkflow`）可以通过轮询日志文件检测此 Tag，从而精确判定任务是否完成，实现基于事件的自动化编排。
