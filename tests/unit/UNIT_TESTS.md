# DP-EVA 单元测试体系文档

## 1. 测试架构与目录说明 (Architecture)

本项目的测试体系遵循“金字塔模型”，将快速、稳定的单元测试作为质量保障的基石。所有单元测试代码均托管于 `dpeva/tests/unit` 目录下，与集成测试（`dpeva/test`）物理隔离，确保测试策略的清晰性。

### 1.1 目录结构
```text
dpeva/tests/
├── unit/                       # [核心] 单元测试根目录
│   ├── conftest.py             # 全局 Fixture 与 Mock 工厂定义
│   ├── test_calculator_uq.py   # UQCalculator 核心算法测试 (含数值稳定性)
│   ├── test_filter_uq.py       # UQFilter 筛选策略全覆盖测试
│   ├── test_parser.py          # I/O 解析器测试 (基于临时文件)
│   ├── test_sampling.py        # DIRECT 采样器逻辑测试
│   ├── feature/                # 特征生成模块测试 [新增]
│   │   └── test_generator.py   # DescriptorGenerator 逻辑测试 (CLI/Python模式)
│   ├── submission/             # 任务提交模块测试 [新增]
│   │   └── test_job_manager.py # JobManager 与 TemplateEngine 测试
│   ├── workflows/              # 工作流执行测试
│   │   ├── test_infer_workflow_exec.py      # 推理工作流 (启动/解析)
│   │   ├── test_train_workflow_init.py      # 训练工作流 (多模型/Slurm)
│   │   ├── test_collect_workflow_routing.py # 采集工作流 (单/多池路由)
│   │   └── test_feature_workflow_env.py     # 特征工作流 (环境注入)
│   └── UNIT_TESTS.md           # 本文档
└── ...
```

### 1.2 职责划分
*   **unit**: 关注函数/类级别的逻辑正确性。不依赖真实的大型模型文件或外部数据库，所有输入均通过 Mock 或合成数据生成，追求毫秒级响应。
*   **unit/workflows**: 关注业务流程编排的正确性。验证配置解析、路径路由、命令构建及外部组件（如 Slurm）的调用参数，但不实际执行耗时任务。
*   **conftest.py**: 集中管理测试数据生成器（如 `mock_predictions_factory`）及环境模拟器（`mock_job_manager`, `real_config_loader`），避免测试代码重复，并统一管理随机种子。

## 2. 源码映射与测试范围 (Mapping & Scope)

| 领域 | 被测源码文件 (`src/dpeva/...`) | 测试文件 (`tests/unit/...`) | 测试核心范围 |
| :--- | :--- | :--- | :--- |
| **不确定度** | `uncertain/calculator.py` | `test_calculator_uq.py` | QbC/RND 公式精度、Auto-UQ (KDE)、数值稳定性 (NaN/Inf) |
| **筛选** | `uncertain/filter.py` | `test_filter_uq.py` | 5种筛选策略 (`strict`, `tangent`...), 几何边界判定 |
| **I/O** | `io/dataproc.py` | `test_parser.py` | `dp test` 输出文件解析、无标签数据兼容性 |
| **采样** | `sampling/direct.py` | `test_sampling.py` | 聚类+分层采样流程、维度一致性 |
| **特征生成** | `feature/generator.py` | `feature/test_generator.py` | **混合模式生成逻辑**、递归目录处理、CLI 命令构建 |
| **任务提交** | `submission/manager.py` | `submission/test_job_manager.py` | **Template 渲染**、Slurm/Local 提交命令封装 |
| **推理流** | `workflows/infer.py` | `workflows/test_infer_workflow_exec.py` | 模型自动发现、命令构建、JobManager 调用、异常处理 |
| **训练流** | `workflows/train.py` | `workflows/test_train_workflow_init.py` | 多模型目录隔离、配置继承、Trainer 初始化 |
| **采集流** | `workflows/collect.py` | `workflows/test_collect_workflow_routing.py` | 单/多数据池路径解析、迭代参数传递 |
| **特征流** | `workflows/feature.py` | `workflows/test_feature_workflow_env.py` | Slurm 环境注入、CLI 参数透传 |

## 3. 测试用例详解 (Test Cases)

### 3.1 核心算法测试
*   **不确定度计算 (`test_calculator_uq.py`)**: 验证 QbC/RND/Auto-UQ 算法的数学精确性及对 NaN/Inf 的鲁棒性。
*   **筛选策略 (`test_filter_uq.py`)**: 全面覆盖 `strict`, `tangent_lo` 等筛选边界逻辑。
*   **I/O 解析 (`test_parser.py`)**: 验证 DeepMD 输出解析及无标签数据的处理。
*   **采样逻辑 (`test_sampling.py`)**: 验证 DIRECT 采样流程的维度一致性。

### 3.2 模块功能测试 (Modules) **[新增]**
*   **特征生成 (`feature/test_generator.py`)**:
    *   `test_cli_generation_command_construction`: 验证 `dp eval-desc` 命令参数构建（含 `-s`, `-m`, `--head`）。
    *   `test_python_generation_recursion`: 验证 Python 模式下对嵌套目录结构（Root -> Group -> System）的递归处理能力。
*   **任务提交 (`submission/test_job_manager.py`)**:
    *   `test_generate_script_slurm`: 验证 SBATCH 指令生成（`-J`, `-p`, `--gpus-per-node`）。
    *   `test_submit_slurm`: 验证 `sbatch` 命令调用。

### 3.3 工作流执行测试 (Workflows)
*   **推理工作流 (`test_infer_workflow_exec.py`)**:
    *   `test_init_model_discovery`: 验证能否自动识别嵌套目录下的 `model.ckpt.pt`。
    *   `test_run_command_generation`: 验证生成的 `dp test` 命令参数是否正确。
*   **训练工作流 (`test_train_workflow_init.py`)**:
    *   `test_training_workflow_init_multi_model`: 验证多模型训练时的目录创建及配置参数继承。
*   **采集工作流 (`test_collect_workflow_routing.py`)**:
    *   `test_collect_single_pool_routing`: 验证单数据池模式下的路径解析。
    *   `test_collect_multi_pool_routing`: 验证多数据池（联合采样）模式下的参数分离。

## 4. 测试数据与 Mock 策略

### 4.1 动态数据工厂 (`mock_predictions_factory`)
位于 `conftest.py`，用于动态生成 `MockDPTestResults` 对象。
*   **优势**: 允许测试按需指定原子数、帧数、预测值和 Ground Truth，无需依赖外部文件。

### 4.2 真实配置加载器 (`real_config_loader`)
位于 `conftest.py`，用于加载项目中的真实配置文件，并动态替换其中的绝对路径为测试临时路径。
*   **优势**: 确保单元测试使用的配置结构与真实生产环境一致，避免配置漂移。

### 4.3 作业管理器 Mock (`mock_job_manager`)
位于 `conftest.py`，通过 `unittest.mock.patch` 拦截 `JobManager` 的提交动作。
*   **优势**: 允许在无 Slurm 环境下验证作业提交逻辑和脚本生成内容。

## 5. 运行与调试指南

### 5.1 基础命令
```bash
# 运行所有单元测试
pytest tests/unit

# 运行特定模块
pytest tests/unit/feature
pytest tests/unit/submission
```

### 5.2 覆盖率检查
```bash
# 生成覆盖率报告 (终端显示)
pytest tests/unit --cov=dpeva --cov-report=term-missing
```

## 6. 覆盖率阈值与质量门禁

本项目对核心模块实施严格的质量门禁：

| 模块 | 最低行覆盖率 (Line Cov) | 失败策略 |
| :--- | :--- | :--- |
| `dpeva.uncertain` | **100%** | CI 阻断 (Block Merge) |
| `dpeva.workflows` | **85%** | CI 警告 |
| `dpeva.sampling` | **90%** | CI 警告 |
| `dpeva.io` | **85%** | CI 警告 |
| `dpeva.feature` | **90%** | CI 警告 |
| `dpeva.submission` | **95%** | CI 警告 |

## 7. 维护规范

1.  **命名约定**:
    *   测试文件: `test_<module_name>.py`
    *   工作流测试: `workflows/test_<workflow>_<feature>.py`
2.  **变更流程**:
    *   修改业务逻辑前，先运行现有测试确保通过。
    *   新增功能必须同步添加对应的单元测试。
3.  **Mock 原则**:
    *   对于文件 I/O，使用 `tmp_path`。
    *   对于外部耗时调用（如 `dp test`, `sbatch`），必须使用 Mock。
