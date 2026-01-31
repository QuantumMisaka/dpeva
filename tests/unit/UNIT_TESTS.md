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
│   └── UQ_TEST_REPORT.md       # 本文档
└── ...
```

### 1.2 职责划分
*   **unit**: 关注函数/类级别的逻辑正确性。不依赖真实的大型模型文件或外部数据库，所有输入均通过 Mock 或合成数据生成，追求毫秒级响应。
*   **conftest.py**: 集中管理测试数据生成器（如 `mock_predictions_factory`），避免测试代码重复，并统一管理随机种子。

## 2. 源码映射与测试范围 (Mapping & Scope)

| 被测源码文件 (`src/dpeva/...`) | 测试文件 (`tests/unit/...`) | 测试核心范围 | 关键业务上下文 |
| :--- | :--- | :--- | :--- |
| `uncertain/calculator.py` | `test_calculator_uq.py` | QbC/RND 公式精度、Auto-UQ (KDE)、数值稳定性 (NaN/Inf) | 主动学习中的不确定度量化计算，直接决定样本筛选质量。 |
| `uncertain/filter.py` | `test_filter_uq.py` | 5种筛选策略 (`strict`, `tangent`...), 几何边界判定 | 决定哪些样本被视为“高价值”候选点。 |
| `io/dataproc.py` | `test_parser.py` | `dp test` 输出文件解析、无标签数据兼容性 | 处理 DeepMD 产生的异构数据格式。 |
| `sampling/direct.py` | `test_sampling.py` | 聚类+分层采样流程、维度一致性 | 在高维特征空间中进行多样性采样。 |

## 3. 测试用例详解 (Test Cases)

### 3.1 不确定度计算 (`test_calculator_uq.py`)
*   **`test_compute_qbc_rnd_golden_value`**
    *   **目的**: 验证 QbC/RND 计算公式的数学精确性。
    *   **策略**: 使用 `numpy` 手算“黄金值”与 `UQCalculator` 输出比对。
    *   **断言**: 相对误差 < 1e-5。
*   **`test_compute_qbc_rnd_robustness_nan`**
    *   **目的**: 验证 **"Clamp-and-Clean"** 策略对异常模型的防御能力。
    *   **输入**: 包含 `NaN` 的预测力矩阵。
    *   **预期**: 系统记录 Warning 日志，并将结果替换为 `Infinity`（最大不确定度），而非崩溃或输出 0。
*   **`test_calculate_trust_lo_gaussian`**
    *   **目的**: 验证 Auto-UQ (KDE) 算法对标准分布的响应。
    *   **输入**: 均值 0.5、方差 0.1 的高斯分布数据。
    *   **预期**: 计算出的 `trust_lo` 阈值应落在理论峰值右侧下降点附近 (约 0.617)。

### 3.2 筛选策略 (`test_filter_uq.py`)
*   **`test_filter_counts_sanity`** (参数化)
    *   **目的**: 验证所有筛选 Scheme (`strict`, `tangent_lo` 等) 的分类逻辑完备性。
    *   **策略**: 在 `[0, 0.3]` 区间构造 31x31 的密集网格点，统计 Candidate/Accurate/Failed 分区数量。
*   **`test_filter_tangent_lo_boundary`**
    *   **目的**: 精确测试切线筛选边界。
    *   **输入**: 恰好落在 $x+y = 2 \cdot lo$ 线上的点。
    *   **断言**: 验证不等号方向是否符合预期（Candidate 包含边界）。

### 3.3 I/O 解析 (`test_parser.py`)
*   **`test_parser_basic`**
    *   **目的**: 验证标准 `.e.out` / `.f.out` 文件解析。
    *   **Mock**: 使用 `pytest.tmp_path` 创建包含注释和数据的临时文件。
*   **`test_parser_no_ground_truth`**
    *   **目的**: 验证无标签数据（全 0 占位）场景下的解析与标识位设置。

## 4. 测试数据与 Mock 策略

### 4.1 动态数据工厂 (`mock_predictions_factory`)
位于 `conftest.py`，用于动态生成 `MockDPTestResults` 对象。
*   **优势**: 允许测试按需指定原子数、帧数、预测值和 Ground Truth，无需依赖外部文件。
*   **示例**:
    ```python
    # 生成一个包含 5 个原子，且带有 NaN 异常的预测结果
    f0 = np.zeros((5, 3)); f0[0,0] = np.nan
    p0 = mock_predictions_factory(f0, [5])
    ```

### 4.2 文件系统 Mock
使用 PyTest 内置的 `tmp_path` fixture。
*   **策略**: 在内存/临时目录中动态创建测试文件，测试结束后自动清理。
*   **应用**: `test_parser.py` 中用于模拟 DeepMD 的输出目录结构。

### 4.3 随机性控制
所有涉及随机生成的测试（如采样、分布模拟）均显式设置随机种子：
```python
np.random.seed(42)
```
确保测试结果在任何环境下具备**可重复性 (Reproducibility)**。

## 5. 运行与调试指南

### 5.1 基础命令
```bash
# 运行所有单元测试
pytest tests/unit

# 运行特定测试文件
pytest tests/unit/test_calculator_uq.py

# 运行特定用例 (支持模糊匹配)
pytest -k "robustness"
```

### 5.2 覆盖率检查
```bash
# 生成覆盖率报告 (终端显示)
pytest tests/unit --cov=dpeva.uncertain --cov=dpeva.sampling --cov-report=term-missing

# 生成 HTML 详细报告
pytest tests/unit --cov=dpeva --cov-report=html
# 打开 htmlcov/index.html 查看
```

### 5.3 调试模式
*   `-s`: 显示标准输出 (print/logging)，用于调试日志逻辑。
*   `-v`: 详细模式，显示每个 Passed/Failed 的用例名。
*   `--pdb`: 测试失败时自动进入 Python 调试器。

## 6. 覆盖率阈值与质量门禁

本项目对核心模块实施严格的质量门禁：

| 模块 | 最低行覆盖率 (Line Cov) | 失败策略 |
| :--- | :--- | :--- |
| `dpeva.uncertain` | **100%** | CI 阻断 (Block Merge) |
| `dpeva.sampling` | **90%** | CI 警告 |
| `dpeva.io` | **85%** | CI 警告 |

**CI 配置片段 (.github/workflows/ci.yml)**:
```yaml
- name: Enforce Coverage
  run: |
    pytest tests/unit --cov=dpeva.uncertain --cov-fail-under=100
```

## 7. 维护规范

1.  **命名约定**:
    *   测试文件: `test_<module_name>.py` 或 `test_<feature>_uq.py`
    *   测试类: `Test<ClassName>`
    *   测试函数: `test_<function_name>_<condition>`
2.  **变更流程**:
    *   修改业务逻辑前，先运行现有测试确保通过。
    *   新增功能必须同步添加对应的单元测试。
    *   修复 Bug 时，必须先编写一个**复现该 Bug 的失败测试**（TDD 思想），再修复代码。
3.  **废弃策略**:
    *   随着重构（如 `test_calculator.py` 被 `test_calculator_uq.py` 取代），应及时删除旧测试文件或标记为 `@pytest.mark.skip`。

## 8. 附录：依赖与环境

*   **测试框架**: `pytest >= 7.0`
*   **插件**: `pytest-cov`, `pytest-mock` (可选)
*   **核心依赖**: `numpy`, `pandas`, `scipy`, `sklearn`
