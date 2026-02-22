# DP-EVA 集成测试审查报告

**日期**: 2026-02-22
**审查对象**: `/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/tests/integration`

## 1. 总体评估

DP-EVA 的集成测试体系（主要集中在 `test_slurm_multidatapool_e2e.py`）设计思路清晰，通过模拟真实 Slurm 环境下的完整业务流程（Feature -> Training -> Inference -> Collect），有效地覆盖了核心链路。测试代码采用了模块化设计（`orchestrator.py`, `data_minimizer.py`, `slurm_utils.py`），具有较好的可读性和结构。

然而，审查发现当前测试体系存在以下主要问题：
1.  **强依赖外部环境**: 严重依赖 Slurm 集群环境及特定的数据路径（`test/test-for-multiple-datapool`），导致测试在非生产环境或 CI/CD 流水线中难以运行。
2.  **覆盖场景单一**: 目前仅有一个 "Happy Path" 的端到端测试，缺乏对异常流程（如任务失败、资源不足、数据损坏）和边界条件的覆盖。
3.  **执行效率低**: 依赖真实的 Slurm 调度和文件系统 IO，测试执行时间长（超时设置达 2 小时），反馈周期慢。
4.  **断言力度不足**: 大部分断言仅检查文件是否存在，未深入验证文件内容的正确性（如数值精度、数据完整性）。
5.  **Flaky 风险**: 依赖 `wait_for_text_in_file` 轮询日志文件，若日志输出延迟或截断，极易导致 Flaky Test。

## 2. 详细审查发现

### 2.1 测试覆盖与场景
*   **覆盖范围**: 仅覆盖了 `Slurm` 后端的完整流程。缺失 `Local` 后端的集成测试，也未覆盖 `2-DIRECT` 等高级采样策略。
*   **数据依赖**: 测试数据通过 `data_minimizer.py` 从 `test/test-for-multiple-datapool` 复制并裁剪。这种硬编码的源路径使得测试环境搭建成本高，且容易因源数据变动导致测试失败。
*   **缺失场景**:
    *   任务提交失败或超时处理。
    *   部分模型训练失败后的容错机制。
    *   空数据集或极端数据分布下的采样表现。

### 2.2 代码质量与规范
*   **命名规范**: 遵循了 Python 标准命名规范，函数名清晰。
*   **模块化**: 将数据准备、任务编排、Slurm 工具类分离，结构良好。
*   **硬编码**: 存在大量硬编码路径和配置（如 `DPA-3.1-3M.pt`, `mptrj-FeCOH`），降低了测试的灵活性。
*   **类型注解**: 大部分函数有类型注解，但部分关键函数（如 `_load_template`）返回 `dict` 类型过于宽泛。

### 2.3 稳定性与效率 (Flaky Tests)
*   **日志轮询**: `wait_for_text_in_file` 是潜在的 Flaky 来源。如果文件系统同步延迟，或者日志被覆盖，测试将无故超时。建议结合 `squeue` 状态检查。
*   **超时设置**: 默认超时时间过长（7200秒），掩盖了性能回退问题。
*   **资源竞争**: 并发运行多个测试时，固定的工作目录结构可能导致文件冲突。

### 2.4 断言有效性
*   **弱断言**:
    *   `assert (work_dir / ... / "model.ckpt.pt").exists()`: 仅检查文件存在，未验证模型是否有效（如加载检查）。
    *   `assert (work_dir / ... / "final_df.csv").exists()`: 未验证生成的 CSV 行数是否符合预期，内容是否为空。

## 3. 改进建议与行动计划

### 3.1 增强测试健壮性与独立性
*   **Mock 数据生成**: 不再依赖外部 `test/` 目录，而是使用 `dpdata` 和 `numpy` 在测试运行时动态生成微型合成数据（Synthetic Data）。这将彻底解除环境依赖。
*   **本地集成测试**: 增加基于 `Local` 后端的集成测试，使用 `subprocess` 模拟任务执行，使测试能在无 Slurm 环境下运行（CI 友好）。

### 3.2 提升断言质量
*   **内容校验**:
    *   读取 `final_df.csv`，验证行数是否 > 0，且包含预期列。
    *   简单加载生成的模型（如果环境允许）或检查模型文件大小/Header。
    *   解析 `results.e.out`，验证误差统计是否在合理范围内（避免数值爆炸）。

### 3.3 优化执行效率
*   **数据裁剪**: 进一步缩小测试数据集规模（如仅使用 2-5 帧数据，极简网络结构），将端到端测试时间控制在 5 分钟以内。
*   **Mock Slurm**: 对于逻辑验证，可以 Mock `sbatch`/`squeue` 命令，仅验证生成的脚本内容和提交流程，而无需真实调度。

### 3.4 扩展测试场景
*   **异常注入**: 模拟任务脚本返回非零退出码，验证 Orchestrator 的错误捕获机制。
*   **配置变体**: 增加测试用例覆盖 `2-DIRECT` 采样、`User Defined` UQ 阈值等配置组合。

## 4. 推荐的重构路线图

1.  **短期 (Quick Wins)**:
    *   增加对输出文件内容的断言（CSV 读取校验）。
    *   将硬编码路径改为可配置参数或 fixture。
    *   优化 `wait_for_text_in_file`，增加对 `squeue` 状态的辅助检查，减少等待死锁。

2.  **中期 (Infrastructure)**:
    *   创建 `SyntheticDataFixture`，移除对 `test-for-multiple-datapool` 的依赖。
    *   实现 `LocalOrchestrator` 测试套件，作为 CI 的主力集成测试。

3.  **长期 (Coverage)**:
    *   引入 Mock Slurm 框架，实现全场景的调度逻辑测试。
