# DP-EVA v0.4.6 代码修复与更新开发方案

**文档状态**: 已完成 (Done)
**版本**: v1.1
**日期**: 2026-03-02
**目标版本**: v0.4.7

## 1. 概述

本方案旨在针对《DP-EVA v0.4.6 项目状态总结报告》中识别的“错误处理静默”、“默认值不一致”、“Slurm 后端覆写隐晦”及“文档版本脱节”等风险点，以及用户提出的“CollectionWorkflow 日志增强”需求，制定具体的代码修复与开发计划。通过本次迭代（v0.4.7），将提升框架的健壮性、配置一致性、可维护性及用户体验。

## 2. 风险评估与修复计划

| ID | 风险/需求描述 | 优先级 | 修复策略 | 预计工时 | 负责人 |
| :--- | :--- | :---: | :--- | :---: | :--- |
| **R1** | **错误处理静默** (`__init__.py`)：环境检查异常被 `pass` 吞掉 | **P0** | 将静默吞掉改为捕获异常并打印 `UserWarning`，保留不崩溃特性但明确提示风险。 | 2h | TBD |
| **R2** | **默认值不一致** (`SamplingManager`)：内部默认值 0.1 与全局 0.5 不符 | **P1** | 统一引用 `dpeva.constants` 中的全局常量，消除硬编码。 | 2h | TBD |
| **R3** | **Slurm 后端覆写隐晦** (`CollectionWorkflow`) | **P2** | 增加显式日志提示当前运行模式；在 `developer-guide` 中补充该架构设计的说明。 | 2h | TBD |
| **R4** | **文档与代码脱节** (Audit Report & Examples) | **P2** | 更新审计报告状态；在 `examples/README` 增加环境路径修改的高亮提示。 | 1h | TBD |
| **F1** | **Collection 日志增强**：缺失目标池、采样数、剩余数的帧数统计 | **P1** | 在 `CollectionWorkflow` 的关键节点增加统计日志输出。 | 2h | TBD |

**总计预计工时**: 1.5人天 (9h)

## 3. 技术实现路径

### 3.1 [R1] 修复错误静默 (Error Handling)

**目标**: 遵循 "Errors should never pass silently" 原则。

**代码改动**:
- **文件**: `src/dpeva/__init__.py`
- **逻辑**:
  ```python
  import warnings
  
  try:
      check_deepmd_version()
  except ImportError:
      # 仅当 dpeva 自身依赖缺失导致无法检查时
      warnings.warn("Failed to import dependencies for environment check.", ImportWarning)
  except Exception as e:
      # 捕获检查过程中的其他错误（如 dp 命令不存在），打印警告但不阻断 import
      warnings.warn(f"DeepMD-kit environment check failed: {e}", UserWarning)
  ```
- **验证**: 编写单元测试，模拟 `check_deepmd_version` 抛出异常，断言 `warnings.warn` 被触发。

### 3.2 [R2] 统一默认值 (Consistency)

**目标**: 确保 `SamplingManager` 无论通过 Config 还是直接初始化，默认行为一致。

**代码改动**:
- **文件**: `src/dpeva/sampling/manager.py`
- **逻辑**:
  - 引入常量: `from dpeva.constants import DEFAULT_DIRECT_THR_INIT`
  - 修改 `__init__`: `self.direct_thr_init = self.config.get("direct_thr_init", DEFAULT_DIRECT_THR_INIT)`
- **验证**: 编写单元测试，实例化不带 config 的 `SamplingManager`，断言其 `direct_thr_init` 为 `0.5`。

### 3.3 [R3] 增强 Slurm 后端透明度 (Observability)

**目标**: 让开发者/用户在查看日志时明确知道当前是 "Worker via Slurm" 模式。

**代码改动**:
- **文件**: `src/dpeva/workflows/collect.py`
- **逻辑**:
  - 在 `CollectionWorkflow.__init__` 检测 `DPEVA_INTERNAL_BACKEND` 环境变量时：
    ```python
    if env_backend:
        self.logger.info(f"⚡ Running in Internal Worker Mode (Backend override: '{env_backend}'). "
                         f"This process is likely executed by a Slurm job.")
    ```
- **文档**: 在 `docs/guides/developer-guide.md` 添加 "Architecture Decision Record (ADR)" 章节，解释为何使用环境变量覆写后端。

### 3.4 [R4] 文档同步 (Documentation)

**内容**:
- **Audit Report**: 在 `docs/reports/2026-02-28-Code-Audit-Report.md` 头部添加说明：“注：本报告指出的硬编码问题已在 v0.4.6+ 修复。”
- **Examples**: 在 `examples/README.md` 顶部添加 Blockquote 警告：“⚠️ **注意**：运行示例前，请务必修改 `input.json` 中的 `env_setup` 路径，使其匹配您的集群环境。”

### 3.5 [F1] Collection 日志增强 (Logging Enhancement)

**目标**: 提供清晰的数据流转统计信息。

**代码改动**:
- **文件**: `src/dpeva/workflows/collect.py`
- **逻辑**:
  1.  **目标池数据统计**: 在 Phase 1 开始时（或加载描述符后），统计并输出所有 Pool 的总帧数。
      ```python
      # 在 _log_initial_stats 方法中增加总数统计
      total_frames = stats['num_frames'].sum()
      self.logger.info(f"Total candidate frames in pool: {total_frames}")
      ```
  2.  **采样结果统计**: 在 Phase 2 结束时（`_log_sampling_stats`），输出采样的总帧数。
      ```python
      # 在 _log_sampling_stats 方法中增加总数统计
      total_sampled = stats_sampled['sampled_frames'].sum()
      self.logger.info(f"Total sampled frames: {total_sampled}")
      ```
  3.  **剩余数据统计**: 在 Phase 3 导出 dpdata 后（`export_dpdata` 返回统计值，或在 Workflow 中计算），输出剩余数据的帧数。
      *   需要修改 `CollectionIOManager.export_dpdata` 让其返回具体帧数统计，而不仅仅是 system 个数。
      *   或者在 Workflow 中基于 `df_final` 和 `df_candidate` 简单计算：`remaining_frames = total_candidate_frames - total_sampled_frames`（注意：这是基于 DataFrame 的逻辑统计，与实际导出的 dpdata 系统数可能略有不同，建议以 DataFrame 统计为准，因为这是业务层面的“剩余”）。
      *   *决策*: 采用 DataFrame 统计并在 Workflow 中输出，保持 IO Manager 职责单纯。
      ```python
      # 在 run 方法 Phase 3 附近
      total_remaining = total_frames - total_sampled
      self.logger.info(f"Total remaining frames (to be exported as other_dpdata): {total_remaining}")
      ```

## 4. 资源分配

- **人力**: 后端开发工程师 1 名。
- **环境**:
  - 本地开发环境 (Python 3.8+, DeepMD-kit installed)。
  - (可选) Slurm 测试环境用于验证 R3 日志显示，若无则通过 Mock 环境变量验证。
- **依赖**: 无新增第三方依赖。

## 5. 测试计划

### 5.1 单元测试 (Unit Testing)
- **范围**: `tests/unit/utils/test_env_check.py` (新增 import 异常测试), `tests/unit/sampling/test_sampling_manager.py` (新增默认值测试).
- **执行**: `pytest tests/unit`

### 5.2 集成测试 (Integration Testing)
- **范围**: 运行现有的 `tests/integration/test_slurm_multidatapool_e2e.py` (若环境允许) 或 `tests/integration/test_collect_workflow_routing.py`。
- **重点**:
  - 检查 Workflow 初始化日志中是否包含新的模式提示。
  - 检查日志中是否包含 "Total candidate frames", "Total sampled frames", "Total remaining frames" 等关键信息。

### 5.3 回归测试 (Regression)
- 确保 `dpeva collect` 在常规 Local 模式下不受影响。
- 确保 `import dpeva` 在正常环境下无多余 Warning。

## 6. 上线与回滚

### 6.1 部署步骤
1.  完成代码修改与本地测试。
2.  提交 PR，通过 CI (Lint + Unit Tests)。
3.  合并至 `main` 分支。
4.  打标 `v0.4.7` 并发布 PyPI/Repo。

### 6.2 回滚策略
- 若发现 v0.4.7 引入阻塞性 Bug（如 import 崩溃），立即回退至 v0.4.6 Tag，并发布 v0.4.7.post1 或 v0.4.8。

## 7. 验收标准 (Definition of Done)

- [ ] **代码质量**: 所有新代码通过 `ruff` 检查，无新增 Pylint 错误。
- [ ] **测试覆盖**: 修改模块的单元测试通过率 100%，整体覆盖率不低于 80%。
- [ ] **功能验证**:
  - [ ] 模拟 `dp` 缺失环境，`import dpeva` 打印 Warning 但不报错。
  - [ ] `SamplingManager` 默认阈值确认为 0.5。
  - [ ] Slurm 任务日志中包含 "Internal Worker Mode" 提示。
  - [ ] Collection 日志清晰输出：初始总帧数、采样总帧数、剩余总帧数。
- [ ] **文档**: 风险报告状态已更新，示例警告已添加。
