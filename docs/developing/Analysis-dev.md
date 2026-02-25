# DP-EVA Analysis System Refactoring Plan

## 1. 现状分析 (Status Analysis)

### 1.1 AnalysisWorkflow 功能清单
AnalysisWorkflow 目前作为一个独立的 CLI 工具运行，其核心功能包括：
*   **数据加载**: 使用 `DPTestResultParser` 解析单目录下的 DeepMD 测试结果（energy, force, virial）。
*   **成分推断**: 尝试通过文件名正则匹配（如 `H2O.e.out`）推断化学成分，这种方式极易在非标准命名下失败。
*   **统计分析**: 调用 `StatsCalculator` 计算 RMSE/MAE，并尝试通过最小二乘法拟合原子能量以计算结合能。
*   **图表输出**: 生成能量、力、维里、结合能的分布图 (`dist_*.png`)、对角图 (`parity_*.png`) 和误差分布图 (`error_dist_*.png`)。
*   **报告生成**: 输出 `metrics.json` (标量指标), `metrics_summary.csv` (汇总), `cohesive_energy_pred_stats.json`。

### 1.2 与 InferenceWorkflow.analyze_results 的对比
| 对比维度 | AnalysisWorkflow (`run`) | InferenceWorkflow (`analyze_results`) | 判定 |
| :--- | :--- | :--- | :--- |
| **核心算法** | `StatsCalculator` | `StatsCalculator` | **100% 重复** |
| **绘图引擎** | `InferenceVisualizer` | `InferenceVisualizer` | **100% 重复** |
| **数据源质量** | **低 (Low)**: 依赖文件名正则猜测成分 | **高 (High)**: 使用 `dpdata` 读取原始测试集 | 逻辑差异 |
| **输出产物** | 单个 JSON/CSV | 包含分布详情的 JSON + 聚合 CSV | 部分重合 |
| **处理范围** | 单一结果目录 | 遍历所有模型子目录 (Ensemble) | 范围差异 |

### 1.3 核心问题与技术债务
1.  **代码重复 (DRY Violation)**: `AnalysisManager` 和 `InferenceAnalysisManager` 两个类有 90% 的代码逻辑是复制粘贴的。
2.  **数据源脆弱性**: AnalysisWorkflow 缺乏可靠的成分信息来源，导致结合能计算功能经常失效。
3.  **功能割裂**: 独立分析流无法处理多模型数据，无法进行不确定度分析。

## 2. 重构方案 (Refactoring Plan)

### Phase 1: 核心统一 (Core Unification)
**目标**: 合并两个 Manager，消除重复代码。
*   [ ] 创建 `UnifiedAnalysisManager` (在 `dpeva.analysis.managers`)。
    *   将 `InferenceAnalysisManager` 的逻辑迁移至此。
    *   支持可选的 `atom_counts` 输入。
*   [ ] 重构 `InferenceWorkflow` 使用新 Manager。
*   [ ] 重构 `AnalysisWorkflow` 使用新 Manager。
*   [ ] 删除旧的 `InferenceAnalysisManager`。

### Phase 2: 数据源健壮化 (Robust Data Source)
**目标**: 解决 AnalysisWorkflow 无法正确计算结合能的问题。
*   [ ] 在 `AnalysisConfig` 中增加 `data_path` 字段。
*   [ ] 集成 `dpdata` 加载逻辑到 `AnalysisIOManager`。

### Phase 3: 系综分析能力 (Ensemble Capability) - DEFERRED
**Status**: Deferred. Decision made to keep AnalysisWorkflow focused on single-model analysis. Ensemble analysis belongs to CollectionWorkflow.

**目标**: 赋予 CollectionWorkflow 高级分析能力。
*   [ ] 实现 `EnsembleStatsCalculator` 计算不确定度 (in CollectionWorkflow context).
*   [ ] 绘制 "Parity Plot with Error Bars" 图表 (Error vs Uncertainty).


## 4. Code Review & Validation (2026-02-25)

### 4.1 Functional Validation
*   **AnalysisWorkflow**:
    *   Validated via `tests/unit/workflows/test_analysis_workflow.py`.
    *   Single-model analysis correctly generates `metrics.json` and plots.
    *   Legacy regex fallback for composition parsing works.
    *   New `dpdata` integration correctly loads composition from `data_path`.
*   **CollectionWorkflow**:
    *   Ensemble analysis logic remains intact in `UQManager`.
    *   Workflow correctly delegates to `UQManager` and `SamplingManager`.

### 4.2 Architectural Assessment
*   **Manager Pattern**: Successfully applied. `UnifiedAnalysisManager` acts as the single source of truth for standard DP metrics calculation.
*   **Separation of Concerns**:
    *   `AnalysisWorkflow` -> Single Point Accuracy (RMSE/MAE).
    *   `CollectionWorkflow` -> Ensemble Uncertainty (QbC/RND).
    *   Boundary is clear and respected.

### 4.3 Code Quality & Debt
*   **Duplication**: Reduced by ~400 lines by removing `InferenceAnalysisManager`.
*   **Remaining Debt**: `UQManager` still has some custom error calculation logic that could eventually use `StatsCalculator`.

### 4.4 Status Summary
*   **Phase 1 (Core Unification)**: ✅ Completed.
*   **Phase 2 (Robust Data)**: ✅ Completed.
*   **Phase 3 (Ensemble)**: ⏸️ Deferred (Targeting CollectionWorkflow).

