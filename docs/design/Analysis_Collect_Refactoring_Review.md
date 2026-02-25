# Analysis & Collection Workflow Refactoring Review

## 1. Overview
This document consolidates the functional goals, responsibilities, and usage scenarios for the `AnalysisWorkflow` and `CollectionWorkflow` after recent refactoring. It clarifies the boundary between single-model analysis and ensemble analysis.

## 2. Functional Goals & Responsibilities

### 2.1 AnalysisWorkflow (`dpeva analysis`)
*   **Goal**: Provide detailed, point-to-point statistical analysis and visualization for a **single** DeepMD test result directory.
*   **Responsibility**:
    *   Load test results (energy, force, virial) from a specific directory.
    *   Load atomic composition info (robustly via `dpdata` or legacy regex).
    *   Calculate metrics: RMSE, MAE for Energy/Force/Virial.
    *   Calculate physical properties: Cohesive Energy (if composition available).
    *   Generate visualizations: Parity plots, Error distribution, Value distribution.
    *   Output standard reports: `metrics.json`, `metrics_summary.csv`.
*   **Usage Scenario**:
    *   User wants to manually inspect the performance of a specific model training run.
    *   User wants to re-analyze a result folder with different reference energies.
    *   Automated pipeline step *after* a single model inference (though typically handled by InferenceWorkflow internally).
*   **Key Manager**: `UnifiedAnalysisManager` (Shared with InferenceWorkflow).

### 2.2 CollectionWorkflow (`dpeva collect`)
*   **Goal**: Perform **Ensemble Analysis** and **Active Learning** data selection.
*   **Responsibility**:
    *   Load predictions from **multiple** models (the ensemble).
    *   **Uncertainty Quantification (UQ)**: Calculate QbC (Variance) and RND (Deviation) metrics.
    *   **Data Filtering**: Select candidate structures based on UQ thresholds.
    *   **Sampling**: Perform diversity sampling (e.g., DIRECT, FPS) on filtered candidates.
    *   **Ensemble Evaluation**: If ground truth exists, analyze "Error vs Uncertainty" relationship.
*   **Usage Scenario**:
    *   The final step of the DP-EVA loop.
    *   Selecting new training data from a large pool of unlabeled structures.
    *   Evaluating the reliability of the model ensemble.
*   **Key Manager**: `UQManager` (for analysis), `SamplingManager` (for selection).

## 3. Architecture & Boundary

| Feature | AnalysisWorkflow | CollectionWorkflow |
| :--- | :--- | :--- |
| **Input** | Single Result Directory | Multiple Model Predictions |
| **Focus** | Accuracy (RMSE/MAE) | Uncertainty (Std/Deviation) |
| **Logic** | `StatsCalculator` | `UQCalculator` |
| **Output** | Metrics & Distribution Plots | Selected Data & UQ Plots |
| **Ensemble?** | **NO** | **YES** |

## 4. Code Review & Status (Updated: 2026-02-25)

### 4.1 Refactoring Status
*   **Phase 1 (Core Unification)**: **Completed**.
    *   `UnifiedAnalysisManager` implemented in `dpeva.analysis.managers`.
    *   `InferenceAnalysisManager` deprecated and removed.
    *   `InferenceWorkflow` and `AnalysisWorkflow` updated to use `UnifiedAnalysisManager`.
*   **Phase 2 (Robust Data Source)**: **Completed**.
    *   `AnalysisIOManager` updated to use `dpdata` for robust composition loading.
    *   `AnalysisConfig` updated to include `data_path`.
*   **Phase 3 (Ensemble Capability)**: **Deferred**.
    *   Ensemble analysis remains the exclusive responsibility of `CollectionWorkflow`.

### 4.2 Code Quality Assessment
*   **Duplication**: Significantly reduced. The core logic for calculating RMSE/MAE and plotting distributions is now centralized in `UnifiedAnalysisManager`, shared by both Inference and Analysis workflows.
*   **Robustness**: Improved. `AnalysisWorkflow` no longer crashes on complex filenames thanks to `dpdata` integration.
*   **Architecture**: DDD (Domain-Driven Design) pattern is consistently applied across managers (`IOManager`, `ExecutionManager`, `AnalysisManager`).

### 4.3 Future Development (Roadmap)

#### 4.3.1 Parity Plot with Error Bars
*   **Target**: `CollectionWorkflow` (and potentially `InferenceWorkflow` summary).
*   **Description**: Plot Ground Truth vs Ensemble Mean Prediction, with error bars representing Ensemble Standard Deviation.
*   **Value**: Visualizes how well the ensemble uncertainty captures the true error.
*   **Status**: **Planned** (Deferred).

#### 4.3.2 Unified Error Calculation
*   **Observation**: `UQManager` currently implements its own error calculation logic.
*   **Plan**: In the future, `UQManager` should delegate error calculation to `StatsCalculator` (via `UnifiedAnalysisManager`) to ensure consistent metric definitions across the project.
