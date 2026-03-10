---
title: Archived Document
status: archived
audience: Historians
last-updated: 2026-03-09
---

# Spec Mode Review Report: Labeling Module

## 1. Review Summary

*   **Review Date**: 2026-03-05
*   **Target Version**: DP-EVA v0.2.0 (Labeling Integration)
*   **Scope**: `src/dpeva`, `docs`, `examples`
*   **Overall Score**: 95/100 (High Integration)

## 2. Checklist & Verification

### 2.1 Code Implementation (`src/dpeva/labeling`)
| Component | Requirement | Status | Verification Evidence |
| :--- | :--- | :--- | :--- |
| **Generator** | Configurable PP/Orb paths | ✅ | `AbacusGenerator.__init__` reads `pp_map`/`orb_map` from config. |
| | Geometry Classification | ✅ | `_judge_vacuum` correctly identifies Cluster/Layer/String/Bulk. |
| | K-Point Auto-gen | ✅ | `_set_kpoints` implements `kpt_criteria` logic. |
| **Packer** | Dynamic task packing | ✅ | `TaskPacker` supports custom `tasks_per_job` and prefixes. |
| **PostProcess** | Cohesive Energy Calc | ✅ | `compute_metrics` implements E0 fitting and calculation. |
| | Data Cleaning | ✅ | `filter_data` supports `NaN` thresholds for safety. |
| | Export Formats | ✅ | Supports `deepmd/npy`, `mixed`, `extxyz`. |
| **Strategy** | Resubmission Logic | ✅ | `ResubmissionStrategy` handles `mixing_beta` adjustment loop. |
| **Manager** | Workflow Orchestration | ✅ | `LabelingManager` connects all components via `LabelingConfig`. |

### 2.2 Documentation (`docs`)
| Document | Status | Notes |
| :--- | :--- | :--- |
| `specs/spec.md` | ✅ | Updated to reflect final architecture (Packer, Generator, etc.). |
| `recipes/README.md` | ✅ | Added Labeling quick start guide. |

### 2.3 User Examples (`examples`)
| Example | Status | Notes |
| :--- | :--- | :--- |
| `recipes/labeling/config_cpu.json` | ✅ | Valid JSON, includes `cleaning_thresholds` and `ref_energies`. |
| `recipes/labeling/config_gpu.json` | ✅ | Valid JSON, optimized for GPU partition. |
| `recipes/labeling/run_labeling.sh` | ✅ | Correct CLI entry point (`dpeva label`). |

## 3. Discrepancy Analysis

| Spec Requirement | Current Implementation | Impact | Resolution |
| :--- | :--- | :--- | :--- |
| **Visualization** | `converged_data_view.py` has plotting. `postprocess.py` does not. | **Low**. Intentional design decision to decouple compute from viz. | Visualization logic to be migrated to `src/dpeva/analysis` (See Plan). |
| **Config Schema** | Spec listed `strategies`, code uses `attempt_params`. | **None**. Code is more descriptive. | Updated Spec to match Code. |

## 4. Archiving Actions

The following development items are **100% Complete** and ready for archiving:

1.  **Labeling Core Logic**: `generator.py`, `packer.py`, `strategy.py`.
2.  **Labeling Workflow**: `LabelingWorkflow` class and CLI integration.
3.  **Config Schema**: `LabelingConfig` definition.

**Action**: Tag `v0.2.0-labeling-rc1`.

## 5. Risk Assessment

*   **Risk**: `ref_energies` auto-fitting might fail for single-element datasets if E0 is not provided.
    *   *Mitigation*: Added `NaN` safety check in `filter_data`.
*   **Risk**: Slurm `squeue` polling might be flaky on some clusters.
    *   *Mitigation*: `_monitor_slurm_jobs` has basic error handling, but backoff strategy could be improved.

---

# Architecture Plan: Visualization Module Decoupling

## 1. Objectives

> **Status Update (2026-03-05)**: The visualization decoupling functionality described in this plan is **not implemented**. Only the plotting style/settings module (`dpeva.utils.visual_style`) has been isolated as part of the Labeling module integration. The core decoupling of `converged_data_view.py` remains pending.

*   **Decoupling**: Remove heavy dependencies (`matplotlib`, `seaborn`) from core compute nodes (`labeling`, `training`).
*   **Performance**: Optimize rendering for large datasets (100k+ points).
*   **Flexibility**: Support headless rendering (CI/CD) and interactive notebooks.

## 2. Analysis of Legacy `converged_data_view.py`

*   **Dependencies**: `matplotlib.pyplot`, `seaborn`.
*   **Coupling**: Data loading (`DPDataLoader`) is tightly coupled with Plotting (`DataVisualizer`).
*   **Hardcoding**: Plot styles are mixed within plotting functions.

## 3. Proposed Architecture

### 3.1 Layered Design

1.  **Data Layer (`dpeva.analysis.data`)**:
    *   Responsible for loading `deepmd/npy`, `csv`, or `json` stats.
    *   Outputs: `pandas.DataFrame` or `numpy.ndarray`.
2.  **Render Layer (`dpeva.analysis.rendering`)**:
    *   `BaseRenderer`: Abstract interface (`plot_scatter`, `plot_dist`, `save`).
    *   `MatplotlibRenderer`: Implementation using MPL/Seaborn.
    *   `PlotlyRenderer` (Future): Interactive HTML output.
3.  **Facade Layer (`dpeva.analysis.facade`)**:
    *   High-level API: `plot_convergence_report(data_dir)`, `plot_uq_analysis(data_dir)`.

### 3.2 Migration Plan

#### Phase 1: Extraction (T+3 Days)
*   Extract `DataVisualizer` from `converged_data_view.py`.
*   Refactor into `src/dpeva/analysis/plotters/labeling_plotter.py`.
*   Ensure it accepts `pd.DataFrame` input (decoupled from `dpdata` loading).

#### Phase 2: Abstraction (T+7 Days)
*   Define `dpeva.utils.visual_style` as the single source of truth for styles (Done).
*   Create `BasePlotter` class to standardize `save_and_show` logic.

#### Phase 3: Integration (T+10 Days)
*   Create `dpeva analyze labeling` CLI command.
*   It loads data from `outputs/labeling` and invokes `LabelingPlotter`.

## 4. Deliverables

1.  `src/dpeva/analysis/` module.
2.  `dpeva analyze` CLI command.
3.  Unit tests for plot generation (mocking file output).
4.  Migration guide for users of `converged_data_view.py`.
