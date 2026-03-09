---
title: Archived Document
status: archived
audience: Historians
last-updated: 2026-03-09
---

# DP-EVA Labeling Module Specification

## 1. Overview
This document specifies the design for integrating First Principles (FP) labeling capabilities into DP-EVA. The goal is to refactor existing standalone scripts (`utils/fp`) into a modular, robust, and configurable `labeling` module within `src/dpeva`.

## 2. Architecture
The design follows the project's Domain-Driven Design (DDD) principles.

### 2.1 New Modules
*   **`src/dpeva/labeling/`**: Core domain logic for labeling.
    *   **`generation.py`**: Handles input file generation (converting `dpdata` to DFT inputs).
    *   **`postprocess.py`**: Handles result parsing, validation, cleaning, and export.
    *   **`strategy.py`**: Defines resubmission strategies for failed tasks (e.g., parameter tuning).
*   **`src/dpeva/managers/labeling_manager.py`**: Application service that orchestrates the labeling components.
*   **`src/dpeva/workflows/labeling.py`**: High-level workflow for the labeling process.

### 2.2 Component Details

#### 2.2.1 Input Generation (`AbacusGenerator`)
*   **Responsibility**: Convert atomic structures to ABACUS input files (`INPUT`, `STRU`, `KPT`).
*   **Key Features**:
    *   **Geometry Analysis**: Automatic vacuum detection, dimension classification (Cluster, 1D, 2D, Bulk).
    *   **Heuristics**: Auto-KPoint generation, Magnetism initialization, Cell standardization (axis swapping).
    *   **Configurable**: Pseudopotentials (PP), Basis sets (Orb), and DFT parameters must be injected via config, not hardcoded.
*   **Source**: Refactored from `FeCHO_fp_set.py`.

#### 2.2.2 Execution Strategy (`ResubmissionStrategy`)
*   **Responsibility**: Define how to handle task failures (non-convergence).
*   **Logic**:
    *   Level 0: Default parameters (e.g., `mixing_beta=0.4`).
    *   Level 1: Aggressive mixing (e.g., `mixing_beta=0.1`).
    *   Level 2: Conservative mixing (e.g., `mixing_beta=0.025`).
*   **Source**: Refactored from `batch_modify_input.py`.

#### 2.2.3 Post-Processing (`AbacusPostProcessor`)
*   **Responsibility**: Parse results, validate convergence, remove outliers, and format data.
*   **Key Features**:
    *   **Metric Calculation**: Cohesive energy, Pressure, Max Force.
    *   **Anomaly Detection**: Filter out unphysical structures based on thresholds.
    *   **Export**: Convert back to `deepmd/npy` or `extxyz` for training.
*   **Source**: Refactored from `converged_data_view.py`.

#### 2.2.4 Job Management Integration
*   **Task Packing**: Integrate `subjob_dist.py` logic into `JobManager` or a helper to bundle small DFT tasks into larger Slurm jobs to avoid queue limits.
*   **Submission**: Use existing `JobManager` for generic Slurm/Local submission.

## 3. Configuration Schema
New configuration sections in `LabelingConfig`:

```python
class LabelingConfig(BaseModel):
    dft_params: Dict[str, Any]  # Base INPUT parameters
    pp_map: Dict[str, str]      # Element -> PP file
    orb_map: Dict[str, str]     # Element -> Orb file
    pp_dir: str
    orb_dir: str
    attempt_params: List[Dict]  # List of retry parameters
    cleaning_thresholds: Dict   # Energy/Force/Stress limits
    ref_energies: Dict[str, float] # Reference energies
```

## 4. Workflow Flow
1.  **Input Phase**:
    *   Load `sampled_dpdata`.
    *   `LabelingManager.generate_inputs(data, config)`.
    *   Output: Directory structure with ABACUS inputs.
2.  **Execution Phase**:
    *   `JobManager.submit_batch(task_dirs)`.
    *   Monitor completion.
3.  **Correction Phase (Loop)**:
    *   Check convergence.
    *   For unconverged: Apply `ResubmissionStrategy` (modify inputs).
    *   Resubmit.
4.  **Output Phase**:
    *   `LabelingManager.collect_results(dirs)`.
    *   `LabelingManager.clean_data(data)`.
    *   Merge with training pool.

## 5. Decoupling Verification
*   **Input Setup**: Can be run independently to generate files for manual inspection.
*   **Execution**: Can run on any set of directories provided they contain `INPUT`/`STRU`.
*   **Output**: Can process any directory tree of ABACUS results, agnostic of how they were generated.
