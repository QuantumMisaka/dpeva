# Labeling Workflow Development Documentation

## 1. Overview
The Labeling Workflow is a critical component of DP-EVA, responsible for generating high-quality First-Principles (FP) data for Deep Potential training. It orchestrates the entire process from data loading, input generation, task submission, monitoring, result processing, to data cleaning and export.

This document details the architecture, module responsibilities, and recent refactoring efforts to decouple structure analysis from input generation.

## 2. Architecture & Modules

### 2.1 Workflow Diagram
```mermaid
graph TD
    A[LabelingWorkflow] --> B[LabelingManager]
    B --> C[StructureAnalyzer]
    B --> D[AbacusGenerator]
    B --> E[TaskPacker]
    B --> F[JobManager]
    B --> G[AbacusPostProcessor]
    B --> H[ResubmissionStrategy]

    subgraph "Phase 1: Preparation"
    C -- Analyze --> I[Structure Type & Preprocessed Atoms]
    I --> B
    B -- Metadata --> D
    D -- Generate --> J[Input Files (INPUT/STRU/KPT)]
    J --> E
    E -- Pack --> K[Job Bundles (N_50_X)]
    end

    subgraph "Phase 2: Execution"
    K --> F
    F -- Submit --> L[Slurm/Local]
    L --> M[Results]
    end

    subgraph "Phase 3: Post-processing"
    M --> G
    G -- Check Convergence --> N{Converged?}
    N -- No --> H
    H -- Adjust Params --> D
    N -- Yes --> O[Clean & Export]
    end
```

### 2.2 Module Responsibilities

#### `dpeva.workflows.labeling.LabelingWorkflow`
- **Role**: High-level orchestrator.
- **Responsibility**: Manages the lifecycle (Load -> Prepare -> Execute -> Export), handles the main retry loop, and coordinates between Manager and JobManager.

#### `dpeva.labeling.manager.LabelingManager`
- **Role**: Domain logic coordinator.
- **Responsibility**:
    - Iterates over datasets and systems.
    - Calls `StructureAnalyzer` to determine structure type (`cluster`, `bulk`, etc.).
    - Constructs hierarchical directory paths (`inputs/[dataset]/[type]/[task]`).
    - Delegated input generation to `AbacusGenerator`.
    - Manages task packing via `TaskPacker`.
    - Processes results and applies `ResubmissionStrategy`.
    - Exports cleaned data and anomalies.

#### `dpeva.labeling.structure.StructureAnalyzer` (New)
- **Role**: Structure analysis engine.
- **Responsibility**:
    - **Preprocessing**: Wrapping, centering, and heuristic atom shifting.
    - **Vacuum Detection**: Determining vacuum presence along axes.
    - **Classification**: Identifying structure dimensionality (0D Cluster, 1D String, 2D Layer, 3D Bulk) and special types (Cubic Cluster).
    - **Standardization**: Swapping lattice vectors to align vacuum with Z-axis (for Layer/String).
- **Interface**: `analyze(atoms) -> (processed_atoms, stru_type, vacuum_status)`

#### `dpeva.labeling.generator.AbacusGenerator`
- **Role**: Input file writer.
- **Responsibility**:
    - Translates atomic structures and DFT parameters into ABACUS file formats (`INPUT`, `STRU`, `KPT`).
    - Applies structure-specific settings (e.g., dipole correction for Layers) based on metadata provided by `StructureAnalyzer`.
    - **Refactoring Note**: Logic for *determining* structure type has been moved to `StructureAnalyzer`, leaving this class focused on *writing*.

#### `dpeva.labeling.packer.TaskPacker`
- **Role**: Task bundler.
- **Responsibility**: Recursively scans for tasks and moves them into flat job directories (`N_50_X`) to optimize scheduler throughput.

## 3. Refactoring Report

### 3.1 Motivation
Previous implementations coupled structure analysis logic (e.g., vacuum detection) tightly within `AbacusGenerator`. However, `LabelingManager` also needed this information to determine directory structures *before* generation. This caused:
1.  **Code Redundancy**: Logic duplication or awkward cross-calling.
2.  **Unclear Boundaries**: Generator doing analysis tasks.

### 3.2 Changes Implemented
1.  **Extracted `StructureAnalyzer`**: Created `src/dpeva/labeling/structure.py` encapsulating all geometry analysis logic.
2.  **Simplified `AbacusGenerator`**: Removed `_judge_vacuum`, `_preprocess_structure`, and `_swap_crystal_lattice` methods. It now accepts `stru_type` and `vacuum_status` as arguments.
3.  **Updated `LabelingManager`**: Instantiates `StructureAnalyzer` (via `generator.analyzer` helper or directly) to perform analysis first, determines paths, then invokes generation.

### 3.3 Benefits
- **Separation of Concerns**: Analysis vs. Generation.
- **Testability**: `StructureAnalyzer` can be tested independently with synthetic Atoms objects.
- **Consistency**: Directory naming and INPUT parameter settings now strictly rely on the same analysis source.

## 4. Next Steps
1.  **Unit Tests**: Write dedicated tests for `StructureAnalyzer` covering edge cases (empty structures, near-vacuum thresholds).
2.  **Integration**: Verify the full workflow with the new decoupled architecture on the test dataset.
3.  **Documentation**: Update API documentation to reflect the new module structure.
