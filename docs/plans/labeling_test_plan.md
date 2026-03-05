# Test & Verification Plan: Labeling Module

## 1. Overview
This document outlines the testing strategy for the newly implemented Labeling Module (`src/dpeva/labeling`). The goal is to ensure robustness, correctness, and regression prevention through a comprehensive suite of Unit and Integration tests.

## 2. Unit Testing Plan
**Target**: 100% coverage for core logic classes.
**Location**: `tests/unit/labeling/`

### 2.1 Generator Tests (`test_generator.py`)
*   **Test Case 1: Geometry Classification**
    *   Input: Synthetic Atoms objects (Bulk, Layer, Cluster).
    *   Verify: `_judge_vacuum` returns correct boolean list.
    *   Verify: `stru_type` logic correctly identifies 0D/1D/2D/3D.
*   **Test Case 2: K-Point Generation**
    *   Input: Unit cell with known dimensions.
    *   Verify: `_set_kpoints` respects `kpt_criteria`.
    *   Verify: Vacuum directions always set K=1.
*   **Test Case 3: Magnetic Moments**
    *   Input: `mag_map` config + Atoms with Fe/O.
    *   Verify: `get_initial_magnetic_moments` matches config.

### 2.2 Strategy Tests (`test_strategy.py`)
*   **Test Case 1: Parameter Injection**
    *   Input: Mock `INPUT` file content + attempt params `{"mixing_beta": 0.1}`.
    *   Verify: Output content has modified `mixing_beta`.
    *   Verify: Original comments/formatting preserved.
*   **Test Case 2: Parameter Appending**
    *   Input: `INPUT` file missing a key.
    *   Verify: New key appended to end of file.

### 2.3 PostProcess Tests (`test_postprocess.py`)
*   **Test Case 1: Convergence Check**
    *   Input: Mock log files (one success, one fail).
    *   Verify: `check_convergence` returns True/False correctly.
*   **Test Case 2: Metric Calculation**
    *   Input: Mock `dpdata.System` with known Energy/Virial.
    *   Verify: `pressure_gpa` calculated correctly (check unit conversion).
    *   Verify: `cohesive_energy` calculated correctly using E0.
*   **Test Case 3: E0 Fitting**
    *   Input: 3 frames with different composition (A, B, AB).
    *   Verify: Least Squares correctly solves for E_A and E_B.

## 3. Integration Testing Plan
**Target**: End-to-end workflow validation.
**Location**: `tests/integration/labeling/`

### 3.1 Mock Execution Test (`test_workflow_mock.py`)
*   **Scenario**: Run full `LabelingWorkflow` but mock the `JobManager.submit` and `subprocess.run`.
*   **Setup**:
    *   Prepare `tests/data/sampled_dpdata` (small subset).
    *   Config: `backend="local"`, `mock_submission=True`.
*   **Execution**:
    *   `workflow.run()`
*   **Verification**:
    *   Check `inputs/` directory created.
    *   Check `INPUT/STRU` files exist.
    *   Check `run_batch.py` script generated.
    *   Check `task_packer` correctly grouped folders.

### 3.2 End-to-End Simulation (`test_workflow_e2e.py`)
*   **Scenario**: Full run with local execution (requires `abacus` mock or actual binary).
*   **Mock Binary**: Create a dummy `abacus` script that writes "charge density convergence is achieved" to `running_scf.log` and fake `OUT.ABACUS` data.
*   **Verification**:
    *   System detects convergence.
    *   System moves task to `CONVERGED`.
    *   System exports `outputs/cleaned.hdf5`.

## 4. Documentation & Examples
*   **Developer Guide**: Update `docs/guides/developer-guide.md` with "How to add new Generator rules".
*   **User Recipes**: Verify `examples/recipes/labeling/run_labeling.py` works out-of-the-box.

## 5. Timeline
*   **Week 1**: Implement Unit Tests (Generator, Strategy).
*   **Week 2**: Implement PostProcess tests + Mock Integration test.
*   **Week 3**: CI Pipeline integration (GitHub Actions / GitLab CI).
