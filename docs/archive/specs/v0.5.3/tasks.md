# Tasks: Labeling Module Implementation

## Phase 1: Core Domain Logic (`src/dpeva/labeling`) [COMPLETED]
- [x] **Task 1.1**: Create `src/dpeva/labeling/` directory and `__init__.py`.
- [x] **Task 1.2**: Implement `src/dpeva/labeling/generator.py`.
    -   Port `FeCHO_fp_set.py` logic.
    -   Implement `AbacusGenerator` class.
    -   Extract geometry analysis (vacuum, dimension) into helper methods.
    -   Ensure PP/Orb paths are configurable.
- [x] **Task 1.3**: Implement `src/dpeva/labeling/strategy.py`.
    -   Port `batch_modify_input.py` logic.
    -   Implement `ResubmissionStrategy` class to handle parameter modifications.
- [x] **Task 1.4**: Implement `src/dpeva/labeling/postprocess.py`.
    -   Port `converged_data_view.py` logic.
    -   Implement `AbacusPostProcessor` class.
    -   Ensure plotting and cleaning logic is modular.

## Phase 2: Manager & Integration (`src/dpeva/managers`) [COMPLETED]
- [x] **Task 2.1**: Implement `src/dpeva/managers/labeling_manager.py`.
    -   Create `LabelingManager` class.
    -   Methods: `prepare_tasks`, `check_convergence`, `apply_strategy`, `collect_data`.
- [x] **Task 2.2**: Update `src/dpeva/config.py` (or equivalent).
    -   Add `LabelingConfig` schema.

## Phase 3: Workflow (`src/dpeva/workflows`) [COMPLETED]
- [x] **Task 3.1**: Implement `src/dpeva/workflows/labeling.py`.
    -   Create `LabelingWorkflow` class.
    -   Connect `LabelingManager` and `JobManager`.
    -   Implement the Loop (Submit -> Check -> Retry -> Collect).

## Phase 4: Verification & Testing [COMPLETED]
- [x] **Task 4.1**: Create unit tests for `AbacusGenerator` (geometry detection).
- [x] **Task 4.2**: Create integration test using `test/fp-setting` data.
    -   Step 1: Generate inputs from `sampled_dpdata` and compare with `abacus-inputs`.
    -   Step 2: Mock execution or check logic.
    -   Step 3: Process `conveged_data_view` data and verify cleaning logic.
