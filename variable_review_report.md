# DP-EVA Variable Review & Risk Assessment Report

**Date:** 2026-02-02
**Auditor:** Trae AI Assistant
**Scope:** `src/dpeva` (Core Library) & `runner` (Usage Examples)

---

## 1. Executive Summary

This report presents a systematic review of all user-configurable parameters in the DP-EVA codebase. The review is based on a static AST analysis (`parameter_audit.json`) and manual inspection of usage patterns in the `runner` directory.

**Key Statistics:**
- **Total User-Configurable Parameters:** 273
- **Explicit Parameters (Function Args):** 213
- **Implicit Parameters (`config.get`):** 60 (High Risk)
- **Parameters with Missing Docstrings:** 116 (42.5% Coverage Gap)
- **Parameters with Default Values:** 147

**Overall Health Assessment:**
The project follows a "config-dict" pattern common in research code, but this leads to significant **implicit dependencies** where valid configuration keys are hidden from the API surface. Docstring coverage is below acceptable standards for a production-grade library, with nearly half of the parameters undocumented.

---

## 2. Risk Assessment Matrix

We have categorized identified risks into three levels:

| Risk Level | Description | Impact | Count |
| :--- | :--- | :--- | :--- |
| 🔴 **High** | Implicit dependencies, logic conflicts, hardcoded environment paths. | Runtime errors, non-portability, hidden API surface. | ~65 |
| 🟡 **Medium** | Missing docstrings, environment-coupled defaults (e.g., `os.getcwd()`). | Developer confusion, unexpected behavior in new envs. | ~120 |
| 🔵 **Low** | Magic numbers, cosmetic defaults. | Maintenance burden. | ~80 |

---

## 3. Detailed Variable Review by Module

### 3.1. Workflows (`src/dpeva/workflows/`)

This module contains the highest density of implicit parameters.

#### `collect.py` - `CollectionWorkflow`
**Status:** 🔴 **Critical**
**Issues:**
1.  **Implicit Config Injection**: The `__init__` method accepts a generic `config` dict, but internally relies on 25+ keys via `self.config.get()`.
2.  **Logic Conflict / Magic Numbers**:
    *   **Variable**: `uq_trust_ratio`
    *   **Code Default**: `0.33` (Implicit)
    *   **Runner Config**: `0.50` (User Example)
    *   **Risk**: Inconsistent defaults lead to irreproducible experiments.
3.  **Variable**: `uq_qbc_trust_lo` / `uq_qbc_trust_hi`
    *   **Default**: `None`
    *   **Risk**: If `uq_trust_mode='manual'`, these are required but strictly checked later, leading to runtime crashes rather than early validation errors.

| Variable | Problem | Risk | Suggested Fix |
| :--- | :--- | :--- | :--- |
| `project` | Implicit `config.get` | 🔴 | Add as explicit arg in `__init__`. |
| `uq_select_scheme` | Implicit `config.get`, default `'tangent_lo'` | 🔴 | Add explicit arg with `Literal` type hint. |
| `uq_trust_ratio` | Magic number `0.33` | 🟡 | Define constant `DEFAULT_TRUST_RATIO = 0.33`. |
| `backend` | Implicit `config.get` | 🔴 | Unify with `JobManager` backend enum. |

#### `train.py` - `TrainingWorkflow`
**Status:** 🟡 **Medium**
**Issues:**
1.  **Environment Coupling**:
    *   **Variable**: `work_dir`
    *   **Default**: `os.getcwd()` (Implicit)
    *   **Risk**: Writing to the current directory by default can clutter the project root or fail in read-only container environments.
2.  **Implicit Model Paths**:
    *   **Variable**: `base_model_path`
    *   **Default**: `None` (Implicit)
    *   **Risk**: `runner/config.json` hardcodes `DPA-3.1-3M.pt`. Code should enforce this as a required argument if no valid default exists.

### 3.2. Feature Generation (`src/dpeva/feature/`)

#### `generator.py` - `DescriptorGenerator`
**Status:** 🟡 **Medium**
**Issues:**
1.  **Hardware Assumption**:
    *   **Variable**: `omp_threads`
    *   **Default**: `24`
    *   **Risk**: Assumes a high-core-count server. On a laptop or shared node, this will cause severe performance degradation or CPU thrashing.
    *   **Fix**: Default to `os.cpu_count()` or a safer low value (e.g., 4).
2.  **Missing Docs**: `model_path`, `head` have basic docs, but `sys` and `nopbc` in internal methods lack descriptions.

### 3.3. Uncertainty (`src/dpeva/uncertain/`)

#### `filter.py` - `UQFilter`
**Status:** 🟡 **Medium**
**Issues:**
1.  **Documentation Gap**:
    *   `rnd_trust_lo`, `rnd_trust_hi` lack clear explanation of how they interact with the primary `trust_lo`.
2.  **Default Value Logic**:
    *   `rnd_trust_lo` defaults to `None`. The code logic falls back to `trust_lo`. This behavior should be explicitly documented.

### 3.4. Submission (`src/dpeva/submission/`)

#### `manager.py` - `JobManager`
**Status:** 🟢 **Good** (Mostly)
**Issues:**
1.  **Default Directory**: `working_dir` defaults to `'.'`. Acceptable for a submission manager but should be noted.
2.  **Type Safety**: `mode` is typed as `Literal['local', 'slurm']` which is excellent practice.

---

## 4. Docstring Remediation Plan

We identified 116 parameters with missing docstrings. Below are examples of how to remediate them.

### Example 1: `ParallelTrainer.__init__`

**Current:**
```python
def __init__(self, base_config_path, work_dir, num_models=4, ...):
    pass
```

**Proposed:**
```python
def __init__(
    self, 
    base_config_path: str, 
    work_dir: str, 
    num_models: int = 4,
    ...
):
    """
    Initialize the parallel trainer.

    Args:
        base_config_path (str): Path to the base training configuration file (JSON).
        work_dir (str): Directory where training artifacts and logs will be saved.
        num_models (int, optional): Number of models to train in the ensemble. Defaults to 4.
            Must be at least 2 for UQ calculation.
    """
```

### Example 2: `DescriptorGenerator` (Fixing Hardware Default)

**Current:**
```python
def __init__(self, ..., omp_threads=24):
    """Number of OMP threads (default: 24)."""
```

**Proposed:**
```python
def __init__(self, ..., omp_threads: int = 4):
    """
    Args:
        omp_threads (int, optional): Number of OpenMP threads for descriptor calculation.
            Defaults to 4. WARNING: Setting this too high on shared resources
            may degrade performance.
    """
```

---

## 5. Recommended Actions & Roadmap

### Phase 1: Documentation (Immediate)
1.  **Fill Gaps**: Apply the docstring updates to all 116 missing parameters identified in `parameter_audit.json`.
2.  **Clarify Defaults**: Explicitly document the "fallback behavior" for implicit parameters (e.g., "If `uq_trust_ratio` is not provided in config, defaults to 0.33").

### Phase 2: Safety & Defaults (Short-term)
1.  **Fix `omp_threads`**: Change default from `24` to a safer value (`4` or `os.cpu_count() // 2`).
2.  **Centralize Constants**: Move magic numbers (like `0.33`, `0.12`, `0.22`) to a `defaults.py` module.
3.  **Sanitize Paths**: Ensure no default paths refer to `/home/pku-jianghong/...`. (Audit showed most are `./` or `None`, which is good).

### Phase 3: Refactoring (Long-term)
1.  **Adopt Pydantic**: Replace the `config` dictionary in `CollectionWorkflow` and others with a Pydantic `BaseModel`.
    *   **Benefit**: Free type validation, auto-generated schema/docs, no more "implicit" parameters.
    *   **Example**:
        ```python
        class CollectionConfig(BaseModel):
            project: str
            uq_select_scheme: Literal['tangent_lo', 'strict'] = 'tangent_lo'
            uq_trust_ratio: float = Field(0.33, ge=0.0, le=1.0)
        ```
2.  **Explicit APIs**: Refactor `__init__` methods to take the Config object or explicit arguments, banning `**kwargs` or opaque `config` dicts for core logic.

---

**End of Report**
