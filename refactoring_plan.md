# DP-EVA Parameter Refactoring Plan

## 1. Architecture Overview

We will transition from implicit `dict` configuration to explicit `Pydantic` models. This ensures type safety, validation, and self-documentation.

### 1.1 New Module Structure

```
src/dpeva/
├── constants.py       # Centralized default values (e.g., DEFAULT_OMP_THREADS)
├── config.py          # Pydantic models (BaseConfig, TrainConfig, etc.)
└── workflows/
    └── ...            # Workflows updated to use Config objects
```

### 1.2 Naming Standardization

| Old Name | New Name | Scope |
| :--- | :--- | :--- |
| `testing_head` | `model_head` | Collection |
| `finetune_head_name` | `model_head` | Train |
| `head` | `model_head` | Infer/Feature |
| `modelpath` | `model_path` | Feature |
| `omp_threads` | `omp_threads` | Global (Default: 1) |
| `backend` | `submission.backend` | Global (Nested) |
| `slurm_config` | `submission.slurm_config` | Global (Nested) |

---

## 2. Configuration Models (Draft)

### 2.1 Base Configuration (`dpeva.config.BaseConfig`)
Handles common infrastructure settings.

```python
class SubmissionConfig(BaseModel):
    backend: Literal["local", "slurm"] = "local"
    slurm_config: Dict[str, Any] = Field(default_factory=dict)
    env_setup: Union[str, List[str]] = ""

class BaseWorkflowConfig(BaseModel):
    submission: SubmissionConfig = Field(default_factory=SubmissionConfig)
    omp_threads: int = Field(default=DEFAULT_OMP_THREADS, ge=1)
    work_dir: Path = Field(default_factory=Path.cwd)
```

### 2.2 Feature Config (`dpeva.config.FeatureConfig`)

```python
class FeatureConfig(BaseWorkflowConfig):
    data_path: Path
    model_path: Path
    model_head: str = "OC20M"
    format: str = "deepmd/npy"
    output_mode: Literal["atomic", "structural"] = "atomic"
    batch_size: int = Field(1000, gt=0)
    mode: Literal["cli", "python"] = "cli"
    # Legacy alias support
    @model_validator(mode='before')
    def align_legacy_keys(cls, values):
        # map 'head' -> 'model_head', 'modelpath' -> 'model_path'
        ...
```

### 2.3 Training Config (`dpeva.config.TrainingConfig`)

```python
class TrainingConfig(BaseWorkflowConfig):
    base_model_path: Path
    num_models: int = Field(4, ge=2)
    training_mode: Literal["init", "cont"] = Field("cont", alias="mode")
    model_head: str = Field("Hybrid_Perovskite", alias="finetune_head_name")
    ...
```

### 2.4 Inference Config (`dpeva.config.InferenceConfig`)

```python
class InferenceConfig(BaseWorkflowConfig):
    data_path: Path
    model_head: str = Field("Hybrid_Perovskite", alias="head")
    ...
```

### 2.5 Collection Config (`dpeva.config.CollectionConfig`)

```python
class CollectionConfig(BaseWorkflowConfig):
    project: str = "stage9-2"
    # UQ Parameters
    uq_select_scheme: Literal["tangent_lo", "strict", ...] = "tangent_lo"
    uq_trust_ratio: float = Field(0.33, ge=0.0, le=1.0)
    # ...
```

---

## 3. Implementation Steps

1.  **Dependencies**: Add `pydantic` to `pyproject.toml`.
2.  **Constants**: Create `src/dpeva/constants.py`.
3.  **Config Module**: Create `src/dpeva/config.py` with full Pydantic models and legacy aliases.
4.  **Workflow Refactoring**:
    -   Modify `__init__` to accept `Union[Dict, ConfigModel]`.
    -   If `Dict`, parse it into `ConfigModel` (validating immediately).
    -   Replace `self.config.get()` calls with attribute access `self.config.field`.
5.  **Validation**: Run existing tests.

## 4. Risk Mitigation

-   **Backward Compatibility**: The `model_validator(mode='before')` in Pydantic will handle old keys transparently, issuing warnings if needed.
-   **Gradual Rollout**: Workflows can be updated one by one.
