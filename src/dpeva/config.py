"""
Centralized Configuration Management using Pydantic V2.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

from dpeva.constants import (
    DEFAULT_OMP_THREADS,
    DEFAULT_BACKEND,
    DEFAULT_NUM_MODELS,
    DEFAULT_DP_BACKEND,
    VALID_DP_BACKENDS,
    DEFAULT_DESC_BATCH_SIZE,
    DEFAULT_DESC_OUTPUT_MODE,
    DEFAULT_FEATURE_MODE,
    DEFAULT_INFER_TASK_NAME,
    DEFAULT_RESULTS_PREFIX,
    DEFAULT_INPUT_JSON,
    DEFAULT_PROJECT_DIR,
    DEFAULT_COLLECT_ROOT_DIR,
    DEFAULT_TESTING_DIR,
    DEFAULT_UQ_SCHEME,
    DEFAULT_UQ_TRUST_RATIO,
    DEFAULT_UQ_TRUST_WIDTH,
    DEFAULT_DIRECT_K,
    DEFAULT_DIRECT_THR_INIT,
    DEFAULT_STEP1_N_CLUSTERS,
    DEFAULT_STEP1_THRESHOLD,
    DEFAULT_STEP2_N_CLUSTERS,
    DEFAULT_STEP2_THRESHOLD,
    DEFAULT_STEP2_K,
    DEFAULT_STEP2_SELECTION,
    DEFAULT_ANALYSIS_OUTPUT_DIR,
    FIG_DPI,
)

class SubmissionConfig(BaseModel):
    """Configuration for job submission."""
    backend: Literal["local", "slurm"] = Field(
        default=DEFAULT_BACKEND, 
        description="Execution backend."
    )
    slurm_config: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Slurm specific parameters (partition, nodes, etc)."
    )
    env_setup: Union[str, List[str]] = Field(
        default="", 
        description="Environment setup commands."
    )

    @field_validator("env_setup")
    @classmethod
    def validate_env_setup(cls, v):
        if isinstance(v, list):
            return "\n".join(v)
        return v

class BaseWorkflowConfig(BaseModel):
    """Base configuration for all workflows."""
    model_config = ConfigDict(extra='ignore', populate_by_name=True, protected_namespaces=())

    work_dir: Path = Field(
        default_factory=Path.cwd, 
        description="Working directory."
    )
    omp_threads: Union[int, Literal["auto"]] = Field(
        default=DEFAULT_OMP_THREADS, 
        description="OpenMP threads (set to 'auto' to use all cores)."
    )
    dp_backend: str = Field(
        default=DEFAULT_DP_BACKEND,
        description="DeepMD-kit backend (e.g. 'pt', 'tf', 'jax', 'pd')."
    )
    submission: SubmissionConfig = Field(
        default_factory=SubmissionConfig, 
        description="Submission configuration."
    )
    
    @field_validator("dp_backend")
    @classmethod
    def validate_dp_backend(cls, v):
        if v not in VALID_DP_BACKENDS:
            raise ValueError(f"Invalid dp_backend '{v}'. Valid options: {VALID_DP_BACKENDS}")
        return v

    @field_validator("omp_threads")
    @classmethod
    def validate_omp_threads(cls, v):
        if v == "auto":
            return os.cpu_count() or 1
        if isinstance(v, int) and v < 1:
            raise ValueError("omp_threads must be >= 1")
        return v

    @model_validator(mode='before')
    @classmethod
    def extract_flat_submission_config(cls, data: Any) -> Any:
        """
        Allow flat config to populate nested submission config for backward compatibility.
        e.g. {'backend': 'slurm'} -> {'submission': {'backend': 'slurm'}}
        """
        if isinstance(data, dict):
            # If submission is not explicitly provided, try to build it from flat keys
            if "submission" not in data:
                backend = data.get("backend", DEFAULT_BACKEND)
                slurm_config = data.get("slurm_config", {})
                env_setup = data.get("env_setup", "")
                
                # Only create if there's something non-default or if backend is present
                # But to be safe, we always populate it
                data["submission"] = {
                    "backend": backend,
                    "slurm_config": slurm_config,
                    "env_setup": env_setup
                }
        return data

class FeatureConfig(BaseWorkflowConfig):
    """Configuration for Feature Generation Workflow."""
    data_path: Path = Field(..., description="Path to dataset.")
    model_path: Path = Field(..., description="Path to model file.")
    model_head: str = Field(..., description="Model head name.")
    
    output_mode: Literal["atomic", "structural"] = DEFAULT_DESC_OUTPUT_MODE
    batch_size: int = Field(DEFAULT_DESC_BATCH_SIZE, gt=0)
    mode: Literal["cli", "python"] = DEFAULT_FEATURE_MODE
    
    savedir: Optional[Path] = None

    @model_validator(mode='after')
    def set_default_savedir(self):
        if self.savedir is None:
            model_name = self.model_path.stem
            data_name = self.data_path.name
            self.savedir = Path(f"desc-{model_name}-{data_name}")
        return self

class AnalysisConfig(BaseModel):
    """Configuration for Analysis Workflow (Post-processing)."""
    model_config = ConfigDict(extra='ignore', populate_by_name=True)
    
    result_dir: Path = Field(..., description="Path to DP test results directory.")
    output_dir: Path = Field(Path(DEFAULT_ANALYSIS_OUTPUT_DIR), description="Output directory for analysis results.")
    type_map: List[str] = Field(..., description="Atom type map (e.g. ['Fe', 'C']).")
    data_path: Optional[Path] = Field(None, description="Path to original dataset (for robust composition loading).")
    ref_energies: Dict[str, float] = Field(default_factory=dict, description="Reference energies per element for cohesive energy calculation.")

class InferenceConfig(BaseWorkflowConfig):
    """Configuration for Inference Workflow."""
    data_path: Path = Field(..., description="Path to test dataset.")
    model_head: Optional[str] = Field(None, description="Model head name (optional for frozen models).")
    results_prefix: str = Field(DEFAULT_RESULTS_PREFIX, description="Output file prefix.")
    task_name: str = DEFAULT_INFER_TASK_NAME
    ref_energies: Dict[str, float] = Field(default_factory=dict, description="Reference energies per element for cohesive energy calculation.")

class TrainingConfig(BaseWorkflowConfig):
    """Configuration for Training Workflow."""
    base_model_path: Path = Field(..., description="Path to base model.")
    num_models: int = Field(DEFAULT_NUM_MODELS, ge=3, description="Number of models (>=3 for UQ).")
    training_mode: Literal["init", "cont"] = Field("init")
    model_head: str = Field(..., description="Model head name.")
    
    input_json_path: Path = Path(DEFAULT_INPUT_JSON)
    seeds: Optional[List[int]] = None
    training_seeds: Optional[List[int]] = None
    template_path: Optional[Path] = None
    training_data_path: Optional[Path] = None

class CollectionConfig(BaseWorkflowConfig):
    """Configuration for Collection (Active Learning) Workflow."""
    project: str = Field(DEFAULT_PROJECT_DIR, description="Project directory.")
    
    # Paths
    desc_dir: Path = Field(..., description="Descriptor directory.")
    testdata_dir: Path = Field(..., description="Test data directory.")
    
    # Optional Paths (for Joint Sampling)
    training_data_dir: Optional[Path] = None
    training_desc_dir: Optional[Path] = None
    
    root_savedir: Path = Path(DEFAULT_COLLECT_ROOT_DIR)
    
    # Testing
    testing_dir: str = DEFAULT_TESTING_DIR
    results_prefix: str = Field(DEFAULT_RESULTS_PREFIX, description="Output file prefix.")
    fig_dpi: int = Field(FIG_DPI, description="DPI for visualization figures.")
    
    # UQ Parameters
    num_models: int = Field(DEFAULT_NUM_MODELS, ge=3, description="Number of models for UQ calculation.")
    uq_select_scheme: Literal["tangent_lo", "strict", "circle_lo", "crossline_lo", "loose"] = DEFAULT_UQ_SCHEME
    uq_trust_mode: Literal["auto", "manual", "no_filter"] = "auto"
    uq_trust_ratio: float = Field(DEFAULT_UQ_TRUST_RATIO, ge=0.0, le=1.0)
    uq_trust_width: float = Field(DEFAULT_UQ_TRUST_WIDTH, gt=0.0)
    
    # Specific Trust Bounds (Manual Mode)
    uq_qbc_trust_lo: Optional[float] = None
    uq_qbc_trust_hi: Optional[float] = None
    uq_rnd_rescaled_trust_lo: Optional[float] = None
    uq_rnd_rescaled_trust_hi: Optional[float] = None
    
    # Specific Trust Params (Overrides)
    uq_qbc_trust_ratio: Optional[float] = None
    uq_qbc_trust_width: Optional[float] = None
    uq_rnd_rescaled_trust_ratio: Optional[float] = None
    uq_rnd_rescaled_trust_width: Optional[float] = None

    # Auto UQ Bounds
    uq_auto_bounds: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    
    # Sampling
    sampler_type: Literal["direct", "2-direct"] = "direct"
    
    # Standard DIRECT Parameters
    direct_n_clusters: Optional[int] = Field(None, gt=0, description="Target number of clusters for Standard DIRECT.")
    
    direct_k: int = Field(DEFAULT_DIRECT_K, ge=1)
    direct_thr_init: float = Field(DEFAULT_DIRECT_THR_INIT, ge=0.0)
    
    # 2-DIRECT Parameters
    step1_n_clusters: Optional[int] = Field(DEFAULT_STEP1_N_CLUSTERS, gt=1, description="Number of clusters for Step 1. If None, dynamic clustering based on threshold.")
    step1_threshold: float = Field(DEFAULT_STEP1_THRESHOLD, gt=0.0, description="Threshold for Step 1 Birch clustering.")
    step2_n_clusters: Optional[int] = Field(DEFAULT_STEP2_N_CLUSTERS, gt=1, description="Number of clusters for Step 2. If None, dynamic clustering based on threshold.")
    step2_threshold: float = Field(DEFAULT_STEP2_THRESHOLD, gt=0.0, description="Threshold for Step 2 Birch clustering.")
    step2_k: int = Field(DEFAULT_STEP2_K, ge=1, description="Number of samples per atomic cluster in Step 2.")
    step2_selection: Literal["smallest", "center", "random"] = Field(
        DEFAULT_STEP2_SELECTION, 
        description="Selection criteria for Step 2. 'smallest': fewest atoms; 'center': closest to cluster centroid; 'random': random sampling."
    )
    
    # Config File Path (for self-submission)
    config_path: Optional[Path] = None

    @model_validator(mode='after')
    def validate_manual_trust_bounds(self):
        if self.uq_trust_mode == "manual":
            # QbC Validation & Calculation
            if self.uq_qbc_trust_lo is None:
                raise ValueError("In 'manual' trust mode, uq_qbc_trust_lo must be provided.")
            
            if self.uq_qbc_trust_hi is None:
                # Use specific width if available, else global
                width = self.uq_qbc_trust_width if self.uq_qbc_trust_width is not None else self.uq_trust_width
                if width is not None:
                    self.uq_qbc_trust_hi = self.uq_qbc_trust_lo + width
                else:
                    raise ValueError("In 'manual' trust mode, provide uq_qbc_trust_hi OR width.")
            
            # RND Validation & Calculation
            if self.uq_rnd_rescaled_trust_lo is None:
                raise ValueError("In 'manual' trust mode, uq_rnd_rescaled_trust_lo must be provided.")
                
            if self.uq_rnd_rescaled_trust_hi is None:
                width = self.uq_rnd_rescaled_trust_width if self.uq_rnd_rescaled_trust_width is not None else self.uq_trust_width
                if width is not None:
                    self.uq_rnd_rescaled_trust_hi = self.uq_rnd_rescaled_trust_lo + width
                else:
                    raise ValueError("In 'manual' trust mode, provide uq_rnd_rescaled_trust_hi OR width.")
                    
        return self
