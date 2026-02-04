"""
Centralized constants and default values for DP-EVA.
"""
from typing import Final, Optional

# Infrastructure Defaults
DEFAULT_OMP_THREADS: Final[int] = 4
DEFAULT_BACKEND: Final[str] = "local"
DEFAULT_SLURM_PARTITION: Final[Optional[str]] = None
DEFAULT_WALLTIME: Final[str] = "24:00:00"
DEFAULT_SLURM_GPUS_PER_NODE: Final[Optional[int]] = None
DEFAULT_SLURM_CPUS_PER_TASK: Final[Optional[int]] = None
DEFAULT_SLURM_QOS: Final[Optional[str]] = None
DEFAULT_SLURM_NODELIST: Final[Optional[str]] = None
DEFAULT_SLURM_NODES: Final[int] = 1
DEFAULT_SLURM_NTASKS: Final[int] = 1
DEFAULT_SLURM_CUSTOM_HEADERS: Final[str] = ""

# Model Defaults
DEFAULT_NUM_MODELS: Final[int] = 4
DEFAULT_DP_BACKEND: Final[str] = "pt"
VALID_DP_BACKENDS: Final[list] = ["pt", "tf", "jax", "pd"]
MIN_DEEPMD_VERSION: Final[str] = "3.1.2"

# Feature/Descriptor Defaults
DEFAULT_FEATURE_MODE: Final[str] = "cli"
DEFAULT_DESC_BATCH_SIZE: Final[int] = 1000
DEFAULT_DESC_OUTPUT_MODE: Final[str] = "atomic"

# Inference/Training Defaults
DEFAULT_INFER_TASK_NAME: Final[str] = "test"
DEFAULT_RESULTS_PREFIX: Final[str] = "results"
DEFAULT_INPUT_JSON: Final[str] = "input.json"

# Collection/Active Learning Defaults
DEFAULT_PROJECT_DIR: Final[str] = "./"
DEFAULT_COLLECT_ROOT_DIR: Final[str] = "dpeva_uq_post"
DEFAULT_TESTING_DIR: Final[str] = "test_results"

# Visualization
FIG_DPI: Final[int] = 300

# UQ Defaults
DEFAULT_UQ_SCHEME: Final[str] = "tangent_lo"
DEFAULT_UQ_TRUST_RATIO: Final[float] = 0.33
DEFAULT_UQ_TRUST_WIDTH: Final[float] = 0.25
DEFAULT_NUM_SELECTION: Final[int] = 100
DEFAULT_DIRECT_K: Final[int] = 1
DEFAULT_DIRECT_THR_INIT: Final[float] = 0.5

# Workflow Tags
WORKFLOW_FINISHED_TAG: Final[str] = "DPEVA_TAG: WORKFLOW_FINISHED"

# Filenames
FILENAME_STATS_JSON: Final[str] = "statistics.json"
FILENAME_SUMMARY_CSV: Final[str] = "inference_summary.csv"

# Column Names
COL_DESC_PREFIX: Final[str] = "desc_stru_"
COL_UQ_QBC: Final[str] = "uq_qbc_for"
COL_UQ_RND: Final[str] = "uq_rnd_for"
