import os
import shutil
import glob
import logging
import sys
from typing import Union, Dict
import numpy as np
import pandas as pd

from dpeva.io.dataproc import DPTestResultParser
from dpeva.io.types import PredictionData
from dpeva.io.dataset import load_systems
from dpeva.config import CollectionConfig
from dpeva.constants import WORKFLOW_FINISHED_TAG, COL_DESC_PREFIX, COL_UQ_QBC, COL_UQ_RND, DEFAULT_LOG_FILE
from dpeva.sampling.direct import BirchClustering, DIRECTSampler, SelectKFromClusters
from dpeva.sampling.two_step_direct import TwoStepDIRECTSampler
from dpeva.uncertain.calculator import UQCalculator
from dpeva.uncertain.filter import UQFilter
from dpeva.uncertain.visualization import UQVisualizer
from dpeva.submission.manager import JobManager
from dpeva.submission.templates import JobConfig

# Helpers
def get_sys_name(dataname):
    return dataname.rsplit("-", 1)[0]

def get_pool_name(sys_name):
    d = os.path.dirname(sys_name)
    return d if d else "root"

class CollectionWorkflow:
    """
    Orchestrates the Collection pipeline:
    Data Loading -> UQ Calculation -> Filtering -> DIRECT Sampling -> Visualization -> Export
    """

    def __init__(self, config: Union[Dict, CollectionConfig], config_path=None):
        """
        Initialize the Collection Workflow.

        Args:
            config (Union[dict, CollectionConfig]): Configuration object or dictionary.
            config_path (str, optional): Path to the configuration file. 
        """
        self._setup_logger()
        
        # 1. Configuration Loading
        if isinstance(config, dict):
            # Inject config_path if present
            if config_path and "config_path" not in config:
                config["config_path"] = config_path
            self.config = CollectionConfig(**config)
        else:
            self.config = config
            if config_path and self.config.config_path is None:
                 try:
                     self.config.config_path = config_path
                 except Exception:
                     pass

        self.config_path = str(self.config.config_path) if self.config.config_path else None
        
        # 2. Basic Attributes
        self.project = self.config.project
        self.uq_scheme = self.config.uq_select_scheme
        
        # Backend
        # Allow environment override for internal recursion prevention
        env_backend = os.environ.get("DPEVA_INTERNAL_BACKEND")
        if env_backend:
            self.logger.info(f"Overriding backend to '{env_backend}' via DPEVA_INTERNAL_BACKEND environment variable.")
            self.backend = env_backend
        else:
            self.backend = self.config.submission.backend
            
        self.slurm_config = self.config.submission.slurm_config
        
        # Paths
        self.testing_dir = self.config.testing_dir
        self.testing_head = self.config.results_prefix
        self.desc_dir = str(self.config.desc_dir)
        self.testdata_dir = str(self.config.testdata_dir)
        
        self.training_data_dir = str(self.config.training_data_dir) if self.config.training_data_dir else None
        self.training_desc_dir = str(self.config.training_desc_dir) if self.config.training_desc_dir else None
        
        self.root_savedir = str(self.config.root_savedir)
        self.view_savedir = os.path.join(self.project, self.root_savedir, "view")
        self.dpdata_savedir = os.path.join(self.project, self.root_savedir, "dpdata")
        self.df_savedir = os.path.join(self.project, self.root_savedir, "dataframe")
        
        # 3. Validation & Setup
        self._validate_config()
        self._ensure_dirs()
        self._configure_file_logging()
        
        # UQ Parameters
        self.uq_trust_mode = self.config.uq_trust_mode
        self.global_trust_ratio = self.config.uq_trust_ratio
        self.global_trust_width = self.config.uq_trust_width
        
        # Resolve QbC Parameters
        self.uq_qbc_params = {
            "ratio": self.config.uq_qbc_trust_ratio if self.config.uq_qbc_trust_ratio is not None else self.global_trust_ratio,
            "width": self.config.uq_qbc_trust_width if self.config.uq_qbc_trust_width is not None else self.global_trust_width,
            "lo": self.config.uq_qbc_trust_lo,
            "hi": self.config.uq_qbc_trust_hi
        }
        
        # Resolve RND Parameters
        self.uq_rnd_params = {
            "ratio": self.config.uq_rnd_rescaled_trust_ratio if self.config.uq_rnd_rescaled_trust_ratio is not None else self.global_trust_ratio,
            "width": self.config.uq_rnd_rescaled_trust_width if self.config.uq_rnd_rescaled_trust_width is not None else self.global_trust_width,
            "lo": self.config.uq_rnd_rescaled_trust_lo,
            "hi": self.config.uq_rnd_rescaled_trust_hi
        }

        # Validate Parameters based on Mode
        if self.uq_trust_mode == "manual":
            self._validate_manual_params(self.uq_qbc_params, "uq_qbc")
            self._validate_manual_params(self.uq_rnd_params, "uq_rnd")
        elif self.uq_trust_mode == "auto":
            pass
        elif self.uq_trust_mode == "no_filter":
            pass
        
        # Map back to instance variables
        self.uq_qbc_trust_lo = self.uq_qbc_params.get("lo")
        self.uq_qbc_trust_hi = self.uq_qbc_params.get("hi")
        self.uq_rnd_trust_lo = self.uq_rnd_params.get("lo")
        self.uq_rnd_trust_hi = self.uq_rnd_params.get("hi")
        
        # UQ Auto Bounds
        self.uq_auto_bounds = self.config.uq_auto_bounds
        
        # Sampling Parameters
        self.direct_k = self.config.direct_k
        self.direct_thr_init = self.config.direct_thr_init

    def _validate_manual_params(self, params, name):
        """
        Validates and fills parameters for Manual mode.
        Requires 'lo' to be present.
        Calculates 'hi' from 'width' if needed.
        """
        lo = params.get("lo")
        hi = params.get("hi")
        width = params.get("width")
        
        if lo is None:
            raise ValueError(f"[{name}] 'lo' value must be specified in 'manual' mode!")
            
        if hi is None:
            if width is not None:
                params["hi"] = lo + width
                self.logger.info(f"{name}: Calculated hi={params['hi']:.4f} from lo={lo}, width={width}")
            else:
                raise ValueError(f"[{name}] Either 'hi' or 'width' must be specified with 'lo'!")
        else:
            # Check consistency
            if width is not None:
                calc_width = hi - lo
                if abs(calc_width - width) > 1e-5:
                     self.logger.error(f"[{name}] Configuration Conflict: lo={lo}, hi={hi} implies width={calc_width:.4f}, but width is set to {width:.4f}")
                     raise ValueError(f"[{name}] Configuration Conflict: lo + width != hi")
            else:
                params["width"] = hi - lo
                self.logger.info(f"{name}: Derived width {params['width']:.4f} from lo={lo}, hi={hi}")

    def _clamp_trust_lo(self, value, bounds, name="UQ"):
        """
        Clamps the auto-calculated trust_lo value within specified bounds.
        
        Args:
            value (float): The auto-calculated value.
            bounds (dict): Dictionary containing 'lo_min' and/or 'lo_max'.
            name (str): Name of the parameter for logging.
            
        Returns:
            float: The clamped value.
        """
        if value is None:
            return None
            
        lo_min = bounds.get("lo_min")
        lo_max = bounds.get("lo_max")
        
        clamped_value = value
        
        if lo_min is not None and clamped_value < lo_min:
            self.logger.warning(f"{name}: Auto-calculated value {value:.4f} is lower than min bound {lo_min}. Clamping to {lo_min}.")
            clamped_value = lo_min
            
        if lo_max is not None and clamped_value > lo_max:
            self.logger.warning(f"{name}: Auto-calculated value {value:.4f} is higher than max bound {lo_max}. Clamping to {lo_max}.")
            clamped_value = lo_max
            
        return clamped_value
        
    def _setup_logger(self):
        """Sets up the logging configuration."""
        self.logger = logging.getLogger(__name__)

    def _configure_file_logging(self):
        """Configures file logging to the output directory."""
        log_file = os.path.join(self.project, self.root_savedir, DEFAULT_LOG_FILE)
        
        # Check if handler already exists to avoid duplicates
        for h in self.logger.handlers:
            if isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(log_file):
                return

        file_handler = logging.FileHandler(log_file, mode='w')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.info(f"Logging configured to file: {log_file}")

    def _validate_config(self):
        """Validates that necessary configuration paths exist."""
        # Basic validation
        if not os.path.exists(self.project):
            self.logger.error(f"Project directory {self.project} not found!")
            raise ValueError(f"Project directory {self.project} not found!")
            
        # Pydantic ensures keys exist, but we still check path existence
        if not os.path.exists(self.desc_dir):
             self.logger.error(f"Descriptor directory not found: {self.desc_dir}")
             raise ValueError(f"Descriptor directory not found: {self.desc_dir}")
             
        if not os.path.exists(self.testdata_dir):
             self.logger.error(f"Test data directory not found: {self.testdata_dir}")
             raise ValueError(f"Test data directory not found: {self.testdata_dir}")

    def _ensure_dirs(self):
        """Creates necessary output directories if they don't exist."""
        for d in [self.view_savedir, self.dpdata_savedir, self.df_savedir]:
            if not os.path.exists(d):
                os.makedirs(d)

    def _load_descriptors(self, desc_dir, label="descriptors", target_names=None, expected_frames=None):
        """
        Loads descriptors from a directory.
        Only supports flat structure (*.npy) as nested structure is deprecated.
        
        Args:
            desc_dir (str): Path to the descriptor directory.
            label (str): Label for logging purposes.
            target_names (list): List of system names (without index) to load specifically. 
                               If provided, loads only these systems in this order.
            expected_frames (dict): Optional dict {sys_name: n_frames} to enforce consistency.

        Returns:
            tuple: (desc_datanames, desc_stru)
        """
        self.logger.info(f"Loading {label} from {desc_dir}")
        
        desc_datanames = []
        desc_stru = []
        
        if target_names:
            self.logger.info(f"Loading {len(target_names)} specific systems based on target names.")
            
            for sys_name in target_names:
                # Construct possible paths
                # 1. Flat: desc_dir/sys_name.npy
                
                # Try direct path first (most common for multi-pool: desc_dir/Dataset/System.npy)
                path_flat = os.path.join(desc_dir, f"{sys_name}.npy")
                
                # Fallback: Check for flat basename match (e.g. desc_dir/System.npy even if sys_name is Dataset/System)
                path_flat_base = os.path.join(desc_dir, f"{os.path.basename(sys_name)}.npy")
                
                if os.path.exists(path_flat):
                    f = path_flat
                elif os.path.exists(path_flat_base):
                    # In Single Data Pool mode, sys_name might be 'pool/sys' but desc is just 'sys.npy'
                    # This is a valid compatibility match, not a warning condition.
                    self.logger.info(f"Matched descriptor via basename (Single-Pool Compatible): {path_flat_base} for system {sys_name}")
                    f = path_flat_base
                else:
                    self.logger.error(f"Descriptor file not found for system: {sys_name}. Expected at {path_flat}")
                    raise FileNotFoundError(f"Descriptor file missing for {sys_name}")
                
                try:
                    one_desc = np.load(f)
                    
                    # Consistency Check
                    if expected_frames and sys_name in expected_frames:
                        n_exp = expected_frames[sys_name]
                        n_got = one_desc.shape[0]
                        if n_got != n_exp:
                            if n_got > n_exp:
                                self.logger.warning(f"Descriptor frame mismatch for {sys_name}: Expected {n_exp}, Got {n_got}. Truncating.")
                                one_desc = one_desc[:n_exp]
                            else:
                                self.logger.error(f"Descriptor frame mismatch for {sys_name}: Expected {n_exp}, Got {n_got}. Missing frames!")
                                raise ValueError(f"Missing descriptor frames for {sys_name}")
                                
                except Exception as e:
                    self.logger.error(f"Failed to load descriptor file {f}: {e}")
                    raise
                    
                # Use sys_name as keyname to ensure match with target
                keyname = sys_name
                
                # Optimized: List comprehension for faster string generation
                desc_datanames.extend([f"{keyname}-{i}" for i in range(len(one_desc))])
                
                # Mean pooling and L2 normalization per frame
                one_desc_stru = np.mean(one_desc, axis=1) # (n_frames, n_desc)
                
                # L2 Normalization
                stru_modulo = np.linalg.norm(one_desc_stru, axis=1, keepdims=True)
                one_desc_stru_norm = one_desc_stru / (stru_modulo + 1e-12)
                desc_stru.append(one_desc_stru_norm)
                
        else:
            # Check for different file patterns
            flat_pattern = os.path.join(desc_dir, "*.npy")
            
            # Determine which pattern to use
            # Priority: User explicit wildcard > Flat (*.npy)
            
            if '*' in desc_dir:
                 desc_pattern = desc_dir
            else:
                 desc_pattern = flat_pattern
                     
            desc_iter_list = sorted(glob.glob(desc_pattern))
            
            if not desc_iter_list:
                 self.logger.warning(f"No {label} found in {desc_dir}")
                 return [], np.array([])
            
            for f in desc_iter_list:
                # Determine keyname based on structure
                # Flat: .../sysname.npy -> sysname
                # Flat structure (or user wildcard matching npy)
                keyname = os.path.basename(f).replace('.npy', '')
                     
                try:
                    one_desc = np.load(f)
                except Exception as e:
                    self.logger.error(f"Failed to load descriptor file {f}: {e}")
                    continue
                    
                # Optimized: List comprehension for faster string generation
                desc_datanames.extend([f"{keyname}-{i}" for i in range(len(one_desc))])
                
                # Mean pooling and L2 normalization per frame
                one_desc_stru = np.mean(one_desc, axis=1) # (n_frames, n_desc)
                
                # L2 Normalization
                stru_modulo = np.linalg.norm(one_desc_stru, axis=1, keepdims=True)
                one_desc_stru_norm = one_desc_stru / (stru_modulo + 1e-12)
                desc_stru.append(one_desc_stru_norm)
        
        if len(desc_stru) > 0:
            desc_stru = np.concatenate(desc_stru, axis=0)
        else:
            return [], np.array([])
            
        return desc_datanames, desc_stru

    def _count_frames_in_data(self, data_dir, fmt="auto"):
        """Counts total frames in dataset to verify consistency."""
        try:
            # Use load_systems for robust loading and auto-detection
            systems = load_systems(data_dir, fmt=fmt)
            total_frames = sum(len(sys) for sys in systems)
            return total_frames
        except Exception as e:
            self.logger.warning(f"Failed to count frames in {data_dir}: {e}")
            return 0

    def _submit_to_slurm(self):
        """
        Submits the workflow to Slurm backend.
        Optimized to use self-invocation pattern to avoid generating redundant files.
        """
        self.logger.info("Backend is set to 'slurm'. Preparing submission...")
        
        project_abs = os.path.abspath(self.project)
        if not os.path.exists(project_abs):
            os.makedirs(project_abs)

        # 1. Identify Runner and Config
        # If config_path is provided (e.g. from run_uq_collect.py), use it.
        if self.config_path:
            config_abs_path = os.path.abspath(self.config_path)
            
            self.logger.info(f"Using Self-Invocation Mode via CLI:")
            self.logger.info(f"  - Config: {config_abs_path}")
            
            # Construct command to re-run using the standard CLI module
            # We use 'python -m dpeva.cli collect' to ensure we use the installed package
            # and avoid depending on sys.argv[0] (which might be a random script)
            python_exe = sys.executable
            cmd = f"{python_exe} -m dpeva.cli collect {config_abs_path}"
            
            # Set environment variable for the job
            env_setup = "export DPEVA_INTERNAL_BACKEND=local\n"
            
        else:
            # Require config_path for Slurm mode
            self.logger.error("config_path is required for Slurm submission!")
            raise ValueError("config_path is missing. Cannot submit to Slurm.")

        # 2. Generate Slurm Script via JobManager
        # Defaults based on user request
        partition = self.slurm_config.get("partition", "CPU-MISC")
        ntasks = self.slurm_config.get("ntasks", 4)
        qos = self.slurm_config.get("qos", "rush-cpu")
        job_name = self.slurm_config.get("job_name", "dpeva_collect")
        
        job_conf = JobConfig(
            command=cmd,
            job_name=job_name,
            partition=partition,
            ntasks=ntasks,
            cpus_per_task=self.slurm_config.get("cpus_per_task", 1),
            qos=qos,
            output_log=os.path.join(project_abs, "collect_slurm.out"),
            error_log=os.path.join(project_abs, "collect_slurm.err"),
            nodes=1, # Default
            env_setup=env_setup
        )
        
        manager = JobManager(mode="slurm")
        
        # Use a consistent script name
        script_name = "submit_collect.slurm"
        script_path = os.path.join(project_abs, script_name)
        
        self.logger.info(f"Submitting job from {project_abs}")
        
        manager.generate_script(job_conf, script_path)
        manager.submit(script_path, working_dir=project_abs)

    def _prepare_features_for_direct(self, df_candidate, df_desc):
        """
        Prepares features for DIRECT sampling, handling joint sampling logic if training data is provided.
        
        Returns:
            tuple: (features_for_direct, use_joint_sampling, n_candidates)
        """
        use_joint_sampling = False
        train_desc_stru = np.array([])
        
        if self.training_desc_dir:
            self.logger.info(f"Training descriptors provided at {self.training_desc_dir}. Attempting joint sampling.")
            _, train_desc_stru = self._load_descriptors(self.training_desc_dir, "training descriptors")
            
            if len(train_desc_stru) > 0:
                # Consistency Check
                if self.training_data_dir:
                    self.logger.info(f"Verifying training data consistency from {self.training_data_dir}")
                    n_train_frames = self._count_frames_in_data(self.training_data_dir)
                    if n_train_frames != len(train_desc_stru):
                        self.logger.error(f"Training set mismatch: Found {n_train_frames} frames in data but {len(train_desc_stru)} frames in descriptors.")
                        raise ValueError("Training data frame count mismatch with descriptors!")
                    else:
                        self.logger.info(f"Training set verified: {n_train_frames} frames.")
                else:
                    self.logger.warning("Training descriptors provided but training data path not specified. "
                                        "Consistency check skipped. Proceeding with joint sampling.")
                
                use_joint_sampling = True
                self.logger.info(f"Joint sampling enabled: {len(df_candidate)} candidates + {len(train_desc_stru)} training samples.")
            else:
                self.logger.warning("Training descriptors provided but empty or failed to load. Falling back to candidate-only sampling.")
        
        # Prepare Features for DIRECT
        candidate_features = df_candidate[[col for col in df_desc.columns if col.startswith(COL_DESC_PREFIX)]].values
        n_candidates = len(candidate_features)
        
        if use_joint_sampling:
            # Combine [Candidate; Training]
            combined_features = np.vstack([candidate_features, train_desc_stru])
            self.logger.info(f"Combined feature shape: {combined_features.shape}")
            features_for_direct = combined_features
        else:
            features_for_direct = candidate_features
            
        return features_for_direct, use_joint_sampling, n_candidates

    def _log_initial_stats(self, desc_datanames):
        """Logs initial data statistics."""
        self.logger.info("="*40)
        self.logger.info("       INITIAL DATA STATISTICS")
        self.logger.info("="*40)
        
        # Create a temporary DF for stats
        df_stats_init = pd.DataFrame({"dataname": desc_datanames})
        df_stats_init["sys_name"] = df_stats_init["dataname"].apply(get_sys_name)
        df_stats_init["pool_name"] = df_stats_init["sys_name"].apply(get_pool_name)
        
        stats_init = df_stats_init.groupby("pool_name").agg(
            num_systems=("sys_name", "nunique"),
            num_frames=("dataname", "count")
        )
        self.logger.info(f"\n{stats_init}")
        self.logger.info("="*40)
        return stats_init

    def _run_uq_analysis(self, vis):
        """Executes UQ calculation and visualization."""
        # 1. Load Data & Calculate UQ
        self.logger.info("Loading the test results")
        preds = []
        num_models = self.config.num_models
        for i in range(num_models):
            # Construct path using os.path.join for robustness
            res_path_prefix = os.path.join(self.project, str(i), self.testing_dir, self.testing_head)
            
            # DPTestResultParser expects result_dir and head.
            # To maintain compatibility with existing structure where res_path_prefix 
            # might include the file prefix, we pass "." as result_dir and the full prefix as head.
            parser = DPTestResultParser(result_dir=".", head=res_path_prefix)
            parsed_dict = parser.parse()
            
            pred_data = PredictionData(
                energy=parsed_dict["energy"],
                force=parsed_dict["force"],
                virial=parsed_dict["virial"],
                has_ground_truth=parsed_dict["has_ground_truth"],
                dataname_list=parsed_dict["dataname_list"],
                datanames_nframe=parsed_dict["datanames_nframe"]
            )
            preds.append(pred_data)
            
        has_ground_truth = preds[0].has_ground_truth
        
        self.logger.info("Dealing with force difference between 0 head prediction and existing label")
        self.logger.info("Dealing with atomic force and average 1, 2, 3")
        self.logger.info("Dealing with atomic force UQ by DPGEN formula, in QbC and RND-like")
        self.logger.info("Dealing with QbC force UQ")
        self.logger.info("Dealing with RND-like force UQ")
        
        calculator = UQCalculator()
        uq_results = calculator.compute_qbc_rnd(preds)
        
        self.logger.info("Aligning UQ-RND to UQ-QbC by RobustScaler (Median/IQR alignment)")
        uq_rnd_rescaled = calculator.align_scales(uq_results[COL_UQ_QBC], uq_results[COL_UQ_RND])
        
        # Auto-calculate thresholds if mode is auto
        if self.uq_trust_mode == "auto":
            self.logger.info(f"UQ Trust Mode is AUTO.")
            
            # Calculate for QbC
            _qbc_ratio = self.uq_qbc_params["ratio"]
            _qbc_width = self.uq_qbc_params["width"]
            qbc_bounds = self.uq_auto_bounds.get("qbc", {})
            
            self.logger.info(f"Calculating QbC thresholds with ratio={_qbc_ratio}, width={_qbc_width}")
            if qbc_bounds:
                self.logger.info(f"  - Using Bounds: {qbc_bounds}")
            
            calc_lo_qbc = calculator.calculate_trust_lo(uq_results[COL_UQ_QBC], ratio=_qbc_ratio)
            
            # Apply Bounds
            calc_lo_qbc = self._clamp_trust_lo(calc_lo_qbc, qbc_bounds, "QbC Trust Lo")
            
            if calc_lo_qbc is not None:
                self.uq_qbc_trust_lo = calc_lo_qbc
                self.logger.info(f"Auto-calculated QbC Trust Lo: {self.uq_qbc_trust_lo:.4f}")
            else:
                self.logger.warning(f"Auto-calculation for QbC failed. Fallback to manual Trust Lo: {self.uq_qbc_trust_lo}")
                if self.uq_qbc_trust_lo is None:
                    raise ValueError("Auto-UQ failed and no fallback 'lo' provided for QbC!")
                
            self.uq_qbc_trust_hi = self.uq_qbc_trust_lo + _qbc_width
            self.logger.info(f"Final QbC Trust Range: [{self.uq_qbc_trust_lo:.4f}, {self.uq_qbc_trust_hi:.4f}]")
            
            # Calculate for RND Rescaled
            _rnd_ratio = self.uq_rnd_params["ratio"]
            _rnd_width = self.uq_rnd_params["width"]
            rnd_bounds = self.uq_auto_bounds.get("rnd", {})
            
            self.logger.info(f"Calculating RND thresholds with ratio={_rnd_ratio}, width={_rnd_width}")
            if rnd_bounds:
                self.logger.info(f"  - Using Bounds: {rnd_bounds}")
                
            calc_lo_rnd = calculator.calculate_trust_lo(uq_rnd_rescaled, ratio=_rnd_ratio)
            
            # Apply Bounds
            calc_lo_rnd = self._clamp_trust_lo(calc_lo_rnd, rnd_bounds, "RND Trust Lo")
            
            if calc_lo_rnd is not None:
                self.uq_rnd_trust_lo = calc_lo_rnd
                self.logger.info(f"Auto-calculated RND-rescaled Trust Lo: {self.uq_rnd_trust_lo:.4f}")
            else:
                self.logger.warning(f"Auto-calculation for RND-rescaled failed. Fallback to manual Trust Lo: {self.uq_rnd_trust_lo}")
                if self.uq_rnd_trust_lo is None:
                    raise ValueError("Auto-UQ failed and no fallback 'lo' provided for RND!")
                
            self.uq_rnd_trust_hi = self.uq_rnd_trust_lo + _rnd_width
            self.logger.info(f"Final RND-rescaled Trust Range: [{self.uq_rnd_trust_lo:.4f}, {self.uq_rnd_trust_hi:.4f}]")
        
        # Stats for UQ variables
        self.logger.info("Calculating statistics for UQ variables (QbC, RND, RND_rescaled)")
        df_uq_stats = pd.DataFrame({
            "UQ_QbC": uq_results[COL_UQ_QBC],
            "UQ_RND": uq_results[COL_UQ_RND],
            "UQ_RND_rescaled": uq_rnd_rescaled
        })
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.float_format', '{:.4f}'.format)
        stats_desc = df_uq_stats.describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99])
        self.logger.info(f"UQ Statistics:\n{stats_desc}")

        # 2. Visualization (UQ Distributions)
        self.logger.info("Plotting and saving the figures of UQ-force")
        vis.plot_uq_distribution(uq_results[COL_UQ_QBC], uq_results[COL_UQ_RND])
        
        self.logger.info("Plotting and saving the figures of UQ-force rescaled")
        vis.plot_uq_distribution(uq_results[COL_UQ_QBC], uq_results[COL_UQ_RND], uq_rnd_rescaled)
        
        self.logger.info("Plotting and saving the figures of UQ-QbC-force with UQ trust range")
        vis.plot_uq_with_trust_range(uq_results[COL_UQ_QBC], "UQ-QbC-force", "UQ-QbC-force.png", 
                                     self.uq_qbc_trust_lo, self.uq_qbc_trust_hi)
        
        self.logger.info("Plotting and saving the figures of UQ-RND-force-rescaled with UQ trust range")
        vis.plot_uq_with_trust_range(uq_rnd_rescaled, "UQ-RND-force-rescaled", "UQ-RND-force-rescaled.png",
                                     self.uq_qbc_trust_lo, self.uq_qbc_trust_hi)
        
        if has_ground_truth:
            self.logger.info("Plotting and saving the figures of UQ-force vs force diff")
            vis.plot_uq_vs_error(uq_results[COL_UQ_QBC], uq_results[COL_UQ_RND], uq_results["diff_maxf_0_frame"])
            
            self.logger.info("Plotting and saving the figures of UQ-force-rescaled vs force diff")
            vis.plot_uq_vs_error(uq_results[COL_UQ_QBC], uq_rnd_rescaled, uq_results["diff_maxf_0_frame"], rescaled=True)
            
        self.logger.info("Calculating the difference between UQ-qbc and UQ-rnd-rescaled")
        self.logger.info("Plotting and saving the figures of UQ-diff")
        vis.plot_uq_diff_parity(uq_results[COL_UQ_QBC], uq_rnd_rescaled, 
                                diff_maxf=uq_results["diff_maxf_0_frame"] if has_ground_truth else None)
        
        if has_ground_truth:
            self.logger.info("Plotting and saving the figures of UQ-qbc-force and UQ-rnd-force-rescaled vs force diff")
            # Creating temp df for visualization
            df_temp = pd.DataFrame({
                COL_UQ_QBC: uq_results[COL_UQ_QBC],
                "uq_rnd_for_rescaled": uq_rnd_rescaled,
                "diff_maxf_0_frame": uq_results["diff_maxf_0_frame"]
            })
            vis.plot_uq_fdiff_scatter(df_temp, self.uq_scheme, 
                                  self.uq_qbc_trust_lo, self.uq_qbc_trust_hi, 
                                  self.uq_rnd_trust_lo, self.uq_rnd_trust_hi)

        return uq_results, uq_rnd_rescaled, preds, has_ground_truth

    def _run_uq_filtering(self, uq_results, uq_rnd_rescaled, preds, has_ground_truth, vis):
        """Executes filtering based on UQ results."""
        # 3. Data Preparation & Filtering
        self.logger.info("Dealing with Selection in Target dpdata")
        datanames_ind_list = [f"{i[0]}-{i[1]}" for i in preds[0].dataname_list]
        
        # Extract unique system names in order for ordered loading
        # dataname_list contains [name, index, natom]
        unique_system_names = []
        seen = set()
        for item in preds[0].dataname_list:
            name = item[0]
            if name not in seen:
                seen.add(name)
                unique_system_names.append(name)
        
        self.logger.info(f"Identified {len(unique_system_names)} unique systems from inference results.")
        
        # Build expected frames dict for consistency check
        expected_frames_dict = {}
        for sys_name in unique_system_names:
            n_frames = preds[0].datanames_nframe.get(sys_name, 0)
            if n_frames > 0:
                expected_frames_dict[sys_name] = n_frames
        
        data_dict_uq = {
            "dataname": datanames_ind_list,
            COL_UQ_QBC: uq_results[COL_UQ_QBC],
            "uq_rnd_for_rescaled": uq_rnd_rescaled,
            COL_UQ_RND: uq_results[COL_UQ_RND],
        }
        if has_ground_truth:
            data_dict_uq["diff_maxf_0_frame"] = uq_results["diff_maxf_0_frame"]
            
        df_uq = pd.DataFrame(data_dict_uq)
        
        # Load descriptors (Candidates) - Using Ordered Loading with Consistency Check
        desc_datanames, desc_stru = self._load_descriptors(self.desc_dir, "candidate descriptors", 
                                                         target_names=unique_system_names,
                                                         expected_frames=expected_frames_dict)
        
        if len(desc_stru) == 0:
            raise ValueError("No candidate descriptors loaded!")
        
        # Log Initial Stats
        stats_init = self._log_initial_stats(desc_datanames)

        self.logger.info(f"Collecting data to dataframe and do UQ selection")
        df_desc = pd.DataFrame(desc_stru, columns=[f"{COL_DESC_PREFIX}{i}" for i in range(desc_stru.shape[1])])
        df_desc["dataname"] = desc_datanames
        
        # Verify consistency for candidates
        if len(df_desc) != len(df_uq):
             self.logger.warning(f"Mismatch: UQ data has {len(df_uq)} frames, but descriptors have {len(df_desc)} frames.")
             
        # Optimized: Use concat if aligned (much faster), else merge
        if len(df_uq) == len(df_desc) and np.array_equal(df_uq["dataname"].values, df_desc["dataname"].values):
            self.logger.info("Dataframes are aligned. Using optimized concat.")
            # Drop duplicate dataname col from df_desc
            df_desc_vals = df_desc.drop(columns=["dataname"])
            df_uq_desc = pd.concat([df_uq, df_desc_vals], axis=1)
        else:
            self.logger.info("Dataframes not aligned. Falling back to merge.")
            df_uq_desc = pd.merge(df_uq, df_desc, on="dataname")
        
        self.logger.info(f"Save df_uq_desc dataframe to {self.df_savedir}/df_uq_desc.csv")
        df_uq_desc.to_csv(f"{self.df_savedir}/df_uq_desc.csv", index=True)
        
        # Apply Filtering
        uq_filter = UQFilter(self.uq_scheme, self.uq_qbc_trust_lo, self.uq_qbc_trust_hi, 
                             self.uq_rnd_trust_lo, self.uq_rnd_trust_hi)
        
        df_candidate, df_accurate, df_failed = uq_filter.filter(df_uq_desc)
        
        # Logging Stats
        self.logger.info(f"UQ scheme: {self.uq_scheme} between QbC and RND-like")
        self.logger.info(f"UQ selection information : {self.uq_scheme}")
        self.logger.info(f"Total number of structures: {len(df_uq_desc)}")
        self.logger.info(f"Accurate structures: {len(df_accurate)}, Precentage: {len(df_accurate) / len(df_uq_desc) * 100:.2f}%")
        self.logger.info(f"Candidate structures: {len(df_candidate)}, Precentage: {len(df_candidate) / len(df_uq_desc) * 100:.2f}%")
        self.logger.info(f"Failed structures: {len(df_failed)}, Precentage: {len(df_failed) / len(df_uq_desc) * 100:.2f}%")
        
        # Add Identity Labels
        df_uq = uq_filter.get_identity_labels(df_uq, df_candidate, df_accurate)
        
        self.logger.info(f"Save df_uq dataframe to {self.df_savedir}/df_uq.csv after UQ selection and identication")
        df_uq.to_csv(f"{self.df_savedir}/df_uq.csv", index=True)
        
        self.logger.info(f"Save df_uq_desc_candidate dataframe to {self.df_savedir}/df_uq_desc_sampled-UQ.csv")
        df_candidate.to_csv(f"{self.df_savedir}/df_uq_desc_sampled-UQ.csv", index=True)
        
        # Visualize Selection
        self.logger.info("Plotting and saving the figure of UQ-identity in QbC-RND 2D space")
        vis.plot_uq_identity_scatter(df_uq, self.uq_scheme, 
                              self.uq_qbc_trust_lo, self.uq_qbc_trust_hi, 
                              self.uq_rnd_trust_lo, self.uq_rnd_trust_hi)
                              
        if has_ground_truth:
            self.logger.info("Plotting and saving the figure of UQ-Candidate vs Max Force Diff")
            vis.plot_candidate_vs_error(df_uq, df_candidate)

        return df_candidate, df_uq, df_desc, unique_system_names, df_uq_desc, stats_init, uq_filter, df_accurate

    def _run_nofilter_setup(self):
        """Sets up data for 'no_filter' mode (skips UQ)."""
        self.logger.info("UQ Trust Mode is 'no_filter'. Skipping UQ calculation and filtering.")
        
        # Load descriptors
        desc_datanames, desc_stru = self._load_descriptors(self.desc_dir, "candidate descriptors")
        
        if len(desc_stru) == 0:
            raise ValueError("No candidate descriptors loaded!")
            
        # Build df_desc
        df_desc = pd.DataFrame(desc_stru, columns=[f"{COL_DESC_PREFIX}{i}" for i in range(desc_stru.shape[1])])
        df_desc["dataname"] = desc_datanames
        
        # Initial Stats
        stats_init = self._log_initial_stats(desc_datanames)
        
        # Setup df_uq (dummy) and df_candidate
        df_uq = df_desc[["dataname"]].copy()
        df_uq["uq_identity"] = "candidate"
        
        df_uq_desc = df_desc.copy()
        df_candidate = df_desc.copy() # All are candidates
        df_candidate["uq_identity"] = "candidate"
        
        # Derive unique_system_names for export
        unique_system_names = sorted(list(set(get_sys_name(d) for d in desc_datanames)))
        self.logger.info(f"Identified {len(unique_system_names)} unique systems from descriptors.")
        
        has_ground_truth = False # No preds loaded
        
        return df_candidate, df_uq, df_desc, unique_system_names, df_uq_desc, stats_init, has_ground_truth

    def _load_atomic_features_for_candidates(self, df_candidate):
        """
        Loads atomic features only for the frames present in df_candidate.
        
        Args:
            df_candidate (pd.DataFrame): DataFrame containing 'dataname' of candidates.
            
        Returns:
            tuple: (X_atom_list, n_atoms_list)
                   X_atom_list: List of (N_atoms, D) arrays.
                   n_atoms_list: List of ints.
        """
        self.logger.info("Loading atomic features for 2-DIRECT sampling...")
        
        dataname_to_atom_features = {}
        dataname_to_natoms = {}
        
        # Identify which systems are needed
        candidate_datanames = set(df_candidate["dataname"])
        
        # Group needed frames by system to minimize file I/O
        sys_to_frames = {}
        for dn in candidate_datanames:
            sys_name, idx = dn.rsplit("-", 1)
            idx = int(idx)
            if sys_name not in sys_to_frames:
                sys_to_frames[sys_name] = set()
            sys_to_frames[sys_name].add(idx)
            
        # Iterate over systems and load data
        for sys_name in sys_to_frames:
            # Construct path (reuse logic from _load_descriptors roughly)
            path_flat = os.path.join(self.desc_dir, f"{sys_name}.npy")
            path_flat_base = os.path.join(self.desc_dir, f"{os.path.basename(sys_name)}.npy")
            
            if os.path.exists(path_flat):
                f = path_flat
            elif os.path.exists(path_flat_base):
                f = path_flat_base
            else:
                self.logger.warning(f"Could not find descriptor file for {sys_name} during atomic load.")
                continue
                
            try:
                # Use mmap_mode to avoid loading entire file into memory
                one_desc = np.load(f, mmap_mode='r') 
                
                # Extract specific frames
                needed_indices = sys_to_frames[sys_name]
                
                for idx in needed_indices:
                    if idx < len(one_desc):
                        # Force copy to memory
                        feats = np.array(one_desc[idx]) 
                        dn = f"{sys_name}-{idx}"
                        dataname_to_atom_features[dn] = feats
                        dataname_to_natoms[dn] = feats.shape[0]
                    else:
                        self.logger.warning(f"Frame index {idx} out of bounds for {sys_name}")
                        
            except Exception as e:
                self.logger.error(f"Failed to load atomic features for {sys_name}: {e}")
                
        # Build ordered lists matching df_candidate
        X_atom_list = []
        n_atoms_list = []
        
        for dn in df_candidate["dataname"]:
            if dn in dataname_to_atom_features:
                X_atom_list.append(dataname_to_atom_features[dn])
                n_atoms_list.append(dataname_to_natoms[dn])
            else:
                self.logger.error(f"Missing atomic features for candidate {dn}")
                raise ValueError(f"Missing atomic features for {dn}")
                
        return X_atom_list, n_atoms_list

    def run(self):
        """
        Executes the main collection workflow.
        """
        if self.backend == "slurm":
            self._submit_to_slurm()
            return

        self.logger.info(f"Initializing selection in {self.project} ---")
        
        vis = UQVisualizer(self.view_savedir, dpi=self.config.fig_dpi)
        uq_filter = None

        if self.uq_trust_mode == "no_filter":
             df_candidate, df_uq, df_desc, unique_system_names, df_uq_desc, stats_init, has_ground_truth = self._run_nofilter_setup()
        else:
             uq_results, uq_rnd_rescaled, preds, has_ground_truth = self._run_uq_analysis(vis)
             df_candidate, df_uq, df_desc, unique_system_names, df_uq_desc, stats_init, uq_filter, df_accurate = self._run_uq_filtering(uq_results, uq_rnd_rescaled, preds, has_ground_truth, vis)
        
        # 4. DIRECT Sampling
        if len(df_candidate) == 0:
            self.logger.warning("No structures selected! Skipping DIRECT selection.")
            return

        self.logger.info(f"Doing DIRECT Selection on UQ-selected data")
        
        # --- Joint with Training Data Logic ---
        features_for_direct, use_joint_sampling, n_candidates = self._prepare_features_for_direct(df_candidate, df_desc)
        
        # --- Select Sampler Strategy ---
        sampler_type = self.config.sampler_type
        
        # Initialize num_selection for logging (will be calculated if dynamic)
        num_selection = 0
        
        if sampler_type == "2-direct":
            self.logger.info("Using 2-DIRECT Sampling Strategy.")
            
            # Dynamic Mode Warning for 2-DIRECT
            if self.config.step1_n_clusters is None or self.config.step2_n_clusters is None:
                missing_params = []
                if self.config.step1_n_clusters is None: missing_params.append("step1_n_clusters")
                if self.config.step2_n_clusters is None: missing_params.append("step2_n_clusters")
                
                self.logger.warning(
                    f"2-DIRECT Dynamic Mode (ADVANCED): The following cluster counts are unset: {missing_params}. "
                    f"Using thresholds (step1_thr={self.config.step1_threshold}, step2_thr={self.config.step2_threshold}) to determine counts dynamically. "
                    "For predictable sampling budget, it is STRONGLY RECOMMENDED to set these parameters explicitly."
                )
                                    
            if use_joint_sampling:
                self.logger.warning("2-DIRECT currently DOES NOT support joint sampling with training data in the same way as DIRECT. "
                                    "Training data will be ignored for 2-DIRECT step 2 (atomic sampling). "
                                    "Only candidates will be processed.")
                # Decision: Fallback to candidate-only features for 2-DIRECT to ensure stability.
                features_for_direct = features_for_direct[:n_candidates]
                use_joint_sampling = False 
                self.logger.info("Disabled joint sampling for 2-DIRECT.")

            # Load Atomic Features
            X_atom_list, n_atoms_list = self._load_atomic_features_for_candidates(df_candidate)
            
            sampler = TwoStepDIRECTSampler(
                step1_clustering=BirchClustering(n=self.config.step1_n_clusters, threshold_init=self.config.step1_threshold),
                step2_clustering=BirchClustering(n=self.config.step2_n_clusters, threshold_init=self.config.step2_threshold),
                step2_selection=SelectKFromClusters(k=self.config.step2_k, selection_criteria=self.config.step2_selection, n_sites=[1])
            )
            
            # Run 2-DIRECT
            # features_for_direct corresponds to X_stru (aligned with df_candidate)
            sampling_result = sampler.fit_transform(features_for_direct, X_atom_list, n_atoms_list)
            
            DIRECT_selected_indices = sampling_result["selected_indices"]
            all_pca_features = sampling_result["PCAfeatures"]
            
            # 2-DIRECT uses the PCA from Step 1 for visualization
            explained_variance = sampler.step1_sampler.pca.pca.explained_variance_
            
        else:
            # Standard DIRECT
            self.logger.info("Using Standard DIRECT Sampling Strategy.")
            
            # Level 1: Explicit Control (Recommended)
            if self.config.direct_n_clusters is not None:
                n_clusters = self.config.direct_n_clusters
                num_selection = n_clusters * self.direct_k
                self.logger.info(f"Explicit cluster count set: {n_clusters}")
                
            # Level 3: Dynamic Clustering (Advanced)
            else:
                self.logger.warning(
                    "Dynamic Clustering Mode (ADVANCED): `direct_n_clusters` is not set. "
                    f"Cluster count will be determined dynamically by `direct_thr_init`={self.direct_thr_init}. "
                    "For predictable sampling budget, it is STRONGLY RECOMMENDED to set `direct_n_clusters` explicitly."
                )
                n_clusters = None # BirchClustering supports n=None
            
            sampler = DIRECTSampler(
                structure_encoder=None,
                clustering=BirchClustering(n=n_clusters, threshold_init=self.direct_thr_init),
                select_k_from_clusters=SelectKFromClusters(k=self.direct_k),
            )
            
            sampling_result = sampler.fit_transform(features_for_direct)
            
            selected_indices_raw = sampling_result["selected_indices"]
            all_pca_features = sampling_result["PCAfeatures"]
            explained_variance = sampler.pca.pca.explained_variance_
            
            # Handle Joint Sampling Indices
            if use_joint_sampling:
                DIRECT_selected_indices = [idx for idx in selected_indices_raw if idx < n_candidates]
            else:
                DIRECT_selected_indices = selected_indices_raw

        # --- Post-Sampling Common Logic ---
        selected_PC_dim = len([e for e in explained_variance if e > 1])
        
        # Calculate num_selection for logging if it was 0 (2-direct)
        if num_selection == 0:
            num_selection = len(DIRECT_selected_indices)
        
        if use_joint_sampling:
             n_from_training = len(selected_indices_raw) - len(DIRECT_selected_indices)
             self.logger.info(f"DIRECT Selection Result (Joint Mode):")
             self.logger.info(f"  - Target Total Representatives (num_selection): {num_selection}")
             self.logger.info(f"  - Actually Found Representatives: {len(selected_indices_raw)}")
             self.logger.info(f"  - Selected from Training Set (Ignored): {n_from_training}")
             self.logger.info(f"  - Selected from Candidate Set (New Samples): {len(DIRECT_selected_indices)}")
             
             all_features_viz = all_pca_features / explained_variance[:selected_PC_dim]
             viz_selected_indices = selected_indices_raw
        else:
             self.logger.info(f"DIRECT Selection Result:")
             self.logger.info(f"  - Strategy: {sampler_type}")
             self.logger.info(f"  - Actually Selected Samples: {len(DIRECT_selected_indices)}")
             
             all_features_viz = all_pca_features / explained_variance[:selected_PC_dim]
             viz_selected_indices = DIRECT_selected_indices
        
        # Calculate PCA for ALL data (df_uq_desc) for visualization background
        try:
            self.logger.info("Projecting all data onto PCA space for visualization...")
            all_desc_features = df_uq_desc[[col for col in df_desc.columns if col.startswith(COL_DESC_PREFIX)]].values
            
            # Transform using the appropriate PCA model
            if sampler_type == "2-direct":
                pca_model = sampler.step1_sampler.pca
            else:
                pca_model = sampler.pca
                
            full_pca_features = pca_model.transform(all_desc_features)
            
            if full_pca_features.shape[1] > selected_PC_dim:
                full_pca_features = full_pca_features[:, :selected_PC_dim]
                
            full_features_viz = full_pca_features / explained_variance[:selected_PC_dim]
            
        except Exception as e:
            self.logger.warning(f"Failed to project all data for visualization: {e}. Skipping background plot.")
            full_features_viz = None

        df_final = df_candidate.iloc[DIRECT_selected_indices]
        
        self.logger.info(f"Saving df_uq_desc_selected_final dataframe to {self.df_savedir}/df_uq_desc_sampled-final.csv")
        df_final.to_csv(f"{self.df_savedir}/df_uq_desc_sampled-final.csv", index=True)

        # --- Enhanced Logging: Sampling Stats & Consistency Check ---
        self.logger.info("="*40)
        self.logger.info("       SAMPLING STATISTICS")
        self.logger.info("="*40)
        
        # 1. Sampled Stats
        df_final_stats = df_final.copy()
        df_final_stats["sys_name"] = df_final_stats["dataname"].apply(get_sys_name)
        df_final_stats["pool_name"] = df_final_stats["sys_name"].apply(get_pool_name)
        
        stats_sampled = df_final_stats.groupby("pool_name").agg(
            sampled_frames=("dataname", "count")
        )
        
        # 2. Remaining Stats
        # Merge sampled counts into init stats
        # Note: stats_init includes "ALL" row, but stats_sampled does not yet.
        
        stats_merged = stats_init.join(stats_sampled).fillna(0)
        # "ALL" row in stats_sampled will be NaN after join, need to compute sum
        
        stats_merged["sampled_frames"] = stats_merged["sampled_frames"].astype(int)
        
        # Re-calculate ALL row for sampled_frames
        total_sampled = stats_merged.loc[stats_merged.index != "ALL", "sampled_frames"].sum()
        stats_merged.loc["ALL", "sampled_frames"] = total_sampled
        
        # Re-populate ALL row for num_systems and num_frames (in case they were NaN'd or zeroed by join logic if index didn't match perfectly, though join on index should be fine)
        # Actually stats_init already had ALL. The join keeps it.
        # But let's ensure it's correct.
        total_systems = stats_merged.loc[stats_merged.index != "ALL", "num_systems"].sum()
        total_frames = stats_merged.loc[stats_merged.index != "ALL", "num_frames"].sum()
        stats_merged.loc["ALL", "num_systems"] = total_systems
        stats_merged.loc["ALL", "num_frames"] = total_frames

        stats_merged["remaining_frames"] = stats_merged["num_frames"] - stats_merged["sampled_frames"]
        
        # Calculate remaining systems (Systems that have not been FULLY sampled? Or just systems present in remaining?)
        # "Remaining System Number": If we interpret as "Systems that have at least 1 frame remaining".
        
        # Get set of sampled datanames
        sampled_datanames_set = set(df_final["dataname"])
        
        # Filter df_stats_init to find remaining rows
        # We need to recreate df_stats_init here or reuse it from stats_init?
        # stats_init is aggregated. We need the raw df to filter.
        # It's better to reconstruct it locally as before since we didn't return the raw df.
        
        # Reconstruct df_stats_init for remaining calc
        # Or better: return df_stats_init from _log_initial_stats? No, just recreate it, it's cheap.
        # Actually I can't recreate it easily without desc_datanames.
        # df_desc has dataname.
        desc_datanames_for_stats = df_desc["dataname"]
        df_stats_init_temp = pd.DataFrame({"dataname": desc_datanames_for_stats})
        df_stats_init_temp["sys_name"] = df_stats_init_temp["dataname"].apply(get_sys_name)
        df_stats_init_temp["pool_name"] = df_stats_init_temp["sys_name"].apply(get_pool_name)

        df_remaining = df_stats_init_temp[~df_stats_init_temp["dataname"].isin(sampled_datanames_set)]
        
        stats_remaining_sys = df_remaining.groupby("pool_name").agg(
            remaining_systems=("sys_name", "nunique")
        )
        
        # Calculate ALL for remaining systems
        total_remaining_sys = stats_remaining_sys["remaining_systems"].sum()
        
        stats_merged = stats_merged.join(stats_remaining_sys).fillna(0)
        stats_merged["remaining_systems"] = stats_merged["remaining_systems"].astype(int)
        
        stats_merged.loc["ALL", "remaining_systems"] = total_remaining_sys
        
        # Ensure integer display by explicitly converting to int
        cols_to_print = ['num_systems', 'num_frames', 'sampled_frames', 'remaining_systems', 'remaining_frames']
        # Convert to int to remove decimals, then to string for logging
        # Using to_string() avoids the global float_format setting if dtypes are int
        stats_display = stats_merged[cols_to_print].astype(int)
        self.logger.info(f"\n{stats_display}")
        
        # 3. Consistency Check
        # Use .loc["ALL"] to get totals
        total_init_frames = stats_merged.loc["ALL", "num_frames"]
        total_sampled_frames = stats_merged.loc["ALL", "sampled_frames"]
        total_remaining_frames = stats_merged.loc["ALL", "remaining_frames"]
        
        if abs(total_init_frames - (total_sampled_frames + total_remaining_frames)) > 1e-5: # Use tolerance for float check, though these are ints
             self.logger.error(f"Consistency Check FAILED: {total_init_frames} != {total_sampled_frames} + {total_remaining_frames}")
             raise ValueError("Frame consistency check failed after sampling!")
        else:
             self.logger.info(f"Consistency Check PASSED: Total({int(total_init_frames)}) == Sampled({int(total_sampled_frames)}) + Remaining({int(total_remaining_frames)})")
             
        self.logger.info("="*40)
        # ------------------------------------------------------------
        
        # 5. Visualization (Sampling)
        self.logger.info(f"Visualization of DIRECT results compared with Random")
        
        # Random Simulation
        # In Joint mode, we compare DIRECT (on Joint) vs Random (on Joint)
        np.random.seed(42)
        if len(all_features_viz) >= num_selection:
             manual_selection_index = np.random.choice(len(all_features_viz), num_selection, replace=False)
        else:
             manual_selection_index = np.arange(len(all_features_viz))
        
        # Coverage Scores
        def calculate_feature_coverage_score(all_features, selected_indices, n_bins=100):
            if len(selected_indices) == 0: return 0
            selected_features = all_features[selected_indices]
            # Use fixed bins based on min/max of ALL features
            bins = np.linspace(min(all_features), max(all_features), n_bins)
            n_all = np.count_nonzero(np.histogram(all_features, bins=bins)[0])
            n_select = np.count_nonzero(np.histogram(selected_features, bins=bins)[0])
            return n_select / n_all if n_all > 0 else 0

        def calculate_all_FCS(all_features, selected_indices):
            return [calculate_feature_coverage_score(all_features[:, i], selected_indices) 
                    for i in range(all_features.shape[1])]

        # Use viz_selected_indices (Joint Selection) for Coverage Calculation
        scores_DIRECT = calculate_all_FCS(all_features_viz, viz_selected_indices)
        scores_MS = calculate_all_FCS(all_features_viz, manual_selection_index)
        
        self.logger.info(f"Visualization of final selection results in PCA space")
        
        # Add uq_identity to df_uq_desc for visualizer usage
        if uq_filter:
            df_uq_desc = uq_filter.get_identity_labels(df_uq_desc, df_candidate, df_accurate)
        else:
            # For no_filter, we manually added uq_identity="candidate" earlier
            pass
        
        # Ensure df_candidate has identity label for visualization
        df_candidate = df_candidate.copy()
        df_candidate["uq_identity"] = "candidate"
        
        # Pass n_candidates to visualizer to distinguish Candidate/Training in plots
        # Note: We pass df_candidate instead of df_uq_desc because in Joint mode, 
        # PCA is performed on [Candidates; Training], not the full UQ set.
        # Passing df_candidate ensures length alignment with all_features_viz[:n_candidates].
        PCs_df = vis.plot_pca_analysis(explained_variance, selected_PC_dim, all_features_viz, 
                                      viz_selected_indices, manual_selection_index,
                                      scores_DIRECT, scores_MS, 
                                      df_candidate, df_final.index,
                                      n_candidates=n_candidates if use_joint_sampling else None,
                                      full_features=full_features_viz)

        
        # Save Final PCA Data (Candidates Only)
        # Note: PCs_df returned by plot_pca_analysis might be Joint size if we are not careful.
        # Ideally we want PCs for the df_uq dataframe.
        # But all_features_viz is Joint.
        # Let's extract candidate PCs from all_features_viz for saving to final_df.csv (which corresponds to df_uq)
        
        if use_joint_sampling:
            # First n_candidates rows correspond to df_uq (candidates)
            # Re-scale back or just use viz features? 
            # Usually we save the features used for visualization.
            candidate_pcs = all_features_viz[:n_candidates, :2] # Save first 2 PCs
        else:
            candidate_pcs = all_features_viz[:, :2]
            
        df_pcs = pd.DataFrame(candidate_pcs, columns=['PC1', 'PC2'])
        
        # Add PC info to df_candidate first (since they align)
        # We need to make sure indices align.
        # df_candidate is a slice of df_uq_desc, possibly with gaps in index?
        # But `candidate_features` was extracted from `df_candidate`.
        # So `candidate_pcs` corresponds row-by-row to `df_candidate`.
        
        df_candidate = df_candidate.reset_index(drop=False) # Keep original index if needed, but here we just want to merge PCs
        # Actually, let's just assign columns directly if lengths match
        if len(df_pcs) == len(df_candidate):
            df_candidate["PC1"] = df_pcs["PC1"]
            df_candidate["PC2"] = df_pcs["PC2"]
        else:
            self.logger.error(f"Logic Error: df_candidate len {len(df_candidate)} != PC len {len(df_pcs)}")
            
        # Now merge PC info back to df_uq
        # df_uq has all frames. df_candidate has a subset.
        # We want final_df.csv to have all frames, with PC1/PC2 filled for candidates, NaN for others.
        
        # Merge key: 'dataname' is safest.
        # df_uq has 'dataname'. df_candidate has 'dataname'.
        
        df_pcs_subset = df_candidate[["dataname", "PC1", "PC2"]]
        df_final_all = pd.merge(df_uq, df_pcs_subset, on="dataname", how="left")
        
        self.logger.info(f"Save final_df.csv with PC coordinates to {self.df_savedir}/final_df.csv")
        df_final_all.to_csv(f"{self.df_savedir}/final_df.csv", index=True)

        
        # 6. Export dpdata
        self.logger.info(f"Sampling dpdata based on selected indices")
        
        # Optimized: Pre-calculate sampled indices map for O(1) lookup
        # df_final['dataname'] format: "sys_name-index"
        sampled_indices_map = {}
        for dataname in df_final['dataname']:
            # Use rsplit to handle potential dashes in sys_name (though dataname format is fixed)
            sys_name, idx_str = dataname.rsplit('-', 1)
            try:
                idx = int(idx_str)
                if sys_name not in sampled_indices_map:
                    sampled_indices_map[sys_name] = set()
                sampled_indices_map[sys_name].add(idx)
            except ValueError:
                self.logger.warning(f"Failed to parse dataname: {dataname}")

        self.logger.info(f"Loading the target testing data from {self.testdata_dir}")
        
        # Use ordered unique_system_names to load data
        # This ensures alignment with df_final and prevents missing/extra systems
        # Use fmt="auto" to automatically detect if it's mixed or npy
        test_data = load_systems(self.testdata_dir, fmt="auto", target_systems=unique_system_names)
            
        self.logger.info(f"Dumping sampled and other dpdata to {self.dpdata_savedir}")
        
        # Clean up existing directories
        if os.path.exists(f"{self.dpdata_savedir}/sampled_dpdata"):
            shutil.rmtree(f"{self.dpdata_savedir}/sampled_dpdata")
        if os.path.exists(f"{self.dpdata_savedir}/other_dpdata"):
            shutil.rmtree(f"{self.dpdata_savedir}/other_dpdata")
        
        os.makedirs(f"{self.dpdata_savedir}/sampled_dpdata")
        os.makedirs(f"{self.dpdata_savedir}/other_dpdata")

        count_sampled_sys = 0
        count_other_sys = 0

        for sys in test_data:
            # Match frames based on dataname (keyname-index)
            # Use the target_name we stored, or fallback to short_name if missing
            sys_name = getattr(sys, "target_name", sys.short_name)
            
            # Optimized: O(1) Lookup instead of O(N*M) loop
            sampled_set = sampled_indices_map.get(sys_name, set())
            
            # Since sys indices are 0..len(sys)-1
            # We filter sampled_set to ensure it's within range (safety)
            n_frames = len(sys)
            valid_sampled = {i for i in sampled_set if i < n_frames}
            
            # Convert to sorted list for dpdata
            sampled_indices = sorted(list(valid_sampled))
            
            # Other indices are the complement
            # Using set difference is faster than iterating
            all_indices = set(range(n_frames))
            other_indices = sorted(list(all_indices - valid_sampled))
            
            # Save Sampled
            if sampled_indices:
                try:
                    sys_sampled = sys.sub_system(sampled_indices)
                    # Construct path: sampled_dpdata/sys_name (sys_name might be Pool/System)
                    save_path = os.path.join(self.dpdata_savedir, "sampled_dpdata", sys_name)
                    # Ensure parent dir exists (for nested sys_name)
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    # to_deepmd_npy creates the target directory
                    sys_sampled.to_deepmd_npy(save_path)
                    count_sampled_sys += 1
                except Exception as e:
                    self.logger.error(f"Failed to save sampled frames for {sys_name}: {e}")

            # Save Other
            if other_indices:
                try:
                    sys_other = sys.sub_system(other_indices)
                    save_path = os.path.join(self.dpdata_savedir, "other_dpdata", sys_name)
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    sys_other.to_deepmd_npy(save_path)
                    count_other_sys += 1
                except Exception as e:
                    self.logger.error(f"Failed to save other frames for {sys_name}: {e}")
                    
        self.logger.info(f"Saved {count_sampled_sys} systems to sampled_dpdata")
        self.logger.info(f"Saved {count_other_sys} systems to other_dpdata")
        
        self.logger.info(WORKFLOW_FINISHED_TAG)
