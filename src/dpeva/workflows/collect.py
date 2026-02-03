import os
import shutil
import glob
import logging
import sys
import numpy as np
import pandas as pd
import dpdata

from dpeva.io.dataproc import DPTestResultParser
from dpeva.io.types import PredictionData
from dpeva.io.dataset import load_systems
from dpeva.sampling.direct import BirchClustering, DIRECTSampler, SelectKFromClusters
from dpeva.uncertain.calculator import UQCalculator
from dpeva.uncertain.filter import UQFilter
from dpeva.uncertain.visualization import UQVisualizer
from dpeva.submission.manager import JobManager
from dpeva.submission.templates import JobConfig

class CollectionWorkflow:
    """
    Orchestrates the Collection pipeline:
    Data Loading -> UQ Calculation -> Filtering -> DIRECT Sampling -> Visualization -> Export
    """

    def __init__(self, config, config_path=None):
        """
        Initialize the Collection Workflow.

        Args:
            config (dict): Configuration dictionary containing:
                - project (str): Project name/directory (default: "stage9-2").
                - uq_select_scheme (str): UQ filtering scheme (default: "tangent_lo").
                - testing_dir (str): Subdir for test results (default: "test-val-npy").
                - testing_head (str): Head name for test results (default: "results").
                - desc_dir (str): Path to descriptor directory (Required).
                - desc_filename (str): Descriptor filename (default: "desc.npy").
                - testdata_dir (str): Path to test data directory (Required).
                - training_data_dir (str): Path to training data (Optional, for joint sampling).
                - training_desc_dir (str): Path to training descriptors (Optional).
                - root_savedir (str): Root directory for outputs (default: "dpeva_uq_post").
                - uq_trust_mode (str): 'manual' or 'auto'.
                - uq_trust_ratio (float): Global UQ trust ratio (default: 0.33).
                - uq_trust_width (float): Global UQ trust width (default: 0.25).
                - uq_qbc_trust_lo/hi/ratio/width: QbC specific parameters.
                - uq_rnd_rescaled_trust_lo/hi/ratio/width: RND specific parameters.
                - uq_auto_bounds (dict): Bounds for auto-UQ ('qbc', 'rnd').
                - num_selection (int): Total number of structures to select (default: 100).
                - direct_k (int): Number of clusters for DIRECT sampling (default: 1).
                - direct_thr_init (float): Initial threshold for DIRECT clustering (default: 0.5).
                - backend (str): 'local' or 'slurm' (default: 'local').
                - slurm_config (dict): Dict containing Slurm parameters.
            config_path (str, optional): Path to the configuration file. 
                                         Used for optimized Slurm self-submission.
        """
        self.config = config
        self.config_path = config_path
        self._setup_logger()
        self._validate_config()
        
        self.project = config.get("project", "stage9-2")
        self.uq_scheme = config.get("uq_select_scheme", "tangent_lo")
        
        # Backend Configuration
        # Allow environment override for internal recursion prevention (Slurm Worker Mode)
        env_backend = os.environ.get("DPEVA_INTERNAL_BACKEND")
        if env_backend:
            self.logger.info(f"Overriding backend to '{env_backend}' via DPEVA_INTERNAL_BACKEND environment variable.")
            self.backend = env_backend
        else:
            self.backend = config.get("backend", "local")
            
        self.slurm_config = config.get("slurm_config", {})
        
        # Paths
        self.testing_dir = config.get("testing_dir", "test-val-npy")
        self.testing_head = config.get("testing_head", "results")
        self.desc_dir = config.get("desc_dir")
        self.desc_filename = config.get("desc_filename", "desc.npy")
        self.testdata_dir = config.get("testdata_dir")
        # testdata_fmt is no longer needed as we use auto-detection
        
        # Training Set Paths (used in joint DIRECT for diversity maximization)
        self.training_data_dir = config.get("training_data_dir")
        self.training_desc_dir = config.get("training_desc_dir")
        
        # Save Dirs
        self.root_savedir = config.get("root_savedir", "dpeva_uq_post")
        self.view_savedir = os.path.join(self.project, self.root_savedir, "view")
        self.dpdata_savedir = os.path.join(self.project, self.root_savedir, "dpdata")
        self.df_savedir = os.path.join(self.project, self.root_savedir, "dataframe")
        
        self._ensure_dirs()
        self._configure_file_logging()
        
        # UQ Parameters
        self.uq_trust_mode = config.get("uq_trust_mode")
        
        # 1. Resolve Global Defaults
        self.global_trust_ratio = config.get("uq_trust_ratio", 0.33)
        self.global_trust_width = config.get("uq_trust_width", 0.25)
        
        # 2. Resolve QbC Parameters
        self.uq_qbc_params = {
            "ratio": config.get("uq_qbc_trust_ratio", self.global_trust_ratio),
            "width": config.get("uq_qbc_trust_width", self.global_trust_width),
            "lo": config.get("uq_qbc_trust_lo"),
            "hi": config.get("uq_qbc_trust_hi")
        }
        
        # 3. Resolve RND Parameters
        self.uq_rnd_params = {
            "ratio": config.get("uq_rnd_rescaled_trust_ratio", self.global_trust_ratio),
            "width": config.get("uq_rnd_rescaled_trust_width", self.global_trust_width),
            "lo": config.get("uq_rnd_rescaled_trust_lo"),
            "hi": config.get("uq_rnd_rescaled_trust_hi")
        }
        
        # Validate Parameters based on Mode
        if self.uq_trust_mode == "manual":
            self._validate_manual_params(self.uq_qbc_params, "uq_qbc")
            self._validate_manual_params(self.uq_rnd_params, "uq_rnd")
        elif self.uq_trust_mode == "auto":
            pass
        else:
            if not self.uq_trust_mode:
                self.logger.info("uq_trust_mode not set. Defaulting to 'manual'.")
                self.uq_trust_mode = "manual"
                self._validate_manual_params(self.uq_qbc_params, "uq_qbc")
                self._validate_manual_params(self.uq_rnd_params, "uq_rnd")
            else:
                raise ValueError(f"Unknown uq_trust_mode: {self.uq_trust_mode}")

        # Map back to instance variables
        self.uq_qbc_trust_lo = self.uq_qbc_params.get("lo")
        self.uq_qbc_trust_hi = self.uq_qbc_params.get("hi")
        self.uq_rnd_trust_lo = self.uq_rnd_params.get("lo")
        self.uq_rnd_trust_hi = self.uq_rnd_params.get("hi")
        
        # UQ Auto Bounds
        self.uq_auto_bounds = config.get("uq_auto_bounds", {})
        
        # Sampling Parameters
        self.num_selection = config.get("num_selection", 100)
        self.direct_k = config.get("direct_k", 1)
        self.direct_thr_init = config.get("direct_thr_init", 0.5)

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
        log_file = os.path.join(self.project, self.root_savedir, "collection.log")
        
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
        if not os.path.exists(self.config.get("project", ".")):
            self.logger.error(f"Project directory {self.config.get('project')} not found!")
            raise ValueError(f"Project directory {self.config.get('project')} not found!")
            
        required_keys = ["desc_dir", "testdata_dir"]
        for key in required_keys:
            if not self.config.get(key):
                self.logger.error(f"Missing required configuration: {key}")
                raise ValueError(f"Missing required configuration: {key}")

    def _ensure_dirs(self):
        """Creates necessary output directories if they don't exist."""
        for d in [self.view_savedir, self.dpdata_savedir, self.df_savedir]:
            if not os.path.exists(d):
                os.makedirs(d)

    def _load_descriptors(self, desc_dir, desc_filename="desc.npy", label="descriptors", target_names=None, expected_frames=None):
        """
        Loads descriptors from a directory.
        Supports both nested structure (sys/desc.npy) and flat structure (sys.npy).
        Prioritizes flat structure (*.npy) over nested structure (*/desc.npy).
        
        Args:
            desc_dir (str): Path to the descriptor directory.
            desc_filename (str): Name of the descriptor file (default: "desc.npy") used in nested structure.
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
                # 2. Nested: desc_dir/sys_name/desc.npy
                # 3. Nested 3-level (Dataset/System): desc_dir/Dataset/System.npy or desc_dir/Dataset/System/desc.npy
                
                # Try direct path first (most common for multi-pool: desc_dir/Dataset/System.npy)
                path_flat = os.path.join(desc_dir, f"{sys_name}.npy")
                path_nested = os.path.join(desc_dir, sys_name, desc_filename)
                
                # Fallback: Check for flat basename match (e.g. desc_dir/System.npy even if sys_name is Dataset/System)
                path_flat_base = os.path.join(desc_dir, f"{os.path.basename(sys_name)}.npy")
                
                if os.path.exists(path_flat):
                    f = path_flat
                elif os.path.exists(path_nested):
                    f = path_nested
                elif os.path.exists(path_flat_base):
                    # In Single Data Pool mode, sys_name might be 'pool/sys' but desc is just 'sys.npy'
                    # This is a valid compatibility match, not a warning condition.
                    self.logger.info(f"Matched descriptor via basename (Single-Pool Compatible): {path_flat_base} for system {sys_name}")
                    f = path_flat_base
                else:
                    self.logger.error(f"Descriptor file not found for system: {sys_name}. Expected at {path_flat} or {path_nested}")
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
            nested_pattern = os.path.join(desc_dir, "*", desc_filename)
            
            # Determine which pattern to use
            # Priority: User explicit wildcard > Flat (*.npy) > Nested (*/desc.npy)
            
            if '*' in desc_dir:
                 desc_pattern = desc_dir
            elif len(glob.glob(flat_pattern)) > 0:
                 desc_pattern = flat_pattern
            elif len(glob.glob(nested_pattern)) > 0:
                 self.logger.warning(f"Using deprecated nested descriptor structure: {nested_pattern}. "
                                     "Please switch to flat *.npy structure, which is the default in `dp eval-desc`.")
                 desc_pattern = nested_pattern
            else:
                 # Default fallback if nothing found (will return empty later)
                 desc_pattern = flat_pattern
                     
            desc_iter_list = sorted(glob.glob(desc_pattern))
            
            if not desc_iter_list:
                 self.logger.warning(f"No {label} found in {desc_dir}")
                 return [], np.array([])
            
            for f in desc_iter_list:
                # Determine keyname based on structure
                # Flat: .../sysname.npy -> sysname
                # Nested: .../sysname/desc.npy -> sysname
                if f.endswith(desc_filename) and os.path.basename(f) == desc_filename:
                     # Nested structure
                     keyname = os.path.basename(os.path.dirname(f))
                else:
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
            # Use sys.argv[0] as the runner script path if it's a python script
            runner_script = os.path.abspath(sys.argv[0])
            
            self.logger.info(f"Using Self-Invocation Mode:")
            self.logger.info(f"  - Runner: {runner_script}")
            self.logger.info(f"  - Config: {config_abs_path}")
            
            # Construct command to re-run the same script
            # We add DPEVA_INTERNAL_BACKEND=local to prevent infinite recursion
            python_exe = sys.executable
            cmd = f"{python_exe} -u {runner_script} --config {config_abs_path}"
            
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
            _, train_desc_stru = self._load_descriptors(self.training_desc_dir, self.desc_filename, "training descriptors")
            
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
        candidate_features = df_candidate[[col for col in df_desc.columns if col.startswith("desc_stru_")]].values
        n_candidates = len(candidate_features)
        
        if use_joint_sampling:
            # Combine [Candidate; Training]
            combined_features = np.vstack([candidate_features, train_desc_stru])
            self.logger.info(f"Combined feature shape: {combined_features.shape}")
            features_for_direct = combined_features
        else:
            features_for_direct = candidate_features
            
        return features_for_direct, use_joint_sampling, n_candidates

    def run(self):
        """
        Executes the main collection workflow:
        1. Load prediction results and calculate UQ.
        2. Filter data based on UQ thresholds (manual or auto).
        3. Visualize UQ distributions and filtering results.
        4. Perform DIRECT sampling on candidate structures.
        5. Export sampled structures as dpdata.
        """
        if self.backend == "slurm":
            self._submit_to_slurm()
            return

        self.logger.info(f"Initializing selection in {self.project} ---")
        
        # 1. Load Data & Calculate UQ
        self.logger.info("Loading the test results")
        preds = []
        for i in range(4):
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
        uq_results = calculator.compute_qbc_rnd(preds[0], preds[1], preds[2], preds[3])
        
        self.logger.info("Aligning UQ-RND to UQ-QbC by RobustScaler (Median/IQR alignment)")
        uq_rnd_rescaled = calculator.align_scales(uq_results["uq_qbc_for"], uq_results["uq_rnd_for"])
        
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
            
            calc_lo_qbc = calculator.calculate_trust_lo(uq_results["uq_qbc_for"], ratio=_qbc_ratio)
            
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
            "UQ_QbC": uq_results["uq_qbc_for"],
            "UQ_RND": uq_results["uq_rnd_for"],
            "UQ_RND_rescaled": uq_rnd_rescaled
        })
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.float_format', '{:.4f}'.format)
        stats_desc = df_uq_stats.describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99])
        self.logger.info(f"UQ Statistics:\n{stats_desc}")

        # 2. Visualization (UQ Distributions)
        vis = UQVisualizer(self.view_savedir, dpi=self.config.get("fig_dpi", 150))
        
        self.logger.info("Plotting and saving the figures of UQ-force")
        vis.plot_uq_distribution(uq_results["uq_qbc_for"], uq_results["uq_rnd_for"])
        
        self.logger.info("Plotting and saving the figures of UQ-force rescaled")
        vis.plot_uq_distribution(uq_results["uq_qbc_for"], uq_results["uq_rnd_for"], uq_rnd_rescaled)
        
        self.logger.info("Plotting and saving the figures of UQ-QbC-force with UQ trust range")
        vis.plot_uq_with_trust_range(uq_results["uq_qbc_for"], "UQ-QbC-force", "UQ-QbC-force.png", 
                                     self.uq_qbc_trust_lo, self.uq_qbc_trust_hi)
        
        self.logger.info("Plotting and saving the figures of UQ-RND-force-rescaled with UQ trust range")
        vis.plot_uq_with_trust_range(uq_rnd_rescaled, "UQ-RND-force-rescaled", "UQ-RND-force-rescaled.png",
                                     self.uq_qbc_trust_lo, self.uq_qbc_trust_hi)
        
        if has_ground_truth:
            self.logger.info("Plotting and saving the figures of UQ-force vs force diff")
            vis.plot_uq_vs_error(uq_results["uq_qbc_for"], uq_results["uq_rnd_for"], uq_results["diff_maxf_0_frame"])
            
            self.logger.info("Plotting and saving the figures of UQ-force-rescaled vs force diff")
            vis.plot_uq_vs_error(uq_results["uq_qbc_for"], uq_rnd_rescaled, uq_results["diff_maxf_0_frame"], rescaled=True)
            
        self.logger.info("Calculating the difference between UQ-qbc and UQ-rnd-rescaled")
        self.logger.info("Plotting and saving the figures of UQ-diff")
        vis.plot_uq_diff_parity(uq_results["uq_qbc_for"], uq_rnd_rescaled, 
                                diff_maxf=uq_results["diff_maxf_0_frame"] if has_ground_truth else None)
        
        if has_ground_truth:
            self.logger.info("Plotting and saving the figures of UQ-qbc-force and UQ-rnd-force-rescaled vs force diff")
            # Creating temp df for visualization
            df_temp = pd.DataFrame({
                "uq_qbc_for": uq_results["uq_qbc_for"],
                "uq_rnd_for_rescaled": uq_rnd_rescaled,
                "diff_maxf_0_frame": uq_results["diff_maxf_0_frame"]
            })
            vis.plot_uq_fdiff_scatter(df_temp, self.uq_scheme, 
                                  self.uq_qbc_trust_lo, self.uq_qbc_trust_hi, 
                                  self.uq_rnd_trust_lo, self.uq_rnd_trust_hi)

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
            "uq_qbc_for": uq_results["uq_qbc_for"],
            "uq_rnd_for_rescaled": uq_rnd_rescaled,
            "uq_rnd_for": uq_results["uq_rnd_for"],
        }
        if has_ground_truth:
            data_dict_uq["diff_maxf_0_frame"] = uq_results["diff_maxf_0_frame"]
            
        df_uq = pd.DataFrame(data_dict_uq)
        
        # Load descriptors (Candidates) - Using Ordered Loading with Consistency Check
        desc_datanames, desc_stru = self._load_descriptors(self.desc_dir, self.desc_filename, "candidate descriptors", 
                                                         target_names=unique_system_names,
                                                         expected_frames=expected_frames_dict)
        
        if len(desc_stru) == 0:
            raise ValueError("No candidate descriptors loaded!")
        
        # --- Enhanced Logging: Initial Data Stats ---
        self.logger.info("="*40)
        self.logger.info("       INITIAL DATA STATISTICS")
        self.logger.info("="*40)
        
        # Helper to parse sys_name from dataname (assumes format "sysname-index")
        def get_sys_name(dataname):
            return dataname.rsplit("-", 1)[0]
        
        # Helper to get pool name (dirname of sys_name)
        def get_pool_name(sys_name):
            d = os.path.dirname(sys_name)
            return d if d else "root"

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
        # --------------------------------------------

        self.logger.info(f"Collecting data to dataframe and do UQ selection")
        df_desc = pd.DataFrame(desc_stru, columns=[f"desc_stru_{i}" for i in range(desc_stru.shape[1])])
        df_desc["dataname"] = desc_datanames
        
        # Verify consistency for candidates
        if len(df_desc) != len(df_uq):
             self.logger.warning(f"Mismatch: UQ data has {len(df_uq)} frames, but descriptors have {len(df_desc)} frames.")
             # Since we added per-system check, this global mismatch should ideally not happen 
             # unless some system was skipped entirely or dataname_list logic is flawed.
             
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
        
        # 4. DIRECT Sampling
        if len(df_candidate) == 0:
            self.logger.warning("No structures selected by UQ scheme! Skipping DIRECT selection.")
            return

        self.logger.info(f"Doing DIRECT Selection on UQ-selected data")
        
        # --- Joint with Training Data Logic ---
        use_joint_sampling = False
        train_desc_stru = np.array([])
        
        if self.training_desc_dir:
            self.logger.info(f"Training descriptors provided at {self.training_desc_dir}. Attempting joint sampling.")
            _, train_desc_stru = self._load_descriptors(self.training_desc_dir, self.desc_filename, "training descriptors")
            
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
        candidate_features = df_candidate[[col for col in df_desc.columns if col.startswith("desc_stru_")]].values
        
        if use_joint_sampling:
            # Combine [Candidate; Training]
            combined_features = np.vstack([candidate_features, train_desc_stru])
            self.logger.info(f"Combined feature shape: {combined_features.shape}")
            features_for_direct = combined_features
        else:
            features_for_direct = candidate_features
            
        # DIRECT Sampling
        DIRECT_sampler = DIRECTSampler(
            structure_encoder=None,
            clustering=BirchClustering(n=self.num_selection // self.direct_k, threshold_init=self.direct_thr_init),
            select_k_from_clusters=SelectKFromClusters(k=self.direct_k),
        )
        
        DIRECT_selection = DIRECT_sampler.fit_transform(features_for_direct)
        
        selected_indices_raw = DIRECT_selection["selected_indices"]
        all_pca_features = DIRECT_selection["PCAfeatures"]
        explained_variance = DIRECT_sampler.pca.pca.explained_variance_
        selected_PC_dim = len([e for e in explained_variance if e > 1])
        
        # Filter indices to keep only candidates for Export
        n_candidates = len(candidate_features)
        
        if use_joint_sampling:
            # Filter: Keep indices < n_candidates for final export
            DIRECT_selected_indices = [idx for idx in selected_indices_raw if idx < n_candidates]
            n_from_training = len(selected_indices_raw) - len(DIRECT_selected_indices)
            self.logger.info(f"DIRECT Selection Result (Joint Mode):")
            self.logger.info(f"  - Target Total Representatives (num_selection): {self.num_selection}")
            self.logger.info(f"  - Actually Found Representatives: {len(selected_indices_raw)}")
            self.logger.info(f"  - Selected from Training Set (Ignored): {n_from_training}")
            self.logger.info(f"  - Selected from Candidate Set (New Samples): {len(DIRECT_selected_indices)}")
            
            # For Visualization: Use Joint Features and Joint Selection
            # This ensures coverage plots reflect the Joint Space
            all_features_viz = all_pca_features / explained_variance[:selected_PC_dim]
            viz_selected_indices = selected_indices_raw
            
        else:
            DIRECT_selected_indices = selected_indices_raw
            self.logger.info(f"DIRECT Selection Result (Normal Mode):")
            self.logger.info(f"  - Target New Samples (num_selection): {self.num_selection}")
            self.logger.info(f"  - Actually Selected Samples: {len(DIRECT_selected_indices)}")
            
            all_features_viz = all_pca_features / explained_variance[:selected_PC_dim]
            viz_selected_indices = DIRECT_selected_indices
        
        # Calculate PCA for ALL data (df_uq_desc) for visualization background
        # Note: We must use the same PCA projection and scaling as all_features_viz
        try:
            self.logger.info("Projecting all data onto PCA space for visualization...")
            all_desc_features = df_uq_desc[[col for col in df_desc.columns if col.startswith("desc_stru_")]].values
            
            # Use the fitted PCA from DIRECT_sampler
            full_pca_features = DIRECT_sampler.pca.transform(all_desc_features)
            
            # Ensure dimension matches selected_PC_dim
            if full_pca_features.shape[1] > selected_PC_dim:
                full_pca_features = full_pca_features[:, :selected_PC_dim]
                
            # Apply the same scaling
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
        df_remaining = df_stats_init[~df_stats_init["dataname"].isin(sampled_datanames_set)]
        
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
        if len(all_features_viz) >= self.num_selection:
             manual_selection_index = np.random.choice(len(all_features_viz), self.num_selection, replace=False)
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
        df_uq_desc = uq_filter.get_identity_labels(df_uq_desc, df_candidate, df_accurate)
        
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
        
        self.logger.info("DPEVA_TAG: WORKFLOW_FINISHED")
