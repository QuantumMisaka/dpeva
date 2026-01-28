import os
import shutil
import glob
import logging
import numpy as np
import pandas as pd
import dpdata
from copy import deepcopy

from dpeva.io.dataproc import DPTestResults
from dpeva.sampling.direct import BirchClustering, DIRECTSampler, SelectKFromClusters
from dpeva.uncertain.calculator import UQCalculator
from dpeva.uncertain.filter import UQFilter
from dpeva.uncertain.visualization import UQVisualizer

class CollectionWorkflow:
    """
    Orchestrates the Collection pipeline:
    Data Loading -> UQ Calculation -> Filtering -> DIRECT Sampling -> Visualization -> Export
    """

    def __init__(self, config):
        """
        Initialize the Collection Workflow.

        Args:
            config (dict): Configuration dictionary containing:
                - project: Project name/directory.
                - uq_select_scheme: UQ filtering scheme (e.g., 'tangent_lo').
                - testing_dir/head: Path components for test results.
                - desc_dir/filename: Path components for descriptors.
                - testdata_dir: Path to test data (ground truth/structures).
                - uq_trust_mode: 'manual' or 'auto'.
                - uq_trust_ratio/width: Global UQ parameters.
                - uq_qbc_trust_lo/hi/ratio/width: QbC specific parameters.
                - uq_rnd_rescaled_trust_lo/hi/ratio/width: RND specific parameters.
                - num_selection: Total number of structures to select.
                - direct_k: Number of clusters for DIRECT sampling.
        """
        self.config = config
        self._setup_logger()
        self._validate_config()
        
        self.project = config.get("project", "stage9-2")
        self.uq_scheme = config.get("uq_select_scheme", "tangent_lo")
        
        # Paths
        self.testing_dir = config.get("testing_dir", "test-val-npy")
        self.testing_head = config.get("testing_head", "results")
        self.desc_dir = config.get("desc_dir")
        self.desc_filename = config.get("desc_filename", "desc.npy")
        self.testdata_dir = config.get("testdata_dir")
        self.testdata_fmt = config.get("testdata_fmt", "deepmd/npy")
        
        # Training Set Paths (used in joint DIRECT for diversity maximization)
        self.training_data_dir = config.get("training_data_dir")
        self.training_desc_dir = config.get("training_desc_dir")
        
        # Save Dirs
        self.root_savedir = config.get("root_savedir", "dpeva_uq_post")
        self.view_savedir = os.path.join(self.project, self.root_savedir, "view")
        self.dpdata_savedir = os.path.join(self.project, self.root_savedir, "dpdata")
        self.df_savedir = os.path.join(self.project, self.root_savedir, "dataframe")
        
        self._ensure_dirs()
        
        # UQ Parameters
        self.uq_trust_mode = config.get("uq_trust_mode", "manual")
        
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
        if self.uq_qbc_params["lo"] is None:
             self.uq_qbc_params["lo"] = 0.12
             
        self._validate_and_fill_trust_params(self.uq_qbc_params, "uq_qbc")
        
        # Map back to instance variables for compatibility
        self.uq_qbc_trust_lo = self.uq_qbc_params["lo"]
        self.uq_qbc_trust_hi = self.uq_qbc_params["hi"]
        
        # 3. Resolve RND Parameters
        rnd_lo_default = config.get("uq_rnd_rescaled_trust_lo")
        if rnd_lo_default is None:
            rnd_lo_default = self.uq_qbc_trust_lo
            
        self.uq_rnd_params = {
            "ratio": config.get("uq_rnd_rescaled_trust_ratio", self.global_trust_ratio),
            "width": config.get("uq_rnd_rescaled_trust_width", self.global_trust_width),
            "lo": rnd_lo_default,
            "hi": config.get("uq_rnd_rescaled_trust_hi")
        }
        
        self._validate_and_fill_trust_params(self.uq_rnd_params, "uq_rnd")
        
        self.uq_rnd_trust_lo = self.uq_rnd_params["lo"]
        self.uq_rnd_trust_hi = self.uq_rnd_params["hi"]
        
        # Sampling Parameters
        self.num_selection = config.get("num_selection", 100)
        self.direct_k = config.get("direct_k", 1)
        self.direct_thr_init = config.get("direct_thr_init", 0.5)

    def _validate_and_fill_trust_params(self, params, name):
        """
        Validates consistency between lo, hi, and width.
        Fills missing values if possible.

        Args:
            params (dict): Dictionary with keys 'lo', 'hi', 'width'.
            name (str): Name of the parameter set (for logging).
        """
        lo = params.get("lo")
        hi = params.get("hi")
        width = params.get("width")
        
        if lo is not None and hi is not None:
            # Both Lo and Hi specified -> Check width consistency
            calculated_width = hi - lo
            if width is not None:
                if abs(calculated_width - width) > 1e-5:
                     explicit_width_key = f"{name}_trust_width"
                     global_width_key = "uq_trust_width"
                     
                     has_explicit_width = (explicit_width_key in self.config) or (global_width_key in self.config)
                     
                     if has_explicit_width:
                         self.logger.error(f"Configuration Conflict in {name}: lo={lo}, hi={hi} implies width={calculated_width:.4f}, "
                                           f"but width is set to {width:.4f}")
                         raise ValueError(f"Configuration Conflict in {name}: lo + width != hi")
                     else:
                         # No explicit width, so we update the derived width
                         params["width"] = calculated_width
                         self.logger.info(f"{name}: Derived width {calculated_width:.4f} from lo={lo}, hi={hi}")

        elif lo is not None and hi is None:
            # Lo specified, Hi missing -> Use width to calculate Hi
            if width is not None:
                params["hi"] = lo + width
                self.logger.info(f"{name}: Calculated hi={params['hi']:.4f} from lo={lo}, width={width}")
            else:
                pass
                
        elif lo is None and hi is not None:
             # Hi specified, Lo missing -> Use width to calculate Lo
             if width is not None:
                 params["lo"] = hi - width
                 self.logger.info(f"{name}: Calculated lo={params['lo']:.4f} from hi={hi}, width={width}")
        
    def _setup_logger(self):
        """Sets up the logging configuration."""
        self.logger = logging.getLogger(__name__)

    def _validate_config(self):
        """Validates that necessary configuration paths exist."""
        # Basic validation
        if not os.path.exists(self.config["project"]):
            self.logger.error(f"Project directory {self.config['project']} not found!")
            raise ValueError(f"Project directory {self.config['project']} not found!")

    def _ensure_dirs(self):
        """Creates necessary output directories if they don't exist."""
        for d in [self.view_savedir, self.dpdata_savedir, self.df_savedir]:
            if not os.path.exists(d):
                os.makedirs(d)

    def _load_descriptors(self, desc_dir, desc_filename="desc.npy", label="descriptors"):
        """
        Loads descriptors from a directory.
        Supports both nested structure (sys/desc.npy) and flat structure (sys.npy).
        Prioritizes flat structure (*.npy) over nested structure (*/desc.npy).
        
        Args:
            desc_dir (str): Path to the descriptor directory.
            desc_filename (str): Name of the descriptor file (default: "desc.npy") used in nested structure.
            label (str): Label for logging purposes.

        Returns:
            tuple: (desc_datanames, desc_stru)
        """
        self.logger.info(f"Loading {label} from {desc_dir}")
        
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
                 
        desc_datanames = []
        desc_stru = []
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
                
            for i in range(len(one_desc)):
                desc_datanames.append(f"{keyname}-{i}")
            
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

    def _count_frames_in_data(self, data_dir, fmt="deepmd/npy"):
        """Counts total frames in dataset to verify consistency."""
        total_frames = 0
        found_dirs = sorted(glob.glob(os.path.join(data_dir, "*")))
        for d in found_dirs:
            if not os.path.isdir(d):
                continue
            try:
                # Use dpdata to count frames quickly if possible, or just load
                # Loading might be slow for huge datasets, but necessary for consistency check
                # A lightweight check is to look at set.000/box.npy
                # But let's use dpdata for robustness
                try:
                    sys = dpdata.LabeledSystem(d, fmt=fmt)
                except:
                    sys = dpdata.System(d, fmt=fmt)
                total_frames += len(sys)
            except Exception:
                pass
        return total_frames

    def run(self):
        """
        Executes the main collection workflow:
        1. Load prediction results and calculate UQ.
        2. Filter data based on UQ thresholds (manual or auto).
        3. Visualize UQ distributions and filtering results.
        4. Perform DIRECT sampling on candidate structures.
        5. Export sampled structures as dpdata.
        """
        self.logger.info(f"Initializing selection in {self.project} ---")
        
        # 1. Load Data & Calculate UQ
        self.logger.info("Loading the test results")
        preds = [
            DPTestResults(f"./{self.project}/{i}/{self.testing_dir}/{self.testing_head}")
            for i in range(4)
        ]
        
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
            
            self.logger.info(f"Calculating QbC thresholds with ratio={_qbc_ratio} and width={_qbc_width}")
            calc_lo_qbc = calculator.calculate_trust_lo(uq_results["uq_qbc_for"], ratio=_qbc_ratio)
            if calc_lo_qbc is not None:
                self.uq_qbc_trust_lo = calc_lo_qbc
                self.logger.info(f"Auto-calculated QbC Trust Lo: {self.uq_qbc_trust_lo:.4f}")
            else:
                self.logger.warning(f"Auto-calculation for QbC failed. Fallback to manual Trust Lo: {self.uq_qbc_trust_lo:.4f}")
                
            self.uq_qbc_trust_hi = self.uq_qbc_trust_lo + _qbc_width
            self.logger.info(f"Final QbC Trust Range: [{self.uq_qbc_trust_lo:.4f}, {self.uq_qbc_trust_hi:.4f}]")
            
            # Calculate for RND Rescaled
            _rnd_ratio = self.uq_rnd_params["ratio"]
            _rnd_width = self.uq_rnd_params["width"]
            
            self.logger.info(f"Calculating RND thresholds with ratio={_rnd_ratio} and width={_rnd_width}")
            calc_lo_rnd = calculator.calculate_trust_lo(uq_rnd_rescaled, ratio=_rnd_ratio)
            if calc_lo_rnd is not None:
                self.uq_rnd_trust_lo = calc_lo_rnd
                self.logger.info(f"Auto-calculated RND-rescaled Trust Lo: {self.uq_rnd_trust_lo:.4f}")
            else:
                self.logger.warning(f"Auto-calculation for RND-rescaled failed. Fallback to manual Trust Lo: {self.uq_rnd_trust_lo:.4f}")
                
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
        
        data_dict_uq = {
            "dataname": datanames_ind_list,
            "uq_qbc_for": uq_results["uq_qbc_for"],
            "uq_rnd_for_rescaled": uq_rnd_rescaled,
            "uq_rnd_for": uq_results["uq_rnd_for"],
        }
        if has_ground_truth:
            data_dict_uq["diff_maxf_0_frame"] = uq_results["diff_maxf_0_frame"]
            
        df_uq = pd.DataFrame(data_dict_uq)
        
        # Load descriptors (Candidates)
        desc_datanames, desc_stru = self._load_descriptors(self.desc_dir, self.desc_filename, "candidate descriptors")
        
        if len(desc_stru) == 0:
            raise ValueError("No candidate descriptors loaded!")
        
        self.logger.info(f"Collecting data to dataframe and do UQ selection")
        df_desc = pd.DataFrame(desc_stru, columns=[f"desc_stru_{i}" for i in range(desc_stru.shape[1])])
        df_desc["dataname"] = desc_datanames
        
        # Verify consistency for candidates
        if len(df_desc) != len(df_uq):
             self.logger.warning(f"Mismatch: UQ data has {len(df_uq)} frames, but descriptors have {len(df_desc)} frames.")
             # We can proceed if merge handles it (inner join), but it's risky.
             # Ideally they should match.
        
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
            self.logger.info(f"DIRECT selected {len(selected_indices_raw)} total samples. "
                             f"After filtering training samples, {len(DIRECT_selected_indices)} candidates remain.")
            
            # For Visualization: Use Joint Features and Joint Selection
            # This ensures coverage plots reflect the Joint Space
            all_features_viz = all_pca_features / explained_variance[:selected_PC_dim]
            viz_selected_indices = selected_indices_raw
            
        else:
            DIRECT_selected_indices = selected_indices_raw
            all_features_viz = all_pca_features / explained_variance[:selected_PC_dim]
            viz_selected_indices = DIRECT_selected_indices
        
        df_final = df_candidate.iloc[DIRECT_selected_indices]
        
        self.logger.info(f"Saving df_uq_desc_selected_final dataframe to {self.df_savedir}/df_uq_desc_sampled-final.csv")
        df_final.to_csv(f"{self.df_savedir}/df_uq_desc_sampled-final.csv", index=True)
        
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
                                      n_candidates=n_candidates if use_joint_sampling else None)
        
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
        
        # Ensure length matches
        if len(df_pcs) == len(df_uq):
            df_alldataPC_visual = pd.concat([df_uq.reset_index(drop=True), df_pcs.reset_index(drop=True)], axis=1)
            df_alldataPC_visual.to_csv(f"{self.df_savedir}/final_df.csv", index=True)
        else:
            self.logger.warning(f"Skipping final_df.csv: Length mismatch df_uq={len(df_uq)} vs PCs={len(df_pcs)}")

        
        # 6. Export dpdata
        self.logger.info(f"Sampling dpdata based on selected indices")
        sampled_datanames = df_final['dataname'].to_list()
        
        self.logger.info(f"Loading the target testing data from {self.testdata_dir}")
        
        test_data = [] 
        # Check if testdata_dir has subdirectories
        found_dirs = sorted(glob.glob(os.path.join(self.testdata_dir, "*")))
        
        for d in found_dirs:
            if not os.path.isdir(d):
                continue
            try:
                sys = dpdata.LabeledSystem(d, fmt=self.testdata_fmt)
            except:
                try:
                    sys = dpdata.System(d, fmt=self.testdata_fmt)
                except Exception as e:
                    self.logger.warning(f"Failed to load {d}: {e}")
                    continue
            test_data.append(sys)
            
        sampled_dpdata = dpdata.MultiSystems()
        other_dpdata = dpdata.MultiSystems()
        
        for sys in test_data:
            # Match frames based on dataname (keyname-index)
            sys_name = sys.short_name
            
            for i in range(len(sys)):
                dataname_sys = f"{sys_name}-{i}"
                if dataname_sys in sampled_datanames:
                    sampled_dpdata.append(sys[i])
                else:
                    other_dpdata.append(sys[i])
                    
        self.logger.info(f'Sampled dpdata: {sampled_dpdata}')
        self.logger.info(f'Other dpdata: {other_dpdata}')
        
        self.logger.info(f"Dumping sampled and other dpdata to {self.dpdata_savedir}")
        
        if os.path.exists(f"{self.dpdata_savedir}/sampled_dpdata"):
            shutil.rmtree(f"{self.dpdata_savedir}/sampled_dpdata")
        if os.path.exists(f"{self.dpdata_savedir}/other_dpdata"):
            shutil.rmtree(f"{self.dpdata_savedir}/other_dpdata")
            
        sampled_dpdata.to_deepmd_npy(f"{self.dpdata_savedir}/sampled_dpdata")
        other_dpdata.to_deepmd_npy(f"{self.dpdata_savedir}/other_dpdata")
        
        self.logger.info("All Done!")
