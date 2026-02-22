import os
import sys
import logging
from typing import Union, Dict
import pandas as pd
import numpy as np

from dpeva.config import CollectionConfig
from dpeva.constants import WORKFLOW_FINISHED_TAG, COL_DESC_PREFIX, COL_UQ_QBC, COL_UQ_RND
from dpeva.uncertain.visualization import UQVisualizer
from dpeva.submission.manager import JobManager
from dpeva.submission.templates import JobConfig

# New Managers
from dpeva.io.collection import CollectionIOManager
from dpeva.uncertain.manager import UQManager
from dpeva.sampling.manager import SamplingManager

def get_sys_name(dataname):
    return dataname.rsplit("-", 1)[0]

def get_pool_name(sys_name):
    d = os.path.dirname(sys_name)
    return d if d else "root"

class CollectionWorkflow:
    """
    Orchestrates the Collection pipeline (Refactored).
    Delegates responsibilities to dedicated Managers:
    - CollectionIOManager: I/O & Data Loading
    - UQManager: UQ Calculation & Filtering
    - SamplingManager: DIRECT Sampling
    """

    def __init__(self, config: Union[Dict, CollectionConfig], config_path=None):
        self.logger = logging.getLogger(__name__)
        
        # 1. Config Loading
        if isinstance(config, dict):
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
        
        # 2. Setup Context
        self.project = self.config.project
        self.root_savedir = str(self.config.root_savedir)
        
        # Initialize Managers
        self.io_manager = CollectionIOManager(self.project, self.root_savedir)
        self.uq_manager = UQManager(
            project_dir=self.project,
            testing_dir=self.config.testing_dir,
            testing_head=self.config.results_prefix,
            uq_config={
                "trust_mode": self.config.uq_trust_mode,
                "scheme": self.config.uq_select_scheme,
                "auto_bounds": self.config.uq_auto_bounds,
                "qbc_params": {
                    "ratio": self.config.uq_qbc_trust_ratio or self.config.uq_trust_ratio,
                    "width": self.config.uq_qbc_trust_width or self.config.uq_trust_width,
                    "lo": self.config.uq_qbc_trust_lo,
                    "hi": self.config.uq_qbc_trust_hi
                },
                "rnd_params": {
                    "ratio": self.config.uq_rnd_rescaled_trust_ratio or self.config.uq_trust_ratio,
                    "width": self.config.uq_rnd_rescaled_trust_width or self.config.uq_trust_width,
                    "lo": self.config.uq_rnd_rescaled_trust_lo,
                    "hi": self.config.uq_rnd_rescaled_trust_hi
                }
            },
            num_models=self.config.num_models
        )
        self.sampling_manager = SamplingManager(self.config.model_dump())
        
        # Backend Handling
        env_backend = os.environ.get("DPEVA_INTERNAL_BACKEND")
        if env_backend:
            self.logger.info(f"Overriding backend to '{env_backend}'")
            self.backend = env_backend
        else:
            self.backend = self.config.submission.backend
            
        self.slurm_config = self.config.submission.slurm_config

        # 3. Setup Environment
        self._validate_config()
        self.io_manager.ensure_dirs()
        self.io_manager.configure_logging()
        
    def _validate_config(self):
        if not os.path.exists(self.project):
            raise ValueError(f"Project directory {self.project} not found!")
        if not os.path.exists(self.config.desc_dir):
            raise ValueError(f"Descriptor directory not found: {self.config.desc_dir}")
        if not os.path.exists(self.config.testdata_dir):
            raise ValueError(f"Test data directory not found: {self.config.testdata_dir}")

    def run(self):
        if self.backend == "slurm":
            self._submit_to_slurm()
            return
        self.logger.info(f"Initializing selection in {self.project} ---")
        vis = UQVisualizer(self.io_manager.view_savedir, dpi=self.config.fig_dpi)

        # ---------------------------------------------------------
        # Phase 1: UQ Analysis & Filtering
        # ---------------------------------------------------------
        if self.config.uq_trust_mode == "no_filter":
            self.logger.info("UQ Trust Mode is 'no_filter'. Skipping UQ.")
            # Load Descriptors directly
            desc_datanames, desc_stru = self.io_manager.load_descriptors(str(self.config.desc_dir), "candidate descriptors")
            if len(desc_stru) == 0: raise ValueError("No descriptors loaded!")
            
            # Construct dummy frames
            df_desc = pd.DataFrame(desc_stru, columns=[f"{COL_DESC_PREFIX}{i}" for i in range(desc_stru.shape[1])])
            df_desc["dataname"] = desc_datanames
            
            df_uq = df_desc[["dataname"]].copy()
            df_uq["uq_identity"] = "candidate"
            
            df_uq_desc = df_desc.copy()
            df_candidate = df_desc.copy()
            df_candidate["uq_identity"] = "candidate"
            
            # Identify systems for later use
            unique_system_names = sorted(list(set(get_sys_name(d) for d in desc_datanames)))
            
            # Stats
            self._log_initial_stats(desc_datanames)
            
        else:
            # 1.1 Load Predictions
            preds, has_gt = self.uq_manager.load_predictions()
            
            # 1.2 Compute UQ
            uq_results, uq_rnd_rescaled = self.uq_manager.run_analysis(preds)
            
            # 1.3 Auto Threshold
            self.uq_manager.run_auto_threshold(uq_results, uq_rnd_rescaled)
            
            # 1.4 Visualization (UQ Distributions)
            vis.plot_uq_distribution(uq_results[COL_UQ_QBC], uq_results[COL_UQ_RND], uq_rnd_rescaled)
            vis.plot_uq_with_trust_range(uq_results[COL_UQ_QBC], "UQ-QbC-force", "UQ-QbC-force.png",
                                        self.uq_manager.qbc_params["lo"], self.uq_manager.qbc_params["hi"])
            
            # 1.5 Prepare Data for Filtering
            unique_system_names = []
            seen = set()
            for item in preds[0].dataname_list:
                name = item[0]
                if name not in seen:
                    seen.add(name)
                    unique_system_names.append(name)
            
            expected_frames = {sys: preds[0].datanames_nframe.get(sys, 0) for sys in unique_system_names}
            
            # Load Descriptors
            desc_datanames, desc_stru = self.io_manager.load_descriptors(
                str(self.config.desc_dir), "candidate descriptors", 
                target_names=unique_system_names, expected_frames=expected_frames
            )
            
            # Build DataFrames
            df_desc = pd.DataFrame(desc_stru, columns=[f"{COL_DESC_PREFIX}{i}" for i in range(desc_stru.shape[1])])
            df_desc["dataname"] = desc_datanames
            
            datanames_ind_list = [f"{i[0]}-{i[1]}" for i in preds[0].dataname_list]
            data_dict_uq = {
                "dataname": datanames_ind_list,
                COL_UQ_QBC: uq_results[COL_UQ_QBC],
                "uq_rnd_for_rescaled": uq_rnd_rescaled,
                COL_UQ_RND: uq_results[COL_UQ_RND],
            }
            if has_gt:
                data_dict_uq["diff_maxf_0_frame"] = uq_results["diff_maxf_0_frame"]
            
            df_uq_raw = pd.DataFrame(data_dict_uq)
            
            # Merge
            if len(df_uq_raw) == len(df_desc) and np.array_equal(df_uq_raw["dataname"].values, df_desc["dataname"].values):
                 df_uq_desc = pd.concat([df_uq_raw, df_desc.drop(columns=["dataname"])], axis=1)
            else:
                 df_uq_desc = pd.merge(df_uq_raw, df_desc, on="dataname")
            
            self.io_manager.save_dataframe(df_uq_desc, "df_uq_desc.csv")
            
            # 1.6 Filtering
            df_candidate, df_accurate, df_failed, uq_filter = self.uq_manager.run_filtering(df_uq_desc)
            
            # Add Labels
            df_uq = uq_filter.get_identity_labels(df_uq_raw, df_candidate, df_accurate)
            self.io_manager.save_dataframe(df_uq, "df_uq.csv")
            self.io_manager.save_dataframe(df_candidate, "df_uq_desc_sampled-UQ.csv")
            
            # 1.7 Visualize Selection
            vis.plot_uq_identity_scatter(df_uq, self.config.uq_select_scheme,
                                        self.uq_manager.qbc_params["lo"], self.uq_manager.qbc_params["hi"],
                                        self.uq_manager.rnd_params["lo"], self.uq_manager.rnd_params["hi"])
            
            # 1.8 Error Analysis (if Ground Truth exists)
            if has_gt:
                self.logger.info("Ground Truth available. Plotting UQ vs Error...")
                vis.plot_uq_vs_error(
                    uq_results[COL_UQ_QBC], 
                    uq_results[COL_UQ_RND], 
                    uq_results["diff_maxf_0_frame"]
                )
                if uq_rnd_rescaled is not None:
                    vis.plot_uq_vs_error(
                        uq_results[COL_UQ_QBC], 
                        uq_rnd_rescaled, 
                        uq_results["diff_maxf_0_frame"],
                        rescaled=True
                    )
                    
                    diff_maxf = uq_results["diff_maxf_0_frame"]
                    vis.plot_uq_diff_parity(uq_results[COL_UQ_QBC], uq_rnd_rescaled, diff_maxf)

            self._log_initial_stats(desc_datanames)


        # ---------------------------------------------------------
        # Phase 2: Sampling
        # ---------------------------------------------------------
        if len(df_candidate) == 0:
            self.logger.warning("No candidates selected. Skipping Sampling.")
            self.logger.info(WORKFLOW_FINISHED_TAG)
            return

        # 2.1 Prepare Features (Joint Logic)
        train_desc_stru = np.array([])
        if self.config.training_desc_dir:
            _, train_desc_stru = self.io_manager.load_descriptors(str(self.config.training_desc_dir), "training descriptors")
            
        features, use_joint, n_candidates = self.sampling_manager.prepare_features(df_candidate, df_desc, train_desc_stru)
        
        # 2.2 Load Atomic Features if needed (2-DIRECT)
        X_atom, n_atoms = None, None
        if self.sampling_manager.sampler_type == "2-direct":
            X_atom, n_atoms = self.io_manager.load_atomic_features(str(self.config.desc_dir), df_candidate)
            
        # Prepare background features (full pool) for visualization
        background_features = df_desc[[col for col in df_desc.columns if col.startswith(COL_DESC_PREFIX)]].values

        # 2.3 Execute Sampling
        sampling_results = self.sampling_manager.execute_sampling(features, X_atom, n_atoms, background_features=background_features)
        
        selected_indices = sampling_results["selected_indices"]
        pca_features = sampling_results["pca_features"]
        explained_var = sampling_results["explained_variance"]
        random_indices = sampling_results["random_indices"]
        scores_direct = sampling_results["scores_direct"]
        scores_random = sampling_results["scores_random"]
        full_pca_features = sampling_results.get("full_pca_features")
        
        # 2.4 Handle Joint Indices
        if use_joint:
            final_indices = [idx for idx in selected_indices if idx < n_candidates]
        else:
            final_indices = selected_indices
            
        df_final = df_candidate.iloc[final_indices]
        self.io_manager.save_dataframe(df_final, "final_df.csv")
        
        # 2.5 Sampling Stats & Visualization
        self._log_sampling_stats(df_final)
        
        self.logger.info("Visualizing Sampling Results (PCA & Coverage)...")
        # For PCA visualization, we need the full PCA features
        # If Joint Sampling was used, 'features' includes training data
        # We need to distinguish candidates vs training in the plot
        
        # In Joint mode, 'n_candidates' is the split point
        # For plot_pca_analysis, we pass 'features' as 'all_features'
        
        vis.plot_pca_analysis(
            explained_variance=explained_var,
            selected_PC_dim=pca_features.shape[1],
            all_features=pca_features,  # This is PCA-transformed 'features'
            direct_indices=selected_indices,
            random_indices=random_indices,
            scores_direct=scores_direct,
            scores_random=scores_random,
            df_uq=df_candidate, # df_candidate corresponds to the first n_candidates rows of features
            final_indices=final_indices,
            n_candidates=n_candidates if use_joint else None,
            full_features=full_pca_features # We pass the full features for background plot
        )
        
        # ---------------------------------------------------------
        # Phase 3: Export
        # ---------------------------------------------------------
        self.io_manager.export_dpdata(str(self.config.testdata_dir), df_final, unique_system_names)
        self.logger.info(WORKFLOW_FINISHED_TAG)


    def _submit_to_slurm(self):
        """Delegates Slurm submission to JobManager (inline for now as it's simple)."""
        # Logic kept similar to original but simplified
        if not self.config_path: raise ValueError("Config path required for Slurm.")
        
        project_abs = os.path.abspath(self.project)
        if not os.path.exists(project_abs): os.makedirs(project_abs)
        
        cmd = f"{sys.executable} -m dpeva.cli collect {os.path.abspath(self.config_path)}"
        
        # Determine Environment Setup
        # Prioritize config.submission.env_setup, fallback to "export DPEVA_INTERNAL_BACKEND=local" logic
        # But actually, we need BOTH if we want env setup + backend override.
        # User provided env_setup might contain conda activate etc.
        
        user_env_setup = self.config.submission.env_setup
        if isinstance(user_env_setup, list):
            user_env_setup = "\n".join(user_env_setup)
        elif user_env_setup is None:
            user_env_setup = ""
            
        # Ensure backend override is present
        internal_backend_setup = "export DPEVA_INTERNAL_BACKEND=local"
        
        final_env_setup = f"{user_env_setup}\n{internal_backend_setup}"
        
        job_conf = JobConfig(
            command=cmd,
            job_name=self.slurm_config.get("job_name", "dpeva_collect"),
            partition=self.slurm_config.get("partition", "CPU-MISC"),
            qos=self.slurm_config.get("qos"),
            ntasks=self.slurm_config.get("ntasks", 4),
            output_log=os.path.join(project_abs, "collect_slurm.out"),
            error_log=os.path.join(project_abs, "collect_slurm.err"),
            env_setup=final_env_setup
        )
        
        manager = JobManager(mode="slurm")
        script_path = os.path.join(project_abs, "submit_collect.slurm")
        manager.generate_script(job_conf, script_path)
        manager.submit(script_path, working_dir=project_abs)

    def _log_initial_stats(self, desc_datanames):
        """Logs initial stats."""
        df = pd.DataFrame({"dataname": desc_datanames})
        df["pool"] = df["dataname"].apply(lambda x: get_pool_name(get_sys_name(x)))
        stats = df.groupby("pool").agg(num_systems=("dataname", lambda x: x.apply(get_sys_name).nunique()), num_frames=("dataname", "count"))
        self.logger.info(f"Initial Stats:\n{stats}")
        return stats

    def _log_sampling_stats(self, df_final):
        """Logs sampling stats."""
        df_final["pool"] = df_final["dataname"].apply(lambda x: get_pool_name(get_sys_name(x)))
        stats_sampled = df_final.groupby("pool").agg(sampled_frames=("dataname", "count"))
        self.logger.info(f"Sampled Stats:\n{stats_sampled}")
