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

class CollectWorkflow:
    """
    Orchestrates the Collection pipeline:
    Data Loading -> UQ Calculation -> Filtering -> DIRECT Sampling -> Visualization -> Export
    """

    def __init__(self, config):
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
        self.testdata_string = config.get("testdata_string", "O*")
        
        # Save Dirs
        self.root_savedir = config.get("root_savedir", "dpeva_uq_post")
        self.view_savedir = os.path.join(self.project, self.root_savedir, "view")
        self.dpdata_savedir = os.path.join(self.project, self.root_savedir, "dpdata")
        self.df_savedir = os.path.join(self.project, self.root_savedir, "dataframe")
        
        self._ensure_dirs()
        
        # UQ Parameters
        self.uq_qbc_trust_lo = config.get("uq_qbc_trust_lo", 0.12)
        self.uq_qbc_trust_hi = config.get("uq_qbc_trust_hi", 0.22)
        self.uq_rnd_trust_lo = config.get("uq_rnd_rescaled_trust_lo", self.uq_qbc_trust_lo)
        self.uq_rnd_trust_hi = config.get("uq_rnd_rescaled_trust_hi", self.uq_qbc_trust_hi)
        
        # Sampling Parameters
        self.num_selection = config.get("num_selection", 100)
        self.direct_k = config.get("direct_k", 1)
        self.direct_thr_init = config.get("direct_thr_init", 0.5)

    def _setup_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            filemode='w',
            filename=self.config.get("log_filename", "UQ-DIRECT-selection.log"),
        )
        self.logger = logging.getLogger(__name__)

    def _validate_config(self):
        # Basic validation
        if not os.path.exists(self.config["project"]):
            self.logger.error(f"Project directory {self.config['project']} not found!")
            raise ValueError(f"Project directory {self.config['project']} not found!")

    def _ensure_dirs(self):
        for d in [self.view_savedir, self.dpdata_savedir, self.df_savedir]:
            if not os.path.exists(d):
                os.makedirs(d)

    def run(self):
        self.logger.info(f"Initializing selection in {self.project} ---")
        
        # 1. Load Data & Calculate UQ
        self.logger.info("Loading the test results")
        preds = [
            DPTestResults(f"./{self.project}/{i}/{self.testing_dir}/{self.testing_head}")
            for i in range(4)
        ]
        
        self.logger.info("Dealing with force difference between 0 head prediction and existing label")
        self.logger.info("Dealing with atomic force and average 1, 2, 3")
        self.logger.info("Dealing with atomic force UQ by DPGEN formula, in QbC and RND-like")
        self.logger.info("Dealing with QbC force UQ")
        self.logger.info("Dealing with RND-like force UQ")
        
        calculator = UQCalculator()
        uq_results = calculator.compute_qbc_rnd(preds[0], preds[1], preds[2], preds[3])
        
        self.logger.info("Aligning UQ-RND to UQ-QbC by Z-Score")
        uq_rnd_rescaled = calculator.align_scales(uq_results["uq_qbc_for"], uq_results["uq_rnd_for"])
        
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
                                     self.uq_qbc_trust_lo, self.uq_qbc_trust_hi) # Note: using same thresholds as QbC per original script logic for visualization boundary
        
        self.logger.info("Plotting and saving the figures of UQ-force vs force diff")
        vis.plot_uq_vs_error(uq_results["uq_qbc_for"], uq_results["uq_rnd_for"], uq_results["diff_maxf_0_frame"])
        
        self.logger.info("Plotting and saving the figures of UQ-force-rescaled vs force diff")
        vis.plot_uq_vs_error(uq_results["uq_qbc_for"], uq_rnd_rescaled, uq_results["diff_maxf_0_frame"], rescaled=True)
        
        self.logger.info("Calculating the difference between UQ-qbc and UQ-rnd-rescaled")
        self.logger.info("Plotting and saving the figures of UQ-diff")
        self.logger.info("Plotting and saving the figures of UQ-diff vs UQ")
        self.logger.info("Plotting and saving the figures of UQ-diff vs force diff")
        vis.plot_uq_diff_parity(uq_results["uq_qbc_for"], uq_rnd_rescaled, uq_results["diff_maxf_0_frame"])
        
        self.logger.info("Plotting and saving the figures of UQ-qbc-force and UQ-rnd-force-rescaled vs force diff")
        # Creating temp df for visualization
        df_temp = pd.DataFrame({
            "uq_qbc_for": uq_results["uq_qbc_for"],
            "uq_rnd_for_rescaled": uq_rnd_rescaled,
            "uq_rnd_for": uq_results["uq_rnd_for"],
            "diff_maxf_0_frame": uq_results["diff_maxf_0_frame"]
        })
        # This call actually handles the "Plotting ... vs force diff" logic internally in vis
        vis.plot_2d_uq_scatter(df_temp, self.uq_scheme, 
                              self.uq_qbc_trust_lo, self.uq_qbc_trust_hi, 
                              self.uq_rnd_trust_lo, self.uq_rnd_trust_hi)

        # 3. Data Preparation & Filtering
        self.logger.info("Dealing with Selection in Target dpdata")
        datanames_ind_list = [f"{i[0]}-{i[1]}" for i in preds[0].dataname_list]
        
        df_uq = pd.DataFrame({
            "dataname": datanames_ind_list,
            "uq_qbc_for": uq_results["uq_qbc_for"],
            "uq_rnd_for_rescaled": uq_rnd_rescaled,
            "uq_rnd_for": uq_results["uq_rnd_for"],
            "diff_maxf_0_frame": uq_results["diff_maxf_0_frame"]
        })
        
        # Load descriptors
        self.logger.info(f"Loading the target descriptors from {self.testdata_dir}") # Log message per original
        desc_string_test = f'{self.desc_dir}/*/{self.desc_filename}'
        desc_datanames = []
        desc_stru = []
        desc_iter_list = sorted(glob.glob(desc_string_test))
        
        for f in desc_iter_list:
            directory, _ = os.path.split(f)
            _, keyname = os.path.split(directory)
            one_desc = np.load(f)
            for i in range(len(one_desc)):
                desc_datanames.append(f"{keyname}-{i}")
                desc_stru.append(np.mean(one_desc[i], axis=0).reshape(1, -1))
        
        desc_stru = np.concatenate(desc_stru, axis=0)
        
        self.logger.info(f"Collecting data to dataframe and do UQ selection")
        df_desc = pd.DataFrame(desc_stru, columns=[f"desc_stru_{i}" for i in range(desc_stru.shape[1])])
        df_desc["dataname"] = desc_datanames
        
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
        # We need to pass the updated df_uq which now has 'uq_identity'
        vis.plot_2d_uq_scatter(df_uq, self.uq_scheme, 
                              self.uq_qbc_trust_lo, self.uq_qbc_trust_hi, 
                              self.uq_rnd_trust_lo, self.uq_rnd_trust_hi)
                              
        self.logger.info("Plotting and saving the figure of UQ-Candidate in QbC space against Max Force Diff")
        self.logger.info("Plotting and saving the figure of UQ-Candidate in RND-rescaled space against Max Force Diff")
        vis.plot_candidate_vs_error(df_uq, df_candidate)
        
        # 4. DIRECT Sampling
        self.logger.info(f"Doing DIRECT Selection on UQ-selected data")
        
        DIRECT_sampler = DIRECTSampler(
            structure_encoder=None,
            clustering=BirchClustering(n=self.num_selection // self.direct_k, threshold_init=self.direct_thr_init),
            select_k_from_clusters=SelectKFromClusters(k=self.direct_k),
        )
        
        desc_features = [col for col in df_desc.columns if col.startswith("desc_stru_")]
        DIRECT_selection = DIRECT_sampler.fit_transform(df_candidate[desc_features].values)
        
        DIRECT_selected_indices = DIRECT_selection["selected_indices"]
        explained_variance = DIRECT_sampler.pca.pca.explained_variance_
        selected_PC_dim = len([e for e in explained_variance if e > 1])
        
        # Normalize features for coverage calc
        DIRECT_selection["PCAfeatures_unweighted"] = DIRECT_selection["PCAfeatures"] / explained_variance[:selected_PC_dim]
        all_features = DIRECT_selection["PCAfeatures_unweighted"]
        
        df_final = df_candidate.iloc[DIRECT_selected_indices]
        
        self.logger.info(f"Saving df_uq_desc_selected_final dataframe to {self.df_savedir}/df_uq_desc_sampled-final.csv")
        df_final.to_csv(f"{self.df_savedir}/df_uq_desc_sampled-final.csv", index=True)
        
        # 5. Visualization (Sampling)
        self.logger.info(f"Visualization of DIRECT results compared with Random")
        
        # Random Simulation
        np.random.seed(42)
        manual_selection_index = np.random.choice(len(all_features), self.num_selection, replace=False)
        
        # Coverage Scores
        def calculate_feature_coverage_score(all_features, selected_indices, n_bins=100):
            selected_features = all_features[selected_indices]
            # Use fixed bins based on min/max of ALL features
            bins = np.linspace(min(all_features), max(all_features), n_bins)
            n_all = np.count_nonzero(np.histogram(all_features, bins=bins)[0])
            n_select = np.count_nonzero(np.histogram(selected_features, bins=bins)[0])
            return n_select / n_all if n_all > 0 else 0

        def calculate_all_FCS(all_features, selected_indices):
            return [calculate_feature_coverage_score(all_features[:, i], selected_indices) 
                    for i in range(all_features.shape[1])]

        scores_DIRECT = calculate_all_FCS(all_features, DIRECT_selection["selected_indices"])
        scores_MS = calculate_all_FCS(all_features, manual_selection_index)
        
        self.logger.info(f"Visualization of final selection results in PCA space")
        
        # Pass df_uq (full data) to visualizer to help it reconstruct global indices if needed
        # But our visualizer helper just plots what we give it or recalculates PCA
        # We need to map final indices to global context for the visualizer
        
        # Note: df_candidate.index gives indices relative to df_uq_desc (original concat)
        # df_final.index gives the same global indices
        # We need to pass these global indices to visualizer
        
        # Add uq_identity to df_uq_desc for visualizer usage
        df_uq_desc = uq_filter.get_identity_labels(df_uq_desc, df_candidate, df_accurate)
        
        PCs_df = vis.plot_pca_analysis(explained_variance, selected_PC_dim, all_features, 
                                      DIRECT_selected_indices, manual_selection_index,
                                      scores_DIRECT, scores_MS, 
                                      df_uq_desc, df_final.index)
        
        self.logger.info(f"Saving the PCA view of UQ-DIRECT sampling to {self.view_savedir}/Final_sampled_PCAview.png")
        
        # Save Final PCA Data
        df_alldataPC_visual = pd.concat([df_uq, PCs_df], axis=1)
        df_alldataPC_visual.to_csv(f"{self.df_savedir}/final_df.csv", index=True)
        
        # 6. Export dpdata
        self.logger.info(f"Sampling dpdata based on selected indices")
        sampled_datanames = df_final['dataname'].to_list()
        
        self.logger.info(f"Loading the target testing data from {self.testdata_dir}")
        test_data = dpdata.MultiSystems.from_dir(f"{self.testdata_dir}", 
                                                 f"{self.testdata_string}", 
                                                 fmt=f"{self.testdata_fmt}")
        
        sampled_dpdata = dpdata.MultiSystems()
        other_dpdata = dpdata.MultiSystems()
        
        for lbsys in test_data:
            for ind, sys in enumerate(lbsys):
                dataname_sys = f"{sys.short_name}-{ind}"
                if dataname_sys in sampled_datanames:
                    sampled_dpdata.append(sys)
                else:
                    other_dpdata.append(sys)
                    
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
