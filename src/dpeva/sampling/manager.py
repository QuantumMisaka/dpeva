import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional

from dpeva.sampling.direct import BirchClustering, DIRECTSampler, SelectKFromClusters
from dpeva.sampling.two_step_direct import TwoStepDIRECTSampler
from dpeva.constants import COL_DESC_PREFIX

class SamplingManager:
    """
    Manages DIRECT Sampling Workflow:
    - Feature preparation (Joint Sampling logic)
    - Sampler Selection (DIRECT vs 2-DIRECT)
    - Execution
    """
    
    def __init__(self, config_dict: Dict):
        self.config = config_dict
        self.logger = logging.getLogger(__name__)
        
        # Unpack critical params
        self.sampler_type = self.config.get("sampler_type", "direct")
        self.direct_k = self.config.get("direct_k", 1)
        self.direct_thr_init = self.config.get("direct_thr_init", 0.1)

    def prepare_features(self, df_candidate: pd.DataFrame, df_desc: pd.DataFrame, 
                        train_desc_stru: np.ndarray) -> Tuple[np.ndarray, bool, int]:
        """
        Prepares feature matrix, potentially merging with training data.
        Returns: (features, use_joint, n_candidates)
        """
        candidate_features = df_candidate[[col for col in df_desc.columns if col.startswith(COL_DESC_PREFIX)]].values
        n_candidates = len(candidate_features)
        
        use_joint = False
        features = candidate_features
        
        if len(train_desc_stru) > 0:
            if self.sampler_type == "2-direct":
                self.logger.warning("2-DIRECT does not support Joint Sampling yet. Ignoring training data.")
            else:
                self.logger.info(f"Joint Sampling: Merging {n_candidates} candidates with {len(train_desc_stru)} training samples.")
                features = np.vstack([candidate_features, train_desc_stru])
                use_joint = True
                
        return features, use_joint, n_candidates

    def execute_sampling(self, features: np.ndarray, 
                        atom_features: Optional[List[np.ndarray]] = None,
                        atom_counts: Optional[List[int]] = None) -> Tuple[List[int], np.ndarray, np.ndarray]:
        """
        Runs the sampler.
        Returns: (selected_indices, pca_features, explained_variance)
        """
        if self.sampler_type == "2-direct":
            return self._run_2_direct(features, atom_features, atom_counts)
        else:
            return self._run_direct(features)

    def _run_direct(self, features):
        self.logger.info("Running Standard DIRECT...")
        n_clusters = self.config.get("direct_n_clusters")
        
        sampler = DIRECTSampler(
            structure_encoder=None,
            clustering=BirchClustering(n=n_clusters, threshold_init=self.direct_thr_init),
            select_k_from_clusters=SelectKFromClusters(k=self.direct_k)
        )
        
        res = sampler.fit_transform(features)
        return res["selected_indices"], res["PCAfeatures"], sampler.pca.pca.explained_variance_

    def _run_2_direct(self, features, atom_features, atom_counts):
        self.logger.info("Running 2-DIRECT...")
        if atom_features is None:
            raise ValueError("Atomic features required for 2-DIRECT")
            
        sampler = TwoStepDIRECTSampler(
            step1_clustering=BirchClustering(
                n=self.config.get("step1_n_clusters"), 
                threshold_init=self.config.get("step1_threshold")
            ),
            step2_clustering=BirchClustering(
                n=self.config.get("step2_n_clusters"), 
                threshold_init=self.config.get("step2_threshold")
            ),
            step2_selection=SelectKFromClusters(
                k=self.config.get("step2_k"), 
                selection_criteria=self.config.get("step2_selection"), 
                n_sites=[1]
            )
        )
        
        res = sampler.fit_transform(features, atom_features, atom_counts)
        # 2-DIRECT uses Step 1 PCA for visualization usually
        expl_var = sampler.step1_sampler.pca.pca.explained_variance_
        return res["selected_indices"], res["PCAfeatures"], expl_var
