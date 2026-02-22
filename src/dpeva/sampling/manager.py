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
                        atom_counts: Optional[List[int]] = None,
                        background_features: Optional[np.ndarray] = None,
                        n_candidates: Optional[int] = None) -> Dict:
        """
        Runs the sampler.
        """
        if self.sampler_type == "2-direct":
            return self._run_2_direct(features, atom_features, atom_counts, background_features)
        else:
            return self._run_direct(features, background_features, n_candidates)

    def _calc_coverage(self, all_pca, selected_indices, n_bins=50):
        """Calculates grid-based coverage score for each PC dimension."""
        n_dims = all_pca.shape[1]
        scores = []
        for d in range(n_dims):
            vals_all = all_pca[:, d]
            vals_sel = all_pca[selected_indices, d]
            
            # Use fixed number of bins based on full range
            hist_all, edges = np.histogram(vals_all, bins=n_bins)
            hist_sel, _ = np.histogram(vals_sel, bins=edges)
            
            # Count occupied bins
            n_occ_all = np.sum(hist_all > 0)
            n_occ_sel = np.sum(hist_sel > 0)
            
            if n_occ_all == 0:
                scores.append(0.0)
            else:
                scores.append(n_occ_sel / n_occ_all)
        return np.array(scores)

    def _run_direct(self, features, background_features=None, n_candidates=None):
        self.logger.info("Running Standard DIRECT...")
        n_clusters = self.config.get("direct_n_clusters")
        
        sampler = DIRECTSampler(
            structure_encoder=None,
            clustering=BirchClustering(n=n_clusters, threshold_init=self.direct_thr_init),
            select_k_from_clusters=SelectKFromClusters(k=self.direct_k)
        )
        
        res = sampler.fit_transform(features, n_candidates=n_candidates)
        
        # Calculate scores and random baseline for visualization
        selected_indices = res["selected_indices"]
        pca_features = res["PCAfeatures"]
        
        # Calculate random baseline
        n_samples = len(selected_indices)
        n_total = len(features)
        if n_samples < n_total:
            random_indices = np.random.choice(n_total, n_samples, replace=False)
        else:
            random_indices = np.arange(n_total)
            
        # Calculate Coverage Scores
        scores_direct = self._calc_coverage(pca_features, selected_indices)
        scores_random = self._calc_coverage(pca_features, random_indices)
        
        # Transform background features if provided
        full_pca_features = None
        if background_features is not None:
            # DIRECTSampler.pca is the PCA step (usually a Pipeline or just PCA)
            # Inspecting dpeva.sampling.direct: DIRECTSampler.pca is the first step
            # Actually DIRECTSampler.fit_transform calls self.pca.fit_transform(X)
            # So self.pca is the fitted transformer
            full_pca_features = sampler.pca.transform(background_features)
        
        return {
            "selected_indices": selected_indices,
            "pca_features": pca_features,
            "explained_variance": sampler.pca.pca.explained_variance_,
            "random_indices": random_indices,
            "scores_direct": scores_direct,
            "scores_random": scores_random,
            "full_pca_features": full_pca_features
        }

    def _run_2_direct(self, features, atom_features, atom_counts, background_features=None):
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
        
        selected_indices = res["selected_indices"]
        pca_features = res["PCAfeatures"]
        
        # Random baseline
        n_samples = len(selected_indices)
        n_total = len(features)
        if n_samples < n_total:
            random_indices = np.random.choice(n_total, n_samples, replace=False)
        else:
            random_indices = np.arange(n_total)
            
        # Calculate Coverage Scores
        scores_direct = self._calc_coverage(pca_features, selected_indices)
        scores_random = self._calc_coverage(pca_features, random_indices)

        # Transform background features if provided
        # 2-DIRECT structure is complex, usually step1_sampler does PCA
        full_pca_features = None
        if background_features is not None:
             full_pca_features = sampler.step1_sampler.pca.transform(background_features)
            
        return {
            "selected_indices": selected_indices,
            "pca_features": pca_features,
            "explained_variance": expl_var,
            "random_indices": random_indices,
            "scores_direct": scores_direct,
            "scores_random": scores_random,
            "full_pca_features": full_pca_features
        }
