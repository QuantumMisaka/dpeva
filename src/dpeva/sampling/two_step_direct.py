"""Two-step DIRECT sampling."""

from __future__ import annotations

import logging
import numpy as np

from .clustering import BirchClustering
from .direct import DIRECTSampler
from .stratified_sampling import SelectKFromClusters

logger = logging.getLogger(__name__)


class TwoStepDIRECTSampler:
    """
    Two-step DIRECT sampling strategy.
    
    Step 1: Partition structures into global clusters based on structural descriptors.
    Step 2: For each cluster, sample structures with the smallest number of atoms 
            based on atomic descriptors.
    """

    def __init__(
        self,
        step1_clustering=None,
        step2_clustering=None,
        step2_selection=None,
    ):
        """
        Initialize TwoStepDIRECTSampler.

        Args:
            step1_clustering: Clustering algorithm for Step 1 (Structural Partitioning).
                Defaults to BirchClustering(n=40, threshold_init=0.5).
            step2_clustering: Clustering algorithm for Step 2 (Atomic Feature Clustering).
                Defaults to BirchClustering(n=93, threshold_init=0.1).
            step2_selection: Selection strategy for Step 2.
                Defaults to SelectKFromClusters(k=5, selection_criteria="smallest").
                Note: n_sites will be populated dynamically during transform.
        """
        # Note: Defaults here are handled by Config or caller (CollectionWorkflow)
        # We provide sensible defaults just in case it's used standalone.
        # But for project consistency, we should rely on Config passed values.
        
        self.step1_clustering = (
            step1_clustering
            if step1_clustering is not None
            else BirchClustering(n=None, threshold_init=0.5)
        )
        self.step2_clustering = (
            step2_clustering
            if step2_clustering is not None
            else BirchClustering(n=None, threshold_init=0.1)
        )
        self.step2_selection = (
            step2_selection
            if step2_selection is not None
            else SelectKFromClusters(k=5, selection_criteria="smallest", n_sites=[1])
        )

        # Sampler for Step 1 (Structural)
        # We only use clustering from DIRECTSampler logic
        self.step1_sampler = DIRECTSampler(
            structure_encoder=None,
            clustering=self.step1_clustering,
            select_k_from_clusters=None, # No selection in step 1, just partition
        )

    def fit_transform(self, X_stru, X_atom_list, n_atoms_list):
        """
        Perform two-step sampling.

        Args:
            X_stru: (N_structures, D_stru) Structural descriptors (mean-pooled).
            X_atom_list: List of length N_structures. Each element is an array 
                         of shape (N_atoms_i, D_atom) containing atomic descriptors.
            n_atoms_list: List of length N_structures containing atom counts.

        Returns:
            dict: {
                "selected_indices": List of selected structure indices.
                "step1_labels": Cluster labels from Step 1.
                "PCAfeatures": PCA features from Step 1 (for visualization).
            }
        """
        logger.info("Starting Step 1: Structural Partitioning...")
        
        # Step 1: Structural Partitioning
        # DIRECTSampler.fit_transform returns dict with "labels" and "PCAfeatures"
        # Since select_k_from_clusters is None, it returns clustering data directly
        step1_result = self.step1_sampler.fit_transform(X_stru)
        step1_labels = step1_result["labels"]
        step1_pca_features = step1_result["PCAfeatures"]

        logger.info(f"Step 1 finished. Partitioned into {len(set(step1_labels))} clusters.")

        sampled_structure_ids = []
        unique_labels = sorted(list(set(step1_labels)))

        logger.info("Starting Step 2: Atomic Feature Sampling per cluster...")
        
        for label in unique_labels:
            # Indices of structures in this cluster
            structure_ids = np.where(step1_labels == label)[0]
            
            if len(structure_ids) == 0:
                continue

            # Gather atomic features for this cluster
            # We flatten all atoms from all structures in this cluster into one big array
            atom_features = []
            natoms_per_site = [] # Maps each atom back to the n_atoms of its parent structure
            structure_idx_per_site = [] # Maps each atom back to its parent structure index
            
            for struct_idx in structure_ids:
                # Append atomic features
                atom_features.append(X_atom_list[struct_idx])
                
                # Number of atoms in this structure
                n_atoms = n_atoms_list[struct_idx]
                
                # Append n_atoms for each atom in this structure
                natoms_per_site.extend([n_atoms] * n_atoms)
                
                # Append structure index for each atom
                structure_idx_per_site.extend([struct_idx] * n_atoms)
            
            # Concatenate all atomic features
            if len(atom_features) > 0:
                X_atom_cluster = np.concatenate(atom_features, axis=0)
            else:
                continue

            # Configure Step 2 Sampler
            # We need to pass n_sites (natoms_per_site) to the selection criteria
            # We clone the base configuration to avoid side effects
            current_selector = SelectKFromClusters(
                k=self.step2_selection.k,
                selection_criteria=self.step2_selection.selection_criteria,
                n_sites=natoms_per_site,
                allow_duplicate=self.step2_selection.allow_duplicate
            )
            
            # Use DIRECTSampler for Step 2 (Atomic)
            step2_sampler = DIRECTSampler(
                structure_encoder=None,
                clustering=self.step2_clustering,
                select_k_from_clusters=current_selector
            )
            
            # Run sampling
            # This returns indices into X_atom_cluster (i.e., which atoms were selected)
            step2_result = step2_sampler.fit_transform(X_atom_cluster)
            selected_atom_indices = step2_result["selected_indices"]
            
            # Map selected atoms back to structures
            # We want the set of unique structures that contain the selected atoms
            selected_structs = set()
            for atom_idx in selected_atom_indices:
                selected_structs.add(structure_idx_per_site[atom_idx])
            
            sampled_structure_ids.extend(list(selected_structs))
        
        # Remove duplicates globally (though set logic above handles per-cluster duplicates)
        sampled_structure_ids = sorted(list(set(sampled_structure_ids)))
        
        logger.info(f"Step 2 finished. Selected {len(sampled_structure_ids)} unique structures.")

        return {
            "selected_indices": sampled_structure_ids,
            "step1_labels": step1_labels,
            "PCAfeatures": step1_pca_features,
        }
