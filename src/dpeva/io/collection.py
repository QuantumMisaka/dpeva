import os
import glob
import logging
import shutil
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Set

from dpeva.io.dataset import load_systems

class CollectionIOManager:
    """
    Handles Data I/O for Collection Workflow:
    - Descriptor loading
    - Atomic feature loading
    - Result export (dpdata, csv)
    - Logging setup
    """
    
    def __init__(self, project_dir: str, root_savedir: str):
        self.project_dir = project_dir
        self.root_savedir = root_savedir
        self.logger = logging.getLogger(__name__)
        
        # Derived paths
        self.view_savedir = os.path.join(self.project_dir, self.root_savedir, "view")
        self.dpdata_savedir = os.path.join(self.project_dir, self.root_savedir, "dpdata")
        self.df_savedir = os.path.join(self.project_dir, self.root_savedir, "dataframe")
        
    def ensure_dirs(self):
        """Creates necessary output directories."""
        for d in [self.view_savedir, self.dpdata_savedir, self.df_savedir]:
            if not os.path.exists(d):
                os.makedirs(d)

    def count_frames(self, data_dir: str, fmt: str = "auto") -> int:
        """Counts total frames in dataset."""
        try:
            systems = load_systems(data_dir, fmt=fmt)
            return sum(len(sys) for sys in systems)
        except Exception as e:
            self.logger.warning(f"Failed to count frames in {data_dir}: {e}")
            return 0

    def load_descriptors(self, desc_dir: str, label: str = "descriptors", 
                        target_names: Optional[List[str]] = None, 
                        expected_frames: Optional[Dict[str, int]] = None) -> Tuple[List[str], np.ndarray]:
        """
        Load descriptors from a directory.

        Supports loading specific systems by name or all .npy files in the directory.

        Args:
            desc_dir (str): Directory containing descriptor files (.npy).
            label (str): Label for logging (e.g., "descriptors", "atomic features").
            target_names (Optional[List[str]]): List of specific system names to load.
            expected_frames (Optional[Dict[str, int]]): Dictionary mapping system names to expected frame counts for validation.

        Returns:
            Tuple[List[str], np.ndarray]:
                - List of datanames (e.g., "sys_name-0", "sys_name-1").
                - Concatenated array of structural descriptors (N_frames, N_features).
        
        Raises:
            FileNotFoundError: If a target system's descriptor file is missing.
        """
        self.logger.info(f"Loading {label} from {desc_dir}")
        
        desc_datanames = []
        desc_stru = []
        
        if target_names:
            self.logger.info(f"Loading {len(target_names)} specific systems based on target names.")
            for sys_name in target_names:
                path_flat = os.path.join(desc_dir, f"{sys_name}.npy")
                path_flat_base = os.path.join(desc_dir, f"{os.path.basename(sys_name)}.npy")
                
                if os.path.exists(path_flat):
                    f = path_flat
                elif os.path.exists(path_flat_base):
                    self.logger.info(f"Matched descriptor via basename: {path_flat_base}")
                    f = path_flat_base
                else:
                    self.logger.error(f"Descriptor file not found for system: {sys_name}")
                    raise FileNotFoundError(f"Descriptor file missing for {sys_name}")
                
                self._load_single_descriptor(f, sys_name, expected_frames, desc_datanames, desc_stru)
                
        else:
            # Glob loading
            pattern = desc_dir if '*' in desc_dir else os.path.join(desc_dir, "*.npy")
            desc_iter_list = sorted(glob.glob(pattern))
            
            if not desc_iter_list:
                self.logger.warning(f"No {label} found in {desc_dir}")
                return [], np.array([])
            
            for f in desc_iter_list:
                keyname = os.path.basename(f).replace('.npy', '')
                try:
                    self._load_single_descriptor(f, keyname, expected_frames, desc_datanames, desc_stru)
                except Exception as e:
                    self.logger.error(f"Failed to load {f}: {e}")
                    continue
        
        if len(desc_stru) > 0:
            return desc_datanames, np.concatenate(desc_stru, axis=0)
        return [], np.array([])

    def _load_single_descriptor(self, f_path: str, sys_name: str, expected_frames: Optional[Dict], 
                               desc_datanames: List, desc_stru: List):
        """
        Helper to load and process a single descriptor file.

        Loads .npy, validates frame count, computes structural descriptor (mean pooling),
        and normalizes it. Appends results to lists in-place.
        """
        one_desc = np.load(f_path)
        
        # Consistency Check
        if expected_frames and sys_name in expected_frames:
            n_exp = expected_frames[sys_name]
            n_got = one_desc.shape[0]
            if n_got != n_exp:
                if n_got > n_exp:
                    self.logger.warning(f"Descriptor mismatch for {sys_name}: Exp {n_exp}, Got {n_got}. Truncating.")
                    one_desc = one_desc[:n_exp]
                else:
                    raise ValueError(f"Missing frames for {sys_name}: Exp {n_exp}, Got {n_got}")

        # Add datanames
        desc_datanames.extend([f"{sys_name}-{i}" for i in range(len(one_desc))])
        
        # Normalize
        one_desc_stru = np.mean(one_desc, axis=1)
        stru_modulo = np.linalg.norm(one_desc_stru, axis=1, keepdims=True)
        desc_stru.append(one_desc_stru / (stru_modulo + 1e-12))

    def load_atomic_features(self, desc_dir: str, df_candidate: pd.DataFrame) -> Tuple[List[np.ndarray], List[int]]:
        """
        Load atomic features for 2-DIRECT sampling.

        Uses mmap to load only necessary frames from descriptor files.

        Args:
            desc_dir (str): Directory containing descriptor files.
            df_candidate (pd.DataFrame): DataFrame containing candidate systems/frames.

        Returns:
            Tuple[List[np.ndarray], List[int]]:
                - List of atomic feature arrays (N_atoms, N_features).
                - List of atom counts for each frame.
        
        Raises:
            ValueError: If atomic features are missing for a required system.
        """
        self.logger.info("Loading atomic features...")
        dataname_to_feat = {}
        dataname_to_natoms = {}
        
        candidate_datanames = set(df_candidate["dataname"])
        sys_to_frames = {}
        
        for dn in candidate_datanames:
            sys_name, idx = dn.rsplit("-", 1)
            sys_to_frames.setdefault(sys_name, set()).add(int(idx))
            
        for sys_name, indices in sys_to_frames.items():
            f_path = os.path.join(desc_dir, f"{sys_name}.npy")
            if not os.path.exists(f_path):
                f_path = os.path.join(desc_dir, f"{os.path.basename(sys_name)}.npy")
                
            if not os.path.exists(f_path):
                self.logger.warning(f"Missing atomic features for {sys_name}")
                continue
                
            try:
                # mmap load
                one_desc = np.load(f_path, mmap_mode='r')
                for idx in indices:
                    if idx < len(one_desc):
                        feat = np.array(one_desc[idx])
                        dn = f"{sys_name}-{idx}"
                        dataname_to_feat[dn] = feat
                        dataname_to_natoms[dn] = feat.shape[0]
            except Exception as e:
                self.logger.error(f"Error loading atomic features {sys_name}: {e}")
                
        X_list = []
        n_list = []
        for dn in df_candidate["dataname"]:
            if dn in dataname_to_feat:
                X_list.append(dataname_to_feat[dn])
                n_list.append(dataname_to_natoms[dn])
            else:
                raise ValueError(f"Missing atomic features for {dn}")
                
        return X_list, n_list

    def save_dataframe(self, df: pd.DataFrame, filename: str):
        """
        Save dataframe to CSV.

        Args:
            df (pd.DataFrame): The DataFrame to save.
            filename (str): The filename (relative to df_savedir).
        """
        path = os.path.join(self.df_savedir, filename)
        self.logger.info(f"Saving dataframe to {path}")
        df.to_csv(path, index=True)

    def export_dpdata(self, testdata_dir: str, df_final: pd.DataFrame, unique_system_names: List[str]):
        """
        Export sampled and remaining systems to dpdata format.

        Splits the original dataset into 'sampled_dpdata' and 'other_dpdata' based on selected indices.

        Args:
            testdata_dir (str): Directory containing original test data (dpdata format).
            df_final (pd.DataFrame): DataFrame containing selected frames.
            unique_system_names (List[str]): List of unique system names to process.
        """
        self.logger.info(f"Exporting dpdata from {testdata_dir}")
        
        # Map construction
        sampled_indices_map = {}
        for dataname in df_final['dataname']:
            sys_name, idx_str = dataname.rsplit('-', 1)
            try:
                sampled_indices_map.setdefault(sys_name, set()).add(int(idx_str))
            except ValueError:
                pass
                
        # Clean dirs
        for sub in ["sampled_dpdata", "other_dpdata"]:
            p = os.path.join(self.dpdata_savedir, sub)
            if os.path.exists(p): shutil.rmtree(p)
            os.makedirs(p)
            
        test_data = load_systems(testdata_dir, fmt="auto", target_systems=unique_system_names)
        
        count_sampled = 0
        count_other = 0
        
        for sys in test_data:
            sys_name = getattr(sys, "target_name", sys.short_name)
            sampled_set = sampled_indices_map.get(sys_name, set())
            n_frames = len(sys)
            
            valid_sampled = sorted([i for i in sampled_set if i < n_frames])
            other_indices = sorted(list(set(range(n_frames)) - set(valid_sampled)))
            
            if valid_sampled:
                try:
                    sys.sub_system(valid_sampled).to_deepmd_npy(
                        os.path.join(self.dpdata_savedir, "sampled_dpdata", sys_name)
                    )
                    count_sampled += 1
                except Exception as e:
                    self.logger.error(f"Failed to export sampled {sys_name}: {e}")
                    
            if other_indices:
                try:
                    sys.sub_system(other_indices).to_deepmd_npy(
                        os.path.join(self.dpdata_savedir, "other_dpdata", sys_name)
                    )
                    count_other += 1
                except Exception as e:
                    self.logger.error(f"Failed to export other {sys_name}: {e}")
                    
        self.logger.info(f"Exported {count_sampled} sampled systems, {count_other} other systems.")
