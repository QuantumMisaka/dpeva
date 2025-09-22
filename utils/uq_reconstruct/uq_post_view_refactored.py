"""
Refactored UQ Post-Processing and Visualization Module

This module provides an object-oriented approach to uncertainty quantification (UQ) 
post-processing and visualization for DeepMD models. It maintains all original 
functionality while organizing code into clear, reusable classes.

Author: Refactored from original uq-post-view.py
"""

import logging
import os
import shutil
import glob
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from copy import deepcopy

import dpdata
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from dpeva.sampling.direct import BirchClustering, DIRECTSampler, SelectKFromClusters
from dpeva.io.dataproc import DPTestResults


@dataclass
class UQConfig:
    """
    Configuration class for UQ post-processing parameters.
    
    This class encapsulates all configuration parameters used in the UQ workflow,
    providing a clean interface for parameter management and validation.
    
    Attributes:
        project (str): Project name/directory
        uq_select_scheme (str): UQ selection scheme (strict, circle_lo, tangent_lo, crossline_lo, loose)
        testing_dir (str): Testing directory name
        testing_head (str): Testing results file prefix
        desc_dir (str): Descriptor directory path
        desc_filename (str): Descriptor file name
        testdata_dir (str): Test data directory path
        testdata_fmt (str): Test data format
        testdata_string (str): Test data pattern string
        kde_bw_adjust (float): KDE bandwidth adjustment factor
        fig_dpi (int): Figure DPI for saved plots
        root_savedir (str): Root save directory name
        uq_qbc_trust_lo (float): QbC trust lower threshold
        uq_qbc_trust_hi (float): QbC trust upper threshold
        uq_rnd_rescaled_trust_lo (float): RND rescaled trust lower threshold
        uq_rnd_rescaled_trust_hi (float): RND rescaled trust upper threshold
        num_selection (int): Number of structures to select
        direct_k (int): DIRECT sampling k parameter
        direct_thr_init (float): DIRECT threshold initialization
    """
    # Project settings
    project: str = "stage9-2"
    uq_select_scheme: str = "tangent_lo"  # strict, circle_lo, tangent_lo, crossline_lo, loose
    testing_dir: str = "test-val-npy"
    testing_head: str = "results"
    
    # Descriptor loading settings
    desc_dir: str = None
    desc_filename: str = "desc.npy"
    
    # Test data settings
    testdata_dir: str = None
    testdata_fmt: str = "deepmd/npy"
    testdata_string: str = "O*"  # for correspondence
    
    # Figure settings
    kde_bw_adjust: float = 0.5
    fig_dpi: int = 150
    
    # Save settings
    root_savedir: str = "dpeva_uq_post"
    
    # Selection settings
    uq_qbc_trust_lo: float = 0.12
    uq_qbc_trust_hi: float = 0.22
    uq_rnd_rescaled_trust_lo: float = 0.12
    uq_rnd_rescaled_trust_hi: float = 0.22
    num_selection: int = 100
    direct_k: int = 1
    direct_thr_init: float = 0.5
    
    def __post_init__(self):
        """Initialize derived paths and validate configuration."""
        if self.desc_dir is None:
            self.desc_dir = f"{self.project}/desc_other"
        if self.testdata_dir is None:
            self.testdata_dir = f"{self.project}/other_dpdata"
            
        self.view_savedir = f"./{self.project}/{self.root_savedir}/view"
        self.dpdata_savedir = f"./{self.project}/{self.root_savedir}/dpdata"
        self.df_savedir = f"./{self.project}/{self.root_savedir}/dataframe"
        
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        uq_options = ["strict", "circle_lo", "crossline_lo", "tangent_lo", "loose"]
        if self.uq_select_scheme not in uq_options:
            raise ValueError(f"UQ selection scheme {self.uq_select_scheme} not supported! Please choose from {uq_options}.")
        
        if ((self.uq_qbc_trust_lo >= self.uq_qbc_trust_hi) 
            or (self.uq_rnd_rescaled_trust_lo >= self.uq_rnd_rescaled_trust_hi)):
            raise ValueError("Low trust threshold should be lower than High trust threshold!")


class DirectoryManager:
    """
    Manages directory creation and validation for UQ workflow.
    
    This class handles all directory-related operations including validation
    of required directories and creation of output directories.
    """
    
    def __init__(self, config: UQConfig):
        """
        Initialize DirectoryManager with configuration.
        
        Args:
            config (UQConfig): Configuration object containing directory paths
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def validate_directories(self):
        """
        Validate that all required input directories exist.
        
        Raises:
            ValueError: If any required directory is missing
        """
        self.logger.info(f"Initializing selection in {self.config.project} ---")
        
        required_dirs = [
            (self.config.project, "Project directory"),
            (f"{self.config.project}/0/{self.config.testing_dir}", "Testing directory"),
            (self.config.desc_dir, "Descriptor directory"),
            (self.config.testdata_dir, "Testdata directory")
        ]
        
        for dir_path, dir_name in required_dirs:
            if not os.path.exists(dir_path):
                self.logger.error(f"{dir_name} {dir_path} not found!")
                raise ValueError(f"{dir_name} {dir_path} not found!")
    
    def create_output_directories(self):
        """Create output directories if they don't exist."""
        output_dirs = [
            self.config.view_savedir,
            self.config.dpdata_savedir,
            self.config.df_savedir
        ]
        
        for dir_path in output_dirs:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)


class DataLoader:
    """
    Handles loading of test results, descriptors, and test data.
    
    This class encapsulates all data loading operations, providing a clean
    interface for accessing different types of input data.
    """
    
    def __init__(self, config: UQConfig):
        """
        Initialize DataLoader with configuration.
        
        Args:
            config (UQConfig): Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def load_test_results(self) -> Dict[int, DPTestResults]:
        """
        Load DeepMD test results from multiple models.
        
        Returns:
            Dict[int, DPTestResults]: Dictionary mapping model index to test results
        """
        self.logger.info("Loading the test results")
        
        test_results = {}
        for i in range(4):  # Load results from models 0, 1, 2, 3
            result_path = f"./{self.config.project}/{i}/{self.config.testing_dir}/{self.config.testing_head}"
            test_results[i] = DPTestResults(result_path)
        
        return test_results
    
    def load_test_data(self) -> dpdata.MultiSystems:
        """
        Load target testing data.
        
        Returns:
            dpdata.MultiSystems: Loaded test data
        """
        self.logger.info(f"Loading the target testing data from {self.config.testdata_dir}")
        
        return dpdata.MultiSystems.from_dir(
            self.config.testdata_dir,
            self.config.testdata_string,
            fmt=self.config.testdata_fmt
        )
    
    def load_descriptors(self) -> Tuple[List[str], np.ndarray]:
        """
        Load descriptors from descriptor files.
        
        Returns:
            Tuple[List[str], np.ndarray]: Descriptor names and structure descriptors
        """
        self.logger.info(f"Loading the target descriptors from {self.config.testdata_dir}")
        
        desc_string_test = f'{self.config.desc_dir}/*/{self.config.desc_filename}'
        desc_datanames = []
        desc_stru = []
        desc_iter_list = sorted(glob.glob(desc_string_test))
        
        for f in desc_iter_list:
            # Extract dirname of desc.npy from descriptors/*
            directory, _ = os.path.split(f)
            _, keyname = os.path.split(directory)
            one_desc = np.load(f)  # nframe, natoms, ndesc
            
            for i in range(len(one_desc)):
                desc_dataname = f"{keyname}-{i}"
                desc_datanames.append(desc_dataname)
                # Mean the atomic descriptors to structure descriptors
                one_desc_stru = np.mean(one_desc[i], axis=0).reshape(1, -1)
                desc_stru.append(one_desc_stru)
        
        desc_stru = np.concatenate(desc_stru, axis=0)
        return desc_datanames, desc_stru


class UQCalculator:
    """
    Calculates uncertainty quantification metrics from test results.
    
    This class implements the DPGEN formula for calculating atomic force
    uncertainties using both QbC (Query by Committee) and RND-like methods.
    """
    
    def __init__(self, config: UQConfig):
        """
        Initialize UQCalculator with configuration.
        
        Args:
            config (UQConfig): Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def calculate_force_differences(self, test_results: Dict[int, DPTestResults]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate force differences between model 0 prediction and existing label.
        
        Args:
            test_results (Dict[int, DPTestResults]): Test results from multiple models
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Max and RMS force differences per frame
        """
        self.logger.info("Dealing with force difference between 0 head prediction and existing label")
        
        dp_test_results_0 = test_results[0]
        diff_f_0 = np.sqrt(dp_test_results_0.diff_fx**2 + dp_test_results_0.diff_fy**2 + dp_test_results_0.diff_fz**2)
        
        # Map diff_f_0 to each structure with max force diff
        index = 0
        diff_maxf_0_frame = []
        diff_rmsf_0_frame = []
        
        for item in dp_test_results_0.dataname_list:
            natom = item[2]
            diff_f_0_item = diff_f_0[index:index + natom]
            diff_maxf_0_frame.append(np.max(diff_f_0_item))
            diff_rmsf_0_frame.append(np.sqrt(np.mean(diff_f_0_item**2)))
            index += natom
        
        return np.array(diff_maxf_0_frame), np.array(diff_rmsf_0_frame)
    
    def extract_atomic_forces(self, test_results: Dict[int, DPTestResults]) -> Dict[str, np.ndarray]:
        """
        Extract atomic forces from test results.
        
        Args:
            test_results (Dict[int, DPTestResults]): Test results from multiple models
            
        Returns:
            Dict[str, np.ndarray]: Dictionary containing force components and expected values
        """
        self.logger.info("Dealing with atomic force and average 1, 2, 3")
        
        forces = {}
        for i in range(4):
            forces[f'fx_{i}'] = test_results[i].data_f['pred_fx']
            forces[f'fy_{i}'] = test_results[i].data_f['pred_fy']
            forces[f'fz_{i}'] = test_results[i].data_f['pred_fz']
        
        # Calculate expected values (average of models 1, 2, 3)
        forces['fx_expt'] = np.mean((forces['fx_1'], forces['fx_2'], forces['fx_3']), axis=0)
        forces['fy_expt'] = np.mean((forces['fy_1'], forces['fy_2'], forces['fy_3']), axis=0)
        forces['fz_expt'] = np.mean((forces['fz_1'], forces['fz_2'], forces['fz_3']), axis=0)
        
        return forces
    
    def calculate_qbc_uq(self, forces: Dict[str, np.ndarray], test_results: Dict[int, DPTestResults]) -> np.ndarray:
        """
        Calculate QbC (Query by Committee) uncertainty quantification.
        
        Args:
            forces (Dict[str, np.ndarray]): Force components dictionary
            test_results (Dict[int, DPTestResults]): Test results from multiple models
            
        Returns:
            np.ndarray: QbC UQ values per structure
        """
        self.logger.info("Dealing with QbC force UQ")
        
        # Calculate QbC force UQ using DPGEN formula
        fx_qbc_square_diff = np.mean(((forces['fx_1'] - forces['fx_expt'])**2, 
                                     (forces['fx_2'] - forces['fx_expt'])**2, 
                                     (forces['fx_3'] - forces['fx_expt'])**2), axis=0)
        fy_qbc_square_diff = np.mean(((forces['fy_1'] - forces['fy_expt'])**2, 
                                     (forces['fy_2'] - forces['fy_expt'])**2, 
                                     (forces['fy_3'] - forces['fy_expt'])**2), axis=0)
        fz_qbc_square_diff = np.mean(((forces['fz_1'] - forces['fz_expt'])**2, 
                                     (forces['fz_2'] - forces['fz_expt'])**2, 
                                     (forces['fz_3'] - forces['fz_expt'])**2), axis=0)
        
        f_qbc_stddiff = np.sqrt(fx_qbc_square_diff + fy_qbc_square_diff + fz_qbc_square_diff)
        
        # Assign atomic force stddiff to each structure and get UQ by max atomic force diff
        index = 0
        uq_qbc_for_list = []
        
        for item in test_results[0].dataname_list:
            natom = item[2]
            f_qbc_stddiff_item = f_qbc_stddiff[index:index + natom]
            uq_qbc_for_list.append(np.max(f_qbc_stddiff_item))
            index += natom
        
        return np.array(uq_qbc_for_list)
    
    def calculate_rnd_uq(self, forces: Dict[str, np.ndarray], test_results: Dict[int, DPTestResults]) -> np.ndarray:
        """
        Calculate RND-like uncertainty quantification.
        
        Args:
            forces (Dict[str, np.ndarray]): Force components dictionary
            test_results (Dict[int, DPTestResults]): Test results from multiple models
            
        Returns:
            np.ndarray: RND UQ values per structure
        """
        self.logger.info("Dealing with RND-like force UQ")
        
        # Calculate RND-like force UQ
        fx_rnd_square_diff = np.mean(((forces['fx_1'] - forces['fx_0'])**2, 
                                     (forces['fx_2'] - forces['fx_0'])**2, 
                                     (forces['fx_3'] - forces['fx_0'])**2), axis=0)
        fy_rnd_square_diff = np.mean(((forces['fy_1'] - forces['fy_0'])**2, 
                                     (forces['fy_2'] - forces['fy_0'])**2, 
                                     (forces['fy_3'] - forces['fy_0'])**2), axis=0)
        fz_rnd_square_diff = np.mean(((forces['fz_1'] - forces['fz_0'])**2, 
                                     (forces['fz_2'] - forces['fz_0'])**2, 
                                     (forces['fz_3'] - forces['fz_0'])**2), axis=0)
        
        f_rnd_stddiff = np.sqrt(fx_rnd_square_diff + fy_rnd_square_diff + fz_rnd_square_diff)
        
        # Assign atomic force stddiff to each structure and get UQ by max atomic force diff
        index = 0
        uq_rnd_for_list = []
        
        for item in test_results[0].dataname_list:
            natom = item[2]
            f_rnd_stddiff_item = f_rnd_stddiff[index:index + natom]
            uq_rnd_for_list.append(np.max(f_rnd_stddiff_item))
            index += natom
        
        return np.array(uq_rnd_for_list)
    
    def rescale_rnd_to_qbc(self, uq_qbc_for: np.ndarray, uq_rnd_for: np.ndarray) -> np.ndarray:
        """
        Align RND to QbC by Z-Score normalization.
        
        Args:
            uq_qbc_for (np.ndarray): QbC UQ values
            uq_rnd_for (np.ndarray): RND UQ values
            
        Returns:
            np.ndarray: Rescaled RND UQ values
        """
        self.logger.info("Aligning UQ-RND to UQ-QbC by Z-Score")
        
        scaler_qbc_for = StandardScaler()
        scaler_rnd_for = StandardScaler()
        
        uq_qbc_for_scaled = scaler_qbc_for.fit_transform(uq_qbc_for.reshape(-1, 1)).flatten()
        uq_rnd_for_scaled = scaler_rnd_for.fit_transform(uq_rnd_for.reshape(-1, 1)).flatten()
        uq_rnd_for_rescaled = scaler_qbc_for.inverse_transform(uq_rnd_for_scaled.reshape(-1, 1)).flatten()
        
        return uq_rnd_for_rescaled


class UQSelector:
    """
    Handles UQ-based structure selection using different selection schemes.
    
    This class implements various selection strategies for identifying
    candidate, accurate, and failed structures based on UQ thresholds.
    """
    
    def __init__(self, config: UQConfig):
        """
        Initialize UQSelector with configuration.
        
        Args:
            config (UQConfig): Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def select_structures(self, df_uq_desc: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Select structures based on UQ selection scheme.
        
        Args:
            df_uq_desc (pd.DataFrame): DataFrame containing UQ values and descriptors
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Candidate, accurate, and failed structures
        """
        uq_x = df_uq_desc["uq_qbc_for"]
        uq_y = df_uq_desc["uq_rnd_for_rescaled"]
        
        uq_x_lo = self.config.uq_qbc_trust_lo
        uq_y_lo = self.config.uq_rnd_rescaled_trust_lo
        uq_x_hi = self.config.uq_qbc_trust_hi
        uq_y_hi = self.config.uq_rnd_rescaled_trust_hi
        
        if self.config.uq_select_scheme == "strict":
            return self._strict_selection(df_uq_desc, uq_x, uq_y, uq_x_lo, uq_x_hi, uq_y_lo, uq_y_hi)
        elif self.config.uq_select_scheme == "circle_lo":
            return self._circle_lo_selection(df_uq_desc, uq_x, uq_y, uq_x_lo, uq_x_hi, uq_y_lo, uq_y_hi)
        elif self.config.uq_select_scheme == "tangent_lo":
            return self._tangent_lo_selection(df_uq_desc, uq_x, uq_y, uq_x_lo, uq_x_hi, uq_y_lo, uq_y_hi)
        elif self.config.uq_select_scheme == "crossline_lo":
            return self._crossline_lo_selection(df_uq_desc, uq_x, uq_y, uq_x_lo, uq_x_hi, uq_y_lo, uq_y_hi)
        elif self.config.uq_select_scheme == "loose":
            return self._loose_selection(df_uq_desc, uq_x, uq_y, uq_x_lo, uq_x_hi, uq_y_lo, uq_y_hi)
        else:
            raise ValueError(f"UQ selection scheme {self.config.uq_select_scheme} not supported!")
    
    def _strict_selection(self, df_uq_desc, uq_x, uq_y, uq_x_lo, uq_x_hi, uq_y_lo, uq_y_hi):
        """Strict selection: QbC and RND-like are both trustable."""
        df_uq = df_uq_desc[["dataname", "uq_qbc_for", "uq_rnd_for_rescaled", "uq_rnd_for", "diff_maxf_0_frame"]]
        
        df_uq_desc_candidate = df_uq_desc[
            (uq_x >= uq_x_lo) & (uq_x <= uq_x_hi) & (uq_y >= uq_y_lo) & (uq_y <= uq_y_hi)
        ]
        df_uq_accurate = df_uq[
            ((uq_x < uq_x_lo) & (uq_y < uq_y_hi)) | ((uq_x < uq_x_hi) & (uq_y < uq_y_lo))
        ]
        df_uq_failed = df_uq[
            (uq_x > uq_x_hi) | (uq_y > uq_y_hi)
        ]
        
        return df_uq_desc_candidate, df_uq_accurate, df_uq_failed
    
    def _circle_lo_selection(self, df_uq_desc, uq_x, uq_y, uq_x_lo, uq_x_hi, uq_y_lo, uq_y_hi):
        """Balance selection: QbC and RND-like are trustable in a circle balance way."""
        df_uq = df_uq_desc[["dataname", "uq_qbc_for", "uq_rnd_for_rescaled", "uq_rnd_for", "diff_maxf_0_frame"]]
        
        df_uq_desc_candidate = df_uq_desc[
            ((uq_x <= uq_x_hi) & (uq_y <= uq_y_hi)) &
            ((uq_x-uq_x_hi)**2 + (uq_y-uq_y_hi)**2 <= (uq_x_hi - uq_x_lo)**2 + (uq_y_hi - uq_y_lo)**2)
        ]
        df_uq_accurate = df_uq[
            ((uq_x-uq_x_hi)**2 + (uq_y-uq_y_hi)**2 > (uq_x_lo - uq_x_hi)**2 + (uq_y_lo - uq_y_hi)**2) & 
            ((uq_x < uq_x_hi) & (uq_y < uq_y_hi))
        ]
        df_uq_failed = df_uq[
            (uq_x > uq_x_hi) | (uq_y > uq_y_hi)
        ]
        
        return df_uq_desc_candidate, df_uq_accurate, df_uq_failed
    
    def _tangent_lo_selection(self, df_uq_desc, uq_x, uq_y, uq_x_lo, uq_x_hi, uq_y_lo, uq_y_hi):
        """Balance selection: QbC and RND-like are trustable in a tangent-circle balance way."""
        df_uq = df_uq_desc[["dataname", "uq_qbc_for", "uq_rnd_for_rescaled", "uq_rnd_for", "diff_maxf_0_frame"]]
        
        df_uq_desc_candidate = df_uq_desc[
            ((uq_x <= uq_x_hi) & (uq_y <= uq_y_hi)) &
            ((uq_x-uq_x_lo)*(uq_x_lo-uq_x_hi) + (uq_y-uq_y_lo)*(uq_y_lo-uq_y_hi) <= 0)
        ]
        df_uq_accurate = df_uq[
            ((uq_x-uq_x_lo)*(uq_x_lo-uq_x_hi) + (uq_y-uq_y_lo)*(uq_y_lo-uq_y_hi) > 0) & 
            ((uq_x < uq_x_hi) & (uq_y < uq_y_hi))
        ]
        df_uq_failed = df_uq[
            (uq_x > uq_x_hi) | (uq_y > uq_y_hi)
        ]
        
        return df_uq_desc_candidate, df_uq_accurate, df_uq_failed
    
    def _crossline_lo_selection(self, df_uq_desc, uq_x, uq_y, uq_x_lo, uq_x_hi, uq_y_lo, uq_y_hi):
        """Balance selection: QbC and RND-like are trustable in a crossline balance way."""
        df_uq = df_uq_desc[["dataname", "uq_qbc_for", "uq_rnd_for_rescaled", "uq_rnd_for", "diff_maxf_0_frame"]]
        
        df_uq_desc_candidate = df_uq_desc[
            (uq_x <= uq_x_hi) & (uq_y <= uq_y_hi) &
            (uq_x_lo * uq_y + (uq_y_hi - uq_y_lo) * uq_x >= uq_x_lo * uq_y_hi) &
            (uq_x * uq_y_lo + (uq_x_hi - uq_x_lo) * uq_y >= uq_x_hi * uq_y_lo)
        ]
        df_uq_accurate = df_uq[
            (uq_x_lo * uq_y + (uq_y_hi - uq_y_lo) * uq_x < uq_x_lo * uq_y_hi) |
            (uq_x * uq_y_lo + (uq_x_hi - uq_x_lo) * uq_y < uq_x_hi * uq_y_lo)
        ]
        df_uq_failed = df_uq[
            (uq_x > uq_x_hi) | (uq_y > uq_y_hi)
        ]
        
        return df_uq_desc_candidate, df_uq_accurate, df_uq_failed
    
    def _loose_selection(self, df_uq_desc, uq_x, uq_y, uq_x_lo, uq_x_hi, uq_y_lo, uq_y_hi):
        """Loose selection: QbC or RND-like is either trustable."""
        df_uq = df_uq_desc[["dataname", "uq_qbc_for", "uq_rnd_for_rescaled", "uq_rnd_for", "diff_maxf_0_frame"]]
        
        df_uq_desc_candidate = df_uq_desc[
            ((uq_x >= uq_x_lo) & (uq_x <= uq_x_hi)) | ((uq_y >= uq_y_lo) & (uq_y <= uq_y_hi))
        ]
        df_uq_accurate = df_uq[
            (uq_x < uq_x_lo) & (uq_y < uq_y_lo)
        ]
        df_uq_failed = df_uq[
            (uq_x > uq_x_hi) | (uq_y > uq_y_hi)
        ]
        
        return df_uq_desc_candidate, df_uq_accurate, df_uq_failed


class UQVisualizer:
    """
    Handles all visualization tasks for UQ analysis.
    
    This class provides methods for creating various plots and visualizations
    to analyze UQ distributions, correlations, and selection results.
    """
    
    def __init__(self, config: UQConfig):
        """
        Initialize UQVisualizer with configuration.
        
        Args:
            config (UQConfig): Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Set matplotlib parameters
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['font.size'] = 10
    
    def plot_uq_distributions(self, uq_qbc_for: np.ndarray, uq_rnd_for: np.ndarray, uq_rnd_for_rescaled: np.ndarray):
        """
        Plot UQ distributions using KDE plots.
        
        Args:
            uq_qbc_for (np.ndarray): QbC UQ values
            uq_rnd_for (np.ndarray): RND UQ values
            uq_rnd_for_rescaled (np.ndarray): Rescaled RND UQ values
        """
        self.logger.info("Plotting and saving the figures of UQ-force")
        
        # Original UQ distributions
        plt.figure(figsize=(8, 6))
        sns.kdeplot(uq_qbc_for, color="blue", label="UQ-QbC", bw_adjust=self.config.kde_bw_adjust)
        sns.kdeplot(uq_rnd_for, color="red", label="UQ-RND", bw_adjust=self.config.kde_bw_adjust)
        plt.title("Distribution of UQ-force by KDEplot")
        plt.xlabel("UQ Value")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"./{self.config.view_savedir}/UQ-force.png", dpi=self.config.fig_dpi)
        plt.close()
        
        # Rescaled UQ distributions
        self.logger.info("Plotting and saving the figures of UQ-force rescaled")
        plt.figure(figsize=(8, 6))
        sns.kdeplot(uq_qbc_for, color="blue", label="UQ-QbC", bw_adjust=self.config.kde_bw_adjust)
        sns.kdeplot(uq_rnd_for_rescaled, color="red", label="UQ-RND-rescaled", bw_adjust=self.config.kde_bw_adjust)
        plt.title("Distribution of UQ-force by KDEplot")
        plt.xlabel("UQ Value")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"./{self.config.view_savedir}/UQ-force-rescaled.png", dpi=self.config.fig_dpi)
        plt.close()
    
    def plot_uq_with_trust_ranges(self, uq_qbc_for: np.ndarray, uq_rnd_for_rescaled: np.ndarray):
        """
        Plot UQ distributions with trust range indicators.
        
        Args:
            uq_qbc_for (np.ndarray): QbC UQ values
            uq_rnd_for_rescaled (np.ndarray): Rescaled RND UQ values
        """
        uq_x_lo = self.config.uq_qbc_trust_lo
        uq_x_hi = self.config.uq_qbc_trust_hi
        
        # QbC with trust range
        self.logger.info("Plotting and saving the figures of UQ-QbC-force with UQ trust range")
        plt.figure(figsize=(8, 6))
        sns.kdeplot(uq_qbc_for, color="blue", bw_adjust=self.config.kde_bw_adjust)
        plt.title("Distribution of UQ-QbC-force by KDEplot")
        plt.xlabel("UQ-QbC Value")
        plt.ylabel("Density")
        plt.grid(True)
        plt.axvline(uq_x_lo, color='purple', linestyle='--', linewidth=1)
        plt.axvline(uq_x_hi, color='purple', linestyle='--', linewidth=1)
        plt.axvspan(np.min(uq_qbc_for), uq_x_lo, alpha=0.1, color='green')
        plt.axvspan(uq_x_lo, uq_x_hi, alpha=0.1, color='yellow')
        plt.axvspan(uq_x_hi, np.max(uq_qbc_for), alpha=0.1, color='red')
        plt.savefig(f"./{self.config.view_savedir}/UQ-QbC-force.png", dpi=self.config.fig_dpi)
        plt.close()
        
        # RND rescaled with trust range
        self.logger.info("Plotting and saving the figures of UQ-RND-force-rescaled with UQ trust range")
        plt.figure(figsize=(8, 6))
        sns.kdeplot(uq_rnd_for_rescaled, color="blue", bw_adjust=self.config.kde_bw_adjust)
        plt.title("Distribution of UQ-RND-force-rescaled by KDEplot")
        plt.xlabel("UQ-RND-rescaled Value")
        plt.ylabel("Density")
        plt.grid(True)
        plt.axvline(uq_x_lo, color='purple', linestyle='--', linewidth=1)
        plt.axvline(uq_x_hi, color='purple', linestyle='--', linewidth=1)
        plt.axvspan(np.min(uq_rnd_for_rescaled), uq_x_lo, alpha=0.1, color='green')
        plt.axvspan(uq_x_lo, uq_x_hi, alpha=0.1, color='yellow')
        plt.axvspan(uq_x_hi, np.max(uq_rnd_for_rescaled), alpha=0.1, color='red')
        plt.savefig(f"./{self.config.view_savedir}/UQ-RND-force-rescaled.png", dpi=self.config.fig_dpi)
        plt.close()
    
    def plot_uq_vs_force_diff(self, uq_qbc_for: np.ndarray, uq_rnd_for: np.ndarray, 
                             uq_rnd_for_rescaled: np.ndarray, diff_maxf_0_frame: np.ndarray):
        """
        Plot UQ values vs force differences.
        
        Args:
            uq_qbc_for (np.ndarray): QbC UQ values
            uq_rnd_for (np.ndarray): RND UQ values
            uq_rnd_for_rescaled (np.ndarray): Rescaled RND UQ values
            diff_maxf_0_frame (np.ndarray): Max force differences per frame
        """
        # Original UQ vs force diff
        self.logger.info("Plotting and saving the figures of UQ-force vs force diff")
        plt.figure(figsize=(8, 6))
        plt.scatter(uq_qbc_for, diff_maxf_0_frame, color="blue", label="QbC", s=20)
        plt.scatter(uq_rnd_for, diff_maxf_0_frame, color="red", label="RND", s=20)
        plt.title("UQ vs Force Diff")
        plt.xlabel("UQ Value")
        plt.ylabel("True Max Force Diff")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"./{self.config.view_savedir}/UQ-force-fdiff-parity.png", dpi=self.config.fig_dpi)
        plt.close()
        
        # Rescaled UQ vs force diff
        self.logger.info("Plotting and saving the figures of UQ-force-rescaled vs force diff")
        plt.figure(figsize=(8, 6))
        plt.scatter(uq_qbc_for, diff_maxf_0_frame, color="blue", label="QbC", s=20)
        plt.scatter(uq_rnd_for_rescaled, diff_maxf_0_frame, color="red", label="RND-rescaled", s=20)
        plt.title("UQ vs Force Diff")
        plt.xlabel("UQ Value")
        plt.ylabel("True Max Force Diff")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"./{self.config.view_savedir}/UQ-force-rescaled-fdiff-parity.png", dpi=self.config.fig_dpi)
        plt.close()
    
    def plot_uq_difference_analysis(self, uq_qbc_for: np.ndarray, uq_rnd_for_rescaled: np.ndarray, 
                                   diff_maxf_0_frame: np.ndarray):
        """
        Plot UQ difference analysis.
        
        Args:
            uq_qbc_for (np.ndarray): QbC UQ values
            uq_rnd_for_rescaled (np.ndarray): Rescaled RND UQ values
            diff_maxf_0_frame (np.ndarray): Max force differences per frame
        """
        uq_diff_for_scaled_to_qbc = np.abs(uq_rnd_for_rescaled - uq_qbc_for)
        
        # UQ diff vs UQ
        self.logger.info("Plotting and saving the figures of UQ-diff vs UQ")
        plt.figure(figsize=(8, 6))
        plt.scatter(uq_diff_for_scaled_to_qbc, uq_qbc_for, color="blue", label="UQ-qbc-for", s=20)
        plt.scatter(uq_diff_for_scaled_to_qbc, uq_rnd_for_rescaled, color="red", label="UQ-rnd-for-rescaled", s=20)
        plt.title("UQ-diff vs UQ")
        plt.xlabel("UQ-diff Value")
        plt.ylabel("UQ Value")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"./{self.config.view_savedir}/UQ-diff-UQ-parity.png", dpi=self.config.fig_dpi)
        plt.close()
        
        # UQ diff vs force diff
        self.logger.info("Plotting and saving the figures of UQ-diff vs force diff")
        plt.figure(figsize=(8, 6))
        plt.scatter(uq_diff_for_scaled_to_qbc, diff_maxf_0_frame, color="blue", label="UQ-diff-force", s=20)
        plt.title("UQ-diff vs Force Diff")
        plt.xlabel("UQ-diff Value")
        plt.ylabel("True Max Force Diff")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"./{self.config.view_savedir}/UQ-diff-fdiff-parity.png", dpi=self.config.fig_dpi)
        plt.close()
    
    def plot_uq_2d_scatter_with_selection(self, df_uq: pd.DataFrame):
        """
        Plot 2D scatter plot of UQ values with selection regions.
        
        Args:
            df_uq (pd.DataFrame): DataFrame containing UQ values and identities
        """
        self.logger.info("Plotting and saving the figures of UQ-qbc-force and UQ-rnd-force-rescaled vs force diff")
        
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df_uq, 
                        x="uq_qbc_for", 
                        y="uq_rnd_for_rescaled", 
                        hue="diff_maxf_0_frame", 
                        palette="Reds",
                        alpha=0.8,
                        s=60)
        plt.title("UQ-QbC and UQ-RND vs Max Force Diff", fontsize=14)
        plt.grid(True)
        plt.xlabel("UQ-QbC Value", fontsize=12)
        plt.ylabel("UQ-RND-rescaled Value", fontsize=12)
        plt.legend(title="Max Force Diff", fontsize=10)
        
        # Set ticks
        ax = plt.gca()
        x_major_locator = mtick.MultipleLocator(0.1)
        y_major_locator = mtick.MultipleLocator(0.1)
        ax.xaxis.set_major_locator(x_major_locator)
        ax.yaxis.set_major_locator(y_major_locator)
        
        # Plot selection boundaries
        self._plot_selection_boundaries(ax)
        
        plt.savefig(f"./{self.config.view_savedir}/UQ-force-qbc-rnd-fdiff-scatter.png", dpi=self.config.fig_dpi)
        plt.close()
    
    def _plot_selection_boundaries(self, ax):
        """Plot selection boundaries based on UQ selection scheme."""
        uq_x_lo = self.config.uq_qbc_trust_lo
        uq_y_lo = self.config.uq_rnd_rescaled_trust_lo
        uq_x_hi = self.config.uq_qbc_trust_hi
        uq_y_hi = self.config.uq_rnd_rescaled_trust_hi
        
        # Common boundaries
        plt.plot([0, uq_x_hi], [uq_y_hi, uq_y_hi], color='black', linestyle='--', linewidth=2)
        plt.plot([uq_x_hi, uq_x_hi], [0, uq_y_hi], color='black', linestyle='--', linewidth=2)
        
        if self.config.uq_select_scheme == "strict":
            plt.plot([uq_x_lo, uq_x_lo], [uq_y_lo, uq_y_hi], color='purple', linestyle='--', linewidth=2)
            plt.plot([uq_x_lo, uq_x_hi], [uq_y_lo, uq_y_lo], color='purple', linestyle='--', linewidth=2)
        elif self.config.uq_select_scheme == "circle_lo":
            center = (uq_x_hi, uq_y_hi)
            radius = np.sqrt((uq_x_lo - uq_x_hi)**2 + (uq_y_lo - uq_y_hi)**2)
            theta = np.linspace(np.pi, 1.5*np.pi, 100)
            x_val = center[0] + radius * np.cos(theta)
            y_val = center[1] + radius * np.sin(theta)
            plt.plot(x_val, y_val, color="purple", linestyle="--", linewidth=2)
        elif self.config.uq_select_scheme == "tangent_lo":
            x_val = np.linspace(0, uq_x_hi, 100)
            y_val = - (uq_y_hi - uq_y_lo) / (uq_x_hi - uq_x_lo) * (x_val - uq_x_lo) + uq_y_lo
            x_val = x_val[y_val < uq_y_hi]
            y_val = y_val[y_val < uq_y_hi]
            plt.plot(x_val, y_val, color="purple", linestyle="--", linewidth=2)
        elif self.config.uq_select_scheme == "crossline_lo":
            x_val, y_val = self._balance_linear_func(uq_x_lo, uq_x_hi, uq_y_lo, uq_y_hi, (0, uq_x_hi), 100)
            x_val = x_val[y_val < uq_y_hi]
            y_val = y_val[y_val < uq_y_hi]
            plt.plot(x_val, y_val, color="purple", linestyle="--", linewidth=2)
        elif self.config.uq_select_scheme == "loose":
            plt.plot([uq_x_lo, uq_x_lo], [0, uq_y_lo], color='purple', linestyle='--', linewidth=2)
            plt.plot([0, uq_x_lo], [uq_y_lo, uq_y_lo], color='purple', linestyle='--', linewidth=2)
    
    def _balance_linear_func(self, x_lo, x_hi, y_lo, y_hi, x_range=(0, 10), num_points=20):
        """Helper function for crossline selection boundary."""
        x_val = np.linspace(x_range[0], x_range[1], num_points)
        delta_y = y_hi - y_lo
        delta_x = x_hi - x_lo
        y1 = (y_hi * x_lo - delta_y * x_val) / x_lo
        y2 = (y_lo * x_hi - y_lo * x_val) / delta_x
        y = np.max((y1, y2), axis=0)
        return x_val, y
    
    def plot_selection_identity_scatter(self, df_uq: pd.DataFrame):
        """
        Plot UQ identity scatter plot.
        
        Args:
            df_uq (pd.DataFrame): DataFrame containing UQ values and identities
        """
        self.logger.info("Plotting and saving the figure of UQ-identity in QbC-RND 2D space")
        
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df_uq, 
                        x="uq_qbc_for", 
                        y="uq_rnd_for_rescaled", 
                        hue="uq_identity", 
                        palette={"candidate": "orange", "accurate": "green", "failed": "red"},
                        alpha=0.5,
                        s=60)
        plt.title("UQ QbC+RND Selection View", fontsize=14)
        plt.grid(True)
        plt.xlabel("UQ-QbC Value", fontsize=12)
        plt.ylabel("UQ-RND-rescaled Value", fontsize=12)
        plt.legend(title="Identity", fontsize=10)
        
        # Set ticks
        ax = plt.gca()
        x_major_locator = mtick.MultipleLocator(0.1)
        y_major_locator = mtick.MultipleLocator(0.1)
        ax.xaxis.set_major_locator(x_major_locator)
        ax.yaxis.set_major_locator(y_major_locator)
        
        # Plot selection boundaries
        self._plot_selection_boundaries(ax)
        
        plt.savefig(f"./{self.config.view_savedir}/UQ-force-qbc-rnd-identity-scatter.png", dpi=self.config.fig_dpi)
        plt.close()
    
    def plot_candidate_analysis(self, uq_qbc_for: np.ndarray, uq_rnd_for_rescaled: np.ndarray,
                               diff_maxf_0_frame: np.ndarray, df_uq_desc_candidate: pd.DataFrame):
        """
        Plot candidate structure analysis.
        
        Args:
            uq_qbc_for (np.ndarray): QbC UQ values
            uq_rnd_for_rescaled (np.ndarray): Rescaled RND UQ values
            diff_maxf_0_frame (np.ndarray): Max force differences per frame
            df_uq_desc_candidate (pd.DataFrame): Candidate structures DataFrame
        """
        # QbC candidate analysis
        self.logger.info("Plotting and saving the figure of UQ-Candidate in QbC space against Max Force Diff")
        plt.figure(figsize=(8, 6))
        plt.scatter(uq_qbc_for, diff_maxf_0_frame, color="blue", label="UQ-QbC", s=20)
        plt.scatter(df_uq_desc_candidate["uq_qbc_for"], df_uq_desc_candidate["diff_maxf_0_frame"], 
                   color="orange", label="Candidate", s=20)
        plt.title("UQ vs Force Diff")
        plt.xlabel("UQ Value")
        plt.ylabel("True Max Force Diff")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"./{self.config.view_savedir}/UQ-QbC-Candidate-fdiff-parity.png", dpi=self.config.fig_dpi)
        plt.close()
        
        # RND candidate analysis
        self.logger.info("Plotting and saving the figure of UQ-Candidate in RND-rescaled space against Max Force Diff")
        plt.figure(figsize=(8, 6))
        plt.scatter(uq_rnd_for_rescaled, diff_maxf_0_frame, color="blue", label="UQ-RND-rescaled", s=20)
        plt.scatter(df_uq_desc_candidate["uq_rnd_for_rescaled"], df_uq_desc_candidate["diff_maxf_0_frame"], 
                   color="orange", label="Candidate", s=20)
        plt.title("UQ vs Force Diff")
        plt.xlabel("UQ Value")
        plt.ylabel("True Max Force Diff")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"./{self.config.view_savedir}/UQ-RND-Candidate-fdiff-parity.png", dpi=self.config.fig_dpi)
        plt.close()


class DIRECTProcessor:
    """
    Handles DIRECT sampling and related analysis.
    
    This class manages the DIRECT sampling process, PCA analysis,
    and coverage score calculations.
    """
    
    def __init__(self, config: UQConfig):
        """
        Initialize DIRECTProcessor with configuration.
        
        Args:
            config (UQConfig): Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def perform_direct_sampling(self, df_uq_desc_candidate: pd.DataFrame, desc_stru: np.ndarray) -> Dict:
        """
        Perform DIRECT sampling on candidate structures.
        
        Args:
            df_uq_desc_candidate (pd.DataFrame): Candidate structures DataFrame
            desc_stru (np.ndarray): Structure descriptors
            
        Returns:
            Dict: DIRECT sampling results
        """
        self.logger.info(f"Doing DIRECT Selection on UQ-selected data")
        
        DIRECT_sampler = DIRECTSampler(
            structure_encoder=None,
            clustering=BirchClustering(
                n=self.config.num_selection // self.config.direct_k, 
                threshold_init=self.config.direct_thr_init
            ),
            select_k_from_clusters=SelectKFromClusters(k=self.config.direct_k),
        )
        
        desc_features = [f"desc_stru_{i}" for i in range(desc_stru.shape[1])]
        DIRECT_selection = DIRECT_sampler.fit_transform(df_uq_desc_candidate[desc_features].values)
        
        explained_variance = DIRECT_sampler.pca.pca.explained_variance_
        selected_PC_dim = len([e for e in explained_variance if e > 1])
        DIRECT_selection["PCAfeatures_unweighted"] = DIRECT_selection["PCAfeatures"] / explained_variance[:selected_PC_dim]
        
        return {
            'DIRECT_selection': DIRECT_selection,
            'DIRECT_sampler': DIRECT_sampler,
            'explained_variance': explained_variance,
            'selected_PC_dim': selected_PC_dim
        }
    
    def visualize_direct_results(self, direct_results: Dict):
        """
        Visualize DIRECT sampling results.
        
        Args:
            direct_results (Dict): DIRECT sampling results
        """
        explained_variance = direct_results['explained_variance']
        selected_PC_dim = direct_results['selected_PC_dim']
        DIRECT_selection = direct_results['DIRECT_selection']
        
        # Explained variance plot
        self.logger.info(f"Visualization of DIRECT results compared with Random")
        plt.figure(figsize=(8, 6))
        plt.plot(
            range(1, selected_PC_dim+6+1),
            explained_variance[:selected_PC_dim+6],
            "o-",
        )
        plt.xlabel(r"i$^{\mathrm{th}}$ PC", size=12)
        plt.ylabel("Explained variance", size=12)
        ax = plt.gca()
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        plt.savefig(f"{self.config.view_savedir}/explained_variance.png", dpi=150)
        plt.close()
        
        # PCA feature coverage plots
        all_features = DIRECT_selection["PCAfeatures_unweighted"]
        DIRECT_selected_indices = DIRECT_selection["selected_indices"]
        
        self._plot_PCAfeature_coverage(all_features, DIRECT_selected_indices, "DIRECT")
        
        # Random selection for comparison
        np.random.seed(42)
        manual_selection_index = np.random.choice(len(all_features), self.config.num_selection, replace=False)
        self._plot_PCAfeature_coverage(all_features, manual_selection_index, "Random")
        
        # Coverage score comparison
        self._plot_coverage_scores(all_features, DIRECT_selected_indices, manual_selection_index)
    
    def _plot_PCAfeature_coverage(self, all_features: np.ndarray, selected_indices: np.ndarray, method: str):
        """Plot PCA feature coverage."""
        plt.figure(figsize=(8, 6))
        selected_features = all_features[selected_indices]
        plt.plot(all_features[:, 0], all_features[:, 1], "*", alpha=0.6, label=f"All {len(all_features):,} structures")
        plt.plot(
            selected_features[:, 0],
            selected_features[:, 1],
            "*",
            alpha=0.6,
            label=f"{method} sampled {len(selected_features):,}",
        )
        plt.legend(frameon=False, fontsize=10, reverse=True)
        plt.ylabel("PC 2", size=12)
        plt.xlabel("PC 1", size=12)
        plt.savefig(f"{self.config.view_savedir}/{method}_PCA_feature_coverage.png", dpi=self.config.fig_dpi)
        plt.close()
    
    def _calculate_feature_coverage_score(self, all_features: np.ndarray, selected_indices: np.ndarray, n_bins: int = 100) -> float:
        """Calculate feature coverage score."""
        selected_features = all_features[selected_indices]
        n_all = np.count_nonzero(
            np.histogram(all_features, bins=np.linspace(min(all_features), max(all_features), n_bins))[0]
        )
        n_select = np.count_nonzero(
            np.histogram(selected_features, bins=np.linspace(min(all_features), max(all_features), n_bins))[0]
        )
        return n_select / n_all
    
    def _calculate_all_FCS(self, all_features: np.ndarray, selected_indices: np.ndarray, b_bins: int = 100) -> List[float]:
        """Calculate all feature coverage scores."""
        select_scores = [
            self._calculate_feature_coverage_score(all_features[:, i], selected_indices, n_bins=b_bins)
            for i in range(all_features.shape[1])
        ]
        return select_scores
    
    def _plot_coverage_scores(self, all_features: np.ndarray, DIRECT_selected_indices: np.ndarray, 
                             manual_selection_index: np.ndarray):
        """Plot coverage score comparison."""
        scores_DIRECT = self._calculate_all_FCS(all_features, DIRECT_selected_indices, b_bins=100)
        scores_MS = self._calculate_all_FCS(all_features, manual_selection_index, b_bins=100)
        
        x = np.arange(len(scores_DIRECT))
        x_ticks = [f"PC {n+1}" for n in range(len(x))]
        plt.figure(figsize=(15, 4))
        plt.bar(
            x + 0.6,
            scores_DIRECT,
            width=0.3,
            label=rf"DIRECT, $\overline{{\mathrm{{Coverage\ score}}}}$ = {np.mean(scores_DIRECT):.3f}",
        )
        plt.bar(
            x + 0.3,
            scores_MS,
            width=0.3,
            label=rf"Random, $\overline{{\mathrm{{Coverage\ score}}}}$ = {np.mean(scores_MS):.3f}",
        )
        plt.xticks(x + 0.45, x_ticks, size=12)
        plt.yticks(np.linspace(0, 1.0, 6), size=12)
        plt.ylabel("Coverage score", size=12)
        plt.legend(shadow=True, loc="lower right", fontsize=12)
        plt.savefig(f"{self.config.view_savedir}/coverage_score.png", dpi=self.config.fig_dpi)
        plt.close()
    
    def visualize_final_selection_pca(self, df_uq_desc: pd.DataFrame, df_uq_desc_candidate: pd.DataFrame,
                                     df_uq_desc_selected_final: pd.DataFrame, desc_stru: np.ndarray):
        """
        Visualize final selection results in PCA space.
        
        Args:
            df_uq_desc (pd.DataFrame): All structures DataFrame
            df_uq_desc_candidate (pd.DataFrame): Candidate structures DataFrame
            df_uq_desc_selected_final (pd.DataFrame): Final selected structures DataFrame
            desc_stru (np.ndarray): Structure descriptors
        """
        self.logger.info("Visualization of final selection results in PCA space")
        
        # Perform PCA on all descriptors
        desc_features = [f"desc_stru_{i}" for i in range(desc_stru.shape[1])]
        pca = PCA(n_components=2)
        all_pca_features = pca.fit_transform(df_uq_desc[desc_features].values)
        candidate_pca_features = pca.transform(df_uq_desc_candidate[desc_features].values)
        selected_pca_features = pca.transform(df_uq_desc_selected_final[desc_features].values)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(all_pca_features[:, 0], all_pca_features[:, 1], 
                   c='lightgray', alpha=0.5, s=20, label=f'All structures ({len(df_uq_desc)})')
        plt.scatter(candidate_pca_features[:, 0], candidate_pca_features[:, 1], 
                   c='orange', alpha=0.7, s=30, label=f'Candidates ({len(df_uq_desc_candidate)})')
        plt.scatter(selected_pca_features[:, 0], selected_pca_features[:, 1], 
                   c='red', alpha=0.9, s=50, label=f'Final selected ({len(df_uq_desc_selected_final)})')
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.title('Final Selection Results in PCA Space')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{self.config.view_savedir}/final_selection_PCA.png", dpi=self.config.fig_dpi)
        plt.close()


class DataProcessor:
    """
    Handles data processing, sampling, and export operations.
    
    This class manages the final data processing steps including
    dpdata sampling and export of selected structures.
    """
    
    def __init__(self, config: UQConfig):
        """
        Initialize DataProcessor with configuration.
        
        Args:
            config (UQConfig): Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def save_dataframes(self, df_uq_desc: pd.DataFrame, df_uq_desc_candidate: pd.DataFrame,
                       df_uq_accurate: pd.DataFrame, df_uq_failed: pd.DataFrame,
                       df_uq_desc_selected_final: pd.DataFrame):
        """
        Save all DataFrames to files.
        
        Args:
            df_uq_desc (pd.DataFrame): All structures DataFrame
            df_uq_desc_candidate (pd.DataFrame): Candidate structures DataFrame
            df_uq_accurate (pd.DataFrame): Accurate structures DataFrame
            df_uq_failed (pd.DataFrame): Failed structures DataFrame
            df_uq_desc_selected_final (pd.DataFrame): Final selected structures DataFrame
        """
        self.logger.info("Saving DataFrames")
        
        dataframes = {
            'df_uq_desc.pkl': df_uq_desc,
            'df_uq_desc_candidate.pkl': df_uq_desc_candidate,
            'df_uq_accurate.pkl': df_uq_accurate,
            'df_uq_failed.pkl': df_uq_failed,
            'df_uq_desc_selected_final.pkl': df_uq_desc_selected_final
        }
        
        for filename, df in dataframes.items():
            filepath = f"{self.config.df_savedir}/{filename}"
            df.to_pickle(filepath)
            self.logger.info(f"Saved {filename} with shape {df.shape}")
    
    def perform_dpdata_sampling(self, df_uq_desc_selected_final: pd.DataFrame, 
                               testdata: dpdata.MultiSystems) -> dpdata.MultiSystems:
        """
        Perform dpdata sampling based on selected structures.
        
        Args:
            df_uq_desc_selected_final (pd.DataFrame): Final selected structures DataFrame
            testdata (dpdata.MultiSystems): Test data MultiSystems
            
        Returns:
            dpdata.MultiSystems: Sampled MultiSystems
        """
        self.logger.info("Doing dpdata sampling")
        
        # Create correspondence between datanames and testdata
        testdata_datanames = []
        for i, sys in enumerate(testdata.systems.values()):
            for j in range(len(sys)):
                testdata_datanames.append(f"{list(testdata.systems.keys())[i]}-{j}")
        
        # Find selected indices in testdata
        selected_datanames = df_uq_desc_selected_final["dataname"].tolist()
        selected_indices = []
        
        for dataname in selected_datanames:
            if dataname in testdata_datanames:
                selected_indices.append(testdata_datanames.index(dataname))
        
        # Sample from testdata
        sampled_systems = dpdata.MultiSystems()
        current_idx = 0
        
        for sys_name, sys in testdata.systems.items():
            sys_indices = []
            for i in range(len(sys)):
                if current_idx in selected_indices:
                    sys_indices.append(i)
                current_idx += 1
            
            if sys_indices:
                sampled_sys = sys.sub_system(sys_indices)
                sampled_systems.append(sampled_sys, sys_name)
        
        return sampled_systems
    
    def export_sampled_data(self, sampled_systems: dpdata.MultiSystems):
        """
        Export sampled data to files.
        
        Args:
            sampled_systems (dpdata.MultiSystems): Sampled MultiSystems
        """
        self.logger.info("Exporting sampled data")
        
        # Export to deepmd/npy format
        sampled_systems.to_deepmd_npy(f"{self.config.dpdata_savedir}/sampled_deepmd_npy")
        
        # Export to VASP POSCAR format
        poscar_dir = f"{self.config.dpdata_savedir}/sampled_poscar"
        if not os.path.exists(poscar_dir):
            os.makedirs(poscar_dir)
        
        frame_count = 0
        for sys_name, sys in sampled_systems.systems.items():
            for i in range(len(sys)):
                frame = sys.sub_system([i])
                frame.to_vasp_poscar(f"{poscar_dir}/POSCAR_{frame_count:04d}")
                frame_count += 1
        
        self.logger.info(f"Exported {frame_count} structures to {poscar_dir}")


class UQPostProcessor:
    """
    Main orchestrator class for UQ post-processing workflow.
    
    This class coordinates all components to execute the complete
    UQ post-processing and visualization pipeline.
    """
    
    def __init__(self, config: UQConfig):
        """
        Initialize UQPostProcessor with configuration.
        
        Args:
            config (UQConfig): Configuration object
        """
        self.config = config
        self.logger = self._setup_logging()
        
        # Initialize components
        self.dir_manager = DirectoryManager(config)
        self.data_loader = DataLoader(config)
        self.uq_calculator = UQCalculator(config)
        self.uq_selector = UQSelector(config)
        self.visualizer = UQVisualizer(config)
        self.direct_processor = DIRECTProcessor(config)
        self.data_processor = DataProcessor(config)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def run_complete_workflow(self):
        """
        Execute the complete UQ post-processing workflow.
        
        This method orchestrates all steps of the UQ analysis pipeline
        from data loading to final structure selection and export.
        """
        self.logger.info("Starting UQ post-processing workflow")
        
        # Step 1: Setup and validation
        self.dir_manager.validate_directories()
        self.dir_manager.create_output_directories()
        
        # Step 2: Data loading
        test_results = self.data_loader.load_test_results()
        testdata = self.data_loader.load_test_data()
        desc_datanames, desc_stru = self.data_loader.load_descriptors()
        
        # Step 3: UQ calculations
        diff_maxf_0_frame, diff_rmsf_0_frame = self.uq_calculator.calculate_force_differences(test_results)
        forces = self.uq_calculator.extract_atomic_forces(test_results)
        uq_qbc_for = self.uq_calculator.calculate_qbc_uq(forces, test_results)
        uq_rnd_for = self.uq_calculator.calculate_rnd_uq(forces, test_results)
        uq_rnd_for_rescaled = self.uq_calculator.rescale_rnd_to_qbc(uq_qbc_for, uq_rnd_for)
        
        # Step 4: Create comprehensive DataFrame
        df_uq_desc = self._create_comprehensive_dataframe(
            desc_datanames, desc_stru, uq_qbc_for, uq_rnd_for, 
            uq_rnd_for_rescaled, diff_maxf_0_frame, diff_rmsf_0_frame
        )
        
        # Step 5: Visualizations
        self.visualizer.plot_uq_distributions(uq_qbc_for, uq_rnd_for, uq_rnd_for_rescaled)
        self.visualizer.plot_uq_with_trust_ranges(uq_qbc_for, uq_rnd_for_rescaled)
        self.visualizer.plot_uq_vs_force_diff(uq_qbc_for, uq_rnd_for, uq_rnd_for_rescaled, diff_maxf_0_frame)
        self.visualizer.plot_uq_difference_analysis(uq_qbc_for, uq_rnd_for_rescaled, diff_maxf_0_frame)
        
        # Step 6: Structure selection
        df_uq_desc_candidate, df_uq_accurate, df_uq_failed = self.uq_selector.select_structures(df_uq_desc)
        
        # Add identity column for visualization
        df_uq_all = self._add_identity_column(df_uq_desc, df_uq_desc_candidate, df_uq_accurate, df_uq_failed)
        
        # Step 7: Selection visualizations
        self.visualizer.plot_uq_2d_scatter_with_selection(df_uq_all)
        self.visualizer.plot_selection_identity_scatter(df_uq_all)
        self.visualizer.plot_candidate_analysis(uq_qbc_for, uq_rnd_for_rescaled, diff_maxf_0_frame, df_uq_desc_candidate)
        
        # Step 8: DIRECT sampling
        direct_results = self.direct_processor.perform_direct_sampling(df_uq_desc_candidate, desc_stru)
        self.direct_processor.visualize_direct_results(direct_results)
        
        # Step 9: Final selection
        df_uq_desc_selected_final = self._get_final_selection(df_uq_desc_candidate, direct_results)
        
        # Step 10: Final visualizations
        self.direct_processor.visualize_final_selection_pca(
            df_uq_desc, df_uq_desc_candidate, df_uq_desc_selected_final, desc_stru
        )
        
        # Step 11: Data processing and export
        self.data_processor.save_dataframes(
            df_uq_desc, df_uq_desc_candidate, df_uq_accurate, df_uq_failed, df_uq_desc_selected_final
        )
        
        sampled_systems = self.data_processor.perform_dpdata_sampling(df_uq_desc_selected_final, testdata)
        self.data_processor.export_sampled_data(sampled_systems)
        
        self.logger.info("UQ post-processing workflow completed successfully")
        self._print_summary(df_uq_desc, df_uq_desc_candidate, df_uq_accurate, df_uq_failed, df_uq_desc_selected_final)
    
    def _create_comprehensive_dataframe(self, desc_datanames: List[str], desc_stru: np.ndarray,
                                       uq_qbc_for: np.ndarray, uq_rnd_for: np.ndarray,
                                       uq_rnd_for_rescaled: np.ndarray, diff_maxf_0_frame: np.ndarray,
                                       diff_rmsf_0_frame: np.ndarray) -> pd.DataFrame:
        """Create comprehensive DataFrame with all data."""
        # Create descriptor columns
        desc_features = [f"desc_stru_{i}" for i in range(desc_stru.shape[1])]
        desc_df = pd.DataFrame(desc_stru, columns=desc_features)
        
        # Create main DataFrame
        df_uq_desc = pd.DataFrame({
            'dataname': desc_datanames,
            'uq_qbc_for': uq_qbc_for,
            'uq_rnd_for': uq_rnd_for,
            'uq_rnd_for_rescaled': uq_rnd_for_rescaled,
            'diff_maxf_0_frame': diff_maxf_0_frame,
            'diff_rmsf_0_frame': diff_rmsf_0_frame
        })
        
        # Combine with descriptors
        df_uq_desc = pd.concat([df_uq_desc, desc_df], axis=1)
        
        return df_uq_desc
    
    def _add_identity_column(self, df_uq_desc: pd.DataFrame, df_uq_desc_candidate: pd.DataFrame,
                            df_uq_accurate: pd.DataFrame, df_uq_failed: pd.DataFrame) -> pd.DataFrame:
        """Add identity column for visualization."""
        df_uq_all = df_uq_desc[["dataname", "uq_qbc_for", "uq_rnd_for_rescaled", "diff_maxf_0_frame"]].copy()
        df_uq_all["uq_identity"] = "other"
        
        # Set identities
        candidate_mask = df_uq_all["dataname"].isin(df_uq_desc_candidate["dataname"])
        accurate_mask = df_uq_all["dataname"].isin(df_uq_accurate["dataname"])
        failed_mask = df_uq_all["dataname"].isin(df_uq_failed["dataname"])
        
        df_uq_all.loc[candidate_mask, "uq_identity"] = "candidate"
        df_uq_all.loc[accurate_mask, "uq_identity"] = "accurate"
        df_uq_all.loc[failed_mask, "uq_identity"] = "failed"
        
        return df_uq_all
    
    def _get_final_selection(self, df_uq_desc_candidate: pd.DataFrame, direct_results: Dict) -> pd.DataFrame:
        """Get final selected structures from DIRECT results."""
        selected_indices = direct_results['DIRECT_selection']['selected_indices']
        return df_uq_desc_candidate.iloc[selected_indices].copy()
    
    def _print_summary(self, df_uq_desc: pd.DataFrame, df_uq_desc_candidate: pd.DataFrame,
                      df_uq_accurate: pd.DataFrame, df_uq_failed: pd.DataFrame,
                      df_uq_desc_selected_final: pd.DataFrame):
        """Print workflow summary."""
        self.logger.info("="*60)
        self.logger.info("WORKFLOW SUMMARY")
        self.logger.info("="*60)
        self.logger.info(f"Total structures: {len(df_uq_desc)}")
        self.logger.info(f"Candidate structures: {len(df_uq_desc_candidate)}")
        self.logger.info(f"Accurate structures: {len(df_uq_accurate)}")
        self.logger.info(f"Failed structures: {len(df_uq_failed)}")
        self.logger.info(f"Final selected structures: {len(df_uq_desc_selected_final)}")
        self.logger.info(f"Selection scheme: {self.config.uq_select_scheme}")
        self.logger.info(f"Results saved to: {self.config.root_savedir}")
        self.logger.info("="*60)


def main():
    """
    Main function to run the UQ post-processing workflow.
    
    This function creates the configuration and runs the complete workflow.
    Users can modify the UQConfig parameters as needed for their specific use case.
    """
    # Create configuration
    config = UQConfig()
    
    # Create and run processor
    processor = UQPostProcessor(config)
    processor.run_complete_workflow()


if __name__ == "__main__":
    main()