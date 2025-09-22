#!/usr/bin/env python3
"""
UQ Post-Processing and Visualization Tool - Object-Oriented Refactored Version

This module provides an object-oriented refactoring of the original uq-post-view.py script.
It maintains the same functionality and workflow while organizing the code into clear,
modular classes for better maintainability and reusability.

Author: Refactored from original uq-post-view.py
Date: 2024
"""

import logging
import os
import shutil
import glob
from copy import deepcopy
from typing import Dict, List, Tuple, Optional, Any

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


class UQConfig:
    """
    Configuration manager for UQ post-processing parameters.
    
    This class centralizes all configuration parameters used throughout
    the UQ post-processing workflow, making it easy to manage and modify
    settings in one place.
    """
    
    def __init__(self):
        """Initialize UQ configuration with default parameters."""
        # 项目和测试配置变量
        self.project = "stage9-2"  # str: 项目名称，用于定位项目相关的所有目录和文件
        self.uq_select_scheme = "tangent_lo"  # str: UQ选择方案，定义数据选择策略 (strict, circle_lo, tangent_lo, crossline_lo, loose)
        self.testing_dir = "test-val-npy"  # str: 测试目录名称，存储测试结果的子目录
        self.testing_head = "results"  # str: 测试结果文件前缀，用于识别测试结果文件
        
        # 描述符加载配置变量
        self.desc_dir = f"{self.project}/desc_other"  # str: 描述符目录路径，存储结构描述符文件
        self.desc_filename = "desc.npy"  # str: 描述符文件名，numpy格式的描述符数据文件
        
        # 测试数据配置变量
        self.testdata_dir = f"{self.project}/other_dpdata"  # str: 测试数据目录路径，存储dpdata格式的测试数据
        self.testdata_fmt = "deepmd/npy"  # str: 测试数据格式，指定dpdata加载格式
        self.testdata_string = "O*"  # str: 测试数据匹配字符串，用于筛选特定的测试数据文件
        
        # 图形设置变量
        self.kde_bw_adjust = 0.5  # float: KDE带宽调整参数，控制核密度估计的平滑程度 (0.1-2.0)
        self.fig_dpi = 150  # int: 图形分辨率，设置保存图片的DPI值 (72-300)
        
        # 保存路径配置变量
        self.root_savedir = "dpeva_uq_post"  # str: 根保存目录名称，所有输出文件的根目录
        self.view_savedir = f"./{self.project}/{self.root_savedir}/view"  # str: 可视化结果保存目录，存储生成的图表文件
        self.dpdata_savedir = f"./{self.project}/{self.root_savedir}/dpdata"  # str: dpdata结果保存目录，存储选择的数据结构
        self.df_savedir = f"./{self.project}/{self.root_savedir}/dataframe"  # str: 数据框保存目录，存储pandas DataFrame结果
        
        # 选择策略配置变量
        self.uq_qbc_trust_lo = 0.12  # float: QbC UQ信任下限阈值，低于此值认为预测可信 (0.0-1.0)
        self.uq_qbc_trust_hi = 0.22  # float: QbC UQ信任上限阈值，高于此值认为预测不可信 (0.0-1.0)
        self.uq_rnd_rescaled_trust_lo = self.uq_qbc_trust_lo  # float: 重缩放RND UQ信任下限，与QbC对齐后的下限
        self.uq_rnd_rescaled_trust_hi = self.uq_qbc_trust_hi  # float: 重缩放RND UQ信任上限，与QbC对齐后的上限
        self.num_selection = 100  # int: 选择数量，DIRECT算法要选择的最终样本数量 (10-1000)
        self.direct_k = 1  # int: DIRECT聚类参数，每个聚类选择的样本数量 (1-10)
        self.direct_thr_init = 0.5  # float: DIRECT初始阈值，聚类算法的初始距离阈值 (0.1-1.0)
        
        # UQ选择方案验证变量
        self.uq_options = ["strict", "circle_lo", "crossline_lo", "tangent_lo", "loose"]  # List[str]: 支持的UQ选择方案列表
        
        # Setup logging
        self._setup_logging()
        
        # Validate configuration
        self._validate_config()
        
        # Create directories
        self._create_directories()
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            filemode='w',
            filename="UQ-DIRECT-selection.log",
        )
        self.logger = logging.getLogger(__name__)
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        # Check directories exist
        required_dirs = [
            self.project,
            f"{self.project}/0/{self.testing_dir}",
            self.desc_dir,
            self.testdata_dir
        ]
        
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                self.logger.error(f"Required directory {dir_path} not found!")
                raise ValueError(f"Required directory {dir_path} not found!")
        
        # Validate UQ selection scheme
        if self.uq_select_scheme not in self.uq_options:
            self.logger.error(f"UQ selection scheme {self.uq_select_scheme} not supported! Please choose from {self.uq_options}.")
            raise ValueError(f"UQ selection scheme {self.uq_select_scheme} not supported! Please choose from {self.uq_options}.")
        
        # Validate trust thresholds
        if ((self.uq_qbc_trust_lo >= self.uq_qbc_trust_hi) or 
            (self.uq_rnd_rescaled_trust_lo >= self.uq_rnd_rescaled_trust_hi)):
            raise ValueError("Low trust threshold should be lower than High trust threshold!")
    
    def _create_directories(self) -> None:
        """Create necessary output directories."""
        for dir_path in [self.view_savedir, self.dpdata_savedir, self.df_savedir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)


class UQDataProcessor:
    """
    Data processor for UQ calculations and transformations.
    
    This class handles all UQ-related data processing including force difference
    calculations, UQ metric computations, and data scaling operations.
    """
    
    def __init__(self, config: UQConfig):
        """
        Initialize UQ data processor.
        
        Args:
            config: UQConfig instance containing all configuration parameters
        """
        # 配置引用变量
        self.config = config  # UQConfig: 配置对象引用，提供对所有配置参数的访问
        self.logger = config.logger  # logging.Logger: 日志记录器对象，用于记录数据处理过程中的信息
        
        # 原始数据容器变量
        self.dp_test_results = {}  # Dict[int, DPTestResults]: 测试结果字典，存储4个模型的DPTestResults对象 (键: 0-3)
        self.force_data = {}  # Dict[str, np.ndarray]: 力数据字典，存储各模型的力预测值和期望值
        self.uq_metrics = {}  # Dict[str, np.ndarray]: UQ指标字典，存储计算得到的各种不确定性指标
        self.scalers = {}  # Dict[str, StandardScaler]: 缩放器字典，存储用于数据标准化的sklearn缩放器对象
    
    def load_test_results(self) -> None:
        """Load DP test results from all models."""
        self.logger.info("Loading the test results")
        
        for i in range(4):
            result_path = f"./{self.config.project}/{i}/{self.config.testing_dir}/{self.config.testing_head}"
            self.dp_test_results[i] = DPTestResults(result_path)
    
    def calculate_force_differences(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate force differences between model 0 predictions and labels.
        
        Returns:
            Tuple of (max_force_diff_per_frame, rms_force_diff_per_frame)
        """
        self.logger.info("Dealing with force difference between 0 head prediction and existing label")
        
        dp_0 = self.dp_test_results[0]
        diff_f_0 = np.sqrt(dp_0.diff_fx**2 + dp_0.diff_fy**2 + dp_0.diff_fz**2)
        
        # Map diff_f_0 to each structure with max force diff
        index = 0
        diff_maxf_0_frame = []
        diff_rmsf_0_frame = []
        
        for item in dp_0.dataname_list:
            natom = item[2]
            diff_f_0_item = diff_f_0[index:index + natom]
            diff_maxf_0_frame.append(np.max(diff_f_0_item))
            diff_rmsf_0_frame.append(np.sqrt(np.mean(diff_f_0_item**2)))
            index += natom
        
        return np.array(diff_maxf_0_frame), np.array(diff_rmsf_0_frame)
    
    def extract_force_predictions(self) -> Dict[str, np.ndarray]:
        """
        Extract force predictions from all models.
        
        Returns:
            Dictionary containing force predictions for each model and expected values
        """
        self.logger.info("Dealing with atomic force and average 1, 2, 3")
        
        forces = {}
        for i in range(4):
            dp = self.dp_test_results[i]
            forces[f'fx_{i}'] = dp.data_f['pred_fx']
            forces[f'fy_{i}'] = dp.data_f['pred_fy']
            forces[f'fz_{i}'] = dp.data_f['pred_fz']
        
        # Calculate expected values (average of models 1, 2, 3)
        forces['fx_expt'] = np.mean([forces['fx_1'], forces['fx_2'], forces['fx_3']], axis=0)
        forces['fy_expt'] = np.mean([forces['fy_1'], forces['fy_2'], forces['fy_3']], axis=0)
        forces['fz_expt'] = np.mean([forces['fz_1'], forces['fz_2'], forces['fz_3']], axis=0)
        
        self.force_data = forces
        return forces
    
    def calculate_qbc_uq(self) -> np.ndarray:
        """
        Calculate QbC (Query by Committee) UQ metrics.
        
        Returns:
            Array of QbC UQ values per structure
        """
        self.logger.info("Dealing with QbC force UQ")
        
        forces = self.force_data
        
        # Calculate QbC force UQ using DPGEN formula
        fx_qbc_square_diff = np.mean([
            (forces['fx_1'] - forces['fx_expt'])**2,
            (forces['fx_2'] - forces['fx_expt'])**2,
            (forces['fx_3'] - forces['fx_expt'])**2
        ], axis=0)
        
        fy_qbc_square_diff = np.mean([
            (forces['fy_1'] - forces['fy_expt'])**2,
            (forces['fy_2'] - forces['fy_expt'])**2,
            (forces['fy_3'] - forces['fy_expt'])**2
        ], axis=0)
        
        fz_qbc_square_diff = np.mean([
            (forces['fz_1'] - forces['fz_expt'])**2,
            (forces['fz_2'] - forces['fz_expt'])**2,
            (forces['fz_3'] - forces['fz_expt'])**2
        ], axis=0)
        
        f_qbc_stddiff = np.sqrt(fx_qbc_square_diff + fy_qbc_square_diff + fz_qbc_square_diff)
        
        # Assign atomic force stddiff to each structure and get UQ by max atomic force diff
        index = 0
        uq_qbc_for_list = []
        
        for item in self.dp_test_results[0].dataname_list:
            natom = item[2]
            f_qbc_stddiff_item = f_qbc_stddiff[index:index + natom]
            uq_qbc_for_list.append(np.max(f_qbc_stddiff_item))
            index += natom
        
        uq_qbc_for = np.array(uq_qbc_for_list)
        self.uq_metrics['qbc'] = uq_qbc_for
        return uq_qbc_for
    
    def calculate_rnd_uq(self) -> np.ndarray:
        """
        Calculate RND-like UQ metrics.
        
        Returns:
            Array of RND UQ values per structure
        """
        self.logger.info("Dealing with RND-like force UQ")
        
        forces = self.force_data
        
        # Calculate RND-like force UQ
        fx_rnd_square_diff = np.mean([
            (forces['fx_1'] - forces['fx_0'])**2,
            (forces['fx_2'] - forces['fx_0'])**2,
            (forces['fx_3'] - forces['fx_0'])**2
        ], axis=0)
        
        fy_rnd_square_diff = np.mean([
            (forces['fy_1'] - forces['fy_0'])**2,
            (forces['fy_2'] - forces['fy_0'])**2,
            (forces['fy_3'] - forces['fy_0'])**2
        ], axis=0)
        
        fz_rnd_square_diff = np.mean([
            (forces['fz_1'] - forces['fz_0'])**2,
            (forces['fz_2'] - forces['fz_0'])**2,
            (forces['fz_3'] - forces['fz_0'])**2
        ], axis=0)
        
        f_rnd_stddiff = np.sqrt(fx_rnd_square_diff + fy_rnd_square_diff + fz_rnd_square_diff)
        
        # Assign atomic force stddiff to each structure and get UQ by max atomic force diff
        index = 0
        uq_rnd_for_list = []
        
        for item in self.dp_test_results[0].dataname_list:
            natom = item[2]
            f_rnd_stddiff_item = f_rnd_stddiff[index:index + natom]
            uq_rnd_for_list.append(np.max(f_rnd_stddiff_item))
            index += natom
        
        uq_rnd_for = np.array(uq_rnd_for_list)
        self.uq_metrics['rnd'] = uq_rnd_for
        return uq_rnd_for
    
    def align_uq_metrics(self, uq_qbc_for: np.ndarray, uq_rnd_for: np.ndarray) -> np.ndarray:
        """
        Align RND UQ to QbC UQ using Z-Score normalization.
        
        Args:
            uq_qbc_for: QbC UQ values
            uq_rnd_for: RND UQ values
            
        Returns:
            Rescaled RND UQ values aligned to QbC scale
        """
        self.logger.info("Aligning UQ-RND to UQ-QbC by Z-Score")
        
        scaler_qbc_for = StandardScaler()
        scaler_rnd_for = StandardScaler()
        
        uq_qbc_for_scaled = scaler_qbc_for.fit_transform(uq_qbc_for.reshape(-1, 1)).flatten()
        uq_rnd_for_scaled = scaler_rnd_for.fit_transform(uq_rnd_for.reshape(-1, 1)).flatten()
        uq_rnd_for_rescaled = scaler_qbc_for.inverse_transform(uq_rnd_for_scaled.reshape(-1, 1)).flatten()
        
        self.scalers['qbc'] = scaler_qbc_for
        self.scalers['rnd'] = scaler_rnd_for
        self.uq_metrics['rnd_rescaled'] = uq_rnd_for_rescaled
        
        return uq_rnd_for_rescaled


class UQVisualizer:
    """
    Visualization manager for UQ analysis plots.
    
    This class handles all plotting and visualization tasks for UQ analysis,
    including distribution plots, scatter plots, and selection visualizations.
    """
    
    def __init__(self, config: UQConfig):
        """
        Initialize UQ visualizer.
        
        Args:
            config: UQConfig instance containing configuration parameters
        """
        # 配置和数据引用变量
        self.config = config  # UQConfig: 配置对象引用，提供绘图参数和设置
        self.logger = config.logger  # logging.Logger: 日志记录器对象，用于记录可视化过程中的信息
        
        # 绘图样式配置变量
        plt.rcParams['xtick.direction'] = 'in'  # str: X轴刻度方向，设置为向内
        plt.rcParams['ytick.direction'] = 'in'  # str: Y轴刻度方向，设置为向内
        plt.rcParams['font.size'] = 10  # int: 字体大小，设置图表中文字的默认大小
    
    def plot_uq_distributions(self, uq_qbc_for: np.ndarray, uq_rnd_for: np.ndarray, 
                             uq_rnd_for_rescaled: np.ndarray) -> None:
        """
        Plot UQ distribution comparisons.
        
        Args:
            uq_qbc_for: QbC UQ values
            uq_rnd_for: Original RND UQ values
            uq_rnd_for_rescaled: Rescaled RND UQ values
        """
        # Plot original UQ distributions
        self.logger.info("Plotting and saving the figures of UQ-force")
        plt.figure(figsize=(8, 6))
        sns.kdeplot(uq_qbc_for, color="blue", label="UQ-QbC", bw_adjust=self.config.kde_bw_adjust)
        sns.kdeplot(uq_rnd_for, color="red", label="UQ-RND", bw_adjust=self.config.kde_bw_adjust)
        plt.title("Distribution of UQ-force by KDEplot")
        plt.xlabel("UQ Value")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.config.view_savedir}/UQ-force.png", dpi=self.config.fig_dpi)
        plt.close()
        
        # Plot rescaled UQ distributions
        self.logger.info("Plotting and saving the figures of UQ-force rescaled")
        plt.figure(figsize=(8, 6))
        sns.kdeplot(uq_qbc_for, color="blue", label="UQ-QbC", bw_adjust=self.config.kde_bw_adjust)
        sns.kdeplot(uq_rnd_for_rescaled, color="red", label="UQ-RND-rescaled", bw_adjust=self.config.kde_bw_adjust)
        plt.title("Distribution of UQ-force by KDEplot")
        plt.xlabel("UQ Value")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.config.view_savedir}/UQ-force-rescaled.png", dpi=self.config.fig_dpi)
        plt.close()
    
    def plot_uq_trust_ranges(self, uq_qbc_for: np.ndarray, uq_rnd_for_rescaled: np.ndarray) -> None:
        """
        Plot UQ distributions with trust range indicators.
        
        Args:
            uq_qbc_for: QbC UQ values
            uq_rnd_for_rescaled: Rescaled RND UQ values
        """
        # Plot QbC with trust range
        self.logger.info("Plotting and saving the figures of UQ-QbC-force with UQ trust range")
        plt.figure(figsize=(8, 6))
        sns.kdeplot(uq_qbc_for, color="blue", bw_adjust=self.config.kde_bw_adjust)
        plt.title("Distribution of UQ-QbC-force by KDEplot")
        plt.xlabel("UQ-QbC Value")
        plt.ylabel("Density")
        plt.grid(True)
        
        # Add trust range indicators
        plt.axvline(self.config.uq_qbc_trust_lo, color='purple', linestyle='--', linewidth=1)
        plt.axvline(self.config.uq_qbc_trust_hi, color='purple', linestyle='--', linewidth=1)
        plt.axvspan(np.min(uq_qbc_for), self.config.uq_qbc_trust_lo, alpha=0.1, color='green')
        plt.axvspan(self.config.uq_qbc_trust_lo, self.config.uq_qbc_trust_hi, alpha=0.1, color='yellow')
        plt.axvspan(self.config.uq_qbc_trust_hi, np.max(uq_qbc_for), alpha=0.1, color='red')
        
        plt.savefig(f"{self.config.view_savedir}/UQ-QbC-force.png", dpi=self.config.fig_dpi)
        plt.close()
        
        # Plot RND rescaled with trust range
        self.logger.info("Plotting and saving the figures of UQ-RND-force-rescaled with UQ trust range")
        plt.figure(figsize=(8, 6))
        sns.kdeplot(uq_rnd_for_rescaled, color="blue", bw_adjust=self.config.kde_bw_adjust)
        plt.title("Distribution of UQ-RND-force-rescaled by KDEplot")
        plt.xlabel("UQ-RND-rescaled Value")
        plt.ylabel("Density")
        plt.grid(True)
        
        # Add trust range indicators
        plt.axvline(self.config.uq_qbc_trust_lo, color='purple', linestyle='--', linewidth=1)
        plt.axvline(self.config.uq_qbc_trust_hi, color='purple', linestyle='--', linewidth=1)
        plt.axvspan(np.min(uq_rnd_for_rescaled), self.config.uq_qbc_trust_lo, alpha=0.1, color='green')
        plt.axvspan(self.config.uq_qbc_trust_lo, self.config.uq_qbc_trust_hi, alpha=0.1, color='yellow')
        plt.axvspan(self.config.uq_qbc_trust_hi, np.max(uq_rnd_for_rescaled), alpha=0.1, color='red')
        
        plt.savefig(f"{self.config.view_savedir}/UQ-RND-force-rescaled.png", dpi=self.config.fig_dpi)
        plt.close()
    
    def plot_uq_vs_force_diff(self, uq_qbc_for: np.ndarray, uq_rnd_for: np.ndarray, 
                             uq_rnd_for_rescaled: np.ndarray, diff_maxf_0_frame: np.ndarray) -> None:
        """
        Plot UQ values vs force differences.
        
        Args:
            uq_qbc_for: QbC UQ values
            uq_rnd_for: Original RND UQ values
            uq_rnd_for_rescaled: Rescaled RND UQ values
            diff_maxf_0_frame: Maximum force differences per frame
        """
        # Plot original UQ vs force diff
        self.logger.info("Plotting and saving the figures of UQ-force vs force diff")
        plt.figure(figsize=(8, 6))
        plt.scatter(uq_qbc_for, diff_maxf_0_frame, color="blue", label="QbC", s=20)
        plt.scatter(uq_rnd_for, diff_maxf_0_frame, color="red", label="RND", s=20)
        plt.title("UQ vs Force Diff")
        plt.xlabel("UQ Value")
        plt.ylabel("True Max Force Diff")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.config.view_savedir}/UQ-force-fdiff-parity.png", dpi=self.config.fig_dpi)
        plt.close()
        
        # Plot rescaled UQ vs force diff
        self.logger.info("Plotting and saving the figures of UQ-force-rescaled vs force diff")
        plt.figure(figsize=(8, 6))
        plt.scatter(uq_qbc_for, diff_maxf_0_frame, color="blue", label="QbC", s=20)
        plt.scatter(uq_rnd_for_rescaled, diff_maxf_0_frame, color="red", label="RND-rescaled", s=20)
        plt.title("UQ vs Force Diff")
        plt.xlabel("UQ Value")
        plt.ylabel("True Max Force Diff")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.config.view_savedir}/UQ-force-rescaled-fdiff-parity.png", dpi=self.config.fig_dpi)
        plt.close()
    
    def plot_uq_difference_analysis(self, uq_qbc_for: np.ndarray, uq_rnd_for_rescaled: np.ndarray, 
                                   diff_maxf_0_frame: np.ndarray) -> None:
        """
        Plot UQ difference analysis.
        
        Args:
            uq_qbc_for: QbC UQ values
            uq_rnd_for_rescaled: Rescaled RND UQ values
            diff_maxf_0_frame: Maximum force differences per frame
        """
        uq_diff_for_scaled_to_qbc = np.abs(uq_rnd_for_rescaled - uq_qbc_for)
        
        # Plot UQ difference vs UQ values
        self.logger.info("Plotting and saving the figures of UQ-diff vs UQ")
        plt.figure(figsize=(8, 6))
        plt.scatter(uq_diff_for_scaled_to_qbc, uq_qbc_for, color="blue", label="UQ-qbc-for", s=20)
        plt.scatter(uq_diff_for_scaled_to_qbc, uq_rnd_for_rescaled, color="red", label="UQ-rnd-for-rescaled", s=20)
        plt.title("UQ-diff vs UQ")
        plt.xlabel("UQ-diff Value")
        plt.ylabel("UQ Value")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.config.view_savedir}/UQ-diff-UQ-parity.png", dpi=self.config.fig_dpi)
        plt.close()
        
        # Plot UQ difference vs force diff
        self.logger.info("Plotting and saving the figures of UQ-diff vs force diff")
        plt.figure(figsize=(8, 6))
        plt.scatter(uq_diff_for_scaled_to_qbc, diff_maxf_0_frame, color="blue", label="UQ-diff-force", s=20)
        plt.title("UQ-diff vs Force Diff")
        plt.xlabel("UQ-diff Value")
        plt.ylabel("True Max Force Diff")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.config.view_savedir}/UQ-diff-fdiff-parity.png", dpi=self.config.fig_dpi)
        plt.close()
    
    def plot_uq_selection_scatter(self, df_uq: pd.DataFrame) -> None:
        """
        Plot UQ selection results in 2D scatter plot.
        
        Args:
            df_uq: DataFrame containing UQ data and selection results
        """
        self.logger.info("Plotting and saving the figure of UQ-identity in QbC-RND 2D space")
        
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df_uq, 
                       x="uq_qbc_for", 
                       y="uq_rnd_for_rescaled", 
                       hue="uq_identity", 
                       palette={"candidate": "orange", 
                               "accurate": "green", 
                               "failed": "red"},
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
        self._plot_selection_boundaries()
        
        plt.savefig(f"{self.config.view_savedir}/UQ-force-qbc-rnd-identity-scatter.png", dpi=self.config.fig_dpi)
        plt.close()
    
    def _plot_selection_boundaries(self) -> None:
        """
        Plot selection boundaries based on UQ selection scheme.
        """
        uq_x_lo = self.config.uq_qbc_trust_lo
        uq_y_lo = self.config.uq_rnd_rescaled_trust_lo
        uq_x_hi = self.config.uq_qbc_trust_hi
        uq_y_hi = self.config.uq_rnd_rescaled_trust_hi
        
        # Plot basic boundaries
        plt.plot([0, uq_x_hi], [uq_y_hi, uq_y_hi], color='black', linestyle='--', linewidth=2)
        plt.plot([uq_x_hi, uq_x_hi], [0, uq_y_hi], color='black', linestyle='--', linewidth=2)
        
        # Plot scheme-specific boundaries
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
    
    def _balance_linear_func(self, x_lo: float, x_hi: float, y_lo: float, y_hi: float, 
                           x_range: Tuple[float, float], num_points: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate balance linear function for crossline selection scheme.
        
        Args:
            x_lo, x_hi, y_lo, y_hi: Trust threshold coordinates
            x_range: Range for x values
            num_points: Number of points to generate
            
        Returns:
            Tuple of (x_values, y_values)
        """
        x_val = np.linspace(x_range[0], x_range[1], num_points)
        delta_y = y_hi - y_lo
        delta_x = x_hi - x_lo
        y1 = (y_hi * x_lo - delta_y * x_val) / x_lo
        y2 = (y_lo * x_hi - y_lo * x_val) / delta_x
        y = np.max((y1, y2), axis=0)
        return x_val, y


class UQSelector:
    """
    UQ-based data selector implementing various selection schemes.
    
    This class handles the logic for selecting data points based on UQ metrics
    using different selection schemes (strict, circle_lo, tangent_lo, etc.).
    """
    
    def __init__(self, config: UQConfig):
        """
        Initialize UQ selector.
        
        Args:
            config: UQConfig instance containing configuration parameters
        """
        # 配置和数据引用变量
        self.config = config  # UQConfig: 配置对象引用，提供选择方案和阈值参数
        self.logger = config.logger  # logging.Logger: 日志记录器对象，用于记录选择过程中的信息
    
    def select_by_uq_scheme(self, df_uq_desc: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Select data points based on configured UQ selection scheme.
        
        Args:
            df_uq_desc: DataFrame containing UQ metrics and descriptors
            
        Returns:
            Tuple of (candidate_df, accurate_df, failed_df)
        """
        uq_x = df_uq_desc["uq_qbc_for"]
        uq_y = df_uq_desc["uq_rnd_for_rescaled"]
        
        uq_x_lo = self.config.uq_qbc_trust_lo
        uq_y_lo = self.config.uq_rnd_rescaled_trust_lo
        uq_x_hi = self.config.uq_qbc_trust_hi
        uq_y_hi = self.config.uq_rnd_rescaled_trust_hi
        
        if self.config.uq_select_scheme == "strict":
            return self._strict_selection(df_uq_desc, uq_x, uq_y, uq_x_lo, uq_y_lo, uq_x_hi, uq_y_hi)
        
        elif self.config.uq_select_scheme == "circle_lo":
            return self._circle_lo_selection(df_uq_desc, uq_x, uq_y, uq_x_lo, uq_y_lo, uq_x_hi, uq_y_hi)
        
        elif self.config.uq_select_scheme == "tangent_lo":
            return self._tangent_lo_selection(df_uq_desc, uq_x, uq_y, uq_x_lo, uq_y_lo, uq_x_hi, uq_y_hi)
        
        elif self.config.uq_select_scheme == "crossline_lo":
            return self._crossline_lo_selection(df_uq_desc, uq_x, uq_y, uq_x_lo, uq_y_lo, uq_x_hi, uq_y_hi)
        
        elif self.config.uq_select_scheme == "loose":
            return self._loose_selection(df_uq_desc, uq_x, uq_y, uq_x_lo, uq_y_lo, uq_x_hi, uq_y_hi)
        
        else:
            raise ValueError(f"UQ selection scheme {self.config.uq_select_scheme} not supported!")
    
    def _strict_selection(self, df_uq_desc: pd.DataFrame, uq_x: pd.Series, uq_y: pd.Series,
                         uq_x_lo: float, uq_y_lo: float, uq_x_hi: float, uq_y_hi: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Strict selection: QbC and RND-like are both trustable."""
        df_uq = df_uq_desc[["dataname", "uq_qbc_for", "uq_rnd_for_rescaled", "uq_rnd_for", "diff_maxf_0_frame"]]
        
        df_candidate = df_uq_desc[
            (uq_x >= uq_x_lo) & (uq_x <= uq_x_hi) & (uq_y >= uq_y_lo) & (uq_y <= uq_y_hi)
        ]
        
        df_accurate = df_uq[
            ((uq_x < uq_x_lo) & (uq_y < uq_y_hi)) | ((uq_x < uq_x_hi) & (uq_y < uq_y_lo))
        ]
        
        df_failed = df_uq[
            (uq_x > uq_x_hi) | (uq_y > uq_y_hi)
        ]
        
        return df_candidate, df_accurate, df_failed
    
    def _circle_lo_selection(self, df_uq_desc: pd.DataFrame, uq_x: pd.Series, uq_y: pd.Series,
                            uq_x_lo: float, uq_y_lo: float, uq_x_hi: float, uq_y_hi: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Circle balance selection: QbC and RND-like are trustable in a circle balance way."""
        df_uq = df_uq_desc[["dataname", "uq_qbc_for", "uq_rnd_for_rescaled", "uq_rnd_for", "diff_maxf_0_frame"]]
        
        df_candidate = df_uq_desc[
            ((uq_x <= uq_x_hi) & (uq_y <= uq_y_hi)) &
            ((uq_x-uq_x_hi)**2 + (uq_y-uq_y_hi)**2 <= (uq_x_hi - uq_x_lo)**2 + (uq_y_hi - uq_y_lo)**2)
        ]
        
        df_accurate = df_uq[
            ((uq_x-uq_x_hi)**2 + (uq_y-uq_y_hi)**2 > (uq_x_lo - uq_x_hi)**2 + (uq_y_lo - uq_y_hi)**2) & 
            ((uq_x < uq_x_hi) & (uq_y < uq_y_hi))
        ]
        
        df_failed = df_uq[
            (uq_x > uq_x_hi) | (uq_y > uq_y_hi)
        ]
        
        return df_candidate, df_accurate, df_failed
    
    def _tangent_lo_selection(self, df_uq_desc: pd.DataFrame, uq_x: pd.Series, uq_y: pd.Series,
                             uq_x_lo: float, uq_y_lo: float, uq_x_hi: float, uq_y_hi: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Tangent balance selection: QbC and RND-like are trustable in a tangent-circle balance way."""
        df_uq = df_uq_desc[["dataname", "uq_qbc_for", "uq_rnd_for_rescaled", "uq_rnd_for", "diff_maxf_0_frame"]]
        
        df_candidate = df_uq_desc[
            ((uq_x <= uq_x_hi) & (uq_y <= uq_y_hi)) &
            ((uq_x-uq_x_lo)*(uq_x_lo-uq_x_hi) + (uq_y-uq_y_lo)*(uq_y_lo-uq_y_hi) <= 0)
        ]
        
        df_accurate = df_uq[
            ((uq_x-uq_x_lo)*(uq_x_lo-uq_x_hi) + (uq_y-uq_y_lo)*(uq_y_lo-uq_y_hi) > 0) & 
            ((uq_x < uq_x_hi) & (uq_y < uq_y_hi))
        ]
        
        df_failed = df_uq[
            (uq_x > uq_x_hi) | (uq_y > uq_y_hi)
        ]
        
        return df_candidate, df_accurate, df_failed
    
    def _crossline_lo_selection(self, df_uq_desc: pd.DataFrame, uq_x: pd.Series, uq_y: pd.Series,
                               uq_x_lo: float, uq_y_lo: float, uq_x_hi: float, uq_y_hi: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Crossline balance selection: QbC and RND-like are trustable in a crossline balance way."""
        df_uq = df_uq_desc[["dataname", "uq_qbc_for", "uq_rnd_for_rescaled", "uq_rnd_for", "diff_maxf_0_frame"]]
        
        df_candidate = df_uq_desc[
            (uq_x <= uq_x_hi) & (uq_y <= uq_y_hi) &
            (uq_x_lo * uq_y + (uq_y_hi - uq_y_lo) * uq_x >= uq_x_lo * uq_y_hi) &
            (uq_x * uq_y_lo + (uq_x_hi - uq_x_lo) * uq_y >= uq_x_hi * uq_y_lo)
        ]
        
        df_accurate = df_uq[
            (uq_x_lo * uq_y + (uq_y_hi - uq_y_lo) * uq_x < uq_x_lo * uq_y_hi) |
            (uq_x * uq_y_lo + (uq_x_hi - uq_x_lo) * uq_y < uq_x_hi * uq_y_lo)
        ]
        
        df_failed = df_uq[
            (uq_x > uq_x_hi) | (uq_y > uq_y_hi)
        ]
        
        return df_candidate, df_accurate, df_failed
    
    def _loose_selection(self, df_uq_desc: pd.DataFrame, uq_x: pd.Series, uq_y: pd.Series,
                        uq_x_lo: float, uq_y_lo: float, uq_x_hi: float, uq_y_hi: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Loose selection: QbC or RND-like is either trustable."""
        df_uq = df_uq_desc[["dataname", "uq_qbc_for", "uq_rnd_for_rescaled", "uq_rnd_for", "diff_maxf_0_frame"]]
        
        df_candidate = df_uq_desc[
            ((uq_x >= uq_x_lo) & (uq_x <= uq_x_hi)) | ((uq_y >= uq_y_lo) & (uq_y <= uq_y_hi))
        ]
        
        df_accurate = df_uq[
            (uq_x < uq_x_lo) & (uq_y < uq_y_lo)
        ]
        
        df_failed = df_uq[
            (uq_x > uq_x_hi) | (uq_y > uq_y_hi)
        ]
        
        return df_candidate, df_accurate, df_failed


class DIRECTSamplerWrapper:
    """
    Wrapper for DIRECT sampling functionality.
    
    This class encapsulates the DIRECT sampling algorithm and provides
    methods for feature coverage analysis and visualization.
    """
    
    def __init__(self, config: UQConfig):
        """
        Initialize DIRECT sampler wrapper.
        
        Args:
            config: UQConfig instance containing configuration parameters
        """
        # 配置和数据引用变量
        self.config = config  # UQConfig: 配置对象引用，提供DIRECT采样参数
        self.logger = config.logger  # logging.Logger: 日志记录器对象，用于记录采样过程中的信息
        
        # DIRECT采样相关变量
        self.sampler = None  # DIRECTSampler: DIRECT采样器对象，初始化为None，在采样时创建
        self.selection_results = None  # Dict: 采样结果字典，包含选中的索引、PCA特征等信息，初始为None
    
    def perform_direct_sampling(self, df_candidate: pd.DataFrame, desc_features: List[str]) -> pd.DataFrame:
        """
        Perform DIRECT sampling on candidate structures.
        
        Args:
            df_candidate: DataFrame containing candidate structures
            desc_features: List of descriptor feature column names
            
        Returns:
            DataFrame containing selected structures
        """
        self.logger.info("Doing DIRECT Selection on UQ-selected data")
        
        self.sampler = DIRECTSampler(
            structure_encoder=None,
            clustering=BirchClustering(
                n=self.config.num_selection // self.config.direct_k, 
                threshold_init=self.config.direct_thr_init
            ),
            select_k_from_clusters=SelectKFromClusters(k=self.config.direct_k),
        )
        
        self.selection_results = self.sampler.fit_transform(df_candidate[desc_features].values)
        selected_indices = self.selection_results["selected_indices"]
        
        # Calculate explained variance and PCA features
        explained_variance = self.sampler.pca.pca.explained_variance_
        selected_PC_dim = len([e for e in explained_variance if e > 1])
        self.selection_results["PCAfeatures_unweighted"] = (
            self.selection_results["PCAfeatures"] / explained_variance[:selected_PC_dim]
        )
        
        df_selected = df_candidate.iloc[selected_indices]
        return df_selected
    
    def visualize_direct_results(self) -> None:
        """
        Visualize DIRECT sampling results including explained variance and feature coverage.
        """
        if self.selection_results is None:
            raise ValueError("DIRECT sampling must be performed before visualization")
        
        self.logger.info("Visualization of DIRECT results compared with Random")
        
        # Plot explained variance
        explained_variance = self.sampler.pca.pca.explained_variance_
        selected_PC_dim = len([e for e in explained_variance if e > 1])
        
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
        
        # Plot PCA feature coverage
        all_features = self.selection_results["PCAfeatures_unweighted"]
        selected_indices = self.selection_results["selected_indices"]
        
        self._plot_pca_feature_coverage(all_features, selected_indices, "DIRECT")
        
        # Compare with random selection
        np.random.seed(42)
        manual_selection_index = np.random.choice(len(all_features), self.config.num_selection, replace=False)
        self._plot_pca_feature_coverage(all_features, manual_selection_index, "Random")
        
        # Plot coverage score comparison
        self._plot_coverage_score_comparison(all_features, selected_indices, manual_selection_index)
    
    def _plot_pca_feature_coverage(self, all_features: np.ndarray, selected_indices: np.ndarray, method: str) -> None:
        """
        Plot PCA feature coverage for given selection method.
        
        Args:
            all_features: All PCA features
            selected_indices: Indices of selected features
            method: Selection method name
        """
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
        """
        Calculate feature coverage score for given selection.
        
        Args:
            all_features: All features
            selected_indices: Indices of selected features
            n_bins: Number of bins for histogram
            
        Returns:
            Coverage score
        """
        selected_features = all_features[selected_indices]
        n_all = np.count_nonzero(
            np.histogram(all_features, bins=np.linspace(min(all_features), max(all_features), n_bins))[0]
        )
        n_select = np.count_nonzero(
            np.histogram(selected_features, bins=np.linspace(min(all_features), max(all_features), n_bins))[0]
        )
        return n_select / n_all
    
    def _calculate_all_fcs(self, all_features: np.ndarray, selected_indices: np.ndarray, b_bins: int = 100) -> List[float]:
        """
        Calculate feature coverage scores for all dimensions.
        
        Args:
            all_features: All features
            selected_indices: Indices of selected features
            b_bins: Number of bins for histogram
            
        Returns:
            List of coverage scores for each dimension
        """
        select_scores = [
            self._calculate_feature_coverage_score(all_features[:, i], selected_indices, n_bins=b_bins)
            for i in range(all_features.shape[1])
        ]
        return select_scores
    
    def _plot_coverage_score_comparison(self, all_features: np.ndarray, direct_indices: np.ndarray, random_indices: np.ndarray) -> None:
        """
        Plot coverage score comparison between DIRECT and random selection.
        
        Args:
            all_features: All PCA features
            direct_indices: DIRECT selected indices
            random_indices: Random selected indices
        """
        scores_direct = self._calculate_all_fcs(all_features, direct_indices, b_bins=100)
        scores_random = self._calculate_all_fcs(all_features, random_indices, b_bins=100)
        
        x = np.arange(len(scores_direct))
        x_ticks = [f"PC {n+1}" for n in range(len(x))]
        
        plt.figure(figsize=(15, 4))
        plt.bar(
            x + 0.6,
            scores_direct,
            width=0.3,
            label=rf"DIRECT, $\overline{{\mathrm{{Coverage\ score}}}}$ = {np.mean(scores_direct):.3f}",
        )
        plt.bar(
            x + 0.3,
            scores_random,
            width=0.3,
            label=rf"Random, $\overline{{\mathrm{{Coverage\ score}}}}$ = {np.mean(scores_random):.3f}",
        )
        plt.xticks(x + 0.45, x_ticks, size=12)
        plt.yticks(np.linspace(0, 1.0, 6), size=12)
        plt.ylabel("Coverage score", size=12)
        plt.legend(shadow=True, loc="lower right", fontsize=12)
        plt.savefig(f"{self.config.view_savedir}/coverage_score.png", dpi=self.config.fig_dpi)
        plt.close()


class UQPostProcessor:
    """
    Main UQ post-processing coordinator class.
    
    This class orchestrates the entire UQ post-processing workflow by coordinating
    the various components (data processing, visualization, selection, sampling).
    """
    
    def __init__(self, config: Optional[UQConfig] = None):
        """
        Initialize UQ post-processor.
        
        Args:
            config: Optional UQConfig instance. If None, default config is created.
        """
        # 配置和日志变量
        self.config = config or UQConfig()  # UQConfig: 配置对象，提供所有工作流参数
        self.logger = self.config.logger  # logging.Logger: 日志记录器对象，用于记录整个工作流过程
        
        # 组件实例变量
        self.data_processor = UQDataProcessor(self.config)  # UQDataProcessor: 数据处理器，负责加载和处理UQ数据
        self.visualizer = UQVisualizer(self.config)  # UQVisualizer: 可视化器，负责生成各种图表
        self.selector = UQSelector(self.config)  # UQSelector: 选择器，负责基于UQ指标的数据选择
        self.direct_sampler = DIRECTSamplerWrapper(self.config)  # DIRECTSamplerWrapper: DIRECT采样器包装器，负责多样性采样
        
        # 数据容器变量
        self.test_data = None  # dpdata.MultiSystems: 测试数据集，包含原子结构和力信息，初始为None
        self.descriptors = None  # np.ndarray: 描述符数据，用于结构特征表示，初始为None
        self.df_uq_desc = None  # pd.DataFrame: 合并的UQ指标和描述符数据框，初始为None
        self.final_selection = None  # pd.DataFrame: 最终选择的数据结果，初始为None
    
    def run_full_workflow(self) -> None:
        """
        Execute the complete UQ post-processing workflow.
        
        This method runs all steps of the UQ analysis pipeline:
        1. Load and process test results
        2. Calculate UQ metrics
        3. Generate visualizations
        4. Perform UQ-based selection
        5. Apply DIRECT sampling
        6. Generate final outputs
        """
        self.logger.info(f"Initializing selection in {self.config.project} ---")
        
        # Step 1: Load and process data
        self._load_and_process_data()
        
        # Step 2: Calculate UQ metrics
        uq_metrics = self._calculate_uq_metrics()
        
        # Step 3: Generate visualizations
        self._generate_visualizations(uq_metrics)
        
        # Step 4: Perform UQ selection
        selection_results = self._perform_uq_selection()
        
        # Step 5: Apply DIRECT sampling
        final_selection = self._apply_direct_sampling(selection_results)
        
        # Step 6: Generate final outputs
        self._generate_final_outputs(final_selection)
        
        self.logger.info("All Done!")
    
    def _load_and_process_data(self) -> None:
        """
        Load test results, target data, and descriptors.
        """
        # Load test results
        self.data_processor.load_test_results()
        
        # Load target testing data
        self.logger.info(f"Loading the target testing data from {self.config.testdata_dir}")
        self.test_data = dpdata.MultiSystems.from_dir(
            self.config.testdata_dir, 
            self.config.testdata_string, 
            fmt=self.config.testdata_fmt
        )
        
        # Load descriptors
        self._load_descriptors()
    
    def _load_descriptors(self) -> None:
        """
        Load and process descriptor data.
        """
        self.logger.info(f"Loading the target descriptors from {self.config.testdata_dir}")
        
        desc