"""
模块化的UQ后处理和可视化分析工具
该文件实现了与uq-post-view.py相同的功能，但采用模块化设计
"""
import logging
import sys
import os

# 将src目录添加到Python路径中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# log setting 必须放在最前面，形成一个global的logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filemode='w',
    filename="UQ-DIRECT-selection.log",
)

logger = logging.getLogger(__name__)


import os
import shutil
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import pandas as pd
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import dpdata

# 导入dpeva包中的现有模块
from dpeva.io.dataproc import DPTestResults
from dpeva.sampling.direct import DIRECTSampler, BirchClustering, SelectKFromClusters


class UQConfig:
    """UQ分析的配置参数类"""
    
    def __init__(self, project="."):
        # testing setting
        self.project = project
        self.uq_select_scheme = "tangent_lo"  # strict, circle_lo, tangent_lo, crossline_lo, loose
        self.testing_dir = "test-val-npy"
        self.testing_head = "results"
        
        # descriptor loading setting
        self.desc_dir = f"{self.project}/desc_other"
        self.desc_filename = "desc.npy"
        
        # testdata setting
        self.testdata_dir = f"{self.project}/other_dpdata"
        self.testdata_fmt = "deepmd/npy"
        self.testdata_string = "O*"  # for correspondence
        
        # figure setting
        self.kde_bw_adjust = 0.5
        self.fig_dpi = 150
        
        # save setting
        self.root_savedir = "dpeva_uq_post"
        self.view_savedir = f"./{self.project}/{self.root_savedir}/view"
        self.dpdata_savedir = f"./{self.project}/{self.root_savedir}/dpdata"
        self.df_savedir = f"./{self.project}/{self.root_savedir}/dataframe"
        
        # selection setting
        self.uq_qbc_trust_lo = 0.12
        self.uq_qbc_trust_hi = 0.22
        self.uq_rnd_rescaled_trust_lo = self.uq_qbc_trust_lo
        self.uq_rnd_rescaled_trust_hi = self.uq_qbc_trust_hi
        self.num_selection = 100
        self.direct_k = 1
        self.direct_thr_init = 0.5
        self.n_bins = 100


class UQDataLoader:
    """UQ分析的数据加载器"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def load_test_results(self):
        """加载测试结果数据"""
        self.logger.info("Loading the test results")
        dp_test_results_0 = DPTestResults(f"./{self.config.project}/0/{self.config.testing_dir}/{self.config.testing_head}")
        dp_test_results_1 = DPTestResults(f"./{self.config.project}/1/{self.config.testing_dir}/{self.config.testing_head}")
        dp_test_results_2 = DPTestResults(f"./{self.config.project}/2/{self.config.testing_dir}/{self.config.testing_head}")
        dp_test_results_3 = DPTestResults(f"./{self.config.project}/3/{self.config.testing_dir}/{self.config.testing_head}")
        return [dp_test_results_0, dp_test_results_1, dp_test_results_2, dp_test_results_3]
    
    def load_test_data(self):
        """加载测试数据"""
        self.logger.info(f"Loading the target testing data from {self.config.testdata_dir}")
        test_data = dpdata.MultiSystems.from_dir(
            f"{self.config.testdata_dir}", 
            f"{self.config.testdata_string}", 
            fmt=f"{self.config.testdata_fmt}")
        return test_data
    
    def load_descriptors(self):
        """加载描述符数据"""
        self.logger.info(f"Loading the target descriptors from {self.config.testdata_dir}")
        desc_string_test = f'{self.config.desc_dir}/*/{self.config.desc_filename}'
        desc_datanames = []
        desc_stru = []
        desc_iter_list = sorted(glob.glob(desc_string_test))
        for f in desc_iter_list:
            # extract dirname of desc.npy from descriptors/*
            directory, _ = os.path.split(f)
            _, keyname = os.path.split(directory)
            one_desc = np.load(f)  # nframe, natoms, ndesc
            for i in range(len(one_desc)):
                desc_dataname = f"{keyname}-{i}"
                desc_datanames.append(desc_dataname)
                # mean the atomic descriptors to structure descriptors
                one_desc_stru = np.mean(one_desc[i], axis=0).reshape(1, -1)
                desc_stru.append(one_desc_stru)
        desc_stru = np.concatenate(desc_stru, axis=0)
        return desc_stru, desc_datanames


class UQCalculator:
    """UQ数据处理器"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def get_force_prediction_differences(self, dp_test_results_0):
        """获取各结构零号微调头与实际值之间的受力差，用来与UQ结果做可视化比较。"""
        self.logger.info("Dealing with force difference between 0 head prediction and existing label")
        diff_f_0 = np.sqrt(dp_test_results_0.diff_fx**2 + dp_test_results_0.diff_fy**2 + dp_test_results_0.diff_fz**2)
        # map diff_f_0 to each structures with max force diff
        index = 0
        diff_maxf_0_frame = []
        diff_rmsf_0_frame = []
        for item in dp_test_results_0.dataname_list:
            natom = item[2]
            diff_f_0_item = diff_f_0[index:index + natom]
            diff_maxf_0_frame.append(np.max(diff_f_0_item))
            diff_rmsf_0_frame.append(np.sqrt(np.mean(diff_f_0_item**2)))
            index += natom
        diff_maxf_0_frame = np.array(diff_maxf_0_frame)
        diff_rmsf_0_frame = np.array(diff_rmsf_0_frame)
        return diff_maxf_0_frame, diff_rmsf_0_frame
    
    def get_atomic_forces(self, dp_test_results_list):
        """获取各结构中的原子受力，并对非零号原子受力做平均"""
        dp_test_results_0, dp_test_results_1, dp_test_results_2, dp_test_results_3 = dp_test_results_list
        
        fx_0 = dp_test_results_0.data_f['pred_fx']
        fy_0 = dp_test_results_0.data_f['pred_fy']
        fz_0 = dp_test_results_0.data_f['pred_fz']
        fx_1 = dp_test_results_1.data_f['pred_fx']
        fy_1 = dp_test_results_1.data_f['pred_fy']
        fz_1 = dp_test_results_1.data_f['pred_fz']
        fx_2 = dp_test_results_2.data_f['pred_fx']
        fy_2 = dp_test_results_2.data_f['pred_fy']
        fz_2 = dp_test_results_2.data_f['pred_fz']
        fx_3 = dp_test_results_3.data_f['pred_fx']
        fy_3 = dp_test_results_3.data_f['pred_fy']
        fz_3 = dp_test_results_3.data_f['pred_fz'] 
        
        fx_expt = np.mean((fx_1, fx_2, fx_3), axis=0)
        fy_expt = np.mean((fy_1, fy_2, fy_3), axis=0)
        fz_expt = np.mean((fz_1, fz_2, fz_3), axis=0)
        
        atomic_forces = (fx_0, fy_0, fz_0), (fx_1, fy_1, fz_1), (fx_2, fy_2, fz_2), (fx_3, fy_3, fz_3), (fx_expt, fy_expt, fz_expt)
        
        return atomic_forces
    
    def calculate_force_stddiff_qbc(self, atomic_forces):
        """calculating standard deviation in atomic forces for UQ-QbC"""
        self.logger.info("Dealing with QbC force UQ by DPGEN formula")
        fx_1, fy_1, fz_1 = atomic_forces[1]
        fx_2, fy_2, fz_2 = atomic_forces[2]
        fx_3, fy_3, fz_3 = atomic_forces[3]
        fx_expt, fy_expt, fz_expt = atomic_forces[4]
        
        fx_square_diff_qbc = np.mean(((fx_1 - fx_expt)**2, (fx_2 - fx_expt)**2, (fx_3 - fx_expt)**2), axis=0) 
        fy_square_diff_qbc = np.mean(((fy_1 - fy_expt)**2, (fy_2 - fy_expt)**2, (fy_3 - fy_expt)**2), axis=0) 
        fz_square_diff_qbc = np.mean(((fz_1 - fz_expt)**2, (fz_2 - fz_expt)**2, (fz_3 - fz_expt)**2), axis=0) 
        f_qbc_stddiff = np.sqrt(fx_square_diff_qbc + fy_square_diff_qbc + fz_square_diff_qbc)
        return f_qbc_stddiff
    
    def calculate_force_stddiff_rnd(self, atomic_forces):
        """calculating standard deviation in atomic forces for UQ-RND-like"""
        self.logger.info("Dealing with RND-like force UQ")
        fx_0, fy_0, fz_0 = atomic_forces[0]
        fx_1, fy_1, fz_1 = atomic_forces[1]
        fx_2, fy_2, fz_2 = atomic_forces[2]
        fx_3, fy_3, fz_3 = atomic_forces[3]
        
        fx_square_diff_rnd = np.mean(((fx_1 - fx_0)**2, (fx_2 - fx_0)**2, (fx_3 - fx_0)**2), axis=0) 
        fy_square_diff_rnd = np.mean(((fy_1 - fy_0)**2, (fy_2 - fy_0)**2, (fy_3 - fy_0)**2), axis=0) 
        fz_square_diff_rnd = np.mean(((fz_1 - fz_0)**2, (fz_2 - fz_0)**2, (fz_3 - fz_0)**2), axis=0) 
        f_stddiff_rnd = np.sqrt(fx_square_diff_rnd + fy_square_diff_rnd + fz_square_diff_rnd)
        return f_stddiff_rnd
    
    def assign_uq_to_structures(self, f_stddiff_qbc, f_stddiff_rnd, dataname_list):
        """对每个结构，基于不同类型原子受力标准偏差分配Qbc和RND-like的UQ值"""
        # assign atomic force stddiff to each structure and get UQ by max atomic force diff
        index = 0
        uq_qbc_for_list = []
        uq_rnd_for_list = []
        for item in dataname_list:
            natom = item[2]
            f_stddiff_qbc_item = f_stddiff_qbc[index:index + natom]
            uq_qbc_for_list.append(np.max(f_stddiff_qbc_item))
            f_stddiff_rnd_item = f_stddiff_rnd[index:index + natom]
            uq_rnd_for_list.append(np.max(f_stddiff_rnd_item))
            index += natom
        uq_qbc_for = np.array(uq_qbc_for_list)
        uq_rnd_for = np.array(uq_rnd_for_list)
        return uq_qbc_for, uq_rnd_for
    
    def align_uq_values(self, uq_qbc_for, uq_rnd_for):
        """通过Z-Score方法对齐RND和QbC的UQ值，最终使用的UQ是UQ-QbC和UQ-RND_rescaled"""
        self.logger.info("Aligning UQ-RND to UQ-QbC by Z-Score")
        scaler_qbc_for = StandardScaler()
        scaler_rnd_for = StandardScaler()
        uq_qbc_for_scaled = scaler_qbc_for.fit_transform(uq_qbc_for.reshape(-1,1)).flatten()
        uq_rnd_for_scaled = scaler_rnd_for.fit_transform(uq_rnd_for.reshape(-1,1)).flatten()
        uq_rnd_for_rescaled = scaler_qbc_for.inverse_transform(uq_rnd_for_scaled.reshape(-1,1)).flatten()
        return uq_rnd_for_rescaled


class UQVisualizer:
    """UQ结果可视化器"""
    
    def __init__(self, config: UQConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        plt.rcParams['xtick.direction'] = 'in'  # set the direction of xticks inside
        plt.rcParams['ytick.direction'] = 'in'  # set the direction of yticks inside
        plt.rcParams['font.size'] = 10  # set the font size
        
    def _plot_uq_region(self, plt, uq_select_scheme, uq_x_lo, uq_x_hi, uq_y_lo, uq_y_hi):
        """
        绘制UQ选择区域的边界线

        Args:
            plt: matplotlib.pyplot 对象，用于绘图
            uq_select_scheme: str, UQ选择方案，可选值为 "strict", "circle_lo", "tangent_lo", "crossline_lo", "loose"
            uq_x_lo: float, QbC UQ值的下限
            uq_x_hi: float, QbC UQ值的上限
            uq_y_lo: float, RND-like UQ值的下限
            uq_y_hi: float, RND-like UQ值的上限
        """
        plt.plot([0, uq_x_hi], [uq_y_hi, uq_y_hi], color='black', linestyle='--', linewidth=2)
        plt.plot([uq_x_hi, uq_x_hi], [0, uq_y_hi], color='black', linestyle='--', linewidth=2)
        if uq_select_scheme == "strict":
            plt.plot([uq_x_lo, uq_x_lo], [uq_y_lo, uq_y_hi], color='purple', linestyle='--', linewidth=2)
            plt.plot([uq_x_lo, uq_x_hi], [uq_y_lo, uq_y_lo], color='purple', linestyle='--', linewidth=2)
        elif uq_select_scheme == "circle_lo":
            # ((uq_x-uq_x_hi)** 2 + (uq_y-uq_y_hi)**2 >= (uq_x_lo - uq_x_hi)**2 + (uq_y_lo - uq_y_hi)**2)
            center = (uq_x_hi, uq_y_hi)
            radius = np.sqrt((uq_x_lo - uq_x_hi)**2 + (uq_y_lo - uq_y_hi)**2)
            theta = np.linspace(np.pi, 1.5*np.pi, 100)
            x_val = center[0] + radius * np.cos(theta)
            y_val = center[1] + radius * np.sin(theta)
            plt.plot(x_val, y_val, color="purple", linestyle="--", linewidth=2)
        elif uq_select_scheme == "tangent_lo":
            # ((uq_x-uq_x_hi)*(uq_x_hi-uq_x_lo) + (uq_y-uq_y_hi)*(uq_y_hi-uq_y_lo) >= 0)
            x_val = np.linspace(0, uq_x_hi, 100)
            y_val = - (uq_y_hi - uq_y_lo) / (uq_x_hi - uq_x_lo) * (x_val - uq_x_lo) + uq_y_lo
            x_val = x_val[y_val < uq_y_hi]
            y_val = y_val[y_val < uq_y_hi]
            plt.plot(x_val, y_val, color="purple", linestyle="--", linewidth=2)
        elif uq_select_scheme == "crossline_lo":
            def balance_linear_func(x_lo, x_hi, y_lo, y_hi, x_range=(0,10), num_points=20):
                # y1 = (y_hi*x_lo - (y_hi - y_lo)*x)/x_lo
                # y2 = (y_lo*x_hi - y_lo*x)/(x_hi - x_lo)
                x_val = np.linspace(x_range[0], x_range[1], num_points)
                delta_y = y_hi - y_lo
                delta_x = x_hi - x_lo
                y1 = (y_hi * x_lo - delta_y * x_val) / x_lo
                y2 = (y_lo * x_hi - y_lo * x_val) / delta_x
                y = np.max((y1, y2), axis=0)
                return x_val, y
            x_val, y_val = balance_linear_func(
                uq_x_lo, uq_x_hi, uq_y_lo, uq_y_hi, (0, uq_x_hi), 100)
            # filter the x_val and y_val in the range of uq_x_lo and uq_x_hi
            x_val = x_val[y_val < uq_y_hi]
            y_val = y_val[y_val < uq_y_hi]
            plt.plot(x_val, y_val, color="purple", linestyle="--", linewidth=2)
        elif uq_select_scheme == "loose":
            plt.plot([uq_x_lo, uq_x_lo], [0, uq_y_lo], color='purple', linestyle='--', linewidth=2)
            plt.plot([0, uq_x_lo], [uq_y_lo, uq_y_lo], color='purple', linestyle='--', linewidth=2)
        else:
            raise ValueError(f"UQ selection scheme {uq_select_scheme} not supported!")
    
    def plot_uq_force_distribution(self, uq_qbc_for, uq_rnd_for, uq_rnd_for_rescaled, diff_maxf_0_frame):
        """绘制UQ-force分布图"""
        self.logger.info("Plotting and saving the figures of UQ-force")
        plt.figure(figsize=(8, 6))
        sns.kdeplot(uq_qbc_for, color="blue", label="UQ-QbC", bw_adjust=self.config.kde_bw_adjust)
        sns.kdeplot(uq_rnd_for, color="red", label="UQ-RND", bw_adjust=self.config.kde_bw_adjust)
        plt.title(f"UQ-force Distribution")
        plt.xlabel("UQ Value")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"./{self.config.view_savedir}/UQ-force.png", dpi=self.config.fig_dpi)
        plt.close()
        
        # 绘制重缩放的UQ力分布图
        self.logger.info("Plotting and saving the figures of UQ-force rescaled")
        plt.figure(figsize=(8, 6))
        sns.kdeplot(uq_qbc_for, color="blue", label="UQ-QbC", bw_adjust=self.config.kde_bw_adjust)
        sns.kdeplot(uq_rnd_for_rescaled, color="red", label="UQ-RND-rescaled", bw_adjust=self.config.kde_bw_adjust)
        plt.xlabel("UQ Value")
        plt.ylabel("Frequency")
        plt.title("UQ Force Distribution (Rescaled)")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"./{self.config.view_savedir}/UQ-force-rescaled.png", dpi=self.config.fig_dpi)
        plt.close()
        
        # 绘制带信任范围的UQ-QbC-force分布图
        self.logger.info("Plotting and saving the figures of UQ-QbC-force with UQ trust range")
        plt.figure(figsize=(8, 6))
        sns.kdeplot(uq_qbc_for, color="blue", bw_adjust=self.config.kde_bw_adjust)
        plt.title("Distribution of UQ-QbC-force by KDEplot")
        plt.xlabel("UQ-QbC Value")
        plt.ylabel("Density")
        plt.grid(True)
        plt.axvline(self.config.uq_qbc_trust_lo, color='purple', linestyle='--', linewidth=1)
        plt.axvline(self.config.uq_qbc_trust_hi, color='purple', linestyle='--', linewidth=1)
        # 显示UQ信任范围
        plt.axvspan(np.min(uq_qbc_for), self.config.uq_qbc_trust_lo, alpha=0.1, color='green')
        plt.axvspan(self.config.uq_qbc_trust_lo, self.config.uq_qbc_trust_hi, alpha=0.1, color='yellow')
        plt.axvspan(self.config.uq_qbc_trust_hi, np.max(uq_qbc_for), alpha=0.1, color='red')
        plt.savefig(f"./{self.config.view_savedir}/UQ-QbC-force.png", dpi=self.config.fig_dpi)
        plt.close()
        
        # 绘制带信任范围的UQ-RND-force-rescaled分布图
        self.logger.info("Plotting and saving the figures of UQ-RND-force-rescaled with UQ trust range")
        plt.figure(figsize=(8, 6))
        sns.kdeplot(uq_rnd_for_rescaled, color="blue", bw_adjust=self.config.kde_bw_adjust)
        plt.title("Distribution of UQ-RND-force-rescaled by KDEplot")
        plt.xlabel("UQ-RND-rescaled Value")
        plt.ylabel("Density")
        plt.grid(True)
        plt.axvline(self.config.uq_rnd_rescaled_trust_lo, color='purple', linestyle='--', linewidth=1)
        plt.axvline(self.config.uq_rnd_rescaled_trust_hi, color='purple', linestyle='--', linewidth=1)
        # 显示UQ信任范围
        plt.axvspan(np.min(uq_rnd_for_rescaled), self.config.uq_rnd_rescaled_trust_lo, alpha=0.1, color='green')
        plt.axvspan(self.config.uq_rnd_rescaled_trust_lo, self.config.uq_rnd_rescaled_trust_hi, alpha=0.1, color='yellow')
        plt.axvspan(self.config.uq_rnd_rescaled_trust_hi, np.max(uq_rnd_for_rescaled), alpha=0.1, color='red')
        plt.savefig(f"./{self.config.view_savedir}/UQ-RND-force-rescaled.png", dpi=self.config.fig_dpi)
        plt.close()
        
        # 绘制UQ-force vs force diff散点图
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
        
        # 绘制UQ-force-rescaled vs force diff散点图
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
    
    def plot_uq_difference(self, uq_qbc_for, uq_rnd_for_rescaled, diff_maxf_0_frame):
        """绘制两种UQ之差相关的可视化图"""
        # UQ差异散点图
        self.logger.info("Calculating |UQ-QbC - UQ-RND-rescaled| as UQ-diff")
        uq_diff = np.abs(uq_qbc_for - uq_rnd_for_rescaled)
        
        # UQ-diff vs UQ散点图
        self.logger.info("Plotting and saving the figures of UQ-diff vs UQ")
        plt.figure(figsize=(8, 6))
        plt.scatter(uq_diff, uq_qbc_for, color="blue", label="UQ-qbc-for", s=20)
        plt.scatter(uq_diff, uq_rnd_for_rescaled, color="red", label="UQ-rnd-for-rescaled", s=20)
        plt.title("UQ-diff vs UQ")
        plt.xlabel("UQ-diff Value")
        plt.ylabel("UQ Value")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"./{self.config.view_savedir}/UQ-diff-UQ-parity.png", dpi=self.config.fig_dpi)
        plt.close()
        
        # UQ-diff vs True Force diff散点图
        self.logger.info("Plotting and saving the figures of UQ-diff vs force diff")
        plt.figure(figsize=(8, 6))
        plt.scatter(uq_diff, diff_maxf_0_frame, color="blue", label="UQ-diff-force", s=20)
        plt.title("UQ-diff vs Force Diff")
        plt.xlabel("UQ-diff Value")
        plt.ylabel("True Max Force Diff")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"./{self.config.view_savedir}/UQ-diff-fdiff-parity.png", dpi=self.config.fig_dpi)
        plt.close()
        
    def plot_candidate_uq_scatter(self, uq_qbc_for, uq_rnd_for_rescaled, diff_maxf_0_frame, df_uq_desc_candidate):
        """绘制候选结构的UQ散点图"""
        # UQ-QbC候选结构力差异散点图
        self.logger.info("Plotting and saving the figure of UQ-Candidate in QbC space against Max Force Diff")
        plt.figure(figsize=(8, 6))
        plt.scatter(uq_qbc_for, diff_maxf_0_frame, color="blue", label="UQ-QbC", s=20)
        plt.scatter(df_uq_desc_candidate["uq_qbc_for"], df_uq_desc_candidate["diff_maxf_0_frame"], 
                   color="orange", label="Candidate", s=20)
        plt.title("UQ-Candidate in QbC space vs Force Diff")
        plt.xlabel("UQ-QbC Value")
        plt.ylabel("True Max Force Diff")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"./{self.config.view_savedir}/UQ-QbC-Candidate-fdiff-parity.png", dpi=self.config.fig_dpi)
        plt.close()
        
        # UQ-RND候选结构力差异散点图
        self.logger.info("Plotting and saving the figure of UQ-Candidate in RND-rescaled space against Max Force Diff")
        plt.figure(figsize=(8, 6))
        plt.scatter(uq_rnd_for_rescaled, diff_maxf_0_frame, color="blue", label="UQ-RND-rescaled", s=20)
        plt.scatter(df_uq_desc_candidate["uq_rnd_for_rescaled"], df_uq_desc_candidate["diff_maxf_0_frame"], 
                   color="orange", label="Candidate", s=20)
        plt.title("UQ-Candidate in RND-rescaled space vs Force Diff")
        plt.xlabel("UQ-RND-rescaled Value")
        plt.ylabel("True Max Force Diff")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"./{self.config.view_savedir}/UQ-RND-Candidate-fdiff-parity.png", dpi=self.config.fig_dpi)
        plt.close()


    def plot_uq_qbc_rnd_identity_scatter(self, df_uq, df_uq_desc_candidate, df_uq_accurate, df_uq_failed,
                                      uq_select_scheme: str, uq_x_hi: float, uq_y_hi: float, uq_x_lo: float, uq_y_lo: float):
        """绘制UQ-identity散点图"""
        
        # UQ-identity散点图
        self.logger.info("Plotting and saving the figure of UQ-identity in QbC-RND 2D space")
        plt.figure(figsize=(8, 6))
        candidate_count = len(df_uq_desc_candidate)
        accurate_count = len(df_uq_accurate)
        failed_count = len(df_uq_failed)
        # 绘制散点图，创建候选结构、准确结构和失败结构的数据框
        sns.scatterplot(data=df_uq, 
                        x="uq_qbc_for", 
                        y="uq_rnd_for_rescaled", 
                        hue="uq_identity", 
                        palette={"candidate": "orange", 
                                "accurate": "green", 
                                "failed": "red"},
                        alpha=0.5,
                        s=60)
        
        # 自定义图例标签，包含计数信息
        legend_labels = {
            "candidate": f"candidate: {candidate_count}",
            "accurate": f"accurate: {accurate_count}",
            "failed": f"failed: {failed_count}"
        }
        
        # 获取当前图例的句柄和标签
        handles, _ = plt.gca().get_legend_handles_labels()
        # 创建新的图例标签
        new_labels = [legend_labels.get(label, label) for label in ["candidate", "accurate", "failed"]]
        # 添加图例
        plt.legend(handles, new_labels, title="UQ-Identity", fontsize=10)
        plt.title("UQ QbC+RND Selection View", fontsize=14)
        plt.grid(True)
        plt.xlabel("UQ-QbC Value", fontsize=12)
        plt.ylabel("UQ-RND-rescaled Value", fontsize=12)
        
        # 设置坐标轴刻度
        ax = plt.gca()
        x_major_locator = mtick.MultipleLocator(0.1)
        y_major_locator = mtick.MultipleLocator(0.1)
        ax.xaxis.set_major_locator(x_major_locator)
        ax.yaxis.set_major_locator(y_major_locator)
        
        # plot the UQ region of trust inside the plot
        self._plot_uq_region(plt, uq_select_scheme, uq_x_lo, uq_x_hi, uq_y_lo, uq_y_hi)
        
        plt.savefig(f"./{self.config.view_savedir}/UQ-force-qbc-rnd-identity-scatter.png", dpi=self.config.fig_dpi)
        plt.close()
        

    def plot_uq_qbc_rnd_maxfdiff_scatter(self, df_uq, diff_maxf_0_frame, 
                                      uq_select_scheme: str, uq_x_hi: float, uq_y_hi: float, uq_x_lo: float, uq_y_lo: float):
        """绘制UQ-qbc-force和UQ-rnd-force-rescaled vs max force diff的散点图"""
        self.logger.info("Plotting and saving the figures of UQ-qbc-force and UQ-rnd-force-rescaled vs max force diff")
        # 创建包含UQ-QbC、UQ-RND-rescaled和Max Force Diff的数据框
        df_uq_maxfor = pd.DataFrame({"UQ-QbC": df_uq["uq_qbc_for"], "UQ-RND-rescaled": df_uq["uq_rnd_for_rescaled"], "Max Force Diff": diff_maxf_0_frame})
        
        # 绘制散点图
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df_uq_maxfor, 
                        x="UQ-QbC", 
                        y="UQ-RND-rescaled", 
                        hue="Max Force Diff", 
                        palette="Reds",
                        alpha=0.8,
                        s=60)
        plt.title("UQ-QbC and UQ-RND vs Max Force Diff", fontsize=14)
        plt.xlabel("UQ-QbC Value", fontsize=12)
        plt.ylabel("UQ-RND-rescaled Value", fontsize=12)
        plt.legend(title="Max Force Diff", fontsize=10)
        plt.grid(True)
        
        # 设置坐标轴刻度
        ax = plt.gca()
        x_major_locator = mtick.MultipleLocator(0.1)
        y_major_locator = mtick.MultipleLocator(0.1)
        ax.xaxis.set_major_locator(x_major_locator)
        ax.yaxis.set_major_locator(y_major_locator)
        
        # 绘制UQ信任区域边界
        self._plot_uq_region(plt, uq_select_scheme, uq_x_lo, uq_x_hi, uq_y_lo, uq_y_hi)
        
        plt.savefig(f"./{self.config.view_savedir}/UQ-force-qbc-rnd-fdiff-scatter.png", dpi=self.config.fig_dpi)
        plt.close()


class UQSelector:
    """UQ数据选择器"""
    
    def __init__(self, config: UQConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def create_dataframes(self, dp_test_results_0, uq_qbc_for, uq_rnd_for_rescaled, uq_rnd_for, diff_maxf_0_frame, desc_stru, desc_datanames):
        """create Pandas DataFrames for maintaining table including dataname, UQ ,force diff and descriptor"""
        self.logger.info("Collecting data to dataframe and do UQ selection")
        datanames_ind_list = [f"{i[0]}-{i[1]}" for i in dp_test_results_0.dataname_list]
        data_dict_uq = {"dataname": datanames_ind_list, 
                     "uq_qbc_for": uq_qbc_for, 
                     "uq_rnd_for_rescaled": uq_rnd_for_rescaled,
                     "uq_rnd_for": uq_rnd_for,
                     "diff_maxf_0_frame": diff_maxf_0_frame,
                     }
        df_uq = pd.DataFrame(data_dict_uq)
        
        df_desc = pd.DataFrame(desc_stru, 
                               columns=[f"desc_stru_{i}" for i in range(desc_stru.shape[1])])
        # normalize the descriptor to modulo=1
        
        df_desc["dataname"] = desc_datanames
        df_uq_desc = pd.merge(df_uq, df_desc, on="dataname")
        # save the complete df_uq_desc dataframe
        self.logger.info(f"Save df_uq_desc dataframe to {self.config.df_savedir}/df_uq_desc.csv")
        df_uq_desc.to_csv(f"{self.config.df_savedir}/df_uq_desc.csv", index=True)
        return df_uq, df_uq_desc
    
    def select_candidates(self, df_uq, df_uq_desc):
        """Candidates Selection via chosen UQ trust region and UQ select scheme"""
        # simplify variable name
        uq_x = df_uq["uq_qbc_for"]
        uq_y = df_uq["uq_rnd_for_rescaled"]
        uq_x_lo = self.config.uq_qbc_trust_lo
        uq_y_lo = self.config.uq_rnd_rescaled_trust_lo
        uq_x_hi = self.config.uq_qbc_trust_hi
        uq_y_hi = self.config.uq_rnd_rescaled_trust_hi
        
        # uq selection
        if self.config.uq_select_scheme == "strict":
            # strict selection: QbC and RND-like are both trustable
            df_uq_desc_candidate = df_uq_desc[
                (uq_x >= uq_x_lo)
                & (uq_x <= uq_x_hi) 
                & (uq_y >= uq_y_lo) 
                & (uq_y <= uq_y_hi)
            ]
            df_uq_accurate = df_uq[
                ((uq_x < uq_x_lo) 
                 & (uq_y < uq_y_hi)) 
                | ((uq_x< uq_x_hi) 
                 & (uq_y < uq_y_lo))
            ]
            df_uq_failed = df_uq[
                (uq_x > uq_x_hi)
                | (uq_y > uq_y_hi) 
            ]
        elif self.config.uq_select_scheme == "circle_lo":
            # balance selection: QbC and RND-like are trustable in a circle balance way
            df_uq_desc_candidate = df_uq_desc[
                ((uq_x <= uq_x_hi) & (uq_y <= uq_y_hi)) &
                ((uq_x-uq_x_hi)** 2 + (uq_y-uq_y_hi)**2 
                   <= (uq_x_hi - uq_x_lo)**2 + (uq_y_hi - uq_y_lo)**2)
            ]
            df_uq_accurate = df_uq[
                ((uq_x-uq_x_hi)** 2 + (uq_y-uq_y_hi)**2 
                   > (uq_x_lo - uq_x_hi)**2 + (uq_y_lo - uq_y_hi)**2) 
                & ((uq_x < uq_x_hi) & (uq_y < uq_y_hi))
                ]
            df_uq_failed = df_uq[
                (uq_x > uq_x_hi) |
                (uq_y > uq_y_hi) 
            ]
        elif self.config.uq_select_scheme == "tangent_lo":
            # balance selection: QbC and RND-like are trustable in a tangent-circle balance way
            df_uq_desc_candidate = df_uq_desc[
                ((uq_x <= uq_x_hi) & (uq_y <= uq_y_hi)) &
                ((uq_x-uq_x_lo)*(uq_x_lo-uq_x_hi) 
                    + (uq_y-uq_y_lo)*(uq_y_lo-uq_y_hi) <= 0)
            ]
            df_uq_accurate = df_uq[
                ((uq_x-uq_x_lo)*(uq_x_lo-uq_x_hi) 
                    + (uq_y-uq_y_lo)*(uq_y_lo-uq_y_hi) > 0) 
                & ((uq_x < uq_x_hi) & (uq_y < uq_y_hi)) 
                ]
            df_uq_failed = df_uq[
                (uq_x > uq_x_hi) |
                (uq_y > uq_y_hi)
            ]
        elif self.config.uq_select_scheme == "crossline_lo":
            # balance selection: QbC and RND-like are trustable in a croseline balance way
            df_uq_desc_candidate = df_uq_desc[
                (uq_x <= uq_x_hi) & 
                (uq_y <= uq_y_hi) &
                (uq_x_lo * uq_y + (uq_y_hi - uq_y_lo) * uq_x 
                    >= uq_x_lo * uq_y_hi) &
                (uq_x * uq_y_lo + (uq_x_hi - uq_x_lo) * uq_y 
                    >= uq_x_hi * uq_y_lo)
            ]
            df_uq_accurate = df_uq[
                (uq_x_lo * uq_y + (uq_y_hi - uq_y_lo) * uq_x 
                    < uq_x_lo * uq_y_hi) |
                (uq_x * uq_y_lo + (uq_x_hi - uq_x_lo) * uq_y 
                    < uq_x_hi * uq_y_lo)
            ]
            df_uq_failed = df_uq[
                (uq_x > uq_x_hi) |
                (uq_y > uq_y_hi) 
            ]
        elif self.config.uq_select_scheme == "loose":
            # loose selection: QbC or RND-like is either trustable
            df_uq_desc_candidate = df_uq_desc[
                ((uq_x >= uq_x_lo)
                & (uq_x <= uq_x_hi)) 
                | ((uq_y >= uq_y_lo) 
                & (uq_y <= uq_y_hi))
            ]
            df_uq_accurate = df_uq[
                (uq_x < uq_x_lo) &
                (uq_y < uq_y_lo)
            ]
            df_uq_failed = df_uq[
                (uq_x > uq_x_hi) |
                (uq_y > uq_y_hi) 
            ]
        else:
            raise ValueError(f"UQ selection scheme {self.config.uq_select_scheme} not supported!")
        
        # 添加UQ选择信息的日志输出
        self.logger.info(f"UQ scheme: {self.config.uq_select_scheme} between QbC and RND-like")
        self.logger.info(f"UQ selection information : {self.config.uq_select_scheme}")
        self.logger.info(f"Total number of structures: {len(df_uq_desc)}")
        self.logger.info(f"Accurate structures: {len(df_uq_accurate)}, Precentage: {len(df_uq_accurate) / len(df_uq_desc) * 100:.2f}%")
        self.logger.info(f"Candidate structures: {len(df_uq_desc_candidate)}, Precentage: {len(df_uq_desc_candidate) / len(df_uq_desc) * 100:.2f}%")
        self.logger.info(f"Failed structures: {len(df_uq_failed)}, Precentage: {len(df_uq_failed) / len(df_uq_desc) * 100:.2f}%")
        
        # store the selection information in df_uq DataFrame
        df_uq['uq_identity'] = np.where(df_uq['dataname'].isin(df_uq_desc_candidate['dataname']), 'candidate',
                                        np.where(df_uq['dataname'].isin(df_uq_accurate['dataname']), 'accurate', 'failed'))
        
        # save the DataFrame
        self.logger.info(f"Save df_uq dataframe to {self.config.df_savedir}/df_uq.csv after UQ selection and identication")
        df_uq.to_csv(f"{self.config.df_savedir}/df_uq.csv", index=True)
        self.logger.info(f"Save df_uq_desc_candidate dataframe to {self.config.df_savedir}/df_uq_desc_sampled-UQ.csv")
        df_uq_desc_candidate.to_csv(f"{self.config.df_savedir}/df_uq_desc_sampled-UQ.csv", index=True)
        
        return df_uq, df_uq_desc_candidate, df_uq_accurate, df_uq_failed


class DIRECTSamplerWrapper:
    """DIRECT采样器包装器"""
    
    def __init__(self, config: UQConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def sample(self, df_uq_desc_candidate):
        """使用DIRECT算法进行采样"""
        self.logger.info(f"Doing DIRECT Selection on UQ-selected data")
        DIRECT_sampler = DIRECTSampler(
            structure_encoder=None,
            clustering=BirchClustering(
                n=self.config.num_selection // self.config.direct_k, 
                threshold_init=self.config.direct_thr_init),
            select_k_from_clusters=SelectKFromClusters(k=self.config.direct_k),
        )
        desc_features = [f"desc_stru_{i}" for i in range(len([col for col in df_uq_desc_candidate.columns if col.startswith("desc_stru_")]))]
        DIRECT_selection = DIRECT_sampler.fit_transform(df_uq_desc_candidate[desc_features].values)
        DIRECT_selected_indices = DIRECT_selection["selected_indices"]
        explained_variance = DIRECT_sampler.pca.pca.explained_variance_
        selected_PC_dim = len([e for e in explained_variance if e > 1])
        DIRECT_selection["PCAfeatures_unweighted"] = DIRECT_selection["PCAfeatures"] / explained_variance[:selected_PC_dim]
        all_features = DIRECT_selection["PCAfeatures_unweighted"]
        
        df_uq_desc_selected_final = df_uq_desc_candidate.iloc[DIRECT_selected_indices]
        return DIRECT_sampler, DIRECT_selection, DIRECT_selected_indices, all_features, df_uq_desc_selected_final
    
    def save_sampled_DataFrame(self, df_uq_desc_selected_final):
        """Save the sampled DataFrame"""
        self.logger.info(f"Saving df_uq_desc_selected_final dataframe to {self.config.df_savedir}/df_uq_desc_sampled-final.csv")
        df_uq_desc_selected_final.to_csv(f"{self.config.df_savedir}/df_uq_desc_sampled-final.csv", index=True)
    
    def visualize_direct_results(self, DIRECT_sampler, DIRECT_selection, DIRECT_selected_indices, all_features):
        """Visualize the results of DIRECT selection"""
        explained_variance = DIRECT_sampler.pca.pca.explained_variance_
        selected_PC_dim = len([e for e in explained_variance if e > 1])
        
        # Explained Variance
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
        plt.savefig(f"{self.config.view_savedir}/explained_variance.png", dpi=self.config.fig_dpi)
        plt.close()
        
        # PCA feature coverage
        self._plot_PCAfeature_coverage(all_features, DIRECT_selected_indices, "DIRECT", self.config.fig_dpi, self.config.view_savedir)
        
        # Simulate a manual selection by random
        np.random.seed(42)
        manual_selection_index = np.random.choice(len(all_features), 
                                                  self.config.num_selection, 
                                                  replace=False)
        self._plot_PCAfeature_coverage(all_features, manual_selection_index, "Random", self.config.fig_dpi, self.config.view_savedir)
        
        # Feature coverage score comparison
        scores_DIRECT = self._calculate_all_FCS(all_features, DIRECT_selection["selected_indices"], self.config.n_bins)
        scores_MS = self._calculate_all_FCS(all_features, manual_selection_index, self.config.n_bins)
        
        # Plot the feature coverage score comparison
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
    
    def _plot_PCAfeature_coverage(self, all_features, selected_indices, method="DIRECT", dpi=150, savedir='.'):
        """Draw the PCA feature coverage plot"""
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
        plt.savefig(f"{savedir}/{method}_PCA_feature_coverage.png", dpi=dpi)
        plt.close()
    
    def _calculate_feature_coverage_score(self, all_features, selected_indices, n_bins=100):
        """计算特征覆盖分数"""
        selected_features = all_features[selected_indices]
        n_all = np.count_nonzero(
            np.histogram(all_features, bins=np.linspace(all_features.min(), all_features.max(), n_bins))[0]
        )
        n_select = np.count_nonzero(
            np.histogram(selected_features, bins=np.linspace(all_features.min(), all_features.max(), n_bins))[0]
        )
        return n_select / n_all
    
    def _calculate_all_FCS(self, all_features, selected_indices, b_bins=100):
        """计算所有特征的覆盖分数"""
        select_scores = [
            self._calculate_feature_coverage_score(all_features[:, i], selected_indices, n_bins=b_bins)
            for i in range(all_features.shape[1])
        ]
        return select_scores


class UQPostProcessor:
    """UQ后处理主类"""
    
    def __init__(self, project="."):
        # 初始化各个模块
        self.config = UQConfig(project)
        self.logger = logging.getLogger(__name__)
        self.data_loader = UQDataLoader(self.config)
        self.processor = UQCalculator(self.config)
        self.visualizer = UQVisualizer(self.config)
        self.selector = UQSelector(self.config)
        self.sampler = DIRECTSamplerWrapper(self.config)
        self._check_directories()
        
    
    def _check_directories(self):
        """检查和创建必要的目录"""
        self.logger.info(f"Initializing selection in {self.config.project} ---")
        if os.path.exists(self.config.project) == False:
            self.logger.error(f"Project directory {self.config.project} not found!")
            raise ValueError(f"Project directory {self.config.project} not found!")
        if os.path.exists(f"{self.config.project}/0/{self.config.testing_dir}") == False:
            self.logger.error(f"Testing directory {self.config.testing_dir} not found!")
            raise ValueError(f"Testing directory {self.config.testing_dir} not found!")
        if os.path.exists(self.config.desc_dir) == False:
            self.logger.error(f"Descriptor directory {self.config.desc_dir} not found!")
            raise ValueError(f"Descriptor directory {self.config.desc_dir} not found!")
        if os.path.exists(self.config.testdata_dir) == False:
            self.logger.error(f"Testdata directory {self.config.testdata_dir} not found!")
            raise ValueError(f"Testdata directory {self.config.testdata_dir} not found!")
        uq_options = ["strict", 
                      "circle_lo", 
                      "crossline_lo", 
                      "tangent_lo",
                      "loose"]
        if self.config.uq_select_scheme not in uq_options:
            self.logger.error(f"UQ selection scheme {self.config.uq_select_scheme} not supported! Please choose from {uq_options}.")
            raise ValueError(f"UQ selection scheme {self.config.uq_select_scheme} not supported! Please choose from {uq_options}.")
        if ((self.config.uq_qbc_trust_lo >= self.config.uq_qbc_trust_hi) 
            or (self.config.uq_rnd_rescaled_trust_lo >= self.config.uq_rnd_rescaled_trust_hi)):
            raise ValueError("Low trust threshold should be lower than High trust threshold !")

        if os.path.exists(self.config.view_savedir) == False:
            os.makedirs(self.config.view_savedir)
        if os.path.exists(self.config.dpdata_savedir) == False:
            os.makedirs(self.config.dpdata_savedir)
        if os.path.exists(self.config.df_savedir) == False:
            os.makedirs(self.config.df_savedir)
    
    def run_workflow(self):
        """运行完整的UQ后处理工作流"""
        # 1. 数据加载
        dp_test_results_list = self.data_loader.load_test_results()
        test_data = self.data_loader.load_test_data()
        desc_stru, desc_datanames = self.data_loader.load_descriptors()
        
        # 2. 数据处理
        diff_maxf_0_frame, diff_rmsf_0_frame = self.processor.get_force_prediction_differences(dp_test_results_list[0])
        atomic_forces = self.processor.get_atomic_forces(dp_test_results_list)
        f_stddiff_qbc = self.processor.calculate_force_stddiff_qbc(atomic_forces)
        f_stddiff_rnd = self.processor.calculate_force_stddiff_rnd(atomic_forces)
        uq_qbc_for, uq_rnd_for = self.processor.assign_uq_to_structures(f_stddiff_qbc, f_stddiff_rnd, dp_test_results_list[0].dataname_list)
        uq_rnd_for_rescaled = self.processor.align_uq_values(uq_qbc_for, uq_rnd_for)
        
        # 3. 可视化
        self.visualizer.plot_uq_force_distribution(uq_qbc_for, uq_rnd_for, uq_rnd_for_rescaled, diff_maxf_0_frame)
        self.visualizer.plot_uq_difference(uq_qbc_for, uq_rnd_for_rescaled, diff_maxf_0_frame)
        
        # 4. 数据选择
        # 进一步重构方向：在df_uq出来之后，所有传值全部用df进行，不再单独传df里面有的值。
        df_uq, df_uq_desc = self.selector.create_dataframes(
            dp_test_results_list[0], uq_qbc_for, uq_rnd_for_rescaled, uq_rnd_for, diff_maxf_0_frame, desc_stru, desc_datanames)
        df_uq, df_uq_desc_candidate, df_uq_accurate, df_uq_failed = self.selector.select_candidates(df_uq, df_uq_desc)
        
        # 4.1 候选结构可视化
        self.visualizer.plot_candidate_uq_scatter(uq_qbc_for, uq_rnd_for_rescaled, diff_maxf_0_frame, df_uq_desc_candidate)
        # 4.2 UQ的UQ-Identity和UQ-max-force散点图可视化
        self.visualizer.plot_uq_qbc_rnd_identity_scatter(df_uq, df_uq_desc_candidate, df_uq_accurate, df_uq_failed,
                                                        self.config.uq_select_scheme,
                                                        self.config.uq_qbc_trust_hi, self.config.uq_rnd_rescaled_trust_hi,
                                                        self.config.uq_qbc_trust_lo, self.config.uq_rnd_rescaled_trust_lo)
        self.visualizer.plot_uq_qbc_rnd_maxfdiff_scatter(df_uq, diff_maxf_0_frame, 
                                                        self.config.uq_select_scheme,
                                                        self.config.uq_qbc_trust_hi, self.config.uq_rnd_rescaled_trust_hi,
                                                        self.config.uq_qbc_trust_lo, self.config.uq_rnd_rescaled_trust_lo)
        # 5. DIRECT采样
        DIRECT_sampler, DIRECT_selection, DIRECT_selected_indices, all_features, df_uq_desc_selected_final = self.sampler.sample(df_uq_desc_candidate)
        
        # 6. 保存采样的数据框
        self.sampler.save_sampled_DataFrame(df_uq_desc_selected_final)
        
        # 7. 可视化DIRECT结果
        self.sampler.visualize_direct_results(DIRECT_sampler, DIRECT_selection, DIRECT_selected_indices, all_features)
        
        # 8. 最终选择结果在PCA空间中的可视化
        self._visualize_final_selection_in_pca_space(df_uq, df_uq_desc, desc_stru, df_uq_desc_candidate, df_uq_desc_selected_final)
        
        # 9. 基于采样结果从dpdata数据类中取样并保存sampled_dpdata和other_dpdata
        self._sample_and_save_dpdata(test_data, df_uq_desc, df_uq_desc_selected_final)
        
        self.logger.info("UQ Post-processing workflow completed!")
    
    def _visualize_final_selection_in_pca_space(self, df_uq, df_uq_desc, desc_stru, df_uq_desc_candidate, df_uq_desc_selected_final):
        """在PCA空间中可视化最终选择结果"""
        desc_features = [f"desc_stru_{i}" for i in range(desc_stru.shape[1])]
        X = df_uq_desc[desc_features].values
        pca = PCA(n_components=2)
        PCs_alldata = pca.fit_transform(X)
        PCs_df = pd.DataFrame(data=PCs_alldata, columns=['PC1', 'PC2'])
        df_alldataPC_visual = pd.concat([df_uq, PCs_df], axis=1)
        
        # 保存数据框
        df_alldataPC_visual.to_csv(f"{self.config.df_savedir}/final_df.csv", index=True)
        
        # 全局索引
        UQ_selected_indices_global = df_uq_desc_candidate.index
        final_selected_indices_global = df_uq_desc_selected_final.index
        num_all = len(df_uq.index)
        num_selected_UQ = len(UQ_selected_indices_global)
        num_selected_final = len(final_selected_indices_global)
        
        # 绘制PCA视图
        plt.figure(figsize=(10, 8))
        plt.scatter(PCs_alldata[:, 0], 
                    PCs_alldata[:, 1], 
                    marker="*", 
                    color="gray", 
                    label=f"All {num_all} structures", 
                    alpha=0.7,
                    s=15)
        # 绘制仅通过UQ选择的结构
        plt.scatter(PCs_alldata[UQ_selected_indices_global, 0], 
                    PCs_alldata[UQ_selected_indices_global, 1], 
                    marker="*", 
                    color="blue", 
                    label=f"UQ sampled {num_selected_UQ}", 
                    alpha=0.7,
                    s=30)
        # 绘制通过UQ-DIRECT选择的结构
        plt.scatter(PCs_alldata[final_selected_indices_global, 0], 
                    PCs_alldata[final_selected_indices_global, 1], 
                    marker="*", 
                    color="red", 
                    label=f"UQ-DIRECT sampled {num_selected_final}", 
                    s=30)
        plt.title(f"PCA of UQ-DIRECT sampling", fontsize=14)
        plt.xlabel("PC1", size=12)
        plt.ylabel("PC2", size=12)
        plt.legend(frameon=False, fontsize=12, reverse=True)
        self.logger.info(f"Saving the PCA view of UQ-DIRECT sampling to {self.config.view_savedir}/Final_sampled_PCAview.png")
        plt.savefig(f"{self.config.view_savedir}/Final_sampled_PCAview.png", dpi=self.config.fig_dpi)
    
    def _sample_and_save_dpdata(self, test_data, df_uq_desc, df_uq_desc_selected_final):
        """基于采样结果从dpdata数据类中取样并保存sampled_dpdata和other_dpdata"""
        self.logger.info(f"Sampling dpdata based on selected indices")
        sampled_datanames = df_uq_desc.iloc[df_uq_desc_selected_final.index]['dataname'].to_list()
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
        self.logger.info(f"Dumping sampled and other dpdata to {self.config.dpdata_savedir}")
        
        # 删除现有的dpdata目录
        if os.path.exists(f"{self.config.dpdata_savedir}/sampled_dpdata"):
            shutil.rmtree(f"{self.config.dpdata_savedir}/sampled_dpdata")
        if os.path.exists(f"{self.config.dpdata_savedir}/other_dpdata"):
            shutil.rmtree(f"{self.config.dpdata_savedir}/other_dpdata")
        
        # 保存dpdata
        sampled_dpdata.to_deepmd_npy(f"{self.config.dpdata_savedir}/sampled_dpdata")
        other_dpdata.to_deepmd_npy(f"{self.config.dpdata_savedir}/other_dpdata")


# 主函数
if __name__ == "__main__":
    # 创建UQ后处理器实例
    uq_processor = UQPostProcessor(".")
    
    # 运行工作流
    uq_processor.run_workflow()