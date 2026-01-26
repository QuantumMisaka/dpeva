"""Module for handling DeepMD test results."""

from copy import deepcopy
import numpy as np


class DPTestResults:
    """Class to load and manage DeepMD test results.
    
    This class handles the loading of test results from DeepMD models and provides
    methods to access energy, force data and structure information. The results from DeepMD test
    must be calculated first and loaded via this class.
    
    Args:
        type_map (list): List of atom types
        data_e (np.ndarray): Energy data from test results
        data_f (np.ndarray): Force data from test results
        dataname_list (list): List of [dataname, frame_index, natom] for each structure
        datanames_nframe (dict): Dictionary mapping dataname to number of frames
        diff_e (np.ndarray): Energy differences between predicted and actual values
        diff_fx (np.ndarray): Force x-component differences
        diff_fy (np.ndarray): Force y-component differences
        diff_fz (np.ndarray): Force z-component differences
    """
    
    def __init__(self, headname, type_map=None):
        self.type_map = type_map if type_map else ["H","C","O","Fe"]
        self.get_dptest_detail(headname)
    
    def get_natom(self, dataname):
        """Get natoms from dataname.
        
        Args:
            dataname: Name of the data sample
            
        Returns:
            int: Number of atoms in the sample
        """
        natom = 0
        name_string = deepcopy(dataname)
        for ele in self.type_map:
            name_string = name_string.replace(ele, f" {ele},")
        ele_num_pair_list = name_string.strip().split(" ")
        for ind, ele_string in enumerate(ele_num_pair_list):
            natom += int(ele_string.split(',')[1])
        return natom
    
    def get_dataname(self, filename):
        """Read dataname and nframe from test result file.
        
        Args:
            filename: Path to the test result file
            
        Returns:
            tuple: (datanames_nframe_list, datanames_nframe_dict)
        """
        # 从对应的test result*.txt文件中读取每个数据对应的LabeledSystem以及nframe, 目前能读取deepmd/npy数据测试后结果
        datanames_indice_dict = {}
        datanames_nframe_list = []
        datanames_nframe_dict = {}
        # read the "# DataName" line from the file and its index of columns

        with open(filename, 'r') as f:
            for i, line in enumerate(f):
                if line.startswith("# "):
                    dirname = line.split(" ")[1]
                    dataname = dirname.split("/")[-1][:-1]
                    datanames_indice_dict[dataname] = i
        full_index = i
        len_data = len(datanames_indice_dict)
        for count, dataname in enumerate(datanames_indice_dict):
            if count == len_data - 1:
                datanames_nframe_dict[dataname] = full_index - datanames_indice_dict[dataname]
            else:
                list_indice = list(datanames_indice_dict.values())
                datanames_nframe_dict[dataname] = list_indice[count + 1] - list_indice[count] - 1
        # flatten the datanames_nframe
        for dataname, count in datanames_nframe_dict.items():
            for i in range(count):
                natom = self.get_natom(dataname)
                datanames_nframe_list.append([dataname, i, natom])
        return datanames_nframe_list, datanames_nframe_dict

    def get_dptest_detail(self, headname):
        """Read test results data from test result file
        
        Args:
            headname: Base name for the test result files
        """
        self.data_e = np.genfromtxt(f"{headname}.e_peratom.out", names=[f"data_Energy", f"pred_Energy"])
        self.data_f = np.genfromtxt(f"{headname}.f.out", names=["data_fx", "data_fy", "data_fz", "pred_fx", "pred_fy", "pred_fz"])
        self.dataname_list, self.datanames_nframe = self.get_dataname(f"{headname}.e_peratom.out")

        # Check if ground truth exists (if all data values are 0, assume it doesn't exist)
        is_force_zero = np.all(self.data_f['data_fx'] == 0) and \
                        np.all(self.data_f['data_fy'] == 0) and \
                        np.all(self.data_f['data_fz'] == 0)
        is_energy_zero = np.all(self.data_e['data_Energy'] == 0)
        
        if is_force_zero and is_energy_zero:
            self.has_ground_truth = False
            self.diff_e = None
            self.diff_fx = None
            self.diff_fy = None
            self.diff_fz = None
        else:
            self.has_ground_truth = True
            # 计算pred和data的差值
            self.diff_e = self.data_e[f"pred_Energy"] - self.data_e[f"data_Energy"]
            self.diff_fx = self.data_f[f'pred_fx'] - self.data_f[f'data_fx']
            self.diff_fy = self.data_f[f'pred_fy'] - self.data_f[f'data_fy']
            self.diff_fz = self.data_f[f'pred_fz'] - self.data_f[f'data_fz']