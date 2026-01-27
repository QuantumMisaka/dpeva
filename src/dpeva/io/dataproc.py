"""Module for handling DeepMD test results."""

import re
from collections import Counter
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union
import logging
import numpy as np
import os

class TestResultParser:
    """
    Parses the output results from `dp test` command.
    Handles both energy, force, and virial outputs, and detects if ground truth is available.
    """
    
    def __init__(self, result_dir: str, head: str = "results", type_map: List[str] = None):
        """
        Initialize the parser.
        
        Args:
            result_dir (str): Directory containing the test results (e.g. *.e.out files).
            head (str): The head name used in dp test output files (e.g. "results").
            type_map (list): List of atom types. If None, defaults to ["H", "C", "O", "Fe"].
        """
        self.result_dir = result_dir
        self.head = head
        self.type_map = type_map if type_map else ["H", "C", "O", "Fe"]
        self.logger = logging.getLogger(__name__)
        
        self.data_e: Optional[np.ndarray] = None
        self.data_f: Optional[np.ndarray] = None
        self.data_v: Optional[np.ndarray] = None
        
        self.has_ground_truth = False
        self.parsed_data = {}

    def parse(self) -> Dict:
        """
        Parse the result files in the directory.
        
        Returns:
            dict: A dictionary containing parsed data arrays and metadata.
        """
        # File paths based on dp test output convention
        e_file = os.path.join(self.result_dir, f"{self.head}.e_peratom.out")
        f_file = os.path.join(self.result_dir, f"{self.head}.f.out")
        v_file = os.path.join(self.result_dir, f"{self.head}.v.out")
        
        if not os.path.exists(e_file):
            self.logger.error(f"Energy file not found: {e_file}")
            raise FileNotFoundError(f"Energy file not found: {e_file}")

        self.logger.info(f"Parsing test results from {self.result_dir}")
        
        # Load Energy
        # Format: data_E pred_E
        try:
            self.data_e = np.genfromtxt(e_file, names=["data_e", "pred_e"])
        except Exception as e:
            self.logger.error(f"Failed to parse energy file: {e}")
            raise

        # Load Forces
        if os.path.exists(f_file):
            # Format: data_fx data_fy data_fz pred_fx pred_fy pred_fz
            self.data_f = np.genfromtxt(f_file, names=["data_fx", "data_fy", "data_fz", "pred_fx", "pred_fy", "pred_fz"])
        else:
            self.logger.warning(f"Force file not found: {f_file}")
            self.data_f = None
            
        # Load Virial (Optional)
        # Prioritize v_peratom.out as user requested, else fallback to v.out
        vp_file = os.path.join(self.result_dir, f"{self.head}.v_peratom.out")
        
        if os.path.exists(vp_file):
            try:
                # 9 for data, 9 for pred
                names = [f"data_v{i}" for i in range(9)] + [f"pred_v{i}" for i in range(9)]
                self.data_v = np.genfromtxt(vp_file, names=names)
                self.logger.info(f"Loaded virial from {vp_file}")
            except Exception as e:
                self.logger.warning(f"Failed to parse v_peratom file: {e}")
                self.data_v = None
        elif os.path.exists(v_file):
            try:
                # 9 for data, 9 for pred
                names = [f"data_v{i}" for i in range(9)] + [f"pred_v{i}" for i in range(9)]
                self.data_v = np.genfromtxt(v_file, names=names)
                self.logger.info(f"Loaded virial from {v_file}")
            except Exception as e:
                self.logger.warning(f"Failed to parse virial file: {e}")
                self.data_v = None
        else:
            self.data_v = None

        # Parse Data Names and Frame Info from Energy file comments
        dataname_list, datanames_nframe = self._get_dataname_info(e_file)
        
        # Check Ground Truth
        self._check_ground_truth()
        
        self.parsed_data = {
            "energy": self.data_e,
            "force": self.data_f,
            "virial": self.data_v,
            "has_ground_truth": self.has_ground_truth,
            "dataname_list": dataname_list,
            "datanames_nframe": datanames_nframe
        }
        
        return self.parsed_data

    def get_composition_list(self) -> Tuple[List[Dict[str, int]], List[int]]:
        """
        Reconstruct atom counts list and atom number list from parsed dataname_list.
        Requires dataname_list to be populated (call parse() first).
        
        Returns:
            Tuple[List[Dict[str, int]], List[int]]: 
                - atom_counts_list: List of atom counts dict for each frame.
                - atom_num_list: List of total atom numbers for each frame.
        """
        dataname_list = self.parsed_data.get("dataname_list", [])
        if not dataname_list:
            self.logger.warning("dataname_list is empty. Did you call parse()?")
            return [], []
            
        atom_counts_list = []
        atom_num_list = []
        
        # Sort type_map by length descending to ensure correct regex matching (e.g. Fe before F)
        sorted_types = sorted(self.type_map, key=len, reverse=True)
        pattern = f"({'|'.join(sorted_types)})(\d+)"
        
        for item in dataname_list:
            dataname = item[0]
            counts = Counter()
            matches = re.findall(pattern, dataname)
            
            for elem, count_str in matches:
                counts[elem] += int(count_str)
                
            n_atoms = sum(counts.values())
            
            atom_counts_list.append(counts)
            atom_num_list.append(n_atoms)
            
        return atom_counts_list, atom_num_list

    def _check_ground_truth(self):
        """
        Check if the data columns contain actual ground truth or are just placeholders (zeros).
        """
        # Heuristic: If all energy and force data are exactly zero, likely no ground truth.
        is_e_zero = np.all(np.abs(self.data_e['data_e']) < 1e-12)
        
        is_f_zero = True
        if self.data_f is not None:
            is_f_zero = np.all(np.abs(self.data_f['data_fx']) < 1e-12) and \
                        np.all(np.abs(self.data_f['data_fy']) < 1e-12) and \
                        np.all(np.abs(self.data_f['data_fz']) < 1e-12)
                        
        if is_e_zero and is_f_zero:
            self.has_ground_truth = False
            self.logger.info("Detected all-zero data columns. Assuming NO ground truth.")
        else:
            self.has_ground_truth = True
            self.logger.info("Detected non-zero data columns. Assuming ground truth exists.")

    def _get_dataname_info(self, filename: str) -> Tuple[List, Dict]:
        """
        Extract system names and frame counts from the comment lines of the output file.
        """
        datanames_indice_dict = {}
        datanames_nframe_list = []
        datanames_nframe_dict = {}
        
        with open(filename, 'r') as f:
            lines = f.readlines()
            
        # Parse indices
        for i, line in enumerate(lines):
            if line.startswith("# "):
                parts = line.strip().split(" ")
                if len(parts) >= 2:
                    dirname = parts[1]
                    dataname = os.path.basename(dirname.rstrip('/'))
                    datanames_indice_dict[dataname] = i

        full_index = len(lines)
        sorted_indices = sorted(datanames_indice_dict.items(), key=lambda x: x[1])
        
        for idx, (dataname, start_line) in enumerate(sorted_indices):
            if idx == len(sorted_indices) - 1:
                end_line = full_index
            else:
                end_line = sorted_indices[idx+1][1]
            
            n_frames = end_line - start_line - 1
            datanames_nframe_dict[dataname] = n_frames
            
            natom = self._get_natom_from_name(dataname)
            
            for i in range(n_frames):
                datanames_nframe_list.append([dataname, i, natom])
                
        return datanames_nframe_list, datanames_nframe_dict

    def _get_natom_from_name(self, dataname: str) -> int:
        """
        Estimate number of atoms from dataname string (heuristic).
        """
        try:
            natom = 0
            name_string = deepcopy(dataname)
            for ele in self.type_map:
                name_string = name_string.replace(ele, f" {ele},")
            ele_num_pair_list = name_string.strip().split(" ")
            for ele_string in ele_num_pair_list:
                if ',' in ele_string:
                    count = ele_string.split(',')[1]
                    if count:
                        natom += int(count)
            return natom if natom > 0 else 1 # Fallback
        except:
            return 1 # Fallback


class DPTestResults:
    """
    Legacy wrapper class to load and manage DeepMD test results.
    Maintained for backward compatibility. Uses TestResultParser internally.
    """
    
    def __init__(self, headname, type_map=None):
        self.type_map = type_map if type_map else ["H","C","O","Fe"]
        # Assuming current directory as original code did
        self.parser = TestResultParser(result_dir=".", head=headname, type_map=self.type_map)
        
        try:
            self.results = self.parser.parse()
            self._map_results()
        except Exception as e:
            logging.error(f"Failed to initialize DPTestResults: {e}")
            raise

    def _map_results(self):
        """Map parsed results to legacy attributes."""
        # Map Energy
        self.data_e = self.results["energy"]
        if self.data_e is not None:
             # Rename fields for compatibility: data_e/pred_e -> data_Energy/pred_Energy
             # Check current dtypes first to avoid errors
             current_dtypes = self.data_e.dtype.descr
             new_dtypes = []
             for name, dtype in current_dtypes:
                 if name == 'data_e':
                     new_dtypes.append(('data_Energy', dtype))
                 elif name == 'pred_e':
                     new_dtypes.append(('pred_Energy', dtype))
                 else:
                     new_dtypes.append((name, dtype))
             
             self.data_e = self.data_e.astype(new_dtypes)

        # Map Forces
        self.data_f = self.results["force"]
        
        self.dataname_list = self.results["dataname_list"]
        self.datanames_nframe = self.results["datanames_nframe"]
        self.has_ground_truth = self.results["has_ground_truth"]
        
        if self.has_ground_truth and self.data_e is not None and self.data_f is not None:
            # Calculate diffs using new field names
            self.diff_e = self.data_e['pred_Energy'] - self.data_e['data_Energy']
            self.diff_fx = self.data_f['pred_fx'] - self.data_f['data_fx']
            self.diff_fy = self.data_f['pred_fy'] - self.data_f['data_fy']
            self.diff_fz = self.data_f['pred_fz'] - self.data_f['data_fz']
        else:
            self.diff_e = None
            self.diff_fx = None
            self.diff_fy = None
            self.diff_fz = None
            
    def get_natom(self, dataname):
        """Proxy to parser's method."""
        return self.parser._get_natom_from_name(dataname)
    
    def get_dataname(self, filename):
        """Legacy method proxy."""
        return self.dataname_list, self.datanames_nframe

    def get_dptest_detail(self, headname):
        """Legacy method, no-op as work is done in init."""
        pass
