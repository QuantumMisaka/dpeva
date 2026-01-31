"""Module for handling DeepMD test results."""

import re
from collections import Counter
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union
import logging
import numpy as np
import os

class DPTestResultParser:
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
        vp_file = os.path.join(self.result_dir, f"{self.head}.v_peratom.out")
        v_file = os.path.join(self.result_dir, f"{self.head}.v.out")
        
        if not os.path.exists(e_file):
            self.logger.error(f"Energy file not found: {e_file}")
            raise FileNotFoundError(f"Energy file not found: {e_file}")

        self.logger.info(f"Parsing test results from {self.result_dir}")
        
        # Load Energy
        # Format: data_E pred_E
        try:
            self.logger.info(f"Loading energy file: {e_file}")
            self.data_e = np.genfromtxt(e_file, names=["data_e", "pred_e"])
        except Exception as e:
            self.logger.error(f"Failed to parse energy file: {e}")
            raise

        # Load Forces
        if os.path.exists(f_file):
            # Format: data_fx data_fy data_fz pred_fx pred_fy pred_fz
            try:
                self.logger.info(f"Loading force file: {f_file}")
                self.data_f = np.genfromtxt(f_file, names=["data_fx", "data_fy", "data_fz", "pred_fx", "pred_fy", "pred_fz"])
            except Exception as e:
                self.logger.error(f"Failed to parse force file: {e}")
                raise
        else:
            self.logger.warning(f"Force file not found: {f_file}")
            self.data_f = None
            
        # Load Virial (Optional)
        # Prioritize v_peratom.out as user requested, else fallback to v.out
        if os.path.exists(vp_file):
            try:
                self.logger.info(f"Loading virial file: {vp_file}")
                # 9 for data, 9 for pred
                names = [f"data_v{i}" for i in range(9)] + [f"pred_v{i}" for i in range(9)]
                self.data_v = np.genfromtxt(vp_file, names=names)
                self.logger.info(f"Loaded virial from {vp_file}")
            except Exception as e:
                self.logger.warning(f"Failed to parse v_peratom file: {e}")
                self.data_v = None
        elif os.path.exists(v_file):
            try:
                self.logger.info(f"Loading virial file: {v_file}")
                # 9 for data, 9 for pred
                names = [f"data_v{i}" for i in range(9)] + [f"pred_v{i}" for i in range(9)]
                self.data_v = np.genfromtxt(v_file, names=names)
                self.logger.info(f"Loaded virial from {v_file}")
            except Exception as e:
                self.logger.warning(f"Failed to parse virial file: {e}")
                self.data_v = None
        else:
            self.data_v = None

        try:
            # Parse Data Names and Frame Info from Energy file comments
            dataname_list, datanames_nframe = self._get_dataname_info(e_file)
        except Exception as e:
            self.logger.error(f"Failed to parse dataname info: {e}")
            raise
        
        try:
            # Check Ground Truth
            self._check_ground_truth()
        except Exception as e:
            self.logger.error(f"Failed to check ground truth: {e}")
            raise
        
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
        pattern = rf"({'|'.join(sorted_types)})(\d+)"
        
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
        with open(filename, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if line.startswith('#'):
                    parts = line.split(':')
                    if len(parts) >= 2:
                        raw_path = parts[0].strip().lstrip('#').strip()
                        # Use heuristic to extract Pool/System if available to avoid name collision
                        path_clean = os.path.normpath(raw_path)
                        path_parts = path_clean.split(os.sep)
                        # Filter out dot but keep .. to handle relative paths correctly
                        path_parts = [p for p in path_parts if p and p != '.']
                        
                        if len(path_parts) >= 2 and path_parts[-2] != '..':
                            # Likely Pool/System structure (e.g. pool/sys or ../pool/sys)
                            dataname = f"{path_parts[-2]}/{path_parts[-1]}"
                        else:
                            # Fallback to basename (e.g. sys or ../sys)
                            dataname = path_parts[-1]
                            
                        datanames_indice_dict[dataname] = i
        
        datanames_nframe_list = []
        datanames_nframe_dict = {}
        
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
