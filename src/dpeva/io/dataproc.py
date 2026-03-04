"""Module for handling DeepMD test results."""

import re
from collections import Counter
from typing import Dict, List, Optional, Tuple
import logging
import numpy as np
import os

class DPTestResultParser:
    """
    Parses the output results from `dp test` command.
    Handles both energy, force, and virial outputs, and detects if ground truth is available.
    """
    
    def __init__(self, result_dir: str, head: str = "results", type_map: List[str] = None, testdata_dir: str = None):
        """
        Initialize the parser.
        
        Args:
            result_dir (str): Directory containing the test results (e.g. *.e.out files).
            head (str): The head name used in dp test output files (e.g. "results").
            type_map (list): List of atom types. If None, defaults to ["H", "C", "O", "Fe"].
            testdata_dir (str): Path to original test data for fallback atom count verification.
        """
        self.result_dir = result_dir
        self.head = head
        self.type_map = type_map if type_map else ["H", "C", "O", "Fe"]
        self.testdata_dir = testdata_dir
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
            # Pass f_file to use structure-based atom counting
            dataname_list, datanames_nframe = self._get_dataname_info(e_file, f_file)
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
        Updates self.has_ground_truth based on heuristic check of zero-values.
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

    def _get_dataname_info(self, e_file: str, f_file: str = None) -> Tuple[List, Dict]:
        """
        Extract system names, frame counts, and atom counts from output files.
        Uses structure-based parsing: natom = N_force_lines / N_frames.

        Args:
            e_file (str): Path to the energy output file (e.g., .e_peratom.out).
            f_file (str): Path to the force output file (e.g., .f.out).

        Returns:
            Tuple[List, Dict]: 
                - datanames_nframe_list: List of [dataname, frame_idx, natom].
                - datanames_nframe_dict: Dict mapping dataname to number of frames.
        """
        def parse_indices(filename):
            indices = {}
            with open(filename, 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    if line.startswith('#'):
                        parts = line.split(':')
                        if len(parts) >= 2:
                            raw_path = parts[0].strip().lstrip('#').strip()
                            path_clean = os.path.normpath(raw_path)
                            path_parts = path_clean.split(os.sep)
                            path_parts = [p for p in path_parts if p and p != '.']
                            
                            if len(path_parts) >= 2 and path_parts[-2] != '..':
                                dataname = f"{path_parts[-2]}/{path_parts[-1]}"
                            else:
                                dataname = path_parts[-1]
                            indices[dataname] = i
            return indices, len(lines)

        e_indices, e_total_lines = parse_indices(e_file)
        
        f_indices = {}
        f_total_lines = 0
        if f_file and os.path.exists(f_file):
            f_indices, f_total_lines = parse_indices(f_file)
        
        datanames_nframe_list = []
        datanames_nframe_dict = {}
        
        sorted_e_indices = sorted(e_indices.items(), key=lambda x: x[1])
        
        for idx, (dataname, e_start) in enumerate(sorted_e_indices):
            # 1. Calculate N_frames
            if idx == len(sorted_e_indices) - 1:
                e_end = e_total_lines
            else:
                e_end = sorted_e_indices[idx+1][1]
            
            n_frames = e_end - e_start - 1
            datanames_nframe_dict[dataname] = n_frames
            
            # 2. Calculate natom
            natom = 1 # Default fallback
            natom_source = "fallback" # Track source for logging
            
            if f_file and dataname in f_indices:
                f_start = f_indices[dataname]
                
                # Find f_end
                # Assuming force file has same system order/keys, but safer to look it up
                sorted_f_indices = sorted(f_indices.items(), key=lambda x: x[1])
                # Find index of current dataname in sorted f keys
                f_idx = -1
                for i, (name, _) in enumerate(sorted_f_indices):
                    if name == dataname:
                        f_idx = i
                        break
                
                if f_idx != -1:
                    if f_idx == len(sorted_f_indices) - 1:
                        f_end = f_total_lines
                    else:
                        f_end = sorted_f_indices[f_idx+1][1]
                    
                    n_force_lines = f_end - f_start - 1
                    
                    if n_frames > 0:
                        calculated_natom = n_force_lines / n_frames
                        if abs(calculated_natom - round(calculated_natom)) < 1e-5:
                            natom = int(round(calculated_natom))
                            natom_source = "force_file"
                        else:
                            self.logger.warning(f"Non-integer natom ratio for {dataname}: {n_force_lines}/{n_frames} = {calculated_natom}. Using fallback logic.")
            else:
                 if f_file: # Only warn if we expected force file to exist
                     self.logger.warning(f"Could not find system {dataname} in force file.")

            # Fallback 2: Try looking up in testdata_dir if available
            if natom_source != "force_file" and self.testdata_dir and os.path.exists(self.testdata_dir):
                from dpeva.io.dataset import load_systems
                
                # Try locating system
                # Name might be "Pool/System" or just "System"
                sys_path = os.path.join(self.testdata_dir, dataname)
                if not os.path.exists(sys_path):
                    basename = os.path.basename(dataname)
                    sys_path_base = os.path.join(self.testdata_dir, basename)
                    if os.path.exists(sys_path_base):
                        sys_path = sys_path_base
                
                if os.path.exists(sys_path):
                    try:
                        systems = load_systems(sys_path)
                        if systems and len(systems) > 0:
                            natom = len(systems[0]["atom_types"])
                            natom_source = "testdata_dir"
                            self.logger.info(f"Resolved atom count {natom} for {dataname} from testdata_dir.")
                    except Exception as e:
                        self.logger.debug(f"Failed to load system from testdata_dir for {dataname}: {e}")

            if natom_source == "fallback":
                 self.logger.warning(f"Using fallback natom=1 for {dataname}.")

            for i in range(n_frames):
                datanames_nframe_list.append([dataname, i, natom])
                
        return datanames_nframe_list, datanames_nframe_dict


