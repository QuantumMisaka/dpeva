import os
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional

from dpeva.io.dataproc import DPTestResultParser
from dpeva.io.types import PredictionData
from dpeva.uncertain.calculator import UQCalculator
from dpeva.uncertain.filter import UQFilter
from dpeva.constants import COL_UQ_QBC, COL_UQ_RND

class UQManager:
    """
    Manages UQ Analysis Workflow:
    - Loading predictions
    - Running UQ Calculation
    - Auto-thresholding
    - Filtering
    """
    
    def __init__(self, project_dir: str, testing_dir: str, testing_head: str, 
                 uq_config: Dict, num_models: int, testdata_dir: str = None):
        self.project_dir = project_dir
        self.testing_dir = testing_dir
        self.testing_head = testing_head
        self.uq_config = uq_config
        self.num_models = num_models
        self.testdata_dir = testdata_dir
        self.logger = logging.getLogger(__name__)
        
        self.calculator = UQCalculator()
        
        # Unpack Config
        self.trust_mode = uq_config.get("trust_mode", "auto")
        self.scheme = uq_config.get("scheme", "tangent_lo")
        self.auto_bounds = uq_config.get("auto_bounds", {})
        
        # These will be populated after analysis
        self.qbc_params = uq_config.get("qbc_params", {}).copy()
        self.rnd_params = uq_config.get("rnd_params", {}).copy()

    def load_predictions(self) -> Tuple[List[PredictionData], bool]:
        """
        Load prediction results for all models.

        Returns:
            Tuple[List[PredictionData], bool]: List of parsed predictions and a boolean indicating if ground truth exists.
        """
        self.logger.info(f"Loading test results for {self.num_models} models...")
        preds = []
        for i in range(self.num_models):
            path_prefix = os.path.join(self.project_dir, str(i), self.testing_dir, self.testing_head)
            result_dir = os.path.dirname(path_prefix)
            head_name = os.path.basename(path_prefix)
            
            parser = DPTestResultParser(result_dir=result_dir, head=head_name, testdata_dir=self.testdata_dir)
            parsed = parser.parse()
            preds.append(PredictionData(
                energy=parsed["energy"],
                force=parsed["force"],
                virial=parsed["virial"],
                has_ground_truth=parsed["has_ground_truth"],
                dataname_list=parsed["dataname_list"],
                datanames_nframe=parsed["datanames_nframe"]
            ))
            
        # Optional: Cross-verify atom counts with testdata_dir
        if self.testdata_dir and os.path.exists(self.testdata_dir) and len(preds) > 0:
            self._verify_atom_counts_list(preds[0].dataname_list)
            
        return preds, preds[0].has_ground_truth

    def _verify_atom_counts_list(self, dataname_list: List[List]):
        """
        Verify parsed atom counts against actual systems in testdata_dir.
        
        Args:
            dataname_list: List of [dataname, frame_idx, natom]
        """
        from dpeva.io.dataset import load_systems
        
        self.logger.info("Cross-verifying atom counts with original data...")
        
        # Extract unique systems and their parsed natoms
        unique_systems = {}
        for item in dataname_list:
            name, _, natom = item
            if name not in unique_systems:
                unique_systems[name] = natom
        
        verified_count = 0
        mismatch_count = 0
        
        for name, parsed_natom in unique_systems.items():
            # Try to locate system in testdata_dir
            sys_path = os.path.join(self.testdata_dir, name)
            if not os.path.exists(sys_path):
                # Try name as basename if not found (for single pool case)
                basename = os.path.basename(name)
                sys_path_base = os.path.join(self.testdata_dir, basename)
                if os.path.exists(sys_path_base):
                    sys_path = sys_path_base
                else:
                    self.logger.debug(f"Verification: System {name} not found in {self.testdata_dir}, skipping.")
                    continue
            
            try:
                # Load system
                systems = load_systems(sys_path)
                if not systems:
                    continue
                    
                # Check the first system
                real_natom = len(systems[0]["atom_types"])
                
                if parsed_natom != real_natom:
                    self.logger.error(f"ATOM COUNT MISMATCH for {name}: Parsed={parsed_natom}, Actual={real_natom}!")
                    mismatch_count += 1
                else:
                    verified_count += 1
                    
            except Exception as e:
                self.logger.warning(f"Verification failed for {name}: {e}")
        
        if mismatch_count > 0:
            self.logger.warning(f"Verification completed: {verified_count} matched, {mismatch_count} MISMATCHES.")
        else:
            self.logger.info(f"Verification completed: {verified_count} systems matched successfully.")

    def run_analysis(self, preds: List[PredictionData]) -> Tuple[Dict, np.ndarray]:
        """
        Run UQ calculation and scale alignment.

        Args:
            preds (List[PredictionData]): List of prediction data objects.

        Returns:
            Tuple[Dict, np.ndarray]: UQ results dictionary and rescaled RND array.
        """
        self.logger.info("Running UQ Calculation (QbC & RND)...")
        uq_results = self.calculator.compute_qbc_rnd(preds)
        
        self.logger.info("Aligning UQ-RND to UQ-QbC scales...")
        uq_rnd_rescaled = self.calculator.align_scales(uq_results[COL_UQ_QBC], uq_results[COL_UQ_RND])
        
        return uq_results, uq_rnd_rescaled

    def run_auto_threshold(self, uq_results: Dict, uq_rnd_rescaled: np.ndarray):
        """Calculates and updates trust thresholds if mode is 'auto'."""
        if self.trust_mode != "auto":
            return

        self.logger.info("Auto-calculating UQ thresholds...")
        
        # QbC
        qbc_ratio = self.qbc_params.get("ratio", 0.33)
        qbc_lo = self.calculator.calculate_trust_lo(uq_results[COL_UQ_QBC], ratio=qbc_ratio)
        qbc_lo = self._clamp(qbc_lo, self.auto_bounds.get("qbc", {}), "QbC")
        
        if qbc_lo is not None:
            self.qbc_params["lo"] = qbc_lo
            self.qbc_params["hi"] = qbc_lo + self.qbc_params.get("width", 0.25)
            self.logger.info(f"Auto QbC: [{self.qbc_params['lo']:.4f}, {self.qbc_params['hi']:.4f}]")
            
        # RND
        rnd_ratio = self.rnd_params.get("ratio", 0.33)
        rnd_lo = self.calculator.calculate_trust_lo(uq_rnd_rescaled, ratio=rnd_ratio)
        rnd_lo = self._clamp(rnd_lo, self.auto_bounds.get("rnd", {}), "RND")
        
        if rnd_lo is not None:
            self.rnd_params["lo"] = rnd_lo
            self.rnd_params["hi"] = rnd_lo + self.rnd_params.get("width", 0.25)
            self.logger.info(f"Auto RND: [{self.rnd_params['lo']:.4f}, {self.rnd_params['hi']:.4f}]")

    def _clamp(self, value, bounds, name):
        """
        Clamp value within bounds.

        Args:
            value (float): Value to clamp.
            bounds (Dict): Dictionary containing 'lo_min' and 'lo_max'.
            name (str): Name of the value for logging.

        Returns:
            float: Clamped value.
        """
        if value is None: return None
        lo_min, lo_max = bounds.get("lo_min"), bounds.get("lo_max")
        if lo_min is not None and value < lo_min:
            self.logger.warning(f"{name} < {lo_min}, clamping.")
            return lo_min
        if lo_max is not None and value > lo_max:
            self.logger.warning(f"{name} > {lo_max}, clamping.")
            return lo_max
        return value

    def run_filtering(self, df_uq_desc: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, UQFilter]:
        """Executes filtering and returns (candidate, accurate, failed, filter_obj)."""
        uq_filter = UQFilter(
            scheme=self.scheme,
            trust_lo=self.qbc_params["lo"],
            trust_hi=self.qbc_params["hi"],
            rnd_trust_lo=self.rnd_params["lo"],
            rnd_trust_hi=self.rnd_params["hi"]
        )
        
        self.logger.info(f"Filtering with scheme {self.scheme}...")
        cand, acc, fail = uq_filter.filter(df_uq_desc)
        
        self.logger.info(f"Filter Results: Cand={len(cand)}, Acc={len(acc)}, Fail={len(fail)}")
        return cand, acc, fail, uq_filter
