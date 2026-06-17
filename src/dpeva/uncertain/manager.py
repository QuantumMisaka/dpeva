import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

from dpeva.io.dataproc import DPTestResultParser
from dpeva.io.types import PredictionData
from dpeva.uncertain.calculator import UQCalculator
from dpeva.uncertain.filter import UQFilter
from dpeva.constants import (
    COL_UQ_DPOSE_ENERGY_ENSEMBLE_MEAN,
    COL_UQ_DPOSE_ENERGY_ENSEMBLE_N_MEMBERS,
    COL_UQ_DPOSE_ENERGY_ENSEMBLE_PATH,
    COL_UQ_DPOSE_ENERGY_ENSEMBLE_STD,
    COL_UQ_DPOSE_ENERGY_ENSEMBLE_STD_PER_ATOM,
    COL_UQ_LLPR_ALPHA,
    COL_UQ_LLPR_CALIBRATED,
    COL_UQ_LLPR_ENERGY_PER_ATOM,
    COL_UQ_LLPR_ENERGY_TOTAL,
    COL_UQ_QBC,
    COL_UQ_RND,
)
from dpeva.uncertain.dpose import DPOSEEnsemble
from dpeva.uncertain.llpr import LLPRCalibrator, LLPRState, ShallowEnsembleSampler

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
            
        has_ground_truth = all(pred.has_ground_truth for pred in preds)
        if not has_ground_truth:
            self.logger.info("At least one model lacks valid ground truth. Ground-truth-dependent plotting will be disabled.")
            self.logger.warning("Detected missing/invalid ground truth in target pool (including near-zero energy labels <1e-4). Treating the pool as unlabeled and enabling no-label analysis/plot branches.")
        return preds, has_ground_truth

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

    def run_llpr_analysis(
        self,
        train_features: np.ndarray,
        candidate_features: np.ndarray,
        candidate_atom_counts: np.ndarray | None = None,
        residuals: np.ndarray | None = None,
        calibration_uncertainties: np.ndarray | None = None,
        mean_energy: np.ndarray | None = None,
        last_layer_weights: np.ndarray | None = None,
    ) -> Dict:
        """
        Run DeepMD last-layer LLPR analysis and return collect-ready columns.

        Force-level DPOSE requires a differentiable DeepMDTorchDPOSEAdapter and is
        not available from detached last-layer feature arrays.
        """
        llpr_targets = self.uq_config.get("llpr_targets", "energy")
        self.logger.info(
            "Running LLPR/DPOSE UQ: targets=%s, train_shape=%s, candidate_shape=%s",
            llpr_targets,
            np.asarray(train_features).shape,
            np.asarray(candidate_features).shape,
        )
        if llpr_targets in {"force", "energy_force"}:
            raise RuntimeError(
                "Force-level DPOSE requires DeepMDTorchDPOSEAdapter; detached "
                "last-layer feature arrays only support llpr_targets='energy'."
            )
        regularizer = float(self.uq_config.get("llpr_regularizer", 1e-8))
        feature_normalization = self.uq_config.get("llpr_feature_normalization", "mean")
        self.logger.info(
            "LLPR covariance settings: regularizer=%s, feature_normalization=%s, "
            "state_path=%s, save_state_path=%s",
            regularizer,
            feature_normalization,
            self.uq_config.get("llpr_state_path"),
            self.uq_config.get("llpr_save_state_path"),
        )
        alpha = 1.0
        calibrated = False
        state_path = self.uq_config.get("llpr_state_path")
        if state_path:
            base_state = LLPRState.load_npz(state_path)
            alpha = base_state.alpha
            calibrated = base_state.calibrated
            feature_normalization = base_state.feature_normalization
            self.logger.info(
                "Loaded LLPR state: path=%s, feature_dimension=%d, alpha=%s, calibrated=%s",
                state_path,
                base_state.feature_dimension,
                np.array2string(np.asarray(base_state.alpha), precision=6),
                base_state.calibrated,
            )
        else:
            base_state = LLPRState.from_training_features(
                train_features,
                regularizer=regularizer,
                alpha=alpha,
                calibrated=calibrated,
                feature_normalization=feature_normalization,
            )
        if residuals is not None and calibration_uncertainties is not None:
            method = self.uq_config.get("llpr_calibration_method", "squared_residuals")
            self.logger.info("Calibrating LLPR alpha with method=%s", method)
            alpha = LLPRCalibrator(method=method).fit_alpha(
                residuals,
                calibration_uncertainties,
            )
            calibrated = True
            state = LLPRState.from_training_features(
                train_features,
                regularizer=regularizer,
                alpha=alpha,
                calibrated=True,
                feature_normalization=feature_normalization,
            )
        else:
            state = base_state

        save_state_path = self.uq_config.get("llpr_save_state_path")
        if save_state_path:
            state.save_npz(save_state_path)
            self.logger.info("Saved LLPR state: path=%s", save_state_path)

        prediction = state.predict_uncertainty(candidate_features)
        per_atom = prediction.per_atom
        if candidate_atom_counts is not None:
            atom_counts = np.asarray(candidate_atom_counts, dtype=float)
            if atom_counts.shape != prediction.total.shape:
                raise ValueError(
                    "candidate_atom_counts must have the same length as candidate_features."
                )
            if np.any(atom_counts <= 0):
                raise ValueError("candidate_atom_counts must be positive.")
            per_atom = prediction.total / atom_counts
        result = {
            COL_UQ_LLPR_ENERGY_TOTAL: prediction.total,
            COL_UQ_LLPR_ENERGY_PER_ATOM: per_atom,
            COL_UQ_LLPR_ALPHA: prediction.alpha,
            COL_UQ_LLPR_CALIBRATED: prediction.calibrated,
        }
        self.logger.info(
            "LLPR energy UQ completed: frames=%d, alpha=%s, calibrated=%s, "
            "feature_dimension=%d, energy_total_range=[%.6g, %.6g], "
            "energy_per_atom_range=[%.6g, %.6g]",
            len(prediction.total),
            np.array2string(np.asarray(prediction.alpha), precision=6),
            prediction.calibrated,
            state.feature_dimension,
            float(np.min(prediction.total)),
            float(np.max(prediction.total)),
            float(np.min(per_atom)),
            float(np.max(per_atom)),
        )
        n_members = self.uq_config.get("llpr_num_ensemble_members")
        if n_members:
            strict = self.uq_config.get("llpr_strict_metatrain_parity", True)
            if mean_energy is None and strict:
                raise RuntimeError(
                    "DPOSE energy_ensemble requires base DeepMD energy. "
                    "Provide llpr_candidate_energy_path or disable llpr_num_ensemble_members."
                )
            if last_layer_weights is None and self.uq_config.get("llpr_strict_metatrain_parity", True):
                raise RuntimeError(
                    "DPOSE energy_ensemble requires real last-layer weights. "
                    "Disable llpr_num_ensemble_members or provide last_layer_weights."
                )
            if mean_energy is None:
                raise RuntimeError("mean_energy is required to generate energy_ensemble.")
            if last_layer_weights is not None:
                sampler = DPOSEEnsemble(
                    state=state,
                    weights=last_layer_weights,
                    n_members=int(n_members),
                    random_seed=self.uq_config.get("llpr_random_seed"),
                )
            else:
                sampler = ShallowEnsembleSampler(
                    state,
                    n_members=int(n_members),
                    random_seed=self.uq_config.get("llpr_random_seed"),
                )
            result["energy_ensemble"] = sampler.sample_energy_ensemble(
                candidate_features,
                mean_energy,
            )
            ensemble = result["energy_ensemble"]
            ensemble_std = np.std(ensemble, axis=1, ddof=1)
            result[COL_UQ_DPOSE_ENERGY_ENSEMBLE_MEAN] = ensemble.mean(axis=1)
            result[COL_UQ_DPOSE_ENERGY_ENSEMBLE_STD] = ensemble_std
            result[COL_UQ_DPOSE_ENERGY_ENSEMBLE_STD_PER_ATOM] = ensemble_std / (
                np.asarray(candidate_atom_counts, dtype=float)
                if candidate_atom_counts is not None
                else np.ones(ensemble.shape[0], dtype=float)
            )
            result[COL_UQ_DPOSE_ENERGY_ENSEMBLE_N_MEMBERS] = int(n_members)
            ensemble_output_path = self.uq_config.get("llpr_ensemble_output_path")
            if ensemble_output_path:
                ensemble_output_path = Path(ensemble_output_path)
                ensemble_output_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(ensemble_output_path, ensemble)
                result[COL_UQ_DPOSE_ENERGY_ENSEMBLE_PATH] = str(ensemble_output_path)
            self.logger.info(
                "DPOSE energy ensemble generated: frames=%d, members=%d, "
                "weight_source=%s, output_path=%s, std_per_atom_range=[%.6g, %.6g]",
                ensemble.shape[0],
                ensemble.shape[1],
                self.uq_config.get("llpr_weight_source", "provided"),
                self.uq_config.get("llpr_ensemble_output_path"),
                float(np.min(result[COL_UQ_DPOSE_ENERGY_ENSEMBLE_STD_PER_ATOM])),
                float(np.max(result[COL_UQ_DPOSE_ENERGY_ENSEMBLE_STD_PER_ATOM])),
            )
        self.logger.info("LLPR output columns: %s", ", ".join(sorted(result)))
        return result
        
    def log_uq_statistics(self, uq_results: Dict, uq_rnd_rescaled: np.ndarray):
        """
        Log detailed statistics for UQ variables (similar to pandas describe).
        
        Args:
            uq_results (Dict): UQ results dictionary containing COL_UQ_QBC and COL_UQ_RND.
            uq_rnd_rescaled (np.ndarray): Rescaled RND values.
        """
        self.logger.info("Calculating statistics for UQ variables (QbC, RND, RND_rescaled)")
        
        try:
            # Construct temporary DataFrame for statistics
            df_stats = pd.DataFrame({
                "UQ_QbC": uq_results[COL_UQ_QBC],
                "UQ_RND": uq_results[COL_UQ_RND],
                "UQ_RND_rescaled": uq_rnd_rescaled
            })
            
            # Calculate describe stats
            # percentiles=[.25, .5, .75, .95, .99] to match user requirement
            stats = df_stats.describe(percentiles=[.25, .5, .75, .95, .99])
            
            # Format the output string to align columns beautifully
            # We use to_string() which provides a nice table format by default
            stats_str = stats.to_string(float_format=lambda x: "{:.4f}".format(x))
            
            self.logger.info(f"UQ Statistics:\n{stats_str}")
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate UQ statistics: {e}")

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
        if value is None:
            return None
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
