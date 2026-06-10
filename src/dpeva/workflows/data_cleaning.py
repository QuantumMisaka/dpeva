import json
import logging
import os
import shutil
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from dpeva.config import DataCleaningConfig
from dpeva.constants import LOG_FILE_CLEAN, WORKFLOW_FINISHED_TAG
from dpeva.io.dataproc import DPTestResultParser
from dpeva.io.dataset import load_systems
from dpeva.utils.logs import setup_workflow_logger
from dpeva.utils.security import normalize_sys_name, safe_join


class DataCleaningWorkflow:
    def __init__(self, config: Union[Dict, DataCleaningConfig]):
        self.logger = logging.getLogger(__name__)
        self.config = DataCleaningConfig(**config) if isinstance(config, dict) else config
        self.output_dir = str(self.config.output_dir)
        self._validate_config()
        os.makedirs(self.output_dir, exist_ok=True)
        setup_workflow_logger(
            logger_name="dpeva",
            work_dir=self.output_dir,
            filename=LOG_FILE_CLEAN,
            capture_stdout=True,
        )

    def _validate_config(self):
        if not os.path.exists(self.config.dataset_dir):
            raise ValueError(f"Dataset directory not found: {self.config.dataset_dir}")
        if not os.path.exists(self.config.result_dir):
            raise ValueError(f"Result directory not found: {self.config.result_dir}")

    def run(self):
        parser = DPTestResultParser(
            result_dir=str(self.config.result_dir),
            head=self.config.results_prefix,
            testdata_dir=str(self.config.dataset_dir),
        )
        parsed = parser.parse()
        if not parsed.get("has_ground_truth", False):
            raise ValueError("Data cleaning requires labeled inference results with ground truth columns.")
        metrics_df = self._build_metrics_dataframe(parsed)
        target_systems = sorted(metrics_df["system"].unique().tolist())
        systems = load_systems(str(self.config.dataset_dir), fmt="auto", target_systems=target_systems)
        if not systems:
            raise ValueError(f"No systems loaded from dataset directory: {self.config.dataset_dir}")
        keep_indices_map, decision_map, summary = self._select_frames(metrics_df, systems)
        self._export_cleaned_dataset(systems, keep_indices_map)
        frame_df = self._build_frame_report(metrics_df, decision_map)
        self._write_outputs(frame_df, summary)
        self.logger.info(WORKFLOW_FINISHED_TAG)

    def _build_metrics_dataframe(self, parsed: Dict) -> pd.DataFrame:
        dataname_list = parsed["dataname_list"]
        data_e = np.atleast_1d(parsed["energy"])
        n_frames = len(dataname_list)
        if len(data_e) != n_frames:
            raise ValueError(f"Energy frame count mismatch: results={len(data_e)} dataset_meta={n_frames}")
        force_diff = self._compute_force_frame_max_diff(parsed.get("force"), dataname_list)
        stress_diff = self._compute_stress_frame_max_diff(parsed.get("virial"), n_frames)
        records = []
        for idx, item in enumerate(dataname_list):
            system_name = item[0]
            frame_index = int(item[1])
            natom = int(item[2])
            records.append(
                {
                    "system": system_name,
                    "frame": frame_index,
                    "natom": natom,
                    "dataname": f"{system_name}-{frame_index}",
                    "energy_diff": float(abs(float(data_e[idx]["pred_e"]) - float(data_e[idx]["data_e"]))),
                    "force_max_diff": None if force_diff is None else float(force_diff[idx]),
                    "stress_max_diff": None if stress_diff is None else float(stress_diff[idx]),
                }
            )
        return pd.DataFrame.from_records(records)

    def _compute_force_frame_max_diff(self, force_data: Optional[np.ndarray], dataname_list: List[List]) -> Optional[np.ndarray]:
        if force_data is None:
            if self.config.force_max_diff_threshold is not None:
                raise ValueError("Force threshold is set but force predictions are missing.")
            return None
        force_arr = np.atleast_1d(force_data)
        dfx = force_arr["pred_fx"] - force_arr["data_fx"]
        dfy = force_arr["pred_fy"] - force_arr["data_fy"]
        dfz = force_arr["pred_fz"] - force_arr["data_fz"]
        atom_diff = np.sqrt(dfx**2 + dfy**2 + dfz**2)
        frame_max_diff = []
        offset = 0
        for item in dataname_list:
            natom = int(item[2])
            if offset + natom > len(atom_diff):
                raise ValueError(f"Force atom count mismatch near system={item[0]} frame={item[1]}")
            frame_max_diff.append(float(np.max(atom_diff[offset: offset + natom])))
            offset += natom
        if offset != len(atom_diff):
            raise ValueError("Force atom count mismatch: trailing atoms in results not mapped to frames.")
        return np.asarray(frame_max_diff)

    def _compute_stress_frame_max_diff(self, virial_data: Optional[np.ndarray], n_frames: int) -> Optional[np.ndarray]:
        if virial_data is None:
            if self.config.stress_max_diff_threshold is not None:
                raise ValueError("Stress threshold is set but virial predictions are missing.")
            return None
        virial_arr = np.atleast_1d(virial_data)
        if len(virial_arr) != n_frames:
            raise ValueError(f"Virial frame count mismatch: results={len(virial_arr)} dataset_meta={n_frames}")
        frame_max_diff = []
        for row in virial_arr:
            data_v = np.array([row[f"data_v{i}"] for i in range(9)], dtype=float)
            pred_v = np.array([row[f"pred_v{i}"] for i in range(9)], dtype=float)
            frame_max_diff.append(float(np.max(np.abs(pred_v - data_v))))
        return np.asarray(frame_max_diff)

    def _select_frames(self, metrics_df: pd.DataFrame, systems: List) -> Tuple[Dict[str, List[int]], Dict[Tuple[str, int], Dict], Dict]:
        thresholds = {
            "energy_diff": self.config.energy_diff_threshold,
            "force_max_diff": self.config.force_max_diff_threshold,
            "stress_max_diff": self.config.stress_max_diff_threshold,
        }
        enabled = {k: v for k, v in thresholds.items() if v is not None}
        if not enabled:
            self.logger.warning("No thresholds configured. Cleaning runs in passthrough mode.")
        metrics_map = {}
        for row in metrics_df.to_dict(orient="records"):
            key = (str(row["system"]), int(row["frame"]))
            if key in metrics_map:
                raise ValueError(f"Duplicate frame metrics detected for system={key[0]} frame={key[1]}")
            metrics_map[key] = row
        count_by_system = metrics_df.groupby("system").size().to_dict()
        used_keys = set()
        keep_indices_map: Dict[str, List[int]] = {}
        decision_map: Dict[Tuple[str, int], Dict] = {}
        trigger_counts = {k: 0 for k in enabled}
        total_frames = 0
        dropped_frames = 0
        for sys in systems:
            sys_name = str(getattr(sys, "target_name", sys.short_name))
            n_frames = len(sys)
            keep_indices: List[int] = []
            fallback_df = None
            for frame_idx in range(n_frames):
                key = (sys_name, frame_idx)
                row = metrics_map.get(key)
                if row is None and not self.config.strict_alignment:
                    if fallback_df is None:
                        expected_count = count_by_system.get(sys_name, 0)
                        if expected_count != n_frames:
                            raise ValueError(
                                f"Alignment failed for system={sys_name}: dataset_frames={n_frames}, result_frames={expected_count}"
                            )
                        fallback_df = (
                            metrics_df[metrics_df["system"] == sys_name]
                            .sort_values("frame")
                            .reset_index(drop=True)
                            .to_dict(orient="records")
                        )
                    row = fallback_df[frame_idx]
                if row is None:
                    raise ValueError(
                        f"Missing frame alignment for system={sys_name} frame={frame_idx}. "
                        f"Enable strict_alignment=false only when per-system frame counts are identical."
                    )
                used_keys.add((str(row["system"]), int(row["frame"])))
                reasons = []
                for metric_name, threshold in enabled.items():
                    value = row.get(metric_name)
                    if value is None or not np.isfinite(value) or float(value) > float(threshold):
                        reasons.append(metric_name)
                        trigger_counts[metric_name] += 1
                keep = len(reasons) == 0
                if keep:
                    keep_indices.append(frame_idx)
                else:
                    dropped_frames += 1
                total_frames += 1
                decision_map[(sys_name, frame_idx)] = {"keep": keep, "drop_reasons": reasons}
            keep_indices_map[sys_name] = keep_indices
        unused_keys = set(metrics_map.keys()) - used_keys
        if self.config.strict_alignment and unused_keys:
            sample = sorted(list(unused_keys))[:3]
            raise ValueError(f"Unused inference frames detected after alignment. Examples: {sample}")
        summary = {
            "dataset_dir": str(self.config.dataset_dir),
            "result_dir": str(self.config.result_dir),
            "output_dir": self.output_dir,
            "results_prefix": self.config.results_prefix,
            "strict_alignment": self.config.strict_alignment,
            "thresholds": {
                "energy_diff_threshold": self.config.energy_diff_threshold,
                "force_max_diff_threshold": self.config.force_max_diff_threshold,
                "stress_max_diff_threshold": self.config.stress_max_diff_threshold,
            },
            "frames": {
                "total": int(total_frames),
                "kept": int(total_frames - dropped_frames),
                "dropped": int(dropped_frames),
            },
            "drop_trigger_counts": {k: int(v) for k, v in trigger_counts.items()},
        }
        return keep_indices_map, decision_map, summary

    def _export_cleaned_dataset(self, systems: List, keep_indices_map: Dict[str, List[int]]):
        clean_root = safe_join(self.output_dir, "cleaned_dpdata")
        dropped_root = safe_join(self.output_dir, "filtered_out_dpdata")
        for root in (clean_root, dropped_root):
            if os.path.exists(root):
                shutil.rmtree(root)
            os.makedirs(root)
        for sys in systems:
            sys_name = str(getattr(sys, "target_name", sys.short_name))
            normalized_name = normalize_sys_name(sys_name)
            n_frames = len(sys)
            keep_indices = sorted(set(keep_indices_map.get(sys_name, [])))
            drop_indices = sorted(set(range(n_frames)) - set(keep_indices))
            if keep_indices:
                clean_path = safe_join(clean_root, normalized_name)
                sys.sub_system(keep_indices).to_deepmd_npy(clean_path)
            if drop_indices:
                dropped_path = safe_join(dropped_root, normalized_name)
                sys.sub_system(drop_indices).to_deepmd_npy(dropped_path)

    def _build_frame_report(self, metrics_df: pd.DataFrame, decision_map: Dict[Tuple[str, int], Dict]) -> pd.DataFrame:
        report_rows = []
        for row in metrics_df.to_dict(orient="records"):
            key = (str(row["system"]), int(row["frame"]))
            decision = decision_map.get(key)
            if decision is None and not self.config.strict_alignment:
                decision = {"keep": True, "drop_reasons": []}
            if decision is None:
                raise ValueError(f"Missing decision entry for system={key[0]} frame={key[1]}")
            report_rows.append(
                {
                    **row,
                    "keep": bool(decision["keep"]),
                    "drop_reasons": ",".join(decision["drop_reasons"]),
                }
            )
        return pd.DataFrame.from_records(report_rows)

    def _write_outputs(self, frame_df: pd.DataFrame, summary: Dict):
        report_path = safe_join(self.output_dir, "frame_metrics.csv")
        summary_path = safe_join(self.output_dir, "cleaning_summary.json")
        frame_df.to_csv(report_path, index=False)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        self.logger.info(
            f"Cleaning completed: kept={summary['frames']['kept']} dropped={summary['frames']['dropped']} "
            f"out of total={summary['frames']['total']}."
        )
