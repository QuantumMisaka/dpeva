"""
Labeling Manager
================

Orchestrates the labeling process: input generation, task packing, and result collection.
"""

import os
import logging
import shutil
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple

import pandas as pd
from dpeva.config import LabelingConfig
from dpeva.labeling.generator import AbacusGenerator
from dpeva.labeling.packer import TaskPacker
from dpeva.labeling.postprocess import AbacusPostProcessor
from dpeva.labeling.strategy import ResubmissionStrategy

import dpdata

logger = logging.getLogger(__name__)

class LabelingManager:
    """
    Manager for the labeling workflow.
    """

    def __init__(self, config: Union[LabelingConfig, Dict[str, Any]]):
        if isinstance(config, LabelingConfig):
            self.config_obj = config
            self.config = config.model_dump()
        else:
            self.config = config
            self.config_obj = None

        self.generator = AbacusGenerator(self.config)
        self.packer = TaskPacker(self.config.get("tasks_per_job", 50))
        self.postprocessor = AbacusPostProcessor(self.config)
        self.strategy = ResubmissionStrategy(self.config.get("attempt_params"))
        
        self.work_dir = Path(self.config.get("work_dir", "."))
        self.input_dir = self.work_dir / "inputs"
        self.output_dir = self.work_dir / "outputs"
        self.converged_dir = self.work_dir / "CONVERGED"
        self.bad_converged_dir = self.work_dir / "BAD_CONVERGED"
        self.identity_map_path = self.output_dir / "task_identity_map.json"

    def prepare_tasks(self, dataset_map: Dict[str, dpdata.MultiSystems], task_prefix: str = "task") -> List[Path]:
        """
        Generate input files and pack tasks.
        
        Args:
            dataset_map: Dictionary mapping dataset names to MultiSystems.
            
        Returns:
            List[Path]: List of packed job directories (e.g., N_50_0).
        """
        total_systems = sum(len(ms) for ms in dataset_map.values())
        logger.info(f"Generating inputs for {len(dataset_map)} datasets ({total_systems} systems)...")
        self._reset_prepare_workspace()
        
        generated_count = 0
        
        for dataset_name, input_data in dataset_map.items():
            for i, system in enumerate(input_data):
                atoms_list = system.to_ase_structure()
                # Use target_name (system name) if available, else fallback
                sys_name = getattr(system, "target_name", getattr(system, "short_name", f"sys_{i}"))
                
                for f_idx, atoms in enumerate(atoms_list):
                    task_name = f"{sys_name}_{f_idx}"
                    
                    try:
                        # Use new analyze method from analyzer
                        pre_atoms, stru_type, vacuum_status = self.generator.analyzer.analyze(atoms)
                        
                        # Construct hierarchical path: inputs/dataset/stru_type/task_name
                        # Note: dataset_name is the top level now
                        task_dir = self.input_dir / dataset_name / stru_type / task_name
                        
                        # Generate files using pre-analyzed data
                        self.generator.generate(
                            pre_atoms, 
                            task_dir, 
                            task_name, 
                            stru_type, 
                            vacuum_status, 
                            dataset_name=dataset_name,
                            system_name=sys_name
                        )
                        generated_count += 1
                    except Exception as e:
                        logger.error(f"Failed to generate input for {task_name}: {e}")
        
        logger.info(f"Generated {generated_count} tasks.")
        
        # Pack tasks
        # Note: packer needs to scan recursively now because of deeper structure
        packed_job_dirs = self.packer.pack(self.input_dir)
        return packed_job_dirs

    def _reset_prepare_workspace(self):
        if self.input_dir.exists():
            logger.info(f"Resetting prepare workspace: {self.input_dir}")
            shutil.rmtree(self.input_dir)
        self.input_dir.mkdir(parents=True, exist_ok=True)

    def generate_runner_script(self, job_dir: Path) -> str:
        """
        Generate the content of a Python runner script for a packed job directory.
        This script will iterate over all subdirectories and run ABACUS.
        """
        # Ensure omp_threads is set (default to 1 if not in config)
        omp_threads = self.config.get("omp_threads", 1)
        if omp_threads == "auto":
            import os
            omp_threads = os.cpu_count() or 1
            
        script_content = f"""
import os
import subprocess
import sys
import shlex
from pathlib import Path

# Set OMP_NUM_THREADS
os.environ["OMP_NUM_THREADS"] = "{omp_threads}"

def run_abacus_tasks():
    root_dir = Path.cwd()
    # Assume we are inside a packed directory (e.g. N_50_X)
    # Iterate over all subdirectories (tasks)
    task_dirs = [d for d in root_dir.iterdir() if d.is_dir()]
    task_dirs.sort()
    
    print(f"Found {{len(task_dirs)}} tasks in {{root_dir}}")
    
    for task_dir in task_dirs:
        # Resolve task_dir to absolute path if it's relative
        # Though task_dirs from iterdir() are usually absolute or relative to cwd.
        # Let's ensure it's a Path object.
        task_dir_path = Path(task_dir)
        print(f"Running task: {{task_dir_path.name}}")
        os.chdir(task_dir_path)
        
        # Command to run ABACUS
        # We assume 'abacus' is in PATH or loaded via modules
        # Check environment variable for ABACUS command
        abacus_cmd = os.environ.get("ABACUS_COMMAND", "abacus")
        
        # Check SLURM_NTASKS for MPI
        slurm_ntasks = os.environ.get("SLURM_NTASKS", "1")
        
        abacus_args = shlex.split(abacus_cmd)
        if int(slurm_ntasks) > 1:
            cmd = ["mpirun", "-np", str(slurm_ntasks), *abacus_args]
        else:
            cmd = abacus_args
            
        try:
            with open("abacus.out", "w") as outfile:
                subprocess.run(cmd, check=True, stdout=outfile, stderr=subprocess.STDOUT)
            print(f"Task {{task_dir_path.name}} completed.")
        except subprocess.CalledProcessError as e:
            print(f"Task {{task_dir_path.name}} failed: {{e}}")
        except Exception as e:
            print(f"Task {{task_dir_path.name}} error: {{e}}")
        finally:
            os.chdir(root_dir)

if __name__ == "__main__":
    run_abacus_tasks()
"""
        return script_content

    def process_results(self, packed_job_dirs: List[Path]) -> Tuple[List[Path], List[Path]]:
        converged_tasks, _, failed_tasks = self.extract_results(packed_job_dirs)
        return converged_tasks, failed_tasks

    def extract_results(self, packed_job_dirs: List[Path]) -> Tuple[List[Path], List[Path], List[Path]]:
        logger.info("Processing results...")
        self.converged_dir.mkdir(parents=True, exist_ok=True)
        self.bad_converged_dir.mkdir(parents=True, exist_ok=True)
        
        converged_tasks = []
        bad_converged_tasks = []
        failed_tasks = []
        bad_reason_counter: Dict[str, int] = {}
        
        for job_dir in packed_job_dirs:
            if not job_dir.exists():
                logger.warning(f"Job directory {job_dir} does not exist.")
                continue
            
            # Recursive scan for tasks (contain INPUT)
            # Use same logic as packer to identify tasks
            for input_file in job_dir.rglob("INPUT"):
                # Skip if INPUT is in an output directory (e.g. OUT.ABACUS/INPUT)
                if "OUT." in input_file.parent.name:
                    continue
                    
                task_dir = input_file.parent
                if not task_dir.is_dir():
                    continue
                identity = self._resolve_task_identity(task_dir, self.input_dir)
                self._write_task_identity(task_dir, identity)
                self._update_identity_map(task_dir, identity)

                status, reason = self.postprocessor.classify_task_status(task_dir)
                if status == "converged":
                    converged_tasks.append(task_dir)
                    self._move_task_dir(task_dir, self.converged_dir, "CONVERGED")
                    continue
                if status == "bad_converged":
                    bad_converged_tasks.append(task_dir)
                    bad_reason_counter[reason] = bad_reason_counter.get(reason, 0) + 1
                    self._move_task_dir(task_dir, self.bad_converged_dir, "BAD_CONVERGED")
                    continue
                if status == "failed":
                    failed_tasks.append(task_dir)
                    continue
                failed_tasks.append(task_dir)

        logger.info(
            f"Converged: {len(converged_tasks)}, "
            f"Bad-Converged: {len(bad_converged_tasks)}, Failed: {len(failed_tasks)}"
        )
        if bad_reason_counter:
            for reason, count in sorted(bad_reason_counter.items()):
                logger.warning(f"Bad-Converged reason {reason}: {count}")
        return converged_tasks, bad_converged_tasks, failed_tasks

    def _build_target_subpath(self, task_dir: Path) -> Path:
        meta = self._resolve_task_identity(task_dir, self.input_dir)
        return Path(meta["dataset"]) / meta["type"] / meta["task_name"]

    def _move_task_dir(self, task_dir: Path, destination_root: Path, destination_name: str):
        try:
            target_subpath = self._build_target_subpath(task_dir)
            target_parent = destination_root / target_subpath.parent
            target_parent.mkdir(parents=True, exist_ok=True)
            final_target = destination_root / target_subpath
            final_target = self._dedupe_target_path(final_target)
            shutil.move(str(task_dir), str(final_target))
        except Exception as e:
            logger.error(f"Failed to move {task_dir} to {destination_name}: {e}")

    def apply_attempt_params(self, failed_tasks: List[Path], attempt: int):
        """
        Apply parameters for the next attempt to failed tasks.
        """
        logger.info(f"Applying parameters for attempt {attempt} to {len(failed_tasks)} tasks...")
        fixed_count = 0
        for task_dir in failed_tasks:
            if self.strategy.apply(task_dir, attempt):
                fixed_count += 1
        
        logger.info(f"Parameters applied to {fixed_count} tasks.")
        return fixed_count

    def repack_failed_tasks(self, failed_tasks: List[Path]) -> List[Path]:
        """
        Repack failed tasks into new N_50_X directories for resubmission.
        Since failed tasks are still in their original N_50_X directories (unless moved),
        we might need to consolidate them if they are sparse.
        
        However, simplicity first: If we keep them in place, we just re-submit the original N_50_X jobs?
        No, because converged tasks are moved out.
        So N_50_X only contains failed tasks now.
        We can just return the list of N_50_X directories that still contain tasks.
        """
        job_dirs = set()
        for task_dir in failed_tasks:
            job_dirs.add(task_dir.parent)
        
        return list(job_dirs)

    def collect_and_export(self):
        logger.info("Collecting converged results...")
        if not self.converged_dir.exists():
            logger.warning("CONVERGED directory not found.")
        task_dirs = self._collect_converged_task_dirs()
        systems, df, df_clean = self._build_metrics_data(task_dirs)
        if not df.empty:
            self._export_filtered_results(systems, df, df_clean)
        failed_tasks_info = self._collect_failed_tasks_info()
        stats = self._aggregate_stats(df, df_clean, failed_tasks_info)
        report = self._build_stats_report(stats, df, df_clean, failed_tasks_info)
        self._log_stats(stats, report["consistency"])
        self._write_stats_report(report)
        return report

    def rebuild_statistics_report(self) -> Dict[str, Any]:
        logger.info("Rebuilding labeling statistics report...")
        task_dirs = self._collect_converged_task_dirs()
        _, df, df_clean = self._build_metrics_data(task_dirs)
        failed_tasks_info = self._collect_failed_tasks_info()
        stats = self._aggregate_stats(df, df_clean, failed_tasks_info)
        report = self._build_stats_report(stats, df, df_clean, failed_tasks_info)
        self._log_stats(stats, report["consistency"])
        self._write_stats_report(report, filename="labeling_stats_report.repaired.json")
        return report

    def _collect_converged_task_dirs(self) -> List[Path]:
        task_dirs = []
        if self.converged_dir.exists():
            for root, _, files in os.walk(self.converged_dir):
                if "INPUT" in files and "STRU" in files and "OUT." not in Path(root).name:
                    task_dirs.append(Path(root))
        task_dirs.sort(key=lambda p: str(p))
        return task_dirs

    def _read_task_meta(self, task_dir: Path, base_dir: Path) -> Dict[str, str]:
        identity = self._resolve_task_identity(task_dir, base_dir)
        if identity["dataset"] == "unknown" or identity["type"] == "unknown":
            logger.warning(f"Task identity unresolved, fallback to unknown branch: {task_dir}")
        return identity

    def _build_metrics_data(self, task_dirs: List[Path]) -> Tuple[List[dpdata.LabeledSystem], pd.DataFrame, pd.DataFrame]:
        systems: List[dpdata.LabeledSystem] = []
        valid_systems = []
        skipped_systems = 0
        task_registry = []
        for task_dir in task_dirs:
            labeled_system = self.postprocessor.load_data(task_dir)
            if labeled_system:
                valid_systems.append(labeled_system)
                meta = self._read_task_meta(task_dir, self.converged_dir)
                task_registry.append({
                    "sys_idx": len(valid_systems) - 1,
                    "dataset": meta["dataset"],
                    "type": meta["type"],
                    "task_name": meta["task_name"],
                })
            else:
                skipped_systems += 1
        logger.info(f"Loaded {len(valid_systems)} converged systems from {len(task_dirs)} directories.")
        if skipped_systems > 0:
            logger.warning(f"Skipped {skipped_systems} converged directories due to incomplete parsed data.")
        if not valid_systems:
            return systems, pd.DataFrame(), pd.DataFrame()
        systems = list(valid_systems)
        metrics_frames = []
        for sys_idx, labeled_system in enumerate(valid_systems):
            single_systems = dpdata.MultiSystems()
            single_systems.append(labeled_system)
            df_single = self.postprocessor.compute_metrics(single_systems)
            if df_single.empty:
                continue
            df_single = df_single.copy()
            df_single["sys_idx"] = sys_idx
            metrics_frames.append(df_single)
        if metrics_frames:
            df = pd.concat(metrics_frames, ignore_index=True)
        else:
            df = pd.DataFrame()
        registry_df = pd.DataFrame(task_registry)
        if not registry_df.empty:
            if registry_df["sys_idx"].duplicated().any():
                raise ValueError("Duplicate sys_idx found in task registry.")
            df = df.merge(registry_df, on="sys_idx", how="left")
        else:
            df["dataset"] = "unknown"
            df["type"] = "unknown"
        if "dataset" not in df.columns:
            df["dataset"] = "unknown"
        if "type" not in df.columns:
            df["type"] = "unknown"
        df["dataset"] = df["dataset"].fillna("unknown")
        df["type"] = df["type"].fillna("unknown")
        df_clean = self.postprocessor.filter_data(df)
        return systems, df, df_clean

    def _export_filtered_results(self, systems: List[dpdata.LabeledSystem], df: pd.DataFrame, df_clean: pd.DataFrame):
        output_format = self.config.get("output_format", "deepmd/npy")
        cleaned_dir = self.output_dir / "cleaned"
        anomaly_dir = self.output_dir / "anomalies"
        self.postprocessor.export_data(systems, df_clean, cleaned_dir, format=output_format)
        if len(df) <= len(df_clean):
            return
        df_anomalies = df.drop(df_clean.index)
        self.postprocessor.export_data(systems, df_anomalies, anomaly_dir, format=output_format)
        self._export_anomaly_extxyz(systems, df_anomalies, anomaly_dir / "extxyz")

    def _export_anomaly_extxyz(self, systems: List[dpdata.LabeledSystem], df_anomalies: pd.DataFrame, extxyz_dir: Path):
        extxyz_dir.mkdir(exist_ok=True, parents=True)
        top_anomalies = df_anomalies.sort_values("max_force", ascending=False).head(50)
        for _, row in top_anomalies.iterrows():
            try:
                si, fi = int(row["sys_idx"]), int(row["frame_idx"])
                atoms = systems[si][fi].to_ase_structure()[0]
                ds_name = row.get("dataset", "unknown")
                atoms.write(extxyz_dir / f"anomaly_{ds_name}_{si}_{fi}.extxyz")
            except Exception as e:
                logger.warning(f"Failed to export anomaly extxyz: {e}")

    def _collect_failed_tasks_info(self) -> List[Dict[str, str]]:
        failed_tasks_info = []
        if not self.input_dir.exists():
            return failed_tasks_info
        for input_file in self.input_dir.rglob("INPUT"):
            if "OUT." in input_file.parent.name:
                continue
            task_dir = input_file.parent
            if not task_dir.is_dir():
                continue
            meta = self._read_task_meta(task_dir, self.input_dir)
            failed_tasks_info.append({"dataset": meta["dataset"], "type": meta["type"]})
        return failed_tasks_info

    def _aggregate_stats(self, df: pd.DataFrame, df_clean: pd.DataFrame, failed_tasks_info: List[Dict[str, str]]) -> Dict[str, Dict[str, Dict[str, int]]]:
        stats = {}
        all_datasets = set()
        if not df.empty and "dataset" in df.columns:
            all_datasets.update(df["dataset"].unique())
        for failed in failed_tasks_info:
            all_datasets.add(failed["dataset"])
        for ds in sorted(all_datasets):
            stats[ds] = {}
            ds_df = df[df["dataset"] == ds] if not df.empty and "dataset" in df.columns else pd.DataFrame()
            ds_clean = df_clean[df_clean["dataset"] == ds] if not df_clean.empty and "dataset" in df_clean.columns else pd.DataFrame()
            ds_failed = [x for x in failed_tasks_info if x["dataset"] == ds]
            all_types = set()
            if not ds_df.empty:
                all_types.update(ds_df["type"].unique())
            all_types.update(x["type"] for x in ds_failed)
            for t in sorted(all_types):
                t_df = ds_df[ds_df["type"] == t] if not ds_df.empty else pd.DataFrame()
                t_clean_df = ds_clean[ds_clean["type"] == t] if not ds_clean.empty else pd.DataFrame()
                n_conv = len(t_df)
                n_clean = len(t_clean_df)
                n_fail = sum(1 for x in ds_failed if x["type"] == t)
                stats[ds][t] = {
                    "total": n_conv + n_fail,
                    "conv": n_conv,
                    "fail": n_fail,
                    "clean": n_clean,
                    "filt": n_conv - n_clean,
                }
        return stats

    def _log_stats(self, stats: Dict[str, Dict[str, Dict[str, int]]], consistency: Dict[str, Any]):
        logger.info("=== Labeling Statistics Report ===")
        g_total = sum(v["total"] for ds in stats.values() for v in ds.values())
        g_conv = sum(v["conv"] for ds in stats.values() for v in ds.values())
        g_fail = sum(v["fail"] for ds in stats.values() for v in ds.values())
        g_clean = sum(v["clean"] for ds in stats.values() for v in ds.values())
        g_filt = sum(v["filt"] for ds in stats.values() for v in ds.values())
        logger.info(f"Global: Total={g_total}, Converged={g_conv}, Failed={g_fail}, Cleaned={g_clean}, Filtered={g_filt}")
        logger.info(f"Report trusted={consistency['trusted']}")
        if not consistency["trusted"]:
            for issue in consistency["errors"]:
                logger.error(f"Stats consistency check failed: {issue}")
        for ds, types in stats.items():
            d_total = sum(v["total"] for v in types.values())
            d_conv = sum(v["conv"] for v in types.values())
            d_fail = sum(v["fail"] for v in types.values())
            d_clean = sum(v["clean"] for v in types.values())
            d_filt = sum(v["filt"] for v in types.values())
            logger.info(f"  Dataset: {ds:<20} (Total={d_total}, Conv={d_conv}, Fail={d_fail}, Clean={d_clean}, Filt={d_filt})")
            dataset_hint = self.postprocessor.build_no_contribution_hint(d_conv, d_clean)
            if dataset_hint:
                logger.info(f"    {dataset_hint}")
            for t, v in types.items():
                logger.info(f"    Type: {t:<15} -> Total={v['total']}, Conv={v['conv']}, Fail={v['fail']}, Clean={v['clean']}, Filt={v['filt']}")
                type_hint = self.postprocessor.build_no_contribution_hint(v["conv"], v["clean"])
                if type_hint:
                    logger.info(f"      {type_hint}")

    @staticmethod
    def _is_valid_identity_value(value: Any) -> bool:
        if not isinstance(value, str):
            return False
        if not value.strip():
            return False
        return True

    @staticmethod
    def _is_packed_bundle_name(name: str) -> bool:
        return bool(re.match(r"^N_\d+_\d+$", name))

    @classmethod
    def _normalize_identity(
        cls,
        dataset_name: Optional[str],
        stru_type: Optional[str],
        task_name: Optional[str],
        fallback_task_name: str,
    ) -> Dict[str, str]:
        ds = dataset_name if cls._is_valid_identity_value(dataset_name) else "unknown"
        st = stru_type if cls._is_valid_identity_value(stru_type) else "unknown"
        tn = task_name if cls._is_valid_identity_value(task_name) else fallback_task_name
        if cls._is_packed_bundle_name(ds):
            ds = "unknown"
        if cls._is_packed_bundle_name(st):
            st = "unknown"
        return {"dataset": ds, "type": st, "task_name": tn}

    @staticmethod
    def _read_json_file(path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    return loaded
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return None
        return None

    def _infer_identity_from_path(self, task_dir: Path, base_dir: Path) -> Dict[str, str]:
        try:
            rel_path = task_dir.relative_to(base_dir)
        except ValueError:
            return self._normalize_identity(None, None, task_dir.name, task_dir.name)
        if len(rel_path.parts) >= 3 and not self._is_packed_bundle_name(rel_path.parts[0]):
            return self._normalize_identity(rel_path.parts[0], rel_path.parts[1], task_dir.name, task_dir.name)
        return self._normalize_identity(None, None, task_dir.name, task_dir.name)

    def _write_task_identity(self, task_dir: Path, identity: Dict[str, str]):
        identity_file = task_dir / "task_identity.json"
        identity_file.write_text(json.dumps(identity, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_identity_map(self) -> Dict[str, Dict[str, str]]:
        loaded = self._read_json_file(self.identity_map_path)
        if loaded is None:
            return {}
        by_task_name = loaded.get("by_task_name")
        if not isinstance(by_task_name, dict):
            return {}
        identity_map = {}
        for k, v in by_task_name.items():
            if not isinstance(v, dict):
                continue
            identity_map[k] = self._normalize_identity(
                v.get("dataset"),
                v.get("type"),
                v.get("task_name", k),
                k,
            )
        return identity_map

    def _save_identity_map(self, identity_map: Dict[str, Dict[str, str]]):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        payload = {"by_task_name": identity_map}
        self.identity_map_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _update_identity_map(self, task_dir: Path, identity: Dict[str, str]):
        identity_map = self._load_identity_map()
        task_name = identity["task_name"]
        existing = identity_map.get(task_name)
        if existing is not None and (existing["dataset"] != identity["dataset"] or existing["type"] != identity["type"]):
            identity_map.pop(task_name, None)
            self._save_identity_map(identity_map)
            return
        identity_map[task_name] = identity
        self._save_identity_map(identity_map)

    def _resolve_task_identity(self, task_dir: Path, base_dir: Path) -> Dict[str, str]:
        meta = self._read_json_file(task_dir / "task_meta.json")
        if meta is not None:
            normalized = self._normalize_identity(
                meta.get("dataset_name"),
                meta.get("stru_type"),
                meta.get("task_name", task_dir.name),
                task_dir.name,
            )
            if normalized["dataset"] != "unknown" and normalized["type"] != "unknown":
                return normalized

        sidecar = self._read_json_file(task_dir / "task_identity.json")
        if sidecar is not None:
            normalized = self._normalize_identity(
                sidecar.get("dataset"),
                sidecar.get("type"),
                sidecar.get("task_name", task_dir.name),
                task_dir.name,
            )
            if normalized["dataset"] != "unknown" and normalized["type"] != "unknown":
                return normalized

        inferred = self._infer_identity_from_path(task_dir, base_dir)
        if inferred["dataset"] != "unknown" and inferred["type"] != "unknown":
            return inferred

        identity_map = self._load_identity_map()
        mapped = identity_map.get(task_dir.name)
        if mapped is not None:
            return mapped
        return self._normalize_identity(None, None, task_dir.name, task_dir.name)

    @staticmethod
    def _dedupe_target_path(target_path: Path) -> Path:
        if not target_path.exists():
            return target_path
        idx = 1
        while True:
            candidate = target_path.parent / f"{target_path.name}_dup{idx}"
            if not candidate.exists():
                return candidate
            idx += 1

    def _build_stats_report(
        self,
        stats: Dict[str, Dict[str, Dict[str, int]]],
        df: pd.DataFrame,
        df_clean: pd.DataFrame,
        failed_tasks_info: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        global_stats = {
            "total": sum(v["total"] for ds in stats.values() for v in ds.values()),
            "conv": sum(v["conv"] for ds in stats.values() for v in ds.values()),
            "fail": sum(v["fail"] for ds in stats.values() for v in ds.values()),
            "clean": sum(v["clean"] for ds in stats.values() for v in ds.values()),
            "filt": sum(v["filt"] for ds in stats.values() for v in ds.values()),
        }
        consistency = self._validate_stats_consistency(stats, df, df_clean, failed_tasks_info)
        return {
            "global": global_stats,
            "branches": stats,
            "consistency": consistency,
        }

    def _validate_stats_consistency(
        self,
        stats: Dict[str, Dict[str, Dict[str, int]]],
        df: pd.DataFrame,
        df_clean: pd.DataFrame,
        failed_tasks_info: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        errors = []
        sum_total = sum(v["total"] for ds in stats.values() for v in ds.values())
        sum_conv = sum(v["conv"] for ds in stats.values() for v in ds.values())
        sum_fail = sum(v["fail"] for ds in stats.values() for v in ds.values())
        sum_clean = sum(v["clean"] for ds in stats.values() for v in ds.values())
        sum_filt = sum(v["filt"] for ds in stats.values() for v in ds.values())

        expected_conv = int(len(df))
        expected_fail = int(len(failed_tasks_info))
        expected_clean = int(len(df_clean))
        expected_total = expected_conv + expected_fail
        expected_filt = expected_conv - expected_clean

        if sum_total != expected_total:
            errors.append(f"total mismatch: details={sum_total}, expected={expected_total}")
        if sum_conv != expected_conv:
            errors.append(f"conv mismatch: details={sum_conv}, expected={expected_conv}")
        if sum_fail != expected_fail:
            errors.append(f"fail mismatch: details={sum_fail}, expected={expected_fail}")
        if sum_clean != expected_clean:
            errors.append(f"clean mismatch: details={sum_clean}, expected={expected_clean}")
        if sum_filt != expected_filt:
            errors.append(f"filt mismatch: details={sum_filt}, expected={expected_filt}")

        return {"trusted": len(errors) == 0, "errors": errors}

    def _write_stats_report(self, report: Dict[str, Any], filename: str = "labeling_stats_report.json"):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.output_dir / filename
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
