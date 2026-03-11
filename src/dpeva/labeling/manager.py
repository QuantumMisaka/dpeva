"""
Labeling Manager
================

Orchestrates the labeling process: input generation, task packing, and result collection.
"""

import os
import logging
import shutil
import json
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
        self.input_dir.mkdir(parents=True, exist_ok=True)
        
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
        """
        Process results from packed job directories.
        Iterates over tasks inside packed jobs (recursive).
        """
        logger.info("Processing results...")
        self.converged_dir.mkdir(parents=True, exist_ok=True)
        
        converged_tasks = []
        failed_tasks = []
        
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
                
                # Check convergence
                if self.postprocessor.check_convergence(task_dir):
                    converged_tasks.append(task_dir)
                    
                    # Move to CONVERGED
                    # Preserve hierarchy relative to inputs/
                    try:
                        # Extract sys_name from task_name (sys_name_fidx)
                        # Warning: sys_name might contain underscores.
                        # We know f_idx is int at end.
                        parts = task_dir.name.rsplit('_', 1)
                        if len(parts) == 2 and parts[1].isdigit():
                            sys_name = parts[0]
                            # Create subdir in CONVERGED
                            # Use metadata if available to reconstruct dataset/type hierarchy
                            # Or infer from task name if possible
                            # Ideally, we should read task_meta.json here, but it's inside task_dir.
                            
                            # Let's try to read metadata for better restoration
                            meta_file = task_dir / "task_meta.json"
                            target_subpath = Path(sys_name) # Default fallback
                            
                            if meta_file.exists():
                                try:
                                    import json
                                    with open(meta_file) as f:
                                        meta = json.load(f)
                                        ds = meta.get("dataset_name", "unknown")
                                        st = meta.get("stru_type", "unknown")
                                        tn = meta.get("task_name", task_dir.name)
                                        target_subpath = Path(ds) / st / tn
                                        # Note: task_dir.name usually equals tn
                                except:
                                    pass
                            else:
                                # Fallback logic
                                parts = task_dir.name.rsplit('_', 1)
                                if len(parts) == 2 and parts[1].isdigit():
                                    sys_name = parts[0]
                                    target_subpath = Path(sys_name) / task_dir.name
                            
                            target_parent = self.converged_dir / target_subpath.parent
                            target_parent.mkdir(parents=True, exist_ok=True)
                            
                            # Move
                            shutil.move(str(task_dir), str(self.converged_dir / target_subpath))
                        else:
                            # Fallback
                            shutil.move(str(task_dir), str(self.converged_dir))
                    except Exception as e:
                        logger.error(f"Failed to move {task_dir} to CONVERGED: {e}")
                else:
                    failed_tasks.append(task_dir)
        
        logger.info(f"Converged: {len(converged_tasks)}, Failed: {len(failed_tasks)}")
        return converged_tasks, failed_tasks

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
        self._log_stats(stats)

    def _collect_converged_task_dirs(self) -> List[Path]:
        task_dirs = []
        if self.converged_dir.exists():
            for root, _, files in os.walk(self.converged_dir):
                if "INPUT" in files and "STRU" in files and "OUT." not in Path(root).name:
                    task_dirs.append(Path(root))
        task_dirs.sort(key=lambda p: str(p))
        return task_dirs

    def _read_task_meta(self, task_dir: Path, base_dir: Path) -> Dict[str, str]:
        dataset_name = "unknown"
        stru_type = "unknown"
        task_name = task_dir.name
        meta_file = task_dir / "task_meta.json"
        if meta_file.exists():
            try:
                with open(meta_file) as f:
                    meta = json.load(f)
                    dataset_name = meta.get("dataset_name", "unknown")
                    stru_type = meta.get("stru_type", "unknown")
                    task_name = meta.get("task_name", task_dir.name)
            except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
                logger.warning(f"Failed to read metadata from {task_dir}: {exc}")
        else:
            try:
                rel_path = task_dir.relative_to(base_dir)
                if len(rel_path.parts) >= 3:
                    dataset_name = rel_path.parts[0]
                    stru_type = rel_path.parts[1]
                elif len(rel_path.parts) >= 1:
                    dataset_name = rel_path.parts[0]
            except ValueError:
                pass
        return {"dataset": dataset_name, "type": stru_type, "task_name": task_name}

    def _build_metrics_data(self, task_dirs: List[Path]) -> Tuple[dpdata.MultiSystems, pd.DataFrame, pd.DataFrame]:
        systems = dpdata.MultiSystems()
        valid_systems = []
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
        logger.info(f"Loaded {len(valid_systems)} converged systems from {len(task_dirs)} directories.")
        if not valid_systems:
            return systems, pd.DataFrame(), pd.DataFrame()
        for labeled_system in valid_systems:
            systems.append(labeled_system)
        df = self.postprocessor.compute_metrics(systems)
        registry_df = pd.DataFrame(task_registry)
        if not registry_df.empty:
            df = df.merge(registry_df, on="sys_idx", how="left")
        else:
            df["dataset"] = "unknown"
            df["type"] = "unknown"
        df_clean = self.postprocessor.filter_data(df)
        return systems, df, df_clean

    def _export_filtered_results(self, systems: dpdata.MultiSystems, df: pd.DataFrame, df_clean: pd.DataFrame):
        output_format = self.config.get("output_format", "deepmd/npy")
        cleaned_dir = self.output_dir / "cleaned"
        anomaly_dir = self.output_dir / "anomalies"
        self.postprocessor.export_data(systems, df_clean, cleaned_dir, format=output_format)
        if len(df) <= len(df_clean):
            return
        df_anomalies = df.drop(df_clean.index)
        self.postprocessor.export_data(systems, df_anomalies, anomaly_dir, format=output_format)
        self._export_anomaly_extxyz(systems, df_anomalies, anomaly_dir / "extxyz")

    def _export_anomaly_extxyz(self, systems: dpdata.MultiSystems, df_anomalies: pd.DataFrame, extxyz_dir: Path):
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

    def _log_stats(self, stats: Dict[str, Dict[str, Dict[str, int]]]):
        logger.info("=== Labeling Statistics Report ===")
        g_total = sum(v["total"] for ds in stats.values() for v in ds.values())
        g_conv = sum(v["conv"] for ds in stats.values() for v in ds.values())
        g_fail = sum(v["fail"] for ds in stats.values() for v in ds.values())
        g_clean = sum(v["clean"] for ds in stats.values() for v in ds.values())
        g_filt = sum(v["filt"] for ds in stats.values() for v in ds.values())
        logger.info(f"Global: Total={g_total}, Converged={g_conv}, Failed={g_fail}, Cleaned={g_clean}, Filtered={g_filt}")
        for ds, types in stats.items():
            d_total = sum(v["total"] for v in types.values())
            d_conv = sum(v["conv"] for v in types.values())
            d_fail = sum(v["fail"] for v in types.values())
            d_clean = sum(v["clean"] for v in types.values())
            d_filt = sum(v["filt"] for v in types.values())
            logger.info(f"  Dataset: {ds:<20} (Total={d_total}, Conv={d_conv}, Fail={d_fail}, Clean={d_clean}, Filt={d_filt})")
            for t, v in types.items():
                logger.info(f"    Type: {t:<15} -> Total={v['total']}, Conv={v['conv']}, Fail={v['fail']}, Clean={v['clean']}, Filt={v['filt']}")
