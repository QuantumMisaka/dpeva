"""
Labeling Workflow
=================

Workflow for First Principles (FP) labeling tasks.
"""

import logging
import time
import subprocess
import re
import sys
import hashlib
from pathlib import Path
from typing import Any, List, Dict, Optional

import dpdata

from dpeva.config import LabelingConfig
from dpeva.labeling.manager import LabelingManager
from dpeva.labeling.integration import DataIntegrationManager
from dpeva.submission.manager import JobManager
from dpeva.submission.array import ArrayTaskSpec
from dpeva.submission.names import normalize_slurm_job_name
from dpeva.submission.templates import JobConfig
from dpeva.constants import WORKFLOW_FINISHED_TAG
from dpeva.io.dataset import load_systems
from dpeva.utils.logs import setup_workflow_logger, close_workflow_logger

logger = logging.getLogger(__name__)

SAI_RANK_MAP_SETUP = "source /opt/sai_config/mps_mapping.d/${SLURM_JOB_PARTITION}.bash"
ACTIVE_SLURM_STATES = {
    "BOOT_FAIL",
    "CONFIGURING",
    "COMPLETING",
    "PENDING",
    "REQUEUED",
    "RESIZING",
    "RUNNING",
    "SIGNALING",
    "STAGE_OUT",
    "SUSPENDED",
}

class LabelingWorkflow:
    """
    Workflow for FP labeling.
    Steps:
    1. Load Data
    2. Generate Inputs & Pack
    3. Loop (Submit -> Wait -> Check -> Resubmit)
    4. Export
    """

    def __init__(self, config: LabelingConfig):
        self.config = config
        self.manager = LabelingManager(config)
        self.job_manager = JobManager(
            mode=config.submission.backend,
            # Pass custom template path if provided in slurm_config
            # SubmissionConfig.slurm_config is a dict
            custom_template_path=config.submission.slurm_config.get("template_path")
        )
        self._packed_job_dirs: List[Path] = []

    def run(self):
        logger.info("Starting Labeling Workflow...")
        packed_job_dirs = self.run_prepare()
        if not packed_job_dirs:
            logger.info(WORKFLOW_FINISHED_TAG)
            return
        self.run_execute(packed_job_dirs=packed_job_dirs)
        self.run_extract(packed_job_dirs=packed_job_dirs)
        self.run_postprocess()
        logger.info(WORKFLOW_FINISHED_TAG)

    def run_prepare(self) -> List[Path]:
        return self._run_with_stage_logging("prepare", self._run_prepare_impl)

    def _run_prepare_impl(self) -> List[Path]:
        logger.info(f"Loading input data from {self.config.input_data_path}")
        dataset_map = self._load_dataset_map()
        packed_job_dirs = self.manager.prepare_tasks(dataset_map)
        self._packed_job_dirs = packed_job_dirs
        if not packed_job_dirs:
            logger.warning("No tasks generated.")
        return packed_job_dirs

    def run_execute(self, packed_job_dirs: Optional[List[Path]] = None) -> List[Path]:
        return self._run_with_stage_logging("execute", self._run_execute_impl, packed_job_dirs)

    def _run_execute_impl(self, packed_job_dirs: Optional[List[Path]] = None) -> List[Path]:
        prepared_job_dirs = packed_job_dirs if packed_job_dirs is not None else self._resolve_packed_job_dirs()
        if not prepared_job_dirs:
            raise ValueError(
                f"No packed jobs found under {(Path(self.config.work_dir) / 'inputs')}. "
                "Please run prepare stage first."
            )
        max_attempts = len(self.config.attempt_params)
        if max_attempts == 0:
            logger.warning("No attempt_params defined. Executing single run with default parameters.")
            max_attempts = 1
        failed_tasks = []
        for attempt in range(max_attempts):
            logger.info(f"=== Execution Attempt {attempt}/{max_attempts - 1} ===")
            if attempt > 0:
                if not failed_tasks:
                    logger.info("No failed tasks to retry. Workflow finished.")
                    break
                if attempt < len(self.config.attempt_params):
                    logger.info(f"Preparing retry for {len(failed_tasks)} tasks...")
                    self.manager.apply_attempt_params(failed_tasks, attempt)
                else:
                    logger.warning(f"No parameters for attempt {attempt}. Skipping parameter application.")
            active_job_dirs = self._collect_active_job_dirs(prepared_job_dirs)
            if not active_job_dirs:
                logger.info("All tasks converged.")
                break
            logger.info(f"Submitting {len(active_job_dirs)} job bundles...")
            job_ids = self._submit_job_dirs(active_job_dirs, attempt)
            if self.config.submission.backend == "slurm":
                self._monitor_slurm_jobs(job_ids)
            else:
                logger.info("Local execution completed.")
            logger.info("Checking convergence...")
            _, failed_tasks = self.manager.process_results(active_job_dirs)
            if not failed_tasks:
                logger.info("All active tasks converged.")
                break
        self._packed_job_dirs = prepared_job_dirs
        return prepared_job_dirs

    def run_postprocess(self):
        self._run_with_stage_logging("postprocess", self._run_postprocess_impl)

    def _run_postprocess_impl(self):
        self.manager.collect_and_export()
        if self.config.integration_enabled:
            integration_manager = DataIntegrationManager(
                deduplicate=self.config.integration_deduplicate,
                output_format=self.config.integration_output_format,
            )
            cleaned_dir = Path(self.config.work_dir) / "outputs" / "cleaned"
            merged_path = self.config.merged_training_data_path or (Path(self.config.work_dir) / "outputs" / "merged_training_data")
            summary = integration_manager.integrate(
                new_labeled_data_path=cleaned_dir,
                merged_output_path=Path(merged_path),
                existing_training_data_path=self.config.existing_training_data_path,
            )
            logger.info(f"Data integration finished: {summary}")

    def run_extract(self, packed_job_dirs: Optional[List[Path]] = None) -> List[Path]:
        return self._run_with_stage_logging("extract", self._run_extract_impl, packed_job_dirs)

    def _run_extract_impl(self, packed_job_dirs: Optional[List[Path]] = None) -> List[Path]:
        prepared_job_dirs = packed_job_dirs if packed_job_dirs is not None else self._resolve_packed_job_dirs()
        if not prepared_job_dirs:
            raise ValueError(
                f"No packed jobs found under {(Path(self.config.work_dir) / 'inputs')}. "
                "Please run prepare stage first."
            )
        converged_tasks, bad_converged_tasks, failed_tasks = self.manager.extract_results(prepared_job_dirs)
        logger.info(
            "Extraction summary: "
            f"converged={len(converged_tasks)}, "
            f"bad_converged={len(bad_converged_tasks)}, failed={len(failed_tasks)}"
        )
        self._packed_job_dirs = prepared_job_dirs
        return prepared_job_dirs

    def _run_with_stage_logging(self, stage_name: str, stage_func, *args):
        log_filename = f"labeling_{stage_name}.log"
        log_path = str(Path(self.config.work_dir) / log_filename)
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        setup_workflow_logger("dpeva", str(self.config.work_dir), log_filename, capture_stdout=True)
        try:
            return stage_func(*args)
        finally:
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            except Exception:
                pass
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            close_workflow_logger("dpeva", log_path)

    def _load_dataset_map(self) -> Dict[str, dpdata.MultiSystems]:
        dataset_map: Dict[str, dpdata.MultiSystems] = {}
        try:
            root_path = Path(self.config.input_data_path)
            if not root_path.exists():
                raise FileNotFoundError(f"Input path not found: {root_path}")
            is_root_system = (root_path / "type.raw").exists() or (root_path / "type_map.raw").exists() or (root_path / "set.000").exists()
            if is_root_system:
                logger.info(f"Detected Single-System mode. Dataset: {root_path.name}")
                ms = dpdata.MultiSystems()
                loaded = load_systems(str(root_path), fmt="auto")
                for s in loaded:
                    ms.append(s)
                dataset_map[root_path.name] = ms
            else:
                subdirs = sorted([d for d in root_path.iterdir() if d.is_dir()])
                is_subdir_system = False
                for d in subdirs:
                    if (d / "type.raw").exists() or (d / "type_map.raw").exists() or (d / "set.000").exists():
                        is_subdir_system = True
                        break
                if is_subdir_system:
                    logger.info(f"Detected Single-Pool mode. Dataset: {root_path.name}")
                    ms = dpdata.MultiSystems()
                    loaded = load_systems(str(root_path), fmt="auto")
                    for s in loaded:
                        ms.append(s)
                    dataset_map[root_path.name] = ms
                else:
                    logger.info("Detected Multi-Pool mode.")
                    for d in subdirs:
                        if d.name.startswith("."):
                            continue
                        logger.info(f"Loading dataset: {d.name}")
                        loaded = load_systems(str(d), fmt="auto")
                        if loaded:
                            ms = dpdata.MultiSystems()
                            for s in loaded:
                                ms.append(s)
                            dataset_map[d.name] = ms
                        else:
                            logger.warning(f"No systems found in {d.name}, skipping.")
            if not dataset_map:
                raise ValueError(f"No valid systems found in {self.config.input_data_path}")
            total_systems = sum(len(ms) for ms in dataset_map.values())
            logger.info(f"Loaded {len(dataset_map)} datasets, {total_systems} systems total.")
        except Exception:
            logger.critical("Failed to load input data", exc_info=True)
            raise
        return dataset_map

    def _resolve_packed_job_dirs(self) -> List[Path]:
        if self._packed_job_dirs:
            return self._packed_job_dirs
        packed_root = Path(self.config.work_dir) / "inputs"
        if not packed_root.exists():
            return []
        packed_dirs = []
        for candidate in sorted(packed_root.iterdir()):
            if not candidate.is_dir():
                continue
            if candidate.name.startswith("N_"):
                packed_dirs.append(candidate)
                continue
            packed_dirs.extend(
                d for d in sorted(candidate.iterdir())
                if d.is_dir() and d.name.startswith("N_")
            )
        return packed_dirs

    @staticmethod
    def _collect_active_job_dirs(packed_job_dirs: List[Path]) -> List[Path]:
        active_job_dirs = []
        for job_dir in packed_job_dirs:
            if any(d.is_dir() for d in job_dir.iterdir()):
                active_job_dirs.append(job_dir)
        return active_job_dirs

    def _submit_job_dirs(self, active_job_dirs: List[Path], attempt: int) -> List[str]:
        if self.config.submission.backend == "slurm" and self.config.submission.slurm_array:
            return self._submit_job_dirs_as_arrays(active_job_dirs, attempt)

        job_ids = []
        for job_dir in active_job_dirs:
            class_config = self._task_class_config_for_job_dir(job_dir)
            launcher_mode = self._class_value(class_config, "launcher_mode", "auto")
            runner_content = self.manager.generate_runner_script(job_dir, launcher_mode=launcher_mode)
            runner_name = "run_batch.py"
            slurm_conf = self._slurm_config_for_class(class_config)
            env_setup = self._env_setup_for_class(class_config, launcher_mode)
            class_name = self._class_value(class_config, "name", None)
            job_name = f"fp_{job_dir.name}_att{attempt}"
            if class_name:
                job_name = f"fp_{class_name}_{job_dir.name}_att{attempt}"
            job_config = JobConfig(
                command="",
                job_name=normalize_slurm_job_name(job_name),
                partition=slurm_conf.get("partition"),
                qos=slurm_conf.get("qos"),
                nodes=slurm_conf.get("nodes", 1),
                ntasks=slurm_conf.get("ntasks", 1),
                gpus_per_node=slurm_conf.get("gpus_per_node"),
                cpus_per_task=slurm_conf.get("cpus_per_task"),
                walltime=slurm_conf.get("walltime", "24:00:00"),
                env_setup=env_setup
            )
            try:
                job_id = self.job_manager.submit_python_script(
                    runner_content,
                    runner_name,
                    job_config,
                    working_dir=str(job_dir)
                )
                job_ids.append(job_id)
            except Exception as e:
                logger.error(f"Failed to submit job for {job_dir}: {e}")
        return job_ids

    def _submit_job_dirs_as_arrays(self, active_job_dirs: List[Path], attempt: int) -> List[str]:
        grouped: Dict[str, List[Path]] = {}
        class_config_by_key: Dict[str, Any] = {}
        for job_dir in active_job_dirs:
            class_config = self._task_class_config_for_job_dir(job_dir)
            raw_class_name = self._class_value(class_config, "name", "default") or "default"
            class_name = str(raw_class_name)
            grouped.setdefault(class_name, []).append(job_dir)
            class_config_by_key[class_name] = class_config

        job_ids: List[str] = []
        for class_name, job_dirs in grouped.items():
            class_config = class_config_by_key[class_name]
            launcher_mode = self._class_value(class_config, "launcher_mode", "auto")
            slurm_conf = self._slurm_config_for_class(class_config)
            env_setup = self._env_setup_for_class(class_config, launcher_mode)
            tasks: List[ArrayTaskSpec] = []
            for idx, job_dir in enumerate(sorted(job_dirs, key=lambda path: str(path))):
                runner_content = self.manager.generate_runner_script(
                    job_dir,
                    launcher_mode=launcher_mode,
                )
                runner_path = job_dir / "run_batch.py"
                runner_path.write_text(runner_content, encoding="utf-8")
                tasks.append(
                    ArrayTaskSpec(
                        index=idx,
                        name=job_dir.name,
                        working_dir=job_dir,
                        argv=[sys.executable, "-u", str(runner_path.resolve())],
                    )
                )

            array_root = Path(self.config.work_dir) / "array_jobs" / f"{self._array_group_slug(class_name)}_attempt_{attempt}"
            job_config = JobConfig(
                command="",
                job_name=normalize_slurm_job_name(f"fp-{class_name}-att{attempt}"),
                partition=slurm_conf.get("partition"),
                qos=slurm_conf.get("qos"),
                nodes=slurm_conf.get("nodes", 1),
                ntasks=slurm_conf.get("ntasks", 1),
                gpus_per_node=slurm_conf.get("gpus_per_node"),
                cpus_per_task=slurm_conf.get("cpus_per_task"),
                walltime=slurm_conf.get("walltime", "24:00:00"),
                env_setup=env_setup,
            )
            try:
                job_id = self.job_manager.submit_array(
                    tasks=tasks,
                    job_config=job_config,
                    manifest_path=str(array_root / "tasks.json"),
                    script_path=str(array_root / "submit_array.slurm"),
                    working_dir=str(array_root),
                    array_task_limit=self.config.submission.slurm_array_task_limit,
                )
                job_ids.append(job_id)
            except Exception as e:
                logger.error(f"Failed to submit array for class {class_name}: {e}")
        if active_job_dirs and not job_ids:
            raise RuntimeError("Failed to submit any Slurm array jobs")
        return job_ids

    @staticmethod
    def _array_group_slug(class_name: str) -> str:
        safe_name = normalize_slurm_job_name(class_name, fallback="default", max_length=80)
        digest = hashlib.sha1(class_name.encode("utf-8")).hexdigest()[:8]
        return f"{safe_name}-{digest}"

    @staticmethod
    def _class_value(class_config: Any, key: str, default: Any = None) -> Any:
        if class_config is None:
            return default
        if isinstance(class_config, dict):
            return class_config.get(key, default)
        return getattr(class_config, key, default)

    def _task_class_config_for_job_dir(self, job_dir: Path) -> Any:
        class_name = self._task_class_name_for_job_dir(job_dir)
        if class_name is None:
            return None
        for class_config in self.config.labeling_task_classes:
            if self._class_value(class_config, "name") == class_name:
                return class_config
        return None

    def _task_class_name_for_job_dir(self, job_dir: Path) -> Optional[str]:
        meta = job_dir / ".dpeva_job_class.json"
        if meta.exists():
            try:
                import json
                loaded = json.loads(meta.read_text(encoding="utf-8"))
                class_name = loaded.get("task_class")
                if isinstance(class_name, str) and class_name:
                    return class_name
            except (OSError, ValueError, TypeError):
                pass
        try:
            rel = job_dir.relative_to(Path(self.config.work_dir) / "inputs")
        except ValueError:
            return None
        if len(rel.parts) >= 2 and rel.parts[1].startswith("N_"):
            return rel.parts[0]
        return None

    def _slurm_config_for_class(self, class_config: Any) -> Dict[str, Any]:
        merged = dict(self.config.submission.slurm_config)
        class_slurm = self._class_value(class_config, "slurm_config", {}) or {}
        merged.update(class_slurm)
        resource_mode = self._class_value(class_config, "resource_mode", None)
        if resource_mode == "single_gpu":
            merged["ntasks"] = 1
            merged["gpus_per_node"] = 1
        elif resource_mode == "multi_gpu_mpi":
            class_ntasks = class_slurm.get("ntasks")
            class_gpus = class_slurm.get("gpus_per_node")
            default_size = class_ntasks or class_gpus or 4
            merged["ntasks"] = class_ntasks or default_size
            merged["gpus_per_node"] = class_gpus or default_size
        return merged

    def _env_setup_for_class(self, class_config: Any, launcher_mode: str) -> str:
        lines = self._env_lines(self.config.submission.env_setup)
        class_env = self._class_value(class_config, "env_setup", "")
        lines.extend(self._env_lines(class_env))
        if launcher_mode == "abacus":
            lines = [
                line for line in lines
                if "mps_mapping.d" not in line and "MAP_OPT" not in line
            ]
        elif launcher_mode == "mpi_abacus":
            if not any("mps_mapping.d" in line for line in lines):
                profile_idx = next(
                    (idx for idx, line in enumerate(lines) if line.strip() == "source /etc/profile"),
                    None,
                )
                insert_idx = profile_idx + 1 if profile_idx is not None else 0
                lines.insert(insert_idx, SAI_RANK_MAP_SETUP)
        return "\n".join(lines)

    @staticmethod
    def _env_lines(env_setup: Any) -> List[str]:
        if env_setup is None:
            return []
        if isinstance(env_setup, list):
            raw_lines = env_setup
        else:
            raw_lines = str(env_setup).splitlines()
        return [line for line in raw_lines if line.strip()]

    def _monitor_slurm_jobs(self, job_ids: List[str], interval: int = 60):
        """Monitor Slurm jobs and wait until they finish."""
        if not job_ids:
            return

        logger.info(f"Monitoring {len(job_ids)} Slurm jobs...")
        
        clean_ids = []
        for jid in job_ids:
            match = re.search(r'\d+', jid)
            if match:
                clean_ids.append(match.group())
        
        if not clean_ids:
            logger.warning("No valid job IDs found to monitor.")
            return

        wait_count = 0
        while True:
            # Check queue via squeue
            cmd = ["squeue", "--job", ",".join(clean_ids), "--noheader", "--format=%i"]
            try:
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
                active_ids = result.stdout.strip().split()
                
                if not active_ids:
                    active_states = self._query_active_slurm_accounting_states(clean_ids)
                    if active_states:
                        if wait_count % 10 == 0:
                            states = ", ".join(sorted(set(active_states)))
                            logger.info(
                                "Slurm accounting still reports active jobs "
                                f"({states}) after an empty squeue response..."
                            )
                        wait_count += 1
                        time.sleep(interval)
                        continue
                    logger.info("All jobs finished.")
                    break
                
                # Log only every 10th interval (10 mins) to reduce spam
                # But keep polling every minute
                if wait_count % 10 == 0:
                    logger.info(f"{len(active_ids)} jobs still running... (waited {wait_count * interval // 60} mins)")
                
                wait_count += 1
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Failed to query Slurm queue: {e}")
                time.sleep(interval)

    @classmethod
    def _query_active_slurm_accounting_states(cls, job_ids: List[str]) -> List[str]:
        cmd = [
            "sacct",
            "-X",
            "-j",
            ",".join(job_ids),
            "--format=State",
            "-n",
            "-P",
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        active_states = []
        for line in result.stdout.splitlines():
            state = cls._base_slurm_state(line)
            if state in ACTIVE_SLURM_STATES:
                active_states.append(state)
        return active_states

    @staticmethod
    def _base_slurm_state(raw_state: str) -> str:
        return raw_state.strip().split("|", 1)[0].split()[0].upper() if raw_state.strip() else ""
