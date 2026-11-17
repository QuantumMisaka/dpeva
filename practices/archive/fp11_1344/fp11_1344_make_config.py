#!/usr/bin/env python3
"""Build the FP11 labeling config for the SAI-1344 16V100 partition."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


PROFILE_LINE = "source /etc/profile"
ABACUS_MODULE_LINE = "module load abacus/LTSv3.10.1-sm70-auto"
ABACUS_PP_LINE = "export ABACUS_PP_PATH=$HOME/PP_ORB/PP"
ABACUS_ORB_LINE = "export ABACUS_ORB_PATH=$HOME/PP_ORB/ORB"
WORK_DIR_1344 = "labeling_workdir_1344"
PARTITION_1344 = "16V100"
WALLTIME_1344 = "04:00:00"
SLURM_ARRAY_TASK_LIMIT = 128
NORMAL_QOS = "flood-gpu"
HIGHMEM_QOS = "flood-gpu"


def _probe_passed(probe: dict[str, Any], key: str) -> bool:
    result = probe.get(key)
    if not isinstance(result, dict):
        return False
    return (
        result.get("accepted_by_sbatch") is True
        and result.get("completed") is True
        and result.get("exit_code") == "0:0"
    )


def _env_lines(env_setup: Any) -> list[str]:
    if env_setup is None:
        return []
    raw_lines: list[Any]
    if isinstance(env_setup, list):
        raw_lines = []
        for item in env_setup:
            raw_lines.extend(str(item).splitlines())
    else:
        raw_lines = str(env_setup).splitlines()
    return [line.strip() for line in raw_lines if str(line).strip()]


def _ensure_profile(env_setup: Any) -> list[str]:
    lines = _env_lines(env_setup)
    rest: list[str] = []
    seen: set[str] = set()

    for line in lines:
        if line in (PROFILE_LINE, ABACUS_MODULE_LINE, ABACUS_PP_LINE, ABACUS_ORB_LINE):
            continue
        if line.startswith("export ABACUS_PP_PATH=") or line.startswith(
            "export ABACUS_ORB_PATH="
        ):
            continue
        if line in seen:
            continue
        rest.append(line)
        seen.add(line)

    return [PROFILE_LINE, ABACUS_MODULE_LINE, ABACUS_PP_LINE, ABACUS_ORB_LINE, *rest]


def _class_env_extras(env_setup: Any) -> list[str]:
    lines = _env_lines(env_setup)
    extras: list[str] = []
    seen: set[str] = set()
    for line in lines:
        if line in (PROFILE_LINE, ABACUS_MODULE_LINE, ABACUS_PP_LINE, ABACUS_ORB_LINE):
            continue
        if line.startswith("export ABACUS_PP_PATH=") or line.startswith(
            "export ABACUS_ORB_PATH="
        ):
            continue
        if line in seen:
            continue
        extras.append(line)
        seen.add(line)
    return extras


def _default_task_classes() -> list[dict[str, Any]]:
    return [
        {
            "name": "normal",
            "selector": {"max_atoms": 180},
            "tasks_per_job": 1,
        },
        {
            "name": "highmem",
            "selector": {"min_atoms": 181},
            "tasks_per_job": 1,
        },
    ]


def _is_multi_gpu_like(task_class: dict[str, Any]) -> bool:
    name = str(task_class.get("name", "")).lower()
    if "highmem" in name or "multi" in name:
        return True
    if task_class.get("resource_mode") == "multi_gpu_mpi":
        return True
    if task_class.get("launcher_mode") == "mpi_abacus":
        return True
    slurm_config = task_class.get("slurm_config")
    if isinstance(slurm_config, dict):
        for key in ("ntasks", "gpus_per_node"):
            try:
                if int(slurm_config.get(key, 0) or 0) > 1:
                    return True
            except (TypeError, ValueError):
                pass
    selector = task_class.get("selector")
    return isinstance(selector, dict) and "min_atoms" in selector and "max_atoms" not in selector


def _normalize_array_task_limit(value: Any) -> int:
    if value is None:
        return SLURM_ARRAY_TASK_LIMIT
    if isinstance(value, bool):
        raise ValueError("slurm_array_task_limit must be a positive integer")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and value.isdigit():
        parsed = int(value)
    else:
        raise ValueError("slurm_array_task_limit must be a positive integer")
    if parsed < 1:
        raise ValueError("slurm_array_task_limit must be a positive integer")
    return parsed


def _apply_single_gpu_class(task_class: dict[str, Any], qos: str) -> None:
    task_class["launcher_mode"] = "abacus"
    task_class["resource_mode"] = "single_gpu"
    task_class.setdefault("tasks_per_job", 1)
    slurm_config = task_class.setdefault("slurm_config", {})
    slurm_config.update(
        {
            "partition": PARTITION_1344,
            "ntasks": 1,
            "gpus_per_node": 1,
            "qos": qos,
            "walltime": WALLTIME_1344,
        }
    )


def _apply_mpi_4gpu_class(task_class: dict[str, Any], qos: str) -> None:
    task_class["launcher_mode"] = "mpi_abacus"
    task_class["resource_mode"] = "multi_gpu_mpi"
    task_class.setdefault("tasks_per_job", 1)
    slurm_config = task_class.setdefault("slurm_config", {})
    slurm_config.update(
        {
            "partition": PARTITION_1344,
            "ntasks": 4,
            "gpus_per_node": 4,
            "qos": qos,
            "walltime": WALLTIME_1344,
        }
    )


def patch_config_for_1344(config: dict[str, Any], probe: dict[str, Any]) -> dict[str, Any]:
    patched = copy.deepcopy(config)
    patched["work_dir"] = WORK_DIR_1344

    submission = patched.setdefault("submission", {})
    submission["backend"] = "slurm"
    submission["slurm_array"] = True
    submission["slurm_array_task_limit"] = _normalize_array_task_limit(
        submission.get("slurm_array_task_limit")
    )
    submission["env_setup"] = _ensure_profile(submission.get("env_setup"))
    submission.setdefault("slurm_config", {}).update(
        {
            "partition": PARTITION_1344,
            "walltime": WALLTIME_1344,
        }
    )

    if not patched.get("labeling_task_classes"):
        patched["labeling_task_classes"] = _default_task_classes()

    single_gpu_supported = _probe_passed(probe, "flood_gpu")
    if single_gpu_supported:
        patched["fp11_1344_resource_strategy"] = "single_gpu_normal"
        for task_class in patched["labeling_task_classes"]:
            task_class["env_setup"] = _class_env_extras(task_class.get("env_setup"))
            if _is_multi_gpu_like(task_class):
                _apply_mpi_4gpu_class(task_class, HIGHMEM_QOS)
            else:
                _apply_single_gpu_class(task_class, NORMAL_QOS)
    else:
        patched["fp11_1344_resource_strategy"] = "multi_gpu_mpi_fallback"
        for task_class in patched["labeling_task_classes"]:
            task_class["env_setup"] = _class_env_extras(task_class.get("env_setup"))
            qos = HIGHMEM_QOS if _is_multi_gpu_like(task_class) else NORMAL_QOS
            _apply_mpi_4gpu_class(task_class, qos)

    return patched


def main() -> int:
    cwd = Path.cwd()
    config_path = cwd / "config_gpu.json"
    probe_path = cwd / "probes" / "qos-single-gpu" / "qos_single_gpu_probe_result.json"
    output_path = cwd / "config_gpu_1344.json"

    config = json.loads(config_path.read_text(encoding="utf-8"))
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    patched = patch_config_for_1344(config, probe)
    output_path.write_text(
        json.dumps(patched, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
