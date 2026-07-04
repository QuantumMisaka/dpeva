#!/usr/bin/env python3
"""Shared FP11 labeling submission helper rules.

This module keeps the FP11 audit, benchmark summary, and config patcher aligned
with DP-EVA's class-aware labeling launcher contract.
"""

from __future__ import annotations

import copy
from typing import Any


RANK_MAP_MARKER = "mps_mapping.d"
DEFAULT_SINGLE_GPU_ENV = ["module load abacus/LTSv3.10.1-sm70-auto"]
DEFAULT_SINGLE_GPU_SLURM = {
    "partition": "4V100",
    "ntasks": 1,
    "gpus_per_node": 1,
    "qos": "flood-1o2gpu",
    "walltime": "02:00:00",
}


def env_lines(env_setup: Any) -> list[str]:
    if env_setup is None:
        return []
    if isinstance(env_setup, list):
        raw = env_setup
    else:
        raw = str(env_setup).splitlines()
    return [line for line in raw if str(line).strip()]


def find_task_class(config: dict[str, Any], class_name: str) -> dict[str, Any] | None:
    for item in config.get("labeling_task_classes", []):
        if item.get("name") == class_name:
            return item
    return None


def audit_single_gpu_class(config: dict[str, Any], class_name: str = "normal") -> list[str]:
    issues: list[str] = []
    task_class = find_task_class(config, class_name)
    if task_class is None:
        issues.append(f"missing labeling task class: {class_name}")
        return issues

    if task_class.get("launcher_mode") != "abacus":
        issues.append(f"{class_name} must use launcher_mode=abacus")
    if task_class.get("resource_mode") != "single_gpu":
        issues.append(f"{class_name} must use resource_mode=single_gpu")

    slurm_config = task_class.get("slurm_config", {})
    if slurm_config.get("ntasks") != 1:
        issues.append(f"{class_name} must set ntasks=1")
    if slurm_config.get("gpus_per_node") != 1:
        issues.append(f"{class_name} must set gpus_per_node=1")

    combined_env = env_lines(config.get("submission", {}).get("env_setup"))
    combined_env.extend(env_lines(task_class.get("env_setup")))
    if any(RANK_MAP_MARKER in line or "MAP_OPT" in line for line in combined_env):
        issues.append(f"{class_name} must not source rank-map or reference MAP_OPT")
    return issues


def build_benchmark_summary(cases: list[dict[str, Any]], sample: dict[str, Any]) -> dict[str, Any]:
    ranked = sorted(
        [
            case for case in cases
            if case.get("mean_scf_iter_s_excluding_cu1") is not None
        ],
        key=lambda case: case["mean_scf_iter_s_excluding_cu1"],
    )
    if not ranked:
        raise ValueError("no benchmark cases contain mean_scf_iter_s_excluding_cu1")
    selected_omp = ranked[0]["omp_threads"]
    return {
        "sample": sample,
        "cases": cases,
        "selected": {
            "work_dir": "labeling_workdir_normal_g1",
            "omp_threads": selected_omp,
            "tasks_per_job": 1,
            "launcher_mode": "abacus",
            "resource_mode": "single_gpu",
            "slurm_config": dict(DEFAULT_SINGLE_GPU_SLURM),
            "env_setup": list(DEFAULT_SINGLE_GPU_ENV),
        },
        "selection_reason": (
            f"OMP{selected_omp} has the lowest stable SCF-iteration time among "
            "completed one-card benchmark cases; normal FP11 tasks stay single-card."
        ),
        "ranking_by_stable_scf_iter_s": [case["omp_threads"] for case in ranked],
    }


def patch_config_from_benchmark_summary(config: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    selected = summary.get("selected")
    if not isinstance(selected, dict):
        raise ValueError("benchmark summary must contain selected settings")

    patched = copy.deepcopy(config)
    patched["work_dir"] = selected["work_dir"]
    patched["omp_threads"] = selected["omp_threads"]
    patched["tasks_per_job"] = selected["tasks_per_job"]
    patched.setdefault("submission", {})["env_setup"] = list(selected.get("env_setup", DEFAULT_SINGLE_GPU_ENV))
    patched.setdefault("submission", {}).setdefault("slurm_config", {}).update(selected["slurm_config"])

    normal_class = {
        "name": "normal",
        "selector": {"max_atoms": 180},
        "tasks_per_job": selected["tasks_per_job"],
        "launcher_mode": selected["launcher_mode"],
        "resource_mode": selected["resource_mode"],
        "slurm_config": dict(selected["slurm_config"]),
        "env_setup": list(selected.get("env_setup", DEFAULT_SINGLE_GPU_ENV)),
    }
    highmem_class = {
        "name": "highmem",
        "selector": {"min_atoms": 181},
        "tasks_per_job": 1,
        "launcher_mode": "mpi_abacus",
        "resource_mode": "multi_gpu_mpi",
        "slurm_config": {
            "partition": "4V100",
            "ntasks": 4,
            "gpus_per_node": 4,
            "qos": "flood-gpu",
            "walltime": "02:00:00",
        },
    }
    patched["labeling_task_classes"] = [normal_class, highmem_class]

    dft = patched.setdefault("dft_params", {})
    if "xc" in dft:
        dft.setdefault("dft_functional", dft.pop("xc"))
    else:
        dft.setdefault("dft_functional", "pbe")
    return patched
