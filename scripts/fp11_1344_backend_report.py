#!/usr/bin/env python3
"""Report FP11 SAI-1344 labeling Slurm-array backend state."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


JOB_ID_RE = re.compile(r"Submitted batch job\s+(\d+)")
LOG_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")
ARRAY_LIMIT_RE = re.compile(r"%(\d+)$")
ABNORMAL_SLURM_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "TIMEOUT",
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _run(command: list[str]) -> str | None:
    if not shutil.which(command[0]):
        return None
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    output = (result.stdout + result.stderr).strip()
    return output or None


def _script_directives(path: Path) -> dict[str, str]:
    directives: dict[str, str] = {}
    if not path.exists():
        return directives
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("#SBATCH"):
            continue
        parts = line.split(maxsplit=2)
        if len(parts) >= 2:
            key = parts[1]
            value = parts[2] if len(parts) == 3 else ""
            if "=" in key:
                key, value = key.split("=", 1)
            directives[key] = value
    return directives


def _array_rows(work_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for script in sorted((work_dir / "array_jobs").glob("*/submit_array.slurm")):
        array_dir = script.parent
        manifest = array_dir / "tasks.json"
        task_count = None
        if manifest.exists():
            task_count = len(_read_json(manifest))
        rows.append(
            {
                "array_dir": str(array_dir),
                "tasks": task_count,
                "directives": _script_directives(script),
            }
        )
    return rows


def _job_ids_from_log(log_path: Path) -> list[str]:
    if not log_path.exists():
        return []
    return JOB_ID_RE.findall(log_path.read_text(encoding="utf-8", errors="replace"))


def _job_ids_from_logs(log_paths: list[Path]) -> tuple[list[str], dict[str, list[str]]]:
    sources: dict[str, list[str]] = {}
    job_ids: list[str] = []
    seen: set[str] = set()

    for log_path in log_paths:
        ids = _job_ids_from_log(log_path)
        sources[str(log_path)] = ids
        for job_id in ids:
            if job_id in seen:
                continue
            seen.add(job_id)
            job_ids.append(job_id)

    return job_ids, sources


def _parse_log_timestamp(line: str) -> datetime | None:
    match = LOG_TS_RE.match(line)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S,%f")


def _submit_elapsed_seconds(log_path: Path) -> list[float]:
    if not log_path.exists():
        return []

    elapsed: list[float] = []
    start: datetime | None = None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        timestamp = _parse_log_timestamp(line)
        if timestamp is None:
            continue
        if "Submitting " in line and " job bundles" in line:
            start = timestamp
            continue
        if start is not None and "Monitoring " in line and " Slurm jobs" in line:
            elapsed.append(round((timestamp - start).total_seconds(), 3))
            start = None
    return elapsed


def _submit_elapsed_seconds_from_logs(log_paths: list[Path]) -> list[float]:
    elapsed: list[float] = []
    for log_path in log_paths:
        elapsed.extend(_submit_elapsed_seconds(log_path))
    return elapsed


def _array_task_limits(arrays: list[dict[str, Any]]) -> list[int]:
    limits = set()
    for row in arrays:
        array_value = row.get("directives", {}).get("--array")
        if not isinstance(array_value, str):
            continue
        match = ARRAY_LIMIT_RE.search(array_value)
        if match:
            limits.add(int(match.group(1)))
    return sorted(limits)


def _slurm_state_counts(sacct_output: str | None) -> dict[str, int]:
    if not sacct_output:
        return {}

    lines = [line for line in sacct_output.splitlines() if line.strip()]
    if not lines:
        return {}
    header = lines[0].split("|")
    try:
        job_index = header.index("JobID")
        state_index = header.index("State")
    except ValueError:
        return {}

    counts: Counter[str] = Counter()
    for line in lines[1:]:
        fields = line.split("|")
        if len(fields) <= max(job_index, state_index):
            continue
        if "." in fields[job_index]:
            continue
        state = fields[state_index].strip()
        if state:
            counts[state] += 1
    return dict(sorted(counts.items()))


def _design_assessment(
    config: dict[str, Any],
    arrays: list[dict[str, Any]],
    state_counts: dict[str, int],
) -> tuple[bool, str]:
    submission = config.get("submission", {})
    array_submission_count = len(arrays)
    total_bundle_count = sum(int(row.get("tasks") or 0) for row in arrays)
    limits = _array_task_limits(arrays)
    abnormal_states = sorted(set(state_counts) & ABNORMAL_SLURM_STATES)

    if submission.get("slurm_array") is not True:
        return False, "Slurm array mode is not enabled."
    if array_submission_count == 0 or total_bundle_count == 0:
        return False, "No Slurm array scripts or bundle manifests were found."
    if not limits and not submission.get("slurm_array_task_limit"):
        return False, "No array throttle was found in config or rendered scripts."
    if abnormal_states:
        return False, "Abnormal Slurm states present: " + ", ".join(abnormal_states)
    if array_submission_count >= total_bundle_count:
        return False, "Array submission count is not smaller than bundle count."
    return True, "Slurm array backend is acceptable for FP11 production."


def _summary(
    config: dict[str, Any],
    arrays: list[dict[str, Any]],
    submit_logs: list[Path],
    job_ids: list[str],
    sacct_output: str | None,
) -> dict[str, Any]:
    state_counts = _slurm_state_counts(sacct_output)
    acceptable, assessment = _design_assessment(config, arrays, state_counts)
    array_submission_count = len(arrays)
    total_bundle_count = sum(int(row.get("tasks") or 0) for row in arrays)
    return {
        "array_submission_count": array_submission_count,
        "total_bundle_count": total_bundle_count,
        "array_task_limits": _array_task_limits(arrays),
        "job_id_count": len(job_ids),
        "slurm_state_counts": state_counts,
        "submit_elapsed_seconds": _submit_elapsed_seconds_from_logs(submit_logs),
        "array_to_bundle_ratio": (
            round(array_submission_count / total_bundle_count, 6)
            if total_bundle_count
            else None
        ),
        "design_acceptable": acceptable,
        "design_assessment": assessment,
    }


def build_report(root: Path, config_name: str) -> dict[str, Any]:
    config_path = root / config_name
    config = _read_json(config_path)
    work_dir = root / str(config["work_dir"])
    execute_log = work_dir / "labeling_execute.log"
    recovery_log = root / "logs" / "fp11_recover_1344.log"
    submit_logs = [execute_log, recovery_log]
    job_ids, job_id_sources = _job_ids_from_logs(submit_logs)
    arrays = _array_rows(work_dir)

    report: dict[str, Any] = {
        "root": str(root),
        "config": str(config_path),
        "work_dir": str(work_dir),
        "strategy": config.get("fp11_1344_resource_strategy"),
        "submission": {
            "backend": config.get("submission", {}).get("backend"),
            "slurm_array": config.get("submission", {}).get("slurm_array"),
            "slurm_array_task_limit": config.get("submission", {}).get("slurm_array_task_limit"),
            "slurm_config": config.get("submission", {}).get("slurm_config"),
        },
        "task_classes": [
            {
                "name": item.get("name"),
                "launcher_mode": item.get("launcher_mode"),
                "resource_mode": item.get("resource_mode"),
                "slurm_config": item.get("slurm_config"),
            }
            for item in config.get("labeling_task_classes", [])
        ],
        "arrays": arrays,
        "job_ids": job_ids,
        "job_id_sources": job_id_sources,
        "tmux": _run(["tmux", "ls"]),
    }

    if job_ids:
        joined = ",".join(job_ids)
        report["squeue"] = _run(
            [
                "squeue",
                "-j",
                joined,
                "-o",
                "%.18i %.9P %.32j %.8u %.2t %.10M %.6D %R",
            ]
        )
        report["sacct"] = _run(
            [
                "sacct",
                "-X",
                "-j",
                joined,
                "--format=JobID,JobName,State,ExitCode,Elapsed,AllocTRES%80",
                "-P",
            ]
        )
    report["summary"] = _summary(config, arrays, submit_logs, job_ids, report.get("sacct"))
    return report


def write_report_files(root: Path, report: dict[str, Any]) -> None:
    (root / "backend_report_1344.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary = report.get("summary", {})
    state_counts = summary.get("slurm_state_counts") or {}
    lines = [
        "# FP11 SAI-1344 Backend Report",
        "",
        f"Array submissions: {summary.get('array_submission_count')}",
        f"Total bundles: {summary.get('total_bundle_count')}",
        f"Array throttle: {summary.get('array_task_limits')}",
        f"Job IDs: {', '.join(report.get('job_ids', []))}",
        "Job ID sources:",
        *[
            f"- {source}: {', '.join(ids) if ids else 'none'}"
            for source, ids in (report.get("job_id_sources") or {}).items()
        ],
        f"Submit elapsed seconds: {summary.get('submit_elapsed_seconds')}",
        f"Slurm states: {state_counts}",
        f"Design acceptable: {summary.get('design_acceptable')}",
        "",
        str(summary.get("design_assessment") or ""),
        "",
    ]
    (root / "backend_report_1344.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="FP11 run root containing config_gpu_1344.json.",
    )
    parser.add_argument("--config", default="config_gpu_1344.json")
    args = parser.parse_args()

    report = build_report(Path(args.root).resolve(), args.config)
    write_report_files(Path(args.root).resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
