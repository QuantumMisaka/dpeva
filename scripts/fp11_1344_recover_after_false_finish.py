#!/usr/bin/env python3
"""Recover FP11 labeling after a transient empty Slurm array monitor result."""

from __future__ import annotations

import argparse
import logging
import subprocess
import time
from pathlib import Path
from typing import Iterable, List

from dpeva.cli import load_and_resolve_config
from dpeva.config import LabelingConfig
from dpeva.workflows.labeling import ACTIVE_SLURM_STATES, LabelingWorkflow


LOG = logging.getLogger("fp11_1344_recover")


def _base_state(raw_state: str) -> str:
    return raw_state.strip().split("|", 1)[0].split()[0].upper() if raw_state.strip() else ""


def _run(cmd: List[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )


def active_slurm_lines(job_ids: Iterable[str]) -> List[str]:
    ids = ",".join(str(job_id) for job_id in job_ids if str(job_id).strip())
    if not ids:
        return []

    squeue = _run(["squeue", "-j", ids, "-h", "-o", "%i %T"])
    active = [line for line in squeue.stdout.splitlines() if line.strip()]
    if active:
        return active

    sacct = _run(["sacct", "-X", "-j", ids, "--format=JobID,State", "-n", "-P"])
    for line in sacct.stdout.splitlines():
        parts = line.split("|")
        if len(parts) < 2:
            continue
        if _base_state(parts[1]) in ACTIVE_SLURM_STATES:
            active.append(line)
    return active


def wait_for_slurm_jobs(job_ids: List[str], interval: int) -> None:
    waited = 0
    while True:
        active = active_slurm_lines(job_ids)
        if not active:
            LOG.info("No active Slurm records remain for %s", ",".join(job_ids))
            return
        LOG.info(
            "%d Slurm records still active for %s after %d min; sample=%s",
            len(active),
            ",".join(job_ids),
            waited // 60,
            active[:5],
        )
        time.sleep(interval)
        waited += interval


def recover(config_path: Path, wait_job_ids: List[str], next_attempt: int, interval: int) -> int:
    if wait_job_ids:
        wait_for_slurm_jobs(wait_job_ids, interval)

    config = LabelingConfig(**load_and_resolve_config(str(config_path)))
    workflow = LabelingWorkflow(config)
    packed_job_dirs = workflow._resolve_packed_job_dirs()
    if not packed_job_dirs:
        raise RuntimeError("No packed FP11 job directories found for recovery")

    LOG.info("Processing results after true completion of %s", ",".join(wait_job_ids))
    _, failed_tasks = workflow.manager.process_results(packed_job_dirs)
    LOG.info("Failed tasks after recovery scan: %d", len(failed_tasks))

    for attempt in range(next_attempt, len(config.attempt_params)):
        if not failed_tasks:
            LOG.info("No failed tasks remain before attempt %d", attempt)
            break

        LOG.info("Applying attempt %d parameters to %d tasks", attempt, len(failed_tasks))
        workflow.manager.apply_attempt_params(failed_tasks, attempt)
        active_job_dirs = workflow._collect_active_job_dirs(packed_job_dirs)
        if not active_job_dirs:
            LOG.info("No active job directories remain before attempt %d", attempt)
            break

        LOG.info("Submitting %d recovered job bundles for attempt %d", len(active_job_dirs), attempt)
        job_ids = workflow._submit_job_dirs(active_job_dirs, attempt)
        workflow._monitor_slurm_jobs(job_ids, interval=interval)
        _, failed_tasks = workflow.manager.process_results(active_job_dirs)
        LOG.info("Failed tasks after attempt %d: %d", attempt, len(failed_tasks))

    LOG.info("Running final extract and postprocess")
    workflow.run_extract(packed_job_dirs=packed_job_dirs)
    workflow.run_postprocess()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--wait-job-id", action="append", default=[])
    parser.add_argument("--next-attempt", type=int, default=2)
    parser.add_argument("--interval", type=int, default=300)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return recover(args.config, args.wait_job_id, args.next_attempt, args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
