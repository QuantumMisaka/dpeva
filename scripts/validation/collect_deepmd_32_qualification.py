#!/usr/bin/env python3
"""Aggregate one immutable qualification directory fail-closed."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REQUIRED_CASES = (
    "pip-freeze", "deepmd-version", "torch-cuda", "gpu",
    "pt-test", "pt-test-ema", "pt-eval-desc", "pt-eval-desc-ema",
    "pt-embed", "pt-embed-ema", "dpa4c-periodic-eval-desc",
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def resolve_job_ref(ref_path: Path) -> tuple[Path, str]:
    pointer = _load(ref_path)
    job_dir_value = pointer.get("job_dir") or pointer.get("external_job_dir")
    job_id = str(pointer.get("job_id", ""))
    if not job_dir_value or not job_id or not re.fullmatch(r"\d+", job_id):
        raise ValueError("job reference lacks a numeric job_id and immutable job_dir")
    job_dir = Path(str(job_dir_value)).expanduser().resolve()
    if job_dir.name.lower() in {"latest", "current", "pending"}:
        raise ValueError("mutable latest/current qualification directory is not accepted")
    if not job_dir.is_dir():
        raise FileNotFoundError(f"recorded qualification directory does not exist: {job_dir}")
    if not (job_dir / "submission.json").is_file():
        raise ValueError("qualification directory is not a recorded submission")
    return job_dir, job_id


def collect_qualification(job_dir: Path, *, job_id: str | None = None, gpu: str | None = None, require_complete: bool = False) -> dict[str, Any]:
    job_dir = job_dir.expanduser().resolve()
    if job_dir.name.lower() in {"latest", "current", "pending"}:
        raise ValueError("mutable latest/current qualification directory is not accepted")
    if not job_dir.is_dir():
        raise FileNotFoundError(job_dir)
    records: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    failed: list[str] = []
    for case in REQUIRED_CASES:
        path = job_dir / "commands" / f"{case}.json"
        if not path.is_file():
            missing.append(case)
            continue
        record = _load(path)
        records[case] = record
        artifact_ok = True
        for item in record.get("artifact_checks", []):
            artifact_path = Path(str(item.get("path", ""))).expanduser()
            if not artifact_path.is_absolute():
                artifact_path = job_dir / artifact_path
            try:
                artifact_path.resolve().relative_to(job_dir)
            except ValueError:
                artifact_ok = False
                continue
            if not artifact_path.exists():
                artifact_ok = False
        if record.get("status") != "finished" or record.get("returncode") != 0 or not artifact_ok:
            failed.append(case)
    environment_missing = [name for name in ("pip-freeze.json", "deepmd-version.json", "torch-cuda.json", "gpu.json") if not (job_dir / "environment" / name).is_file()]
    submission = _load(job_dir / "submission.json") if (job_dir / "submission.json").is_file() else {}
    environment: dict[str, Any] = {}
    environment_invalid: list[str] = []
    for name, key in (("deepmd-version.json", "deepmd_version"), ("torch-cuda.json", "torch_cuda"), ("gpu.json", "gpu"), ("pip-freeze.json", "pip_freeze")):
        path = job_dir / "environment" / name
        if path.is_file():
            try:
                environment[key] = _load(path)
            except (TypeError, ValueError, json.JSONDecodeError):
                environment[key] = {"error": "malformed environment evidence"}
                environment_invalid.append(name)
    status = "finished" if not missing and not failed and not environment_missing and not environment_invalid else "failed"
    gpu_value = gpu
    if gpu_value is None:
        gpu_record = environment.get("gpu", {})
        gpu_value = gpu_record.get("value") if isinstance(gpu_record, dict) else None
    report: dict[str, Any] = {
        "schema_version": "1.0", "qualification": "deepmd-3.2-sai-v100", "status": status,
        "job_id": job_id or str(submission.get("job_id") or os.environ.get("SLURM_JOB_ID", "")),
        "gpu": gpu_value,
        "environment": environment,
        "failed_commands": failed,
        "missing_commands": missing,
        "missing_environment": environment_missing,
        "invalid_environment": environment_invalid,
        "commands": records,
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }
    # Atomic update of the final aggregate inside the already-recorded job.
    target = job_dir / "qualification.json"
    if target.exists():
        existing = _load(target)
        if require_complete and existing.get("status") != "finished":
            raise RuntimeError("qualification evidence is incomplete")
        return existing
    fd, name = tempfile.mkstemp(prefix=".qualification.", dir=str(job_dir), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, target)
    finally:
        try:
            Path(name).unlink()
        except FileNotFoundError:
            pass
    if require_complete and status != "finished":
        raise RuntimeError("qualification evidence is incomplete: " + ", ".join(missing + failed + environment_missing))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--job-dir", type=Path)
    group.add_argument("--job-ref", type=Path)
    parser.add_argument("--job-id")
    parser.add_argument("--gpu")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)
    if args.job_ref:
        job_dir, ref_job_id = resolve_job_ref(args.job_ref)
        job_id = args.job_id or ref_job_id
    else:
        job_dir, job_id = args.job_dir, args.job_id
        if not (job_dir / "submission.json").is_file():
            raise ValueError("--job-dir must point to a recorded submission")
    report = collect_qualification(job_dir, job_id=job_id, gpu=args.gpu, require_complete=args.require_complete)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"qualification collection failed: {exc}", flush=True)
        raise SystemExit(1)
