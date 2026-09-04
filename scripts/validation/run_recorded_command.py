#!/usr/bin/env python3
"""Run one qualification command and persist a machine-readable result."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != "1.0":
        raise ValueError("qualification input must be schema 1.0 object")
    return value


def _spec(config: dict[str, Any], case: str, job_dir: Path) -> tuple[list[str], list[Path]]:
    fixture = Path(config["fixture"]["path"])
    regular = Path(config["models"]["regular"]["path"])
    ema = Path(config["models"]["ema"]["path"])
    output = job_dir / "artifacts" / case
    if case in {"pip-freeze", "deepmd-version", "torch-cuda", "gpu"}:
        specs = {
            "pip-freeze": ([sys.executable, "-m", "pip", "freeze"], job_dir / "environment" / "pip-freeze.json"),
            "deepmd-version": (["dp", "--version"], job_dir / "environment" / "deepmd-version.json"),
            "torch-cuda": ([sys.executable, "-c", "import json, torch; print(json.dumps({'torch': torch.__version__, 'cuda': torch.version.cuda, 'available': torch.cuda.is_available()}))"], job_dir / "environment" / "torch-cuda.json"),
            "gpu": (["nvidia-smi", "-L"], job_dir / "environment" / "gpu.json"),
        }
        return specs[case][0], [specs[case][1]]
    if case.startswith("pt-test"):
        model = ema if case.endswith("-ema") else regular
        return ["dp", "--pt", "test", "-s", str(fixture), "-m", str(model), "-d", str(output)], [output.with_suffix(".e.out")]
    if case.startswith("pt-eval-desc"):
        model = ema if case.endswith("-ema") else regular
        return ["dp", "--pt", "eval-desc", "-s", str(fixture), "-m", str(model), "-o", str(output)], [output]
    if case.startswith("pt-embed"):
        model = ema if case.endswith("-ema") else regular
        return ["dp", "--pt", "embed", "-s", str(fixture), "-m", str(model), "-o", str(output.with_suffix(".hdf5"))], [output.with_suffix(".hdf5")]
    if case == "dpa4c-periodic-eval-desc":
        model_value = config.get("dpa4c_model_path")
        if not model_value:
            raise ValueError("DPEVA_DEEPMD_DPA4C_MODEL is required for dpa4c qualification")
        return ["dp", "--pt-expt", "eval-desc", "-s", str(fixture), "-m", str(Path(model_value)), "-o", str(output)], [output]
    raise ValueError(f"unknown qualification case: {case}")


def run_recorded_command(config_path: Path, job_dir: Path, case: str) -> dict[str, Any]:
    config = _load(config_path)
    command_dir, log_dir = job_dir / "commands", job_dir / "logs"
    command_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    result_path = command_dir / f"{case}.json"
    start = _now()
    try:
        argv, declared = _spec(config, case, job_dir)
        for path in declared:
            path.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(argv, cwd=str(job_dir), capture_output=True, text=True, check=False)
        (log_dir / f"{case}.stdout").write_text(proc.stdout, encoding="utf-8")
        (log_dir / f"{case}.stderr").write_text(proc.stderr, encoding="utf-8")
        if case in {"pip-freeze", "deepmd-version", "gpu"} and proc.returncode == 0:
            # Keep the human-readable command log, while making the value
            # itself machine-readable and tied to its argv/exit status.
            declared[0].write_text(json.dumps({"schema_version": "1.0", "case": case, "argv": argv, "returncode": proc.returncode, "value": proc.stdout}, indent=2) + "\n", encoding="utf-8")
        artifact_checks = [{"path": str(path), "exists": path.exists(), "sha256": _sha256(path) if path.is_file() else None} for path in declared]
        artifacts_ok = all(item["exists"] for item in artifact_checks)
        returncode = int(proc.returncode)
        status = "finished" if returncode == 0 and artifacts_ok else "failed"
        error = None if status == "finished" else ("missing declared artifact" if returncode == 0 else "command returned non-zero")
        record = {"schema_version": "1.0", "case": case, "argv": argv, "job_id": os.environ.get("SLURM_JOB_ID"), "started_at": start, "ended_at": _now(), "returncode": returncode, "status": status, "declared_artifacts": [str(path) for path in declared], "artifact_checks": artifact_checks, "error": error}
    except Exception as exc:
        record = {"schema_version": "1.0", "case": case, "argv": [], "job_id": os.environ.get("SLURM_JOB_ID"), "started_at": start, "ended_at": _now(), "returncode": 2, "status": "failed", "declared_artifacts": [], "artifact_checks": [], "error": str(exc)}
    result_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--job-dir", type=Path, default=None)
    parser.add_argument("--case", required=True)
    args = parser.parse_args(argv)
    job_dir = (args.job_dir or Path(os.environ["DPEVA_QUALIFICATION_DIR"])).expanduser().resolve()
    result = run_recorded_command(args.config.expanduser().resolve(), job_dir, args.case)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "finished" else 1


if __name__ == "__main__":
    raise SystemExit(main())
