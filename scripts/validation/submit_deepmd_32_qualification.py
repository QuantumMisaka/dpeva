#!/usr/bin/env python3
"""Verify qualification inputs and submit exactly one clean Slurm job."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_input(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != "1.0":
        raise ValueError("qualification input must be schema 1.0")
    for role in ("regular", "ema"):
        item = data.get("models", {}).get(role, {})
        artifact = Path(item.get("path", "")).expanduser()
        if not artifact.is_file() or _sha256(artifact) != item.get("sha256"):
            raise ValueError(f"model {role} is absent or SHA-256 changed: {artifact}")
    fixture = Path(data.get("fixture", {}).get("path", "")).expanduser()
    if not fixture.is_dir():
        raise ValueError(f"periodic fixture is absent: {fixture}")
    dpa4c = data.get("dpa4c_model_path")
    if dpa4c:
        dpa4c_path = Path(dpa4c).expanduser()
        if not dpa4c_path.is_file():
            raise ValueError(f"DPA4C model is absent: {dpa4c_path}")
    return data


def parse_job_id(output: str) -> str:
    matches = re.findall(r"Submitted\s+batch\s+job\s+(\d+)", output)
    if not matches:
        matches = re.findall(r"(?m)^\s*(\d+)\s*$", output)
    if len(matches) != 1:
        raise ValueError(f"could not parse exactly one Slurm JobID: {output!r}")
    return matches[0]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        try:
            Path(name).unlink()
        except FileNotFoundError:
            pass


def submit(input_path: Path, slurm_script: Path, write_ref: Path, *, job_root: Path | None = None, dry_run: bool = False) -> dict[str, Any]:
    input_path = input_path.expanduser().resolve()
    slurm_script = slurm_script.expanduser().resolve()
    if not slurm_script.is_file():
        raise FileNotFoundError(slurm_script)
    if not dry_run and os.environ.get("CONDA_PREFIX"):
        raise RuntimeError("qualification submission requires a clean login environment; unset CONDA_PREFIX")
    _load_input(input_path)
    root = (job_root or Path(os.environ.get("DPEVA_QUALIFICATION_ROOT", str(Path.home() / "scratch" / "dpeva-deepmd-qualification")))).expanduser().resolve()
    if dry_run:
        root = root / "dry-run"
    root.mkdir(parents=True, exist_ok=True)
    job_dir = root / f"deepmd-32-{uuid.uuid4().hex}"
    job_dir.mkdir()
    command = ["sbatch", "--export=NONE", str(slurm_script), str(input_path), str(job_dir)]
    if dry_run:
        job_id, output = "DRY-RUN", "Submitted batch job 0"
    else:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        output = result.stdout + result.stderr
        if result.returncode != 0:
            raise RuntimeError(f"sbatch failed ({result.returncode}): {output.strip()}")
        job_id = parse_job_id(output)
    _atomic_json(job_dir / "submission.json", {"schema_version": "1.0", "job_id": job_id, "input": str(input_path), "slurm_script": str(slurm_script), "input_sha256": _sha256(input_path), "script_sha256": _sha256(slurm_script), "command": command, "dry_run": dry_run})
    ref = {"schema_version": "1.0", "job_id": job_id, "job_dir": str(job_dir), "external_job_dir": str(job_dir), "status": "submitted" if not dry_run else "dry-run"}
    _atomic_json(write_ref.expanduser().resolve(), ref)
    print(json.dumps(ref, sort_keys=True))
    return ref


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--slurm-script", type=Path, required=True)
    parser.add_argument("--write-ref", type=Path, required=True)
    parser.add_argument("--job-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    submit(args.input, args.slurm_script, args.write_ref, job_root=args.job_root, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
