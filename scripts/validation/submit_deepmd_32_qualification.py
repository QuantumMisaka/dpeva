#!/usr/bin/env python3
"""Verify qualification inputs and submit exactly one clean Slurm job."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any

from dpeva.compatibility import CapabilityMatrix

try:
    from scripts.validation.deepmd_32_qualification_scope import (
        attestation_specs,
        bind_scope,
        command_cases,
    )
except ModuleNotFoundError:  # direct script execution
    from deepmd_32_qualification_scope import (  # type: ignore[no-redef]
        attestation_specs,
        bind_scope,
        command_cases,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_dir():
        for child in sorted(path.rglob("*")):
            if child.is_file():
                digest.update(str(child.relative_to(path)).encode("utf-8"))
                with child.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
        return digest.hexdigest()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_input(path: Path, *, scope: str | None = None) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != "1.0":
        raise ValueError("qualification input must be schema 1.0")
    effective_scope = bind_scope(data, scope)
    for role in ("regular", "ema"):
        item = data.get("models", {}).get(role, {})
        if not isinstance(item.get("head"), str) or not item["head"].strip():
            raise ValueError(f"model {role} head is required and must be non-empty")
        artifact = Path(item.get("path", "")).expanduser()
        if not artifact.is_file() or _sha256(artifact) != item.get("sha256"):
            raise ValueError(f"model {role} is absent or SHA-256 changed: {artifact}")
    fixture = Path(data.get("fixture", {}).get("path", "")).expanduser()
    fixture_hash = data.get("fixture", {}).get("sha256")
    if not fixture.is_dir() or not isinstance(fixture_hash, str) or _sha256(fixture) != fixture_hash:
        raise ValueError(f"periodic fixture is absent: {fixture}")
    if effective_scope == "all":
        dpa4c = data.get("dpa4c_model_path")
        if not dpa4c:
            raise ValueError("DPEVA_DEEPMD_DPA4C_MODEL is required for qualification")
        dpa4c_path = Path(dpa4c).expanduser()
        if not dpa4c_path.is_file() or not data.get("dpa4c_model_sha256") or _sha256(dpa4c_path) != data["dpa4c_model_sha256"]:
            raise ValueError(f"DPA4C model is absent: {dpa4c_path}")
        if not isinstance(data.get("dpa4c_model_head"), str) or not data["dpa4c_model_head"].strip():
            raise ValueError("dpa4c_model_head is required and must be non-empty")
    if tuple(data.get("required_cases", ())) != command_cases(effective_scope):
        raise ValueError("qualification input required_cases do not match the harness")
    expected_specs = attestation_specs(
        CapabilityMatrix.load_default().records, effective_scope
    )
    if data.get("capability_attestation_specs") != expected_specs:
        raise ValueError("capability attestation specs are stale or do not match manifest")
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


def _exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a launch declaration once; never replace a launch contract."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(name, path)
    finally:
        try:
            Path(name).unlink()
        except FileNotFoundError:
            pass


def submit(
    input_path: Path,
    slurm_script: Path,
    write_ref: Path,
    *,
    scope: str | None = None,
    job_root: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    input_path = input_path.expanduser().resolve()
    slurm_script = slurm_script.expanduser().resolve()
    if not slurm_script.is_file():
        raise FileNotFoundError(slurm_script)
    script_text = slurm_script.read_text(encoding="utf-8")
    repo_root = slurm_script.parents[2]
    if not repo_root.is_dir():
        raise ValueError(f"could not resolve repository root from Slurm script: {slurm_script}")
    directives: dict[str, str] = {}
    for line in script_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("#SBATCH"):
            continue
        tokens = shlex.split(stripped[len("#SBATCH"):].strip())
        if not tokens or not tokens[0].startswith("--"):
            continue
        key, _, value = tokens[0].partition("=")
        if not value and len(tokens) > 1:
            value = tokens[1]
        directives[key] = value
    required_directives = {"--partition": "4V100", "--nodes": "1", "--ntasks": "1", "--gpus-per-node": "1", "--qos": "improper-gpu", "--time": "00:30:00"}
    forbidden_cpu = any(key == "--cpus" or key.startswith("--cpus-") for key in directives)
    forbidden_memory = any(key == "--mem" or key.startswith("--mem-") for key in directives)
    if any(directives.get(key) != value for key, value in required_directives.items()) or forbidden_memory or forbidden_cpu:
        raise ValueError("Slurm script does not satisfy the bounded SAI qualification contract")
    if not dry_run and os.environ.get("CONDA_PREFIX"):
        raise RuntimeError("qualification submission requires a clean login environment; unset CONDA_PREFIX")
    data = _load_input(input_path, scope=scope)
    effective_scope = bind_scope(data, scope)
    root = (job_root or Path(os.environ.get("DPEVA_QUALIFICATION_ROOT", str(Path.home() / "scratch" / "dpeva-deepmd-qualification")))).expanduser().resolve()
    if dry_run:
        root = root / "dry-run"
    root.mkdir(parents=True, exist_ok=True)
    job_dir = root / f"deepmd-32-{uuid.uuid4().hex}"
    job_dir.mkdir()
    nonce = uuid.uuid4().hex
    launch = {
        "schema_version": "1.0", "nonce": nonce,
        "input_path": str(input_path), "input_sha256": _sha256(input_path),
        "slurm_script_path": str(slurm_script), "slurm_script_sha256": _sha256(slurm_script),
        "expected_deepmd_version": "DeePMD-kit v3.2.0", "expected_gpu": "V100",
        "qualification_env_name": "dpeva-dpa4-320",
        "scope": effective_scope,
        "fixture_sha256": data["fixture"]["sha256"],
        "job_dir": str(job_dir), "status": "launched",
    }
    if effective_scope == "all":
        launch["dpa4c_model_sha256"] = data["dpa4c_model_sha256"]
    _exclusive_json(job_dir / "launch.json", launch)
    # SAI's Slurm control plane cancels ``--export=NONE`` jobs before the
    # batch step starts.  ``NIL`` preserves the intended clean environment
    # without triggering Slurm's implicit login-environment reconstruction.
    command = [
        "sbatch",
        "--export=NIL",
        f"--output={job_dir}/slurm-%j.out",
        f"--error={job_dir}/slurm-%j.err",
        str(slurm_script),
        str(input_path),
        str(job_dir),
        str(repo_root),
    ]
    if dry_run:
        job_id, output = "DRY-RUN", "Submitted batch job 0"
    else:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        output = result.stdout + result.stderr
        if result.returncode != 0:
            raise RuntimeError(f"sbatch failed ({result.returncode}): {output.strip()}")
        job_id = parse_job_id(output)
    _exclusive_json(job_dir / "submission.json", {"schema_version": "1.0", "job_id": job_id, "input": str(input_path), "slurm_script": str(slurm_script), "input_sha256": launch["input_sha256"], "script_sha256": launch["slurm_script_sha256"], "nonce": nonce, "job_dir": str(job_dir), "qualification_env_name": launch["qualification_env_name"], "scope": effective_scope, "command": command, "dry_run": dry_run})
    ref = {"schema_version": "1.0", "job_id": job_id, "job_dir": str(job_dir), "external_job_dir": str(job_dir), "nonce": nonce, "scope": effective_scope, "status": "submitted" if not dry_run else "dry-run"}
    _atomic_json(write_ref.expanduser().resolve(), ref)
    print(json.dumps(ref, sort_keys=True))
    return ref


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--slurm-script", type=Path, required=True)
    parser.add_argument("--write-ref", type=Path, required=True)
    parser.add_argument("--scope", choices=("dpa4", "all"))
    parser.add_argument("--job-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    submit(
        args.input,
        args.slurm_script,
        args.write_ref,
        scope=args.scope,
        job_root=args.job_root,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
