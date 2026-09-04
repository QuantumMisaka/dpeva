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

from dpeva.compatibility import CapabilityMatrix

REQUIRED_CASES = (
    "pip-freeze", "deepmd-version", "torch-cuda", "gpu",
    "pt-test", "pt-test-ema", "pt-eval-desc", "pt-eval-desc-ema",
    "pt-embed", "pt-embed-ema", "dpa4c-periodic-eval-desc",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_dir():
        for child in sorted(path.rglob("*")):
            if child.is_file():
                digest.update(str(child.relative_to(path)).encode())
                with child.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
        return digest.hexdigest()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_result(path: Path, record: dict[str, Any]) -> dict[str, Any]:
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


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


def _preflight(config_path: Path, job_dir: Path) -> dict[str, Any]:
    """Revalidate the launch contract on the compute node before any case."""
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "commands").mkdir(parents=True, exist_ok=True)
    (job_dir / "logs").mkdir(parents=True, exist_ok=True)
    result_path = job_dir / "commands" / "preflight.json"
    started = _now()
    argv = ["preflight", str(job_dir / "launch.json")]
    errors: list[str] = []
    checks: list[dict[str, Any]] = []
    try:
        launch = _load(job_dir / "launch.json")
        expected_env = str(launch["qualification_env_name"])
        actual_env = os.environ.get("CONDA_DEFAULT_ENV", "")
        actual_prefix = Path(os.environ.get("CONDA_PREFIX", "")).name
        if actual_env != expected_env or actual_prefix != expected_env:
            errors.append(f"qualification environment mismatch: expected {expected_env}, default={actual_env!r}, prefix={actual_prefix!r}")
        input_path = Path(str(launch["input_path"])).expanduser().resolve()
        script_path = Path(str(launch["slurm_script_path"])).expanduser().resolve()
        if Path(str(launch["job_dir"])).expanduser().resolve() != job_dir.resolve():
            errors.append("launch job directory does not match execution directory")
        if input_path != config_path.resolve() or _sha256(input_path) != launch["input_sha256"]:
            errors.append("qualification input was mutated or is not the launched input")
        if _sha256(script_path) != launch["slurm_script_sha256"]:
            errors.append("Slurm script was mutated after submission")
        config = _load(input_path)
        if tuple(config.get("required_cases", ())) != REQUIRED_CASES:
            errors.append("qualification required_cases do not match harness")
        if launch.get("fixture_sha256") != config.get("fixture", {}).get("sha256"):
            errors.append("qualification fixture hash does not match launch contract")
        expected_specs = [
            {
                "case": case,
                "capability_key": record.key.model_dump(),
                "verification_command": record.verification_command,
                "source": "sai-v100-qualification",
            }
            for record in CapabilityMatrix.load_default().records
            if record.sai_verification_cases
            for case in record.sai_verification_cases
        ]
        if config.get("capability_attestation_specs") != expected_specs:
            errors.append("capability attestation specs do not match manifest")
        for role in ("regular", "ema"):
            item = config["models"][role]
            model = Path(item["path"]).expanduser().resolve()
            if not model.is_file() or _sha256(model) != item["sha256"]:
                errors.append(f"{role} model hash changed")
        dpa4c = config.get("dpa4c_model_path")
        if not dpa4c:
            errors.append("DPEVA_DEEPMD_DPA4C_MODEL is missing from qualification input")
        else:
            dpa4c_path = Path(dpa4c).expanduser()
            if not dpa4c_path.is_file() or not config.get("dpa4c_model_sha256") or _sha256(dpa4c_path) != config["dpa4c_model_sha256"]:
                errors.append("DPA4C model hash changed or path is absent")
        fixture = Path(config["fixture"]["path"]).expanduser().resolve()
        if (
            not fixture.is_dir()
            or not (fixture / "type.raw").is_file()
            or not (fixture / "type_map.raw").is_file()
            or _sha256(fixture) != config["fixture"].get("sha256")
        ):
            errors.append("periodic fixture is not a valid DeepMD/npy root")
        checks.append({"name": "launch", "ok": not errors})
        version = subprocess.run(["dp", "--version"], capture_output=True, text=True, check=False)
        checks.append({"name": "deepmd_version", "argv": ["dp", "--version"], "returncode": version.returncode, "value": version.stdout.strip()})
        if version.returncode != 0 or version.stdout.strip() != launch["expected_deepmd_version"]:
            errors.append("DeepMD version is not exact 3.2.0")
        gpu = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=False)
        checks.append({"name": "gpu", "argv": ["nvidia-smi", "-L"], "returncode": gpu.returncode, "value": gpu.stdout})
        if gpu.returncode != 0 or "V100" not in gpu.stdout.upper():
            errors.append("GPU evidence does not identify V100")
        torch_probe = subprocess.run([sys.executable, "-c", "import json, torch; print(json.dumps({'torch': torch.__version__, 'cuda': torch.version.cuda, 'available': torch.cuda.is_available()}))"], capture_output=True, text=True, check=False)
        torch_value = json.loads(torch_probe.stdout) if torch_probe.returncode == 0 else {}
        checks.append({"name": "torch_cuda", "argv": [sys.executable, "-c", "torch cuda probe"], "returncode": torch_probe.returncode, "value": torch_value})
        if torch_probe.returncode != 0 or not isinstance(torch_value, dict) or not torch_value.get("available") or not torch_value.get("cuda"):
            errors.append("Torch CUDA is unavailable")
        record = {"schema_version": "1.0", "case": "preflight", "argv": argv, "job_id": os.environ.get("SLURM_JOB_ID"), "started_at": started, "ended_at": _now(), "returncode": 0 if not errors else 1, "status": "finished" if not errors else "failed", "declared_artifacts": [str(job_dir / "launch.json")], "artifact_checks": [{"path": str(job_dir / "launch.json"), "exists": True, "sha256": _sha256(job_dir / "launch.json")}], "checks": checks, "error": "; ".join(errors) if errors else None}
    except Exception as exc:
        record = {"schema_version": "1.0", "case": "preflight", "argv": argv, "job_id": os.environ.get("SLURM_JOB_ID"), "started_at": started, "ended_at": _now(), "returncode": 2, "status": "failed", "declared_artifacts": [], "artifact_checks": [], "checks": checks, "error": str(exc)}
    return _write_result(result_path, record)


def run_recorded_command(config_path: Path, job_dir: Path, case: str) -> dict[str, Any]:
    if case == "preflight":
        return _preflight(config_path.expanduser().resolve(), job_dir.expanduser().resolve())
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
        if case in {"pip-freeze", "deepmd-version", "torch-cuda", "gpu"} and proc.returncode == 0:
            # Keep the human-readable command log, while making the value
            # itself machine-readable and tied to its argv/exit status.
            value: Any = proc.stdout
            if case == "torch-cuda":
                value = json.loads(proc.stdout)
                if not isinstance(value, dict) or not {"torch", "cuda", "available"} <= value.keys():
                    raise ValueError("torch-cuda probe did not return torch/cuda/available")
            declared[0].write_text(json.dumps({"schema_version": "1.0", "case": case, "argv": argv, "returncode": proc.returncode, "value": value}, indent=2) + "\n", encoding="utf-8")
        artifact_checks = [{"path": str(path), "exists": path.exists(), "sha256": _sha256(path) if path.exists() else ""} for path in declared]
        artifacts_ok = all(item["exists"] for item in artifact_checks)
        returncode = int(proc.returncode)
        status = "finished" if returncode == 0 and artifacts_ok else "failed"
        error = None if status == "finished" else ("missing declared artifact" if returncode == 0 else "command returned non-zero")
        record = {"schema_version": "1.0", "case": case, "argv": argv, "job_id": os.environ.get("SLURM_JOB_ID"), "started_at": start, "ended_at": _now(), "returncode": returncode, "status": status, "declared_artifacts": [str(path) for path in declared], "artifact_checks": artifact_checks, "error": error}
    except Exception as exc:
        record = {"schema_version": "1.0", "case": case, "argv": [], "job_id": os.environ.get("SLURM_JOB_ID"), "started_at": start, "ended_at": _now(), "returncode": 2, "status": "failed", "declared_artifacts": [], "artifact_checks": [], "error": str(exc)}
    return _write_result(result_path, record)


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
