#!/usr/bin/env python3
"""Aggregate one immutable qualification directory fail-closed."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dpeva.compatibility import CapabilityAttestation, CapabilityKey, CapabilityMatrix

REQUIRED_CASES = (
    "preflight",
    "pip-freeze", "deepmd-version", "torch-cuda", "gpu",
    "pt-test", "pt-test-ema", "pt-eval-desc", "pt-eval-desc-ema",
    "pt-embed", "pt-embed-ema", "dpa4c-periodic-eval-desc",
)
REQUIRED_RECORD_FIELDS = {"schema_version", "case", "argv", "job_id", "started_at", "ended_at", "returncode", "status", "declared_artifacts", "artifact_checks"}
VALID_STATUSES = {"finished", "failed"}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _artifact_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if path.is_dir():
        for child in sorted(path.rglob("*")):
            if child.is_file():
                digest.update(str(child.relative_to(path)).encode())
                with child.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
        return digest.hexdigest()
    return ""


def _validate_record(record: dict[str, Any], case: str, job_dir: Path) -> list[str]:
    errors: list[str] = []
    if set(record) < REQUIRED_RECORD_FIELDS or record.get("schema_version") != "1.0" or record.get("case") != case:
        errors.append("record schema/case mismatch")
    if not isinstance(record.get("argv"), list) or not all(isinstance(value, str) for value in record.get("argv", [])):
        errors.append("argv must be a string list")
    if record.get("job_id") is not None and not isinstance(record.get("job_id"), str):
        errors.append("job_id must be a string or null")
    for key in ("started_at", "ended_at"):
        try:
            datetime.fromisoformat(str(record[key]))
        except (KeyError, TypeError, ValueError):
            errors.append(f"invalid {key}")
    if isinstance(record.get("returncode"), bool) or not isinstance(record.get("returncode"), int):
        errors.append("returncode must be an integer")
    if record.get("status") not in VALID_STATUSES:
        errors.append("invalid status")
    if not isinstance(record.get("declared_artifacts"), list) or not isinstance(record.get("artifact_checks"), list):
        errors.append("artifact fields must be lists")
    if len(record.get("declared_artifacts", [])) != len(record.get("artifact_checks", [])):
        errors.append("declared artifacts/checks length mismatch")
    if not all(isinstance(value, str) for value in record.get("declared_artifacts", [])):
        errors.append("declared artifacts must be strings")
    else:
        declared_paths = {
            str((Path(value).expanduser() if Path(value).expanduser().is_absolute() else job_dir / Path(value).expanduser()).resolve())
            for value in record.get("declared_artifacts", [])
        }
        checked_paths = set()
        for item in record.get("artifact_checks", []):
            if isinstance(item, dict) and isinstance(item.get("path"), str):
                value = Path(item["path"]).expanduser()
                checked_paths.add(str((value if value.is_absolute() else job_dir / value).resolve()))
        if declared_paths != checked_paths:
            errors.append("declared artifacts do not match artifact checks")
    for item in record.get("artifact_checks", []):
        if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not isinstance(item.get("exists"), bool) or not isinstance(item.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", item.get("sha256", "")):
            errors.append("malformed artifact check")
            continue
        artifact = Path(item["path"]).expanduser()
        if not artifact.is_absolute():
            artifact = job_dir / artifact
        try:
            resolved = artifact.resolve()
            resolved.relative_to(job_dir)
        except ValueError:
            errors.append("artifact path escapes job directory")
            continue
        if item.get("exists") is not True or not resolved.exists() or _artifact_sha256(resolved) != item["sha256"]:
            errors.append("artifact missing or hash changed")
    return errors


def _load_bound_attestation_specs(job_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Load specs bound by launch input and report any freshness violation."""

    errors: list[str] = []
    try:
        launch = _load(job_dir / "launch.json")
        input_path = Path(str(launch["input_path"])).expanduser().resolve()
        if _artifact_sha256(input_path) != launch["input_sha256"]:
            errors.append("qualification input hash changed")
        input_payload = _load(input_path)
        raw_specs = input_payload["capability_attestation_specs"]
        if not isinstance(raw_specs, list):
            raise ValueError("capability attestation specs must be a list")
        specs: list[dict[str, Any]] = []
        cases: set[str] = set()
        for raw in raw_specs:
            if not isinstance(raw, dict) or set(raw) != {"case", "capability_key", "verification_command", "source"}:
                raise ValueError("malformed capability attestation spec")
            key = CapabilityKey.model_validate(raw["capability_key"])
            if raw["source"] != "sai-v100-qualification" or not isinstance(raw["case"], str) or raw["case"] in cases:
                raise ValueError("invalid or duplicate capability attestation spec case")
            if not isinstance(raw["verification_command"], str) or not raw["verification_command"]:
                raise ValueError("capability attestation spec command is empty")
            cases.add(raw["case"])
            specs.append({
                "case": raw["case"],
                "capability_key": key.model_dump(),
                "verification_command": raw["verification_command"],
                "source": raw["source"],
            })
        expected = [
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
        if specs != expected:
            errors.append("capability attestation specs are stale relative to manifest")
        return specs, errors
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"invalid launch-bound capability specs: {exc}")
        return [], errors


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


def collect_qualification(job_dir: Path, *, job_id: str | None = None, gpu: str | None = None, require_complete: bool = False, finalize: bool = False) -> dict[str, Any]:
    job_dir = job_dir.expanduser().resolve()
    if job_dir.name.lower() in {"latest", "current", "pending"}:
        raise ValueError("mutable latest/current qualification directory is not accepted")
    if not job_dir.is_dir():
        raise FileNotFoundError(job_dir)
    records: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    unknown: list[str] = []
    malformed: list[str] = []
    failed: list[str] = []
    for case in REQUIRED_CASES:
        path = job_dir / "commands" / f"{case}.json"
        if not path.is_file():
            missing.append(case)
            continue
        try:
            record = _load(path)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            malformed.append(case)
            continue
        records[case] = record
        record_errors = _validate_record(record, case, job_dir)
        if record_errors or record.get("status") != "finished" or record.get("returncode") != 0:
            failed.append(case)
    for extra in (job_dir / "commands").glob("*.json"):
        if extra.stem not in REQUIRED_CASES:
            unknown.append(extra.stem)
    environment_missing = [name for name in ("pip-freeze.json", "deepmd-version.json", "torch-cuda.json", "gpu.json") if not (job_dir / "environment" / name).is_file()]
    identity_errors: list[str] = []
    try:
        submission = _load(job_dir / "submission.json") if (job_dir / "submission.json").is_file() else {}
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        submission = {}
        identity_errors.append("submission.json is malformed")
    submission_job_id = str(submission.get("job_id", ""))
    selected_job_id = str(job_id or submission_job_id or os.environ.get("SLURM_JOB_ID", ""))
    if not re.fullmatch(r"\d+", selected_job_id):
        identity_errors.append("selected job_id must be numeric")
    if not re.fullmatch(r"\d+", submission_job_id):
        identity_errors.append("submission job_id must be numeric")
    elif submission_job_id != selected_job_id:
        identity_errors.append("submission JobID does not match selected JobID")
    for case, record in records.items():
        record_job_id = record.get("job_id")
        if not isinstance(record_job_id, str) or not re.fullmatch(r"\d+", record_job_id):
            identity_errors.append(f"{case} command job_id must be numeric")
        elif record_job_id != selected_job_id:
            identity_errors.append(f"{case} command JobID does not match selected JobID")
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
    # The recorded environment is the only authoritative GPU identity.  The
    # CLI value is retained solely as an expected-value diagnostic and must
    # agree after whitespace normalization.
    gpu_record = environment.get("gpu", {})
    measured_gpu = gpu_record.get("value") if isinstance(gpu_record, dict) else None
    gpu_value = measured_gpu
    if gpu is not None and (not isinstance(measured_gpu, str) or gpu.strip() != measured_gpu.strip()):
        identity_errors.append("collector GPU expectation does not match environment/gpu.json")
    version_record = environment.get("deepmd_version", {})
    version_value = version_record.get("value") if isinstance(version_record, dict) else None
    normalized_version = version_value.strip() if isinstance(version_value, str) else None
    if normalized_version != "DeePMD-kit v3.2.0" and "deepmd-version.json" not in environment_invalid:
        environment_invalid.append("deepmd-version.json")
    if not isinstance(measured_gpu, str) or "v100" not in measured_gpu.lower():
        if "gpu.json" not in environment_invalid:
            environment_invalid.append("gpu.json")
    bound_specs, spec_errors = _load_bound_attestation_specs(job_dir)
    invalid_evidence = [*spec_errors, *identity_errors]
    if not bound_specs:
        invalid_evidence.append("bound capability attestation specs must be non-empty")
    candidate_errors: list[str] = []
    attestations: list[dict[str, Any]] = []
    # Construct candidate attestations before deciding aggregate status. A
    # failed command or construction anomaly must never leak a finished
    # attestation into a failed aggregate.
    if bound_specs and not identity_errors and not environment_invalid and not environment_missing and not spec_errors:
        for spec in bound_specs:
            case = spec["case"]
            command_record = records.get(case)
            if command_record is None:
                candidate_errors.append(f"missing command record for attestation case {case}")
                continue
            try:
                attestation = CapabilityAttestation(
                    status="finished",
                    returncode=command_record["returncode"],
                    capability_key=CapabilityKey.model_validate(spec["capability_key"]),
                    verification_command=spec["verification_command"],
                    deepmd_version=normalized_version,
                    source="sai-v100-qualification",
                    case=case,
                    job_id=selected_job_id,
                    gpu=measured_gpu,
                )
            except Exception as exc:
                candidate_errors.append(f"invalid attestation for {case}: {exc}")
                continue
            attestations.append(attestation.model_dump(mode="json"))
        spec_cases = [spec["case"] for spec in bound_specs]
        attestation_cases = [item.get("case") for item in attestations]
        if len(attestations) != len(spec_cases):
            candidate_errors.append("attestation count does not equal bound spec count")
        if len(attestation_cases) != len(set(attestation_cases)):
            candidate_errors.append("duplicate attestation cases")
        if set(attestation_cases) != set(spec_cases):
            candidate_errors.append("attestation cases do not match bound specs")
    if candidate_errors:
        invalid_evidence.extend(candidate_errors)
        attestations = []
    status = "finished" if (
        not missing and not unknown and not malformed and not failed
        and not environment_missing and not environment_invalid
        and not invalid_evidence and bound_specs and attestations
        and len(attestations) == len({spec["case"] for spec in bound_specs})
    ) else "failed"
    # Attestations are an output of a finished aggregate, never a partial
    # preview of one. This also covers command records that fail artifact or
    # status validation while still carrying a superficially valid returncode.
    if status != "finished":
        attestations = []
    report: dict[str, Any] = {
        "schema_version": "1.0", "qualification": "deepmd-3.2-sai-v100", "status": status,
        "job_id": selected_job_id,
        "gpu": gpu_value,
        "environment": environment,
        "failed_commands": failed,
        "missing_commands": missing,
        "unknown_commands": unknown,
        "malformed_commands": malformed,
        "missing_environment": environment_missing,
        "invalid_environment": environment_invalid,
        "invalid_evidence": invalid_evidence,
        "commands": records,
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }
    report["attestations"] = attestations
    # Inspection never freezes the final aggregate. Only the EXIT trap or an
    # explicitly complete external collection may finalize it.
    target = job_dir / "qualification.json"
    if target.exists():
        existing = _load(target)
        if existing.get("job_id") and report["job_id"] and existing["job_id"] != report["job_id"]:
            raise ValueError("qualification JobID does not match recorded JobID")
        if require_complete and existing.get("status") != "finished":
            raise RuntimeError("qualification evidence is incomplete")
        return existing
    if not finalize and not (require_complete and status == "finished"):
        if require_complete and status != "finished":
            raise RuntimeError("qualification evidence is incomplete: " + ", ".join(missing + unknown + malformed + failed + environment_missing + invalid_evidence))
        return report
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
        raise RuntimeError("qualification evidence is incomplete: " + ", ".join(missing + failed + environment_missing + invalid_evidence))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--job-dir", type=Path)
    group.add_argument("--job-ref", type=Path)
    parser.add_argument("--job-id")
    parser.add_argument("--gpu")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args(argv)
    if args.job_ref:
        job_dir, ref_job_id = resolve_job_ref(args.job_ref)
        job_id = args.job_id or ref_job_id
    else:
        job_dir, job_id = args.job_dir, args.job_id
        if not (job_dir / "launch.json").is_file():
            raise ValueError("--job-dir must point to a recorded launch")
        if not args.finalize and not (job_dir / "submission.json").is_file():
            raise ValueError("--job-dir must point to a recorded submission")
    report = collect_qualification(job_dir, job_id=job_id, gpu=args.gpu, require_complete=args.require_complete, finalize=args.finalize)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"qualification collection failed: {exc}", flush=True)
        raise SystemExit(1)
