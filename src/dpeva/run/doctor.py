from __future__ import annotations

import re
import subprocess
import importlib
from collections.abc import Callable, Sequence
from typing import Any, Literal

from packaging.version import InvalidVersion, Version
from pydantic import BaseModel, Field

from dpeva.constants import (
    LEGACY_MIN_DEEPMD_VERSION,
    MAX_DEEPMD_VERSION,
    MIN_DEEPMD_VERSION,
)


DEEPMD_RUNTIME_DETAIL = (
    f"runtime envelope >= {LEGACY_MIN_DEEPMD_VERSION}, < {MAX_DEEPMD_VERSION}"
)
DEEPMD_QUALIFIED_DETAIL = (
    f"qualified 3.2 lane >= {MIN_DEEPMD_VERSION}, < {MAX_DEEPMD_VERSION}"
)


class DoctorCheck(BaseModel):
    model_config = {"extra": "forbid"}
    name: str
    status: Literal[
        "ok", "missing", "incompatible", "error", "unknown", "unavailable", "skipped"
    ]
    version: str | None = None
    detail: str
    # ``None`` means required for backward-compatible DeepMD checks; optional
    # probes set this explicitly to false so the JSON remains self-describing.
    required: bool | None = Field(default=None, exclude_if=lambda value: value is None)


class DoctorReport(BaseModel):
    model_config = {"extra": "forbid"}
    schema_version: Literal["1.0"] = "1.0"
    status: Literal["ok", "failed"]
    checks: list[DoctorCheck]


def probe_deepmd(
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> DoctorCheck:
    """Probe the ``dp`` executable and return a structured compatibility check."""
    try:
        result = run(["dp", "--version"], check=False, text=True, capture_output=True)
    except FileNotFoundError:
        return DoctorCheck(name="deepmd", status="missing", detail="dp executable not found")
    except OSError as exc:
        return DoctorCheck(name="deepmd", status="error", detail=str(exc))

    output = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
    if result.returncode != 0:
        return DoctorCheck(
            name="deepmd",
            status="error",
            detail=output or f"dp exited {result.returncode}",
        )

    match = re.search(r"v?(\d+(?:\.\d+)+(?:[A-Za-z0-9_.!+\-]*)?)", output)
    if match is None:
        return DoctorCheck(
            name="deepmd",
            status="unknown",
            detail=f"unparsed version: {output}",
        )

    raw = match.group(1)
    try:
        parsed = Version(raw)
    except InvalidVersion:
        return DoctorCheck(
            name="deepmd",
            status="unknown",
            version=raw,
            detail=f"unparsed version: {output}",
        )

    compatible = _version_in_lane(parsed, LEGACY_MIN_DEEPMD_VERSION)
    return DoctorCheck(
        name="deepmd",
        status="ok" if compatible else "incompatible",
        version=raw,
        detail=DEEPMD_RUNTIME_DETAIL,
    )


def _version_in_lane(version: Version, minimum: str) -> bool:
    """Return whether a stable version is inside one bounded DeepMD lane."""
    return (
        not version.is_prerelease
        and Version(minimum) <= version < Version(MAX_DEEPMD_VERSION)
    )


def probe_deepmd_qualification(deepmd: DoctorCheck) -> DoctorCheck:
    """Report the qualified 3.2 lane independently of legacy runtime status."""
    if deepmd.version is None or deepmd.status in {"missing", "error", "unknown"}:
        return DoctorCheck(
            name="deepmd.qualified",
            status="skipped",
            version=deepmd.version,
            detail=f"3.2 qualification unavailable: {deepmd.detail}",
            required=False,
        )

    try:
        parsed = Version(deepmd.version)
    except InvalidVersion:
        return DoctorCheck(
            name="deepmd.qualified",
            status="skipped",
            version=deepmd.version,
            detail=f"3.2 qualification unavailable: {deepmd.detail}",
            required=False,
        )

    if _version_in_lane(parsed, MIN_DEEPMD_VERSION):
        status = "ok"
        detail = DEEPMD_QUALIFIED_DETAIL
    elif _version_in_lane(parsed, LEGACY_MIN_DEEPMD_VERSION):
        status = "skipped"
        detail = (
            f"3.2 qualification not claimed for legacy DeepMD {deepmd.version}"
        )
    else:
        status = "incompatible"
        detail = DEEPMD_QUALIFIED_DETAIL
    return DoctorCheck(
        name="deepmd.qualified",
        status=status,
        version=deepmd.version,
        detail=detail,
        required=False,
    )


def build_doctor_report(
    checks: Sequence[DoctorCheck] | None = None,
    *,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    include_optional: bool = True,
    torch_module: Any | None = None,
    cuda_probe: Callable[[Any], bool] | None = None,
) -> DoctorReport:
    """Build a stable capability report.

    ``checks`` is an injection seam for tests and callers that already have a
    controlled observation.  The default path performs all inexpensive,
    explicit probes.  Hardware and optional integrations are informational;
    the required DeepMD CLI and Python data/runtime packages determine the
    report's top-level status.
    """
    observed = list(checks) if checks is not None else _default_checks(
        run=run, include_optional=include_optional, torch_module=torch_module,
        cuda_probe=cuda_probe,
    )
    status = "ok" if all(item.status == "ok" or item.required is False for item in observed) else "failed"
    return DoctorReport(status=status, checks=observed)


def _probe_command(
    name: str,
    command: list[str],
    *,
    run: Callable[..., subprocess.CompletedProcess[str]],
    required: bool = True,
) -> DoctorCheck:
    try:
        result = run(command, check=False, text=True, capture_output=True)
    except FileNotFoundError:
        return DoctorCheck(name=name, status="missing", detail=f"executable not found: {command[0]}", required=required)
    except OSError as exc:
        return DoctorCheck(name=name, status="error", detail=str(exc), required=required)
    output = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
    if result.returncode != 0:
        return DoctorCheck(name=name, status="error", detail=output or f"exited {result.returncode}", required=required)
    return DoctorCheck(name=name, status="ok", detail=output.splitlines()[0] if output else "command completed", required=required)


def probe_deepmd_operations(
    *,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    required: bool = True,
) -> list[DoctorCheck]:
    """Check the CLI surfaces consumed by the feature/infer pilot."""
    return [
        _probe_command(
            f"deepmd.cli.{operation}", ["dp", operation, "-h"], run=run,
            required=required,
        )
        for operation in ("test", "eval-desc", "embed")
    ]


def _probe_python_package(name: str, *, required: bool) -> DoctorCheck:
    try:
        module = importlib.import_module(name)
    except (ImportError, ModuleNotFoundError) as exc:
        return DoctorCheck(name=name, status="missing", detail=f"package unavailable: {exc}", required=required)
    except Exception as exc:
        return DoctorCheck(name=name, status="error", detail=f"package probe failed: {exc}", required=required)
    version = getattr(module, "__version__", None)
    return DoctorCheck(
        name=name,
        status="ok",
        version=str(version) if version is not None else None,
        detail="import succeeded",
        required=required,
    )


def _probe_torch_cuda(
    torch_module: Any | None = None,
    cuda_probe: Callable[[Any], bool] | None = None,
) -> DoctorCheck:
    try:
        torch = torch_module if torch_module is not None else importlib.import_module("torch")
    except (ImportError, ModuleNotFoundError) as exc:
        return DoctorCheck(name="torch.cuda", status="missing", detail=f"torch unavailable: {exc}", required=False)
    except Exception as exc:
        return DoctorCheck(name="torch.cuda", status="error", detail=f"torch CUDA probe failed: {exc}", required=False)
    available = bool(cuda_probe(torch) if cuda_probe is not None else torch.cuda.is_available())
    return DoctorCheck(
        name="torch.cuda",
        status="ok" if available else "unavailable",
        version=getattr(torch.version, "cuda", None),
        detail="CUDA runtime available" if available else "CUDA runtime unavailable; CPU usage remains supported",
        required=False,
    )


def _default_checks(
    *,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    include_optional: bool = True,
    torch_module: Any | None = None,
    cuda_probe: Callable[[Any], bool] | None = None,
) -> list[DoctorCheck]:
    deepmd = probe_deepmd(run=run)
    checks = [deepmd, probe_deepmd_qualification(deepmd)]
    operations_required = True
    if deepmd.status == "ok" and deepmd.version is not None:
        try:
            operations_required = _version_in_lane(
                Version(deepmd.version), MIN_DEEPMD_VERSION
            )
        except InvalidVersion:
            pass
    checks.extend(probe_deepmd_operations(run=run, required=operations_required))
    checks.append(_probe_python_package("dpdata", required=True))
    checks.append(_probe_python_package("torch", required=True))
    checks.append(_probe_torch_cuda(torch_module=torch_module, cuda_probe=cuda_probe))
    gpu = _probe_command("gpu.visibility", ["nvidia-smi", "-L"], run=run, required=False)
    if gpu.status == "missing":
        gpu.status = "unavailable"
        gpu.detail = "nvidia-smi unavailable; CPU usage remains supported"
    checks.append(gpu)
    if include_optional:
        for package in ("jax", "tensorflow", "mpi4py"):
            checks.append(_probe_python_package(package, required=False))
    return checks
