from __future__ import annotations

import re
import subprocess
from collections.abc import Callable, Sequence

from packaging.version import InvalidVersion, Version
from pydantic import BaseModel

from dpeva.constants import MAX_DEEPMD_VERSION, MIN_DEEPMD_VERSION


class DoctorCheck(BaseModel):
    name: str
    status: str
    version: str | None = None
    detail: str


class DoctorReport(BaseModel):
    schema_version: str = "1.0"
    status: str
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

    compatible = parsed.is_devrelease or (
        Version(MIN_DEEPMD_VERSION) <= parsed < Version(MAX_DEEPMD_VERSION)
    )
    return DoctorCheck(
        name="deepmd",
        status="ok" if compatible else "incompatible",
        version=raw,
        detail=f"required >= {MIN_DEEPMD_VERSION}, < {MAX_DEEPMD_VERSION}",
    )


def build_doctor_report(checks: Sequence[DoctorCheck] | None = None) -> DoctorReport:
    """Build a stable report from supplied checks or the default DeepMD probe."""
    observed = list(checks) if checks is not None else [probe_deepmd()]
    status = "ok" if all(item.status == "ok" for item in observed) else "failed"
    return DoctorReport(status=status, checks=observed)
