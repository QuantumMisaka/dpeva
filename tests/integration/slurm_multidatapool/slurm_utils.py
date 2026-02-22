import os
import re
import time
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Mapping

from ..utils.logs import wait_for_text_in_file


@dataclass(frozen=True)
class SlurmSubmission:
    job_id: Optional[str]
    stdout: str
    stderr: str


def _run(
    args: Sequence[str],
    cwd: Path,
    env: Optional[Mapping[str, str]] = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update({k: str(v) for k, v in env.items()})
    return subprocess.run(
        list(args),
        cwd=str(cwd),
        env=merged_env,
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def submit_cli_and_capture_job_id(
    cli_args: Sequence[str],
    cwd: Path,
    env: Optional[Mapping[str, str]] = None,
) -> SlurmSubmission:
    proc = _run(cli_args, cwd=cwd, env=env, check=True)
    job_id = _parse_job_id(proc.stdout) or _parse_job_id(proc.stderr)
    return SlurmSubmission(job_id=job_id, stdout=proc.stdout, stderr=proc.stderr)


def _parse_job_id(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"Submitted batch job\s+(\d+)", text)
    if not m:
        return None
    return m.group(1)


def wait_for_job_gone_from_queue(
    job_id: str,
    timeout_s: float,
    poll_s: float = 10.0,
    cwd: Optional[Path] = None,
) -> None:
    start = time.time()
    cwd = cwd or Path.cwd()
    while True:
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Timeout waiting for job {job_id} to leave queue")
        proc = _run(["squeue", "-j", job_id, "-h"], cwd=cwd, check=False)
        if proc.returncode != 0:
            return
        if not proc.stdout.strip():
            return
        time.sleep(poll_s)
