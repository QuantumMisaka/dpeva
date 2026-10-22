#!/usr/bin/env python3
"""Run and compare one real ``dp test`` npy/LMDB A/B pair.

This script is intentionally an evidence producer, not a workflow component.
It runs both formats in one process, preserves the raw command logs, and emits
the bounded metric comparison used by the LMDB compatibility report.  Raw logs
and detail files remain in the requested output directory; the JSON summary
contains paths and hashes so a sanitized public copy can be reviewed separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable


METRIC_PATTERNS = {
    "energy_mae_per_atom": re.compile(
        r"Energy\s+MAE/Natoms\s*:\s*([0-9.eE+-]+)"
    ),
    "force_mae": re.compile(r"Force\s+MAE\s*:\s*([0-9.eE+-]+)"),
    "virial_mae_per_atom": re.compile(
        r"Virial\s+MAE/Natoms\s*:\s*([0-9.eE+-]+)"
    ),
    "stress_mae": re.compile(r"Stress\s+MAE\s*:\s*([0-9.eE+-]+)"),
}


def parse_aggregate_metrics(text: str) -> dict[str, float]:
    """Parse the final weighted-average block from ``dp test`` output."""

    lines = text.splitlines()
    aggregate_starts = [
        index
        for index, line in enumerate(lines)
        if "# number of systems" in line
    ]
    if not aggregate_starts:
        raise ValueError("dp test output has no '# number of systems' aggregate block")
    block = lines[aggregate_starts[-1] :]
    values: dict[str, float] = {}
    for name, pattern in METRIC_PATTERNS.items():
        for line in block:
            match = pattern.search(line)
            if match:
                values[name] = float(match.group(1))
                break
        if name not in values:
            raise ValueError(f"dp test aggregate block is missing {name}")
    return values


def compare_metrics(
    npy: dict[str, float], lmdb: dict[str, float], *, threshold: float
) -> dict[str, dict[str, float | bool]]:
    """Return absolute and relative differences for each required metric."""

    result: dict[str, dict[str, float | bool]] = {}
    for name in METRIC_PATTERNS:
        baseline = float(npy[name])
        candidate = float(lmdb[name])
        absolute = abs(baseline - candidate)
        relative = absolute / max(abs(baseline), abs(candidate), 1e-30)
        result[name] = {
            "npy": baseline,
            "lmdb": candidate,
            "absolute_diff": absolute,
            "relative_diff": relative,
            "passed": relative <= threshold,
        }
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        return sha256_file(path)
    for child in sorted(path.rglob("*")):
        if not child.is_file():
            continue
        digest.update(str(child.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(child)))
    return digest.hexdigest()


def _run_command(argv: list[str], log_path: Path) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(argv, capture_output=True, text=True, check=False)
    log_path.write_text(
        "COMMAND: " + " ".join(argv) + "\n\nSTDOUT\n" + result.stdout + "\nSTDERR\n" + result.stderr,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed with return code {result.returncode}: {' '.join(argv)}"
        )
    return result


def _detail_hashes(output_dir: Path, prefix: str) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for suffix in (".e_peratom.out", ".f.out", ".v_peratom.out", ".s.out"):
        path = output_dir / f"{prefix}{suffix}"
        if path.is_file():
            hashes[path.name] = sha256_file(path)
    return hashes


def _verify_lmdb_detail(
    output_dir: Path, prefix: str, lmdb_path: Path
) -> dict[str, object]:
    from dpeva.io.dataproc import DPTestResultParser
    from dpeva.io.lmdb_index import read_lmdb_frame_index

    index = read_lmdb_frame_index(lmdb_path)
    parsed = DPTestResultParser(
        str(output_dir),
        head=prefix,
        testdata_dir=str(lmdb_path),
    ).parse()
    if parsed["dataname_list"] != index.dataname_list:
        raise AssertionError("LMDB detail attribution does not match the frame index")
    return {
        "nframes": len(index.frames),
        "nsystems": len({frame.system_id for frame in index.frames}),
        "datanames_nframe": parsed["datanames_nframe"],
    }


def run_ab(
    *,
    npy_path: Path,
    lmdb_path: Path,
    model_path: Path,
    dp_executable: str,
    output_dir: Path,
    threshold: float,
    job_id: str | None = None,
    head: str | None = None,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    common = [dp_executable, "--pt", "test", "-m", str(model_path), "-n", "0"]
    if head:
        common.extend(["--head", head])
    npy_prefix = "npy-detail"
    lmdb_prefix = "lmdb-detail"
    npy_cmd = [*common, "-s", str(npy_path), "-d", str(output_dir / npy_prefix)]
    lmdb_cmd = [*common, "-s", str(lmdb_path), "-d", str(output_dir / lmdb_prefix)]
    npy_result = _run_command(npy_cmd, output_dir / "npy.log")
    lmdb_result = _run_command(lmdb_cmd, output_dir / "lmdb.log")
    npy_metrics = parse_aggregate_metrics(npy_result.stdout + npy_result.stderr)
    lmdb_metrics = parse_aggregate_metrics(lmdb_result.stdout + lmdb_result.stderr)
    comparison = compare_metrics(npy_metrics, lmdb_metrics, threshold=threshold)
    lmdb_detail = _verify_lmdb_detail(output_dir, lmdb_prefix, lmdb_path)
    passed = all(bool(item["passed"]) for item in comparison.values())
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "status": "finished" if passed else "failed",
        "job_id": job_id,
        "threshold": threshold,
        "commands": {"npy": npy_cmd, "lmdb": lmdb_cmd},
        "inputs": {
            "npy": str(npy_path),
            "lmdb": str(lmdb_path),
            "model": str(model_path),
        },
        "sha256": {
            "npy_tree": sha256_tree(npy_path),
            "lmdb_tree": sha256_tree(lmdb_path),
            "model": sha256_file(model_path),
        },
        "metrics": {
            "npy": npy_metrics,
            "lmdb": lmdb_metrics,
            "comparison": comparison,
        },
        "detail_sha256": {
            "npy": _detail_hashes(output_dir, npy_prefix),
            "lmdb": _detail_hashes(output_dir, lmdb_prefix),
        },
        "lmdb_detail_attribution": lmdb_detail,
    }
    (output_dir / "ab_result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npy", type=Path, required=True)
    parser.add_argument("--lmdb", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dp", default="dp")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=2e-5)
    parser.add_argument("--job-id")
    parser.add_argument("--head")
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        payload = run_ab(
            npy_path=args.npy,
            lmdb_path=args.lmdb,
            model_path=args.model,
            dp_executable=args.dp,
            output_dir=args.output_dir,
            threshold=args.threshold,
            job_id=args.job_id,
            head=args.head,
        )
    except Exception as exc:
        print(f"LMDB A/B failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": payload["status"], "result": str(args.output_dir / "ab_result.json")}))
    return 0 if payload["status"] == "finished" else 1


if __name__ == "__main__":
    raise SystemExit(main())
