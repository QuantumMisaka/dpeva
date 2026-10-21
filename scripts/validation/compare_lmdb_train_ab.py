#!/usr/bin/env python3
"""Compare two bounded ``dp train`` runs and emit LMDB A/B evidence JSON."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable

try:
    from scripts.validation.compare_lmdb_ab import sha256_file, sha256_tree
except ModuleNotFoundError:  # direct script execution
    from compare_lmdb_ab import sha256_file, sha256_tree  # type: ignore[no-redef]


LOSS_FIELDS = (
    "mae_e_val",
    "mae_e_trn",
    "mae_f_val",
    "mae_f_trn",
    "mae_v_val",
    "mae_v_trn",
    "rmse_val",
    "rmse_trn",
)
WALL_PATTERN = re.compile(
    r"Batch\s+(\d+):\s+total wall time\s*=\s*([0-9.]+)\s*s"
)


def parse_lcurve(path: Path) -> dict[int, dict[str, float]]:
    """Parse the compact ``lcurve.out`` table emitted by DeepMD."""

    rows: dict[int, dict[str, float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 10:
            raise ValueError(f"unexpected lcurve row: {line!r}")
        try:
            step = int(parts[0])
            values = [float(item) for item in parts[1:10]]
        except ValueError as exc:
            raise ValueError(f"non-numeric lcurve row: {line!r}") from exc
        rows[step] = {
            "mae_e_val": values[0],
            "mae_e_trn": values[1],
            "mae_f_val": values[2],
            "mae_f_trn": values[3],
            "mae_v_val": values[4],
            "mae_v_trn": values[5],
            "rmse_val": values[6],
            "rmse_trn": values[7],
            "lr": values[8],
        }
    if 200 not in rows:
        raise ValueError(f"lcurve has no step 200: {path}")
    return rows


def parse_wall_times(path: Path) -> dict[int, float]:
    times: dict[int, float] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = WALL_PATTERN.search(line)
        if match:
            times[int(match.group(1))] = float(match.group(2))
    if 100 not in times or 200 not in times:
        raise ValueError(f"training log lacks wall time for steps 100/200: {path}")
    return times


def _relative(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-30)


def compare_train_metrics(
    mixed: dict[int, dict[str, float]],
    lmdb: dict[int, dict[str, float]],
    mixed_wall: dict[int, float],
    lmdb_wall: dict[int, float],
    *,
    loss_threshold: float,
    wall_threshold: float,
) -> dict[str, object]:
    losses: dict[str, object] = {}
    for field in LOSS_FIELDS:
        a = mixed[200][field]
        b = lmdb[200][field]
        relative = _relative(a, b)
        losses[field] = {
            "mixed": a,
            "lmdb": b,
            "relative_diff": relative,
            "passed": relative <= loss_threshold,
        }
    wall: dict[str, object] = {}
    for step in (100, 200):
        a = mixed_wall[step]
        b = lmdb_wall[step]
        relative = _relative(a, b)
        wall[str(step)] = {
            "mixed_s": a,
            "lmdb_s": b,
            "relative_diff": relative,
            "passed": relative <= wall_threshold,
        }
    return {
        "loss_at_step_200": losses,
        "wall_time": wall,
        "loss_threshold": loss_threshold,
        "wall_threshold": wall_threshold,
        "passed": all(bool(item["passed"]) for item in losses.values())
        and all(bool(item["passed"]) for item in wall.values()),
    }


def build_result(
    *,
    mixed_config: Path,
    lmdb_config: Path,
    mixed_data: Path,
    lmdb_data: Path,
    stat_file: Path,
    mixed_log: Path,
    lmdb_log: Path,
    mixed_lcurve: Path,
    lmdb_lcurve: Path,
    mixed_models: Path,
    lmdb_models: Path,
    loss_threshold: float = 0.05,
    wall_threshold: float = 0.10,
    job_id: str | None = None,
    execution_only: bool = False,
) -> dict[str, object]:
    mixed_metrics = parse_lcurve(mixed_lcurve)
    lmdb_metrics = parse_lcurve(lmdb_lcurve)
    mixed_wall = parse_wall_times(mixed_log)
    lmdb_wall = parse_wall_times(lmdb_log)
    comparison = compare_train_metrics(
        mixed_metrics,
        lmdb_metrics,
        mixed_wall,
        lmdb_wall,
        loss_threshold=loss_threshold,
        wall_threshold=wall_threshold,
    )
    mixed_checkpoint = mixed_models / "model.ckpt-200.pt"
    lmdb_checkpoint = lmdb_models / "model.ckpt-200.pt"
    hashes = {
        "mixed_config": sha256_file(mixed_config),
        "lmdb_config": sha256_file(lmdb_config),
        "mixed_data": sha256_tree(mixed_data),
        "lmdb_data": sha256_tree(lmdb_data),
        "stat_file": sha256_file(stat_file),
        "mixed_log": sha256_file(mixed_log),
        "lmdb_log": sha256_file(lmdb_log),
        "mixed_lcurve": sha256_file(mixed_lcurve),
        "lmdb_lcurve": sha256_file(lmdb_lcurve),
        "mixed_checkpoint_200": sha256_file(mixed_checkpoint)
        if mixed_checkpoint.is_file()
        else None,
        "lmdb_checkpoint_200": sha256_file(lmdb_checkpoint)
        if lmdb_checkpoint.is_file()
        else None,
    }
    execution_complete = mixed_checkpoint.is_file() and lmdb_checkpoint.is_file()
    status = (
        "finished"
        if execution_complete and (execution_only or comparison["passed"])
        else "failed"
    )
    return {
        "schema_version": "1.0",
        "status": status,
        "job_id": job_id,
        "step": 200,
        "execution_only": execution_only,
        "metric_equivalence_claimed": not execution_only,
        "metric_equivalence_note": (
            "Not claimed: 200-step training is a stochastic execution smoke. "
            "Loss trajectory differences are recorded for review, not used as "
            "a bitwise format-equivalence gate."
            if execution_only
            else "Loss and wall-time thresholds are applied as declared."
        ),
        "metrics": {
            "mixed": mixed_metrics,
            "lmdb": lmdb_metrics,
            "comparison": comparison,
        },
        "wall_time": {"mixed": mixed_wall, "lmdb": lmdb_wall},
        "sha256": hashes,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mixed-config", type=Path, required=True)
    parser.add_argument("--lmdb-config", type=Path, required=True)
    parser.add_argument("--mixed-data", type=Path, required=True)
    parser.add_argument("--lmdb-data", type=Path, required=True)
    parser.add_argument("--stat-file", type=Path, required=True)
    parser.add_argument("--mixed-log", type=Path, required=True)
    parser.add_argument("--lmdb-log", type=Path, required=True)
    parser.add_argument("--mixed-lcurve", type=Path, required=True)
    parser.add_argument("--lmdb-lcurve", type=Path, required=True)
    parser.add_argument("--mixed-models", type=Path, required=True)
    parser.add_argument("--lmdb-models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--loss-threshold", type=float, default=0.05)
    parser.add_argument("--wall-threshold", type=float, default=0.10)
    parser.add_argument("--job-id")
    parser.add_argument(
        "--execution-only",
        action="store_true",
        help="require both commands/checkpoints, but keep loss differences informational",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        result = build_result(
            mixed_config=args.mixed_config,
            lmdb_config=args.lmdb_config,
            mixed_data=args.mixed_data,
            lmdb_data=args.lmdb_data,
            stat_file=args.stat_file,
            mixed_log=args.mixed_log,
            lmdb_log=args.lmdb_log,
            mixed_lcurve=args.mixed_lcurve,
            lmdb_lcurve=args.lmdb_lcurve,
            mixed_models=args.mixed_models,
            lmdb_models=args.lmdb_models,
            loss_threshold=args.loss_threshold,
            wall_threshold=args.wall_threshold,
            job_id=args.job_id,
            execution_only=args.execution_only,
        )
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"LMDB train A/B failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": result["status"], "result": str(args.output)}))
    return 0 if result["status"] == "finished" else 1


if __name__ == "__main__":
    raise SystemExit(main())
