#!/usr/bin/env python3
"""Measure local fake-command overhead from creating a run manifest."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Sequence

from dpeva.run.recorder import StatusRecorder


def percentile(values: Sequence[float], percent: float) -> float:
    """Return the nearest-rank percentile (1-based rank, no interpolation)."""
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0 < percent <= 100:
        raise ValueError("percent must be in (0, 100]")
    ordered = sorted(values)
    rank = max(1, math.ceil(percent / 100 * len(ordered)))
    return ordered[rank - 1]


def run_benchmark(repetitions: int = 41, warmups: int = 5) -> dict[str, object]:
    if repetitions < 1 or warmups < 0:
        raise ValueError("repetitions must be positive and warmups non-negative")
    command = [sys.executable, "-c", "pass"]
    with tempfile.TemporaryDirectory(prefix="dpeva-manifest-benchmark-") as temporary:
        root = Path(temporary)
        for _ in range(warmups):
            subprocess.run(command, check=True, capture_output=True)
        baseline: list[float] = []
        with_manifest: list[float] = []
        overhead: list[float] = []
        for index in range(repetitions):
            started = time.perf_counter_ns()
            subprocess.run(command, check=True, capture_output=True)
            baseline_ms = (time.perf_counter_ns() - started) / 1_000_000

            run_dir = root / f"run-{index}"
            run_dir.mkdir()
            started = time.perf_counter_ns()
            subprocess.run(command, check=True, capture_output=True)
            StatusRecorder.create(run_dir / "run.json", f"run-{index}", "feature")
            with_manifest_ms = (time.perf_counter_ns() - started) / 1_000_000

            baseline.append(round(baseline_ms, 3))
            with_manifest.append(round(with_manifest_ms, 3))
            overhead.append(round(with_manifest_ms - baseline_ms, 3))
    return {
        "repetitions": repetitions,
        "warmups": warmups,
        "command": command,
        "percentile_method": "nearest-rank (ceil(p/100*n), 1-based)",
        "baseline_ms": baseline,
        "with_manifest_ms": with_manifest,
        "overhead_ms": overhead,
        "median_baseline_ms": round(percentile(baseline, 50), 3),
        "median_with_manifest_ms": round(percentile(with_manifest, 50), 3),
        "median_overhead_ms": round(percentile(overhead, 50), 3),
        "p95_overhead_ms": round(percentile(overhead, 95), 3),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=41)
    parser.add_argument("--warmups", type=int, default=5)
    args = parser.parse_args(argv)
    print(json.dumps(run_benchmark(args.repetitions, args.warmups), separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
