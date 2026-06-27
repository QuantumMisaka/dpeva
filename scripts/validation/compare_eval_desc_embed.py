#!/usr/bin/env python3
"""Compare DeepMD eval-desc descriptors with embed HDF5 descriptors."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


@dataclass(frozen=True)
class Case:
    name: str
    model: Path
    system: Path
    source: str
    stress: bool = False


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def conda_shell_command(env_name: str, args: list[str]) -> list[str]:
    candidates = [
        Path("/opt/devtools/anaconda3/etc/profile.d/conda.sh"),
        Path.home() / ".conda" / "etc" / "profile.d" / "conda.sh",
        Path("/opt/conda/etc/profile.d/conda.sh"),
    ]
    conda_sh = next((path for path in candidates if path.exists()), candidates[0])
    inner = " ".join(shlex.quote(str(part)) for part in args)
    return [
        "bash",
        "-lc",
        f"source {shlex.quote(str(conda_sh))} && conda activate {shlex.quote(env_name)} && {inner}",
    ]


def query_gpu_memory_mib() -> list[int]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except Exception:
        return []
    values = []
    for line in output.splitlines():
        line = line.strip()
        if line:
            try:
                values.append(int(line))
            except ValueError:
                pass
    return values


def parse_time_v(stderr_text: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    patterns = {
        "max_rss_kb": r"Maximum resident set size \(kbytes\):\s*(\d+)",
        "user_seconds": r"User time \(seconds\):\s*([0-9.]+)",
        "system_seconds": r"System time \(seconds\):\s*([0-9.]+)",
        "exit_status_time": r"Exit status:\s*(\d+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, stderr_text)
        if not match:
            continue
        value = match.group(1)
        metrics[key] = float(value) if "." in value else int(value)
    return metrics


def run_monitored(
    name: str,
    cmd: list[str],
    cwd: Path,
    log_dir: Path,
    timeout_s: int,
) -> dict[str, Any]:
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{name}.stdout"
    stderr_path = log_dir / f"{name}.stderr"
    samples: list[dict[str, Any]] = []
    stop_event = threading.Event()
    baseline = query_gpu_memory_mib()
    peak_mib = max(baseline) if baseline else None

    def monitor() -> None:
        nonlocal peak_mib
        while not stop_event.is_set():
            values = query_gpu_memory_mib()
            if values:
                current = max(values)
                peak_mib = current if peak_mib is None else max(peak_mib, current)
                samples.append(
                    {
                        "t": time.time(),
                        "memory_used_mib": values,
                    }
                )
            time.sleep(0.5)

    full_cmd = ["/usr/bin/time", "-v"] + cmd
    start = time.monotonic()
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    proc = subprocess.Popen(
        full_cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout_text, stderr_text = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            stdout_text, stderr_text = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            stdout_text, stderr_text = proc.communicate()
    finally:
        stop_event.set()
        thread.join(timeout=5)
    wall_seconds = time.monotonic() - start

    stdout_path.write_text(stdout_text, encoding="utf-8")
    stderr_path.write_text(stderr_text, encoding="utf-8")

    baseline_peak = max(baseline) if baseline else None
    result = {
        "name": name,
        "command": full_cmd,
        "returncode": proc.returncode,
        "timed_out": timed_out,
        "wall_seconds": wall_seconds,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "gpu_baseline_mib": baseline_peak,
        "gpu_peak_mib": peak_mib,
        "gpu_peak_delta_mib": (
            peak_mib - baseline_peak
            if peak_mib is not None and baseline_peak is not None
            else None
        ),
        "gpu_samples": samples,
    }
    result.update(parse_time_v(stderr_text))
    return result


def find_single_npy(desc_dir: Path) -> Path:
    files = sorted(desc_dir.rglob("*.npy"))
    if len(files) != 1:
        raise RuntimeError(f"Expected one descriptor npy in {desc_dir}, found {len(files)}")
    return files[0]


def load_embed_descriptor(hdf5_path: Path, case_name: str) -> tuple[str, np.ndarray]:
    with h5py.File(hdf5_path, "r") as h5:
        keys = sorted(h5.keys())
        if len(keys) == 1:
            key = keys[0]
        else:
            matching = [k for k in keys if k == case_name or Path(k).name == case_name]
            if len(matching) != 1:
                raise RuntimeError(
                    f"Could not match HDF5 group for {case_name}; groups={keys}"
                )
            key = matching[0]
        return key, np.asarray(h5[key]["descriptor"])


def compare_arrays(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        return {
            "shape_match": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
        }
    ref64 = reference.astype(np.float64, copy=False)
    cand64 = candidate.astype(np.float64, copy=False)
    diff = cand64 - ref64
    abs_diff = np.abs(diff)
    scale = np.maximum(np.abs(ref64), 1e-12)
    rel_diff = abs_diff / scale
    max_abs = float(abs_diff.max()) if abs_diff.size else 0.0
    max_rel = float(rel_diff.max()) if rel_diff.size else 0.0
    rms_abs = float(np.sqrt(np.mean(diff * diff))) if diff.size else 0.0
    significant_digits = None
    if max_rel > 0:
        significant_digits = max(0.0, -math.log10(max_rel))
    return {
        "shape_match": True,
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "reference_dtype": str(reference.dtype),
        "candidate_dtype": str(candidate.dtype),
        "max_abs": max_abs,
        "mean_abs": float(abs_diff.mean()) if abs_diff.size else 0.0,
        "rms_abs": rms_abs,
        "max_rel": max_rel,
        "mean_rel": float(rel_diff.mean()) if rel_diff.size else 0.0,
        "significant_digits_by_max_rel": significant_digits,
        "allclose_rtol_1e-5_atol_1e-6": bool(
            np.allclose(ref64, cand64, rtol=1e-5, atol=1e-6)
        ),
        "allclose_rtol_1e-5_atol_3e-5": bool(
            np.allclose(ref64, cand64, rtol=1e-5, atol=3e-5)
        ),
        "allclose_rtol_1e-4_atol_1e-5": bool(
            np.allclose(ref64, cand64, rtol=1e-4, atol=1e-5)
        ),
        "allclose_rtol_1e-6_atol_1e-8": bool(
            np.allclose(ref64, cand64, rtol=1e-6, atol=1e-8)
        ),
    }


def dataset_summary(system: Path) -> dict[str, Any]:
    coord_files = sorted(system.glob("set.*/coord.npy"))
    frame_count = 0
    atom_counts = set()
    set_shapes = {}
    for path in coord_files:
        arr = np.load(path, mmap_mode="r")
        frame_count += int(arr.shape[0])
        atom_counts.add(int(arr.shape[1] // 3))
        set_shapes[str(path.relative_to(system))] = list(arr.shape)
    return {
        "system": str(system),
        "sets": set_shapes,
        "frames": frame_count,
        "atom_counts": sorted(atom_counts),
    }


def write_markdown_summary(path: Path, results: dict[str, Any]) -> None:
    lines = [
        "# DP embed vs eval-desc raw benchmark",
        "",
        f"- Generated at: `{results['generated_at']}`",
        f"- Old eval-desc environment: `{results['old_env']}`",
        f"- New embed environment: `{results['new_env']}`",
        f"- Timeout per command: `{results['timeout_s']} s`",
        "",
        "## Cases",
        "",
        "| case | source | frames | atoms | model |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for case in results["cases"]:
        summary = case["dataset"]
        lines.append(
            "| {name} | {source} | {frames} | {atoms} | `{model}` |".format(
                name=case["name"],
                source=case["source"],
                frames=summary["frames"],
                atoms=", ".join(str(x) for x in summary["atom_counts"]),
                model=case["model"],
            )
        )

    lines.extend(
        [
            "",
            "## Descriptor Comparisons",
            "",
            "| case | compare | shape | dtypes | max abs | mean abs | max rel | allclose 1e-5/1e-6 |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for case in results["cases"]:
        for label, comp in case.get("comparisons", {}).items():
            if not isinstance(comp, dict) or "shape_match" not in comp:
                continue
            if not comp.get("shape_match"):
                lines.append(
                    f"| {case['name']} | {label} | mismatch | - | - | - | - | false |"
                )
                continue
            shape = "x".join(str(x) for x in comp["reference_shape"])
            dtypes = f"{comp['reference_dtype']} / {comp['candidate_dtype']}"
            lines.append(
                "| {case} | {label} | {shape} | {dtypes} | {max_abs:.6e} | {mean_abs:.6e} | {max_rel:.6e} | {allclose} |".format(
                    case=case["name"],
                    label=label,
                    shape=shape,
                    dtypes=dtypes,
                    max_abs=comp["max_abs"],
                    mean_abs=comp["mean_abs"],
                    max_rel=comp["max_rel"],
                    allclose=comp["allclose_rtol_1e-5_atol_1e-6"],
                )
            )

    lines.extend(
        [
            "",
            "## Runtime And Memory",
            "",
            "| case | method | status | wall s | max RSS GiB | GPU peak delta MiB | output size MiB |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for case in results["cases"]:
        for method, metric in case.get("metrics", {}).items():
            status = "timeout" if metric.get("timed_out") else str(metric.get("returncode"))
            rss_gib = metric.get("max_rss_kb")
            rss_gib = rss_gib / 1024 / 1024 if rss_gib is not None else None
            gpu_delta = metric.get("gpu_peak_delta_mib")
            output_size = metric.get("output_size_mib")
            lines.append(
                "| {case} | {method} | {status} | {wall:.3f} | {rss} | {gpu} | {out} |".format(
                    case=case["name"],
                    method=method,
                    status=status,
                    wall=metric.get("wall_seconds", float("nan")),
                    rss=f"{rss_gib:.3f}" if rss_gib is not None else "-",
                    gpu=f"{gpu_delta:.0f}" if gpu_delta is not None else "-",
                    out=f"{output_size:.3f}" if output_size is not None else "-",
                )
            )

    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def path_size_mib(path: Path) -> float:
    if path.is_file():
        return path.stat().st_size / 1024 / 1024
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total / 1024 / 1024


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument("--old-env", default="dpeva-dpa4-test")
    parser.add_argument("--new-env", default="dpeva-dpa4-embed-test")
    parser.add_argument("--stress", action="store_true")
    args = parser.parse_args()

    root = repo_root()
    out_root = args.output_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    cases = [
        Case(
            name=f"water_data_{idx}",
            source="test/deepmd-kit examples/water/data",
            model=root / "test/deepmd-kit/examples/water/dpa4/lmp/pretrained.pt",
            system=root / f"test/deepmd-kit/examples/water/data/data_{idx}",
        )
        for idx in range(4)
    ]
    if args.stress:
        cases.append(
            Case(
                name="dpa4_plus_oom_216",
                source="practices previous OOM reproducer",
                model=root
                / "practices/dpeva-dpa4-test/dpa4-dpeva-test/DPA4-Plus-OMat24-16M.pt",
                system=root
                / "practices/dpeva-dpa4-test/dpa4-dpeva-test/issue_dpa4_plus_eval_desc_oom/sampled_dpdata/216",
                stress=True,
            )
        )

    results: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "old_env": args.old_env,
        "new_env": args.new_env,
        "timeout_s": args.timeout_s,
        "cases": [],
    }

    for case in cases:
        case_dir = out_root / case.name
        case_dir.mkdir(parents=True, exist_ok=True)
        log_dir = case_dir / "logs"
        old_dir = case_dir / "eval_desc"
        new_eval_native_dir = case_dir / "eval_desc_new_native"
        new_eval_fp32_dir = case_dir / "eval_desc_new_fp32"
        embed_native = case_dir / "embed_native.hdf5"
        embed_fp32 = case_dir / "embed_fp32.hdf5"
        metrics: dict[str, Any] = {}
        comparisons: dict[str, Any] = {}

        old_cmd = conda_shell_command(
            args.old_env,
            [
                "dp",
                "--pt",
                "eval-desc",
                "-m",
                str(case.model),
                "-s",
                str(case.system),
                "-o",
                str(old_dir),
            ],
        )
        metrics["eval_desc_old_native"] = run_monitored(
            "eval_desc_old_native", old_cmd, root, log_dir, args.timeout_s
        )
        if old_dir.exists():
            metrics["eval_desc_old_native"]["output_size_mib"] = path_size_mib(old_dir)

        for dtype, output_dir in [("native", new_eval_native_dir), ("fp32", new_eval_fp32_dir)]:
            eval_new_cmd = conda_shell_command(
                args.new_env,
                [
                    "dp",
                    "--pt",
                    "eval-desc",
                    "-m",
                    str(case.model),
                    "-s",
                    str(case.system),
                    "-o",
                    str(output_dir),
                    "--dtype",
                    dtype,
                ],
            )
            key = f"eval_desc_new_{dtype}"
            metrics[key] = run_monitored(key, eval_new_cmd, root, log_dir, args.timeout_s)
            if output_dir.exists():
                metrics[key]["output_size_mib"] = path_size_mib(output_dir)

        for dtype, output_path in [("native", embed_native), ("fp32", embed_fp32)]:
            embed_cmd = conda_shell_command(
                args.new_env,
                [
                    "dp",
                    "--pt",
                    "embed",
                    "-m",
                    str(case.model),
                    "-s",
                    str(case.system),
                    "-o",
                    str(output_path),
                    "--dtype",
                    dtype,
                ],
            )
            key = f"embed_new_{dtype}"
            metrics[key] = run_monitored(key, embed_cmd, root, log_dir, args.timeout_s)
            if output_path.exists():
                metrics[key]["output_size_mib"] = path_size_mib(output_path)

        if metrics["eval_desc_old_native"].get("returncode") == 0:
            eval_npy = find_single_npy(old_dir)
            eval_desc = np.load(eval_npy)
        else:
            eval_npy = None
            eval_desc = None

        if (
            eval_desc is not None
            and metrics["embed_new_native"].get("returncode") == 0
            and metrics["embed_new_fp32"].get("returncode") == 0
        ):
            native_group, native_desc = load_embed_descriptor(embed_native, case.system.name)
            fp32_group, fp32_desc = load_embed_descriptor(embed_fp32, case.system.name)
            comparisons["embed_native_vs_eval_desc_old_native"] = compare_arrays(
                eval_desc, native_desc
            )
            comparisons["embed_fp32_vs_eval_desc_old_native"] = compare_arrays(
                eval_desc, fp32_desc
            )
            comparisons["hdf5_groups"] = {
                "native": native_group,
                "fp32": fp32_group,
                "eval_desc_npy": str(eval_npy),
            }

        if (
            metrics["eval_desc_new_native"].get("returncode") == 0
            and metrics["embed_new_native"].get("returncode") == 0
        ):
            eval_new_native = np.load(find_single_npy(new_eval_native_dir))
            _, embed_native_desc = load_embed_descriptor(embed_native, case.system.name)
            comparisons["embed_native_vs_eval_desc_new_native"] = compare_arrays(
                eval_new_native, embed_native_desc
            )

        if (
            metrics["eval_desc_new_fp32"].get("returncode") == 0
            and metrics["embed_new_fp32"].get("returncode") == 0
        ):
            eval_new_fp32 = np.load(find_single_npy(new_eval_fp32_dir))
            _, embed_fp32_desc = load_embed_descriptor(embed_fp32, case.system.name)
            comparisons["embed_fp32_vs_eval_desc_new_fp32"] = compare_arrays(
                eval_new_fp32, embed_fp32_desc
            )

        results["cases"].append(
            {
                "name": case.name,
                "source": case.source,
                "model": str(case.model.relative_to(root)),
                "system": str(case.system.relative_to(root)),
                "stress": case.stress,
                "dataset": dataset_summary(case.system),
                "metrics": metrics,
                "comparisons": comparisons,
            }
        )
        (out_root / "results.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )
        write_markdown_summary(out_root / "summary.md", results)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
