---
title: FP11 First Principles Labeling Implementation Plan
status: active
audience: Developers / Operators / AI Agents
last-updated: 2026-07-04
owner: Workflow Owner
---

# FP11 First Principles Labeling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate and run the FP11 ABACUS first-principles labeling workflow, choosing `omp_threads` and `tasks_per_job` from a small GPU benchmark before submitting the full dataset.

**Architecture:** Keep all execution artifacts under `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11` and use the existing `dpeva label` workflow. Use small local helper scripts only for config audit, deterministic one-frame benchmark dataset creation, benchmark summarization, and final config patching; do not change DP-EVA source unless execution exposes a real implementation bug.

**Tech Stack:** DP-EVA CLI, `dpeva-dpa4` conda environment, Pydantic config validation, dpdata/deepmd npy data, ABACUS GPU module on SAI Slurm (`4V100`, `rush-1o2gpu` for benchmark, `flood-1o2gpu` for full run).

---

## Status Update 2026-07-04

Current state: blocked for full FP11 production labeling rerun.

The FP11 production labeling attempt was stopped on 2026-07-03 because the submitted jobs did not follow the current SAI ABACUS launch pattern. The generated DP-EVA batch runner packed multiple task directories into one Slurm job and then launched each task serially with the job-level MPI/GPU allocation. In practice this made ordinary tasks run as two-GPU MPI ABACUS jobs and high-memory tasks run as four-GPU MPI ABACUS jobs, even though most FP11 tasks should be single-card V100 `abacus` runs.

Cancelled work:

- All active `fp_N_*` Slurm jobs owned by `liuzhaoqing` were cancelled.
- Local `dpeva label config_gpu_normal_g2.json --stage execute`, `dpeva label config_gpu_highmem_g4.json --stage execute`, and `dpeva label config_gpu_highmem_extra_g4.json --stage execute` monitor processes were stopped.
- Follow-up verification found no remaining `fp_N_*` queue entries and no matching local execute monitor processes.

Effective completed work:

- Total FP11 frames: 4829.
- Effectively completed and extracted tasks: 580.
- Completed task extraction directory: `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11/completed_tasks`.
- Full completed-task manifest: `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11/fp11_completed_tasks_manifest.json`.
- Human summary: `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11/fp11_completed_tasks_summary.md`.

Convergence confirmation:

- Completed-task extraction used ABACUS SCF convergence markers from `OUT.*/running_scf.log` or `abacus.out`.
- A task is counted as effectively complete only when the logs contain `charge density convergence is achieved` and do not contain `convergence has not been achieved`.
- The extracted completed-task symlink count is 580 and matches the manifest's `unique_completed_tasks`.

Current blockers:

- DP-EVA labeling Slurm dispatch is being refactored; do not restart full FP11 production labeling with the old packed multi-GPU MPI runner.
- The next run must split launcher modes by task class:
  - ordinary single-card tasks: request `--ntasks=1`, `--gpus-per-node=1`, and run `abacus` directly;
  - genuine multi-card ABACUS tasks: source the SAI rank-map script and run `mpirun -np $SLURM_NTASKS --map-by $MAP_OPT abacus`.
- The local command-mode note for this server is `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11/fp11_sai_abacus_submission_notes.md`.
- Remaining FP11 tasks should be resubmitted only after the labeling Slurm backend can preserve the correct single-card versus multi-card launcher semantics.

---

## Current Findings

- Requested path `test/fp11/config_gpu.json` does not exist under the repository root. The matching FP11 directory found in this workspace is `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11`.
- `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11/config_gpu.json` validates as `LabelingConfig` in the `dpeva-dpa4` conda environment.
- The config is structurally close to `examples/recipes/labeling/config_gpu.json`: same PP/ORB maps, ABACUS GPU solver settings, output format, and retry structure. Differences are intentional or review-worthy:
  - `qos` is `flood-1o2gpu` and `walltime` is `12:00:00`, which matches many short single-GPU jobs on SAI.
  - `tasks_per_job` is `20`, not the recipe default `50`; this is plausible for expensive 100-atom ABACUS tasks but must be benchmarked.
  - `integration_enabled` is `true`, but `existing_training_data_path` resolves to `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11/train_data_last`, which currently does not exist.
  - `cleaning_thresholds.cohesive_energy` is `0`, which is accepted by the repo and also used in `examples/recipes/labeling/config_cpu.json`; it means keep frames with cohesive energy per atom `<= 0`.
- PP/ORB files referenced by the config exist under `$HOME/PP_ORB/PP` and `$HOME/PP_ORB/ORB`.
- `sampled_dpdata` contains 10 pools, 561 systems, and 4829 frames. A deterministic 100-atom benchmark candidate is available at `sampled_dpdata/DeepCNT/C62Fe38` with 8 frames.

## File Structure

- Read: `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11/config_gpu.json`
  Validated and then patched after benchmark with chosen `omp_threads` and `tasks_per_job`.
- Read: `examples/recipes/labeling/config_gpu.json`
  Reference GPU labeling recipe for comparison.
- Read: `src/dpeva/config.py`
  `LabelingConfig` schema and path expectations.
- Read: `src/dpeva/workflows/labeling.py`
  Existing prepare/execute/extract/postprocess stage behavior.
- Read: `src/dpeva/labeling/manager.py`
  Existing ABACUS runner generation and `OMP_NUM_THREADS` propagation.
- Create: `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11/fp11_config_audit.py`
  One-shot audit script for config validity, path resolution, recipe comparison, data counts, atom-size distribution, and unresolved integration source.
- Create: `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11/fp11_config_audit.json`
  Machine-readable output from the audit.
- Create: `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11/make_fp11_benchmark.py`
  Deterministically selects one random 95-105 atom frame and writes benchmark configs for `omp_threads` values `1`, `4`, and `8`.
- Create: `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11/benchmark_omp/`
  Contains one-frame benchmark data, generated configs, and benchmark work directories.
- Create: `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11/summarize_fp11_benchmark.py`
  Parses benchmark logs and Slurm accounting output copied into the benchmark directory.
- Create: `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11/fp11_benchmark_summary.json`
  Records elapsed time, convergence status, chosen `omp_threads`, and recommended `tasks_per_job`.
- Create: `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11/patch_fp11_config.py`
  Applies the chosen runtime values to `config_gpu.json` after backing up the original.
- Create: `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11/fp11_run_summary.md`
  Human-readable record of validation, benchmark result, final config values, Slurm job ids, and final labeling output counts.

### Task 1: Audit FP11 Config And Data Contract

**Files:**
- Create: `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11/fp11_config_audit.py`
- Create: `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11/fp11_config_audit.json`
- Read: `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11/config_gpu.json`
- Read: `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/dpeva/examples/recipes/labeling/config_gpu.json`

- [ ] **Step 1: Write the audit script**

Create `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11/fp11_config_audit.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from dpeva.cli import load_and_resolve_config
from dpeva.config import LabelingConfig


FP11 = Path("/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11")
REPO = Path("/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/dpeva")
CONFIG = FP11 / "config_gpu.json"
RECIPE = REPO / "examples/recipes/labeling/config_gpu.json"
OUT = FP11 / "fp11_config_audit.json"


def count_systems(root: Path) -> tuple[int, int, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    for type_raw in sorted(root.rglob("type.raw")):
        system_dir = type_raw.parent
        atom_count = len(type_raw.read_text(encoding="utf-8").split())
        coord = system_dir / "set.000" / "coord.npy"
        frames = int(np.load(coord, mmap_mode="r").shape[0]) if coord.exists() else 0
        rows.append({"path": str(system_dir), "atoms": atom_count, "frames": frames})
    return len(rows), sum(int(row["frames"]) for row in rows), rows


def main() -> None:
    config_raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    recipe_raw = json.loads(RECIPE.read_text(encoding="utf-8"))
    resolved = load_and_resolve_config(str(CONFIG))
    cfg = LabelingConfig(**resolved)

    pp_files = {el: str((Path(cfg.pp_dir) / name).resolve()) for el, name in cfg.pp_map.items()}
    orb_files = {el: str((Path(cfg.orb_dir) / name).resolve()) for el, name in cfg.orb_map.items()}
    missing_pp = {el: path for el, path in pp_files.items() if not Path(path).exists()}
    missing_orb = {el: path for el, path in orb_files.items() if not Path(path).exists()}

    systems, frames, rows = count_systems(Path(cfg.input_data_path))
    near_100 = sorted(rows, key=lambda row: (abs(int(row["atoms"]) - 100), str(row["path"])))[:20]

    comparable_keys = [
        "pp_map",
        "orb_map",
        "mag_map",
        "dft_params",
        "output_format",
        "integration_output_format",
    ]
    recipe_matches = {key: config_raw.get(key) == recipe_raw.get(key) for key in comparable_keys}

    issues: list[str] = []
    if not Path(cfg.input_data_path).exists():
        issues.append(f"input_data_path missing: {cfg.input_data_path}")
    if missing_pp:
        issues.append(f"missing PP files: {missing_pp}")
    if missing_orb:
        issues.append(f"missing ORB files: {missing_orb}")
    if cfg.integration_enabled and cfg.existing_training_data_path and not cfg.existing_training_data_path.exists():
        issues.append(f"integration_enabled=true but existing_training_data_path missing: {cfg.existing_training_data_path}")
    if cfg.submission.backend != "slurm":
        issues.append(f"expected slurm backend, got {cfg.submission.backend}")
    if cfg.submission.slurm_config.get("gpus_per_node") != 1:
        issues.append(f"expected one GPU per benchmark/full single-task job, got {cfg.submission.slurm_config.get('gpus_per_node')}")
    if cfg.submission.slurm_config.get("qos") not in {"flood-1o2gpu", "rush-1o2gpu"}:
        issues.append(f"unexpected SAI one/two-GPU qos: {cfg.submission.slurm_config.get('qos')}")

    report = {
        "config": str(CONFIG),
        "recipe": str(RECIPE),
        "schema_valid": True,
        "resolved_paths": {
            "work_dir": str(cfg.work_dir),
            "input_data_path": str(cfg.input_data_path),
            "pp_dir": str(cfg.pp_dir),
            "orb_dir": str(cfg.orb_dir),
            "existing_training_data_path": str(cfg.existing_training_data_path) if cfg.existing_training_data_path else None,
            "merged_training_data_path": str(cfg.merged_training_data_path) if cfg.merged_training_data_path else None,
        },
        "slurm_config": cfg.submission.slurm_config,
        "omp_threads": cfg.omp_threads,
        "tasks_per_job": cfg.tasks_per_job,
        "recipe_matches": recipe_matches,
        "systems": systems,
        "frames": frames,
        "near_100_atom_candidates": near_100,
        "pp_files": pp_files,
        "orb_files": orb_files,
        "issues": issues,
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if issues:
        raise SystemExit("audit found issues; inspect fp11_config_audit.json")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the audit script**

Run:

```bash
cd /home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11
conda run -n dpeva-dpa4 python fp11_config_audit.py
```

Expected first run:

```text
audit found issues; inspect fp11_config_audit.json
```

Expected known issue in `fp11_config_audit.json`:

```json
"integration_enabled=true but existing_training_data_path missing: /home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11/train_data_last"
```

- [ ] **Step 3: Resolve the integration source**

Use the previous iteration training data already present in this workspace only if it is the intended FP11 base training set:

```bash
cd /home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11
test -d /home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/data/train_data_next
ln -s /home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/data/train_data_next train_data_last
```

If `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/data/train_data_next` is not the intended base set, do not create the symlink. Instead patch `config_gpu.json` before postprocess with:

```json
"integration_enabled": false,
"existing_training_data_path": null,
"merged_training_data_path": null
```

and record that decision in `fp11_run_summary.md`.

- [ ] **Step 4: Re-run the audit**

Run:

```bash
cd /home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11
conda run -n dpeva-dpa4 python fp11_config_audit.py
```

Expected:

```json
"schema_valid": true,
"systems": 561,
"frames": 4829,
"issues": []
```

### Task 2: Create One-Frame OMP Benchmark Configs

**Files:**
- Create: `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11/make_fp11_benchmark.py`
- Create: `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11/benchmark_omp/input_single/`
- Create: `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11/benchmark_omp/config_omp1.json`
- Create: `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11/benchmark_omp/config_omp4.json`
- Create: `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11/benchmark_omp/config_omp8.json`

- [ ] **Step 1: Write the benchmark generator**

Create `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11/make_fp11_benchmark.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import numpy as np


FP11 = Path("/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11")
SOURCE_ROOT = FP11 / "sampled_dpdata"
BENCH_ROOT = FP11 / "benchmark_omp"
INPUT_SINGLE = BENCH_ROOT / "input_single"
SEED = 20260702


def one_frame_array(src: Path, dst: Path, frame: int) -> None:
    arr = np.load(src)
    np.save(dst, arr[frame : frame + 1])


def main() -> None:
    candidates: list[tuple[Path, int, int]] = []
    for type_raw in sorted(SOURCE_ROOT.rglob("type.raw")):
        system_dir = type_raw.parent
        atoms = len(type_raw.read_text(encoding="utf-8").split())
        coord = system_dir / "set.000" / "coord.npy"
        if not coord.exists():
            continue
        frames = int(np.load(coord, mmap_mode="r").shape[0])
        if 95 <= atoms <= 105 and frames > 0:
            candidates.append((system_dir, atoms, frames))
    if not candidates:
        raise SystemExit("no 95-105 atom benchmark candidates found")

    rng = random.Random(SEED)
    system_dir, atoms, frames = rng.choice(candidates)
    frame = rng.randrange(frames)

    if BENCH_ROOT.exists():
        shutil.rmtree(BENCH_ROOT)
    target = INPUT_SINGLE / system_dir.relative_to(SOURCE_ROOT)
    set_target = target / "set.000"
    set_target.mkdir(parents=True)
    shutil.copy2(system_dir / "type.raw", target / "type.raw")
    shutil.copy2(system_dir / "type_map.raw", target / "type_map.raw")
    for name in ["coord.npy", "box.npy", "energy.npy", "force.npy", "virial.npy"]:
        src = system_dir / "set.000" / name
        if src.exists():
            one_frame_array(src, set_target / name, frame)

    base = json.loads((FP11 / "config_gpu.json").read_text(encoding="utf-8"))
    base["input_data_path"] = str(target)
    base["tasks_per_job"] = 1
    base["integration_enabled"] = False
    base["existing_training_data_path"] = None
    base["merged_training_data_path"] = None
    base["submission"]["slurm_config"]["qos"] = "rush-1o2gpu"
    base["submission"]["slurm_config"]["walltime"] = "02:00:00"
    base["submission"]["slurm_config"]["ntasks"] = 1
    base["submission"]["slurm_config"]["gpus_per_node"] = 1
    env_setup = base["submission"].get("env_setup", [])
    if isinstance(env_setup, str):
        env_setup = [line for line in env_setup.splitlines() if line.strip()]
    monitor = "nvidia-smi dmon -s pucvmte -o T > nvdmon_job-${SLURM_JOB_ID}.log &"
    if monitor not in env_setup:
        env_setup.insert(0, monitor)
    base["submission"]["env_setup"] = env_setup

    for omp in [1, 4, 8]:
        cfg = dict(base)
        cfg["omp_threads"] = omp
        cfg["work_dir"] = str(BENCH_ROOT / f"work_omp{omp}")
        (BENCH_ROOT / f"config_omp{omp}.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "seed": SEED,
        "source_system": str(system_dir),
        "atom_count": atoms,
        "source_frames": frames,
        "selected_frame": frame,
        "benchmark_input": str(target),
        "configs": [str(BENCH_ROOT / f"config_omp{omp}.json") for omp in [1, 4, 8]],
    }
    (BENCH_ROOT / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Generate benchmark input and configs**

Run:

```bash
cd /home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11
conda run -n dpeva-dpa4 python make_fp11_benchmark.py
```

Expected:

```json
"atom_count": 95,
"atom_count": 96,
"atom_count": 97,
"atom_count": 98,
"atom_count": 99,
"atom_count": 100,
"atom_count": 101,
"atom_count": 102,
"atom_count": 103,
"atom_count": 104,
"atom_count": 105,
"configs": [
  "/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11/benchmark_omp/config_omp1.json",
  "/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11/benchmark_omp/config_omp4.json",
  "/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11/benchmark_omp/config_omp8.json"
]
```

- [ ] **Step 3: Validate benchmark configs**

Run:

```bash
cd /home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11
conda run -n dpeva-dpa4 python -c "from dpeva.cli import load_and_resolve_config; from dpeva.config import LabelingConfig; [print(LabelingConfig(**load_and_resolve_config(f'benchmark_omp/config_omp{omp}.json')).omp_threads) for omp in (1,4,8)]"
```

Expected:

```text
1
4
8
```

### Task 3: Run OMP Benchmark And Choose Runtime Values

**Files:**
- Create: `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11/summarize_fp11_benchmark.py`
- Create: `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11/fp11_benchmark_summary.json`
- Modify later: `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11/config_gpu.json`

- [ ] **Step 1: Prepare benchmark tasks**

Run:

```bash
cd /home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11
for omp in 1 4 8; do
  conda run -n dpeva-dpa4 dpeva label benchmark_omp/config_omp${omp}.json --stage prepare
done
```

Expected for each OMP value:

```text
Generated 1 tasks.
```

- [ ] **Step 2: Execute benchmark tasks sequentially**

Run:

```bash
cd /home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11
for omp in 1 4 8; do
  echo "=== OMP ${omp} ==="
  date "+%F %T"
  conda run -n dpeva-dpa4 dpeva label benchmark_omp/config_omp${omp}.json --stage execute
  date "+%F %T"
done
```

Expected:

```text
All jobs finished.
All active tasks converged.
```

If one OMP value fails SCF but another converges, keep the converged values only. If all three fail, inspect `benchmark_omp/work_omp*/inputs/N_1_0/*/abacus.out` and stop before modifying the full config.

- [ ] **Step 3: Extract benchmark results**

Run:

```bash
cd /home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11
for omp in 1 4 8; do
  conda run -n dpeva-dpa4 dpeva label benchmark_omp/config_omp${omp}.json --stage extract
done
```

Expected for each OMP value:

```text
Extraction summary: converged=1, bad_converged=0, failed=0
```

- [ ] **Step 4: Write benchmark summarizer**

Create `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11/summarize_fp11_benchmark.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


FP11 = Path("/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11")
BENCH = FP11 / "benchmark_omp"
OUT = FP11 / "fp11_benchmark_summary.json"


def parse_job_id(log_path: Path) -> str | None:
    if not log_path.exists():
        return None
    text = log_path.read_text(errors="ignore")
    matches = re.findall(r"Submitted batch job\s+(\d+)", text)
    return matches[-1] if matches else None


def sacct_elapsed_seconds(job_id: str) -> int | None:
    result = subprocess.run(
        ["sacct", "-j", job_id, "--format=JobID,State,ElapsedRaw", "-P", "-n"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        return None
    seconds: list[int] = []
    for line in result.stdout.splitlines():
        parts = line.split("|")
        if len(parts) < 3:
            continue
        jid, state, elapsed = parts[:3]
        if "." in jid:
            continue
        if not elapsed.isdigit():
            continue
        if state.startswith(("COMPLETED", "FAILED", "TIMEOUT", "CANCELLED")):
            seconds.append(int(elapsed))
    return max(seconds) if seconds else None


def converged(work_dir: Path) -> bool:
    return any((work_dir / "CONVERGED").rglob("running_scf.log")) or any((work_dir / "CONVERGED").rglob("abacus.out"))


def main() -> None:
    rows = []
    for omp in [1, 4, 8]:
        work = BENCH / f"work_omp{omp}"
        log = work / "labeling_execute.log"
        job_id = parse_job_id(log)
        elapsed = sacct_elapsed_seconds(job_id) if job_id else None
        rows.append(
            {
                "omp_threads": omp,
                "work_dir": str(work),
                "job_id": job_id,
                "elapsed_seconds": elapsed,
                "converged": converged(work),
            }
        )

    valid = [row for row in rows if row["converged"] and isinstance(row["elapsed_seconds"], int) and row["elapsed_seconds"] > 0]
    if not valid:
        raise SystemExit("no completed converged benchmark rows with Slurm elapsed time")
    valid.sort(key=lambda row: (int(row["elapsed_seconds"]), int(row["omp_threads"])))
    best = valid[0]
    best_seconds = int(best["elapsed_seconds"])

    if best_seconds <= 180:
        tasks_per_job = 30
    elif best_seconds <= 360:
        tasks_per_job = 20
    elif best_seconds <= 720:
        tasks_per_job = 10
    else:
        tasks_per_job = 5

    summary = {
        "rows": rows,
        "chosen_omp_threads": int(best["omp_threads"]),
        "chosen_elapsed_seconds": best_seconds,
        "recommended_tasks_per_job": tasks_per_job,
        "rule": "Use the fastest converged OMP setting. Target serial bundle length is roughly 1-3 hours under flood-1o2gpu; cap at 30 to keep retry granularity reasonable.",
    }
    OUT.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Summarize and choose values**

Run:

```bash
cd /home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11
conda run -n dpeva-dpa4 python summarize_fp11_benchmark.py
```

Expected:

```json
"chosen_omp_threads": 1 or 4 or 8,
"recommended_tasks_per_job": 5 or 10 or 20 or 30
```

### Task 4: Patch Final Config And Validate Prepare Stage

**Files:**
- Create: `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11/patch_fp11_config.py`
- Modify: `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11/config_gpu.json`
- Create: `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11/config_gpu.json.pre_benchmark`
- Create: `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11/labeling_workdir/inputs/`

- [ ] **Step 1: Write final config patcher**

Create `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11/patch_fp11_config.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from pathlib import Path


FP11 = Path("/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11")
CONFIG = FP11 / "config_gpu.json"
BACKUP = FP11 / "config_gpu.json.pre_benchmark"
SUMMARY = FP11 / "fp11_benchmark_summary.json"


def main() -> None:
    if not BACKUP.exists():
        shutil.copy2(CONFIG, BACKUP)
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    cfg["omp_threads"] = int(summary["chosen_omp_threads"])
    cfg["tasks_per_job"] = int(summary["recommended_tasks_per_job"])
    cfg["submission"]["slurm_config"]["qos"] = "flood-1o2gpu"
    cfg["submission"]["slurm_config"]["walltime"] = "12:00:00"
    cfg["submission"]["slurm_config"]["ntasks"] = 1
    cfg["submission"]["slurm_config"]["gpus_per_node"] = 1
    CONFIG.write_text(json.dumps(cfg, indent=4) + "\n", encoding="utf-8")
    print(f"patched {CONFIG}")
    print(f"omp_threads={cfg['omp_threads']}")
    print(f"tasks_per_job={cfg['tasks_per_job']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Patch and validate config**

Run:

```bash
cd /home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11
conda run -n dpeva-dpa4 python patch_fp11_config.py
conda run -n dpeva-dpa4 python fp11_config_audit.py
```

Expected:

```text
patched /home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11/config_gpu.json
omp_threads is the integer stored in fp11_benchmark_summary.json at chosen_omp_threads
tasks_per_job is the integer stored in fp11_benchmark_summary.json at recommended_tasks_per_job
```

Expected audit:

```json
"issues": []
```

- [ ] **Step 3: Run full prepare stage**

Run:

```bash
cd /home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11
conda run -n dpeva-dpa4 dpeva label config_gpu.json --stage prepare
```

Expected:

```text
Generated 4829 tasks.
```

- [ ] **Step 4: Verify packing count**

Run:

```bash
cd /home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11
find labeling_workdir/inputs -maxdepth 1 -type d -name 'N_*' | wc -l
```

Expected:

```text
ceil(4829 / tasks_per_job)
```

For example, if `tasks_per_job=20`, expected packed jobs are `242`.

### Task 5: Submit Full FP11 Calculation

**Files:**
- Read/Write: `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11/labeling_workdir/`
- Read: `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11/config_gpu.json`

- [ ] **Step 1: Submit and monitor execute stage**

Run:

```bash
cd /home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11
conda run -n dpeva-dpa4 dpeva label config_gpu.json --stage execute
```

Expected during submission:

```text
Submitting one Slurm job bundle per N_* directory under labeling_workdir/inputs.
Monitoring the same number of Slurm jobs.
```

Expected completion:

```text
All jobs finished.
Checking convergence...
```

The execute stage may perform retry attempts using `attempt_params`. Do not interrupt it unless Slurm reports a systemic submission error such as invalid partition/QOS.

- [ ] **Step 2: Inspect Slurm state if the execute stage appears stuck**

Run in a second terminal:

```bash
squeue -u "$USER"
```

Expected:

```text
FP11 jobs appear with names matching fp_N_[0-9]+_[0-9]+_att[0-9]+.
```

If jobs are pending with QOS or partition errors, update only `submission.slurm_config.partition`, `qos`, or `account` according to SAI policy, then rerun `dpeva label config_gpu.json --stage execute`.

- [ ] **Step 3: Extract results after execute exits**

Run:

```bash
cd /home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11
conda run -n dpeva-dpa4 dpeva label config_gpu.json --stage extract
```

Expected:

```text
Extraction summary includes integer converged, bad_converged, and failed counts.
```

If `failed` remains nonzero after all configured attempts, inspect representative failed directories under `labeling_workdir/inputs/N_*/*/abacus.out` and decide whether to add one more `attempt_params` entry or accept the failures as anomalies.

- [ ] **Step 4: Run postprocess and optional integration**

Run:

```bash
cd /home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11
conda run -n dpeva-dpa4 dpeva label config_gpu.json --stage postprocess
```

Expected when integration remains enabled:

```text
Data integration finished:
```

Expected outputs:

```text
labeling_workdir/outputs/cleaned
labeling_workdir/outputs/anomalies
train_data_next/integration_summary.json
```

### Task 6: Verify Outputs And Record Run Summary

**Files:**
- Create: `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11/write_fp11_run_summary.py`
- Create: `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11/fp11_run_summary.md`
- Read: `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11/fp11_config_audit.json`
- Read: `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11/fp11_benchmark_summary.json`
- Read: `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11/labeling_workdir/outputs/`

- [ ] **Step 1: Count final datasets**

Run:

```bash
cd /home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11
find labeling_workdir/CONVERGED -type d -name 'task*' | wc -l
find labeling_workdir/BAD_CONVERGED -type d -name 'task*' | wc -l
find labeling_workdir/outputs/cleaned -name type.raw | wc -l
find labeling_workdir/outputs/anomalies -name type.raw | wc -l
test -f train_data_next/integration_summary.json && cat train_data_next/integration_summary.json
```

Expected:

```text
CONVERGED count is nonzero.
cleaned type.raw count is nonzero unless every converged frame is filtered by configured thresholds.
integration_summary.json exists when integration_enabled=true.
```

- [ ] **Step 2: Write final run summary generator**

Create `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11/write_fp11_run_summary.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


FP11 = Path("/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11")
WORK = FP11 / "labeling_workdir"
OUT = FP11 / "fp11_run_summary.md"


def count_dirs(root: Path, pattern: str) -> int:
    return sum(1 for _ in root.rglob(pattern)) if root.exists() else 0


def count_files(root: Path, name: str) -> int:
    return sum(1 for _ in root.rglob(name)) if root.exists() else 0


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main() -> None:
    audit = load_json(FP11 / "fp11_config_audit.json")
    bench = load_json(FP11 / "fp11_benchmark_summary.json")
    integration = load_json(FP11 / "train_data_next" / "integration_summary.json")
    config = load_json(FP11 / "config_gpu.json")

    lines = [
        "# FP11 Labeling Run Summary",
        "",
        "## Config Validation",
        "",
        f"- Config: `{FP11 / 'config_gpu.json'}`",
        f"- Reference recipe: `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/dpeva/examples/recipes/labeling/config_gpu.json`",
        f"- Audit report: `{FP11 / 'fp11_config_audit.json'}`",
        f"- Dataset systems: `{audit.get('systems')}`",
        f"- Dataset frames: `{audit.get('frames')}`",
        f"- Integration enabled: `{config.get('integration_enabled')}`",
        f"- Integration source: `{config.get('existing_training_data_path')}`",
        "",
        "## Benchmark",
        "",
        f"- Benchmark report: `{FP11 / 'fp11_benchmark_summary.json'}`",
        f"- Chosen `omp_threads`: `{bench.get('chosen_omp_threads')}`",
        f"- Chosen benchmark elapsed seconds: `{bench.get('chosen_elapsed_seconds')}`",
        f"- Chosen `tasks_per_job`: `{bench.get('recommended_tasks_per_job')}`",
        "",
        "## Execution",
        "",
        "- Prepare command: `conda run -n dpeva-dpa4 dpeva label config_gpu.json --stage prepare`",
        "- Execute command: `conda run -n dpeva-dpa4 dpeva label config_gpu.json --stage execute`",
        "- Extract command: `conda run -n dpeva-dpa4 dpeva label config_gpu.json --stage extract`",
        "- Postprocess command: `conda run -n dpeva-dpa4 dpeva label config_gpu.json --stage postprocess`",
        "",
        "## Results",
        "",
        f"- Converged task count: `{count_dirs(WORK / 'CONVERGED', 'task*')}`",
        f"- Bad-converged task count: `{count_dirs(WORK / 'BAD_CONVERGED', 'task*')}`",
        f"- Cleaned system count: `{count_files(WORK / 'outputs' / 'cleaned', 'type.raw')}`",
        f"- Anomaly system count: `{count_files(WORK / 'outputs' / 'anomalies', 'type.raw')}`",
        f"- Integration summary path: `{FP11 / 'train_data_next' / 'integration_summary.json'}`",
        f"- Integration existing frame count: `{integration.get('existing_frame_count')}`",
        f"- Integration new frame count: `{integration.get('new_frame_count')}`",
        f"- Integration merged frame count after dedup: `{integration.get('merged_frame_count_after_dedup')}`",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Generate final run summary**

Run:

```bash
cd /home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11
conda run -n dpeva-dpa4 python write_fp11_run_summary.py
sed -n '1,120p' fp11_run_summary.md
```

Expected:

```text
/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11/fp11_run_summary.md
# FP11 Labeling Run Summary
```

- [ ] **Step 4: Final verification**

Run:

```bash
cd /home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11
test -s fp11_config_audit.json
test -s fp11_benchmark_summary.json
test -s fp11_run_summary.md
test -d labeling_workdir/outputs/cleaned
```

Expected: all commands exit with status `0`.

## Self-Review

- Spec coverage:
  - Requirement 1 is covered by Task 1: config existence, schema validation, recipe comparison, PP/ORB checks, data counts, and explicit issue handling.
  - Requirement 2 is covered by Tasks 2-3: deterministic random one-frame benchmark around 100 atoms for `omp_threads=1,4,8`, then measured selection.
  - Requirement 3 is covered by Tasks 4-6: patch final config, choose `tasks_per_job`, run existing `dpeva label` prepare/execute/extract/postprocess in `dpeva-dpa4`, and verify outputs.
- Placeholder scan:
  - No implementation step contains unresolved `TBD` or missing code. The only human decision is the integration source contract, and the plan provides exact commands for both valid branches.
- Type consistency:
  - All scripts use `LabelingConfig`, `load_and_resolve_config`, `config_gpu.json`, `fp11_benchmark_summary.json`, `omp_threads`, and `tasks_per_job` consistently.
