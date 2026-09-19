---
title: DPA4 Multi-Head Forgetting Study Implementation Plan
status: record
audience: Developers / AI Agents
last-updated: 2026-09-03
owner: Scientific Owner
---

# DPA4 Multi-Head Forgetting Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Quantify forgetting in the existing single-head DPA4-Air MatPES fine-tune, then compare five reproducible two-head fine-tuning schedules that jointly train the MatPES replay head and the Fe-C-H-O `train_data_iter11` head.

**Architecture:** Freeze the exact pretrained checkpoint, single-head checkpoints, data paths, and evaluation command before training. Use DeepMD-kit PyTorch multi-task input with a `Default` MatPES branch that resumes the pretrained single-task model and an `ft2dp` branch with `"finetune_head": "Default"`; both branches share the DPA4 descriptor. Treat `training.numb_steps` as global optimizer updates: four schedules use `ceil(50000 / p_ft2dp)` to target 50,000 expected `ft2dp` updates, while the `replay_lr350e-3` schedule is an explicit low-target-update exception with 100,000 global steps and 20,000 expected `ft2dp` updates. Then evaluate every checkpoint head on complete MatPES and complete `train_data_iter11` data with `dp test -n 0`. The report will distinguish true full-dataset errors from sampled `lcurve` validation values.

**Tech Stack:** DeepMD-kit upstream/develop-compatible PyTorch backend, CUDA 12.6.3, PyTorch 2.11.0+cu126, DPA4/SeZM, `dp --pt train`, `dp --pt test`, Python JSON/CSV parsing, SAI-new Slurm on `16V100`.

---

## Scope and Fixed Inputs

All remote paths below are under `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva`.

- Pretrained MatPES baseline: `models/426/DPA4-MatPES-ZBL-v20260629/checkpoints/DPA4-Air-ZBL-MatPES-v20260629.pt`
- Single-head regular model: `v2.2-ft/runs/dpa4_air_matpes_zbl_train_data_iter11_50k/resume_from_30000_nofullval_numworkers0/models/model.ckpt-50000.pt`
- Single-head EMA model: `v2.2-ft/runs/dpa4_air_matpes_zbl_train_data_iter11_50k/resume_from_30000_nofullval_numworkers0/models/model_ema.ckpt-50000.pt`
- Fe-C-H-O data: `v2.2-ft/data/train_data_iter11`
- MatPES replay data: `multitask-ft-test-20260711/data/matpes-r2scan-2025.1`
- Environment: `/home/pku-jianghong/liuzhaoqing/fp11-sai1344/dpeva/scripts/env/dpeva-dpa4.env`
- Actual runtime environment: `dpeva-dpa4`, CUDA module `cuda/12.6.3`, DeepMD source commit `f73de32e`, PyTorch `2.11.0+cu126`.

The existing single-head input uses `training.training_data.batch_size = "filter:255"` and 50,000 target-head updates. Multi-head DeepMD does not count a selected head's updates separately: one global step samples one branch according to `model_prob`, performs that branch's update, and increments the global step. Therefore the full-run `numb_steps` values are schedule-specific and are not all 50,000. Multi-task DeepMD does not support `training.zero_stage > 0`, so the multi-head inputs will set `zero_stage = 0` as the one required upstream compatibility change. `validating.full_validation`, `ema_full_validation`, and `compiled_infer` remain disabled during training to avoid the previously observed V100 validation OOM; full metrics come from separate `dp test -n 0` jobs.

## Experiment Matrix

`model_prob` is the DeepMD task sampling probability, where `Default` is MatPES and `ft2dp` is Fe-C-H-O. The loss, optimizer, descriptor, EMA, and batch size are controlled across all five schemes. Four schemes target 50,000 expected `ft2dp` updates; the replay-priority exception intentionally tests only 20,000 expected target updates at the same 100,000 global steps as the balanced schedules.

| ID | MatPES `Default` probability | Fe-C-H-O `ft2dp` probability | Expected `ft2dp` updates | Global `training.numb_steps` | `learning_rate.start_lr` | Purpose |
|---|---:|---:|---:|---:|---:|---|
| `balanced_lr350e-3` | 0.50 | 0.50 | 50,000 | 100,000 | `3.5e-4` | Balanced replay/target baseline at the single-head LR |
| `target_lr350e-3` | 0.20 | 0.80 | 50,000 | 62,500 | `3.5e-4` | Prioritize target accuracy while retaining replay |
| `replay_lr350e-3` | 0.80 | 0.20 | 20,000 | 100,000 | `3.5e-4` | Strong retention pressure with reduced target-update budget |
| `target_lr100e-3` | 0.20 | 0.80 | 50,000 | 62,500 | `1.0e-4` | Target-heavy, lower-update-risk schedule |
| `balanced_lr100e-3` | 0.50 | 0.50 | 50,000 | 100,000 | `1.0e-4` | Balanced schedule with reduced shared-descriptor drift |

The remaining learning-rate fields stay as in the single-head input: `stop_lr = 1e-6`, `warmup_ratio = 0.003`, `warmup_start_factor = 0.2`, `decay_phase_ratio = 0.65`, and `decay_type = "cosine"`. The LR schedule is indexed by global step, so its endpoint is reached after the schedule-specific global step count; this is recorded as an intentional consequence of equalizing target-head updates. The `model_prob` values are the only dataset-mixture control; no loss-prefactor or optimizer changes are introduced in this campaign.

The values in the table are expected target-head update counts, not exact per-head counters: upstream DeepMD-kit samples the active head stochastically from `model_prob` and exposes only the global `numb_steps` limit. The standard configuration cannot guarantee an exact count. The campaign therefore records the seed, global step count, and any available sampler/update count; if the runtime does not expose a per-step head count, the report must label the counts as mathematical expectations. The 20,000-update replay exception is intentionally compared against the four 50,000-update schedules to test whether target accuracy can be retained with fewer target-head updates. Enforcing an exact quota would require a custom upstream sampler and is outside this experiment plan.

## Files and Responsibilities

**Create locally and mirror remotely:**

- `v2.2-ft/runs/dpa4_multitask_forgetting_20260711/README.md`: frozen inputs, experiment matrix, job IDs, checkpoint mapping, and interpretation rules.
- `v2.2-ft/runs/dpa4_multitask_forgetting_20260711/generate_inputs.py`: generate five JSON files from the official Air architecture, with exact `Default`/`ft2dp` branch semantics and matrix values.
- `v2.2-ft/runs/dpa4_multitask_forgetting_20260711/run_smoke_4v100.sbatch`: run 100-step smoke tests for all five inputs, one schedule per job or array element.
- `v2.2-ft/runs/dpa4_multitask_forgetting_20260711/run_full_4v100.sbatch`: run one selected matrix schedule on four V100 GPUs; use the same script for all five matrix rows.
- `v2.2-ft/runs/dpa4_multitask_forgetting_20260711/run_baseline_tests_4gpu.sbatch`: evaluate the pretrained, single-head regular, and single-head EMA models on both complete datasets.
- `v2.2-ft/runs/dpa4_multitask_forgetting_20260711/run_multitask_tests_4gpu.sbatch`: evaluate every multi-head regular and EMA checkpoint on the matching head/data pair.
- `v2.2-ft/runs/dpa4_multitask_forgetting_20260711/parse_results.py`: parse `dp test` logs and `lcurve.out` into one CSV and one Markdown summary.
- `v2.2-ft/runs/dpa4_multitask_forgetting_20260711/summary/`: generated `baseline_metrics.csv`, `multitask_metrics.csv`, `lcurve_metrics.csv`, `forgetting_report.md`, and `forgetting_report.json`.

No files under `src/dpeva` are required. This is an experiment campaign using DeepMD-kit directly; DP-EVA is not started during training or testing.

### Task 1: Freeze Inputs and Establish the Single-Head Forgetting Baseline

**Files:**
- Create: `v2.2-ft/runs/dpa4_multitask_forgetting_20260711/README.md`
- Create: `v2.2-ft/runs/dpa4_multitask_forgetting_20260711/run_baseline_tests_4gpu.sbatch`
- Create: `v2.2-ft/runs/dpa4_multitask_forgetting_20260711/parse_results.py`
- Generate: `summary/baseline_metrics.csv`

- [ ] **Step 1: Record immutable input checksums and metadata.**

Run locally and remotely:

```bash
sha256sum \
  model/dpa4/matpes/DPA4-MatPES-ZBL-v20260629/checkpoints/DPA4-Air-ZBL-MatPES-v20260629.pt \
  model/ft2dp-dpa4/ft2dp-dpa4-air-regular-50k.pt \
  model/ft2dp-dpa4/ft2dp-dpa4-air-ema-50k.pt \
  test/matpes-r2scan-2025-1-dp-npy-mixed.tar.gz
```

Record the DeepMD source commit, environment name, CUDA module, model sizes, MatPES system count, and `train_data_iter11` system count in `README.md`. Abort the campaign if any model or dataset checksum differs between local and remote copies.

- [ ] **Step 2: Run baseline tests on complete datasets.**

The Slurm script must request `--partition=16V100`, `--nodes=1`, `--ntasks=1`, `--gpus-per-node=4`, and `--qos=huge-gpu`; it must not request `--cpus-per-task` or `--mem`. Each child command uses one GPU and the environment's Python interpreter explicitly:

```bash
CUDA_VISIBLE_DEVICES="$gpu" "$PY_BIN" "$DP_BIN" --pt test \
  -m "$model" -s "$system" -n 0 -d "$output" [--head "$head"]
```

Run these six baseline evaluations:

```text
pretrained_matpes on matpes-r2scan-2025.1, no --head
pretrained_matpes on train_data_iter11, no --head
single_regular on matpes-r2scan-2025.1, no --head
single_regular on train_data_iter11, no --head
single_ema on matpes-r2scan-2025.1, no --head
single_ema on train_data_iter11, no --head
```

Run at most four children concurrently, then start the remaining two. Write one `.rc` file per child and make the parent job fail if any child return code is nonzero. Do not rely on `grep -L` logic that previously misreported successful child tests.

- [ ] **Step 3: Define the forgetting calculation.**

For each single-head checkpoint, calculate MatPES forgetting relative to the untouched pretrained model:

```text
forgetting_ratio(metric) =
    (single_head_metric - pretrained_metric) / pretrained_metric
```

Report this for energy MAE, energy RMSE, force MAE, force RMSE, virial MAE, and stress MAE. Also report the absolute target-dataset metrics. A positive MatPES ratio means degradation; do not infer forgetting from lcurve values.

- [ ] **Step 4: Verify baseline completeness.**

Expected: Slurm `COMPLETED|0:0`, six child `.rc` files equal to `0`, `systems`/frame counts recorded for every result, and `summary/baseline_metrics.csv` containing the six rows. The baseline report must explicitly state whether single-head regular and EMA models forgot MatPES relative to the original Air model.

### Task 2: Generate and Validate the Five Multi-Head Configurations

**Files:**
- Create: `v2.2-ft/runs/dpa4_multitask_forgetting_20260711/generate_inputs.py`
- Generate: `inputs/balanced_lr350e-3/input.json`, `inputs/target_lr350e-3/input.json`, `inputs/replay_lr350e-3/input.json`, `inputs/target_lr100e-3/input.json`, and `inputs/balanced_lr100e-3/input.json`
- Generate the corresponding `input_v2_compat.json` beside each of those five inputs.

- [ ] **Step 1: Generate the exact multi-head model structure.**

The generator must load `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/multitask-ft-test-20260711/DPA4-Air-ZBL-MatPES-v20260629.json` and construct the branches with this executable logic:

```python
source = json.load(open("/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/multitask-ft-test-20260711/DPA4-Air-ZBL-MatPES-v20260629.json"))["model"]
shared = {
    "matpes_type_map": source["type_map"],
    "air_descriptor": {**source["descriptor"], "type": "dpa4"},
}
def branch():
    return {
        "type": "SeZM",
        "type_map": "matpes_type_map",
        "descriptor": "air_descriptor",
        "fitting_net": source["fitting_net"],
        "bridging_method": "ZBL",
        "bridging_r_inner": 0.5,
        "bridging_r_outer": 0.8,
    }
model_dict = {"Default": branch(), "ft2dp": branch()}
model_dict["ft2dp"]["finetune_head"] = "Default"
model = {"shared_dict": shared, "model_dict": model_dict}
```

The `Default` name is mandatory for direct fine-tuning from a single-task checkpoint: upstream maps a single-task checkpoint to its implicit `Default` branch. The target branch uses the complete pretrained type map so both branches can share the descriptor; DeepMD maps each system's local `type_map.raw` by element name. Verify that Fe, C, H, and O are present in the shared map.

- [ ] **Step 2: Preserve fixed training parameters and apply only matrix controls.**

For each scheme, generate these exact dataset bindings and fixed training values:

```python
data_dict = {
    "Default": {
        "stat_file": f"{run_dir}/matpes.hdf5",
        "training_data": {
            "systems": "/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/multitask-ft-test-20260711/data/matpes-r2scan-2025.1",
            "batch_size": "filter:255",
        },
        "validation_data": {
            "systems": "/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/multitask-ft-test-20260711/data/matpes-r2scan-2025.1",
            "batch_size": 1,
            "numb_batch": 1,
        },
    },
    "ft2dp": {
        "stat_file": f"{run_dir}/ft2dp.hdf5",
        "training_data": {
            "systems": "/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/v2.2-ft/data/train_data_iter11",
            "batch_size": "filter:255",
        },
        "validation_data": {
            "systems": "/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/v2.2-ft/data/train_data_iter11",
            "batch_size": 1,
            "numb_batch": 1,
        },
    },
}
training = {
    "model_prob": {"Default": 0.50, "ft2dp": 0.50},
    "data_dict": data_dict,
    "numb_steps": global_steps_for_scheme,
    "zero_stage": 0,
    "enable_ema": True,
    "ema_decay": 0.999,
}
```

Use the official Air optimizer, loss, descriptor, `use_compile=true`, AMP, and `enable_tf32=true`. Set `model_prob`, `learning_rate.start_lr`, and `training.numb_steps` from the experiment matrix. The generator must use this exact matrix metadata:

```python
matrix = {
    "balanced_lr350e-3": {"model_prob": {"Default": 0.5, "ft2dp": 0.5}, "global_steps": 100000, "expected_ft2dp_updates": 50000},
    "target_lr350e-3": {"model_prob": {"Default": 0.2, "ft2dp": 0.8}, "global_steps": 62500, "expected_ft2dp_updates": 50000},
    "replay_lr350e-3": {"model_prob": {"Default": 0.8, "ft2dp": 0.2}, "global_steps": 100000, "expected_ft2dp_updates": 20000},
    "target_lr100e-3": {"model_prob": {"Default": 0.2, "ft2dp": 0.8}, "global_steps": 62500, "expected_ft2dp_updates": 50000},
    "balanced_lr100e-3": {"model_prob": {"Default": 0.5, "ft2dp": 0.5}, "global_steps": 100000, "expected_ft2dp_updates": 50000},
}
for scheme, values in matrix.items():
    assert values["global_steps"] * values["model_prob"]["ft2dp"] == values["expected_ft2dp_updates"]
```

Record the matrix metadata in `README.md`. Keep full validation disabled during training and set `num_workers=0` in the shell/runtime configuration to avoid the prior shared-memory manager timeout.

- [ ] **Step 3: Run JSON and DeepMD schema validation before GPU jobs.**

Run:

```bash
python -m json.tool inputs/balanced_lr350e-3/input.json >/dev/null
for f in inputs/*/input.json; do
  diff -u inputs/balanced_lr350e-3/input.json "$f" \
    --ignore-matching-lines='model_prob|start_lr|save_dir|disp_file|stat_file' || true
done
```

For each of the five inputs, run the 100-step smoke command from Task 3 with `save_dir` set to its exact matrix-ID directory under `schema_check/` and inspect the generated `input_v2_compat.json`. The acceptance condition is that normalization reaches model construction without `ArgumentKeyError`, `model branch unknown_head does not exist`, `zero_stage` multi-task errors, or type-map errors.

### Task 3: Run Five Multi-Head Smoke Tests Before Full Training

**Files:**
- Create: `v2.2-ft/runs/dpa4_multitask_forgetting_20260711/run_smoke_4v100.sbatch`
- Generate the smoke artifacts under `smoke/balanced_lr350e-3/`, `smoke/target_lr350e-3/`, `smoke/replay_lr350e-3/`, `smoke/target_lr100e-3/`, and `smoke/balanced_lr100e-3/`, including `train.log`, `lcurve.out`, and `models/model.ckpt-100.pt`.

- [ ] **Step 1: Submit one 100-step smoke job per scheme.**

Use the SAI `16V100` partition with four GPUs and the same DDP launch pattern as the completed single-head jobs. To bypass the broken `dp` shebang on the login filesystem, invoke the installed Python interpreter explicitly:

```bash
PY_BIN=$(which python)
DP_BIN=$(which dp)
"$PY_BIN" -m torch.distributed.run --nproc_per_node=4 --standalone --no-python \
  "$PY_BIN" -m deepmd --pt train input.json \
  --finetune "$BASE_MODEL" --skip-neighbor-stat
```

For smoke only, override `training.numb_steps=100`, `save_freq=100`, and `disp_freq=100`; this is a schema/startup test and is not used to estimate the 50,000 target-head updates. Do not change `batch_size`, `model_prob`, or learning rate. Set `save_dir` and `disp_file` inside the scheme's smoke directory.

- [ ] **Step 2: Check each smoke log.**

Require all five jobs to show:

```text
Model branch Default will resume training.
Model branch ft2dp will be fine-tuned.
Shared params of Default.descriptor and ft2dp.descriptor!
Saved model to `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/v2.2-ft/runs/dpa4_multitask_forgetting_20260711/smoke/balanced_lr350e-3/models/model.ckpt-100.pt` (with the corresponding exact matrix-ID path for each other scheme)
```

Reject a scheme if the log contains `Traceback`, `CUDA out of memory`, `device-side assert`, `ProcessGroupNCCL`, `Shared memory manager connection has timed out`, or nonzero Slurm exit code.

### Task 4: Run the Five Multi-Head Fine-Tunes with Controlled Target-Head Budgets

**Files:**
- Create: `v2.2-ft/runs/dpa4_multitask_forgetting_20260711/run_full_4v100.sbatch`
- Generate regular and EMA checkpoints under `runs/balanced_lr350e-3/`, `runs/target_lr350e-3/`, `runs/replay_lr350e-3/`, `runs/target_lr100e-3/`, and `runs/balanced_lr100e-3/`.

- [ ] **Step 1: Submit the full runs only after smoke acceptance.**

Submit five independent jobs, one per scheme, with:

```text
partition=16V100
nodes=1
ntasks=1
gpus-per-node=4
qos=huge-gpu
```

Do not add `--cpus-per-task` or `--mem`. Exclude the previously observed bad nodes `16v100n01,16v100n13,16v100n20,16v100n22,16v100n23` when the scheduler supports `--exclude`; record the actual node in `README.md`.

- [ ] **Step 2: Verify training progress and fixed workload.**

For every job, record the first and last `lcurve.out` rows, average seconds per batch, GPU memory, job elapsed time, and final checkpoint list. Confirm each log reaches the scheme-specific global step count and contains both `Default_*` and `ft2dp_*` metrics. If the runtime exposes a per-step model-selection/update count, record the realized `ft2dp` count and its deviation from the scheme-specific expectation of 50,000, 50,000, 20,000, 50,000, or 50,000; otherwise record the mathematical expectation and explicitly mark the realized count as unavailable. A job is complete only when both regular and EMA checkpoints at the scheme-specific final step exist and Slurm reports `COMPLETED|0:0`.

- [ ] **Step 3: Preserve all artifacts before testing.**

Copy each input, `input_v2_compat.json`, `train.log`, `lcurve.out`, Slurm output/error, `nvdmon` log, and checkpoint SHA256 into the scheme directory. Do not overwrite a failed run; create `retry_<date>` and retain the original failure log.

### Task 5: Evaluate Every Model on Both Complete Datasets

**Files:**
- Create: `v2.2-ft/runs/dpa4_multitask_forgetting_20260711/run_multitask_tests_4gpu.sbatch`
- Generate logs named by exact checkpoint, dataset, and head, for example `tests/balanced_lr350e-3_regular_matpes_Default.log` and `tests/balanced_lr350e-3_regular_ft2dp_ft2dp.log`, with the same naming pattern for the other four schemes and EMA checkpoints.
- Generate: `summary/baseline_metrics.csv`, `summary/multitask_metrics.csv`

- [ ] **Step 1: Test the single-head and pretrained baselines.**

Use `dp --pt test -n 0` on all 186 `train_data_iter11` systems and all MatPES systems. In DeepMD-kit, `-n` is the number of test frames to evaluate; `-n 0` is the sentinel for no frame limit, so the command evaluates every frame in the supplied systems instead of a truncated subset. This is required for comparable full-dataset metrics, not for training. For single-head models omit `--head`; for multi-head models use `--head Default` on MatPES and `--head ft2dp` on `train_data_iter11`.

- [ ] **Step 2: Test every regular and EMA multi-head checkpoint.**

The 20 multi-head evaluations are:

```text
5 schemes x 2 checkpoints (regular, EMA)
  x 2 datasets/heads (Default/MatPES, ft2dp/train_data_iter11)
```

Run four single-GPU child processes concurrently on one four-GPU `16V100` node. Each child must use `CUDA_VISIBLE_DEVICES`, `-n 0`, a unique output directory, and a `.rc` status file. The parent script must wait for all children and fail if any return code is nonzero.

- [ ] **Step 3: Parse metrics with explicit units and counts.**

The parser must retain `systems`, `frames`, `E_MAE`, `E_RMSE`, `E_MAE/N`, `E_RMSE/N`, `F_MAE`, `F_RMSE`, `V_MAE`, `V_RMSE`, `stress_MAE`, and `stress_RMSE`. It must not combine per-atom and total-energy metrics under one column. Confirm each full test log contains `# number of test data` and a weighted-average block.

### Task 6: Compare Target Accuracy and MatPES Forgetting

**Files:**
- Modify: `v2.2-ft/runs/dpa4_multitask_forgetting_20260711/parse_results.py`
- Generate: `summary/lcurve_metrics.csv`
- Generate: `summary/forgetting_report.json`
- Generate: `summary/forgetting_report.md`

- [ ] **Step 1: Parse per-head lcurve train/validation monitors.**

Extract the final-step `Default_trn`, `Default_val`, `ft2dp_trn`, and `ft2dp_val` columns from each scheme, together with the scheme-specific global step and actual `ft2dp` update count. Label these as sampled online monitors, not full-dataset validation. Because the existing dataset configuration uses the same system pool for training and validation, do not call these values an independent held-out validation set.

- [ ] **Step 2: Compute full-dataset target and retention deltas.**

For each multi-head checkpoint calculate:

```text
target_delta(metric) = multi_ft2dp_metric - single_head_metric
matpes_forgetting(metric) =
    (multi_Default_MatPES_metric - pretrained_MatPES_metric)
    / pretrained_MatPES_metric
```

Also calculate `single_head_forgetting` using the Task 1 baseline. Use the same model family and checkpoint type when comparing regular-to-regular and EMA-to-EMA.

- [ ] **Step 3: Rank schedules using a Pareto view.**

Identify:

1. The best Fe-C-H-O target metric among schemes whose MatPES degradation is no worse than the single-head baseline.
2. The best MatPES retention among schemes whose Fe-C-H-O force RMSE is within 10% of the best target scheme.
3. Whether lower LR reduces shared-descriptor drift at comparable target accuracy.
4. Whether `Default` replay probability changes retention monotonically.
5. Whether the `replay_lr350e-3` 20,000-update target exception retains acceptable `train_data_iter11` accuracy relative to the four 50,000-update schedules.

Do not select a winner from one scalar. Report energy, force, virial, and stress separately, with regular and EMA conclusions separated.

- [ ] **Step 4: Write the final report.**

`summary/forgetting_report.md` must contain:

- baseline single-head MatPES forgetting versus the untouched pretrained Air model;
- the exact five-scheme matrix and job/checkpoint mapping;
- target full-test metrics on `train_data_iter11`;
- MatPES full-test metrics for the `Default` head;
- lcurve sampled train/validation monitors with their non-held-out caveat;
- forgetting ratios and target deltas;
- recommended schedule and the evidence supporting it;
- failed jobs, retries, excluded nodes, and any residual data/validation limitations.

### Task 7: Final Verification and Handoff

- [ ] **Step 1: Run artifact integrity checks.**

Run:

```bash
find runs -path '*/models/model.ckpt-*.pt' -o \
           -path '*/models/model_ema.ckpt-*.pt' | sort
sha256sum runs/*/models/model*.pt
python parse_results.py --verify-complete summary/baseline_metrics.csv summary/multitask_metrics.csv
```

Expected: five schemes x two checkpoints x two heads x two datasets in the multi-head table, six baseline rows, all child status files equal to `0`, scheme-specific final checkpoints at global steps 100000/62500/100000/62500/100000, expected `ft2dp` update counts 50000/50000/20000/50000/50000, and no active Slurm jobs for this campaign.

- [ ] **Step 2: Check plan coverage before claiming completion.**

Confirm that the report answers all three requested questions: whether single-head forgetting exists, how five weight/LR schemes compare, and how each model performs on both target and MatPES data. Confirm that every reported metric is traceable to a full `dp test -n 0` log or explicitly labeled as lcurve monitoring.

- [ ] **Step 3: Commit only experiment definitions and report tooling if requested.**

Keep large checkpoints, extracted datasets, and raw Slurm logs on SAI/local experiment storage. Commit the generator, Slurm scripts, parser, README, and compact summary tables only if the user asks to version the campaign artifacts.

## Self-Review Checklist

- The single-head baseline is measured before multi-head runs, so forgetting is relative to the untouched pretrained model rather than inferred from the new campaign.
- The direct single-task-to-multi-task branch semantics use `Default` exactly as required by upstream `FinetuneRuleBuilder`; naming the retained branch `matpes` would incorrectly initialize it randomly.
- All full runs preserve `filter:255`; four runs target 50,000 expected sampled updates on the Fe-C-H-O branch, while `replay_lr350e-3` intentionally targets 20,000. Global `numb_steps` is 100000/62500/100000/62500/100000 according to the matrix.
- `dp test -n 0` means all test frames, not zero test frames; full metrics, not sampled lcurve values, drive the target and forgetting comparisons.
- SAI resource requests obey the `16V100`/four-GPU/QOS rules and omit forbidden CPU/memory requests.
- No source-code changes are planned because the confirmed functionality is implemented in upstream DeepMD-kit and the campaign is an experiment orchestration/reporting task.

---

## Status (2026-09-03 回写)

- **已完成**：五方案多头矩阵 + 基线/单头遗忘评测全部执行并出报告（`v2.2-ft/runs/dpa4_multitask_forgetting_20260711/summary/`，含 `forgetting_report.md`）。
- **核心结论**：单头 50k@3.5e-4 灾难遗忘确证（MatPES E MAE 377×）；多头中 `replay_lr350e-3` 保持最佳、`balanced_lr100e-3` 力误差最均衡、`target_lr350e-3` 域内最佳但保持崩塌。
- **后续衔接**：低 LR 补充 campaign（2026-07-17, auto:255+clip5）single 组已完成并测（lr1e-4 域内 E MAE 3.22 meV）；balanced 组因 16GB V100 OOM 于 9/3 以 `_r2` 重训（32GB）。任务主线见工作区 `docs/tasks/ft2dp-v2.2.md`。
- **当前状态来源**：本计划的原始 checkbox 仅记录当时设计；后续状态、验收和未决项以工作区 `../../../../docs/tasks/ft2dp-v2.2.md` 为准。
