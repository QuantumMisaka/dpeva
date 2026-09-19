---
title: SAI DPA4 Env Rename And Fine-Tune Implementation Plan
status: record
audience: Developers / AI Agents
last-updated: 2026-09-03
owner: Scientific Owner
---

# SAI DPA4 Env Rename And Fine-Tune Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the upstream DeepMD-kit CUDA 12.6.3 environment the canonical `dpeva-dpa4` conda environment on SAI-new, then use it to fine-tune DPA4-Air-ZBL-MatPES on `train_data_iter11` with one 4-GPU job on the `16V100` partition.

**Architecture:** Treat the environment rename as an operational migration with manifest backups and post-migration import checks before any training work. Fine-tuning uses DeepMD-kit directly inside the DP-EVA/DPA4 environment instead of `dpeva train`, because the current DP-EVA training workflow is ensemble-oriented (`num_models >= 3`) and would naturally submit multiple jobs, while the requested resource envelope is one job using 4 V100 GPUs. Run artifacts live under one timestamp-free, named run directory so local and remote copies remain comparable.

**Tech Stack:** SAI-new SSH, Conda, Environment Modules `cuda/12.6.3`, PyTorch `torchrun`, DeepMD-kit PyTorch backend `dp --pt`, Slurm `16V100`, `rush-gpu` for smoke, `huge-gpu` for full 50000-step training.

---

## Key Decisions

- Use `numb_steps: 50000`, not `num_epochs`, for the full run. Upstream DeepMD-kit normalizes `num_epochs`/`num_epoch` into `numb_epoch`; if `numb_steps` is absent, PyTorch training computes `num_steps = ceil(num_epoch * total_numb_batch)`. That makes epoch mode an expected sampling target, while `numb_steps` exactly expresses 50000 batch updates.
- Use `training.training_data.batch_size = "filter:255"`. On `train_data_iter11`, the largest discovered `type.raw` system has 254 atoms, so this filter should not drop any system.
- Keep the pretraining input values from `DPA4-Air-ZBL-MatPES-v20260629.json` except the fields required to run this dataset: training systems, validation systems, training batch size, and training length.
- Use clone-and-remove to implement the conda environment rename. A filesystem `mv` is unsafe for conda environments because scripts and metadata can contain absolute prefixes.
- Remove the old `dpeva-dpa4` environment before cloning. Remove `dpeva-dpa4-cu126-upstream` only after the cloned canonical `dpeva-dpa4` passes import and CLI checks.

## File Structure

- Modify: `scripts/env/dpeva-dpa4.env`
  Canonical DP-EVA/DPA4 environment loader. After the migration it activates `dpeva-dpa4`, loads CUDA 12.6.3, and exports the PyTorch DeepMD-kit runtime variables.
- Delete: `scripts/env/dpeva-dpa4-cu126-upstream.env`
  Temporary loader for the upstream build. Remove after `dpeva-dpa4.env` points to the upstream-backed canonical environment.
- Create remote only: `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/env-manifests/20260710-dpeva-dpa4-rename/`
  Manifest and verification snapshots from before and after the conda migration.
- Create local and remote: `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/v2.2-ft/runs/dpa4_air_matpes_zbl_train_data_iter11_50k/`
  Fine-tuning run directory containing `input.json`, `input.smoke.json`, Slurm scripts, logs, checkpoint output, and summary files. Mirror this under `/home/james/work/ft2dp-dpeva/v2.2-ft/runs/dpa4_air_matpes_zbl_train_data_iter11_50k/`.
- Read remote: `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/models/426/DPA4-MatPES-ZBL-v20260629/inputs/DPA4-Air-ZBL-MatPES-v20260629.json`
  Source input JSON from the downloaded AISSquare package.
- Read remote: `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/models/426/DPA4-MatPES-ZBL-v20260629/checkpoints/DPA4-Air-ZBL-MatPES-v20260629.pt`
  Fine-tune starting checkpoint.
- Read remote: `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/v2.2-ft/data/train_data_iter11`
  Training dataset.

### Task 1: Back Up Current Remote Conda Environments

**Files:**
- Create remote: `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/env-manifests/20260710-dpeva-dpa4-rename/dpeva-dpa4.before.explicit.txt`
- Create remote: `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/env-manifests/20260710-dpeva-dpa4-rename/dpeva-dpa4-cu126-upstream.before.explicit.txt`
- Create remote: `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/env-manifests/20260710-dpeva-dpa4-rename/dpeva-dpa4.before.pip-freeze.txt`
- Create remote: `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/env-manifests/20260710-dpeva-dpa4-rename/dpeva-dpa4-cu126-upstream.before.pip-freeze.txt`
- Create remote: `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/env-manifests/20260710-dpeva-dpa4-rename/preflight.txt`

- [ ] **Step 1: Record both environment prefixes and package manifests**

Run:

```bash
ssh SAI-new 'bash -lc "
set -euo pipefail
source /opt/devtools/anaconda3/etc/profile.d/conda.sh
out=/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/env-manifests/20260710-dpeva-dpa4-rename
mkdir -p \"${out}\"
conda env list | tee \"${out}/preflight.txt\"
conda list -n dpeva-dpa4 --explicit > \"${out}/dpeva-dpa4.before.explicit.txt\"
conda list -n dpeva-dpa4-cu126-upstream --explicit > \"${out}/dpeva-dpa4-cu126-upstream.before.explicit.txt\"
/home/pku-jianghong/liuzhaoqing/.conda/envs/dpeva-dpa4/bin/python -m pip freeze > \"${out}/dpeva-dpa4.before.pip-freeze.txt\"
/home/pku-jianghong/liuzhaoqing/.conda/envs/dpeva-dpa4-cu126-upstream/bin/python -m pip freeze > \"${out}/dpeva-dpa4-cu126-upstream.before.pip-freeze.txt\"
/home/pku-jianghong/liuzhaoqing/.conda/envs/dpeva-dpa4/bin/python -m pip show deepmd-kit | tee \"${out}/dpeva-dpa4.before.deepmd-kit.txt\"
/home/pku-jianghong/liuzhaoqing/.conda/envs/dpeva-dpa4-cu126-upstream/bin/python -m pip show deepmd-kit | tee \"${out}/dpeva-dpa4-cu126-upstream.before.deepmd-kit.txt\"
"
'
```

Expected:

```text
dpeva-dpa4 ... /home/pku-jianghong/liuzhaoqing/.conda/envs/dpeva-dpa4
dpeva-dpa4-cu126-upstream ... /home/pku-jianghong/liuzhaoqing/.conda/envs/dpeva-dpa4-cu126-upstream
Name: deepmd-kit
Version: 3.1.3
Name: deepmd-kit
Version: 3.2.0b1.dev115+gf73de32e2
```

- [ ] **Step 2: Verify no active Slurm job depends on the old environment name**

Run:

```bash
ssh SAI-new 'bash -lc "
set -euo pipefail
squeue -u \"${USER}\" || true
ps -u \"${USER}\" -o pid,ppid,stat,etime,cmd | grep -E \"dpeva-dpa4|dpeva-dpa4-cu126-upstream|dp --pt|torchrun\" | grep -v grep || true
"
'
```

Expected: no running `dp --pt` or `torchrun` process using either environment. If `squeue` shows unrelated completed or pending jobs, record them in the final handoff and proceed only when no running job is using the target environments.

### Task 2: Replace `dpeva-dpa4` With The Upstream-Built Environment

**Files:**
- Modify remote conda env: `/home/pku-jianghong/liuzhaoqing/.conda/envs/dpeva-dpa4`
- Remove remote conda env after validation: `/home/pku-jianghong/liuzhaoqing/.conda/envs/dpeva-dpa4-cu126-upstream`
- Create remote: `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/env-manifests/20260710-dpeva-dpa4-rename/dpeva-dpa4.after.deepmd-kit.txt`
- Create remote: `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/env-manifests/20260710-dpeva-dpa4-rename/dpeva-dpa4.after.verify.txt`

- [ ] **Step 1: Remove the old canonical environment**

Run:

```bash
ssh SAI-new 'bash -lc "
set -euo pipefail
source /opt/devtools/anaconda3/etc/profile.d/conda.sh
conda env remove -n dpeva-dpa4 -y
test ! -d /home/pku-jianghong/liuzhaoqing/.conda/envs/dpeva-dpa4
"
'
```

Expected:

```text
Remove all packages in environment /home/pku-jianghong/liuzhaoqing/.conda/envs/dpeva-dpa4:
```

and the final `test` exits with code `0`.

- [ ] **Step 2: Clone the upstream-built environment into the canonical name**

Run:

```bash
ssh SAI-new 'bash -lc "
set -euo pipefail
source /opt/devtools/anaconda3/etc/profile.d/conda.sh
conda create -n dpeva-dpa4 --clone dpeva-dpa4-cu126-upstream -y
test -x /home/pku-jianghong/liuzhaoqing/.conda/envs/dpeva-dpa4/bin/python
test -x /home/pku-jianghong/liuzhaoqing/.conda/envs/dpeva-dpa4/bin/dp
"
'
```

Expected:

```text
Source:      /home/pku-jianghong/liuzhaoqing/.conda/envs/dpeva-dpa4-cu126-upstream
Destination: /home/pku-jianghong/liuzhaoqing/.conda/envs/dpeva-dpa4
```

- [ ] **Step 3: Verify the cloned canonical environment before deleting the source**

Run:

```bash
ssh SAI-new 'bash -lc "
set -euo pipefail
out=/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/env-manifests/20260710-dpeva-dpa4-rename
/home/pku-jianghong/liuzhaoqing/.conda/envs/dpeva-dpa4/bin/python -m pip show deepmd-kit | tee \"${out}/dpeva-dpa4.after.deepmd-kit.txt\"
/home/pku-jianghong/liuzhaoqing/.conda/envs/dpeva-dpa4/bin/python - <<'"'"'PY'"'"' | tee \"${out}/dpeva-dpa4.after.verify.txt\"
import pathlib
import deepmd
import torch
import dpeva
print(\"deepmd\", deepmd.__version__, pathlib.Path(deepmd.__file__).resolve())
print(\"torch\", torch.__version__, \"cuda\", torch.version.cuda, \"cuda_available\", torch.cuda.is_available())
print(\"dpeva\", pathlib.Path(dpeva.__file__).resolve())
assert deepmd.__version__ == \"3.2.0b1.dev115+gf73de32e2\"
assert torch.__version__.startswith(\"2.11.0\")
assert torch.version.cuda == \"12.6\"
PY
"
'
```

Expected:

```text
Version: 3.2.0b1.dev115+gf73de32e2
deepmd 3.2.0b1.dev115+gf73de32e2 ...
torch 2.11.0+cu126 cuda 12.6 ...
```

- [ ] **Step 4: Remove the temporary upstream-named environment**

Run:

```bash
ssh SAI-new 'bash -lc "
set -euo pipefail
source /opt/devtools/anaconda3/etc/profile.d/conda.sh
conda env remove -n dpeva-dpa4-cu126-upstream -y
test ! -d /home/pku-jianghong/liuzhaoqing/.conda/envs/dpeva-dpa4-cu126-upstream
conda env list | tee /home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/env-manifests/20260710-dpeva-dpa4-rename/env-list.after.txt
"
'
```

Expected: `dpeva-dpa4` remains listed and `dpeva-dpa4-cu126-upstream` is absent.

### Task 3: Update The Canonical Environment Loader

**Files:**
- Modify local: `/home/james/work/ft2dp-dpeva/dpeva/scripts/env/dpeva-dpa4.env`
- Delete local: `/home/james/work/ft2dp-dpeva/dpeva/scripts/env/dpeva-dpa4-cu126-upstream.env`
- Modify remote: `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/dpeva/scripts/env/dpeva-dpa4.env`
- Delete remote: `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/dpeva/scripts/env/dpeva-dpa4-cu126-upstream.env`
- Test remote: source canonical env and import DeepMD-kit/Torch/DP-EVA

- [ ] **Step 1: Replace the local canonical env script**

Use `apply_patch` to make `/home/james/work/ft2dp-dpeva/dpeva/scripts/env/dpeva-dpa4.env` exactly:

```bash
#!/usr/bin/env bash
# DP-EVA DPA4 environment on SAI-new, backed by upstream DeepMD-kit.
# Source this file before running DP-EVA or DeepMD-kit DPA4 validation jobs.

set -euo pipefail

ENV_NAME="${DPEVA_DPA4_ENV_NAME:-dpeva-dpa4}"
CUDA_MODULE_SELECTED="${CUDA_MODULE_BUILT:-cuda/12.6.3}"
CUDA_VERSION_SELECTED="${CUDA_MODULE_SELECTED#cuda/}"

# Lmod init scripts may reference these variables directly; define them before
# enabling nounset-sensitive shells.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="${LD_PRELOAD:-}"

source /opt/devtools/anaconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME}"

source /etc/profile.d/modules.sh 2>/dev/null || true
if command -v module >/dev/null 2>&1; then
    module purge 2>/dev/null || true
    module load "${CUDA_MODULE_SELECTED}"
fi

export CUDA_HOME="/opt/devtools/nvidia/cuda-${CUDA_VERSION_SELECTED}"
export CUDAToolkit_ROOT="${CUDA_HOME}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"

export TORCH_CUDA_ARCH_LIST="7.0"
export CMAKE_CUDA_ARCHITECTURES="70"
export CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=70"

export DP_VARIANT="cuda"
export DP_ENABLE_PYTORCH="1"
export DP_ENABLE_TENSORFLOW="0"
export DP_ENABLE_PADDLE="0"
export DP_ENABLE_JAX="0"
export DP_COMPILE_INFER="${DP_COMPILE_INFER:-0}"
export DP_INTERFACE_PREC="${DP_INTERFACE_PREC:-high}"
```

- [ ] **Step 2: Remove the local temporary env loader**

Run:

```bash
rm -f /home/james/work/ft2dp-dpeva/dpeva/scripts/env/dpeva-dpa4-cu126-upstream.env
```

Expected: `git status --short` no longer shows `?? scripts/env/dpeva-dpa4-cu126-upstream.env`.

- [ ] **Step 3: Sync the canonical env script to SAI-new**

Run:

```bash
scp /home/james/work/ft2dp-dpeva/dpeva/scripts/env/dpeva-dpa4.env \
  SAI-new:/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/dpeva/scripts/env/dpeva-dpa4.env
ssh SAI-new 'rm -f /home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/dpeva/scripts/env/dpeva-dpa4-cu126-upstream.env'
```

Expected: the remote temporary env loader is absent and the canonical loader is present.

- [ ] **Step 4: Verify the canonical env loader on SAI-new**

Run:

```bash
ssh SAI-new 'bash -lc "
set -euo pipefail
source /home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/dpeva/scripts/env/dpeva-dpa4.env
echo CONDA_PREFIX=${CONDA_PREFIX}
echo CUDA_HOME=${CUDA_HOME}
which python
which dp
python - <<'"'"'PY'"'"'
import pathlib
import deepmd
import torch
import dpeva
print(\"deepmd\", deepmd.__version__, pathlib.Path(deepmd.__file__).resolve())
print(\"torch\", torch.__version__, torch.version.cuda)
print(\"dpeva\", pathlib.Path(dpeva.__file__).resolve())
assert deepmd.__version__ == \"3.2.0b1.dev115+gf73de32e2\"
assert torch.__version__.startswith(\"2.11.0\")
assert torch.version.cuda == \"12.6\"
PY
"
'
```

Expected:

```text
CONDA_PREFIX=/home/pku-jianghong/liuzhaoqing/.conda/envs/dpeva-dpa4
CUDA_HOME=/opt/devtools/nvidia/cuda-12.6.3
deepmd 3.2.0b1.dev115+gf73de32e2
torch 2.11.0+cu126 12.6
```

- [ ] **Step 5: Commit the local env script change**

Run:

```bash
cd /home/james/work/ft2dp-dpeva/dpeva
git add scripts/env/dpeva-dpa4.env
git status --short
git commit -m "chore: make dpeva-dpa4 use upstream deepmd"
```

Expected: staged change only touches `scripts/env/dpeva-dpa4.env`. The deleted untracked temporary loader is not part of the commit because it was never tracked.

### Task 4: Generate The Fine-Tuning Input Files

**Files:**
- Create local: `/home/james/work/ft2dp-dpeva/v2.2-ft/runs/dpa4_air_matpes_zbl_train_data_iter11_50k/input.json`
- Create local: `/home/james/work/ft2dp-dpeva/v2.2-ft/runs/dpa4_air_matpes_zbl_train_data_iter11_50k/input.smoke.json`
- Create remote: `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/v2.2-ft/runs/dpa4_air_matpes_zbl_train_data_iter11_50k/input.json`
- Create remote: `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/v2.2-ft/runs/dpa4_air_matpes_zbl_train_data_iter11_50k/input.smoke.json`
- Read remote source: `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/models/426/DPA4-MatPES-ZBL-v20260629/inputs/DPA4-Air-ZBL-MatPES-v20260629.json`

- [ ] **Step 1: Create the local and remote run directories**

Run:

```bash
mkdir -p /home/james/work/ft2dp-dpeva/v2.2-ft/runs/dpa4_air_matpes_zbl_train_data_iter11_50k
ssh SAI-new 'mkdir -p /home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/v2.2-ft/runs/dpa4_air_matpes_zbl_train_data_iter11_50k'
```

Expected: both directories exist.

- [ ] **Step 2: Generate `input.json` and `input.smoke.json` on SAI-new from the pretrained input**

Run:

```bash
ssh SAI-new 'bash -lc "
set -euo pipefail
run=/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/v2.2-ft/runs/dpa4_air_matpes_zbl_train_data_iter11_50k
src=/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/models/426/DPA4-MatPES-ZBL-v20260629/inputs/DPA4-Air-ZBL-MatPES-v20260629.json
data=/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/v2.2-ft/data/train_data_iter11
/home/pku-jianghong/liuzhaoqing/.conda/envs/dpeva-dpa4/bin/python - <<'"'"'PY'"'"'
import copy
import json
from pathlib import Path

run = Path(\"/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/v2.2-ft/runs/dpa4_air_matpes_zbl_train_data_iter11_50k\")
src = Path(\"/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/models/426/DPA4-MatPES-ZBL-v20260629/inputs/DPA4-Air-ZBL-MatPES-v20260629.json\")
data = \"/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/v2.2-ft/data/train_data_iter11\"

cfg = json.loads(src.read_text())
training = cfg.setdefault(\"training\", {})
training_data = training.setdefault(\"training_data\", {})
validation_data = training.setdefault(\"validation_data\", {})

training_data[\"systems\"] = data
training_data[\"batch_size\"] = \"filter:255\"
validation_data[\"systems\"] = data
validation_data[\"batch_size\"] = 1
validation_data[\"numb_batch\"] = 1

for key in (\"num_epochs\", \"num_epoch\", \"numb_epoch\", \"numb_epochs\", \"stop_batch\", \"num_step\", \"num_steps\", \"numb_step\"):
    training.pop(key, None)
training[\"numb_steps\"] = 50000
cfg[\"_comment\"] = \"DPA4-Air-ZBL-MatPES-v20260629 fine-tune on train_data_iter11; batch_size filter:255; numb_steps 50000.\"

smoke = copy.deepcopy(cfg)
smoke[\"training\"][\"numb_steps\"] = 2
smoke[\"training\"][\"save_freq\"] = 1
smoke[\"training\"][\"disp_freq\"] = 1
smoke[\"_comment\"] = \"Smoke test for DPA4-Air-ZBL-MatPES train_data_iter11 fine-tune; 2 steps only.\"

run.mkdir(parents=True, exist_ok=True)
(run / \"input.json\").write_text(json.dumps(cfg, indent=2) + \"\\n\")
(run / \"input.smoke.json\").write_text(json.dumps(smoke, indent=2) + \"\\n\")
print(run / \"input.json\")
print(run / \"input.smoke.json\")
PY
"
'
```

Expected: both input files are written in the remote run directory.

- [ ] **Step 3: Validate the generated input fields**

Run:

```bash
ssh SAI-new 'bash -lc "
set -euo pipefail
run=/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/v2.2-ft/runs/dpa4_air_matpes_zbl_train_data_iter11_50k
/home/pku-jianghong/liuzhaoqing/.conda/envs/dpeva-dpa4/bin/python - <<'"'"'PY'"'"'
import json
from pathlib import Path
run = Path(\"/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/v2.2-ft/runs/dpa4_air_matpes_zbl_train_data_iter11_50k\")
cfg = json.loads((run / \"input.json\").read_text())
smoke = json.loads((run / \"input.smoke.json\").read_text())
tr = cfg[\"training\"]
assert tr[\"training_data\"][\"systems\"] == \"/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/v2.2-ft/data/train_data_iter11\"
assert tr[\"training_data\"][\"batch_size\"] == \"filter:255\"
assert tr[\"validation_data\"][\"systems\"] == tr[\"training_data\"][\"systems\"]
assert tr[\"numb_steps\"] == 50000
assert \"num_epochs\" not in tr
assert \"numb_epoch\" not in tr
assert smoke[\"training\"][\"numb_steps\"] == 2
assert smoke[\"training\"][\"training_data\"][\"batch_size\"] == \"filter:255\"
print(\"validated\", tr[\"numb_steps\"], tr[\"training_data\"][\"batch_size\"], cfg[\"model\"][\"type\"], cfg[\"training\"][\"save_freq\"])
PY
"
'
```

Expected:

```text
validated 50000 filter:255 SeZM 2000
```

- [ ] **Step 4: Confirm `filter:255` keeps all current systems**

Run:

```bash
ssh SAI-new '/home/pku-jianghong/liuzhaoqing/.conda/envs/dpeva-dpa4/bin/python -' <<'PY'
from pathlib import Path
root = Path('/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/v2.2-ft/data/train_data_iter11')
sizes = []
for p in sorted(root.rglob('type.raw')):
    sizes.append((len(p.read_text().split()), p.parent))
assert sizes, 'no type.raw files found'
gt255 = [(n, p) for n, p in sizes if n > 255]
print('systems_with_type_raw', len(sizes))
print('min_atoms', min(n for n, _ in sizes))
print('max_atoms', max(n for n, _ in sizes))
print('gt255', len(gt255))
assert len(gt255) == 0
PY
```

Expected:

```text
systems_with_type_raw 186
min_atoms 2
max_atoms 254
gt255 0
```

- [ ] **Step 5: Mirror generated inputs back to local**

Run:

```bash
rsync -av \
  SAI-new:/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/v2.2-ft/runs/dpa4_air_matpes_zbl_train_data_iter11_50k/input.json \
  SAI-new:/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/v2.2-ft/runs/dpa4_air_matpes_zbl_train_data_iter11_50k/input.smoke.json \
  /home/james/work/ft2dp-dpeva/v2.2-ft/runs/dpa4_air_matpes_zbl_train_data_iter11_50k/
```

Expected: local `input.json` and `input.smoke.json` match the remote files.

### Task 5: Create Slurm Scripts For Smoke And Full Training

**Files:**
- Create local: `/home/james/work/ft2dp-dpeva/v2.2-ft/runs/dpa4_air_matpes_zbl_train_data_iter11_50k/run_smoke_4v100.slurm`
- Create local: `/home/james/work/ft2dp-dpeva/v2.2-ft/runs/dpa4_air_matpes_zbl_train_data_iter11_50k/run_finetune_4v100.slurm`
- Create remote: `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/v2.2-ft/runs/dpa4_air_matpes_zbl_train_data_iter11_50k/run_smoke_4v100.slurm`
- Create remote: `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/v2.2-ft/runs/dpa4_air_matpes_zbl_train_data_iter11_50k/run_finetune_4v100.slurm`

- [ ] **Step 1: Create the smoke Slurm script locally**

Create `/home/james/work/ft2dp-dpeva/v2.2-ft/runs/dpa4_air_matpes_zbl_train_data_iter11_50k/run_smoke_4v100.slurm` with:

```bash
#!/bin/bash
#SBATCH -J dpa4air-it11-smoke
#SBATCH -p 16V100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus-per-node=4
#SBATCH --qos=rush-gpu
#SBATCH -t 00:30:00
#SBATCH --open-mode=truncate
#SBATCH -o slurm-smoke-%j.out
#SBATCH -e slurm-smoke-%j.err

set -euo pipefail

RUN_DIR="/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/v2.2-ft/runs/dpa4_air_matpes_zbl_train_data_iter11_50k"
BASE_MODEL="/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/models/426/DPA4-MatPES-ZBL-v20260629/checkpoints/DPA4-Air-ZBL-MatPES-v20260629.pt"

cd "${RUN_DIR}"
source /home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/dpeva/scripts/env/dpeva-dpa4.env

export OMP_NUM_THREADS=2
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

nvidia-smi dmon -s pucvmte -o T > "nvdmon_smoke_${SLURM_JOB_ID}.log" &

torchrun --nproc_per_node="${SLURM_GPUS_ON_NODE:-4}" \
    --no-python --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    dp --pt train input.smoke.json --finetune "${BASE_MODEL}" --skip-neighbor-stat \
    > train.smoke.log 2>&1

echo "DPEVA_DPA4_SMOKE_FINISHED"
```

- [ ] **Step 2: Create the full Slurm script locally**

Create `/home/james/work/ft2dp-dpeva/v2.2-ft/runs/dpa4_air_matpes_zbl_train_data_iter11_50k/run_finetune_4v100.slurm` with:

```bash
#!/bin/bash
#SBATCH -J dpa4air-it11-ft50k
#SBATCH -p 16V100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus-per-node=4
#SBATCH --qos=huge-gpu
#SBATCH -t 72:00:00
#SBATCH --open-mode=truncate
#SBATCH -o slurm-ft50k-%j.out
#SBATCH -e slurm-ft50k-%j.err

set -euo pipefail

RUN_DIR="/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/v2.2-ft/runs/dpa4_air_matpes_zbl_train_data_iter11_50k"
BASE_MODEL="/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/models/426/DPA4-MatPES-ZBL-v20260629/checkpoints/DPA4-Air-ZBL-MatPES-v20260629.pt"

cd "${RUN_DIR}"
source /home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/dpeva/scripts/env/dpeva-dpa4.env

export OMP_NUM_THREADS=2
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

nvidia-smi dmon -s pucvmte -o T > "nvdmon_ft50k_${SLURM_JOB_ID}.log" &

torchrun --nproc_per_node="${SLURM_GPUS_ON_NODE:-4}" \
    --no-python --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    dp --pt train input.json --finetune "${BASE_MODEL}" --skip-neighbor-stat \
    > train.log 2>&1

echo "DPEVA_DPA4_FINETUNE_FINISHED"
```

- [ ] **Step 3: Check Slurm scripts for shell syntax**

Run:

```bash
bash -n /home/james/work/ft2dp-dpeva/v2.2-ft/runs/dpa4_air_matpes_zbl_train_data_iter11_50k/run_smoke_4v100.slurm
bash -n /home/james/work/ft2dp-dpeva/v2.2-ft/runs/dpa4_air_matpes_zbl_train_data_iter11_50k/run_finetune_4v100.slurm
```

Expected: both commands exit with code `0`.

- [ ] **Step 4: Sync Slurm scripts to SAI-new**

Run:

```bash
rsync -av \
  /home/james/work/ft2dp-dpeva/v2.2-ft/runs/dpa4_air_matpes_zbl_train_data_iter11_50k/run_smoke_4v100.slurm \
  /home/james/work/ft2dp-dpeva/v2.2-ft/runs/dpa4_air_matpes_zbl_train_data_iter11_50k/run_finetune_4v100.slurm \
  SAI-new:/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/v2.2-ft/runs/dpa4_air_matpes_zbl_train_data_iter11_50k/
```

Expected: remote scripts exist with the same byte size as local scripts.

- [ ] **Step 5: Record local run inputs and scripts as operational artifacts**

Run:

```bash
cd /home/james/work/ft2dp-dpeva/dpeva
git status --short
find /home/james/work/ft2dp-dpeva/v2.2-ft/runs/dpa4_air_matpes_zbl_train_data_iter11_50k -maxdepth 1 -type f -print | sort
```

Expected: `git status --short` only reports repository files, and the `find` command lists the local run input and Slurm script artifacts. Do not add the run directory to this repository because `/home/james/work/ft2dp-dpeva/v2.2-ft` is outside `/home/james/work/ft2dp-dpeva/dpeva`.

### Task 6: Submit And Verify The Smoke Job

**Files:**
- Read remote: `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/v2.2-ft/runs/dpa4_air_matpes_zbl_train_data_iter11_50k/train.smoke.log`
- Read remote: `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/v2.2-ft/runs/dpa4_air_matpes_zbl_train_data_iter11_50k/slurm-smoke-<jobid>.out`
- Read remote: `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/v2.2-ft/runs/dpa4_air_matpes_zbl_train_data_iter11_50k/slurm-smoke-<jobid>.err`

- [ ] **Step 1: Submit the smoke job**

Run:

```bash
ssh SAI-new 'bash -lc "
set -euo pipefail
cd /home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/v2.2-ft/runs/dpa4_air_matpes_zbl_train_data_iter11_50k
sbatch run_smoke_4v100.slurm
"
'
```

Expected:

```text
Submitted batch job <jobid>
```

- [ ] **Step 2: Monitor the smoke job**

Run, replacing `<jobid>` with the submitted job id:

```bash
ssh SAI-new 'bash -lc "
squeue -j <jobid>
sacct -j <jobid> --format=JobID,JobName,State,ExitCode,Elapsed,NodeList -P
"
'
```

Expected final state:

```text
<jobid>|dpa4air-it11-smoke|COMPLETED|0:0|...
```

- [ ] **Step 3: Verify smoke logs and checkpoint output**

Run:

```bash
ssh SAI-new 'bash -lc "
set -euo pipefail
run=/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/v2.2-ft/runs/dpa4_air_matpes_zbl_train_data_iter11_50k
cd \"${run}\"
grep -R -nE \"Traceback|RuntimeError|CUDA out of memory|Analysis failed|ERROR\" train.smoke.log slurm-smoke-*.err && exit 1 || true
grep -R \"DPEVA_DPA4_SMOKE_FINISHED\" slurm-smoke-*.out
test -d models
ls -lh train.smoke.log slurm-smoke-*.out slurm-smoke-*.err
find models -maxdepth 1 -type f -print | sort | tail
"
'
```

Expected:

```text
DPEVA_DPA4_SMOKE_FINISHED
models/<checkpoint files>
```

### Task 7: Submit And Monitor The Full 50000-Step Fine-Tune

**Files:**
- Read remote: `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/v2.2-ft/runs/dpa4_air_matpes_zbl_train_data_iter11_50k/train.log`
- Read remote: `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/v2.2-ft/runs/dpa4_air_matpes_zbl_train_data_iter11_50k/models/`

- [ ] **Step 1: Submit the full job only after smoke passes**

Run:

```bash
ssh SAI-new 'bash -lc "
set -euo pipefail
cd /home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/v2.2-ft/runs/dpa4_air_matpes_zbl_train_data_iter11_50k
sbatch run_finetune_4v100.slurm
"
'
```

Expected:

```text
Submitted batch job <jobid>
```

- [ ] **Step 2: Record the full job id and initial allocation state**

Run, replacing `<jobid>`:

```bash
ssh SAI-new 'bash -lc "
squeue -j <jobid>
sacct -j <jobid> --format=JobID,JobName,State,ExitCode,Elapsed,NodeList -P
"
'
```

Expected while pending or running: one job named `dpa4air-it11-ft50k` on partition `16V100` requesting `4` GPUs per node.

- [ ] **Step 3: Monitor lcurve progress without scanning the full dataset**

Run:

```bash
ssh SAI-new 'bash -lc "
run=/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/v2.2-ft/runs/dpa4_air_matpes_zbl_train_data_iter11_50k
cd \"${run}\"
tail -n 40 train.log 2>/dev/null || true
tail -n 20 lcurve.out 2>/dev/null || true
ls -lh models 2>/dev/null || true
"
'
```

Expected during training: `train.log` advances, `lcurve.out` contains step/loss rows, and `models/` contains checkpoint files.

- [ ] **Step 4: Verify completion and final checkpoint**

Run after Slurm reports completion:

```bash
ssh SAI-new 'bash -lc "
set -euo pipefail
run=/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/v2.2-ft/runs/dpa4_air_matpes_zbl_train_data_iter11_50k
cd \"${run}\"
grep -R -nE \"Traceback|RuntimeError|CUDA out of memory|ERROR\" train.log slurm-ft50k-*.err && exit 1 || true
grep -R \"DPEVA_DPA4_FINETUNE_FINISHED\" slurm-ft50k-*.out
test -d models
ls -lh train.log lcurve.out slurm-ft50k-*.out slurm-ft50k-*.err
find models -maxdepth 1 -type f -print | sort | tail
"
'
```

Expected:

```text
DPEVA_DPA4_FINETUNE_FINISHED
models/<final checkpoint files>
```

### Task 8: Mirror Results And Write The Handoff Summary

**Files:**
- Create local: `/home/james/work/ft2dp-dpeva/v2.2-ft/runs/dpa4_air_matpes_zbl_train_data_iter11_50k/`
- Create remote: `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/v2.2-ft/runs/dpa4_air_matpes_zbl_train_data_iter11_50k/RUN_SUMMARY.md`
- Create local: `/home/james/work/ft2dp-dpeva/v2.2-ft/runs/dpa4_air_matpes_zbl_train_data_iter11_50k/RUN_SUMMARY.md`

- [ ] **Step 1: Create a remote run summary**

Run after the full job completes, replacing `<jobid>` with the full job id:

```bash
ssh SAI-new 'bash -lc "
set -euo pipefail
run=/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/v2.2-ft/runs/dpa4_air_matpes_zbl_train_data_iter11_50k
cd \"${run}\"
cat > RUN_SUMMARY.md <<'"'"'EOF'"'"'
# DPA4-Air-ZBL-MatPES train_data_iter11 Fine-Tune

- Environment: dpeva-dpa4
- DeepMD-kit: 3.2.0b1.dev115+gf73de32e2
- CUDA module: cuda/12.6.3
- Torch: 2.11.0+cu126
- Partition: 16V100
- GPUs: 4 V100 on one node
- QOS: huge-gpu
- Base model: /home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/models/426/DPA4-MatPES-ZBL-v20260629/checkpoints/DPA4-Air-ZBL-MatPES-v20260629.pt
- Base input: /home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/models/426/DPA4-MatPES-ZBL-v20260629/inputs/DPA4-Air-ZBL-MatPES-v20260629.json
- Training data: /home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/v2.2-ft/data/train_data_iter11
- Batch size: filter:255
- Steps: 50000
- Full Slurm job: <jobid>
- Output checkpoints: models/
- Note: validation_data points to the same dataset with numb_batch=1 for training-time monitoring because no separate validation set was provided.
EOF
"
'
```

Expected: `RUN_SUMMARY.md` exists in the remote run directory.

- [ ] **Step 2: Mirror the final run directory to local**

Run:

```bash
rsync -av \
  SAI-new:/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/v2.2-ft/runs/dpa4_air_matpes_zbl_train_data_iter11_50k/ \
  /home/james/work/ft2dp-dpeva/v2.2-ft/runs/dpa4_air_matpes_zbl_train_data_iter11_50k/
```

Expected: local run directory contains inputs, Slurm scripts, logs, summary, and the `models/` checkpoint folder.

- [ ] **Step 3: Final verification**

Run:

```bash
ssh SAI-new 'bash -lc "
set -euo pipefail
source /home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/dpeva/scripts/env/dpeva-dpa4.env
python - <<'"'"'PY'"'"'
import pathlib
import deepmd
import torch
run = pathlib.Path('/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/v2.2-ft/runs/dpa4_air_matpes_zbl_train_data_iter11_50k')
models = run / 'models'
assert models.is_dir()
checkpoints = sorted(p for p in models.iterdir() if p.is_file())
assert checkpoints, 'no checkpoint files written'
print('deepmd', deepmd.__version__)
print('torch', torch.__version__, torch.version.cuda)
print('checkpoint_count', len(checkpoints))
print('last_checkpoint', checkpoints[-1], checkpoints[-1].stat().st_size)
PY
"
'
```

Expected:

```text
deepmd 3.2.0b1.dev115+gf73de32e2
torch 2.11.0+cu126 12.6
checkpoint_count <positive-count>
last_checkpoint /home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/v2.2-ft/runs/dpa4_air_matpes_zbl_train_data_iter11_50k/models/<checkpoint> <positive-size>
```

## Self-Review

**Spec coverage:** The plan removes the old `dpeva-dpa4`, promotes `dpeva-dpa4-cu126-upstream` into the canonical `dpeva-dpa4` name, updates the canonical loader, uses the SAI-new CUDA 12.6.3 module, runs on `16V100` with one 4-GPU job, uses DPA4-Air-ZBL-MatPES as the base model, uses `filter:255`, and uses exactly `numb_steps: 50000`.

**Placeholder scan:** The plan contains concrete paths, commands, file contents, and expected outputs. The only replacement token is `<jobid>`, which is produced by `sbatch` and must be substituted with the actual Slurm job id at execution time.

**Type consistency:** The DeepMD input field is consistently `training.numb_steps`; epoch aliases are explicitly removed. The environment name is consistently `dpeva-dpa4` after Task 2. The run directory name is consistently `dpa4_air_matpes_zbl_train_data_iter11_50k`.

---

## Status (2026-09-03 回写)

- **已完成**：环境重命名与 Air/Mini/Neo × `train_data_iter11` 50k 微调全部执行（2026-07-10/11），产物 `model/ft2dp-dpa4/*-50k.pt`。
- **2026-09-03 补记**：本计划全部远端产物因集群迁移滞留旧只读备份，已于 9/3 完成抢救回 `/org/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/`（sha256 全量校验通过，见工作区根 `STORAGE_RESCUE_20260903.md`）。
- **环境事实修正**：当前 `dpeva-dpa4` 为 dev67 构建，而后续 C1/C2 训练实际使用 dev115(`f73de32e`)；GA 3.2.0 环境 `dpeva-dpa4-320` 构建中。所有旧绝对路径 `/home/pku-jianghong/...` 已作废，以 `/org/...` 为准。
- **当前状态来源**：本计划的原始 checkbox 仅记录当时设计；后续状态、验收和未决项以工作区 `../../../../docs/tasks/ft2dp-v2.2.md` 为准。
