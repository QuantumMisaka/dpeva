#!/bin/bash
#SBATCH --job-name=dpeva-llpr-dpose
#SBATCH --partition=4V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=rush-1o2gpu
#SBATCH --output=logs/dpeva-llpr-dpose-%j.out
#SBATCH --error=logs/dpeva-llpr-dpose-%j.err

set -euo pipefail

cd /home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/dpeva

source /opt/devtools/anaconda3/etc/profile.d/conda.sh
conda activate dpeva-dpa4-test

export OMP_NUM_THREADS=1
export DP_INTRA_OP_PARALLELISM_THREADS=1
export DP_INTER_OP_PARALLELISM_THREADS=1
export DP_INFER_BATCH_SIZE=1024
export DPEVA_RUN_DEEPMD_DPOSE_REAL=1
export DPEVA_DEEPMD_HEAD="${DPEVA_DEEPMD_HEAD:-Omat24}"
export DPEVA_DPA3_MODEL_PATH="${DPEVA_DPA3_MODEL_PATH:-tests/integration/data/DPA-3.1-3M.pt}"
export DPEVA_DPA4_MODEL_PATH="${DPEVA_DPA4_MODEL_PATH:-tests/integration/data/DPA4-Mini-OMat24.pt}"

echo "[DPEVA-LLPR-DPOSE] host=$(hostname)"
echo "[DPEVA-LLPR-DPOSE] date=$(date '+%F %T %Z')"
echo "[DPEVA-LLPR-DPOSE] conda_env=${CONDA_DEFAULT_ENV}"
echo "[DPEVA-LLPR-DPOSE] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[DPEVA-LLPR-DPOSE] dpa3_model=${DPEVA_DPA3_MODEL_PATH}"
echo "[DPEVA-LLPR-DPOSE] dpa4_model=${DPEVA_DPA4_MODEL_PATH}"
nvidia-smi

python - <<'PY'
import importlib

for name in ["numpy", "scipy", "pytest", "torch", "deepmd", "dpdata", "ase"]:
    module = importlib.import_module(name)
    print(f"[DPEVA-LLPR-DPOSE] dependency {name}={getattr(module, '__version__', 'unknown')}")
PY

pytest \
  tests/unit/uncertain/test_llpr.py \
  tests/unit/uncertain/test_llpr_manager.py \
  tests/unit/uncertain/test_dpose_adapter.py \
  tests/unit/test_llpr_config.py \
  tests/unit/workflows/test_collect_workflow_routing.py \
  tests/integration/test_real_deepmd_llpr_dpose.py \
  -q -s
