#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/pku-jianghong/liuzhaoqing/fp11-sai1344
FP11_DIR="${ROOT}/fp11"
DPEVA_DIR="${ROOT}/dpeva"
INTERVAL="${1:-300}"
CONDA_BIN=/opt/devtools/anaconda3/bin/conda

cd "${FP11_DIR}"
mkdir -p logs

set +u
source /etc/profile
set -u
export PYTHONPATH="${DPEVA_DIR}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

date '+%F %T %Z'
"${CONDA_BIN}" run --no-capture-output -n dpeva-dpa4 python \
  "${DPEVA_DIR}/scripts/fp11_1344_recover_after_false_finish.py" \
  config_gpu_1344.json \
  --wait-job-id 581295 \
  --next-attempt 2 \
  --interval "${INTERVAL}"

"${CONDA_BIN}" run --no-capture-output -n dpeva-dpa4 python "${DPEVA_DIR}/scripts/fp11_1344_backend_report.py"
date '+%F %T %Z'
