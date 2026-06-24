#!/bin/bash
#SBATCH --job-name=dpeva_embed_cmp_px
#SBATCH --partition=4V100PX
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=rush-1o2gpu
#SBATCH --output=slurm-px-%j.out
#SBATCH --error=slurm-px-%j.err

set -euo pipefail

REPO="/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/dpeva"
cd "${REPO}"

source "${REPO}/scripts/env/dpeva-dpa4-embed-test.env"

python "${REPO}/scripts/validation/compare_eval_desc_embed.py" \
  --output-dir "${REPO}/practices/dpeva-dpa4-test/dpa4-dpeva-test/embed_eval_desc_comparison/results_v4" \
  --timeout-s 900 \
  --stress
