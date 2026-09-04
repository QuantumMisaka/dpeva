#!/usr/bin/env bash
# One bounded, evidence-producing SAI qualification job.  This script does
# not promote a capability; Plan D consumes its recorded evidence separately.
#SBATCH --job-name=dpeva-deepmd32
#SBATCH --partition=4V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=improper-gpu
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -Eeuo pipefail

if [[ "$#" -ne 2 ]]; then
    echo "usage: $0 QUALIFICATION_INPUT IMMUTABLE_JOB_DIR" >&2
    exit 2
fi
readonly DPEVA_QUALIFICATION_INPUT="$1"
readonly DPEVA_QUALIFICATION_DIR="$2"
export DPEVA_QUALIFICATION_INPUT DPEVA_QUALIFICATION_DIR
mkdir -p "$DPEVA_QUALIFICATION_DIR"

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd -P)"
cd "$REPO_ROOT"

job_rc=0
run_case() {
    if ! python "$REPO_ROOT/scripts/validation/run_recorded_command.py" \
        --config "$DPEVA_QUALIFICATION_INPUT" \
        --job-dir "$DPEVA_QUALIFICATION_DIR" \
        --case "$1"; then
        job_rc=1
    fi
}

on_exit() {
    original_rc=$?
    set +e
    python "$REPO_ROOT/scripts/validation/collect_deepmd_32_qualification.py" \
        --job-dir "$DPEVA_QUALIFICATION_DIR" \
        --job-id "${SLURM_JOB_ID:-}" \
        --finalize \
        >/dev/null
    collector_rc=$?
    # An EXIT trap must never turn a failed command into a successful job, nor
    # hide a collector failure after an otherwise successful command sequence.
    if (( original_rc != 0 )); then
        return "$original_rc"
    fi
    return "$collector_rc"
}
trap on_exit EXIT

source "$REPO_ROOT/scripts/env/dpeva-dpa4.env"

run_case preflight
if (( job_rc == 0 )); then
    for case_name in \
        pip-freeze deepmd-version torch-cuda gpu \
        pt-test pt-test-ema pt-eval-desc pt-eval-desc-ema \
        pt-embed pt-embed-ema dpa4c-periodic-eval-desc; do
        run_case "$case_name"
    done
fi

exit "$job_rc"
