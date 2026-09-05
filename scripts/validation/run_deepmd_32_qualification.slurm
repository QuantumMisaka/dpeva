#!/bin/bash
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

if [[ "$#" -ne 3 ]]; then
    echo "usage: $0 QUALIFICATION_INPUT IMMUTABLE_JOB_DIR REPO_ROOT" >&2
    exit 2
fi
readonly DPEVA_QUALIFICATION_INPUT="$1"
readonly DPEVA_QUALIFICATION_DIR="$2"
readonly REPO_ROOT="$3"
export DPEVA_QUALIFICATION_INPUT DPEVA_QUALIFICATION_DIR
mkdir -p "$DPEVA_QUALIFICATION_DIR"
DPEVA_QUALIFICATION_SCOPE="all"
export DPEVA_QUALIFICATION_SCOPE

if [[ ! -d "$REPO_ROOT" ]]; then
    echo "repository root is not a directory: $REPO_ROOT" >&2
    exit 2
fi
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

job_rc=0
run_case() {
    if ! python "$REPO_ROOT/scripts/validation/run_recorded_command.py" \
        --config "$DPEVA_QUALIFICATION_INPUT" \
        --job-dir "$DPEVA_QUALIFICATION_DIR" \
        --scope "$DPEVA_QUALIFICATION_SCOPE" \
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
        --scope "$DPEVA_QUALIFICATION_SCOPE" \
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

# Qualification is intentionally isolated from the ordinary dpeva-dpa4
# development environment.  --export=NIL does not carry this setting in.
export DPEVA_DPA4_ENV_NAME="dpeva-dpa4-320"
source "$REPO_ROOT/scripts/env/dpeva-dpa4.env"

DPEVA_QUALIFICATION_SCOPE="$(
    python -c 'import json, sys; value=json.load(open(sys.argv[1], encoding="utf-8")).get("scope", "all"); assert value in {"dpa4", "all"}; print(value)' \
        "$DPEVA_QUALIFICATION_INPUT"
)"
readonly DPEVA_QUALIFICATION_SCOPE

run_case preflight
if (( job_rc == 0 )); then
    while IFS= read -r case_name; do
        run_case "$case_name"
    done < <(
        python -c 'import json, sys; print("\n".join(json.load(open(sys.argv[1], encoding="utf-8"))["required_cases"]))' \
            "$DPEVA_QUALIFICATION_INPUT"
    )
fi

exit "$job_rc"
