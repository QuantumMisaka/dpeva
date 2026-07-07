#!/usr/bin/env bash
set -Eeuo pipefail

FP11_ROOT="${FP11_ROOT:-/home/pku-jianghong/liuzhaoqing/fp11-sai1344/fp11}"
REPO_ROOT="${REPO_ROOT:-/home/pku-jianghong/liuzhaoqing/fp11-sai1344/dpeva}"
DPEVA_BIN="${DPEVA_BIN:-/home/pku-jianghong/liuzhaoqing/.conda/envs/dpeva-dpa4/bin/dpeva}"
PYTHON_BIN="${PYTHON_BIN:-/home/pku-jianghong/liuzhaoqing/.conda/envs/dpeva-dpa4/bin/python}"
CONFIG_NAME="${CONFIG_NAME:-config_gpu_1344.json}"
EXECUTE_SESSION="${EXECUTE_SESSION:-fp11_1344_execute}"
FINALIZE_SESSION="${FINALIZE_SESSION:-fp11_1344_finalize}"
POLL_SECONDS="${POLL_SECONDS:-300}"

LOG_DIR="${LOG_DIR:-$FP11_ROOT/logs}"
FINALIZE_LOG="${FINALIZE_LOG:-$LOG_DIR/fp11_finalize_1344.log}"
EXTRACT_LOG="${EXTRACT_LOG:-$LOG_DIR/fp11_extract_1344.log}"
POSTPROCESS_LOG="${POSTPROCESS_LOG:-$LOG_DIR/fp11_postprocess_1344.log}"
EXECUTE_WRAPPER_LOG="${EXECUTE_WRAPPER_LOG:-$LOG_DIR/fp11_execute_1344.wrapper.log}"
REPORT_PATH="${REPORT_PATH:-$FP11_ROOT/fp11_1344_backend_report.json}"

usage() {
  cat <<'EOF'
Usage: fp11_1344_finalize_after_execute.sh --detach|--run|--status

Environment overrides:
  FP11_ROOT, REPO_ROOT, DPEVA_BIN, PYTHON_BIN, CONFIG_NAME
  EXECUTE_SESSION, FINALIZE_SESSION, POLL_SECONDS
  LOG_DIR, FINALIZE_LOG, EXTRACT_LOG, POSTPROCESS_LOG, EXECUTE_WRAPPER_LOG, REPORT_PATH
EOF
}

quote() {
  printf "%q" "$1"
}

log() {
  mkdir -p "$LOG_DIR"
  printf "%s %s\n" "$(date "+%F %T %Z")" "$*" >> "$FINALIZE_LOG"
}

wait_for_execute_session() {
  while tmux has-session -t "$EXECUTE_SESSION" 2>/dev/null; do
    log "waiting for tmux session $EXECUTE_SESSION"
    sleep "$POLL_SECONDS"
  done
  log "tmux session $EXECUTE_SESSION is no longer active"
}

require_clean_execute() {
  if [ ! -f "$EXECUTE_WRAPPER_LOG" ]; then
    log "execute wrapper log missing: $EXECUTE_WRAPPER_LOG"
    return 10
  fi
  if ! tail -n 20 "$EXECUTE_WRAPPER_LOG" | grep -q '^exit_code=0$'; then
    log "execute wrapper did not finish with exit_code=0"
    return 10
  fi
}

run_label_stage() {
  local stage="$1"
  local stage_log="$2"
  local rc

  log "starting dpeva label --stage $stage"
  if (cd "$FP11_ROOT" && "$DPEVA_BIN" --no-banner label "$CONFIG_NAME" --stage "$stage" >> "$stage_log" 2>&1); then
    rc=0
  else
    rc=$?
  fi
  log "stage $stage exit_code=$rc log=$stage_log"
  return "$rc"
}

write_backend_report() {
  local rc

  log "writing backend report to $REPORT_PATH"
  if "$PYTHON_BIN" "$REPO_ROOT/scripts/fp11_1344_backend_report.py" "$FP11_ROOT" > "$REPORT_PATH" 2>> "$FINALIZE_LOG"; then
    rc=0
  else
    rc=$?
  fi
  log "backend report exit_code=$rc"
  return "$rc"
}

run_finalize() {
  mkdir -p "$LOG_DIR"
  log "finalize watcher started"
  wait_for_execute_session
  require_clean_execute || exit $?
  run_label_stage extract "$EXTRACT_LOG"
  run_label_stage postprocess "$POSTPROCESS_LOG"
  write_backend_report
  log "finalize watcher completed"
}

detach() {
  mkdir -p "$LOG_DIR"
  if tmux has-session -t "$FINALIZE_SESSION" 2>/dev/null; then
    printf "tmux session already active: %s\n" "$FINALIZE_SESSION"
    exit 0
  fi

  local command
  command="env"
  command+=" FP11_ROOT=$(quote "$FP11_ROOT")"
  command+=" REPO_ROOT=$(quote "$REPO_ROOT")"
  command+=" DPEVA_BIN=$(quote "$DPEVA_BIN")"
  command+=" PYTHON_BIN=$(quote "$PYTHON_BIN")"
  command+=" CONFIG_NAME=$(quote "$CONFIG_NAME")"
  command+=" EXECUTE_SESSION=$(quote "$EXECUTE_SESSION")"
  command+=" FINALIZE_SESSION=$(quote "$FINALIZE_SESSION")"
  command+=" POLL_SECONDS=$(quote "$POLL_SECONDS")"
  command+=" LOG_DIR=$(quote "$LOG_DIR")"
  command+=" FINALIZE_LOG=$(quote "$FINALIZE_LOG")"
  command+=" EXTRACT_LOG=$(quote "$EXTRACT_LOG")"
  command+=" POSTPROCESS_LOG=$(quote "$POSTPROCESS_LOG")"
  command+=" EXECUTE_WRAPPER_LOG=$(quote "$EXECUTE_WRAPPER_LOG")"
  command+=" REPORT_PATH=$(quote "$REPORT_PATH")"
  command+=" bash $(quote "$0") --run"

  tmux new-session -d -s "$FINALIZE_SESSION" "$command"
  printf "started tmux session: %s\n" "$FINALIZE_SESSION"
}

status() {
  tmux ls 2>/dev/null || true
  if [ -f "$FINALIZE_LOG" ]; then
    tail -n 20 "$FINALIZE_LOG"
  fi
}

case "${1:-}" in
  --detach)
    detach
    ;;
  --run)
    run_finalize
    ;;
  --status)
    status
    ;;
  -h|--help|"")
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
