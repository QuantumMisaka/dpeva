#!/bin/bash
set -Eeuo pipefail
python scripts/run_gate.py local "$@"
