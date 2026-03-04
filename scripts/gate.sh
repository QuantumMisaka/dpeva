#!/bin/bash
set -e

# DP-EVA Quality Gate Script
# ---------------------------
# This script runs the minimal set of checks required for code submission.
# It includes:
# 1. Unit Tests (pytest)
# 2. Code Audit (tools/audit.py) - Checks for hardcoded paths, magic numbers, etc.

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== DP-EVA Quality Gate ===${NC}"

# 1. Run Unit Tests
echo -e "\n${GREEN}[1/2] Running Unit Tests...${NC}"
if pytest tests/unit; then
    echo -e "${GREEN}Unit Tests Passed.${NC}"
else
    echo -e "${RED}Unit Tests Failed.${NC}"
    exit 1
fi

# 2. Run Code Audit
# By default, we run in non-strict mode (only fail on ERRORs).
# Use --strict to fail on WARNINGs too.
AUDIT_ARGS=""
if [[ "$1" == "--strict" ]]; then
    AUDIT_ARGS="--strict"
    echo -e "\n${GREEN}[2/2] Running Code Audit (Strict Mode)...${NC}"
else
    echo -e "\n${GREEN}[2/2] Running Code Audit (Standard Mode)...${NC}"
fi

if python scripts/audit.py src/dpeva $AUDIT_ARGS; then
    echo -e "${GREEN}Code Audit Passed.${NC}"
else
    echo -e "${RED}Code Audit Failed.${NC}"
    exit 1
fi

echo -e "\n${GREEN}=== All Checks Passed! ===${NC}"
exit 0
