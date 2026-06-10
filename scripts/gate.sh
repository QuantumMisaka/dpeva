#!/bin/bash
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${GREEN}=== DP-EVA Quality Gate ===${NC}"

echo -e "\n${GREEN}[1/3] Running Ruff...${NC}"
if ruff check src tests scripts; then
    echo -e "${GREEN}Ruff Passed.${NC}"
else
    echo -e "${RED}Ruff Failed.${NC}"
    exit 1
fi

echo -e "\n${GREEN}[2/3] Running Unit Tests...${NC}"
if pytest tests/unit; then
    echo -e "${GREEN}Unit Tests Passed.${NC}"
else
    echo -e "${RED}Unit Tests Failed.${NC}"
    exit 1
fi

AUDIT_ARGS=""
if [[ "$1" == "--strict" ]]; then
    AUDIT_ARGS="--strict"
    echo -e "\n${GREEN}[3/3] Running Code Audit (Strict Mode)...${NC}"
else
    echo -e "\n${GREEN}[3/3] Running Code Audit (Standard Mode)...${NC}"
fi

if python scripts/audit.py src/dpeva $AUDIT_ARGS; then
    echo -e "${GREEN}Code Audit Passed.${NC}"
else
    echo -e "${RED}Code Audit Failed.${NC}"
    exit 1
fi

echo -e "\n${GREEN}=== All Checks Passed! ===${NC}"
exit 0
