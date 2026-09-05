#!/bin/bash
set -Eeuo pipefail

echo "=========================================="
echo "DP-EVA Documentation Verification Script"
echo "=========================================="

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v sphinx-build &> /dev/null; then
    echo "Sphinx not found. Install documentation dependencies first: pip install -e .[docs]"
    exit 1
fi

cd "$PROJECT_ROOT"

echo "Running documentation governance checks..."
python scripts/run_gate.py docs_audit
python scripts/run_gate.py docs_freshness

echo "Building HTML documentation..."
cd docs
make clean
cd "$PROJECT_ROOT"
if python scripts/run_gate.py docs_build; then
    echo "✅ Documentation built successfully."
else
    echo "❌ Documentation build FAILED."
    exit 1
fi

echo "Verifying output artifacts..."
cd docs
REQUIRED_FILES=(
    "build/html/index.html"
    "build/html/api/config.html"
    "build/html/guides/quickstart.html"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  - Found: $file"
    else
        echo "  - Missing: $file"
        echo "❌ Artifact verification FAILED."
        exit 1
    fi
done

echo "=========================================="
echo "🎉 All Checks Passed! You can now commit."
echo "Preview: $(pwd)/build/html/index.html"
echo "=========================================="
