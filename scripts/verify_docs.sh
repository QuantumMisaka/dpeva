#!/bin/bash
set -e

echo "=========================================="
echo "DP-EVA Documentation Verification Script"
echo "=========================================="

if ! command -v sphinx-build &> /dev/null; then
    echo "Sphinx not found. Install documentation dependencies first: pip install -e .[docs]"
    exit 1
fi

echo "Building HTML documentation..."
cd docs
make clean
if make html SPHINXOPTS="-W --keep-going"; then
    echo "✅ Documentation built successfully."
else
    echo "❌ Documentation build FAILED."
    exit 1
fi

echo "Verifying output artifacts..."
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
