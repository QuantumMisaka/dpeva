#!/bin/bash
set -e

echo "=========================================="
echo "DP-EVA Documentation Verification Script"
echo "=========================================="

# 1. Check Dependencies
if ! command -v sphinx-build &> /dev/null; then
    echo "Sphinx not found. Installing doc dependencies..."
    pip install -e .[docs]
fi

# 2. Build Docs
echo "Building HTML documentation..."
cd docs
make clean
# Use -W to turn warnings into errors
if make html SPHINXOPTS="-W --keep-going"; then
    echo "✅ Documentation built successfully."
else
    echo "❌ Documentation build FAILED."
    exit 1
fi

# 3. Verify Artifacts
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
