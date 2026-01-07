#!/bin/bash
# Clean DCM Pipeline Outputs
# ==========================
#
# Removes generated results and temporary files.
# Does NOT remove raw data or source code.
#
# Usage:
#   ./scripts/clean.sh           # Clean results only
#   ./scripts/clean.sh --all     # Clean results + Biogeme artifacts

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

CLEAN_ALL=false
for arg in "$@"; do
    case $arg in
        --all)
            CLEAN_ALL=true
            ;;
    esac
done

echo "Cleaning DCM pipeline outputs..."

# Remove result directories
rm -rf results/mnl/latest
rm -rf results/mxl/latest
rm -rf results/hcm/latest
rm -rf results/final_comparison

# Remove legacy symlinks if they exist
rm -f mnl_small_results mxl_small_results hcm_improved_results 2>/dev/null || true

if [ "$CLEAN_ALL" = true ]; then
    echo "Cleaning Biogeme artifacts..."
    # Remove Biogeme generated files
    rm -f *.pickle *.html 2>/dev/null || true
    rm -f biogeme_model_default_name* 2>/dev/null || true
    rm -rf __pycache__ 2>/dev/null || true
    find . -name "*.pyc" -delete 2>/dev/null || true
fi

echo "Done."
