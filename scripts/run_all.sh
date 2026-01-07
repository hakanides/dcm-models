#!/bin/bash
# DCM Pipeline - Run All Steps
# ============================
#
# This script runs the complete DCM estimation pipeline:
# 1. (Optional) Regenerate scenarios with variation
# 2. (Optional) Generate synthetic data
# 3. Run data QA checks
# 4. Estimate MNL models
# 5. Estimate MXL models
# 6. Estimate HCM models
# 7. Run final comparison
#
# Usage:
#   ./scripts/run_all.sh                    # Run with existing data
#   ./scripts/run_all.sh --regenerate       # Regenerate data first
#   ./scripts/run_all.sh --skip-mxl         # Skip MXL (slow)

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Parse arguments
REGENERATE=false
SKIP_MXL=false
DATA_FILE="data/test_small_sample.csv"

for arg in "$@"; do
    case $arg in
        --regenerate)
            REGENERATE=true
            shift
            ;;
        --skip-mxl)
            SKIP_MXL=true
            shift
            ;;
        --data=*)
            DATA_FILE="${arg#*=}"
            shift
            ;;
    esac
done

echo "============================================================"
echo "DCM ESTIMATION PIPELINE"
echo "============================================================"
echo "Project root: $PROJECT_ROOT"
echo "Data file: $DATA_FILE"
echo ""

# Activate virtual environment if present
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "Activated virtual environment"
fi

# Step 0: Regenerate scenarios (if requested)
if [ "$REGENERATE" = true ]; then
    echo ""
    echo "============================================================"
    echo "Step 0: Regenerating scenarios with variation"
    echo "============================================================"

    # Check if raw scenarios exist
    if [ -f "data/scenarios_raw.csv" ]; then
        python src/simulation/prepare_scenarios.py \
            --input data/scenarios_raw.csv \
            --output data/scenarios_prepared.csv \
            --seed 42
    else
        echo "WARNING: data/scenarios_raw.csv not found, skipping scenario prep"
    fi

    echo ""
    echo "============================================================"
    echo "Step 0b: Generating synthetic data"
    echo "============================================================"

    python src/simulation/simulate_full_data.py \
        --config config/model_config.json \
        --out "$DATA_FILE" \
        --keep_latent
fi

# Step 1: Data QA
echo ""
echo "============================================================"
echo "Step 1: Data Quality Checks"
echo "============================================================"

if [ -f "$DATA_FILE" ]; then
    python src/utils/data_qa.py --data "$DATA_FILE" --max-share 0.85
    QA_EXIT=$?
    if [ $QA_EXIT -ne 0 ]; then
        echo ""
        echo "WARNING: Data QA found issues. Continuing anyway..."
        echo "Consider running with --regenerate to fix data issues."
    fi
else
    echo "ERROR: Data file not found: $DATA_FILE"
    exit 1
fi

# Step 2: MNL Models
echo ""
echo "============================================================"
echo "Step 2: Estimating MNL Models"
echo "============================================================"

mkdir -p results/mnl/latest
python src/models/mnl_model_comparison.py \
    --data "$DATA_FILE" \
    --output results/mnl/latest

# Step 3: MXL Models (optional)
if [ "$SKIP_MXL" = false ]; then
    echo ""
    echo "============================================================"
    echo "Step 3: Estimating MXL Models"
    echo "============================================================"

    mkdir -p results/mxl/latest
    python src/models/mxl_models.py \
        --data "$DATA_FILE" \
        --output results/mxl/latest \
        --draws 2000
else
    echo ""
    echo "============================================================"
    echo "Step 3: Skipping MXL Models (--skip-mxl flag)"
    echo "============================================================"
fi

# Step 4: HCM Models
echo ""
echo "============================================================"
echo "Step 4: Estimating HCM Models"
echo "============================================================"

mkdir -p results/hcm/latest
if [ -f "src/models/hcm_split_latents.py" ]; then
    python src/models/hcm_split_latents.py \
        --data "$DATA_FILE" \
        --output results/hcm/latest
else
    echo "WARNING: HCM script not found, skipping"
fi

# Step 5: Final Comparison
echo ""
echo "============================================================"
echo "Step 5: Final Model Comparison"
echo "============================================================"

mkdir -p results/final_comparison
python src/analysis/final_comparison.py \
    --mnl results/mnl/latest/model_comparison.csv \
    --mxl results/mxl/latest/mxl_comparison.csv \
    --hcm results/hcm/latest/model_comparison.csv \
    --output results/final_comparison

echo ""
echo "============================================================"
echo "PIPELINE COMPLETE"
echo "============================================================"
echo ""
echo "Results saved to:"
echo "  - results/mnl/latest/"
echo "  - results/mxl/latest/"
echo "  - results/hcm/latest/"
echo "  - results/final_comparison/"
echo ""
echo "Key outputs:"
echo "  - results/final_comparison/all_models_comparison.csv"
echo "  - results/final_comparison/model_comparison_plot.png"
