# Issue: Provide a `scripts/run_all.sh` for end-to-end pipeline execution

## Summary
Users currently need to copy/paste multiple bash commands to run the full pipeline, which is error-prone (especially with zsh comment handling). A single script would improve usability.

## Problem Statement
- Multiple scripts need to be run in sequence
- Output directories need to match what `final_comparison.py` expects
- Copy/pasting commands with `#` comments fails in zsh
- No single command to reproduce full results

## Proposed Solution
Create `scripts/run_all.sh` that:
1. Runs data generation (optional, if data doesn't exist)
2. Runs all model estimations
3. Runs final comparison
4. Reports success/failure

## Specification

```bash
#!/bin/bash
# scripts/run_all.sh - Run complete DCM pipeline

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "=== DCM Pipeline ==="
echo "Working directory: $PROJECT_ROOT"

# Check for virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "Activated virtual environment"
fi

# Step 1: Generate data (if needed)
if [ ! -f "data/test_small_sample.csv" ]; then
    echo ""
    echo "=== Step 1: Generating synthetic data ==="
    python src/simulation/simulate_full_data.py
else
    echo ""
    echo "=== Step 1: Data exists, skipping generation ==="
fi

# Step 2: Run MNL models
echo ""
echo "=== Step 2: Running MNL models ==="
python src/models/mnl_model_comparison.py --output results/mnl/latest

# Step 3: Run MXL models
echo ""
echo "=== Step 3: Running MXL models ==="
python src/models/mxl_models.py --output results/mxl/latest

# Step 4: Run HCM models
echo ""
echo "=== Step 4: Running HCM models ==="
python src/models/hcm_split_latents.py --output results/hcm/latest

# Step 5: Create symlinks for final_comparison.py
echo ""
echo "=== Step 5: Setting up result paths ==="
ln -sfn results/mnl/latest mnl_small_results
ln -sfn results/mxl/latest mxl_small_results
ln -sfn results/hcm/latest hcm_improved_results

# Step 6: Run final comparison
echo ""
echo "=== Step 6: Running final comparison ==="
python src/analysis/final_comparison.py

echo ""
echo "=== Pipeline complete ==="
echo "Results available in:"
echo "  - results/mnl/latest/"
echo "  - results/mxl/latest/"
echo "  - results/hcm/latest/"
echo "  - results/final_comparison/"
```

## Additional Scripts

### `scripts/clean.sh`
```bash
#!/bin/bash
# Clean generated outputs
rm -rf results/*/latest
rm -rf final_comparison/
rm -f mnl_small_results mxl_small_results hcm_improved_results
rm -f *.pickle *.html  # Biogeme artifacts
echo "Cleaned outputs"
```

### `scripts/validate_data.sh`
```bash
#!/bin/bash
# Run data QA checks only
python -c "
from src.utils.data_qa import run_all_checks
import pandas as pd
df = pd.read_csv('data/test_small_sample.csv')
run_all_checks(df, {...})
"
```

## Acceptance Criteria
- [ ] `scripts/run_all.sh` runs full pipeline with single command
- [ ] Script is executable (`chmod +x`)
- [ ] Script handles missing data gracefully
- [ ] Script creates necessary symlinks
- [ ] Clear progress output at each step
- [ ] Exits with error code if any step fails

## Affected Files
- `scripts/run_all.sh` (new)
- `scripts/clean.sh` (new, optional)

## Labels
`enhancement`, `usability`, `documentation`
