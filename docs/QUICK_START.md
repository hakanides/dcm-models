# Quick Start Guide

Get up and running with DCM estimation in 5 minutes.

---

## Prerequisites

```bash
pip install biogeme pandas numpy scipy matplotlib
```

---

## Step 1: Navigate to Project

```bash
cd "/path/to/DCM draft3"
```

---

## Step 2: Run All Models (Recommended)

This runs MNL, MXL, and HCM models with automatic comparison:

```bash
python scripts/run_all_models.py
```

**Output:**
- `results/all_models/model_comparison.csv` - Model fit statistics
- `results/all_models/parameter_comparison.csv` - All parameter estimates
- `output/latex/` - LaTeX tables for publication

---

## Step 3: View Results

```bash
# View model comparison
cat results/all_models/model_comparison.csv

# Or in Python:
import pandas as pd
df = pd.read_csv('results/all_models/model_comparison.csv')
print(df.sort_values('AIC'))
```

---

## Understanding Output

### Model Comparison Table

```
Model                    LL       K     AIC      BIC     rho2
M1: Blind Patriotism  -798.16    4  1604.33  1630.40   0.855  <- Best!
M6: Both Secularism   -812.59    5  1635.19  1667.77   0.852
M2: Constructive Pat  -818.57    4  1645.14  1671.20   0.851
...
```

- **LL**: Log-likelihood (higher = better fit)
- **K**: Number of parameters
- **AIC**: Akaike Information Criterion (lower = better)
- **BIC**: Bayesian Information Criterion (lower = better)
- **rho2**: McFadden's R² (higher = better, max 1.0)

### Parameter Estimates

```
Parameter          Estimate   t-stat   Sig
ASC_paid            -0.05     -0.42
B_FEE               -0.78    -10.61    ***
B_DUR               -0.11     -6.55    ***
B_FEE_PatBlind      -0.22     -7.47    ***  <- Significant!
```

- **t-stat > 1.96**: Significant at 5% level (**)
- **t-stat > 2.58**: Significant at 1% level (***)

---

## Common Commands

### Run Different Model Types

```bash
# MNL Models
python src/models/mnl_model_comparison.py --data data/test_validation.csv

# MXL Models (takes longer due to simulation)
python src/models/mxl_models.py --data data/test_validation.csv --draws 2000

# HCM with split latent variables
python src/models/hcm_split_latents.py --data data/test_validation.csv
```

### Generate New Data

```bash
# Basic simulation
python src/simulation/simulate_full_data.py \
    --config config/model_config.json \
    --out data/simulated/my_simulation.csv

# Advanced simulation with random coefficients
python src/simulation/simulate_full_data.py \
    --config config/model_config_advanced.json \
    --out data/simulated/advanced_simulation.csv \
    --keep_latent
```

### Run Full Pipeline

```bash
# Run everything with shell script
./scripts/run_all.sh --data=data/test_validation.csv

# Skip MXL (faster)
./scripts/run_all.sh --skip-mxl
```

### Validate Estimation

```bash
python scripts/validate_estimation.py
```

---

## Troubleshooting

### "Did not converge"

Try:
1. Check `biogeme.toml` uses `simple_bounds_BFGS`
2. Increase `max_iterations` to 10000
3. Simplify model (fewer LV interactions)

### Negative rho²

The model is worse than random choice. Check:
1. Data quality
2. Variable coding
3. Identification issues

---

## Project Structure

```
config/              # Configuration files
scripts/             # Entry point scripts (run_all_models.py)
data/                # Data files (test_validation.csv)
src/                 # Source code
output/              # Generated outputs (LaTeX, logs)
results/             # Estimation results
tests/               # Test suite
docs/                # Documentation
```

---

## Next Steps

1. Read `README.md` for full documentation
2. Read `docs/MODEL_SPECIFICATIONS.md` for model details
3. Modify configurations in `config/`
4. Add your own data in `data/`
