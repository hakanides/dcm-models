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
cd "/Users/hakanmulayim/Desktop/DCM draft3"
```

---

## Step 2: Run HCM Analysis (Recommended)

This runs all 15 HCM model specifications with split latent variables:

```bash
python src/models/hcm_split_latents.py \
    --data data/test_small_sample.csv \
    --output results/hcm/my_analysis
```

**Output:**
- `results/hcm/my_analysis/model_comparison.csv` - Model fit statistics
- `results/hcm/my_analysis/parameters.csv` - All parameter estimates

---

## Step 3: View Results

```bash
# View model comparison
cat results/hcm/my_analysis/model_comparison.csv

# Or in Python:
import pandas as pd
df = pd.read_csv('results/hcm/my_analysis/model_comparison.csv')
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
python src/models/mnl_model_comparison.py --data data/test_small_sample.csv

# MXL Models (takes longer)
python src/models/mxl_models.py --data data/test_small_sample.csv

# Improved HCM (fewer models, faster)
python src/models/hcm_model_improved.py --data data/test_small_sample.csv
```

### Generate New Data

```bash
python src/simulation/dcm_simulator_advanced.py \
    --n_respondents 100 \
    --n_tasks 50 \
    --output data/my_simulation.csv
```

### Compare All Models

```bash
python src/analysis/final_comparison.py
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

## Next Steps

1. Read `README.md` for full documentation
2. Read `docs/MODEL_SPECIFICATIONS.md` for model details
3. Modify model specifications in `src/models/`
4. Add your own data in `data/`
