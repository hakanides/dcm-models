# Quick Start Guide

**Authors: Hakan Mülayim, Giray Girengir, Ataol Azeritürk**

Get up and running with DCM estimation in 5 minutes.

---

## Which Folder Should I Use?

| I want to... | Use | Time |
|--------------|-----|------|
| Learn DCM step-by-step | `models/` | 30 sec - 10 min per model |
| Quick parameter recovery test | `models/` | 30 sec - 10 min |
| Full research pipeline | `full_model/` | 30-60 min total |
| Publication-ready outputs | `full_model/` | 30-60 min total |

**Recommendation:** Start with `models/` to understand the methodology, then graduate to `full_model/` for comprehensive research.

---

## Prerequisites

```bash
pip install biogeme pandas numpy scipy scikit-learn matplotlib
```

---

## Option A: Start with Isolated Models (Recommended)

### Step 1: Run Your First Model

```bash
cd models/mnl_basic
python run.py  # ~30 seconds
```

This will:
1. Simulate data with known true parameters
2. Estimate the model
3. Compare estimates to true values
4. Generate LaTeX tables

### Step 2: Check Results

```bash
cat results/parameter_comparison.csv
ls output/latex/
```

### Step 3: Progress Through Model Hierarchy

| Step | Command | Time | What You Learn |
|------|---------|------|----------------|
| 1 | `cd models/mnl_basic && python run.py` | ~30 sec | Baseline MNL |
| 2 | `cd models/mnl_demographics && python run.py` | ~1 min | Observable heterogeneity |
| 3 | `cd models/mxl_basic && python run.py` | ~2 min | Random coefficients |
| 4 | `cd models/hcm_basic && python run.py` | ~3 min | Latent variables (two-stage) |
| 5 | `cd models/hcm_full && python run.py` | ~5 min | Attenuation bias problem |
| 6 | `cd models/iclv && python run.py` | ~10 min | Simultaneous estimation (solution) |

**Key Insight:** Compare HCM vs ICLV results. ICLV reduces latent variable bias from 50-66% (HCM two-stage) to ~2.4%.

---

## Option B: Full Research Pipeline

When you're ready for comprehensive research with 32+ model specifications:

### Step 1: Navigate to Full Model

```bash
cd full_model
```

### Step 2: Run Complete Pipeline

```bash
python scripts/run_all_models.py  # ~30-60 minutes
```

**What happens:**
1. Loads data from `data/simulated/full_scale_test.csv`
2. Estimates basic MNL models (baseline + demographics)
3. Estimates basic MXL models (random parameters)
4. Estimates basic HCM models (latent variables, two-stage)
5. Estimates ICLV models (simultaneous, unbiased)
6. Estimates Extended Models:
   - 8 MNL specifications (log, quadratic, piecewise, etc.)
   - 8 MXL specifications (normal, lognormal, uniform distributions)
   - 8 HCM specifications (domain separation, interactions)
7. Compares all 32+ models by AIC/BIC
8. Generates LaTeX tables for publication

**Output:**
- `results/all_models/model_comparison.csv` - Model fit statistics
- `results/all_models/parameter_comparison.csv` - All parameter estimates
- `output/latex/MNL/`, `output/latex/MXL/`, `output/latex/HCM/`, `output/latex/ICLV/` - LaTeX tables

### Step 3: View Results

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
Model                    LL       K     AIC      BIC     rho2   Conv
MNL: Basic            -1250.16   3  2506.33  2520.40   0.652   Yes
MNL: Demographics     -1120.59   8  2257.19  2297.77   0.752   Yes
MNL-M2: Log Fee       -1115.23   3  2236.46  2250.53   0.757   Yes
MXL: Random Fee        -980.57  10  1981.14  2041.20   0.821   Yes
MXL-M4: Lognormal     -975.20   10  1970.40  2030.46   0.825   Yes
HCM: Full             -798.16   12  1620.33  1700.40   0.855   Yes
HCM-M7: Domain Sep    -795.50   10  1611.00  1671.06   0.857   Yes
ICLV: Simultaneous    -790.20   15  1610.40  1710.50   0.859   Yes  <- Best!
```

- **LL**: Log-likelihood (higher = better fit)
- **K**: Number of parameters
- **AIC**: Akaike Information Criterion (lower = better)
- **BIC**: Bayesian Information Criterion (lower = better)
- **rho2**: McFadden's R-squared (higher = better, max 1.0)
- **Conv**: Convergence status

### Parameter Estimates

```
Parameter          True    Estimate   SE      t-stat   Bias%   Covered
ASC_paid           5.00     4.85     0.42    11.55    -3.0%   Yes
B_FEE             -0.08    -0.078    0.007  -10.61    -2.5%   Yes
B_DUR             -0.08    -0.082    0.012   -6.55    +2.5%   Yes
B_FEE_PatBlind    -0.10    -0.095    0.010   -9.50    -5.0%   Yes
```

- **True**: Known parameter value (simulation only)
- **Bias%**: Percentage deviation from true
- **Covered**: True value in 95% CI?
- **t-stat > 1.96**: Significant at 5% level

---

## Isolated Model Validation (models/ folder)

Each model in `models/` has its own Data Generating Process (DGP) that exactly matches the model specification. This ensures unbiased parameter recovery.

### How It Works

```
┌─────────────────────────────────────────────────────────────────────┐
│  Step 1: CONFIGURE                                                   │
│  config.json → Define TRUE parameter values                         │
│  • Population size (N individuals, T choice tasks)                   │
│  • True coefficients (ASC, B_FEE, B_DUR, etc.)                      │
│  • Demographics distribution                                         │
│  • Latent variables (for HCM/ICLV)                                  │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Step 2: SIMULATE                                                    │
│  simulate_full_data.py → Generate synthetic data                    │
│  • Draw demographics from config distributions                       │
│  • Compute utilities using TRUE parameters                          │
│  • Simulate choices using Random Utility Model                      │
│  • Output: data/simulated_data.csv                                  │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Step 3: ESTIMATE                                                    │
│  model.py → Recover parameters using Biogeme                        │
│  • Load simulated data                                              │
│  • Estimate model (MLE optimization)                                │
│  • Compare estimates to TRUE values                                 │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Step 4: OUTPUT                                                      │
│  Saved to multiple locations:                                       │
│  • results/parameter_comparison.csv   (estimates vs true)           │
│  • results/estimation_results.csv     (full Biogeme output)         │
│  • output/latex/                      (publication tables)          │
│  • policy_analysis/                   (WTP, elasticities)           │
│  • sample_stats/                      (descriptive statistics)      │
└─────────────────────────────────────────────────────────────────────┘
```

### Config.json Structure

Each model's `config.json` defines the true parameters:

```json
{
  "model_info": {
    "name": "MNL Basic",
    "true_values": {
      "ASC_paid": 5.0,
      "B_FEE": -0.08,
      "B_DUR": -0.08
    }
  },
  "population": {
    "N": 500,
    "T": 10,
    "seed": 42
  }
}
```

**To experiment:** Modify `true_values` in config.json, run `python run.py`, and check if estimates recover your values.

### Output Files

| Location | File | Contents |
|----------|------|----------|
| `results/` | `parameter_comparison.csv` | True vs Estimated, bias %, CI coverage |
| `results/` | `estimation_results.csv` | Full Biogeme output |
| `output/latex/` | `*.tex` | Publication-ready tables |
| `policy_analysis/` | `wtp.csv` | Willingness-to-pay |

### Each Folder Contains

- `config.json` - Model-specific true parameters
- `simulate_full_data.py` - Generates data matching model specification
- `model.py` - Biogeme estimation code
- `run.py` - Orchestrates simulation + estimation
- `results/` - Parameter estimates vs true values
- `output/latex/` - Publication-ready tables

**Key insight:** Compare HCM (~50-66% attenuation) vs ICLV (~2.4% bias) to see why simultaneous estimation matters.

---

## Running Individual Components

### Generate New Simulated Data

```bash
python src/simulation/simulate_full_data.py \
    --config config/model_config.json \
    --output data/simulated/my_data.csv \
    --n_individuals 1000 \
    --n_tasks 10
```

### Run Specific Model Types

```bash
# Basic MNL Models
python src/models/mnl_model_comparison.py --data data/simulated/full_scale_test.csv

# Extended MNL (8 specifications)
python -c "from src.models.mnl_extended import run_mnl_extended; run_mnl_extended('data/simulated/full_scale_test.csv')"

# Basic MXL Models (takes longer due to simulation draws)
python src/models/mxl_models.py --data data/simulated/full_scale_test.csv --draws 500

# Extended MXL (8 specifications)
python -c "from src.models.mxl_extended import run_mxl_extended; run_mxl_extended('data/simulated/full_scale_test.csv', n_draws=500)"

# HCM with 4 latent variable constructs
python src/models/hcm_split_latents.py --data data/simulated/full_scale_test.csv

# Extended HCM (8 specifications)
python -c "from src.models.hcm_extended import run_hcm_extended; run_hcm_extended('data/simulated/full_scale_test.csv')"

# ICLV (simultaneous estimation)
python -c "
from src.models.iclv import estimate_iclv
import pandas as pd
df = pd.read_csv('data/simulated/full_scale_test.csv')
result = estimate_iclv(
    df=df,
    constructs={'pat_blind': ['pat_blind_1', 'pat_blind_2', 'pat_blind_3']},
    choice_col='CHOICE',
    n_draws=500,
    auto_scale=True,
    use_two_stage_start=True
)
print(f'Log-likelihood: {result.log_likelihood:.2f}')
"
```

---

## Extended Model Specifications

### MNL Extended (8 Models)

| Model | Specification | Key Feature |
|-------|--------------|-------------|
| M1 | Basic MNL | Baseline |
| M2 | Log Fee | Diminishing sensitivity |
| M3 | Quadratic Fee | Non-linear effect |
| M4 | Piecewise Fee | Threshold at median |
| M5 | Full Demographics | All demographic interactions |
| M6 | Cross Demographics | Age x Income interactions |
| M7 | Log Fee + Demo | Combined |
| M8 | ASC Demographics | Heterogeneous ASC |

### MXL Extended (8 Models)

| Model | Distribution | Feature |
|-------|-------------|---------|
| M1 | Normal Fee | Random fee coefficient |
| M2 | Normal Fee+Dur | Both random |
| M3 | Lognormal Fee | Always negative |
| M4 | Lognormal Both | Both lognormal |
| M5 | Uniform Fee | Bounded heterogeneity |
| M6 | Random ASC | Alternative-specific |
| M7 | Log Fee Random | Transformed |
| M8 | Demo Shifters | Heterogeneity in mean |

### HCM Extended (8 Models)

| Model | LV Configuration | Feature |
|-------|-----------------|---------|
| M1 | PatBlind on Fee | Single LV effect |
| M2 | SecDL on Fee | Different LV |
| M3 | LVs on Duration | Duration interaction |
| M4 | LVs on ASC | ASC interaction |
| M5 | Quadratic LV | Non-linear LV effect |
| M6 | LV x Demo | LV-demographic interaction |
| M7 | Domain Separation | Patriotism->Fee, Secularism->Dur |
| M8 | Full Specification | All LVs, all attributes |

---

## Configuration Files

| File | Purpose |
|------|---------|
| `config/model_config.json` | TRUE parameters for simulation & validation |
| `config/model_config.json` | Simple config for quick testing |
| `config/items_config.csv` | Likert scale item definitions |
| `biogeme.toml` | Optimizer settings |

---

## Model Progression

The framework estimates models from simple to complex:

```
MNL-Basic      -> Baseline (ASC + fee + duration)
    |
MNL-Demo       -> + Demographic interactions
    |
MNL-Extended   -> 8 functional form specifications
    |
MXL            -> + Random parameters (taste heterogeneity)
    |
MXL-Extended   -> 8 distribution specifications
    |
HCM            -> + Latent variables (two-stage, biased)
    |
HCM-Extended   -> 8 LV configuration specifications
    |
ICLV           -> + Simultaneous estimation (unbiased)
```

Each model builds on the previous, allowing comparison of added complexity.

---

## ICLV Enhanced Features

The ICLV module includes these improvements for publication-quality estimation:

```python
from src.models.iclv import estimate_iclv

result = estimate_iclv(
    df=data,
    constructs={'pat_blind': ['pb1', 'pb2', 'pb3']},
    choice_col='CHOICE',
    n_draws=500,

    # Enhanced features
    auto_scale=True,           # Automatic fee scaling
    use_two_stage_start=True,  # Better starting values
    use_panel=True,            # Panel-corrected SE
    compute_robust_se=True,    # Sandwich estimator
    estimate_lv_correlation=True  # LV correlations
)
```

---

## Troubleshooting

### "Algorithm did not converge"

1. Check `biogeme.toml` uses `simple_bounds_BFGS` optimizer
2. Increase `max_iterations` to 10000
3. Try simpler model specification
4. Enable `use_two_stage_start=True` for ICLV

### Negative rho-squared

Model is worse than random choice. Check:
1. Data quality (missing values, outliers)
2. Variable coding (correct signs)
3. Identification (not too many parameters)

### Import Errors

```bash
# Ensure running from project root
cd /path/to/project

# Add src to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### ICLV Numerical Issues

If ICLV estimation has convergence problems:
1. Enable `auto_scale=True` for large fee values
2. Increase `n_draws` to 500-1000
3. Use `use_two_stage_start=True` for better initialization
4. Check that constructs have at least 3 indicators each

---

## Next Steps

1. Read `README.md` for complete documentation
2. Read `docs/METHODOLOGY.md` for statistical details
3. Read `docs/MODEL_SPECIFICATIONS.md` for model equations
4. Modify `config/model_config.json` for your parameters
5. Add your own data following the format in `data/simulated/`
