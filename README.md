# Discrete Choice Modeling (DCM) Research Project

A comprehensive framework for estimating Discrete Choice Models including Multinomial Logit (MNL), Mixed Logit (MXL), and Hybrid Choice Models (HCM) with latent variables.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Workflow](#workflow)
5. [Data Description](#data-description)
6. [Model Specifications](#model-specifications)
7. [Usage Guide](#usage-guide)
8. [Results Interpretation](#results-interpretation)
9. [Troubleshooting](#troubleshooting)

---

## Project Overview

This project implements a complete DCM estimation pipeline for analyzing citizen preferences regarding military service alternatives. The choice experiment presents respondents with three options:

| Option | Description | Fee | Duration | Exemption |
|--------|-------------|-----|----------|-----------|
| Option 1 | Paid partial service | Varies (0-5.6M) | Varies (1-23 months) | Yes |
| Option 2 | Paid partial service | Varies (0-5.6M) | Varies (1-23 months) | Yes |
| Option 3 | Free standard service | Always 0 | Always 24 months | No |

### Key Features

- **Agent-based data simulation** with realistic behavioral parameters
- **Multiple model types**: MNL, MXL, HCM/ICLV
- **Latent variable integration**: Patriotism and Secularism constructs
- **Robust estimation**: Bounds-constrained optimization with convergence guarantees
- **Comprehensive model comparison**: AIC, BIC, rho-squared, LR tests

---

## Installation

### Requirements

```bash
pip install biogeme pandas numpy scipy matplotlib scikit-learn
```

### Biogeme Configuration

The project uses a custom `biogeme.toml` configuration for robust estimation:

```toml
[Estimation]
optimization_algorithm = "simple_bounds_BFGS"  # Handles parameter bounds

[SimpleBounds]
tolerance = 1e-08           # Tight convergence criterion
max_iterations = 10000      # Sufficient for complex models

[Specification]
numerically_safe = "True"   # Handles numerical issues
```

---

## Project Structure

```
DCM draft3/
├── README.md                      # This documentation
├── biogeme.toml                   # Biogeme configuration
│
├── data/                          # Data files
│   ├── test_small_sample.csv      # Main analysis data (50 respondents)
│   ├── test_simulation.csv        # Simulation output
│   ├── test_advanced.csv          # Advanced simulation
│   └── config/                    # Configuration files
│       ├── items_config.csv       # Likert scale items
│       └── items_config_advanced.csv
│
├── src/                           # Source code
│   ├── simulation/                # Data simulation
│   │   ├── dcm_simulator.py       # Basic simulator
│   │   ├── dcm_simulator_advanced.py  # Advanced with LVs
│   │   ├── simulate_full_data.py  # Full simulation runner
│   │   └── prepare_scenarios.py   # Scenario preparation
│   │
│   ├── models/                    # Model specifications
│   │   ├── mnl_model_comparison.py    # MNL models
│   │   ├── mnl_model_comparison_v2.py # MNL with interactions
│   │   ├── mxl_models.py          # Mixed Logit models
│   │   ├── hcm_model.py           # Basic HCM
│   │   ├── hcm_model_improved.py  # Improved HCM with CFA
│   │   └── hcm_split_latents.py   # HCM with 4 split LVs
│   │
│   ├── estimation/                # Estimation utilities
│   │   ├── robust_estimation.py   # Robust estimation module
│   │   └── validate_estimation.py # Validation diagnostics
│   │
│   └── analysis/                  # Analysis scripts
│       └── final_comparison.py    # Cross-model comparison
│
└── results/                       # Output results
    ├── mnl/                       # MNL results
    ├── mxl/                       # MXL results
    └── hcm/                       # HCM results
```

---

## Workflow

### Complete Analysis Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    1. DATA SIMULATION                            │
│  dcm_simulator_advanced.py → test_small_sample.csv              │
│  - Generates synthetic DCM data                                  │
│  - Creates latent variable indicators                            │
│  - Simulates choice behavior                                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    2. MNL ESTIMATION                             │
│  mnl_model_comparison.py → results/mnl/                         │
│  - Baseline models                                               │
│  - Attribute interactions                                        │
│  - Demographic interactions                                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    3. MXL ESTIMATION                             │
│  mxl_models.py → results/mxl/                                   │
│  - Random parameters                                             │
│  - Taste heterogeneity                                           │
│  - Panel data structure                                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    4. HCM/ICLV ESTIMATION                        │
│  hcm_split_latents.py → results/hcm/                            │
│  - Latent variable estimation (CFA)                              │
│  - LV-attribute interactions                                     │
│  - Multiple specifications                                       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    5. MODEL COMPARISON                           │
│  final_comparison.py → results/final_comparison/                │
│  - AIC/BIC ranking                                               │
│  - Rho-squared comparison                                        │
│  - Best model selection                                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Description

### Choice Variables

| Variable | Description | Range |
|----------|-------------|-------|
| `CHOICE` | Selected alternative (1, 2, or 3) | 1-3 |
| `fee1`, `fee2`, `fee3` | Fee for each alternative (TL) | 0 - 5,600,000 |
| `dur1`, `dur2`, `dur3` | Duration in months | 1-24 |
| `exempt1`, `exempt2`, `exempt3` | Exemption status (binary) | 0/1 |

### Latent Variable Indicators

| Construct | Items | Description |
|-----------|-------|-------------|
| Blind Patriotism | `pat_blind_1` to `pat_blind_5` | Uncritical support for country |
| Constructive Patriotism | `pat_constructive_1` to `pat_constructive_5` | Critical, improvement-oriented |
| Daily Life Secularism | `sec_dl_1` to `sec_dl_5` | Separation in daily practices |
| Faith & Prayer Secularism | `sec_fp_1` to `sec_fp_5` | Separation in religious matters |

### Demographics

| Variable | Description |
|----------|-------------|
| `age_idx` | Age category (1-5) |
| `edu_idx` | Education level (1-5) |
| `income_indiv_idx` | Individual income (1-5) |
| `marital_idx` | Marital status (0/1) |

---

## Model Specifications

### Utility Function Structure

All models follow the Random Utility Maximization (RUM) framework:

```
U_j = V_j + ε_j

where V_j is the systematic utility and ε_j is the random component
```

### MNL Models (Multinomial Logit)

**Basic MNL (Baseline)**
```
V1 = ASC_paid + B_FEE × fee1 + B_DUR × dur1
V2 = ASC_paid + B_FEE × fee2 + B_DUR × dur2
V3 = B_FEE × fee3 + B_DUR × dur3
```

*Note: B_EXEMPT removed - exemption is constant within alternatives (always 1 for paid, always 0 for free). The exemption effect is absorbed into ASC_paid.*

### MXL Models (Mixed Logit)

**Random Parameters**
```
B_FEE_i = B_FEE + σ_FEE × η_i    (individual-specific fee sensitivity)
B_DUR_i = B_DUR + σ_DUR × η_i    (individual-specific duration sensitivity)
```

### HCM Models (Hybrid Choice Models)

The HCM framework integrates latent variables into choice models:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Likert Items   │ --> │ Latent Variable │ --> │ Choice Model    │
│  (Indicators)   │     │ (CFA Scores)    │     │ (Coefficients)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

#### HCM Model Specifications

| Model | Description | Specification |
|-------|-------------|---------------|
| **M0: Baseline MNL** | No LVs | `V = ASC + B_FEE×fee + B_DUR×dur` |
| **M1: Blind Patriotism** | PatBlind on Fee | `B_FEE_i = B_FEE + B_FEE_PB × LV_pat_blind` |
| **M2: Constructive Patriotism** | PatConst on Fee | `B_FEE_i = B_FEE + B_FEE_PC × LV_pat_const` |
| **M3: Daily Secularism** | SecDL on Fee | `B_FEE_i = B_FEE + B_FEE_SDL × LV_sec_dl` |
| **M4: Faith Secularism** | SecFP on Fee | `B_FEE_i = B_FEE + B_FEE_SFP × LV_sec_fp` |
| **M5: Both Patriotism** | Both Pat types on Fee | `B_FEE_i = B_FEE + B_FEE_PB×PB + B_FEE_PC×PC` |
| **M6: Both Secularism** | Both Sec types on Fee | `B_FEE_i = B_FEE + B_FEE_SDL×SDL + B_FEE_SFP×SFP` |
| **M7: PatBlind + SecDL** | Cross-domain | `B_FEE_i = B_FEE + B_FEE_PB×PB + B_FEE_SDL×SDL` |
| **M8: All 4 LVs on Fee** | All LVs → Fee | All 4 LVs affect fee sensitivity |
| **M9: LVs on Duration** | LVs → Duration | `B_DUR_i = B_DUR + B_DUR_PB×PB + B_DUR_SDL×SDL` |
| **M10: LVs on ASC** | LVs → ASC | `ASC_i = ASC + B_ASC_PB×PB + B_ASC_SDL×SDL` |
| **M11: LVs on Fee+Dur** | LVs → Both | LVs affect both fee and duration |
| **M12: Full Patriotism** | Both Pat → Fee+Dur | Patriotism constructs on both attributes |
| **M13: Full Secularism** | Both Sec → Fee+Dur | Secularism constructs on both attributes |
| **M14: Full Model** | All 4 → Fee+Dur | All LVs on both attributes |

#### Latent Variable Estimation (CFA)

The latent variables are estimated using a weighted sum approach:

```python
# For each construct:
1. Calculate item-total correlations
2. Use correlations as weights (minimum 0.1)
3. Compute weighted sum score
4. Standardize to N(0,1)
```

#### Parameter Bounds

To ensure convergence, parameters are bounded:

| Parameter | Lower | Upper | Rationale |
|-----------|-------|-------|-----------|
| B_FEE | -10 | 0 | Fee should be negative (disutility) |
| B_DUR | -5 | 0 | Duration should be negative |
| ASC | -10 | 10 | Alternative-specific constant |
| LV interactions | -2 | 2 | Prevent extreme values |

---

## Usage Guide

### Quick Start

```bash
# 1. Navigate to project directory
cd "/Users/hakanmulayim/Desktop/DCM draft3"

# 2. Run HCM with split latent variables (recommended)
python src/models/hcm_split_latents.py \
    --data data/test_small_sample.csv \
    --output results/hcm/latest

# 3. Run MNL comparison
python src/models/mnl_model_comparison.py \
    --data data/test_small_sample.csv \
    --output results/mnl/latest

# 4. Run final comparison
python src/analysis/final_comparison.py
```

### Running Individual Models

```bash
# MNL Models
python src/models/mnl_model_comparison.py --data data/test_small_sample.csv

# MXL Models
python src/models/mxl_models.py --data data/test_small_sample.csv

# HCM Models (improved version with CFA)
python src/models/hcm_model_improved.py --data data/test_small_sample.csv

# HCM with 4 split latent variables
python src/models/hcm_split_latents.py --data data/test_small_sample.csv
```

### Generating New Simulation Data

```bash
# Basic simulation
python src/simulation/dcm_simulator.py --output data/new_simulation.csv

# Advanced simulation with latent variables
python src/simulation/dcm_simulator_advanced.py \
    --n_respondents 100 \
    --n_tasks 100 \
    --output data/new_advanced.csv
```

---

## Results Interpretation

### Model Fit Statistics

| Statistic | Formula | Interpretation |
|-----------|---------|----------------|
| **Log-Likelihood (LL)** | ln(L) | Higher is better |
| **AIC** | 2K - 2LL | Lower is better (penalizes complexity) |
| **BIC** | K×ln(n) - 2LL | Lower is better (stronger penalty) |
| **Rho-squared (ρ²)** | 1 - LL/LL₀ | 0-1, higher is better |

### Interpreting Coefficients

| Coefficient | Expected Sign | Interpretation |
|-------------|---------------|----------------|
| B_FEE | Negative | Higher fee → lower utility |
| B_DUR | Negative | Longer duration → lower utility |
| ASC_paid | Context-dependent | Preference for paid options |
| B_FEE_PAT | Negative | Patriots more fee-sensitive |
| B_DUR_PAT | Positive | Patriots tolerate longer duration |

### Statistical Significance

| t-statistic | Significance |
|-------------|--------------|
| |t| > 1.645 | p < 0.10 (*) |
| |t| > 1.960 | p < 0.05 (**) |
| |t| > 2.576 | p < 0.01 (***) |

---

## Troubleshooting

### Common Issues

#### 1. "Algorithm did not converge"

**Cause:** Optimizer cannot find maximum likelihood
**Solutions:**
- Check for identification issues (constant variables)
- Use `simple_bounds_BFGS` optimizer
- Increase `max_iterations` in biogeme.toml
- Add reasonable parameter bounds

#### 2. Negative Rho-squared

**Cause:** Model worse than null model
**Solutions:**
- Check for perfect multicollinearity
- Remove constant variables (like B_EXEMPT)
- Simplify model specification
- Check data quality

#### 3. "Bounds will be ignored"

**Cause:** Optimizer doesn't support bounds
**Solution:** Use `simple_bounds_BFGS` or `simple_bounds` algorithm

#### 4. t-statistics showing 0.00

**Cause:** Parameter not identified
**Solutions:**
- Check for linear dependencies
- Remove redundant variables
- Check Hessian eigenvalues

### Key Discovery: Exemption Issue

**Problem:** In this dataset, exemption is constant within alternatives:
- Paid options (1,2): always `exempt = 1`
- Free option (3): always `exempt = 0`

**Impact:** B_EXEMPT is not identifiable - causes:
- Singular Hessian matrix
- Convergence failures
- Negative rho² values

**Solution:** Remove B_EXEMPT from all models. The exemption effect is absorbed into ASC_paid.

---

## References

- Biogeme: https://biogeme.epfl.ch/
- Train, K. (2009). Discrete Choice Methods with Simulation
- Ben-Akiva, M. & Lerman, S. (1985). Discrete Choice Analysis

---

## Contact

DCM Research Team
