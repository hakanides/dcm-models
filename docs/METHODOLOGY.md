# Methodology

**Authors: Hakan Mülayim, Giray Girengir, Ataol Azeritürk**

This document provides comprehensive methodological documentation for the DCM estimation framework, covering the theoretical foundations, estimation procedures, and validation approaches.

---

## Table of Contents

1. [Random Utility Model (RUM) Framework](#1-random-utility-model-rum-framework)
2. [Model Hierarchy](#2-model-hierarchy)
3. [Data Generating Process (DGP)](#3-data-generating-process-dgp)
4. [Measurement Model](#4-measurement-model)
5. [Two-Stage vs Simultaneous Estimation](#5-two-stage-vs-simultaneous-estimation)
6. [Policy Analysis Methodology](#6-policy-analysis-methodology)
7. [Configuration System](#7-configuration-system)
8. [Isolated Model Validation Workflow](#8-isolated-model-validation-workflow)
9. [Extended Model Specifications](#9-extended-model-specifications)
10. [Validation Metrics](#10-validation-metrics)
11. [LaTeX Output Format](#11-latex-output-format)
12. [References](#12-references)

---

## 1. Random Utility Model (RUM) Framework

### 1.1 Mathematical Foundation

The Random Utility Model forms the theoretical basis for all discrete choice models in this framework. For individual n choosing among J alternatives:

```
U_nj = V_nj + ε_nj
```

Where:
- `U_nj` = total (unobserved) utility of alternative j for individual n
- `V_nj` = systematic (observed) utility component
- `ε_nj` = random (unobserved) error term

### 1.2 Choice Rule

Individual n chooses alternative j if and only if:

```
U_nj > U_nk  for all k ≠ j
```

The choice probability is:

```
P_nj = Pr(V_nj + ε_nj > V_nk + ε_nk  for all k ≠ j)
```

### 1.3 Gumbel Distribution and Logit Probabilities

When errors follow the Type I Extreme Value (Gumbel) distribution:

```
F(ε) = exp(-exp(-ε))
```

This yields the closed-form multinomial logit probability:

```
P_nj = exp(V_nj) / Σ_k exp(V_nk)
```

### 1.4 Gumbel Error Generation in DGP

The simulation code generates Gumbel errors using the inverse transform method:

```python
def draw_gumbel_errors(rng, n):
    """Generate Gumbel-distributed errors using inverse CDF method."""
    u = rng.uniform(1e-10, 1-1e-10, size=n)
    return -np.log(-np.log(u))
```

**Why this works:**
- If U ~ Uniform(0,1), then -ln(-ln(U)) ~ Gumbel(0,1)
- The bounds (1e-10, 1-1e-10) prevent numerical overflow at extremes
- This is the standard inverse CDF transformation for Gumbel distribution

### 1.5 Systematic Utility Specification

In this framework, the systematic utility is linear in parameters:

```
V_nj = ASC_j + Σ_a β_a × x_nj,a + Σ_m β_LV,m × η_nm
```

Where:
- `ASC_j` = alternative-specific constant (normalized: ASC_free = 0)
- `β_a` = coefficient for attribute a
- `x_nj,a` = value of attribute a for alternative j faced by individual n
- `β_LV,m` = coefficient for latent variable m
- `η_nm` = value of latent variable m for individual n

---

## 2. Model Hierarchy

The framework implements a progression of models from simple to complex:

### 2.1 Model Comparison Table

| Model | Parameters | Heterogeneity Type | Estimation | LV Bias |
|-------|------------|-------------------|------------|---------|
| MNL Basic | 3 | None (homogeneous) | MLE | N/A |
| MNL Demographics | 8 | Observable (demographics) | MLE | N/A |
| MXL Basic | 4-6 | Unobserved (random β) | SML | N/A |
| HCM Basic | 5-8 | Latent variables | Two-stage | 15-30% |
| HCM Full | 13+ | 4 latent variables | Two-stage | 15-30% |
| ICLV | 10-30 | Latent variables | Simultaneous SML | ~0% |

### 2.2 MNL Basic

The simplest model with fixed coefficients:

```
V_paid = ASC_paid + B_FEE × fee + B_DUR × duration
V_free = 0  (normalized)
```

**Parameters**: ASC_paid, B_FEE, B_DUR

### 2.3 MNL with Demographics

Extends MNL by allowing demographic-specific effects:

```
V_paid = ASC_paid + (B_FEE + B_FEE_AGE × age) × fee
                  + (B_DUR + B_DUR_EDU × edu) × duration
```

**Captures**: Observable heterogeneity through demographic interactions

### 2.4 Mixed Logit (MXL)

Allows coefficients to vary across individuals:

```
β_n,FEE ~ N(μ_FEE, σ²_FEE)
```

**Captures**: Unobserved preference heterogeneity

**Estimation**: Simulated Maximum Likelihood with Halton draws

### 2.5 Hybrid Choice Model (HCM)

Incorporates latent psychological constructs:

```
V_paid = ASC_paid + B_FEE × fee + B_FEE_LV × η_patblind × fee + B_DUR × duration
```

Where `η_patblind` is the latent "Patriotism/Blind Trust" variable

**Problem**: Two-stage estimation causes attenuation bias (see Section 5)

### 2.6 ICLV (Integrated Choice and Latent Variable)

Simultaneously estimates measurement, structural, and choice models:

```
Joint likelihood: L_n = ∫ P(choice|η) × P(indicators|η) × f(η) dη
```

**Advantage**: Unbiased latent variable coefficient estimates

---

## 3. Data Generating Process (DGP)

### 3.1 Workflow Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  Step 1: CONFIGURE                                                  │
│  config.json → Define TRUE parameter values                         │
│  • Population size (N individuals, T choice tasks)                  │
│  • True coefficients (ASC, B_FEE, B_DUR, etc.)                     │
│  • Demographics distribution                                        │
│  • Latent variables (for HCM/ICLV)                                 │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Step 2: SIMULATE                                                   │
│  simulate_full_data.py → Generate synthetic data                    │
│  • Draw demographics from configured distributions                  │
│  • Compute utilities using TRUE parameters                          │
│  • Simulate choices using Random Utility Model                      │
│  • Output: data/simulated_data.csv                                 │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Step 3: ESTIMATE                                                   │
│  model.py → Recover parameters using Biogeme                        │
│  • Load simulated data                                              │
│  • Estimate model (MLE optimization)                                │
│  • Compare estimates to TRUE values                                 │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Step 4: OUTPUT                                                     │
│  Saved to multiple locations:                                       │
│  • results/parameter_comparison.csv   (estimates vs true)           │
│  • results/estimation_results.csv     (full Biogeme output)         │
│  • output/latex/                      (publication tables)          │
│  • policy_analysis/                   (WTP, elasticities)           │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Demographics Generation

Demographics are drawn from categorical distributions specified in config:

```python
# From config.json
"demographics": {
    "age_idx": {
        "type": "categorical",
        "values": [0, 1, 2, 3],
        "probabilities": [0.15, 0.35, 0.30, 0.20]
    },
    "edu_idx": {
        "type": "categorical",
        "values": [0, 1, 2],
        "probabilities": [0.30, 0.45, 0.25]
    }
}
```

### 3.3 Latent Variable Generation

Latent variables are generated from a structural model:

```
η_m = Γ_m0 + Σ_p Γ_mp × X_p + ζ_m
```

Where:
- `Γ_m0` = intercept (often 0 for centering)
- `Γ_mp` = effect of demographic p on latent m
- `ζ_m ~ N(0, σ²_m)` = residual variance

**Implementation** (from `simulate_full_data.py`):

```python
# Structural equation for latent variable
lv_true = (
    config['latent']['intercept'] +
    config['latent']['betas']['age_idx'] * (age_idx - age_center) +
    config['latent']['betas']['edu_idx'] * (edu_idx - edu_center) +
    rng.normal(0, config['latent']['sigma'], n_individuals)
)
```

**Note**: Demographics are centered (e.g., `age_idx - 1.55`) to improve interpretation and reduce collinearity.

### 3.4 Likert Response Generation

Likert items are generated using the ordered probit model:

```python
def generate_likert_responses(lv_values, loadings, thresholds, rng):
    """Generate ordinal Likert responses from latent variables."""
    responses = {}
    for item, loading in loadings.items():
        # Latent continuous response
        y_star = loading * lv_values + rng.normal(0, 1, len(lv_values))

        # Map to ordinal categories using thresholds
        response = np.ones(len(lv_values), dtype=int)  # Start at 1
        for j, tau in enumerate(thresholds):
            response += (y_star > tau).astype(int)

        responses[item] = response

    return responses
```

**Standard thresholds**: `[-1.0, -0.35, 0.35, 1.0]` for 5-point Likert scale

### 3.5 Choice Generation

Choices are simulated using the Random Utility Model:

```python
def simulate_choice(V_alternatives, rng):
    """Simulate choice using RUM with Gumbel errors."""
    n_obs = V_alternatives.shape[0]
    n_alts = V_alternatives.shape[1]

    # Draw Gumbel errors
    gumbel_errors = draw_gumbel_errors(rng, (n_obs, n_alts))

    # Total utility
    U = V_alternatives + gumbel_errors

    # Choice is alternative with maximum utility
    choice = np.argmax(U, axis=1) + 1  # +1 for 1-indexing

    return choice
```

---

## 4. Measurement Model

### 4.1 Ordered Probit Specification

Likert items are modeled as ordinal manifestations of underlying latent variables:

```
y*_k = λ_k × η + ε_k,  where ε_k ~ N(0,1)
y_k = j  if  τ_{j-1} < y*_k ≤ τ_j
```

Where:
- `y*_k` = latent continuous response for item k
- `λ_k` = factor loading (item sensitivity to construct)
- `η` = latent variable value
- `τ_j` = threshold parameters (cutpoints)
- `y_k` = observed ordinal response (1, 2, 3, 4, 5)

### 4.2 Choice Probability for Ordinal Response

```
P(y_k = j | η) = Φ(τ_j - λ_k × η) - Φ(τ_{j-1} - λ_k × η)
```

Where `Φ()` is the standard normal CDF.

### 4.3 Delta Parameterization for Threshold Ordering

**Problem**: Thresholds must be strictly ordered: τ_1 < τ_2 < τ_3 < τ_4

**Solution**: Use delta parameterization to guarantee ordering:

```python
# In ICLV estimation code
tau_1 = Beta('tau_1', -1.0, -3, 3, 0)
delta_2 = Beta('delta_2', 0.6, -3, 3, 0)
delta_3 = Beta('delta_3', 0.6, -3, 3, 0)
delta_4 = Beta('delta_4', 0.6, -3, 3, 0)

# Thresholds with guaranteed ordering
tau_2 = tau_1 + exp(delta_2)  # tau_2 > tau_1 always
tau_3 = tau_2 + exp(delta_3)  # tau_3 > tau_2 always
tau_4 = tau_3 + exp(delta_4)  # tau_4 > tau_3 always
```

**Why this works**: `exp(x) > 0` for all x, so each threshold is strictly greater than the previous.

### 4.4 Scale Identification

The measurement model requires scale normalization:

**Option 1**: Fix first loading to 1
```
λ_1 = 1 (fixed)
λ_2, λ_3, ... estimated freely
```

**Option 2**: Fix latent variance to 1
```
Var(η) = 1 (fixed)
All loadings estimated freely
```

This framework uses Option 1 for interpretability.

### 4.5 Reliability Assessment

**Cronbach's Alpha**:
```
α = (k/(k-1)) × (1 - Σ Var(y_k) / Var(Σ y_k))
```

**Interpretation**:
- α > 0.90: Excellent reliability
- α > 0.80: Good reliability
- α > 0.70: Acceptable reliability
- α < 0.70: Poor reliability (consider revising scale)

---

## 5. Two-Stage vs Simultaneous Estimation

### 5.1 Two-Stage HCM (Causes Attenuation Bias)

**Stage 1**: Estimate latent variable scores from Likert items
```python
# Factor analysis or weighted average
lv_score = weighted_mean(likert_items, loadings)
```

**Stage 2**: Use LV scores as fixed regressors in choice model
```
V = ASC + B_FEE × fee + B_LV × lv_score + ...
```

**Problem**: This treats `lv_score` as error-free, ignoring measurement error.

### 5.2 Attenuation Bias Formula

When using estimated LV scores instead of true values:

```
True model:     U = β_0 + β_LV × LV_true + ε
Estimated:      U = β_0 + β_LV × LV_est + ε

Where: LV_est = LV_true + measurement_error
```

The OLS estimator for β_LV is attenuated:

```
E[β̂_LV] = β_LV × reliability
```

Where:
```
reliability = Var(LV_true) / Var(LV_est) = Var(LV_true) / (Var(LV_true) + Var(error)) < 1
```

### 5.3 Bias Magnitude

| Factor | Effect on Bias |
|--------|---------------|
| Number of Likert items | More items → higher reliability → less bias |
| Factor loadings | Higher loadings → higher reliability → less bias |
| True effect size | Smaller effects more affected (proportionally) |
| Sample size | Does NOT fix bias (only improves precision) |

**Typical reliability** for 4-5 item Likert scales: 0.70-0.85

**Expected attenuation**: LV coefficients underestimated by **15-30%**

### 5.4 ICLV: Simultaneous Estimation (Unbiased)

ICLV integrates over the latent variable distribution in the likelihood:

```
L_n = ∫ P(y_n | η) × Π_k P(I_nk | η) × f(η | X_n) dη
```

**Monte Carlo Integration**:
```
L̂_n = (1/R) Σ_r [ P(y_n | η_r) × Π_k P(I_nk | η_r) ]
```

Where:
- `R` = number of simulation draws (typically 500-1000)
- `η_r` = draw r from the latent variable distribution
- Halton sequences provide more efficient coverage than pseudo-random draws

### 5.5 Comparison Table

| Aspect | Two-Stage HCM | ICLV |
|--------|---------------|------|
| **LV Coefficient Bias** | Attenuated 15-30% | Unbiased |
| **Standard Errors** | Underestimated | Correct |
| **Computational Time** | Fast (~seconds) | Slow (~minutes) |
| **Implementation** | Simple | Complex |
| **When to Use** | Exploratory analysis | Final publication |

### 5.6 Empirical Validation

From the framework's simulation studies:

| Parameter | True | HCM (Two-Stage) | ICLV | HCM Bias | ICLV Bias |
|-----------|------|-----------------|------|----------|-----------|
| B_FEE_PatBlind | -0.10 | -0.067 | -0.097 | -33% | -3% |
| B_FEE_SecDL | -0.08 | -0.052 | -0.078 | -35% | -2.5% |

---

## 6. Policy Analysis Methodology

### 6.1 Willingness-to-Pay (WTP)

**Definition**: The monetary amount an individual is willing to pay to obtain one unit improvement in an attribute.

**Formula**:
```
WTP_attribute = -β_attribute / β_fee × fee_scale
```

**Example**:
```
WTP_duration = -B_DUR / B_FEE × 10,000
             = -(-0.08) / (-0.008) × 10,000
             = -10 × 10,000 = -100,000 TL per month

Interpretation: Respondents require 100,000 TL compensation to accept
one additional month of processing time.
```

### 6.2 WTP Standard Error (Delta Method)

For WTP = -β_num / β_den × scale, the variance is:

```
Var(WTP) ≈ (∂WTP/∂β_num)² × Var(β_num)
         + (∂WTP/∂β_den)² × Var(β_den)
         + 2 × (∂WTP/∂β_num)(∂WTP/∂β_den) × Cov(β_num, β_den)
```

**Partial derivatives**:
```
∂WTP/∂β_num = -1/β_den × scale
∂WTP/∂β_den = β_num/β_den² × scale
```

**Implementation** (from `policy_tools.py`):
```python
def calculate_wtp_with_se(beta_num, beta_den, cov_matrix, scale=10000):
    """Calculate WTP with standard error using delta method."""
    wtp = -beta_num / beta_den * scale

    # Gradient vector
    grad = np.array([
        -1/beta_den * scale,           # d(WTP)/d(beta_num)
        beta_num/beta_den**2 * scale   # d(WTP)/d(beta_den)
    ])

    # Variance via delta method
    var_wtp = grad @ cov_matrix @ grad
    se_wtp = np.sqrt(var_wtp)

    return wtp, se_wtp
```

### 6.3 WTP Distribution for MXL

When coefficients are random:
```
β_fee,i ~ N(μ_fee, σ²_fee)
β_dur,i ~ N(μ_dur, σ²_dur)

WTP_i = -β_dur,i / β_fee,i × fee_scale
```

**Reporting**: Mean, median, std, percentiles (5th, 25th, 50th, 75th, 95th)

**Note**: If β_fee can be positive for some individuals, WTP has undefined moments. Use lognormal distribution for fee coefficient to ensure negative sign.

### 6.4 Own-Price Elasticity

**Definition**: Percentage change in choice probability for alternative j given 1% change in own attribute.

**Formula**:
```
η_jj = β_j × x_j × (1 - P_j)
```

**Point elasticity for fee**:
```
ε_own = B_FEE × fee × (1 - P_paid)
```

### 6.5 Cross-Price Elasticity

**Definition**: Percentage change in choice probability for alternative j given 1% change in attribute of alternative k.

**Formula** (for IIA logit):
```
η_jk = -β × x_k × P_k  (for j ≠ k)
```

### 6.6 Arc Elasticity (Discrete Change)

For larger changes (±10%, ±20%):

```
ε_arc = (P_new - P_baseline) / P_baseline × 100 / % change in attribute
```

**Bootstrap standard errors**: 1000 replicates for confidence intervals

### 6.7 Compensating Variation (Consumer Surplus)

**Definition**: Monetary equivalent of utility change from policy.

**Formula**:
```
CV = -(1/β_fee) × [log-sum(new) - log-sum(baseline)] × fee_scale
```

Where:
```
log-sum = ln(Σ_j exp(V_j))
```

**Interpretation**: Positive CV means policy improves welfare; negative means harm.

### 6.8 Scenario Analysis

**Default scenarios**:
- Fee changes: ±10%, ±20%
- Duration changes: ±5 days, ±10 days
- Combined scenarios

**Output includes**:
- Market share predictions
- Choice probability changes
- Revenue implications (if applicable)

---

## 7. Configuration System

### 7.1 config.json Structure

```json
{
  "model_info": {
    "name": "MNL Basic",
    "description": "Baseline multinomial logit",
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
  },
  "demographics": {
    "age_idx": {
      "type": "categorical",
      "values": [0, 1, 2, 3],
      "probabilities": [0.15, 0.35, 0.30, 0.20],
      "centering": 1.55
    },
    "edu_idx": {
      "type": "categorical",
      "values": [0, 1, 2],
      "probabilities": [0.30, 0.45, 0.25],
      "centering": 0.95
    }
  },
  "latent": {
    "pat_blind": {
      "structural": {
        "intercept": 0,
        "sigma": 1.0,
        "betas": {
          "age_idx": 0.20,
          "edu_idx": -0.15
        }
      },
      "measurement": {
        "items": ["pat_blind_1", "pat_blind_2", "pat_blind_3", "pat_blind_4"],
        "loadings": [1.0, 0.85, 0.78, 0.72],
        "thresholds": [-1.0, -0.35, 0.35, 1.0]
      }
    }
  },
  "choice_model": {
    "fee_scale": 10000,
    "alternatives": ["free", "paid"],
    "attributes": {
      "fee": {"min": 0, "max": 5600000},
      "duration": {"min": 30, "max": 180}
    },
    "parameters": {
      "ASC_paid": 5.0,
      "B_FEE": -0.08,
      "B_DUR": -0.08
    },
    "lv_effects": {
      "B_FEE_PatBlind": -0.10
    }
  }
}
```

### 7.2 Critical Parameters

| Parameter | Purpose | Typical Value |
|-----------|---------|---------------|
| `fee_scale` | Scale fee for numerical stability | 10000 |
| `centering` | Center demographics for interpretation | Mean of distribution |
| `seed` | Reproducibility | 42 |
| `N` | Sample size | 500-2000 |
| `T` | Choice tasks per individual | 8-12 |

### 7.3 Sign Enforcement

Coefficients can be constrained to economically sensible signs:

```json
"sign_constraints": {
  "B_FEE": "negative",
  "B_DUR": "negative",
  "ASC_paid": "none"
}
```

---

## 8. Isolated Model Validation Workflow

### 8.1 Purpose

Each model folder (`models/mnl_basic`, `models/mxl_basic`, etc.) contains a self-contained validation system where:

1. **True parameters** are specified in `config.json`
2. **Data is simulated** using the exact DGP matching the model specification
3. **Parameters are estimated** using Biogeme
4. **Estimates are compared** to true values

This ensures **unbiased parameter recovery** when the model is correctly specified.

### 8.2 Folder Structure

```
models/mnl_basic/
├── config.json          # True parameters
├── simulate_full_data.py # DGP matching model
├── model.py             # Biogeme estimation
├── run.py               # Orchestration script
├── data/
│   └── simulated_data.csv
├── results/
│   ├── parameter_comparison.csv
│   └── estimation_results.csv
├── output/
│   └── latex/
│       ├── parameter_table.tex
│       └── model_summary.tex
└── policy_analysis/
    └── wtp.csv
```

### 8.3 Running a Model

```bash
cd models/mnl_basic
python run.py
```

**What happens**:
1. Loads config.json
2. Calls simulate_full_data.py to generate data
3. Calls model.py to estimate
4. Computes bias, RMSE, coverage
5. Saves results and LaTeX tables

### 8.4 Validation Output

Example `parameter_comparison.csv`:

```
Parameter,True,Estimate,SE,t-stat,Bias,Bias%,CI_Lower,CI_Upper,Covered
ASC_paid,5.00,4.92,0.38,12.95,-0.08,-1.5%,4.18,5.66,Yes
B_FEE,-0.080,-0.079,0.007,-10.61,0.001,+1.8%,-0.093,-0.065,Yes
B_DUR,-0.080,-0.081,0.012,-6.55,-0.001,-0.6%,-0.104,-0.057,Yes
```

---

## 9. Extended Model Specifications

### 9.1 MNL Extended (8 Models)

| Model | Specification | Key Feature |
|-------|--------------|-------------|
| M1 | Linear | Baseline reference |
| M2 | Log(fee) | Diminishing sensitivity |
| M3 | fee + fee² | Quadratic non-linearity |
| M4 | Piecewise | Threshold at median |
| M5 | Full demographics | Age, edu, income interactions |
| M6 | Cross demographics | Age × Income |
| M7 | Log + demographics | Combined |
| M8 | Heterogeneous ASC | ASC varies by demographics |

### 9.2 MXL Extended (8 Models)

| Model | Distribution | Feature |
|-------|-------------|---------|
| M1 | Normal(μ, σ) for fee | Random fee sensitivity |
| M2 | Normal for fee + dur | Both random |
| M3 | -exp(μ + σε) for fee | Lognormal, always negative |
| M4 | Lognormal both | Both sign-constrained |
| M5 | Uniform[a,b] for fee | Bounded heterogeneity |
| M6 | Random ASC | Alternative-specific |
| M7 | Random log(fee) | Transformed scale |
| M8 | μ + γ×demographics | Heterogeneity in mean |

### 9.3 HCM Extended (8 Models)

| Model | LV Configuration | Feature |
|-------|-----------------|---------|
| M1 | PatBlind on fee | Single LV effect |
| M2 | SecDL on fee | Different construct |
| M3 | LVs on duration | Duration interaction |
| M4 | LVs on ASC | ASC interaction |
| M5 | LV + LV² | Non-linear LV effect |
| M6 | LV × demographics | LV-demographic interaction |
| M7 | Domain separation | Patriotism→Fee, Secularism→Duration |
| M8 | Full specification | All LVs, all attributes |

### 9.4 Model Selection Criteria

**Within family** (nested models):
```
LR = 2 × (LL_full - LL_restricted)
df = K_full - K_restricted
p-value from χ² distribution
```

**Across families** (non-nested):
- AIC: 2K - 2LL (prefer lower)
- BIC: K×ln(n) - 2LL (prefer lower)
- Theoretical plausibility

---

## 10. Validation Metrics

### 10.1 Parameter Recovery Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| Bias | estimate - true | Near 0 |
| % Bias | 100 × (estimate - true) / \|true\| | < 10% |
| RMSE | √(Σ(bias²)/n) | Low |
| 95% CI Coverage | % true values in CI | ~95% |

### 10.2 Target Benchmarks

For well-specified models:
- **Bias** < 10% for all parameters
- **Coverage** ≈ 95% (90-97% acceptable)
- **All parameters significant** (|t| > 1.96)

### 10.3 Model Fit Statistics

| Statistic | Formula | Interpretation |
|-----------|---------|----------------|
| Log-likelihood (LL) | Σ ln(P_choice) | Higher = better fit |
| AIC | 2K - 2LL | Lower = better (penalizes complexity) |
| BIC | K×ln(n) - 2LL | Lower = better (stronger penalty) |
| McFadden ρ² | 1 - LL/LL_0 | 0-1 scale, higher = better |

**ρ² interpretation**:
- 0.20-0.40: Good fit for discrete choice models
- \> 0.40: Excellent fit
- < 0.10: Poor fit

---

## 11. LaTeX Output Format

### 11.1 Generated Tables

| File | Contents |
|------|----------|
| `parameter_table.tex` | Estimates with SE, t-stats, bias, coverage |
| `model_summary.tex` | LL, K, AIC, BIC, ρ², convergence |
| `policy_summary.tex` | WTP, market shares |
| `elasticity_table.tex` | Own and cross elasticities |

### 11.2 Significance Stars

```
*** p < 0.001 (|t| > 3.291)
**  p < 0.01  (|t| > 2.576)
*   p < 0.05  (|t| > 1.960)
```

### 11.3 Example Parameter Table

```latex
\begin{table}[htbp]
\centering
\caption{MNL Basic Parameter Estimates}
\begin{tabular}{lrrrrrr}
\hline
Parameter & True & Estimate & SE & t-stat & Bias\% & Covered \\
\hline
ASC\_paid & 5.00 & 4.92*** & 0.38 & 12.95 & -1.5\% & Yes \\
B\_FEE & -0.080 & -0.079*** & 0.007 & -10.61 & +1.8\% & Yes \\
B\_DUR & -0.080 & -0.081*** & 0.012 & -6.55 & -0.6\% & Yes \\
\hline
\end{tabular}
\end{table}
```

---

## 12. References

### Core Discrete Choice Methodology

- Train, K. (2009). *Discrete Choice Methods with Simulation* (2nd ed.). Cambridge University Press.
- Ben-Akiva, M., & Lerman, S. (1985). *Discrete Choice Analysis: Theory and Application to Travel Demand*. MIT Press.
- McFadden, D. (1974). Conditional logit analysis of qualitative choice behavior. In P. Zarembka (Ed.), *Frontiers in Econometrics*.

### Hybrid Choice Models and ICLV

- Ben-Akiva, M., et al. (2002). Hybrid Choice Models: Progress and Challenges. *Marketing Letters*, 13(3), 163-175.
- Walker, J., & Ben-Akiva, M. (2002). Generalized random utility model. *Mathematical Social Sciences*, 43(3), 303-343.
- Vij, A., & Walker, J. (2016). How, when and why integrated choice and latent variable models are latently useful. *Transportation Research Part B*, 90, 208-224.
- Daly, A., et al. (2012). A synthesis of analytical results on integration of latent variables in choice models. *Transportation*.

### Welfare Analysis

- Small, K., & Rosen, H. (1981). Applied welfare economics with discrete choice models. *Econometrica*, 49(1), 105-130.
- McFadden, D. (1999). Computing willingness-to-pay in random utility models. In *Trade, Theory and Econometrics*. Routledge.

### Simulation and Estimation

- Halton, J. (1960). On the efficiency of certain quasi-random sequences of points in evaluating multi-dimensional integrals. *Numerische Mathematik*, 2(1), 84-90.
- Train, K. (2000). Halton sequences for mixed logit. Working Paper, UC Berkeley.

### Robust Inference

- White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity. *Econometrica*, 48(4), 817-838.
- Wooldridge, J. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). MIT Press.

### Measurement and Psychometrics

- Bollen, K. (1989). *Structural Equations with Latent Variables*. Wiley.
- Cronbach, L. (1951). Coefficient alpha and the internal structure of tests. *Psychometrika*, 16(3), 297-334.

---

*Last updated: January 2026*
