# Model Specifications Guide

This document provides detailed specifications for all models implemented in this DCM research project.

---

## Table of Contents

1. [Theoretical Background](#theoretical-background)
2. [MNL Models](#mnl-models)
3. [MNL Extended Models](#mnl-extended-models)
4. [MXL Models](#mxl-models)
5. [MXL Extended Models](#mxl-extended-models)
6. [HCM/ICLV Models](#hcmiclv-models)
7. [HCM Extended Models](#hcm-extended-models)
8. [ICLV Simultaneous Estimation](#iclv-simultaneous-estimation)
9. [Latent Variable Estimation](#latent-variable-estimation)
10. [Model Comparison Framework](#model-comparison-framework)

---

## Theoretical Background

### Random Utility Maximization (RUM)

All models are based on the RUM framework:

```
U_ij = V_ij + e_ij
```

Where:
- `U_ij` = Total utility of alternative j for individual i
- `V_ij` = Systematic (observable) utility
- `e_ij` = Random (unobservable) component

### Choice Probability

The probability that individual i chooses alternative j:

```
P(j|i) = exp(V_ij) / sum_k exp(V_ik)
```

---

## MNL Models

### File: `src/models/mnl_model_comparison.py`

### Model 1: Baseline MNL

**Purpose:** Establish baseline with only choice attributes

**Specification:**
```
V1 = ASC_paid + B_FEE x fee1 + B_DUR x dur1
V2 = ASC_paid + B_FEE x fee2 + B_DUR x dur2
V3 = B_FEE x fee3 + B_DUR x dur3
```

**Parameters:**
| Parameter | Description | Expected Sign |
|-----------|-------------|---------------|
| ASC_paid | Preference for paid options (vs free) | +/- |
| B_FEE | Marginal utility of fee (per 10,000 TL) | Negative |
| B_DUR | Marginal utility of duration (per month) | Negative |

**Note:** Option 3 has `ASC = 0` (normalization)

---

### Model 2: MNL with Demographics

**Purpose:** Test demographic interactions with fee sensitivity

**Specification:**
```
B_FEE_i = B_FEE + B_FEE_AGE x age + B_FEE_EDU x edu + B_FEE_INC x income

V1 = ASC_paid + B_FEE_i x fee1 + B_DUR x dur1
V2 = ASC_paid + B_FEE_i x fee2 + B_DUR x dur2
V3 = B_FEE_i x fee3 + B_DUR x dur3
```

**Interpretation:**
- `B_FEE_AGE > 0`: Older people less fee-sensitive
- `B_FEE_INC > 0`: Higher income less fee-sensitive

---

### Model 3: MNL with Likert Interactions

**Purpose:** Test attitudinal effects using Likert scale averages

**Specification:**
```
B_FEE_i = B_FEE + B_FEE_PAT x pat_blind_idx + B_FEE_SEC x sec_dl_idx

V1 = ASC_paid + B_FEE_i x fee1 + B_DUR x dur1
V2 = ASC_paid + B_FEE_i x fee2 + B_DUR x dur2
V3 = B_FEE_i x fee3 + B_DUR x dur3
```

---

## MNL Extended Models

### File: `src/models/mnl_extended.py`

Eight specifications testing functional forms and demographic interactions.

### M1: Basic MNL (Reference)

```
V_j = ASC_paid + B_FEE x fee_j + B_DUR x dur_j
```

**Parameters:** 3 (ASC_paid, B_FEE, B_DUR)

---

### M2: Log Fee

**Purpose:** Test diminishing sensitivity to fee (Weber-Fechner law)

```
V_j = ASC_paid + B_FEE_LOG x log(fee_j + 1) + B_DUR x dur_j
```

**Interpretation:**
- Log transformation implies proportional sensitivity
- 10% fee increase has same impact regardless of absolute level
- Better fit if respondents process fees proportionally

---

### M3: Quadratic Fee

**Purpose:** Test non-linear fee sensitivity

```
V_j = ASC_paid + B_FEE x fee_j + B_FEE_SQ x fee_j^2 + B_DUR x dur_j
```

**Interpretation:**
- `B_FEE_SQ > 0`: Increasing marginal disutility (sensitivity increases with fee)
- `B_FEE_SQ < 0`: Decreasing marginal disutility (saturation)

---

### M4: Piecewise Fee

**Purpose:** Test threshold effects in fee sensitivity

```
fee_low = min(fee, threshold)
fee_high = max(0, fee - threshold)

V_j = ASC_paid + B_FEE_LOW x fee_low_j + B_FEE_HIGH x fee_high_j + B_DUR x dur_j
```

**Interpretation:**
- Different slopes below/above threshold (median fee)
- Tests if sensitivity changes at certain fee levels

---

### M5: Full Demographics

**Purpose:** All demographic interactions with fee

```
B_FEE_i = B_FEE + B_FEE_AGE x age_c + B_FEE_EDU x edu_c + B_FEE_INC x inc_c

V_j = ASC_paid + B_FEE_i x fee_j + B_DUR x dur_j
```

**Variables:** `_c` suffix indicates centered (mean-subtracted)

---

### M6: Cross Demographics

**Purpose:** Test demographic interaction effects

```
B_FEE_i = B_FEE + B_FEE_AGE x age_c + B_FEE_INC x inc_c + B_FEE_AGExINC x (age_c x inc_c)

V_j = ASC_paid + B_FEE_i x fee_j + B_DUR x dur_j
```

**Interpretation:** Tests if age effect on fee sensitivity varies by income

---

### M7: Log Fee + Demographics

**Purpose:** Combined functional form and demographics

```
B_FEE_LOG_i = B_FEE_LOG + B_FEE_LOG_AGE x age_c + B_FEE_LOG_INC x inc_c

V_j = ASC_paid + B_FEE_LOG_i x log(fee_j + 1) + B_DUR x dur_j
```

---

### M8: ASC Demographics

**Purpose:** Test heterogeneous preferences for paid alternatives

```
ASC_i = ASC_paid + ASC_AGE x age_c + ASC_INC x inc_c

V1 = ASC_i + B_FEE x fee1 + B_DUR x dur1
V2 = ASC_i + B_FEE x fee2 + B_DUR x dur2
V3 = B_FEE x fee3 + B_DUR x dur3
```

**Interpretation:** Tests if base preference for paid options varies by demographics

---

## MXL Models

### File: `src/models/mxl_models.py`

### Model: Mixed Logit with Random Parameters

**Purpose:** Capture unobserved taste heterogeneity

**Specification:**
```
B_FEE_i = B_FEE + sigma_FEE x eta_i,FEE    where eta ~ N(0,1)
B_DUR_i = B_DUR + sigma_DUR x eta_i,DUR    where eta ~ N(0,1)

V1 = ASC_paid + B_FEE_i x fee1 + B_DUR_i x dur1
V2 = ASC_paid + B_FEE_i x fee2 + B_DUR_i x dur2
V3 = B_FEE_i x fee3 + B_DUR_i x dur3
```

**Parameters:**
| Parameter | Description |
|-----------|-------------|
| B_FEE | Mean fee coefficient |
| sigma_FEE | Standard deviation of fee coefficient |
| B_DUR | Mean duration coefficient |
| sigma_DUR | Standard deviation of duration coefficient |

**Estimation:** Simulated Maximum Likelihood with Halton draws

---

## MXL Extended Models

### File: `src/models/mxl_extended.py`

Eight specifications testing distribution assumptions and heterogeneity sources.

### M1: Random Fee (Normal)

```
B_FEE_i ~ N(B_FEE, sigma_FEE^2)

V_j = ASC_paid + B_FEE_i x fee_j + B_DUR x dur_j
```

**Note:** Some individuals may have positive fee coefficient

---

### M2: Random Fee + Duration (Normal)

```
B_FEE_i ~ N(B_FEE, sigma_FEE^2)
B_DUR_i ~ N(B_DUR, sigma_DUR^2)

V_j = ASC_paid + B_FEE_i x fee_j + B_DUR_i x dur_j
```

---

### M3: Lognormal Fee

**Purpose:** Ensure fee coefficient always negative

```
B_FEE_i = -exp(mu_FEE + sigma_FEE x eta_i)    where eta ~ N(0,1)

V_j = ASC_paid + B_FEE_i x fee_j + B_DUR x dur_j
```

**Interpretation:**
- `B_FEE_i < 0` for all individuals (guaranteed)
- Heterogeneity in magnitude, not sign
- Use when economic theory requires negative sign

---

### M4: Lognormal Fee + Duration

```
B_FEE_i = -exp(mu_FEE + sigma_FEE x eta_i)
B_DUR_i = -exp(mu_DUR + sigma_DUR x eta_i)
```

---

### M5: Uniform Fee

**Purpose:** Test bounded heterogeneity

```
B_FEE_i ~ Uniform(B_FEE - spread, B_FEE + spread)

V_j = ASC_paid + B_FEE_i x fee_j + B_DUR x dur_j
```

**Interpretation:** All individuals within bounded range

---

### M6: Random ASC

**Purpose:** Test alternative-specific preference heterogeneity

```
ASC_i ~ N(ASC_paid, sigma_ASC^2)

V1 = ASC_i + B_FEE x fee1 + B_DUR x dur1
V2 = ASC_i + B_FEE x fee2 + B_DUR x dur2
V3 = B_FEE x fee3 + B_DUR x dur3
```

---

### M7: Log Fee Random

**Purpose:** Random parameters on transformed scale

```
B_FEE_LOG_i ~ N(B_FEE_LOG, sigma_FEE_LOG^2)

V_j = ASC_paid + B_FEE_LOG_i x log(fee_j + 1) + B_DUR x dur_j
```

---

### M8: Demographic Shifters

**Purpose:** Heterogeneity in mean varies by demographics

```
B_FEE_mean_i = B_FEE + B_FEE_AGE x age_c + B_FEE_INC x inc_c
B_FEE_i ~ N(B_FEE_mean_i, sigma_FEE^2)

V_j = ASC_paid + B_FEE_i x fee_j + B_DUR x dur_j
```

**Interpretation:** Systematic and random heterogeneity combined

---

## HCM/ICLV Models

### File: `src/models/hcm_split_latents.py`

The Hybrid Choice Model (HCM), also known as Integrated Choice and Latent Variable (ICLV), integrates psychological constructs into the choice model.

### Framework

```
+-------------------------------------------------------------+
|                    STRUCTURAL EQUATIONS                      |
+-------------------------------------------------------------+
|                                                              |
|  Latent Variables:                                           |
|  +--------------+      +--------------+                      |
|  | LV_pat_blind |      | LV_pat_const |  (Patriotism Domain) |
|  +--------------+      +--------------+                      |
|                                                              |
|  +--------------+      +--------------+                      |
|  | LV_sec_dl    |      | LV_sec_fp    |  (Secularism Domain) |
|  +--------------+      +--------------+                      |
|                                                              |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                    MEASUREMENT EQUATIONS                     |
+-------------------------------------------------------------+
|                                                              |
|  LV_pat_blind -> pat_blind_1, pat_blind_2, ..., pat_blind_5  |
|  LV_pat_const -> pat_const_1, pat_const_2, ..., pat_const_5  |
|  LV_sec_dl    -> sec_dl_1, sec_dl_2, ..., sec_dl_5           |
|  LV_sec_fp    -> sec_fp_1, sec_fp_2, ..., sec_fp_5           |
|                                                              |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                    CHOICE MODEL                              |
+-------------------------------------------------------------+
|                                                              |
|  B_FEE_i = B_FEE + sum(B_FEE_LV x LV)                        |
|  B_DUR_i = B_DUR + sum(B_DUR_LV x LV)                        |
|                                                              |
|  V_j = ASC + B_FEE_i x fee_j + B_DUR_i x dur_j               |
|                                                              |
+-------------------------------------------------------------+
```

---

### Section A: Individual LV Effects

#### M0: Baseline MNL (No LVs)

**Purpose:** Reference model without latent variables

```python
V1 = ASC_paid + B_FEE x fee1 + B_DUR x dur1
V2 = ASC_paid + B_FEE x fee2 + B_DUR x dur2
V3 = B_FEE x fee3 + B_DUR x dur3
```

**Parameters:** 3 (ASC_paid, B_FEE, B_DUR)

---

#### M1: Blind Patriotism

**Purpose:** Test if blind patriotism affects fee sensitivity

**Hypothesis:** Blind patriots are more willing to pay for military service -> less fee-sensitive

```python
B_FEE_i = B_FEE + B_FEE_PatBlind x LV_pat_blind

V1 = ASC_paid + B_FEE_i x fee1 + B_DUR x dur1
V2 = ASC_paid + B_FEE_i x fee2 + B_DUR x dur2
V3 = B_FEE_i x fee3 + B_DUR x dur3
```

**Interpretation of B_FEE_PatBlind:**
- If **negative**: Higher blind patriotism -> MORE fee-sensitive
- If **positive**: Higher blind patriotism -> LESS fee-sensitive

---

#### M2: Constructive Patriotism

**Purpose:** Test if constructive patriotism affects fee sensitivity

**Hypothesis:** Constructive patriots value service differently than blind patriots

```python
B_FEE_i = B_FEE + B_FEE_PatConst x LV_pat_const

V1 = ASC_paid + B_FEE_i x fee1 + B_DUR x dur1
V2 = ASC_paid + B_FEE_i x fee2 + B_DUR x dur2
V3 = B_FEE_i x fee3 + B_DUR x dur3
```

---

#### M3: Daily Life Secularism

**Purpose:** Test if daily life secularism affects fee sensitivity

```python
B_FEE_i = B_FEE + B_FEE_SecDL x LV_sec_dl
```

---

#### M4: Faith & Prayer Secularism

**Purpose:** Test if faith-based secularism affects fee sensitivity

```python
B_FEE_i = B_FEE + B_FEE_SecFP x LV_sec_fp
```

---

### Section B: Domain Combinations

#### M5: Both Patriotism Types

**Purpose:** Joint effect of blind and constructive patriotism

```python
B_FEE_i = B_FEE + B_FEE_PatBlind x LV_pat_blind + B_FEE_PatConst x LV_pat_const
```

**Tests:** Do the two patriotism types have different or complementary effects?

---

#### M6: Both Secularism Types

**Purpose:** Joint effect of daily life and faith secularism

```python
B_FEE_i = B_FEE + B_FEE_SecDL x LV_sec_dl + B_FEE_SecFP x LV_sec_fp
```

---

#### M7: Cross-Domain (PatBlind + SecDL)

**Purpose:** Test cross-domain LV interactions

```python
B_FEE_i = B_FEE + B_FEE_PatBlind x LV_pat_blind + B_FEE_SecDL x LV_sec_dl
```

---

#### M8: All 4 LVs on Fee

**Purpose:** Full specification with all LVs affecting fee

```python
B_FEE_i = B_FEE + B_FEE_PB x LV_pat_blind + B_FEE_PC x LV_pat_const
                + B_FEE_SDL x LV_sec_dl + B_FEE_SFP x LV_sec_fp
```

---

## HCM Extended Models

### File: `src/models/hcm_extended.py`

Eight specifications testing LV configurations and interactions.

### M1: PatBlind on Fee

```
B_FEE_i = B_FEE + B_FEE_PatBlind x LV_pat_blind

V_j = ASC_paid + B_FEE_i x fee_j + B_DUR x dur_j
```

**Purpose:** Test single LV effect on price sensitivity

---

### M2: SecDL on Fee

```
B_FEE_i = B_FEE + B_FEE_SecDL x LV_sec_dl

V_j = ASC_paid + B_FEE_i x fee_j + B_DUR x dur_j
```

**Purpose:** Test alternative LV domain

---

### M3: LVs on Duration

```
B_DUR_i = B_DUR + B_DUR_PatBlind x LV_pat_blind + B_DUR_SecDL x LV_sec_dl

V_j = ASC_paid + B_FEE x fee_j + B_DUR_i x dur_j
```

**Purpose:** Test if LVs affect time sensitivity

---

### M4: LVs on ASC

```
ASC_i = ASC_paid + ASC_PatBlind x LV_pat_blind + ASC_SecDL x LV_sec_dl

V1 = ASC_i + B_FEE x fee1 + B_DUR x dur1
V2 = ASC_i + B_FEE x fee2 + B_DUR x dur2
V3 = B_FEE x fee3 + B_DUR x dur3
```

**Purpose:** Test if LVs affect base preference for paid alternatives

---

### M5: Quadratic LV

```
B_FEE_i = B_FEE + B_FEE_PatBlind x LV_pat_blind + B_FEE_PatBlind_SQ x LV_pat_blind^2

V_j = ASC_paid + B_FEE_i x fee_j + B_DUR x dur_j
```

**Purpose:** Test non-linear LV effects

**Interpretation:**
- `B_FEE_PatBlind_SQ > 0`: Effect increases at extremes
- `B_FEE_PatBlind_SQ < 0`: Diminishing effect at extremes

---

### M6: LV x Demographics

```
B_FEE_i = B_FEE + B_FEE_PatBlind x LV_pat_blind + B_FEE_PatBlindxAge x (LV_pat_blind x age_c)

V_j = ASC_paid + B_FEE_i x fee_j + B_DUR x dur_j
```

**Purpose:** Test if LV effect varies by demographics

---

### M7: Domain Separation

```
B_FEE_i = B_FEE + B_FEE_PatBlind x LV_pat_blind
B_DUR_i = B_DUR + B_DUR_SecDL x LV_sec_dl

V_j = ASC_paid + B_FEE_i x fee_j + B_DUR_i x dur_j
```

**Purpose:** Test theoretically-motivated domain separation
- Patriotism -> Fee sensitivity (willingness to pay)
- Secularism -> Duration sensitivity (time preference)

---

### M8: Full Specification

```
B_FEE_i = B_FEE + B_FEE_PatBlind x LV_pat_blind + B_FEE_SecDL x LV_sec_dl
B_DUR_i = B_DUR + B_DUR_PatBlind x LV_pat_blind + B_DUR_SecDL x LV_sec_dl

V_j = ASC_paid + B_FEE_i x fee_j + B_DUR_i x dur_j
```

**Parameters:** 7 (ASC, B_FEE, B_DUR, 4 LV interactions)

**Note:** May have identification issues with small samples

---

## ICLV Simultaneous Estimation

### File: `src/models/iclv/`

The ICLV module provides simultaneous estimation of measurement, structural, and choice models.

### Model Components

**1. Measurement Model (Ordered Probit)**

For each Likert item k:
```
P(I_k = j | eta) = Phi(tau_j - lambda_k x eta) - Phi(tau_{j-1} - lambda_k x eta)
```

**2. Structural Model**

```
eta_m = gamma_m0 + sum_p gamma_mp x X_p + zeta_m
```

**3. Choice Model**

```
V_j = ASC_j + sum_a beta_a x attr_aj + sum_m beta_{LV,m} x eta_m
```

### Estimation

Simulated Maximum Likelihood (SML):
```
L_hat_n = (1/R) sum_r [ P(y_n|eta_r) x prod_k P(I_nk|eta_r) ]
```

### Usage

```python
from src.models.iclv import estimate_iclv

result = estimate_iclv(
    df=data,
    constructs={
        'pat_blind': ['pat_blind_1', 'pat_blind_2', 'pat_blind_3'],
        'sec_dl': ['sec_dl_1', 'sec_dl_2', 'sec_dl_3']
    },
    covariates=['age_c', 'income_c'],
    choice_col='CHOICE',
    attribute_cols=['fee1', 'fee2', 'fee3', 'dur1', 'dur2', 'dur3'],
    lv_effects={
        'pat_blind': 'beta_patriotism',
        'sec_dl': 'beta_secularism'
    },
    n_draws=500,
    auto_scale=True,
    use_two_stage_start=True,
    use_panel=True,
    compute_robust_se=True
)
```

### Enhanced Features

| Feature | Parameter | Description |
|---------|-----------|-------------|
| Auto-scaling | `auto_scale=True` | Scales large fee values |
| Two-stage start | `use_two_stage_start=True` | Better initialization |
| Panel support | `use_panel=True` | Cluster-robust SE |
| Robust SE | `compute_robust_se=True` | Sandwich estimator |
| LV correlation | `estimate_lv_correlation=True` | Estimate correlations |

### Comparison Tools

```python
from src.models.iclv import compare_two_stage_vs_iclv, summarize_attenuation_bias

comparison = compare_two_stage_vs_iclv(hcm_results, iclv_result, true_values)
summary = summarize_attenuation_bias(comparison)

print(f"Mean attenuation: {summary['mean_attenuation_%']:.1f}%")
```

---

## Latent Variable Estimation

### Confirmatory Factor Analysis (CFA) Approach

The project uses a weighted sum score approach for LV estimation:

```python
def estimate_lv(items):
    # 1. Calculate item-total correlations
    total = items.sum(axis=1)
    weights = []
    for item in items.columns:
        corr = correlation(item, total)
        weights.append(max(0.1, corr))  # Minimum weight of 0.1

    # 2. Normalize weights
    weights = weights / sum(weights)

    # 3. Compute weighted sum
    score = (items * weights).sum(axis=1)

    # 4. Standardize to N(0,1)
    score = (score - mean(score)) / std(score)

    return score
```

### LV Correlations

Typical correlations between constructs:

| | pat_blind | pat_const | sec_dl | sec_fp |
|---|---|---|---|---|
| pat_blind | 1.00 | 0.47 | 0.01 | -0.14 |
| pat_const | 0.47 | 1.00 | 0.14 | -0.11 |
| sec_dl | 0.01 | 0.14 | 1.00 | 0.41 |
| sec_fp | -0.14 | -0.11 | 0.41 | 1.00 |

---

## Model Comparison Framework

### Fit Statistics

| Statistic | Formula | Best Model |
|-----------|---------|------------|
| Log-Likelihood | LL = sum ln(P_i) | Highest |
| AIC | 2K - 2LL | Lowest |
| BIC | K x ln(n) - 2LL | Lowest |
| Rho-squared | 1 - LL/LL_0 | Highest (0-1) |
| Adj. Rho-squared | 1 - (LL-K)/LL_0 | Highest |

### Likelihood Ratio Test

For nested models:

```
LR = 2 x (LL_full - LL_restricted)
df = K_full - K_restricted
p-value from chi^2 distribution
```

### Model Selection Strategy

1. **Compare within family** (MNL vs MNL, HCM vs HCM)
2. **Use AIC for prediction**, BIC for explanation
3. **Check parameter signs** match theory
4. **Verify statistical significance** (t > 1.96)
5. **Test for convergence** issues

### Extended Model Selection

| Comparison | Method | Criteria |
|------------|--------|----------|
| MNL M1 vs M2-M8 | LR test (some nested) | p < 0.05 |
| MXL M1 vs M3 | Non-nested | AIC/BIC |
| HCM M1 vs M7 | Non-nested | Theory + AIC |
| HCM vs ICLV | Non-nested | Attenuation correction |

---

## Parameter Bounds Reference

| Parameter Type | Lower | Upper | Rationale |
|----------------|-------|-------|-----------|
| B_FEE | -10 | 0 | Fee always disutility |
| B_DUR | -5 | 0 | Duration always disutility |
| ASC | -10 | 10 | Can be positive or negative |
| LV on Fee | -2 | 2 | Moderate interaction |
| LV on Duration | -1 | 1 | Smaller scale |
| LV on ASC | -5 | 5 | Larger range |
| Sigma (MXL) | 0.001 | 5 | Must be positive |
| Lambda (loading) | 0.1 | 3 | Must be positive |
| Tau (threshold) | -5 | 5 | Ordered |

---

## Best Practices

1. **Start simple**: Begin with baseline MNL
2. **Add complexity gradually**: One LV at a time
3. **Check identification**: Monitor Hessian eigenvalues
4. **Use bounds**: Prevent extreme values
5. **Compare nested models**: Use LR tests
6. **Report robust SEs**: Use sandwich estimator
7. **Test functional forms**: Before adding LVs
8. **Compare two-stage vs ICLV**: Quantify attenuation

---

## Summary: 32+ Model Specifications

| Family | Models | Purpose |
|--------|--------|---------|
| MNL Basic | 3 | Baseline, demographics, Likert |
| MNL Extended | 8 | Functional forms |
| MXL Basic | 2 | Random parameters |
| MXL Extended | 8 | Distribution specifications |
| HCM Basic | 14 | Individual and combined LVs |
| HCM Extended | 8 | LV configurations |
| ICLV | 1+ | Simultaneous estimation |

**Total: 32+ specifications** covering the full hierarchy from simple MNL to unbiased ICLV.
