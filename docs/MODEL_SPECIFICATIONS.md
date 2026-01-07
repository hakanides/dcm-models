# Model Specifications Guide

This document provides detailed specifications for all models implemented in this DCM research project.

---

## Table of Contents

1. [Theoretical Background](#theoretical-background)
2. [MNL Models](#mnl-models)
3. [MXL Models](#mxl-models)
4. [HCM/ICLV Models](#hcmiclv-models)
5. [Latent Variable Estimation](#latent-variable-estimation)
6. [Model Comparison Framework](#model-comparison-framework)

---

## Theoretical Background

### Random Utility Maximization (RUM)

All models are based on the RUM framework:

```
U_ij = V_ij + ε_ij
```

Where:
- `U_ij` = Total utility of alternative j for individual i
- `V_ij` = Systematic (observable) utility
- `ε_ij` = Random (unobservable) component

### Choice Probability

The probability that individual i chooses alternative j:

```
P(j|i) = exp(V_ij) / Σ_k exp(V_ik)
```

---

## MNL Models

### File: `src/models/mnl_model_comparison.py`

### Model 1: Baseline MNL

**Purpose:** Establish baseline with only choice attributes

**Specification:**
```
V1 = ASC_paid + B_FEE × fee1 + B_DUR × dur1
V2 = ASC_paid + B_FEE × fee2 + B_DUR × dur2
V3 = B_FEE × fee3 + B_DUR × dur3
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
B_FEE_i = B_FEE + B_FEE_AGE × age + B_FEE_EDU × edu + B_FEE_INC × income

V1 = ASC_paid + B_FEE_i × fee1 + B_DUR × dur1
V2 = ASC_paid + B_FEE_i × fee2 + B_DUR × dur2
V3 = B_FEE_i × fee3 + B_DUR × dur3
```

**Interpretation:**
- `B_FEE_AGE > 0`: Older people less fee-sensitive
- `B_FEE_INC > 0`: Higher income less fee-sensitive

---

### Model 3: MNL with Likert Interactions

**Purpose:** Test attitudinal effects using Likert scale averages

**Specification:**
```
B_FEE_i = B_FEE + B_FEE_PAT × pat_blind_idx + B_FEE_SEC × sec_dl_idx

V1 = ASC_paid + B_FEE_i × fee1 + B_DUR × dur1
V2 = ASC_paid + B_FEE_i × fee2 + B_DUR × dur2
V3 = B_FEE_i × fee3 + B_DUR × dur3
```

---

## MXL Models

### File: `src/models/mxl_models.py`

### Model: Mixed Logit with Random Parameters

**Purpose:** Capture unobserved taste heterogeneity

**Specification:**
```
B_FEE_i = B_FEE + σ_FEE × η_i,FEE    where η ~ N(0,1)
B_DUR_i = B_DUR + σ_DUR × η_i,DUR    where η ~ N(0,1)

V1 = ASC_paid + B_FEE_i × fee1 + B_DUR_i × dur1
V2 = ASC_paid + B_FEE_i × fee2 + B_DUR_i × dur2
V3 = B_FEE_i × fee3 + B_DUR_i × dur3
```

**Parameters:**
| Parameter | Description |
|-----------|-------------|
| B_FEE | Mean fee coefficient |
| σ_FEE | Standard deviation of fee coefficient |
| B_DUR | Mean duration coefficient |
| σ_DUR | Standard deviation of duration coefficient |

**Estimation:** Simulated Maximum Likelihood with Halton draws

---

## HCM/ICLV Models

### File: `src/models/hcm_split_latents.py`

The Hybrid Choice Model (HCM), also known as Integrated Choice and Latent Variable (ICLV), integrates psychological constructs into the choice model.

### Framework

```
┌─────────────────────────────────────────────────────────────────┐
│                    STRUCTURAL EQUATIONS                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Latent Variables:                                               │
│  ┌──────────────┐      ┌──────────────┐                         │
│  │ LV_pat_blind │      │ LV_pat_const │  (Patriotism Domain)    │
│  └──────────────┘      └──────────────┘                         │
│                                                                  │
│  ┌──────────────┐      ┌──────────────┐                         │
│  │ LV_sec_dl    │      │ LV_sec_fp    │  (Secularism Domain)    │
│  └──────────────┘      └──────────────┘                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    MEASUREMENT EQUATIONS                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  LV_pat_blind → pat_blind_1, pat_blind_2, ..., pat_blind_5      │
│  LV_pat_const → pat_const_1, pat_const_2, ..., pat_const_5      │
│  LV_sec_dl    → sec_dl_1, sec_dl_2, ..., sec_dl_5               │
│  LV_sec_fp    → sec_fp_1, sec_fp_2, ..., sec_fp_5               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    CHOICE MODEL                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  B_FEE_i = B_FEE + Σ(B_FEE_LV × LV)                             │
│  B_DUR_i = B_DUR + Σ(B_DUR_LV × LV)                             │
│                                                                  │
│  V_j = ASC + B_FEE_i × fee_j + B_DUR_i × dur_j                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

### Section A: Individual LV Effects

#### M0: Baseline MNL (No LVs)

**Purpose:** Reference model without latent variables

```python
V1 = ASC_paid + B_FEE × fee1 + B_DUR × dur1
V2 = ASC_paid + B_FEE × fee2 + B_DUR × dur2
V3 = B_FEE × fee3 + B_DUR × dur3
```

**Parameters:** 3 (ASC_paid, B_FEE, B_DUR)

---

#### M1: Blind Patriotism

**Purpose:** Test if blind patriotism affects fee sensitivity

**Hypothesis:** Blind patriots are more willing to pay for military service → less fee-sensitive

```python
B_FEE_i = B_FEE + B_FEE_PatBlind × LV_pat_blind

V1 = ASC_paid + B_FEE_i × fee1 + B_DUR × dur1
V2 = ASC_paid + B_FEE_i × fee2 + B_DUR × dur2
V3 = B_FEE_i × fee3 + B_DUR × dur3
```

**Interpretation of B_FEE_PatBlind:**
- If **negative**: Higher blind patriotism → MORE fee-sensitive
- If **positive**: Higher blind patriotism → LESS fee-sensitive

---

#### M2: Constructive Patriotism

**Purpose:** Test if constructive patriotism affects fee sensitivity

**Hypothesis:** Constructive patriots value service differently than blind patriots

```python
B_FEE_i = B_FEE + B_FEE_PatConst × LV_pat_const

V1 = ASC_paid + B_FEE_i × fee1 + B_DUR × dur1
V2 = ASC_paid + B_FEE_i × fee2 + B_DUR × dur2
V3 = B_FEE_i × fee3 + B_DUR × dur3
```

---

#### M3: Daily Life Secularism

**Purpose:** Test if daily life secularism affects fee sensitivity

```python
B_FEE_i = B_FEE + B_FEE_SecDL × LV_sec_dl
```

---

#### M4: Faith & Prayer Secularism

**Purpose:** Test if faith-based secularism affects fee sensitivity

```python
B_FEE_i = B_FEE + B_FEE_SecFP × LV_sec_fp
```

---

### Section B: Domain Combinations

#### M5: Both Patriotism Types

**Purpose:** Joint effect of blind and constructive patriotism

```python
B_FEE_i = B_FEE + B_FEE_PatBlind × LV_pat_blind + B_FEE_PatConst × LV_pat_const
```

**Tests:** Do the two patriotism types have different or complementary effects?

---

#### M6: Both Secularism Types

**Purpose:** Joint effect of daily life and faith secularism

```python
B_FEE_i = B_FEE + B_FEE_SecDL × LV_sec_dl + B_FEE_SecFP × LV_sec_fp
```

---

#### M7: Cross-Domain (PatBlind + SecDL)

**Purpose:** Test cross-domain LV interactions

```python
B_FEE_i = B_FEE + B_FEE_PatBlind × LV_pat_blind + B_FEE_SecDL × LV_sec_dl
```

---

#### M8: All 4 LVs on Fee

**Purpose:** Full specification with all LVs affecting fee

```python
B_FEE_i = B_FEE + B_FEE_PB × LV_pat_blind + B_FEE_PC × LV_pat_const
                + B_FEE_SDL × LV_sec_dl + B_FEE_SFP × LV_sec_fp
```

---

### Section C: Attribute-Specific Effects

#### M9: LVs on Duration

**Purpose:** Test if LVs affect duration (time) sensitivity

```python
B_DUR_i = B_DUR + B_DUR_PatBlind × LV_pat_blind + B_DUR_SecDL × LV_sec_dl
```

**Hypothesis:** Patriots may tolerate longer service duration

---

#### M10: LVs on ASC

**Purpose:** Test if LVs affect base preference for paid options

```python
ASC_i = ASC_paid + B_ASC_PatBlind × LV_pat_blind + B_ASC_SecDL × LV_sec_dl

V1 = ASC_i + B_FEE × fee1 + B_DUR × dur1
V2 = ASC_i + B_FEE × fee2 + B_DUR × dur2
V3 = B_FEE × fee3 + B_DUR × dur3
```

**Note:** This replaces M10 (LVs on Exemption) since exemption is not identifiable.

---

#### M11: LVs on Fee + Duration

**Purpose:** Full attribute interaction

```python
B_FEE_i = B_FEE + B_FEE_PB × LV_pat_blind + B_FEE_SDL × LV_sec_dl
B_DUR_i = B_DUR + B_DUR_PB × LV_pat_blind + B_DUR_SDL × LV_sec_dl
```

---

### Section D: Full Models

#### M12: Full Patriotism

**Purpose:** Both patriotism types on both fee and duration

```python
B_FEE_i = B_FEE + B_FEE_PB × LV_pat_blind + B_FEE_PC × LV_pat_const
B_DUR_i = B_DUR + B_DUR_PB × LV_pat_blind + B_DUR_PC × LV_pat_const
```

**Parameters:** 7 (ASC, B_FEE, B_DUR, 4 LV interactions)

---

#### M13: Full Secularism

**Purpose:** Both secularism types on both fee and duration

```python
B_FEE_i = B_FEE + B_FEE_SDL × LV_sec_dl + B_FEE_SFP × LV_sec_fp
B_DUR_i = B_DUR + B_DUR_SDL × LV_sec_dl + B_DUR_SFP × LV_sec_fp
```

---

#### M14: Full Model (All LVs)

**Purpose:** All 4 LVs on both fee and duration

```python
B_FEE_i = B_FEE + B_FEE_PB × PB + B_FEE_PC × PC + B_FEE_SDL × SDL + B_FEE_SFP × SFP
B_DUR_i = B_DUR + B_DUR_PB × PB + B_DUR_PC × PC + B_DUR_SDL × SDL + B_DUR_SFP × SFP
```

**Parameters:** 11 (ASC, B_FEE, B_DUR, 8 LV interactions)

**Warning:** High parameter count may cause identification issues with small samples.

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
| Log-Likelihood | LL = Σ ln(P_i) | Highest |
| AIC | 2K - 2LL | Lowest |
| BIC | K×ln(n) - 2LL | Lowest |
| Rho-squared | 1 - LL/LL₀ | Highest (0-1) |
| Adj. Rho-squared | 1 - (LL-K)/LL₀ | Highest |

### Likelihood Ratio Test

For nested models:

```
LR = 2 × (LL_full - LL_restricted)
df = K_full - K_restricted
p-value from χ² distribution
```

### Model Selection Strategy

1. **Compare within family** (MNL vs MNL, HCM vs HCM)
2. **Use AIC for prediction**, BIC for explanation
3. **Check parameter signs** match theory
4. **Verify statistical significance** (t > 1.96)
5. **Test for convergence** issues

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

---

## Best Practices

1. **Start simple**: Begin with baseline MNL
2. **Add complexity gradually**: One LV at a time
3. **Check identification**: Monitor Hessian eigenvalues
4. **Use bounds**: Prevent extreme values
5. **Compare nested models**: Use LR tests
6. **Report robust SEs**: Use sandwich estimator
