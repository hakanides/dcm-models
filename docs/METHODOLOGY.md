# Methodological Notes

This document describes important methodological considerations for the DCM estimation framework.

---

## Table of Contents

1. [Measurement Error in HCM](#measurement-error-in-hybrid-choice-models-hcm)
2. [ICLV: Integrated Choice and Latent Variable Models](#iclv-integrated-choice-and-latent-variable-models)
3. [Enhanced ICLV Features](#enhanced-iclv-features)
4. [Fee Scaling](#fee-scaling)
5. [Panel Data Structure](#panel-data-structure)
6. [Parameter Bounds](#parameter-bounds)
7. [Model Selection](#model-selection)
8. [Random Coefficient Sign Enforcement](#random-coefficient-sign-enforcement)

---

## Measurement Error in Hybrid Choice Models (HCM)

### The Problem

This implementation uses a **two-stage approach** for HCM estimation:

1. **Stage 1**: Estimate latent variables from Likert items using CFA/weighted averages
2. **Stage 2**: Use estimated LVs as fixed regressors in the choice model

This approach treats estimated latent variables as **error-free**, which causes **attenuation bias**: LV effect estimates are systematically biased toward zero.

### Why Does This Happen?

When we use estimated LV scores instead of true LV values:

```
True model:     U = b0 + b1 x LV_true + e
Estimated:      U = b0 + b1 x LV_est + e

Where: LV_est = LV_true + measurement_error
```

The measurement error in `LV_est` attenuates the coefficient `b1`:

```
E[b1_hat] = b1 x reliability
```

Where `reliability = Var(LV_true) / Var(LV_est) < 1`

### Bias Magnitude

The attenuation depends on:

| Factor | Effect on Bias |
|--------|---------------|
| Number of Likert items | More items -> higher reliability -> less bias |
| Item loadings | Higher loadings -> higher reliability -> less bias |
| True effect size | Smaller effects more affected (proportionally) |
| Sample size | Does NOT fix bias, only improves precision |

**Typical reliability** for 4-5 item Likert scales: 0.70-0.85

**Expected attenuation**: LV coefficients underestimated by 15-30%

### Interpretation Guidelines

1. **If LV effects are significant**: They are likely truly significant (conservative test)
2. **If LV effects are NOT significant**: They may still exist (attenuated)
3. **Effect magnitudes**: Underestimated; use for relative comparisons only
4. **Sign and direction**: Generally preserved despite attenuation

### Alternative: Full ICLV Estimation

For unbiased estimates, use Integrated Choice and Latent Variable (ICLV) models with **simultaneous estimation**:

- Measurement model: Likert items -> Latent variables
- Structural model: Demographics -> Latent variables
- Choice model: LVs + attributes -> Utility

### References

- Walker, J., & Ben-Akiva, M. (2002). Generalized random utility model. Mathematical Social Sciences.
- Vij, A., & Walker, J. (2016). How, when and why integrated choice and latent variable models are latently useful. Transportation Research Part B.

---

## ICLV: Integrated Choice and Latent Variable Models

### Overview

The ICLV framework provides **unbiased estimates** by simultaneously estimating:
1. **Measurement Model**: Likert items -> Latent variables
2. **Structural Model**: Demographics -> Latent variables
3. **Choice Model**: Attributes + LVs -> Utility

### Mathematical Formulation

The integrated likelihood for individual n is:

```
L_n = integral P(y_n|eta) x P(I_n|eta) x f(eta|X_n) d_eta
```

Where:
- `y_n` = choice outcome
- `I_n` = Likert indicator responses
- `eta` = latent variable vector
- `X_n` = demographic covariates
- `f(eta|X_n)` = conditional distribution of LVs given demographics

### Simulated Maximum Likelihood (SML)

Since the integral is analytically intractable, we use Monte Carlo simulation:

```
L_hat_n = (1/R) sum_r [ P(y_n|eta_r) x prod_k P(I_nk|eta_r) ]
```

Where:
- `R` = number of simulation draws (typically 500-1000)
- `eta_r` = draw r from the latent variable distribution
- Halton sequences provide more efficient coverage than pseudo-random draws

### Model Components

**1. Measurement Model (Ordered Probit)**

For each Likert item k with J response categories:

```
P(I_k = j | eta) = Phi(tau_j - lambda_k x eta) - Phi(tau_{j-1} - lambda_k x eta)
```

Where:
- `lambda_k` = factor loading for item k
- `tau_j` = threshold parameters
- `Phi()` = standard normal CDF

**2. Structural Model**

Latent variables depend on demographics:

```
eta_m = gamma_m0 + sum_p gamma_mp x X_p + zeta_m
```

Where:
- `gamma_mp` = effect of demographic p on latent m
- `zeta_m ~ N(0, sigma^2_m)` = residual variance

**3. Choice Model**

Utility includes latent variable effects:

```
V_j = ASC_j + sum_a beta_a x attr_aj + sum_m beta_{LV,m} x eta_m
```

### Comparison: Two-Stage vs ICLV

| Aspect | Two-Stage HCM | ICLV |
|--------|---------------|------|
| **Bias** | Attenuated (15-30%) | Unbiased |
| **Standard Errors** | Underestimated | Correct |
| **Computation** | Fast | Slow (simulation) |
| **Implementation** | Simple | Complex |
| **Software** | Any | Specialized |

### When to Use ICLV

**Use ICLV when:**
- Unbiased effect magnitudes are important
- Comparing LV effects across models/studies
- Publication requires correct standard errors
- Measurement quality is moderate (alpha < 0.80)

**Two-stage HCM is acceptable when:**
- Testing significance (conservative)
- Measurement quality is high (alpha > 0.85)
- Computational resources are limited
- Exploratory analysis

### Implementation in This Framework

The ICLV module (`src/models/iclv/`) provides:

```python
from src.models.iclv import estimate_iclv

result = estimate_iclv(
    df=data,
    constructs={'pat_blind': ['pb1', 'pb2', 'pb3', 'pb4']},
    covariates=['age_idx', 'edu_idx'],
    choice_col='CHOICE',
    attribute_cols=['fee1_10k', 'dur1'],
    lv_effects={'pat_blind': 'B_FEE_PatBlind'},
    n_draws=500
)
```

### Identification Requirements

ICLV models require:
1. **At least 3 indicators per construct** (for measurement model)
2. **At least 1 covariate per LV** (for structural identification)
3. **Scale normalization** (fix one loading per construct or fix variance)
4. **Sufficient sample size** (N > 300 typically)

### References

- Ben-Akiva, M., et al. (2002). Hybrid Choice Models: Progress and Challenges. Marketing Letters.
- Vij, A., & Walker, J. (2016). How, when and why integrated choice and latent variable models are latently useful. Transportation Research Part B.
- Daly, A., et al. (2012). A synthesis of analytical results on integration of latent variables in choice models. Transportation.

---

## Enhanced ICLV Features

The ICLV module includes seven major enhancements for publication-quality estimation:

### 1. Auto-Scaling

**Problem**: Large fee values (e.g., 5 million TL) cause numerical instability.

**Solution**: Automatic detection and scaling of large-valued attributes:

```python
result = estimate_iclv(df, ..., auto_scale=True)
```

The module:
- Detects if max(|fee|) > 1000
- Computes optimal scale factor (nearest power of 10)
- Scales attributes before estimation
- Reports original-scale coefficients

### 2. Two-Stage Starting Values

**Problem**: Poor starting values lead to local optima or non-convergence.

**Solution**: Initialize from two-stage estimates:

```python
result = estimate_iclv(df, ..., use_two_stage_start=True)
```

Process:
1. Run factor analysis on Likert items -> LV scores
2. Run logit with LV scores -> choice coefficients
3. Use as starting values for ICLV

### 3. Panel Data Support

**Problem**: Standard errors underestimated when ignoring within-individual correlation.

**Solution**: Panel-aware estimation:

```python
result = estimate_iclv(df, ..., use_panel=True)
```

Implementation:
- Groups observations by individual ID
- Uses same draws across choice tasks for each individual
- Computes cluster-robust standard errors

### 4. Robust Standard Errors

**Problem**: Standard errors rely on correct model specification.

**Solution**: Sandwich (Huber-White) estimator:

```python
result = estimate_iclv(df, ..., compute_robust_se=True)
```

Formula:
```
V_robust = H^{-1} (sum_i g_i g_i') H^{-1}
```

Where:
- `H` = Hessian (expected information)
- `g_i` = score contribution from individual i

### 5. LV Correlation Estimation

**Problem**: Assuming uncorrelated LVs may be unrealistic.

**Solution**: Estimate LV correlations:

```python
result = estimate_iclv(df, ..., estimate_lv_correlation=True)
```

Reports correlation matrix between constructs, useful for:
- Model diagnostics
- Understanding LV relationships
- Identifying potential multicollinearity

### 6. Analytical Gradients

**Problem**: Numerical gradients are slow and less accurate.

**Solution**: Implemented analytical gradients for ordered probit measurement model:

```python
# In measurement.py
gradient_loading()   # d log P / d lambda
gradient_threshold() # d log P / d tau
gradient_lv()        # d log P / d eta
```

Benefits:
- 2-3x faster estimation
- Better convergence properties
- More accurate standard errors

### 7. Comparison Tools

**Problem**: Difficult to quantify attenuation bias.

**Solution**: Built-in comparison functions:

```python
from src.models.iclv import compare_two_stage_vs_iclv, summarize_attenuation_bias

comparison = compare_two_stage_vs_iclv(hcm_results, iclv_result, true_values)
summary = summarize_attenuation_bias(comparison)

print(f"Mean attenuation: {summary['mean_attenuation_%']:.1f}%")
print(f"ICLV RMSE improvement: {summary['rmse_improvement_%']:.1f}%")
```

### Configuration Options

Full configuration via `EstimationConfig`:

```python
from src.models.iclv import EstimationConfig

config = EstimationConfig(
    n_draws=500,
    draw_type='halton',
    optimizer='BFGS',
    max_iter=1000,
    auto_scale=True,
    use_two_stage_start=True,
    use_panel=True,
    compute_robust_se=True,
    estimate_lv_correlation=True
)

result = estimate_iclv(df, ..., config=config)
```

---

## Fee Scaling

### Standard: `fee_scale = 10,000`

All fees are divided by 10,000 before estimation:

```python
fee_scaled = fee_raw / 10000.0
```

**Rationale**:
- Raw fees range 0 to ~5.6 million TL
- Scaled fees range 0 to ~560
- With coefficient ~-0.008, utility contribution = -0.008 x 70 = -0.56 (reasonable)

### Coefficient Interpretation

| Coefficient | Meaning |
|-------------|---------|
| B_FEE = -0.008 | 10,000 TL increase -> -0.008 utility |
| B_FEE = -0.008 | 100,000 TL increase -> -0.08 utility |

**Willingness-to-Pay (WTP)**:
```
WTP for 1 month less duration = -B_DUR / B_FEE x 10,000 TL
```

### Auto-Scaling in ICLV

When `auto_scale=True`, the ICLV module:
1. Detects max attribute magnitude
2. Computes scale factor as `10^floor(log10(max))`
3. Scales during estimation
4. Reports `ScalingInfo` with original-scale conversions

---

## Panel Data Structure

### Why Panel Matters

Each respondent answers T choice tasks. Observations within a respondent are **correlated** due to:
- Persistent taste preferences
- Learning effects
- Fatigue/attention

### Implementation

**MNL models**: Can ignore panel (treats observations as independent)
- Yields consistent but inefficient estimates
- Standard errors may be underestimated

**MXL models**: Must account for panel structure
```python
database.panel('ID')  # Declare panel structure
PanelLikelihoodTrajectory(prob)  # Same draws across tasks
```

**ICLV models**: Panel support via `use_panel=True`
- Same LV draws used across tasks for each individual
- Cluster-robust SE by individual

### Warm-Start Strategy

Models are estimated in sequence, using previous estimates as starting values:

```
MNL-Basic -> MNL-Demographics -> MNL-Full
     |
  Baseline -> MXL models
     |
  Baseline -> HCM models
     |
  Two-stage -> ICLV
```

Benefits:
- Faster convergence
- Better local optima avoidance
- More stable estimates

---

## Parameter Bounds

### Standard Bounds

| Parameter Type | Lower | Upper | Rationale |
|----------------|-------|-------|-----------|
| B_FEE | -10 | 0 | Fee always negative (disutility) |
| B_DUR | -5 | 0 | Duration always negative |
| ASC | -10 | 10 | Can be positive or negative |
| LV on Fee | -2 | 2 | Moderate interaction |
| LV on Duration | -1 | 1 | Smaller scale |
| Sigma (MXL) | 0.001 | 5 | Must be positive |

### Why Use Bounds?

1. **Economic theory**: Fee/duration should be negative
2. **Numerical stability**: Prevents extreme values during optimization
3. **Identification**: Helps with multicollinearity

---

## Model Selection

### Fit Statistics

| Statistic | Formula | Use |
|-----------|---------|-----|
| AIC | 2K - 2LL | Prediction (less penalty) |
| BIC | K x ln(n) - 2LL | Explanation (more penalty) |
| rho^2 | 1 - LL/LL_0 | Overall fit (0-1) |

### Nested Model Testing

For nested models, use Likelihood Ratio test:

```
LR = 2 x (LL_full - LL_restricted)
df = K_full - K_restricted
p-value from chi^2 distribution
```

### Non-Nested Comparison

For non-nested models (MNL vs MXL), use:
- AIC/BIC comparison
- Out-of-sample prediction (cross-validation)
- Theoretical considerations

### Model Progression Strategy

1. Start with baseline MNL
2. Add demographics (test significance)
3. Test functional forms (log, quadratic, piecewise)
4. Add random parameters (MXL)
5. Test distribution specifications
6. Add latent variables (HCM)
7. Compare two-stage vs ICLV
8. Select final model based on AIC/BIC + theory

---

## Random Coefficient Sign Enforcement

### The Issue

When simulating mixed logit data with random coefficients, some coefficients should theoretically always be negative (e.g., cost, time). However, with normal distributions, some individuals may get positive values.

### Current Implementation

Sign enforcement is applied to the **systematic mean** only (before adding random variation):

```python
# Systematic part: beta_sys = base + interactions
# Random part: eta ~ N(0, sigma^2)
# Combined: beta = beta_sys + eta

# Sign enforcement on beta_sys (correct)
if enforce_sign == 'negative':
    beta_sys = -abs(beta_sys)

# beta = beta_sys + eta allows sign flipping if |eta| > |beta_sys|
```

### Why Not Enforce After?

Enforcing signs after adding random variation (e.g., `beta = -abs(beta_sys + eta)`) truncates the distribution and biases heterogeneity estimates.

### Alternative: Lognormal Distributions

For strictly-signed coefficients, use lognormal distributions:

```
# For beta < 0: beta = -exp(mu + sigma*eta)
# For beta > 0: beta = exp(mu + sigma*eta)
```

This ensures correct signs while preserving the full distribution shape.

### Practical Guidance

1. If few individuals have "wrong" signs (< 5%), normal distribution is acceptable
2. For policy analysis requiring strict signs, use lognormal
3. Check sign-flip rate in simulation output

### MXL Extended Models

The extended MXL module tests multiple distributions:

| Model | Distribution | Sign Behavior |
|-------|-------------|---------------|
| M1 | Normal | May flip |
| M3 | Lognormal | Always negative |
| M5 | Uniform | Bounded, may flip |

---

## Extended Model Specifications

### Why Extended Models?

The framework includes 8 specifications for each model family to test:
1. **Functional forms**: Linear, log, quadratic, piecewise
2. **Heterogeneity sources**: Demographics, random parameters, LVs
3. **Distribution assumptions**: Normal, lognormal, uniform

### MNL Extended (8 Models)

| Model | Tests |
|-------|-------|
| M1 | Baseline - reference |
| M2 | Diminishing sensitivity (log fee) |
| M3 | Non-linear effect (quadratic) |
| M4 | Threshold effects (piecewise) |
| M5 | Demographic heterogeneity |
| M6 | Cross-demographic interactions |
| M7 | Combined functional form + demo |
| M8 | Heterogeneous ASC |

### MXL Extended (8 Models)

| Model | Tests |
|-------|-------|
| M1 | Fee heterogeneity |
| M2 | Multi-attribute heterogeneity |
| M3 | Sign-constrained (lognormal) |
| M4 | All lognormal |
| M5 | Bounded heterogeneity (uniform) |
| M6 | ASC heterogeneity |
| M7 | Transformed scale |
| M8 | Heterogeneity in mean |

### HCM Extended (8 Models)

| Model | Tests |
|-------|-------|
| M1 | Single LV effect (PatBlind) |
| M2 | Alternative LV (SecDL) |
| M3 | LVs on different attribute |
| M4 | LVs on ASC |
| M5 | Non-linear LV effect |
| M6 | LV-demographic interaction |
| M7 | Domain separation |
| M8 | Full specification |

### Selection Criteria

1. **Within family**: LR test for nested, AIC/BIC for non-nested
2. **Across families**: Focus on theoretical interpretation
3. **Parsimony**: Prefer simpler models unless gain is substantial
4. **Stability**: Check parameter stability across specifications

---

## Validation with Simulated Data

### Monte Carlo Validation

With simulated data, the framework validates estimation by comparing estimates to true (known) values:

```python
# True parameters from config
true_params = config['choice_model']['parameters']

# Estimated parameters from model
est_params = result.estimates

# Compute metrics
bias = est_params - true_params
pct_bias = 100 * bias / true_params
rmse = sqrt(mean(bias^2))
coverage = sum(in_ci) / n_params
```

### Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| Bias | estimate - true | Near 0 |
| % Bias | 100 x bias / true | < 10% |
| RMSE | sqrt(mean(bias^2)) | Low |
| Coverage | % true in 95% CI | ~95% |

### Attenuation Bias Quantification

For HCM vs ICLV comparison:

```python
# Attenuation = 1 - (HCM estimate / ICLV estimate)
attenuation_pct = 100 * (1 - hcm_coef / iclv_coef)

# Expected: 15-30% for typical measurement quality
```

---

## References

### Core Methodology
- Train, K. (2009). Discrete Choice Methods with Simulation. Cambridge University Press.
- Ben-Akiva, M., & Lerman, S. (1985). Discrete Choice Analysis. MIT Press.

### Hybrid Choice Models
- Ben-Akiva, M., et al. (2002). Hybrid Choice Models: Progress and Challenges. Marketing Letters.
- Walker, J., & Ben-Akiva, M. (2002). Generalized random utility model. Mathematical Social Sciences.

### ICLV
- Vij, A., & Walker, J. (2016). How, when and why integrated choice and latent variable models are latently useful. Transportation Research Part B.
- Daly, A., et al. (2012). A synthesis of analytical results on integration of latent variables in choice models. Transportation.

### Simulation
- Halton, J. (1960). On the efficiency of certain quasi-random sequences. Numerische Mathematik.
- Train, K. (2000). Halton sequences for mixed logit. Working Paper.

### Robust Inference
- White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator. Econometrica.
- Wooldridge, J. (2010). Econometric Analysis of Cross Section and Panel Data. MIT Press.
