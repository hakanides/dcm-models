# Methodological Notes

This document describes important methodological considerations for the DCM estimation framework.

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
True model:     U = β₀ + β₁ × LV_true + ε
Estimated:      U = β₀ + β₁ × LV_est + ε

Where: LV_est = LV_true + measurement_error
```

The measurement error in `LV_est` attenuates the coefficient `β₁`:

```
E[β̂₁] = β₁ × reliability
```

Where `reliability = Var(LV_true) / Var(LV_est) < 1`

### Bias Magnitude

The attenuation depends on:

| Factor | Effect on Bias |
|--------|---------------|
| Number of Likert items | More items → higher reliability → less bias |
| Item loadings | Higher loadings → higher reliability → less bias |
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

- Measurement model: Likert items → Latent variables
- Structural model: Demographics → Latent variables
- Choice model: LVs + attributes → Utility

Software options:
- **Apollo** (R): `apollo_estimate()` with `apollo_lcEM` or full ICLV
- **Biogeme**: Custom likelihood with integrated measurement model
- **Stata**: `gsem` for generalized SEM

### References

- Walker, J., & Ben-Akiva, M. (2002). Generalized random utility model. Mathematical Social Sciences.
- Vij, A., & Walker, J. (2016). How, when and why integrated choice and latent variable models are latently useful. Transportation Research Part B.

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
- With coefficient ~-0.008, utility contribution = -0.008 × 70 ≈ -0.56 (reasonable)

### Coefficient Interpretation

| Coefficient | Meaning |
|-------------|---------|
| B_FEE = -0.008 | 10,000 TL increase → -0.008 utility |
| B_FEE = -0.008 | 100,000 TL increase → -0.08 utility |

**Willingness-to-Pay (WTP)**:
```
WTP for 1 month less duration = -B_DUR / B_FEE × 10,000 TL
```

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

### Warm-Start Strategy

Models are estimated in sequence, using previous estimates as starting values:

```
MNL-Basic → MNL-Demographics → MNL-Full
     ↓
  Baseline → MXL models
     ↓
  Baseline → HCM models
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
| BIC | K×ln(n) - 2LL | Explanation (more penalty) |
| ρ² | 1 - LL/LL₀ | Overall fit (0-1) |

### Nested Model Testing

For nested models, use Likelihood Ratio test:

```
LR = 2 × (LL_full - LL_restricted)
df = K_full - K_restricted
p-value from χ² distribution
```

### Non-Nested Comparison

For non-nested models (MNL vs MXL), use:
- AIC/BIC comparison
- Out-of-sample prediction (cross-validation)
- Theoretical considerations

---

## Random Coefficient Sign Enforcement

### The Issue

When simulating mixed logit data with random coefficients, some coefficients should theoretically always be negative (e.g., cost, time). However, with normal distributions, some individuals may get positive values.

### Current Implementation

Sign enforcement is applied to the **systematic mean** only (before adding random variation):

```python
# Systematic part: β_sys = base + interactions
# Random part: η ~ N(0, σ²)
# Combined: β = β_sys + η

# Sign enforcement on β_sys (correct)
if enforce_sign == 'negative':
    β_sys = -abs(β_sys)

# β = β_sys + η allows sign flipping if |η| > |β_sys|
```

### Why Not Enforce After?

Enforcing signs after adding random variation (e.g., `β = -abs(β_sys + η)`) truncates the distribution and biases heterogeneity estimates.

### Alternative: Lognormal Distributions

For strictly-signed coefficients, use lognormal distributions:

```
# For β < 0: β = -exp(μ + σ*η)
# For β > 0: β = exp(μ + σ*η)
```

This ensures correct signs while preserving the full distribution shape.

### Practical Guidance

1. If few individuals have "wrong" signs (< 5%), normal distribution is acceptable
2. For policy analysis requiring strict signs, use lognormal
3. Check sign-flip rate in simulation output
