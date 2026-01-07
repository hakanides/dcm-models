# Issue: `exempt` variables are constant by alternative, causing collinearity and non-identification

## Summary
In the generated dataset, `exempt1`, `exempt2`, `exempt3` are constant (nunique=1), making them equivalent to alternative labels. This causes perfect collinearity with ASCs and prevents parameter identification.

## Steps to Reproduce
1. Generate synthetic data:
   ```bash
   python src/simulation/simulate_full_data.py
   ```
2. Inspect exemption columns:
   ```python
   import pandas as pd
   df = pd.read_csv('data/test_small_sample.csv')
   print(df[['exempt1', 'exempt2', 'exempt3']].nunique())
   # Output: exempt1=1, exempt2=1, exempt3=1
   ```
3. Run estimation with exemption coefficient + ASCs
4. Observe: Biogeme warning "optimization algorithm did not converge"

## Expected Behavior
Exemption should vary across choice tasks/scenarios if it's being estimated as a coefficient.

## Actual Behavior
- `exempt1` always = 1
- `exempt2` always = 1
- `exempt3` always = 0

## Root Cause Analysis
When `exempt` is constant within each alternative:
- `exempt` becomes a perfect proxy for alternative identity
- ASC and B_EXEMPT become linearly dependent (singular Hessian)
- Optimizer cannot identify separate effects -> flat likelihood ridge -> non-convergence

This is especially problematic for MXL when attempting to estimate random coefficients on `exempt`.

## Suggested Fix
**Option A**: Remove exemption regressor from estimation (rely on ASC only):
```python
# In model specification, remove:
# V[alt] = ... + B_EXEMPT * exempt[alt]
```

**Option B**: Redesign scenario generation so exemption varies:
```python
# In simulate_full_data.py or prepare_scenarios.py
# Ensure exempt varies across scenarios, not just across alternatives
```

**Option C**: Add data validation check that fails if attribute is constant:
```python
def validate_attribute_variation(df, cols, min_unique=2):
    for col in cols:
        if df[col].nunique() < min_unique:
            raise ValueError(f"{col} has insufficient variation (nunique={df[col].nunique()})")
```

## Affected Files
- `src/simulation/simulate_full_data.py`
- `src/simulation/prepare_scenarios.py`
- `src/simulation/dcm_simulator.py`
- `src/models/mnl_model_comparison.py`
- `src/models/mxl_models.py`
- `src/models/hcm_split_latents.py`

## Labels
`bug`, `data-generation`, `convergence`, `identification`
