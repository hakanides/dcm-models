# Issue: Fee magnitudes cause numerical instability (overflow/underflow in exp)

## Summary
Fee values reach ~5.6 million, which with typical scaling and coefficient values creates utility differences of hundreds of units, causing exp() overflow/underflow and numerical instability.

## Steps to Reproduce
1. Generate or load synthetic data:
   ```python
   import pandas as pd
   df = pd.read_csv('data/test_small_sample.csv')
   print(df[['fee1', 'fee2', 'fee3']].describe())
   # fee1 max ~ 5.63e6
   # fee2 max ~ 5.57e6
   # fee3 = 0 (constant)
   ```
2. With scaling `fee/10000`, values become 0-560
3. With coefficient ~ -0.7, utility contribution = -0.7 * 560 = -392
4. Utility differences of hundreds cause `exp(V)` to saturate at 0 or overflow

## Expected Behavior
Utility differences should typically be in range [-10, 10] to avoid numerical issues with softmax/logit probabilities.

## Actual Behavior
- Fee ranges: 0 to ~5.6 million
- After `/10000` scaling: 0 to ~560
- Typical utility difference: hundreds of units
- `exp(large_negative)` -> 0, `exp(large_positive)` -> overflow
- Gradients vanish or explode -> optimization fails

## Suggested Fix
**Option A: Reduce fee magnitudes in data generation:**
```python
# In scenario generation, use more realistic/smaller fee ranges
fee_range = (0, 50000)  # instead of (0, 5_000_000)
```

**Option B: Increase scale factor consistently:**
```python
# In both simulation and estimation
FEE_SCALE = 1_000_000  # instead of 10_000
fee_scaled = fee / FEE_SCALE
```

**Option C: Reduce coefficient magnitudes:**
```python
# Use smaller true beta values in simulation
BETA_FEE = -0.01  # instead of -0.7
```

**Option D: Add scaling diagnostics:**
```python
def check_utility_scale(df, fee_cols, scale, beta_magnitude=1.0):
    for col in fee_cols:
        max_contrib = df[col].max() / scale * beta_magnitude
        if abs(max_contrib) > 20:
            warnings.warn(
                f"{col}: max utility contribution = {max_contrib:.1f}, "
                "may cause numerical issues"
            )
```

## Acceptance Criteria
- [ ] Typical utility differences are in range [-20, 20]
- [ ] No exp overflow/underflow warnings during estimation
- [ ] Consistent scaling between simulation and estimation code

## Affected Files
- `src/simulation/simulate_full_data.py`
- `src/simulation/prepare_scenarios.py`
- `src/simulation/dcm_simulator.py`
- `src/models/mnl_model_comparison.py`
- `src/models/mxl_models.py`
- `src/models/hcm_split_latents.py`

## Labels
`bug`, `numerical-stability`, `convergence`, `high-priority`
