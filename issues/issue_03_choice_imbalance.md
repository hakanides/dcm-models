# Issue: Extreme choice imbalance (~91% alt3) causes quasi-separation and convergence failure

## Status: RESOLVED

**Resolution:** Fee scaling was standardized to 10,000 which fixed utility balance. Choice shares are now ~46%, 46%, 8% (see issue_04_fee_scaling.md). Data QA checks in `src/utils/data_qa.py` now validate choice shares and warn if any alternative exceeds 80% or falls below 2%.

---

## Summary (Historical)
Generated synthetic data shows extreme choice imbalance (alt3 chosen ~91% of the time), which causes quasi-complete separation in logit estimation and prevents convergence.

## Steps to Reproduce
1. Generate synthetic data:
   ```bash
   python src/simulation/simulate_full_data.py
   ```
2. Check choice distribution:
   ```python
   import pandas as pd
   df = pd.read_csv('data/test_small_sample.csv')
   print(df['CHOICE'].value_counts(normalize=True))
   # Output: 3 = 0.9134, 1 = 0.0456, 2 = 0.0410
   ```
3. Run any model estimation
4. Observe: Non-convergence warnings

## Expected Behavior
Choice shares should be reasonably balanced for model identification, typically in the range 0.2-0.5 per alternative for debugging datasets.

## Actual Behavior
- Alternative 3: **91.34%**
- Alternative 1: 4.56%
- Alternative 2: 4.10%

## Root Cause Analysis
When one alternative dominates:
1. Parameters for the dominant alternative try to approach infinity
2. Parameters for rare alternatives are poorly identified
3. Hessian becomes ill-conditioned
4. This mimics quasi-complete/perfect separation in binary logit

In synthetic data, this indicates the experimental design or utility scaling is miscalibrated.

## Suggested Fix
**1. Recalibrate utility parameters in simulator:**
```python
# In dcm_simulator.py, adjust true parameters so utilities are more balanced
# e.g., reduce magnitude of fee coefficient or adjust ASC values
```

**2. Add choice share validation:**
```python
def validate_choice_shares(df, choice_col='CHOICE', max_share=0.8):
    shares = df[choice_col].value_counts(normalize=True)
    for alt, share in shares.items():
        if share > max_share:
            raise ValueError(
                f"Alternative {alt} has share {share:.2%} > {max_share:.0%}. "
                "Consider recalibrating simulation parameters."
            )
```

**3. Add diagnostic output during data generation:**
```python
print("Choice share diagnostics:")
print(df['CHOICE'].value_counts(normalize=True).to_string())
```

## Acceptance Criteria
- [ ] No alternative has >80% choice share in generated test data
- [ ] Data generation script prints choice share diagnostics
- [ ] Validation check fails fast if shares are degenerate

## Affected Files
- `src/simulation/simulate_full_data.py`
- `src/simulation/dcm_simulator.py`
- `src/simulation/dcm_simulator_advanced.py`
- `src/utils/data_qa.py` (add validation)

## Labels
`bug`, `data-generation`, `convergence`, `high-priority`
