# Issue: Likelihood ratio test produces negative values (incorrect sign/nesting)

## Summary
MNL comparison output shows negative LR statistics (e.g., "LR = -2.81, p=1.0"), which indicates either incorrect nesting direction, swapped LL values, or comparison of non-converged models.

## Steps to Reproduce
1. Run MNL comparison:
   ```bash
   python src/models/mnl_model_comparison.py --output results/mnl/latest
   ```
2. Check `likelihood_ratio_tests.csv` or console output
3. Observe: Negative LR statistic values

## Expected Behavior
LR statistic = 2 * (LL_unrestricted - LL_restricted) should be >= 0 when:
- Unrestricted model nests the restricted model
- Both models are converged
- LL values are correctly extracted

## Actual Behavior
```
Model 1 vs Model 2 LR = -2.81, p=1.0
```
Negative LR indicates something is wrong.

## Root Cause Analysis
Possible causes:
1. **Swapped nesting direction**: Restricted/unrestricted models are reversed
2. **Non-converged models**: LL values are not at MLE, so standard LR doesn't apply
3. **Incorrect LL extraction**: Reading wrong field from Biogeme results
4. **Non-nested comparison**: Comparing models that don't have nesting relationship

## Suggested Fix
**1. Verify nesting direction:**
```python
def lr_test(restricted_ll, unrestricted_ll, df_diff):
    """
    restricted_ll: LL of nested (fewer params) model
    unrestricted_ll: LL of full (more params) model
    df_diff: difference in degrees of freedom (should be > 0)
    """
    lr_stat = 2 * (unrestricted_ll - restricted_ll)
    if lr_stat < 0:
        warnings.warn(
            f"Negative LR ({lr_stat:.2f}): check model nesting or convergence"
        )
        return None, None
    p_value = 1 - chi2.cdf(lr_stat, df_diff)
    return lr_stat, p_value
```

**2. Only compare converged models:**
```python
def compare_models(model1, model2):
    if not model1.converged or not model2.converged:
        return {"error": "Cannot compare non-converged models"}
    # ... proceed with LR test
```

**3. Validate nesting relationship:**
```python
def validate_nesting(restricted_params, unrestricted_params):
    """Check that restricted params are subset of unrestricted"""
    if not set(restricted_params).issubset(set(unrestricted_params)):
        raise ValueError("Models are not nested")
```

**4. Add diagnostic output:**
```python
print(f"Restricted model: {restricted_name}")
print(f"  LL = {restricted_ll:.4f}, params = {n_restricted}")
print(f"Unrestricted model: {unrestricted_name}")
print(f"  LL = {unrestricted_ll:.4f}, params = {n_unrestricted}")
print(f"LR = 2 * ({unrestricted_ll:.4f} - {restricted_ll:.4f}) = {lr_stat:.4f}")
```

## Acceptance Criteria
- [ ] LR test validates convergence before computing
- [ ] LR test validates nesting relationship
- [ ] Negative LR triggers warning and returns null/NA
- [ ] Diagnostic output shows LL values used

## Affected Files
- `src/models/mnl_model_comparison.py`
- `src/analysis/final_comparison.py`

## Labels
`bug`, `statistics`, `correctness`, `high-priority`
