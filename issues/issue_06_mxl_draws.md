# Issue: MXL uses too few simulation draws, causing noisy likelihood and instability

## Summary
MXL estimation uses 500 draws, which Biogeme warns is low. Combined with non-converged MNL starting values, this leads to unreliable results.

## Steps to Reproduce
1. Run MXL estimation:
   ```bash
   python src/models/mxl_models.py --output results/mxl/latest
   ```
2. Observe warning in output:
   ```
   draws=500 "low, results may not be meaningful"
   optimization algorithm did not converge
   ```

## Expected Behavior
- MXL should use sufficient draws (2000-10000) for stable likelihood approximation
- Starting values should come from a converged MNL model
- Convergence status should be logged and used for downstream filtering

## Actual Behavior
- Only 500 draws used
- Biogeme warning about insufficient draws
- Non-convergence warnings
- Results may not represent MLE

## Root Cause Analysis
1. **Simulation noise**: With too few draws, the simulated log-likelihood is noisy
2. **Unstable gradients**: Noisy LL -> noisy gradients -> optimizer struggles
3. **Bad starting values**: Using non-converged MNL as starting point propagates problems
4. **Compounding issues**: Data problems (issues #2-5) make this worse

## Suggested Fix
**1. Increase number of draws:**
```python
# In mxl_models.py
N_DRAWS = 2000  # Minimum for debugging
# N_DRAWS = 10000  # For final results
```

**2. Use converged MNL starting values:**
```python
def get_starting_values(mnl_results_path):
    results = pd.read_csv(mnl_results_path)
    # Filter to converged models only
    converged = results[results['converged'] == True]
    if converged.empty:
        raise ValueError("No converged MNL model available for starting values")
    return converged.iloc[0]['parameters']
```

**3. Log convergence status:**
```python
results_df['converged'] = [r.convergence for r in biogeme_results]
results_df['optimizer_message'] = [r.optimization_messages for r in biogeme_results]
```

**4. Staged estimation:**
```python
# Stage 1: Estimate MNL, verify convergence
# Stage 2: Use MNL params as MXL starting values
# Stage 3: Run MXL with increasing draws until stable
```

## Acceptance Criteria
- [ ] MXL uses >= 2000 draws by default
- [ ] Starting values come from verified-converged MNL
- [ ] Convergence status is saved in output CSV
- [ ] Non-converged models are flagged in comparisons

## Affected Files
- `src/models/mxl_models.py`
- `src/analysis/final_comparison.py` (filter by convergence)

## Labels
`bug`, `convergence`, `MXL`, `medium-priority`
