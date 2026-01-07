# Issue: Status quo alternative (alt3) has fully constant attributes, amplifying convergence issues

## Summary
Alternative 3 (status quo/"free" option) has constant attributes (`fee3=0`, `dur3=24`, `exempt3=0`), which combined with 91% choice share makes it mechanically dominant and pushes optimizer toward extreme parameters.

## Steps to Reproduce
1. Load synthetic data:
   ```python
   import pandas as pd
   df = pd.read_csv('data/test_small_sample.csv')
   print(df[['fee3', 'dur3', 'exempt3']].nunique())
   # All = 1 (constant)
   print(df[['fee3', 'dur3', 'exempt3']].iloc[0])
   # fee3=0, dur3=24, exempt3=0
   ```

## Expected Behavior
Status quo alternatives can have fixed attributes, but this should be balanced with:
- Reasonable choice shares across alternatives
- Proper ASC specification
- Consideration of opportunity costs

## Actual Behavior
- `fee3` = 0 (always free)
- `dur3` = 24 (always same duration)
- `exempt3` = 0 (never exempt)
- Combined with ~91% choice share, this creates a nearly deterministic outcome

## Root Cause Analysis
A "free" alternative with no variation:
1. Has maximum utility relative to paid alternatives
2. Is chosen almost deterministically when fee differences are large
3. Provides no information for estimating attribute coefficients for alt3
4. Forces all identification to come from the ~9% of choices selecting alt1/alt2

## Suggested Fix
**Option A: Add opportunity cost to status quo:**
```python
# In scenario generation
# Add implicit cost (time, inconvenience) to "free" alternative
dur3 = np.random.uniform(30, 48)  # Variable wait time
```

**Option B: Allow some attribute variation:**
```python
# Even small variation helps identification
fee3 = np.random.choice([0, 100, 500], p=[0.8, 0.15, 0.05])
```

**Option C: Rebalance paid alternatives to be more competitive:**
```python
# Reduce fee magnitudes so paid alternatives are viable
fee1_range = (100, 5000)  # More reasonable range
```

**Option D: Use ASC-only specification for status quo:**
```python
# Don't estimate coefficients on alt3 attributes (they're constant anyway)
V3 = ASC3  # No attribute terms since they're absorbed
```

## Affected Files
- `src/simulation/simulate_full_data.py`
- `src/simulation/prepare_scenarios.py`
- `src/models/*.py` (model specifications)

## Related Issues
- #3 (Choice imbalance)
- #4 (Fee scaling)

## Labels
`enhancement`, `data-generation`, `experimental-design`
