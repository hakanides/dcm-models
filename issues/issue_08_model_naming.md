# Issue: Models lack explicit names, defaulting to "biogeme_model_default_name"

## Summary
Biogeme prints warnings about undefined model names, making logs and artifacts harder to audit and trace.

## Steps to Reproduce
1. Run any model estimation
2. Observe Biogeme output:
   ```
   You have not defined a name for the model... default [biogeme_model_default_name]
   ```

## Expected Behavior
Each model should have an explicit, descriptive name that appears in:
- Console output
- Log files
- Result CSVs
- Saved model artifacts

## Actual Behavior
All models use default name "biogeme_model_default_name", making it difficult to:
- Distinguish between model runs
- Trace results back to specific configurations
- Debug issues in multi-model comparisons

## Suggested Fix
Set explicit model names in each estimation script:

```python
# In mnl_model_comparison.py
biogeme = bio.BIOGEME(database, logprob, modelName='MNL_baseline')

# In mxl_models.py
biogeme = bio.BIOGEME(database, logprob, modelName='MXL_random_fee')

# In hcm_split_latents.py
biogeme = bio.BIOGEME(database, logprob, modelName='HCM_environmental')
```

Use descriptive names that indicate:
- Model family (MNL, MXL, HCM)
- Key specification differences
- Version/variant if applicable

## Acceptance Criteria
- [ ] All model scripts set explicit `modelName` parameter
- [ ] Model names are descriptive and unique
- [ ] Model names appear in output CSVs for traceability

## Affected Files
- `src/models/mnl_model_comparison.py`
- `src/models/mnl_model_comparison_v2.py`
- `src/models/mxl_models.py`
- `src/models/hcm_model.py`
- `src/models/hcm_model_improved.py`
- `src/models/hcm_split_latents.py`

## Labels
`enhancement`, `usability`, `low-priority`
