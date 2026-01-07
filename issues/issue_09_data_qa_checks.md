# Issue: Add comprehensive data QA checks that fail fast on problematic data

## Summary
Multiple data issues (constant attributes, extreme choice shares, scaling problems) cause downstream estimation failures. Need systematic QA checks that catch these early.

## Problem Statement
Currently, data problems are only discovered when estimation fails with cryptic "did not converge" warnings. This wastes time and makes debugging difficult.

## Proposed Solution
Implement a `data_qa.py` module with validation checks that run before estimation and fail fast with clear error messages.

## Specification

### 1. Attribute Variation Check
```python
def check_attribute_variation(df, attribute_cols, min_unique=2):
    """Ensure attributes vary sufficiently for identification."""
    issues = []
    for col in attribute_cols:
        nunique = df[col].nunique()
        if nunique < min_unique:
            issues.append(f"{col}: nunique={nunique} (need >= {min_unique})")
    if issues:
        raise DataQAError(f"Insufficient attribute variation:\n" + "\n".join(issues))
```

### 2. Choice Share Check
```python
def check_choice_shares(df, choice_col='CHOICE', max_share=0.8, min_share=0.02):
    """Ensure no alternative dominates or is never chosen."""
    shares = df[choice_col].value_counts(normalize=True)
    issues = []
    for alt, share in shares.items():
        if share > max_share:
            issues.append(f"Alt {alt}: {share:.1%} > {max_share:.0%} (too dominant)")
        if share < min_share:
            issues.append(f"Alt {alt}: {share:.1%} < {min_share:.0%} (too rare)")
    if issues:
        raise DataQAError(f"Choice share issues:\n" + "\n".join(issues))
```

### 3. Scaling Check
```python
def check_utility_scaling(df, attribute_cols, scales, max_contribution=20):
    """Warn if attribute ranges could cause numerical issues."""
    warnings = []
    for col in attribute_cols:
        scale = scales.get(col, 1.0)
        max_val = df[col].abs().max() / scale
        if max_val > max_contribution:
            warnings.append(
                f"{col}: max scaled value = {max_val:.1f} "
                f"(may cause exp overflow with typical coefficients)"
            )
    if warnings:
        print("Scaling warnings:\n" + "\n".join(warnings))
```

### 4. Collinearity Check
```python
def check_collinearity(df, attribute_cols, threshold=0.95):
    """Check for near-perfect correlation between attributes."""
    from scipy.stats import pearsonr
    issues = []
    for i, col1 in enumerate(attribute_cols):
        for col2 in attribute_cols[i+1:]:
            corr, _ = pearsonr(df[col1], df[col2])
            if abs(corr) > threshold:
                issues.append(f"{col1} vs {col2}: r={corr:.3f}")
    if issues:
        raise DataQAError(f"High collinearity detected:\n" + "\n".join(issues))
```

### 5. Integration
```python
def run_all_checks(df, config):
    """Run all QA checks before estimation."""
    print("Running data QA checks...")
    check_attribute_variation(df, config['attribute_cols'])
    check_choice_shares(df, config['choice_col'])
    check_utility_scaling(df, config['attribute_cols'], config['scales'])
    check_collinearity(df, config['attribute_cols'])
    print("All QA checks passed!")
```

## Usage in Model Scripts
```python
# At top of estimation script
from src.utils.data_qa import run_all_checks

df = pd.read_csv(data_path)
run_all_checks(df, {
    'attribute_cols': ['fee1', 'fee2', 'dur1', 'dur2', 'exempt1', 'exempt2'],
    'choice_col': 'CHOICE',
    'scales': {'fee1': 10000, 'fee2': 10000, 'dur1': 1, 'dur2': 1}
})
# Proceed with estimation only if checks pass
```

## Acceptance Criteria
- [ ] `src/utils/data_qa.py` implements all check functions
- [ ] Clear error messages indicate exactly what's wrong
- [ ] All model scripts call QA checks before estimation
- [ ] Checks are configurable (thresholds, columns to check)
- [ ] Option to warn vs. fail for each check type

## Affected Files
- `src/utils/data_qa.py` (new/enhanced)
- `src/models/mnl_model_comparison.py`
- `src/models/mxl_models.py`
- `src/models/hcm_split_latents.py`
- `src/simulation/simulate_full_data.py`

## Labels
`enhancement`, `data-validation`, `developer-experience`, `high-priority`
