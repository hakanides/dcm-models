# Archived Model Files

These files have been superseded by newer implementations and are kept here for reference only.

## Do Not Use in Production

| File | Status | Replacement |
|------|--------|-------------|
| `hcm_model.py` | SUPERSEDED | Use `hcm_split_latents.py` |
| `hcm_model_improved.py` | SUPERSEDED | Use `hcm_split_latents.py` |
| `mnl_model_comparison_v2.py` | SUPERSEDED | Use `mnl_model_comparison.py` |

## Why Archived

- **hcm_model.py**: Basic two-stage HCM implementation. Lacks robust parameter bounds and comprehensive model specifications.

- **hcm_model_improved.py**: Intermediate implementation with CFA improvements. Superseded by `hcm_split_latents.py` which has better bounds, more model variants, and cleaner structure.

- **mnl_model_comparison_v2.py**: Alternative MNL implementation without sign constraints. The main `mnl_model_comparison.py` is more complete and well-tested.

## Current Canonical Files

Use these files for production:

- `hcm_split_latents.py` - Canonical HCM with 15 model specifications
- `mnl_model_comparison.py` - Comprehensive MNL comparison
- `mxl_models.py` - Mixed Logit models
- `validation_models.py` - Simulation validation (true LV benchmarks)
