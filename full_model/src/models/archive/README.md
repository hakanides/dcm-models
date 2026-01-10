# Archived Code

This directory contains deprecated code that has been archived rather than deleted.
These files are kept for reference but should NOT be used in active development.

## hcm_split_latents.py

**Archived:** 2026-01-10
**Reason:** Uses two-stage estimation approach which causes 15-50% ATTENUATION BIAS

The two-stage HCM approach:
1. First estimates latent variable scores from indicators
2. Then uses these as fixed explanatory variables in the choice model

This causes attenuation bias because:
- Measurement error in LV scores is ignored
- The LV effects are biased toward zero
- Standard errors are underestimated

**Correct Approach:** Use simultaneous ICLV estimation where LVs and choice model
are estimated jointly via maximum simulated likelihood. See:
- models/hcm_full/model.py
- models/hcm_basic/model.py
- models/iclv/model.py

References:
- Ben-Akiva et al. (2002): "Hybrid Choice Models"
- Raveau et al. (2010): "Sequential vs Simultaneous Estimation"
- Walker & Ben-Akiva (2002): "Generalized Random Utility Model"
