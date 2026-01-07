# Issue: Fee magnitudes cause numerical instability (overflow/underflow in exp)

## Status: RESOLVED

The fee scaling has been standardized to `fee_scale = 10,000` across all code.

## Summary
Fee values reach ~5.6 million TL, which with typical scaling and coefficient values creates utility differences of hundreds of units, causing exp() overflow/underflow and numerical instability.

## Solution Implemented

**Standardized scaling: `fee_scale = 10,000`**

The key insight is that the FEE_SCALE and coefficient magnitude must be jointly calibrated:
- With `fee_scale = 10,000`, a 700,000 TL fee becomes `70` after scaling
- With `base_coef = -0.008`, utility contribution = `-0.008 * 70 = -0.56` (reasonable)

### Files Updated
All files now consistently use `fee_scale = 10,000`:

1. **model_config.json**: `"fee_scale": 10000.0` with `"base_coef": -0.008`
2. **run_all_models.py**: `FEE_SCALE = 10000.0`
3. **src/simulation/simulate_full_data.py**: default `10000.0`
4. **src/utils/data_qa.py**: default `10000.0`
5. **src/models/*.py**: All use `/ 10000.0`

### Coefficient Scaling
When fee_scale changed from 1M to 10k (100x smaller scale → 100x larger scaled values):
- `base_coef` changed from `-0.8` to `-0.008` (100x smaller)
- All fee interaction coefficients scaled down 100x accordingly

## Verification
After the fix:
- Choice shares: ~46%, 46%, 8% (reasonable balance)
- B_FEE recovery: True=-0.008, Est=-0.0069 (14% bias, acceptable)
- All models converge without numerical issues

## Acceptance Criteria
- [x] Typical utility differences are in range [-20, 20]
- [x] No exp overflow/underflow warnings during estimation
- [x] Consistent scaling between simulation and estimation code

## Labels
`resolved`, `numerical-stability`, `convergence`
