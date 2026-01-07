# Issue: `final_comparison.py` cannot find model outputs due to hard-coded paths

## Summary
`final_comparison.py` uses hard-coded paths that don't match the actual output directories created by model scripts, causing "No results found!" errors.

## Steps to Reproduce
1. Run the model scripts:
   ```bash
   python src/models/hcm_split_latents.py --output results/hcm/latest
   python src/models/mnl_model_comparison.py --output results/mnl/latest
   python src/models/mxl_models.py --output results/mxl/latest
   ```
2. Run the final comparison:
   ```bash
   python src/analysis/final_comparison.py
   ```
3. Observe error: **"No results found!"**

## Expected Behavior
`final_comparison.py` should automatically locate outputs in `results/*/latest/` or accept configurable paths.

## Actual Behavior
Script searches hard-coded paths:
- `mnl_small_results/model_comparison.csv`
- `mxl_small_results/mxl_comparison.csv`
- `hcm_improved_results/hcm_comparison.csv`

But actual outputs are in:
- `results/mnl/latest/model_comparison.csv`
- `results/mxl/latest/mxl_comparison.csv`
- `results/hcm/latest/hcm_comparison.csv` (or `results/hcm/hcm_improved_results/`)

## Workaround
Create symlinks manually:
```bash
ln -s results/mnl/latest mnl_small_results
ln -s results/mxl/latest mxl_small_results
ln -s results/hcm/hcm_improved_results hcm_improved_results
```

## Suggested Fix
Option A: Add CLI arguments for result locations:
```python
parser.add_argument('--mnl-results', default='results/mnl/latest')
parser.add_argument('--mxl-results', default='results/mxl/latest')
parser.add_argument('--hcm-results', default='results/hcm/latest')
```

Option B: Auto-discover results using glob patterns:
```python
mnl_paths = glob.glob('results/mnl/*/model_comparison.csv')
```

Option C: Standardize all scripts to write to a canonical structure (e.g., `results/{model_type}/latest/`).

## Affected Files
- `src/analysis/final_comparison.py`

## Labels
`bug`, `pipeline`, `usability`
