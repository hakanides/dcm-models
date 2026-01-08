# Discrete Choice Modeling (DCM) Research Framework

A publication-ready Python framework for discrete choice model estimation with integrated data simulation for validation. Implements the full hierarchy from MNL to ICLV with 32+ model specifications.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Model Hierarchy](#model-hierarchy)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [For Real Data](#for-real-data)
- [API Reference](#api-reference)
- [Methodology Notes](#methodology-notes)

---

## Overview

This framework implements a complete DCM research pipeline designed for academic publication:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SIMULATION MODULE                             │
│  config/model_config_advanced.json → src/simulation/                │
│  • Demographics (age, education, income)                            │
│  • Latent variables (patriotism, secularism constructs)            │
│  • Likert scale responses (5-point ordered probit)                 │
│  • Choice behavior (soft utility maximization)                      │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│                        ESTIMATION MODULE                             │
│  scripts/run_all_models.py → MNL → MXL → HCM → ICLV                │
│  • MNL: Baseline + demographics + extended (8 specs)               │
│  • MXL: Random parameters + extended (8 specs)                     │
│  • HCM: Two-stage latent variables + extended (8 specs)            │
│  • ICLV: Simultaneous estimation (unbiased)                        │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│                     VALIDATION & OUTPUT                              │
│  • Compare estimates to TRUE parameters (simulation validation)     │
│  • Model comparison: AIC, BIC, likelihood ratio tests              │
│  • LaTeX tables for publication                                     │
│  • Parameter recovery metrics (bias, RMSE, coverage)               │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Features

- **32+ Model Specifications**: MNL, MXL, HCM, ICLV with multiple functional forms
- **Simulation-Based Validation**: Known true parameters for rigorous testing
- **Publication-Ready Output**: LaTeX tables, comparison metrics
- **Biogeme Integration**: Industry-standard estimation engine
- **Enhanced ICLV**: Auto-scaling, two-stage initialization, robust SE, panel support

---

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd DCM-Research

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Simulated Data

```bash
python src/simulation/simulate_full_data.py \
    --config config/model_config_advanced.json \
    --output data/simulated/ \
    --n_individuals 1000 \
    --n_tasks 10
```

### 3. Run All Models

```bash
python scripts/run_all_models.py
```

### 4. View Results

```bash
ls results/all_models/
ls output/latex/
```

---

## Project Structure

```
DCM-Research/
├── config/                              # Configuration files
│   ├── model_config_advanced.json       # Main config with true parameters
│   ├── model_config.json                # Simple config for testing
│   ├── items_config.csv                 # Likert scale item definitions
│   └── items_config_advanced.csv        # Advanced item definitions
│
├── src/
│   ├── models/                          # Model specifications
│   │   ├── mnl_basic.py                 # Basic MNL
│   │   ├── mnl_demographics.py          # MNL with demographics
│   │   ├── mnl_extended.py              # 8 MNL specifications
│   │   ├── mxl_basic.py                 # Basic Mixed Logit
│   │   ├── mxl_extended.py              # 8 MXL specifications
│   │   ├── hcm_basic.py                 # Basic HCM (single LV)
│   │   ├── hcm_full.py                  # Full HCM (all LVs)
│   │   ├── hcm_extended.py              # 8 HCM specifications
│   │   ├── hcm_split_latents.py         # Two-stage HCM
│   │   ├── iclv/                        # ICLV module (enhanced)
│   │   ├── model_factory.py             # Model registry
│   │   └── validation_models.py         # True LV benchmark
│   │
│   ├── simulation/                      # Data generation
│   │   ├── dcm_simulator.py             # Core simulator
│   │   ├── dcm_simulator_advanced.py    # With random coefficients
│   │   └── simulate_full_data.py        # CLI wrapper
│   │
│   ├── estimation/                      # Estimation utilities
│   │   ├── robust_estimation.py         # Convergence management
│   │   ├── convergence_diagnostics.py   # Convergence checking
│   │   ├── model_comparison.py          # LR tests, AIC/BIC
│   │   ├── measurement_validation.py    # Cronbach's α, AVE, CR
│   │   ├── bootstrap_inference.py       # Bootstrap CIs
│   │   └── cross_validation.py          # K-fold CV
│   │
│   ├── policy_analysis/                 # Policy applications
│   │   ├── wtp_analysis.py              # Willingness-to-pay
│   │   ├── elasticity.py                # Price/duration elasticities
│   │   └── demand_forecasting.py        # Market share prediction
│   │
│   ├── analysis/                        # Analysis tools
│   │   ├── final_comparison.py          # Cross-model comparison
│   │   └── sensitivity_analysis.py      # Sensitivity analysis
│   │
│   ├── validation/                      # Validation tools
│   │   └── monte_carlo.py               # Monte Carlo studies
│   │
│   └── utils/                           # Utilities
│       ├── latex_output.py              # LaTeX table generation
│       └── visualization.py             # Publication figures
│
├── scripts/
│   ├── run_all_models.py                # Main pipeline script
│   └── validate_estimation.py           # Validation script
│
├── data/
│   ├── raw/                             # Raw scenario definitions
│   │   └── scenarios_prepared.csv
│   └── simulated/                       # Generated simulation data
│
├── output/
│   ├── latex/                           # LaTeX tables by model family
│   │   ├── MNL/
│   │   ├── MXL/
│   │   ├── HCM/
│   │   └── simulation/
│   └── logs/                            # Log files
│
├── results/
│   └── all_models/                      # Model estimation results
│
├── tests/                               # Test suite
│
└── docs/                                # Documentation
    ├── METHODOLOGY.md
    ├── MODEL_SPECIFICATIONS.md
    └── QUICK_START.md
```

---

## Model Hierarchy

### Overview

| Family | Model | Parameters | Purpose |
|--------|-------|------------|---------|
| **MNL** | Basic | ASC, β_fee, β_dur | Baseline |
| | Demographics | + age, income interactions | Observed heterogeneity |
| | Extended (8) | Log, quadratic, piecewise | Functional forms |
| **MXL** | Basic | Random β_fee | Unobserved heterogeneity |
| | Extended (8) | Normal, lognormal, uniform | Distribution specifications |
| **HCM** | Basic | + 1 latent variable | Attitude effects |
| | Full | + 4 latent variables | Full attitude model |
| | Extended (8) | Domain separation, interactions | LV specifications |
| **ICLV** | Simultaneous | Joint estimation | Unbiased (no attenuation) |

### MNL Extended Models

| Model | Specification | Key Parameters |
|-------|--------------|----------------|
| M1 | Basic MNL | ASC_paid, B_FEE, B_DUR |
| M2 | Log Fee | B_FEE_LOG (diminishing sensitivity) |
| M3 | Quadratic Fee | B_FEE, B_FEE_SQ (non-linear) |
| M4 | Piecewise Fee | B_FEE_LOW, B_FEE_HIGH (threshold) |
| M5 | Full Demographics | Age, income, education interactions |
| M6 | Cross Demographics | Age×Income interactions |
| M7 | Log Fee + Demo | Combined specification |
| M8 | ASC Demographics | Heterogeneous ASC |

### MXL Extended Models

| Model | Specification | Distribution |
|-------|--------------|--------------|
| M1 | Random Fee | Normal |
| M2 | Random Fee+Dur | Normal (both) |
| M3 | Lognormal Fee | Always negative |
| M4 | Lognormal Both | Fee and duration |
| M5 | Uniform Fee | Bounded heterogeneity |
| M6 | Random ASC | Alternative-specific |
| M7 | Log Fee Random | Transformed |
| M8 | Demo Shifters | Heterogeneity in mean |

### HCM Extended Models

| Model | Specification | LV Configuration |
|-------|--------------|------------------|
| M1 | PatBlind on Fee | Single LV effect |
| M2 | SecDL on Fee | Different LV |
| M3 | LVs on Duration | Duration interaction |
| M4 | LVs on ASC | ASC interaction |
| M5 | Quadratic LV | Non-linear effect |
| M6 | LV × Demo | LV-demographic interaction |
| M7 | Domain Separation | Patriotism→Fee, Secularism→Dur |
| M8 | Full Specification | All LVs, all attributes |

---

## Usage Examples

### Basic MNL Estimation

```python
from src.models.mnl_basic import estimate_mnl_basic

result = estimate_mnl_basic(
    data_path='data/simulated/full_scale_test.csv',
    config_path='config/model_config_advanced.json',
    output_dir='results/all_models'
)
print(f"Log-likelihood: {result['log_likelihood']:.2f}")
print(f"AIC: {result['aic']:.2f}")
```

### Mixed Logit with Random Coefficients

```python
from src.models.mxl_extended import run_mxl_extended

results, comparison = run_mxl_extended(
    data_path='data/simulated/full_scale_test.csv',
    output_dir='results/all_models',
    n_draws=500
)
print(comparison.to_string())
```

### HCM with Latent Variables

```python
from src.models.hcm_extended import run_hcm_extended

results, comparison = run_hcm_extended(
    data_path='data/simulated/full_scale_test.csv',
    output_dir='results/all_models'
)
```

### ICLV Simultaneous Estimation

```python
from src.models.iclv import estimate_iclv

result = estimate_iclv(
    df=data,
    constructs={
        'pat_blind': ['pat_blind_1', 'pat_blind_2', 'pat_blind_3'],
        'sec_dl': ['sec_dl_1', 'sec_dl_2', 'sec_dl_3']
    },
    covariates=['age_c', 'income_c'],
    choice_col='CHOICE',
    attribute_cols=['fee1', 'fee2', 'fee3', 'dur1', 'dur2', 'dur3'],
    lv_effects={
        'pat_blind': 'beta_patriotism',
        'sec_dl': 'beta_secularism'
    },
    n_draws=500,
    auto_scale=True,      # Automatic fee scaling
    use_panel=True,       # Panel-corrected SE
    use_two_stage_start=True  # Better convergence
)

print(f"Log-likelihood: {result.log_likelihood:.2f}")
print(f"Converged: {result.convergence}")
```

### Run Full Pipeline

```python
# Run all 32+ models
python scripts/run_all_models.py

# Results saved to:
# - results/all_models/model_comparison.csv
# - results/all_models/parameter_estimates.csv
# - output/latex/MNL/, output/latex/MXL/, etc.
```

---

## Configuration

### model_config_advanced.json

```json
{
  "population": {
    "n_individuals": 1000,
    "n_choice_tasks": 10
  },
  "demographics": {
    "age_idx": {"distribution": "categorical", "probabilities": [0.2, 0.3, 0.3, 0.2]},
    "income_indiv_idx": {"distribution": "categorical", "probabilities": [0.25, 0.35, 0.25, 0.15]}
  },
  "latent_variables": {
    "pat_blind": {
      "structural": {"age_idx": 0.15, "income_indiv_idx": -0.10},
      "variance": 1.0
    }
  },
  "measurement_model": {
    "pat_blind_1": {"construct": "pat_blind", "loading": 1.0},
    "pat_blind_2": {"construct": "pat_blind", "loading": 0.85},
    "pat_blind_3": {"construct": "pat_blind", "loading": 0.78}
  },
  "choice_model": {
    "base_terms": [
      {"name": "ASC_paid", "coef": 2.0}
    ],
    "attribute_terms": [
      {"name": "b_fee10k", "attribute": "fee", "scale": 10000, "base_coef": -0.5},
      {"name": "b_dur", "attribute": "dur", "base_coef": -0.025}
    ],
    "lv_interactions": [
      {"name": "b_fee_pat_blind", "attribute": "fee", "lv": "pat_blind", "coef": 0.15}
    ]
  }
}
```

---

## For Real Data

When you have real survey data:

1. **Format Data**: Match the expected column structure
   ```
   ID, CHOICE, fee1, fee2, fee3, dur1, dur2, dur3,
   pat_blind_1, pat_blind_2, pat_blind_3, ...,
   age_idx, income_indiv_idx, ...
   ```

2. **Skip Simulation**: Don't run simulate_full_data.py

3. **Run Estimation**:
   ```bash
   python scripts/run_all_models.py --data path/to/your/data.csv
   ```

4. **Note**: True parameter comparison will be skipped automatically when config doesn't have true values defined.

---

## API Reference

### Models

| Module | Main Function | Description |
|--------|---------------|-------------|
| `mnl_basic` | `estimate_mnl_basic()` | Basic MNL estimation |
| `mnl_extended` | `run_mnl_extended()` | 8 MNL specifications |
| `mxl_basic` | `estimate_mxl_basic()` | Basic MXL estimation |
| `mxl_extended` | `run_mxl_extended()` | 8 MXL specifications |
| `hcm_basic` | `estimate_hcm_basic()` | Single LV HCM |
| `hcm_extended` | `run_hcm_extended()` | 8 HCM specifications |
| `iclv` | `estimate_iclv()` | ICLV simultaneous estimation |

### ICLV Enhanced Features

| Feature | Parameter | Description |
|---------|-----------|-------------|
| Auto-scaling | `auto_scale=True` | Scales large fee values automatically |
| Two-stage start | `use_two_stage_start=True` | Better starting values |
| Panel support | `use_panel=True` | Cluster-robust SE |
| Robust SE | `compute_robust_se=True` | Sandwich estimator |
| LV correlation | `estimate_lv_correlation=True` | Correlated LVs |

### Comparison Tools

```python
from src.models.iclv import compare_two_stage_vs_iclv, summarize_attenuation_bias

# Compare HCM (two-stage) vs ICLV (simultaneous)
comparison = compare_two_stage_vs_iclv(hcm_results, iclv_result, true_values)
summary = summarize_attenuation_bias(comparison)

print(f"Mean attenuation bias: {summary['mean_attenuation_%']:.1f}%")
print(f"ICLV RMSE improvement: {summary['rmse_improvement_%']:.1f}%")
```

---

## Methodology Notes

### Two-Stage HCM vs ICLV

The HCM models use a **two-stage approach**:
1. Estimate latent variables from Likert items (CFA/weighted average)
2. Use LV estimates as fixed regressors in choice model

This causes **attenuation bias** - LV effects are underestimated by ~15-30%.

The ICLV module provides **simultaneous estimation** that eliminates this bias by:
- Integrating over the LV distribution
- Jointly estimating measurement, structural, and choice models
- Using simulation (SML) to handle the integrals

### When to Use Each

| Model | Use When |
|-------|----------|
| MNL | Quick baseline, no heterogeneity |
| MXL | Taste heterogeneity matters |
| HCM (two-stage) | Fast estimation, large samples |
| ICLV | Publication quality, accurate LV effects |

---

## Requirements

```
biogeme>=3.2.0
numpy>=1.20.0
pandas>=1.3.0
scipy>=1.7.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
```

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{dcm_research_framework,
  title = {Discrete Choice Modeling Research Framework},
  author = {DCM Research Team},
  year = {2025},
  url = {https://github.com/your-repo}
}
```

---

## License

MIT License - See LICENSE file for details.

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests: `pytest tests/`
4. Submit pull request

---

## Support

- Issues: [GitHub Issues](https://github.com/your-repo/issues)
- Documentation: See `docs/` folder
- Quick Start: See `docs/QUICK_START.md`
