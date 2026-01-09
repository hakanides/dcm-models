# Discrete Choice Modeling (DCM) Research Framework

**Authors: Hakan Mülayim, Giray Girengir, Ataol Azeritürk**

A Python framework for discrete choice model estimation with integrated data simulation for validation. Implements the full hierarchy from MNL to ICLV with 32+ model specifications.

---

## Two-Tier Architecture

This project provides **two complementary approaches** for different needs:

### `models/` - Start Here (Isolated Validation)

Rigorous statistical validation with matching Data Generating Processes (DGP). Each model folder is self-contained and runs independently.

**Use this when:**
- Learning DCM methodology step-by-step
- Quick parameter recovery tests (~30 sec to ~10 min per model)
- Understanding the MNL → MXL → HCM → ICLV progression
- Limited computational resources
- Teaching or demonstration

```bash
# Example: Run basic MNL validation
cd models/mnl_basic
python run.py  # ~30 seconds
```

### `full_model/` - Full Research Pipeline

Publication-ready framework with 32+ model specifications for comprehensive research.

**Use this when:**
- You have sufficient computational resources (16GB+ RAM recommended)


```bash
# Run complete pipeline
cd full_model
python scripts/run_all_models.py  # ~30-60 minutes
```

---

## Recommended Learning Path

| Step | Model | Folder | Time | What You Learn |
|------|-------|--------|------|----------------|
| 1 | MNL Basic | `models/mnl_basic/` | ~30 sec | Baseline discrete choice |
| 2 | MNL Demographics | `models/mnl_demographics/` | ~1 min | Observable heterogeneity |
| 3 | MXL Basic | `models/mxl_basic/` | ~2 min | Unobserved heterogeneity |
| 4 | HCM Basic | `models/hcm_basic/` | ~3 min | Latent variables (two-stage) |
| 5 | HCM Full | `models/hcm_full/` | ~5 min | Attenuation bias problem |
| 6 | ICLV | `models/iclv/` | ~10 min | Simultaneous estimation (solution) |

**Key Insight:** ICLV simultaneous estimation reduces latent variable bias from 50-66% (HCM two-stage) to 2.4%.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Isolated Model Validation](#isolated-model-validation)
- [Full Pipeline](#full-pipeline)
- [Model Hierarchy](#model-hierarchy)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [For Real Data](#for-real-data)
- [API Reference](#api-reference)
- [Methodology Notes](#methodology-notes)

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

### 2. Run Your First Model (Isolated Validation)

```bash
# Start with basic MNL - simplest model
cd models/mnl_basic
python run.py

# Check results
ls results/
ls output/latex/
```

### 3. Progress Through Model Hierarchy

```bash
# After understanding MNL, try MXL
cd ../mxl_basic
python run.py

# Then HCM to see attenuation bias
cd ../hcm_basic
python run.py

# Finally ICLV to see the solution
cd ../iclv
python run.py
```

### 4. Full Pipeline (When Ready)

```bash
cd full_model
python scripts/run_all_models.py

# Results
ls results/all_models/
ls output/latex/
```

---

## Project Structure

```
DCM-Research/
│
├── models/                              # TIER 1: ISOLATED MODEL VALIDATION
│   │                                    # Start here for learning & quick tests
│   ├── mnl_basic/                       # Basic MNL (~30 sec, <2% bias)
│   ├── mnl_demographics/                # MNL + demographics (~1 min)
│   ├── mxl_basic/                       # MXL random coefficients (~2 min)
│   ├── hcm_basic/                       # HCM single LV (~3 min, shows attenuation)
│   ├── hcm_full/                        # HCM 4 LVs (~5 min, shows attenuation)
│   ├── iclv/                            # ICLV simultaneous (~10 min, unbiased)
│   └── shared/                          # Shared utilities
│       ├── policy_tools.py              # WTP, elasticity, consumer surplus
│       ├── latex_tools.py               # LaTeX table generation
│       ├── sample_stats.py              # Descriptive statistics
│       └── cleanup.py                   # Output management
│
├── full_model/                          # TIER 2: FULL RESEARCH PIPELINE
│   │                                    # Use for comprehensive research
│   ├── src/
│   │   ├── models/                      # 32+ model specifications
│   │   ├── simulation/                  # Data generation
│   │   ├── estimation/                  # Validation utilities
│   │   ├── policy_analysis/             # WTP, elasticity, welfare
│   │   └── utils/                       # LaTeX, visualization
│   ├── scripts/
│   │   └── run_all_models.py            # Main pipeline (~30-60 min)
│   ├── config/
│   │   ├── model_config.json            # True parameters
│   │   └── items_config.csv             # Likert items
│   ├── data/
│   │   ├── raw/scenarios_prepared.csv   # Choice design
│   │   └── simulated/                   # Generated data
│   ├── tests/                           # Comprehensive test suite
│   ├── results/                         # Estimation outputs
│   └── output/latex/                    # Publication tables
│
├── docs/
│   ├── QUICK_START.md                   # 5-minute getting started
│   ├── METHODOLOGY.md                   # Statistical methodology
│   └── MODEL_SPECIFICATIONS.md          # Complete model equations
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Isolated Model Validation

Each model in `models/` has its own Data Generating Process (DGP) that **exactly matches** the model specification. This ensures unbiased parameter recovery when the model is correctly specified.

### How It Works

```
┌─────────────────────────────────────────────────────────────────────┐
│  Step 1: CONFIGURE                                                   │
│  config.json → Define TRUE parameter values                         │
│  • Population size (N individuals, T choice tasks)                   │
│  • True coefficients (ASC, B_FEE, B_DUR, etc.)                      │
│  • Demographics distribution                                         │
│  • Latent variables (for HCM/ICLV)                                  │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Step 2: SIMULATE                                                    │
│  simulate_full_data.py → Generate synthetic data                    │
│  • Draw demographics from config distributions                       │
│  • Compute utilities using TRUE parameters                          │
│  • Simulate choices using Random Utility Model                      │
│  • Output: data/simulated_data.csv                                  │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Step 3: ESTIMATE                                                    │
│  model.py → Recover parameters using Biogeme                        │
│  • Load simulated data                                              │
│  • Estimate model (MLE optimization)                                │
│  • Compare estimates to TRUE values                                 │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Step 4: OUTPUT                                                      │
│  Saved to multiple locations:                                       │
│  • results/parameter_comparison.csv   (estimates vs true)           │
│  • results/estimation_results.csv     (full Biogeme output)         │
│  • output/latex/                      (publication tables)          │
│  • policy_analysis/                   (WTP, elasticities)           │
│  • sample_stats/                      (descriptive statistics)      │
└─────────────────────────────────────────────────────────────────────┘
```

### Config.json Structure

Each model's `config.json` defines the true parameters for simulation:

```json
{
  "model_info": {
    "name": "MNL Basic",
    "true_values": {
      "ASC_paid": 5.0,
      "B_FEE": -0.08,
      "B_DUR": -0.08
    }
  },
  "population": {
    "N": 500,
    "T": 10,
    "seed": 42
  },
  "choice_model": {
    "fee_scale": 10000.0,
    "attribute_terms": [...]
  }
}
```

**Key sections:**
- `true_values`: Parameters you want to recover (ground truth)
- `population.N`: Number of individuals to simulate
- `population.T`: Choice tasks per individual
- `population.seed`: Random seed for reproducibility

### Validation Results

| Model | Folder | True → Estimated | Bias | 95% CI Coverage |
|-------|--------|------------------|------|-----------------|
| MNL Basic | `models/mnl_basic/` | Unbiased | <2% | Yes |
| MNL Demographics | `models/mnl_demographics/` | Unbiased | <2% | Yes |
| MXL Basic | `models/mxl_basic/` | Good | <6% | Yes |
| HCM Basic | `models/hcm_basic/` | **Attenuated** | ~50-66% | No (LV effect) |
| HCM Full | `models/hcm_full/` | **Attenuated** | ~40-70% | No (all LV effects) |
| ICLV | `models/iclv/` | Unbiased | ~2.4% | Yes |

### Output Files

| Location | File | Contents |
|----------|------|----------|
| `results/` | `parameter_comparison.csv` | True vs Estimated values, bias %, CI coverage |
| `results/` | `estimation_results.csv` | Full Biogeme output (SE, t-stats, p-values) |
| `output/latex/` | `*.tex` | Publication-ready LaTeX tables |
| `policy_analysis/` | `wtp.csv`, `elasticity.csv` | Willingness-to-pay, elasticity calculations |
| `sample_stats/` | `descriptives.csv` | Sample descriptive statistics |

### Each Model Folder Contains

```
models/mnl_basic/
├── config.json              # Model-specific true parameters
├── simulate_full_data.py    # Standalone data generator
├── model.py                 # Biogeme estimation
├── run.py                   # Orchestrator (simulate → estimate)
├── data/                    # Generated data
├── results/                 # Estimation results
├── output/latex/            # LaTeX tables
├── policy_analysis/         # WTP, elasticity
└── sample_stats/            # Descriptive statistics
```

---

## Full Pipeline

The `full_model/` folder implements the complete research framework with 32+ model specifications.

### Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SIMULATION MODULE                             │
│  config/model_config.json → src/simulation/                │
│  • Demographics (age, education, income)                            │
│  • Latent variables (patriotism, secularism constructs)            │
│  • Likert scale responses (5-point ordered probit)                 │
│  • Choice behavior (random utility maximization)                    │
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

### Computational Requirements


### Computational Requirements

| Resource | Minimum | Recommended | High-Performance |
|----------|---------|-------------|-------------------|
| RAM | 8 GB | 16 GB | 128+ GB |
| CPU Cores | 2 | 4+ | 32+ cores |
| Disk Space | 1 GB | 5 GB | 50+ GB |
| Time | ~30 min | ~60 min | ~6-10 min |
| GPU Support | Optional | Recommended | CUDA-enabled |

**Note:** High-performance configurations with 10x computational power enable parallel model estimation across all 32+ specifications simultaneously.


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

### Isolated Model (Quick Validation)

```bash
# Run MNL Basic
cd models/mnl_basic
python run.py

# Check results
cat results/parameter_estimates.csv
```

### Full Pipeline (Research)

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
    n_draws=500,
    auto_scale=True,
    use_panel=True
)

print(f"Log-likelihood: {result.log_likelihood:.2f}")
print(f"Converged: {result.convergence}")
```

---

## Configuration

### model_config.json

```json
{
  "population": {
    "n_individuals": 1000,
    "n_choice_tasks": 10
  },
  "demographics": {
    "age_idx": {"distribution": "categorical", "probabilities": [0.2, 0.3, 0.3, 0.2]}
  },
  "latent_variables": {
    "pat_blind": {
      "structural": {"age_idx": 0.15, "income_indiv_idx": -0.10},
      "variance": 1.0
    }
  },
  "choice_model": {
    "base_terms": [{"name": "ASC_paid", "coef": 2.0}],
    "attribute_terms": [
      {"name": "b_fee10k", "attribute": "fee", "scale": 10000, "base_coef": -0.5}
    ]
  }
}
```

---

## For Real Data

When using real survey data:

1. **Format Data**: Match expected column structure
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

---

## API Reference

### Models

| Module | Function | Description |
|--------|----------|-------------|
| `mnl_basic` | `estimate_mnl_basic()` | Basic MNL estimation |
| `mnl_extended` | `run_mnl_extended()` | 8 MNL specifications |
| `mxl_basic` | `estimate_mxl_basic()` | Basic MXL estimation |
| `mxl_extended` | `run_mxl_extended()` | 8 MXL specifications |
| `hcm_basic` | `estimate_hcm_basic()` | Single LV HCM |
| `hcm_extended` | `run_hcm_extended()` | 8 HCM specifications |
| `iclv` | `estimate_iclv()` | ICLV simultaneous |

### ICLV Enhanced Features

| Feature | Parameter | Description |
|---------|-----------|-------------|
| Auto-scaling | `auto_scale=True` | Scales large fee values |
| Two-stage start | `use_two_stage_start=True` | Better convergence |
| Panel support | `use_panel=True` | Cluster-robust SE |
| Robust SE | `compute_robust_se=True` | Sandwich estimator |

---

## Methodology Notes

### Two-Stage HCM vs ICLV

**HCM (two-stage)** causes **attenuation bias** (~15-50%):
1. Estimate latent variables from Likert items
2. Use LV estimates as fixed regressors → measurement error ignored

**ICLV (simultaneous)** eliminates this bias:
1. Integrate over the LV distribution
2. Jointly estimate measurement + choice models

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

```bibtex
@software{dcm_research_framework,
  title = {Discrete Choice Modeling Research Framework},
  author = {Mülayim, Hakan and Girengir, Giray and Azeritürk, Ataol},
  year = {2025}
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

- Documentation: See `docs/` folder
- Quick Start: See `docs/QUICK_START.md`
