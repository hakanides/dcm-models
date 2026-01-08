"""
Run All Models and Compare to True Parameters
==============================================

This script runs MNL, MXL, and HCM models and compares results
to true parameter values from model_config.json.

KNOWN LIMITATION: HARDCODED VARIABLE NAMES
==========================================
This file uses hardcoded variable names (fee1, fee2, fee3, dur1, dur2, dur3)
which limits reusability for different choice experiment designs.

For new projects with different designs (e.g., 4 alternatives, different
attribute names), see src/utils/variable_config.py for a configuration-based
approach that makes variable definitions reusable.

ISSUE #31: CONSTANTS AND MAGIC NUMBERS
======================================
All magic numbers are documented here for transparency:

FEE_SCALE = 10000.0
    - Fee values in Korean Won, scaled by 10,000 for numerical stability
    - e.g., 500,000 KRW → 50 in scaled units
    - MUST match model_config.json's 'fee_scale' value

DEMOGRAPHIC CENTERING (from model_config.json):
    - age_idx: center=2, scale=2 → (age_idx - 2) / 2
    - edu_idx: center=2, scale=2 → (edu_idx - 2) / 2
    - income_indiv_idx: center=3, scale=2 → (income_indiv_idx - 3) / 2
    - income_house_idx: center=3, scale=2 → (income_house_idx - 3) / 2
    - marital_idx: center=1, scale=1 → (marital_idx - 1) / 1
    These values are validated against config at runtime.

PARAMETER BOUNDS:
    - B_FEE: [-10, 0] - fee coefficient must be negative (demand law)
    - B_DUR: [-5, 0] - duration coefficient must be negative
    - SIGMA: [0.001, 1.5] - standard deviation of random coefficients
    - Interactions: [-0.5, 0.5] or [-1, 1] depending on expected effect size

MXL DRAWS:
    - Development: 1000 draws (faster, less stable)
    - Production: 5000 draws (publication quality)

ISSUE #32: VARIABLE NAMING CONVENTIONS
======================================
This project uses multiple naming conventions for historical reasons:

DATA COLUMNS (from simulation):
    - fee1, fee2, fee3: Raw fee values in KRW
    - dur1, dur2, dur3: Duration in days
    - age_idx, edu_idx: Demographic indices (1-4 scale typically)

BIOGEME VARIABLES:
    - fee1_10k, fee2_10k: Fee scaled by 10,000
    - age_c, edu_c: Centered demographics

PARAMETERS (in results):
    - B_FEE: Base fee coefficient
    - B_FEE_AGE: Fee × age interaction
    - B_DUR: Base duration coefficient

CONFIG KEYS (model_config.json):
    - b_fee_scaled_base: True value for B_FEE
    - b_fee_scaled_x_age_idx: True value for B_FEE_AGE

The compare_to_true() function maps between these naming conventions.
"""

import numpy as np
import pandas as pd
import json
import shutil
import os
import sys
from pathlib import Path
import warnings
import logging

# =============================================================================
# PROJECT ROOT SETUP
# =============================================================================
# Determine project root relative to this script's location (scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# ISSUE #39: WARNING AND LOGGING CONFIGURATION
# =============================================================================
# By default, we suppress expected Biogeme optimization warnings.
# These are normal during model estimation:
#   - FutureWarning: Biogeme API deprecation warnings
#   - overflow: Numerical overflow in exp() during early iterations
#   - divide by zero: Can occur during probability calculation
#
# For debugging, set DEBUG_MODE = True to see all warnings
# Warnings are also logged to file for post-hoc investigation
# =============================================================================

DEBUG_MODE = False  # Set True to see all warnings

# Set up logging to file (always capture warnings even if not displayed)
logging.basicConfig(
    filename='dcm_estimation.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def warning_to_log(message, category, filename, lineno, file=None, line=None):
    """Custom warning handler that logs to file."""
    logging.warning(f'{category.__name__}: {message} ({filename}:{lineno})')

# Capture warnings in log
logging.captureWarnings(True)

if DEBUG_MODE:
    # Show all warnings
    warnings.filterwarnings('default')
    print("DEBUG MODE: All warnings will be displayed")
else:
    # Suppress expected warnings but log them
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', message='.*overflow.*')
    warnings.filterwarnings('ignore', message='.*divide by zero.*')
    # Log suppressed warnings
    warnings.showwarning = warning_to_log

import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta, Variable, bioDraws, MonteCarlo, log, PanelLikelihoodTrajectory

from src.utils.latex_output import (
    generate_latex_output,
    cleanup_latex_output,
    generate_all_simulation_latex
)

# Import extended model modules
try:
    from src.models.mnl_extended import run_mnl_extended
    HAS_MNL_EXTENDED = True
except ImportError:
    HAS_MNL_EXTENDED = False

try:
    from src.models.mxl_extended import run_mxl_extended
    HAS_MXL_EXTENDED = True
except ImportError:
    HAS_MXL_EXTENDED = False

try:
    from src.models.hcm_extended import run_hcm_extended
    HAS_HCM_EXTENDED = True
except ImportError:
    HAS_HCM_EXTENDED = False

# =============================================================================
# ISSUE #40: CROSS-VALIDATION MODULE
# =============================================================================
# Cross-validation is available but not integrated by default to save time.
# To use cross-validation for model selection:
#
#   from src.estimation.cross_validation import cross_validate_model, utility_mnl_basic
#
#   # Run 5-fold CV for a model
#   cv_results = cross_validate_model(
#       df=df_num,
#       model_func=model_mnl_basic,
#       utility_func=utility_mnl_basic,
#       n_folds=5,
#       id_col='ID'
#   )
#
#   # Check overfit ratio (>1.1 suggests overfitting)
#   print(f"Overfit ratio: {cv_results['overfit_ratio']:.3f}")
#
# See src/estimation/cross_validation.py for full documentation.
# =============================================================================

# Import identification diagnostics (Issue #14)
try:
    from src.estimation.identification_diagnostics import IdentificationDiagnostics
    HAS_ID_DIAGNOSTICS = True
except ImportError:
    HAS_ID_DIAGNOSTICS = False


# =============================================================================
# CLEANUP FUNCTIONS
# =============================================================================

def cleanup_results_directory(output_dir: Path):
    """Delete all files in output directory before new run."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
        print(f"Cleaned up: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def cleanup_iter_files(project_root: Path = None):
    """Delete all .iter files from project root for clean estimation."""
    if project_root is None:
        project_root = Path.cwd()

    iter_files = list(project_root.glob("__*.iter"))
    if iter_files:
        for iter_file in iter_files:
            iter_file.unlink()
        print(f"Deleted {len(iter_files)} .iter files from project root")

# =============================================================================
# LOAD TRUE PARAMETERS FROM CONFIG
# =============================================================================

def load_true_params(config_path: str = "config/model_config_advanced.json") -> dict:
    """Load true parameter values from config."""
    with open(config_path) as f:
        config = json.load(f)

    true_params = {}

    # ASC
    for term in config['choice_model']['base_terms']:
        if 'paid1' in term.get('apply_to', []) or 'paid2' in term.get('apply_to', []):
            true_params['ASC_paid'] = term['coef']

    # Attribute coefficients
    for term in config['choice_model']['attribute_terms']:
        name = term['name']
        true_params[f'{name}_base'] = term['base_coef']

        for interaction in term.get('interactions', []):
            int_name = f"{name}_x_{interaction['with']}"
            true_params[int_name] = interaction['coef']

    true_params['_fee_scale'] = config['choice_model'].get('fee_scale', 10000)

    return true_params

# =============================================================================
# DATA PREPARATION
# =============================================================================

def load_centering_config(config_path: str = "config/model_config_advanced.json") -> dict:
    """Extract centering/scaling parameters from config for validation."""
    with open(config_path) as f:
        config = json.load(f)

    centering = {}
    for term in config['choice_model'].get('attribute_terms', []):
        for int_spec in term.get('interactions', []):
            var_name = int_spec.get('with', '')
            if 'center' in int_spec and 'scale' in int_spec:
                centering[var_name] = {
                    'center': int_spec['center'],
                    'scale': int_spec['scale']
                }
    return centering


def prepare_data(data_path: str, config_path: str = "config/model_config_advanced.json"):
    """Load and prepare data for all models.

    IMPORTANT: Centering and scaling MUST match model_config.json exactly!
    The simulator uses these values from the config's interaction specifications.
    If you change these, parameter recovery will be biased.

    Config specifies: (variable - center) / scale
    """
    df = pd.read_csv(data_path)

    # Scale fees by 10,000 (standard across all configs and models)
    FEE_SCALE = 10000.0

    # CRITICAL: Validate fee_scale matches config to prevent silent parameter bias
    true_params = load_true_params(config_path)
    config_fee_scale = true_params.get('_fee_scale', 10000.0)
    assert FEE_SCALE == config_fee_scale, (
        f"FEE_SCALE mismatch! Code uses {FEE_SCALE}, config has {config_fee_scale}. "
        f"This would cause {FEE_SCALE/config_fee_scale}x parameter bias."
    )
    df['fee1_10k'] = df['fee1'] / FEE_SCALE
    df['fee2_10k'] = df['fee2'] / FEE_SCALE
    df['fee3_10k'] = df['fee3'] / FEE_SCALE

    # Load centering config for validation
    centering_config = load_centering_config(config_path)

    # Define centering values used in code (must match config!)
    CENTERING = {
        'age_idx': {'center': 2.0, 'scale': 2.0},
        'edu_idx': {'center': 3.0, 'scale': 2.0},
        'income_indiv_idx': {'center': 3.0, 'scale': 2.0},
        'income_house_idx': {'center': 3.0, 'scale': 2.0},
        'marital_idx': {'center': 0.5, 'scale': 0.5},
    }

    # CRITICAL: Validate centering matches config to prevent biased interaction coefficients
    for var_name, code_values in CENTERING.items():
        if var_name in centering_config:
            config_values = centering_config[var_name]
            assert code_values['center'] == config_values['center'], (
                f"Centering mismatch for {var_name}! "
                f"Code uses center={code_values['center']}, config has {config_values['center']}"
            )
            assert code_values['scale'] == config_values['scale'], (
                f"Scale mismatch for {var_name}! "
                f"Code uses scale={code_values['scale']}, config has {config_values['scale']}"
            )

    # Center demographics using validated values
    df['age_c'] = (df['age_idx'] - CENTERING['age_idx']['center']) / CENTERING['age_idx']['scale']
    df['edu_c'] = (df['edu_idx'] - CENTERING['edu_idx']['center']) / CENTERING['edu_idx']['scale']
    df['inc_c'] = (df['income_indiv_idx'] - CENTERING['income_indiv_idx']['center']) / CENTERING['income_indiv_idx']['scale']
    df['inc_hh_c'] = (df['income_house_idx'] - CENTERING['income_house_idx']['center']) / CENTERING['income_house_idx']['scale']
    df['marital_c'] = (df['marital_idx'] - CENTERING['marital_idx']['center']) / CENTERING['marital_idx']['scale']

    # Create Likert proxies (standardized)
    # NOTE: Standardization uses sample mean/std. For out-of-sample prediction,
    # you must save these parameters and apply them consistently.
    lv_items = {
        'pat_blind': ['pat_blind_1', 'pat_blind_2', 'pat_blind_3', 'pat_blind_4'],
        'pat_const': ['pat_constructive_1', 'pat_constructive_2', 'pat_constructive_3', 'pat_constructive_4'],
        'sec_dl': ['sec_dl_1', 'sec_dl_2', 'sec_dl_3', 'sec_dl_4'],
        'sec_fp': ['sec_fp_1', 'sec_fp_2', 'sec_fp_3', 'sec_fp_4'],
    }

    # Store standardization parameters for reproducibility and out-of-sample prediction
    lv_standardization_params = {}

    for lv_name, items in lv_items.items():
        available = [c for c in items if c in df.columns]
        if available:
            proxy = df[available].mean(axis=1)
            proxy_mean = proxy.mean()
            proxy_std = proxy.std()

            # Store parameters for later use (prediction, reporting)
            lv_standardization_params[lv_name] = {
                'mean': float(proxy_mean),
                'std': float(proxy_std),
                'items': available
            }

            df[f'{lv_name}_proxy'] = (proxy - proxy_mean) / proxy_std

    # Save standardization parameters to file for out-of-sample prediction
    output_dir = Path("results/all_models")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "lv_standardization_params.json", 'w') as f:
        json.dump(lv_standardization_params, f, indent=2)
    print(f"  LV standardization params saved to {output_dir / 'lv_standardization_params.json'}")

    # Drop string columns
    string_cols = df.select_dtypes(include=['object']).columns.tolist()
    df_num = df.drop(columns=string_cols)

    # CRITICAL: Check for NaN/Inf in numeric columns (Issue #13)
    nan_counts = df_num.isna().sum()
    inf_counts = np.isinf(df_num.select_dtypes(include=[np.number])).sum()

    if nan_counts.sum() > 0:
        nan_cols = nan_counts[nan_counts > 0]
        print(f"\n⚠️  WARNING: Found NaN values in {len(nan_cols)} columns:")
        for col, count in nan_cols.items():
            print(f"    {col}: {count} NaN values ({count/len(df_num)*100:.1f}%)")
        print("  Biogeme will fail on NaN. Consider imputation or dropping rows.")

    if inf_counts.sum() > 0:
        inf_cols = inf_counts[inf_counts > 0]
        print(f"\n⚠️  WARNING: Found Inf values in {len(inf_cols)} columns:")
        for col, count in inf_cols.items():
            print(f"    {col}: {count} Inf values")
        print("  Biogeme will fail on Inf. Check data transformations.")

    # ISSUE #36: Check for duplicate observations (same choice situation for same person)
    if 'ID' in df_num.columns and 'scenario_id' in df_num.columns:
        duplicate_check = df_num.groupby(['ID', 'scenario_id']).size()
        duplicates = duplicate_check[duplicate_check > 1]
        if len(duplicates) > 0:
            print(f"\n⚠️  WARNING: Found {len(duplicates)} duplicate (ID, scenario) combinations")
            print(f"     Same person seeing same choice situation multiple times")
            print(f"     This can cause perfect collinearity in panel models")
            # Show top duplicates
            top_dups = duplicates.nlargest(5)
            for (id_val, scen_val), count in top_dups.items():
                print(f"       ID={id_val}, scenario={scen_val}: {count} times")

    print(f"\nLoaded {len(df):,} obs from {df['ID'].nunique()} respondents")

    # Choice share balance check (Issue #12 - make actionable)
    shares = {
        1: df['CHOICE'].eq(1).mean(),
        2: df['CHOICE'].eq(2).mean(),
        3: df['CHOICE'].eq(3).mean()
    }
    print(f"Choice shares: 1={shares[1]:.1%}, 2={shares[2]:.1%}, 3={shares[3]:.1%}")

    # IMBALANCE WARNING with severity levels
    severe_imbalance = False
    for alt, share in shares.items():
        if share > 0.80:
            print(f"  🚨 SEVERE: Alternative {alt} has {share:.1%} share - model may not converge!")
            severe_imbalance = True
        elif share > 0.70:
            print(f"  ⚠️  WARNING: Alternative {alt} has {share:.1%} share - may cause identification issues")
        elif share < 0.05:
            print(f"  🚨 SEVERE: Alternative {alt} has only {share:.1%} share - insufficient variation!")
            severe_imbalance = True
        elif share < 0.10:
            print(f"  ⚠️  WARNING: Alternative {alt} has only {share:.1%} share - limited variation")

    if severe_imbalance:
        print("\n  ACTION REQUIRED: Consider pooling alternatives or using weighted sampling")

    return df_num, db.Database('dcm', df_num)


def prepare_mxl_database(df_num):
    """Create database with panel structure for MXL estimation.

    IMPORTANT: For Mixed Logit with panel data, must specify panel structure
    to ensure random coefficients are properly correlated within individuals.
    Without this, MXL treats each observation as independent, which is incorrect
    for repeated choices by the same person.

    Args:
        df_num: Prepared numeric DataFrame (from prepare_data)

    Returns:
        Biogeme Database with panel structure
    """
    database = db.Database('dcm_mxl', df_num)

    if 'ID' in df_num.columns:
        database.panel('ID')
        n_individuals = df_num['ID'].nunique()
        n_obs_per_person = len(df_num) / n_individuals

        # Panel imbalance detection (Issue #11)
        obs_per_person = df_num.groupby('ID').size()
        min_obs = obs_per_person.min()
        max_obs = obs_per_person.max()
        std_obs = obs_per_person.std()

        print(f"  Panel structure: {n_individuals} individuals, {n_obs_per_person:.1f} obs/person avg")
        print(f"  Panel balance: min={min_obs}, max={max_obs}, std={std_obs:.1f}")

        # Check for severe imbalance
        if max_obs > 3 * n_obs_per_person:
            print(f"  ⚠️  WARNING: Panel imbalance detected!")
            print(f"     Some individuals have {max_obs} obs (>{3*n_obs_per_person:.0f} = 3x average)")
            print(f"     This may bias MXL estimates toward frequent respondents")

        # Check if small proportion provides most data
        top_10pct = obs_per_person.nlargest(int(n_individuals * 0.1)).sum()
        total_obs = len(df_num)
        if top_10pct / total_obs > 0.3:
            print(f"  ⚠️  WARNING: Top 10% of respondents provide {top_10pct/total_obs:.0%} of observations")
            print(f"     Consider weighting or balanced sampling")

    return database

# =============================================================================
# STARTING VALUES HELPER (ISSUE #37)
# =============================================================================

def compute_adaptive_starting_values(df_num: pd.DataFrame) -> dict:
    """Compute data-adaptive starting values for Beta parameters.

    ISSUE #37 FIX: Instead of using fixed starting values (e.g., -0.5 for all
    fee interactions), compute reasonable starting values based on the data.

    This helps optimization converge faster and can prevent getting stuck
    in local optima.

    Args:
        df_num: Prepared numeric DataFrame

    Returns:
        Dict of {param_name: suggested_starting_value}

    Example usage:
        start_vals = compute_adaptive_starting_values(df_num)
        B_FEE = Beta('B_FEE', start_vals.get('B_FEE', -0.5), -10, 0, 0)
    """
    starting_values = {}

    # Fee coefficient: rough estimate from utility difference
    # If paid alternatives have higher fees but are still chosen,
    # the ASC must be large or fee sensitivity must be small
    if 'fee1_10k' in df_num.columns:
        avg_fee_diff = df_num['fee1_10k'].mean() - df_num['fee3_10k'].mean()
        # Rough estimate: if fee diff is X and choice shares are equal,
        # fee coefficient must be roughly offset by ASC/X
        if avg_fee_diff != 0:
            # Very rough: assume -0.5 per unit scaled fee is reasonable
            # Scale based on actual fee range
            fee_range = df_num['fee1_10k'].std()
            starting_values['B_FEE'] = -0.5 / max(fee_range, 0.1)
        else:
            starting_values['B_FEE'] = -0.5

    # Duration coefficient: similar logic
    if 'dur1' in df_num.columns:
        dur_std = df_num['dur1'].std()
        # Typically want effect of ~0.05-0.1 per day
        starting_values['B_DUR'] = -0.08 / max(dur_std / 5, 1)

    # Interactions: start at 0 but scaled by demographic variation
    for demo_var in ['age_c', 'edu_c', 'inc_c', 'inc_hh_c', 'marital_c']:
        if demo_var in df_num.columns:
            demo_std = df_num[demo_var].std()
            # Interaction effect: small fraction of main effect
            starting_values[f'B_FEE_{demo_var.upper()}'] = 0.05 * demo_std
            starting_values[f'B_DUR_{demo_var.upper()}'] = 0.02 * demo_std

    # ASC: estimate from choice shares
    if 'CHOICE' in df_num.columns:
        share_1 = (df_num['CHOICE'] == 1).mean()
        share_3 = (df_num['CHOICE'] == 3).mean()
        if share_1 > 0 and share_3 > 0:
            # log-odds ratio gives rough ASC estimate
            log_odds = np.log(share_1 / share_3)
            starting_values['ASC_paid'] = log_odds / 2  # Split between fee and ASC

    return starting_values

# =============================================================================
# MODEL DEFINITIONS
# =============================================================================
#
# KNOWN LIMITATION: WEAK INTERACTION EFFECTS
# ==========================================
# The true interaction coefficients in model_config.json are intentionally
# small (0.02-0.12) relative to base coefficients (-0.8 fee, -0.08 duration).
# This is realistic for DCM research where heterogeneity is often subtle.
#
# IMPLICATIONS FOR ESTIMATION:
# 1. Interaction terms may have t-statistics < 1.96 even when correctly estimated
# 2. Standard errors may exceed coefficient magnitudes
# 3. Large samples (N>1000 individuals) recommended for precise recovery
# 4. Do NOT interpret statistical insignificance as estimation failure
#
# To strengthen effects for validation: Edit model_config.json and multiply
# interaction coefficients by 2-3x, then regenerate data.
# =============================================================================

def model_mnl_basic(database):
    """Basic MNL - just ASC, fee, duration.

    ISSUE #23 NOTE: ASC Values in Different Configs
    ===============================================
    - model_config.json: ASC = 1.5 (simple validation setup)
    - model_config_advanced.json: ASC = 0.25 (realistic HCM setup)

    This is INTENTIONAL - different configs serve different purposes:
    - Basic config (ASC=1.5): Strong preference for paid alternatives,
      makes choice shares easier to validate
    - Advanced config (ASC=0.25): Realistic values for complex models
      where other factors (LVs, interactions) explain preference

    Always match your estimation to the config used for data generation!
    """
    dur1, dur2, dur3 = Variable('dur1'), Variable('dur2'), Variable('dur3')
    fee1, fee2, fee3 = Variable('fee1_10k'), Variable('fee2_10k'), Variable('fee3_10k')
    CHOICE = Variable('CHOICE')

    ASC_paid = Beta('ASC_paid', 1.0, None, None, 0)
    B_FEE = Beta('B_FEE', -0.5, -10, 0, 0)  # Bounded negative
    B_DUR = Beta('B_DUR', -0.05, -5, 0, 0)  # Bounded negative

    V1 = ASC_paid + B_FEE * fee1 + B_DUR * dur1
    V2 = ASC_paid + B_FEE * fee2 + B_DUR * dur2
    V3 = B_FEE * fee3 + B_DUR * dur3

    V = {1: V1, 2: V2, 3: V3}
    av = {1: 1, 2: 1, 3: 1}

    return models.loglogit(V, av, CHOICE), 'MNL-Basic'

def check_coefficient_bounds(result: dict, demographic_range: float = 2.0) -> list:
    """Check if estimated coefficients could produce implausible values.

    For fee: total β_fee should be negative (higher fee = lower utility)
    For duration: total β_dur should be negative (longer duration = lower utility)

    Args:
        result: Estimation result dict with 'betas' key
        demographic_range: Expected max absolute value of centered demographics

    Returns:
        List of warning messages (empty if all OK)
    """
    warnings = []
    betas = result.get('betas', {})

    # Check fee coefficient bounds
    b_fee_base = betas.get('B_FEE', betas.get('B_FEE_MU', 0))
    fee_interactions = [k for k in betas.keys() if k.startswith('B_FEE_') and k not in ['B_FEE_MU', 'B_FEE_SIGMA']]
    max_fee_contribution = sum(abs(betas.get(k, 0)) * demographic_range for k in fee_interactions)

    if b_fee_base + max_fee_contribution > 0:
        warnings.append(
            f"⚠️  Fee coefficient could become positive for extreme demographics! "
            f"Base={b_fee_base:.3f}, max interaction contribution=+{max_fee_contribution:.3f}"
        )

    # Check duration coefficient bounds
    b_dur_base = betas.get('B_DUR', betas.get('B_DUR_MU', 0))
    dur_interactions = [k for k in betas.keys() if k.startswith('B_DUR_') and k not in ['B_DUR_MU', 'B_DUR_SIGMA']]
    max_dur_contribution = sum(abs(betas.get(k, 0)) * demographic_range for k in dur_interactions)

    if b_dur_base + max_dur_contribution > 0:
        warnings.append(
            f"⚠️  Duration coefficient could become positive for extreme demographics! "
            f"Base={b_dur_base:.3f}, max interaction contribution=+{max_dur_contribution:.3f}"
        )

    return warnings


def model_mnl_demographics(database):
    """MNL with demographic interactions.

    BOUNDS NOTE: Interaction bounds [-1, 1] chosen to prevent total coefficient
    from becoming positive even at extreme demographic values (centered range ±2).
    """
    dur1, dur2, dur3 = Variable('dur1'), Variable('dur2'), Variable('dur3')
    fee1, fee2, fee3 = Variable('fee1_10k'), Variable('fee2_10k'), Variable('fee3_10k')
    CHOICE = Variable('CHOICE')
    age_c, edu_c, inc_c = Variable('age_c'), Variable('edu_c'), Variable('inc_c')
    inc_hh_c, marital_c = Variable('inc_hh_c'), Variable('marital_c')

    ASC_paid = Beta('ASC_paid', 1.0, None, None, 0)
    B_FEE = Beta('B_FEE', -0.5, -10, 0, 0)  # Bounded negative

    # Interaction bounds [-1, 1]: with 4 interactions and demographics ±2,
    # max contribution is 4*1*2=8, so B_FEE_i in [-10-8, 0+8] = [-18, 8]
    # Still allows positive total - consider tighter bounds or lognormal if issue
    B_FEE_AGE = Beta('B_FEE_AGE', 0.05, -1, 1, 0)
    B_FEE_EDU = Beta('B_FEE_EDU', 0.05, -1, 1, 0)
    B_FEE_INC = Beta('B_FEE_INC', 0.10, -1, 1, 0)
    B_FEE_INC_H = Beta('B_FEE_INC_H', 0.05, -1, 1, 0)

    B_DUR = Beta('B_DUR', -0.05, -5, 0, 0)  # Bounded negative
    B_DUR_EDU = Beta('B_DUR_EDU', -0.02, -0.5, 0.5, 0)
    B_DUR_INC = Beta('B_DUR_INC', -0.02, -0.5, 0.5, 0)
    B_DUR_INC_H = Beta('B_DUR_INC_H', -0.01, -0.5, 0.5, 0)
    B_DUR_MARITAL = Beta('B_DUR_MARITAL', -0.02, -0.5, 0.5, 0)

    B_FEE_i = B_FEE + B_FEE_AGE*age_c + B_FEE_EDU*edu_c + B_FEE_INC*inc_c + B_FEE_INC_H*inc_hh_c
    B_DUR_i = B_DUR + B_DUR_EDU*edu_c + B_DUR_INC*inc_c + B_DUR_INC_H*inc_hh_c + B_DUR_MARITAL*marital_c

    V1 = ASC_paid + B_FEE_i * fee1 + B_DUR_i * dur1
    V2 = ASC_paid + B_FEE_i * fee2 + B_DUR_i * dur2
    V3 = B_FEE_i * fee3 + B_DUR_i * dur3

    V = {1: V1, 2: V2, 3: V3}
    av = {1: 1, 2: 1, 3: 1}

    return models.loglogit(V, av, CHOICE), 'MNL-Demographics'

def model_mnl_full(database):
    """MNL with demographics + latent proxies.

    BOUNDS NOTE: Tighter interaction bounds to prevent implausible total coefficients.
    """
    dur1, dur2, dur3 = Variable('dur1'), Variable('dur2'), Variable('dur3')
    fee1, fee2, fee3 = Variable('fee1_10k'), Variable('fee2_10k'), Variable('fee3_10k')
    CHOICE = Variable('CHOICE')
    age_c, edu_c, inc_c = Variable('age_c'), Variable('edu_c'), Variable('inc_c')
    inc_hh_c, marital_c = Variable('inc_hh_c'), Variable('marital_c')
    pat_blind, pat_const = Variable('pat_blind_proxy'), Variable('pat_const_proxy')
    sec_dl, sec_fp = Variable('sec_dl_proxy'), Variable('sec_fp_proxy')

    ASC_paid = Beta('ASC_paid', 1.0, None, None, 0)

    B_FEE = Beta('B_FEE', -0.5, -10, 0, 0)  # Bounded negative
    # Tighter bounds [-0.5, 0.5] for interactions to prevent positive total fee coefficient
    B_FEE_AGE = Beta('B_FEE_AGE', 0.05, -0.5, 0.5, 0)
    B_FEE_EDU = Beta('B_FEE_EDU', 0.05, -0.5, 0.5, 0)
    B_FEE_INC = Beta('B_FEE_INC', 0.10, -0.5, 0.5, 0)
    B_FEE_INC_H = Beta('B_FEE_INC_H', 0.05, -0.5, 0.5, 0)
    B_FEE_PAT_B = Beta('B_FEE_PAT_B', -0.05, -0.5, 0.5, 0)
    B_FEE_PAT_C = Beta('B_FEE_PAT_C', -0.02, -0.5, 0.5, 0)
    B_FEE_SEC_D = Beta('B_FEE_SEC_D', 0.02, -0.5, 0.5, 0)
    B_FEE_SEC_F = Beta('B_FEE_SEC_F', 0.02, -0.5, 0.5, 0)

    B_DUR = Beta('B_DUR', -0.05, -5, 0, 0)  # Bounded negative
    B_DUR_EDU = Beta('B_DUR_EDU', -0.02, -0.5, 0.5, 0)
    B_DUR_INC = Beta('B_DUR_INC', -0.02, -0.5, 0.5, 0)
    B_DUR_INC_H = Beta('B_DUR_INC_H', -0.01, -0.5, 0.5, 0)
    B_DUR_MARITAL = Beta('B_DUR_MARITAL', -0.02, -0.5, 0.5, 0)
    B_DUR_PAT_B = Beta('B_DUR_PAT_B', 0.02, -0.5, 0.5, 0)
    B_DUR_PAT_C = Beta('B_DUR_PAT_C', 0.02, -0.5, 0.5, 0)
    B_DUR_SEC_D = Beta('B_DUR_SEC_D', -0.02, -0.5, 0.5, 0)
    B_DUR_SEC_F = Beta('B_DUR_SEC_F', -0.02, -0.5, 0.5, 0)

    B_FEE_i = (B_FEE + B_FEE_AGE*age_c + B_FEE_EDU*edu_c + B_FEE_INC*inc_c + B_FEE_INC_H*inc_hh_c +
               B_FEE_PAT_B*pat_blind + B_FEE_PAT_C*pat_const + B_FEE_SEC_D*sec_dl + B_FEE_SEC_F*sec_fp)
    B_DUR_i = (B_DUR + B_DUR_EDU*edu_c + B_DUR_INC*inc_c + B_DUR_INC_H*inc_hh_c + B_DUR_MARITAL*marital_c +
               B_DUR_PAT_B*pat_blind + B_DUR_PAT_C*pat_const + B_DUR_SEC_D*sec_dl + B_DUR_SEC_F*sec_fp)

    V1 = ASC_paid + B_FEE_i * fee1 + B_DUR_i * dur1
    V2 = ASC_paid + B_FEE_i * fee2 + B_DUR_i * dur2
    V3 = B_FEE_i * fee3 + B_DUR_i * dur3

    V = {1: V1, 2: V2, 3: V3}
    av = {1: 1, 2: 1, 3: 1}

    return models.loglogit(V, av, CHOICE), 'MNL-Full'

# =============================================================================
# MXL (MIXED LOGIT) MODELS
# =============================================================================
# NOTE: MXL estimates random coefficient heterogeneity (σ parameters).
# If using model_config.json (no random coefficients), σ will be ≈ 0.
# Use model_config_advanced.json to generate data with true random coefficients.
#
# ISSUE #29: CORRELATED RANDOM COEFFICIENTS
# =========================================
# Current models assume INDEPENDENT random coefficients (β_fee ⊥ β_dur).
# In reality, time-sensitive people often care about BOTH:
#   - "I want it cheap AND fast"
#   - Correlation between fee and duration sensitivity
#
# KNOWN LIMITATION: Adding correlation requires:
#   1. Cholesky decomposition: L where Σ = L*L'
#   2. Correlated draws: β = μ + L * z where z ~ N(0,I)
#   3. Additional parameters: off-diagonal elements of L
#   4. Significantly more draws for stable estimates
#
# For correlated MXL, consider:
#   - Apollo package in R (easier syntax)
#   - Full covariance specification in Biogeme
#   - The simulator (dcm_simulator_advanced.py) DOES support correlations
#     via the RandomCoefficientsHandler class if needed for validation.
# =============================================================================

def model_mxl_random_fee(database):
    """Mixed Logit with random fee coefficient.

    NOTE: σ will be ~0 if data generated without random coefficients.
    Uses PanelLikelihoodTrajectory for panel data (same draws across choice situations).
    """
    dur1, dur2, dur3 = Variable('dur1'), Variable('dur2'), Variable('dur3')
    fee1, fee2, fee3 = Variable('fee1_10k'), Variable('fee2_10k'), Variable('fee3_10k')
    CHOICE = Variable('CHOICE')

    ASC_paid = Beta('ASC_paid', 1.0, None, None, 0)
    B_DUR = Beta('B_DUR', -0.05, -5, 0, 0)  # Bounded negative

    B_FEE_MU = Beta('B_FEE_MU', -0.5, -10, 0, 0)  # Bounded negative
    # Sigma bounded [0.001, 1.5]: with μ=-0.5, 3σ range is [-5, +4]
    # Upper bound 1.5 allows heterogeneity while keeping most draws negative
    B_FEE_SIGMA = Beta('B_FEE_SIGMA', 0.2, 0.001, 1.5, 0)  # Bounded positive, realistic

    B_FEE_RND = B_FEE_MU + B_FEE_SIGMA * bioDraws('B_FEE_RND', 'NORMAL')

    V1 = ASC_paid + B_FEE_RND * fee1 + B_DUR * dur1
    V2 = ASC_paid + B_FEE_RND * fee2 + B_DUR * dur2
    V3 = B_FEE_RND * fee3 + B_DUR * dur3

    V = {1: V1, 2: V2, 3: V3}
    av = {1: 1, 2: 1, 3: 1}

    prob = models.logit(V, av, CHOICE)
    # PanelLikelihoodTrajectory aggregates probs across choice situations per individual
    return log(MonteCarlo(PanelLikelihoodTrajectory(prob))), 'MXL-RandomFee'

def model_mxl_random_both(database):
    """Mixed Logit with random fee and duration.

    Uses PanelLikelihoodTrajectory for panel data (same draws across choice situations).
    """
    dur1, dur2, dur3 = Variable('dur1'), Variable('dur2'), Variable('dur3')
    fee1, fee2, fee3 = Variable('fee1_10k'), Variable('fee2_10k'), Variable('fee3_10k')
    CHOICE = Variable('CHOICE')

    ASC_paid = Beta('ASC_paid', 1.0, None, None, 0)

    B_FEE_MU = Beta('B_FEE_MU', -0.5, -10, 0, 0)  # Bounded negative
    # Sigma bounded realistically to avoid implausible heterogeneity
    B_FEE_SIGMA = Beta('B_FEE_SIGMA', 0.2, 0.001, 1.5, 0)  # Bounded positive, realistic
    B_DUR_MU = Beta('B_DUR_MU', -0.05, -5, 0, 0)  # Bounded negative
    # Duration sigma: with μ=-0.05, σ=0.5 gives range [-1.55, +1.45], reasonable
    B_DUR_SIGMA = Beta('B_DUR_SIGMA', 0.02, 0.001, 0.5, 0)  # Bounded positive, realistic

    B_FEE_RND = B_FEE_MU + B_FEE_SIGMA * bioDraws('B_FEE_RND', 'NORMAL')
    B_DUR_RND = B_DUR_MU + B_DUR_SIGMA * bioDraws('B_DUR_RND', 'NORMAL')

    V1 = ASC_paid + B_FEE_RND * fee1 + B_DUR_RND * dur1
    V2 = ASC_paid + B_FEE_RND * fee2 + B_DUR_RND * dur2
    V3 = B_FEE_RND * fee3 + B_DUR_RND * dur3

    V = {1: V1, 2: V2, 3: V3}
    av = {1: 1, 2: 1, 3: 1}

    prob = models.logit(V, av, CHOICE)
    # PanelLikelihoodTrajectory aggregates probs across choice situations per individual
    return log(MonteCarlo(PanelLikelihoodTrajectory(prob))), 'MXL-RandomBoth'

# =============================================================================
# HCM (HYBRID CHOICE) MODELS
# =============================================================================
# NOTE: These models use a TWO-STAGE approach with LV proxies.
# This causes ATTENUATION BIAS - LV effects are biased toward zero.
# Significant effects are likely real; non-significant may be attenuated.
# For unbiased estimates, full ICLV with simultaneous estimation is needed.
# =============================================================================

def compute_cronbach_alpha(df: pd.DataFrame, items: list) -> float:
    """Compute Cronbach's alpha reliability coefficient for a set of items.

    Alpha measures internal consistency of multi-item scales.
    Higher alpha (>0.7) indicates more reliable measurement.

    Formula: α = (k / (k-1)) * (1 - Σvar(item) / var(total))

    Returns:
        Cronbach's alpha (0 to 1, higher is better)
    """
    available = [c for c in items if c in df.columns]
    if len(available) < 2:
        return np.nan

    k = len(available)
    item_vars = df[available].var(ddof=1)
    total_var = df[available].sum(axis=1).var(ddof=1)

    if total_var == 0:
        return np.nan

    alpha = (k / (k - 1)) * (1 - item_vars.sum() / total_var)
    return float(alpha)


def model_hcm_basic(database):
    """HCM with latent variable proxies (basic - 2 LVs).

    ATTENUATION BIAS WARNING:
    ========================
    Using proxy scores instead of true latent variables introduces measurement
    error, which attenuates (biases toward zero) the estimated LV effects.

    The true effect is approximately: β_true ≈ β_estimated / reliability

    Where reliability can be estimated using Cronbach's alpha of the items.
    For α=0.8, effects are underestimated by ~20%.

    This is a known limitation of the two-stage approach. For unbiased
    estimates, consider ICLV (Integrated Choice and Latent Variable) models
    that jointly estimate the measurement and structural models.
    """
    dur1, dur2, dur3 = Variable('dur1'), Variable('dur2'), Variable('dur3')
    fee1, fee2, fee3 = Variable('fee1_10k'), Variable('fee2_10k'), Variable('fee3_10k')
    CHOICE = Variable('CHOICE')
    pat_blind = Variable('pat_blind_proxy')
    sec_dl = Variable('sec_dl_proxy')

    ASC_paid = Beta('ASC_paid', 1.0, None, None, 0)
    B_FEE = Beta('B_FEE', -0.5, -10, 0, 0)  # Bounded negative
    B_DUR = Beta('B_DUR', -0.05, -5, 0, 0)  # Bounded negative

    # Latent variable interactions
    B_FEE_PAT = Beta('B_FEE_PAT', -0.05, -2, 2, 0)
    B_FEE_SEC = Beta('B_FEE_SEC', 0.02, -2, 2, 0)
    B_DUR_PAT = Beta('B_DUR_PAT', 0.02, -1, 1, 0)
    B_DUR_SEC = Beta('B_DUR_SEC', -0.02, -1, 1, 0)

    B_FEE_i = B_FEE + B_FEE_PAT * pat_blind + B_FEE_SEC * sec_dl
    B_DUR_i = B_DUR + B_DUR_PAT * pat_blind + B_DUR_SEC * sec_dl

    V1 = ASC_paid + B_FEE_i * fee1 + B_DUR_i * dur1
    V2 = ASC_paid + B_FEE_i * fee2 + B_DUR_i * dur2
    V3 = B_FEE_i * fee3 + B_DUR_i * dur3

    V = {1: V1, 2: V2, 3: V3}
    av = {1: 1, 2: 1, 3: 1}

    return models.loglogit(V, av, CHOICE), 'HCM-Basic'

def model_hcm_full(database):
    """HCM with all 4 latent variable proxies."""
    dur1, dur2, dur3 = Variable('dur1'), Variable('dur2'), Variable('dur3')
    fee1, fee2, fee3 = Variable('fee1_10k'), Variable('fee2_10k'), Variable('fee3_10k')
    CHOICE = Variable('CHOICE')
    pat_blind = Variable('pat_blind_proxy')
    pat_const = Variable('pat_const_proxy')
    sec_dl = Variable('sec_dl_proxy')
    sec_fp = Variable('sec_fp_proxy')

    ASC_paid = Beta('ASC_paid', 1.0, None, None, 0)
    B_FEE = Beta('B_FEE', -0.5, -10, 0, 0)  # Bounded negative
    B_DUR = Beta('B_DUR', -0.05, -5, 0, 0)  # Bounded negative

    # Fee interactions with all LVs
    B_FEE_PAT_B = Beta('B_FEE_PAT_B', -0.05, -2, 2, 0)
    B_FEE_PAT_C = Beta('B_FEE_PAT_C', -0.02, -2, 2, 0)
    B_FEE_SEC_D = Beta('B_FEE_SEC_D', 0.02, -2, 2, 0)
    B_FEE_SEC_F = Beta('B_FEE_SEC_F', 0.02, -2, 2, 0)

    # Duration interactions
    B_DUR_PAT_B = Beta('B_DUR_PAT_B', 0.02, -1, 1, 0)
    B_DUR_SEC_D = Beta('B_DUR_SEC_D', -0.02, -1, 1, 0)

    B_FEE_i = B_FEE + B_FEE_PAT_B*pat_blind + B_FEE_PAT_C*pat_const + B_FEE_SEC_D*sec_dl + B_FEE_SEC_F*sec_fp
    B_DUR_i = B_DUR + B_DUR_PAT_B*pat_blind + B_DUR_SEC_D*sec_dl

    V1 = ASC_paid + B_FEE_i * fee1 + B_DUR_i * dur1
    V2 = ASC_paid + B_FEE_i * fee2 + B_DUR_i * dur2
    V3 = B_FEE_i * fee3 + B_DUR_i * dur3

    V = {1: V1, 2: V2, 3: V3}
    av = {1: 1, 2: 1, 3: 1}

    return models.loglogit(V, av, CHOICE), 'HCM-Full'

# =============================================================================
# WARM-START UTILITIES
# =============================================================================

def get_starting_values(previous_result: dict, target_params: list = None) -> dict:
    """Extract starting values from a previous model result.

    This implements WARM-START: using estimates from a simpler model
    as starting values for a more complex model. This can:
    - Speed up convergence
    - Avoid local optima
    - Improve numerical stability

    Args:
        previous_result: Result dict from estimate_model()
        target_params: List of parameter names to extract (None = all)

    Returns:
        Dict of {param_name: estimated_value}

    Example:
        # Estimate simple model first
        basic_result = estimate_model(database, model_mnl_basic)

        # Use its estimates as starting values for complex model
        starting = get_starting_values(basic_result, ['ASC_paid', 'B_FEE', 'B_DUR'])
        # Then use these values in your Beta() definitions
    """
    if previous_result is None or 'betas' not in previous_result:
        return {}

    betas = previous_result['betas']

    if target_params is None:
        return betas.copy()

    return {p: betas[p] for p in target_params if p in betas}


# =============================================================================
# ESTIMATION AND COMPARISON
# =============================================================================

def get_warm_start_values(previous_result, current_model_params, verbose: bool = False):
    """Extract starting values from previous model for warm-start.

    WARM-START IMPROVEMENT (Issue #17):
    - Now reports which parameters were transferred vs. using defaults
    - Handles parameter name variations (e.g., B_FEE -> B_FEE_MU)

    Args:
        previous_result: Result dict from a simpler model
        current_model_params: List of parameter names in current model
        verbose: If True, print detailed parameter matching info

    Returns:
        Dict of {param_name: starting_value} for matching parameters
    """
    if previous_result is None:
        return {}

    warm_start = {}
    prev_betas = previous_result.get('betas', {})

    # Parameter name variations for smart matching
    # E.g., B_FEE from MNL can warm-start B_FEE_MU in MXL
    param_variations = {
        'B_FEE_MU': ['B_FEE'],
        'B_DUR_MU': ['B_DUR'],
        'B_FEE': ['B_FEE_MU'],
        'B_DUR': ['B_DUR_MU'],
    }

    matched = []
    not_matched = []

    for param in current_model_params:
        if param in prev_betas:
            warm_start[param] = prev_betas[param]
            matched.append(param)
        else:
            # Try variations
            variations = param_variations.get(param, [])
            found = False
            for var in variations:
                if var in prev_betas:
                    warm_start[param] = prev_betas[var]
                    matched.append(f"{param} (from {var})")
                    found = True
                    break
            if not found:
                not_matched.append(param)

    if verbose and (matched or not_matched):
        print(f"    Warm-start: {len(matched)} params transferred, {len(not_matched)} using defaults")
        if not_matched and len(not_matched) <= 5:
            print(f"    New params (using defaults): {', '.join(not_matched)}")

    return warm_start


def estimate_model(database, model_func, n_draws=None, output_dir=None, warm_start=None):
    """Estimate a model and return results.

    Args:
        database: Biogeme database
        model_func: Function returning (logprob, name)
        n_draws: Number of draws for MXL models
        output_dir: Directory for output files
        warm_start: Dict of {param_name: starting_value} for warm-start

    Returns:
        Tuple of (result_dict, biogeme_results) where biogeme_results is the
        raw Biogeme EstimationResults object (for LaTeX generation, etc.)
    """
    logprob, name = model_func(database)

    if n_draws:
        biogeme_obj = bio.BIOGEME(database, logprob, number_of_draws=n_draws)
    else:
        biogeme_obj = bio.BIOGEME(database, logprob)

    # Apply warm-start values if provided
    if warm_start:
        biogeme_obj.change_init_values(warm_start)

    # Set model name with output directory path for HTML files
    model_name = name.replace('-', '_').replace(' ', '_')
    if output_dir:
        biogeme_obj.model_name = str(output_dir / model_name)
    else:
        biogeme_obj.model_name = model_name

    results = biogeme_obj.estimate()

    betas = results.get_beta_values()
    std_errs = {}
    for p in betas:
        try:
            std_errs[p] = results.get_parameter_std_err(p)
        except:
            std_errs[p] = np.nan

    # ISSUE #30 FIX: Validate log-likelihood before storing
    ll = results.final_loglikelihood
    ll_valid = True
    ll_warnings = []

    if np.isnan(ll):
        ll_warnings.append("Log-likelihood is NaN")
        ll_valid = False
    elif np.isinf(ll):
        ll_warnings.append("Log-likelihood is infinite")
        ll_valid = False
    elif ll > 0:
        ll_warnings.append(f"Log-likelihood is positive ({ll:.2f}) - should be negative")
        ll_valid = False

    if ll_warnings:
        print(f"  🚨 LL VALIDATION FAILED:")
        for w in ll_warnings:
            print(f"     - {w}")

    result_dict = {
        'name': name,
        'll': ll,
        'll_valid': ll_valid,
        'k': results.number_of_free_parameters,
        'aic': results.akaike_information_criterion if ll_valid else np.nan,
        'bic': results.bayesian_information_criterion if ll_valid else np.nan,
        'betas': betas,
        'std_errs': std_errs,
        'converged': results.algorithm_has_converged and ll_valid
    }

    # Run identification diagnostics (Issue #14)
    if HAS_ID_DIAGNOSTICS and results.algorithm_has_converged:
        try:
            diag = IdentificationDiagnostics(results)
            report = diag.full_report()

            # Check for identification issues
            id_warnings = []
            if report.get('has_near_zero_eigenvalues'):
                id_warnings.append("Near-zero eigenvalues detected")
            if report.get('is_ill_conditioned'):
                id_warnings.append(f"Ill-conditioned Hessian (κ={report.get('condition_number', 'N/A'):.1e})")
            if report.get('high_correlation_pairs'):
                n_pairs = len(report['high_correlation_pairs'])
                id_warnings.append(f"{n_pairs} high-correlation parameter pairs (>0.95)")
            if report.get('poorly_identified_params'):
                n_poor = len(report['poorly_identified_params'])
                id_warnings.append(f"{n_poor} parameters with SE > 10x value")

            if id_warnings:
                print(f"  ⚠️  IDENTIFICATION ISSUES:")
                for w in id_warnings:
                    print(f"     - {w}")

            result_dict['identification_diagnostics'] = report
        except Exception as e:
            # Diagnostics failed but don't break estimation
            result_dict['identification_diagnostics'] = {'error': str(e)}

    return result_dict, results

def compare_to_true(result, true_params, min_true_value: float = 0.01):
    """Compare estimated parameters to true values.

    ISSUE #24 FIX: Bias% now has minimum threshold to avoid misleading
    percentages for small true values.

    ISSUE #25 FIX: Parameter mapping is data-driven with fallback patterns.

    Args:
        result: Estimation result dict with 'betas' and 'std_errs'
        true_params: Dict of true parameter values from config
        min_true_value: Minimum absolute true value for Bias% calculation.
                       For |true| < min_true_value, Bias% is reported as N/A
                       to avoid misleading percentages (e.g., 100% bias for
                       true=0.001, est=0.002). Default: 0.01

    Returns:
        DataFrame with parameter comparison
    """
    # Parameter name mapping - ISSUE #25: explicit mapping with patterns
    mapping = {
        'ASC_paid': 'ASC_paid',
        'B_FEE': 'b_fee_scaled_base',
        'B_FEE_MU': 'b_fee_scaled_base',
        'B_FEE_AGE': 'b_fee_scaled_x_age_idx',
        'B_FEE_EDU': 'b_fee_scaled_x_edu_idx',
        'B_FEE_INC': 'b_fee_scaled_x_income_indiv_idx',
        'B_FEE_INC_H': 'b_fee_scaled_x_income_house_idx',
        'B_FEE_PAT_B': 'b_fee_scaled_x_pat_blind',
        'B_FEE_PAT_C': 'b_fee_scaled_x_pat_constructive',
        'B_FEE_SEC_D': 'b_fee_scaled_x_sec_dl',
        'B_FEE_SEC_F': 'b_fee_scaled_x_sec_fp',
        'B_DUR': 'b_dur_base',
        'B_DUR_MU': 'b_dur_base',
        'B_DUR_EDU': 'b_dur_x_edu_idx',
        'B_DUR_INC': 'b_dur_x_income_indiv_idx',
        'B_DUR_INC_H': 'b_dur_x_income_house_idx',
        'B_DUR_MARITAL': 'b_dur_x_marital_idx',
        'B_DUR_PAT_B': 'b_dur_x_pat_blind',
        'B_DUR_PAT_C': 'b_dur_x_pat_constructive',
        'B_DUR_SEC_D': 'b_dur_x_sec_dl',
        'B_DUR_SEC_F': 'b_dur_x_sec_fp',
    }

    # ISSUE #25: Fallback pattern matching for unmapped parameters
    def get_true_key(param):
        """Get true parameter key with fallback pattern matching."""
        if param in mapping:
            return mapping[param]

        # Try pattern-based matching
        param_lower = param.lower()

        # B_ATTR_DEMO -> b_attr_x_demo_idx
        if param.startswith('B_') and '_' in param[2:]:
            parts = param[2:].split('_', 1)
            if len(parts) == 2:
                attr, demo = parts
                candidates = [
                    f'b_{attr.lower()}_x_{demo.lower()}_idx',
                    f'b_{attr.lower()}_scaled_x_{demo.lower()}_idx',
                    f'b_{attr.lower()}_x_{demo.lower()}',
                ]
                for cand in candidates:
                    if cand in true_params:
                        return cand

        return None

    comparison = []
    for param, est_val in result['betas'].items():
        true_key = get_true_key(param)
        true_val = true_params.get(true_key) if true_key else None
        se = result['std_errs'].get(param, np.nan)
        t_stat = est_val / se if se and se > 0 else np.nan

        if true_val is not None:
            bias = est_val - true_val

            # ISSUE #24 FIX: Check minimum true value before computing Bias%
            if abs(true_val) >= min_true_value:
                bias_pct = (bias / abs(true_val) * 100)
            else:
                # True value too small - Bias% would be misleading
                bias_pct = np.nan

            ci_lower = est_val - 1.96 * se if not np.isnan(se) else np.nan
            ci_upper = est_val + 1.96 * se if not np.isnan(se) else np.nan
            covered = ci_lower <= true_val <= ci_upper if not np.isnan(ci_lower) else None
        else:
            bias = bias_pct = np.nan
            covered = None

        comparison.append({
            'Parameter': param,
            'True': true_val,
            'Estimated': est_val,
            'SE': se,
            't-stat': t_stat,
            'Bias%': bias_pct,
            'Covered': covered
        })

    return pd.DataFrame(comparison)

def run_all_models(data_path: str = "data/full_scale_test.csv"):
    """Run all models and compare to true parameters."""

    print("="*80)
    print("COMPREHENSIVE MODEL VALIDATION")
    print("="*80)

    # Setup output directory and cleanup
    output_dir = Path("results/all_models")
    cleanup_iter_files()  # Delete .iter files from project root
    cleanup_results_directory(output_dir)  # Clean and recreate output directory
    cleanup_latex_output()  # Clean and recreate latex_output directory
    print(f"Output directory: {output_dir.absolute()}")

    # Load true parameters
    true_params = load_true_params()
    print("\nTrue Parameters (from model_config.json):")
    for k, v in sorted(true_params.items()):
        if not k.startswith('_'):
            print(f"  {k}: {v}")

    # Prepare data
    df, database = prepare_data(data_path)
    n_obs = len(df)
    null_ll = n_obs * np.log(1/3)

    # Generate simulation LaTeX outputs
    print("\nGenerating simulation LaTeX outputs...")
    generate_all_simulation_latex(df)

    # Define models to run
    mnl_models = [
        (model_mnl_basic, None),
        (model_mnl_demographics, None),
        (model_mnl_full, None),
    ]

    # MXL Configuration
    # Set PRODUCTION_MODE = True for publication-quality results (slower)
    # Set PRODUCTION_MODE = False for development/testing (faster)
    PRODUCTION_MODE = True  # Full-scale test with publication-ready draws

    # NOTE: 10,000 draws causes OOM during Hessian calculation with panel data
    # NOTE: 5,000 draws ALSO causes OOM during Hessian with 1000x10 panel data
    # 2,000 draws is minimum for stable estimates with this data size
    MXL_DRAWS = 2000 if PRODUCTION_MODE else 1000

    # ISSUE #28 FIX: Document draw requirements and add stability warning
    # Recommendations from literature:
    # - Minimum 500 draws for exploratory analysis
    # - 1000+ draws for stable estimates
    # - 5000+ draws for publication-quality results
    # - Consider Halton or Sobol sequences instead of pseudo-random
    if MXL_DRAWS < 1000 and not PRODUCTION_MODE:
        print(f"\n⚠️  MXL using {MXL_DRAWS} draws (development mode)")
        print("   For stable estimates, use 1000+ draws (set PRODUCTION_MODE=True)")
        print("   Publication-quality results require 5000+ draws")

    mxl_models = [
        # MXL with panel data is computationally intensive
        # Uses 5000 draws in production mode, 1000 in dev mode
        (model_mxl_random_fee, MXL_DRAWS),
        (model_mxl_random_both, MXL_DRAWS),
    ]

    all_results = []
    baseline_result = None  # For warm-start chain

    # Run MNL models with warm-start chain
    print("\n" + "="*80)
    print("MNL MODELS")
    print("="*80)

    previous_result = None
    for model_func, n_draws in mnl_models:
        print(f"\nEstimating {model_func.__name__}...")
        try:
            # Use warm-start from previous model if available
            warm_start = None
            if previous_result:
                warm_start = get_warm_start_values(previous_result, list(previous_result['betas'].keys()))
                if warm_start:
                    print(f"  Using warm-start from previous model ({len(warm_start)} params)")

            result, biogeme_results = estimate_model(database, model_func, n_draws, output_dir, warm_start)
            result['rho2'] = 1 - (result['ll'] / null_ll)
            all_results.append(result)
            previous_result = result

            # Generate LaTeX output
            if result['converged']:
                generate_latex_output(biogeme_results, result['name'])

            # Save baseline for other model families
            if model_func == model_mnl_basic:
                baseline_result = result

            # Print summary
            print(f"  LL: {result['ll']:.2f} | K: {result['k']} | AIC: {result['aic']:.2f} | ρ²: {result['rho2']:.4f} | Conv: {result['converged']}")

            # Compare key params to true (only if converged - otherwise meaningless)
            if result['converged']:
                comp = compare_to_true(result, true_params)
                key_params = ['ASC_paid', 'B_FEE', 'B_DUR']
                key_comp = comp[comp['Parameter'].isin(key_params)]
                if len(key_comp) > 0 and key_comp['Bias%'].notna().any():
                    mean_bias = key_comp['Bias%'].abs().mean()
                    print(f"  Key params mean |Bias%|: {mean_bias:.1f}%")
            else:
                print(f"  ⚠️  SKIPPING parameter comparison - model did not converge")
        except Exception as e:
            print(f"  ERROR: {e}")

    # Run MXL models
    # NOTE: MXL requires panel structure to properly account for repeated
    # observations by the same individual. Create separate database with panel ID.
    print("\n" + "="*80)
    print("MXL (MIXED LOGIT) MODELS")
    print("="*80)
    print("Creating panel-aware database for MXL...")
    mxl_database = prepare_mxl_database(df)

    previous_mxl_result = None
    for model_func, n_draws in mxl_models:
        print(f"\nEstimating {model_func.__name__} (draws={n_draws})...")
        try:
            # Use warm-start from baseline MNL or previous MXL
            warm_start = None
            if previous_mxl_result:
                warm_start = get_warm_start_values(previous_mxl_result, list(previous_mxl_result['betas'].keys()))
            elif baseline_result:
                warm_start = get_warm_start_values(baseline_result, list(baseline_result['betas'].keys()))
            if warm_start:
                print(f"  Using warm-start ({len(warm_start)} params)")

            result, biogeme_results = estimate_model(mxl_database, model_func, n_draws, output_dir, warm_start)
            result['rho2'] = 1 - (result['ll'] / null_ll)
            all_results.append(result)
            previous_mxl_result = result

            # Generate LaTeX output
            if result['converged']:
                generate_latex_output(biogeme_results, result['name'])

            print(f"  LL: {result['ll']:.2f} | K: {result['k']} | AIC: {result['aic']:.2f} | ρ²: {result['rho2']:.4f} | Conv: {result['converged']}")

            # Compare key params to true (only if converged - otherwise meaningless)
            if result['converged']:
                comp = compare_to_true(result, true_params)
                key_params = ['ASC_paid', 'B_FEE_MU', 'B_DUR', 'B_DUR_MU']
                key_comp = comp[comp['Parameter'].isin(key_params)]
                if len(key_comp) > 0 and key_comp['Bias%'].notna().any():
                    mean_bias = key_comp['Bias%'].abs().mean()
                    print(f"  Key params mean |Bias%|: {mean_bias:.1f}%")
            else:
                print(f"  ⚠️  SKIPPING parameter comparison - model did not converge")
        except Exception as e:
            print(f"  ERROR: {e}")

    # Run HCM models
    print("\n" + "="*80)
    print("HCM (HYBRID CHOICE) MODELS")
    print("="*80)

    # Compute and report reliability (Cronbach's alpha) for LV proxies
    # This is critical for understanding attenuation bias
    lv_items_for_alpha = {
        'pat_blind': ['pat_blind_1', 'pat_blind_2', 'pat_blind_3', 'pat_blind_4'],
        'pat_const': ['pat_constructive_1', 'pat_constructive_2', 'pat_constructive_3', 'pat_constructive_4'],
        'sec_dl': ['sec_dl_1', 'sec_dl_2', 'sec_dl_3', 'sec_dl_4'],
        'sec_fp': ['sec_fp_1', 'sec_fp_2', 'sec_fp_3', 'sec_fp_4'],
    }

    # Use original df for alpha computation (df_num dropped some columns)
    df_for_alpha = df  # df still has Likert columns needed for reliability

    print("\nLATENT VARIABLE RELIABILITY (Cronbach's Alpha):")
    print("-" * 50)
    reliability_info = {}
    for lv_name, items in lv_items_for_alpha.items():
        alpha = compute_cronbach_alpha(df_for_alpha, items)
        reliability_info[lv_name] = alpha
        if pd.notna(alpha):
            attenuation = (1 - alpha) * 100
            print(f"  {lv_name}: α = {alpha:.3f} (LV effects attenuated ~{attenuation:.0f}%)")
        else:
            print(f"  {lv_name}: α = N/A (insufficient items)")

    print("\n⚠️  ATTENUATION BIAS WARNING:")
    print("   Two-stage HCM using proxy scores underestimates LV effects.")
    print("   True effect ≈ estimated / reliability")
    print("   For unbiased estimates, use full ICLV (joint estimation).")
    print("-" * 50)

    hcm_models = [
        (model_hcm_basic, None),
        (model_hcm_full, None),
    ]

    previous_hcm_result = None
    for model_func, n_draws in hcm_models:
        print(f"\nEstimating {model_func.__name__}...")
        try:
            # Use warm-start from baseline MNL or previous HCM
            warm_start = None
            if previous_hcm_result:
                warm_start = get_warm_start_values(previous_hcm_result, list(previous_hcm_result['betas'].keys()))
            elif baseline_result:
                warm_start = get_warm_start_values(baseline_result, list(baseline_result['betas'].keys()))
            if warm_start:
                print(f"  Using warm-start ({len(warm_start)} params)")

            result, biogeme_results = estimate_model(database, model_func, n_draws, output_dir, warm_start)
            result['rho2'] = 1 - (result['ll'] / null_ll)
            all_results.append(result)
            previous_hcm_result = result

            # Generate LaTeX output
            if result['converged']:
                generate_latex_output(biogeme_results, result['name'])

            print(f"  LL: {result['ll']:.2f} | K: {result['k']} | AIC: {result['aic']:.2f} | ρ²: {result['rho2']:.4f} | Conv: {result['converged']}")

            # Compare key params to true (only if converged - otherwise meaningless)
            if result['converged']:
                comp = compare_to_true(result, true_params)
                key_params = ['ASC_paid', 'B_FEE', 'B_DUR']
                key_comp = comp[comp['Parameter'].isin(key_params)]
                if len(key_comp) > 0 and key_comp['Bias%'].notna().any():
                    mean_bias = key_comp['Bias%'].abs().mean()
                    print(f"  Key params mean |Bias%|: {mean_bias:.1f}%")
            else:
                print(f"  ⚠️  SKIPPING parameter comparison - model did not converge")
        except Exception as e:
            print(f"  ERROR: {e}")

    # ==========================================================================
    # ICLV MODELS (Simultaneous Estimation - Unbiased)
    # ==========================================================================
    print("\n" + "="*80)
    print("ICLV MODELS (Simultaneous Estimation)")
    print("="*80)
    print("\nICLV eliminates attenuation bias by jointly estimating")
    print("measurement and choice models via Simulated Maximum Likelihood.")

    try:
        from src.models.iclv import estimate_iclv, ICLVResult

        # Define constructs for ICLV (same as HCM)
        iclv_constructs = {
            'pat_blind': [c for c in df.columns if c.startswith('pat_blind_') and c[-1].isdigit()],
            'pat_const': [c for c in df.columns if c.startswith('pat_constructive_') and c[-1].isdigit()],
            'sec_dl': [c for c in df.columns if c.startswith('sec_dl_') and c[-1].isdigit()],
            'sec_fp': [c for c in df.columns if c.startswith('sec_fp_') and c[-1].isdigit()],
        }

        # Define covariates for structural model
        iclv_covariates = ['age_idx', 'edu_idx', 'income_indiv_idx']
        iclv_covariates = [c for c in iclv_covariates if c in df.columns]

        # Define LV effects in utility
        lv_effects = {
            'pat_blind': 'B_FEE_PatBlind',
            'sec_dl': 'B_FEE_SecDL',
        }

        print(f"\nEstimating ICLV with {len(iclv_constructs)} latent constructs...")
        print(f"  Constructs: {list(iclv_constructs.keys())}")
        print(f"  Covariates: {iclv_covariates}")
        print(f"  LV effects: {list(lv_effects.keys())}")

        # Run ICLV estimation
        iclv_result = estimate_iclv(
            df=df,
            constructs=iclv_constructs,
            covariates=iclv_covariates,
            choice_col='CHOICE',
            attribute_cols=['fee1_10k', 'fee2_10k', 'fee3_10k', 'dur1', 'dur2', 'dur3'],
            lv_effects=lv_effects,
            n_draws=500,
            n_categories=5,
            verbose=True
        )

        # Convert ICLV result to standard format for comparison
        if iclv_result is not None:
            iclv_summary = {
                'name': 'ICLV: Simultaneous',
                'll': iclv_result.log_likelihood if hasattr(iclv_result, 'log_likelihood') else np.nan,
                'k': iclv_result.n_parameters if hasattr(iclv_result, 'n_parameters') else 0,
                'aic': iclv_result.aic if hasattr(iclv_result, 'aic') else np.nan,
                'bic': iclv_result.bic if hasattr(iclv_result, 'bic') else np.nan,
                'rho2': 1 - (iclv_result.log_likelihood / null_ll) if hasattr(iclv_result, 'log_likelihood') else np.nan,
                'converged': iclv_result.converged if hasattr(iclv_result, 'converged') else True,
                'betas': iclv_result.beta if hasattr(iclv_result, 'beta') else {},
                't_stats': {},
            }
            all_results.append(iclv_summary)
            print(f"\n  ICLV estimation complete!")
            if hasattr(iclv_result, 'log_likelihood'):
                print(f"  LL: {iclv_result.log_likelihood:.2f}")

    except ImportError as e:
        print(f"\n  ICLV module not available: {e}")
        print("  Skipping ICLV estimation. Install dependencies or check imports.")
    except Exception as e:
        print(f"\n  ERROR in ICLV estimation: {e}")
        import traceback
        traceback.print_exc()

    # ==========================================================================
    # EXTENDED MODEL SPECIFICATIONS
    # ==========================================================================
    # Run extended model comparisons with alternative functional forms,
    # distributions, and interaction patterns.

    extended_results = {}

    # MNL Extended Models
    if HAS_MNL_EXTENDED:
        print("\n" + "="*80)
        print("EXTENDED MNL MODELS (8 specifications)")
        print("="*80)
        try:
            mnl_ext_results, mnl_ext_df = run_mnl_extended(
                data_path=data_path,
                output_dir=str(output_dir / 'mnl_extended')
            )
            extended_results['mnl_extended'] = {
                'results': mnl_ext_results,
                'comparison': mnl_ext_df
            }
            # Add best extended MNL to main comparison
            if mnl_ext_df is not None and len(mnl_ext_df) > 0:
                best_mnl_ext = mnl_ext_df.loc[mnl_ext_df['AIC'].idxmin()]
                best_mnl_ext_result = next(
                    (r for r in mnl_ext_results if r.name == best_mnl_ext['Model']),
                    None
                )
                if best_mnl_ext_result:
                    all_results.append({
                        'name': f"MNL-Ext: {best_mnl_ext_result.name}",
                        'll': best_mnl_ext_result.ll,
                        'k': best_mnl_ext_result.k,
                        'aic': best_mnl_ext_result.aic,
                        'bic': best_mnl_ext_result.bic,
                        'rho2': best_mnl_ext_result.rho2,
                        'converged': best_mnl_ext_result.converged,
                        'betas': best_mnl_ext_result.params,
                        'std_errs': best_mnl_ext_result.std_errs,
                    })
        except Exception as e:
            print(f"  ERROR in extended MNL: {e}")
    else:
        print("\n  MNL Extended module not available")

    # MXL Extended Models
    if HAS_MXL_EXTENDED:
        print("\n" + "="*80)
        print("EXTENDED MXL MODELS (8 specifications)")
        print("="*80)
        try:
            # Use fewer draws for extended models to save time
            MXL_EXT_DRAWS = 500 if PRODUCTION_MODE else 300
            mxl_ext_results, mxl_ext_df = run_mxl_extended(
                data_path=data_path,
                output_dir=str(output_dir / 'mxl_extended'),
                n_draws=MXL_EXT_DRAWS
            )
            extended_results['mxl_extended'] = {
                'results': mxl_ext_results,
                'comparison': mxl_ext_df
            }
            # Add best extended MXL to main comparison
            if mxl_ext_df is not None and len(mxl_ext_df) > 0:
                best_mxl_ext = mxl_ext_df.loc[mxl_ext_df['AIC'].idxmin()]
                best_mxl_ext_result = next(
                    (r for r in mxl_ext_results if r.name == best_mxl_ext['Model']),
                    None
                )
                if best_mxl_ext_result:
                    all_results.append({
                        'name': f"MXL-Ext: {best_mxl_ext_result.name}",
                        'll': best_mxl_ext_result.ll,
                        'k': best_mxl_ext_result.k,
                        'aic': best_mxl_ext_result.aic,
                        'bic': best_mxl_ext_result.bic,
                        'rho2': best_mxl_ext_result.rho2,
                        'converged': best_mxl_ext_result.converged,
                        'betas': best_mxl_ext_result.params,
                        'std_errs': best_mxl_ext_result.std_errs,
                    })
        except Exception as e:
            print(f"  ERROR in extended MXL: {e}")
    else:
        print("\n  MXL Extended module not available")

    # HCM Extended Models
    if HAS_HCM_EXTENDED:
        print("\n" + "="*80)
        print("EXTENDED HCM MODELS (8 specifications)")
        print("="*80)
        try:
            hcm_ext_results, hcm_ext_df = run_hcm_extended(
                data_path=data_path,
                output_dir=str(output_dir / 'hcm_extended')
            )
            extended_results['hcm_extended'] = {
                'results': hcm_ext_results,
                'comparison': hcm_ext_df
            }
            # Add best extended HCM to main comparison
            if hcm_ext_df is not None and len(hcm_ext_df) > 0:
                best_hcm_ext = hcm_ext_df.loc[hcm_ext_df['AIC'].idxmin()]
                best_hcm_ext_result = next(
                    (r for r in hcm_ext_results if r.name == best_hcm_ext['Model']),
                    None
                )
                if best_hcm_ext_result:
                    all_results.append({
                        'name': f"HCM-Ext: {best_hcm_ext_result.name}",
                        'll': best_hcm_ext_result.ll,
                        'k': best_hcm_ext_result.k,
                        'aic': best_hcm_ext_result.aic,
                        'bic': best_hcm_ext_result.bic,
                        'rho2': best_hcm_ext_result.rho2,
                        'converged': best_hcm_ext_result.converged,
                        'betas': best_hcm_ext_result.params,
                        'std_errs': best_hcm_ext_result.std_errs,
                    })
        except Exception as e:
            print(f"  ERROR in extended HCM: {e}")
    else:
        print("\n  HCM Extended module not available")

    # Save extended model comparisons
    if extended_results:
        print("\n" + "-"*80)
        print("EXTENDED MODEL COMPARISON FILES SAVED:")
        for family, data in extended_results.items():
            if data['comparison'] is not None:
                ext_path = output_dir / family / 'model_comparison.csv'
                print(f"  {ext_path}")

    # Summary table
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)

    summary_data = []
    for r in all_results:
        summary_data.append({
            'Model': r['name'],
            'LL': r['ll'],
            'K': r['k'],
            'AIC': r['aic'],
            'BIC': r['bic'],
            'ρ²': r['rho2'],
            'Conv': 'Yes' if r['converged'] else 'No'
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df['AIC_Rank'] = summary_df['AIC'].rank().astype(int)
    summary_df['BIC_Rank'] = summary_df['BIC'].rank().astype(int)

    print("\n" + summary_df.to_string(index=False))

    # Best model
    best_aic = summary_df.loc[summary_df['AIC'].idxmin()]
    best_bic = summary_df.loc[summary_df['BIC'].idxmin()]

    print(f"\nBest by AIC: {best_aic['Model']} (AIC={best_aic['AIC']:.2f})")
    print(f"Best by BIC: {best_bic['Model']} (BIC={best_bic['BIC']:.2f})")

    # Detailed parameter comparison for best model
    print("\n" + "="*80)
    print(f"DETAILED PARAMETER COMPARISON: {best_aic['Model']}")
    print("="*80)

    best_result = next(r for r in all_results if r['name'] == best_aic['Model'])

    # Only show detailed comparison if best model converged
    if not best_result['converged']:
        print(f"\n⚠️  WARNING: Best model by AIC did not converge!")
        print(f"   Parameter estimates may be unreliable. Consider:")
        print(f"   - Different starting values")
        print(f"   - Simplified model specification")
        print(f"   - More optimization iterations")
        comp_df = pd.DataFrame()  # Empty DataFrame for saving
    else:
        comp_df = compare_to_true(best_result, true_params)

        print(f"\n{'Parameter':<15} {'True':>10} {'Estimated':>10} {'SE':>10} {'t-stat':>8} {'Bias%':>8} {'Covered':>8}")
        print("-"*80)

        for _, row in comp_df.iterrows():
            true_str = f"{row['True']:.4f}" if pd.notna(row['True']) else "N/A"
            est_str = f"{row['Estimated']:.4f}"
            se_str = f"{row['SE']:.4f}" if pd.notna(row['SE']) else "N/A"
            t_str = f"{row['t-stat']:.2f}" if pd.notna(row['t-stat']) else "N/A"
            bias_str = f"{row['Bias%']:+.1f}%" if pd.notna(row['Bias%']) else "N/A"
            cov_str = "Yes" if row['Covered'] == True else "No" if row['Covered'] == False else "N/A"
            print(f"{row['Parameter']:<15} {true_str:>10} {est_str:>10} {se_str:>10} {t_str:>8} {bias_str:>8} {cov_str:>8}")

    # Save results
    output_dir = Path("results/all_models")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(output_dir / "model_comparison.csv", index=False)
    comp_df.to_csv(output_dir / "parameter_comparison.csv", index=False)

    print(f"\nResults saved to: {output_dir}")
    print(f"LaTeX output saved to: output/latex/")

    return all_results, summary_df

if __name__ == '__main__':
    run_all_models()
