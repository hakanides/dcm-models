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
"""

import numpy as np
import pandas as pd
import json
import shutil
import os
from pathlib import Path
import warnings
# Warning suppression for expected Biogeme optimization warnings
# These are normal during model estimation:
#   - FutureWarning: Biogeme API deprecation warnings
#   - overflow: Numerical overflow in exp() during early iterations
#   - divide by zero: Can occur during probability calculation
# To see all warnings for debugging, use: configure_warnings(debug_mode=True)
# from src.utils.logging_config import configure_warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*overflow.*')
warnings.filterwarnings('ignore', message='.*divide by zero.*')

import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta, Variable, bioDraws, MonteCarlo, log, PanelLikelihoodTrajectory


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

def load_true_params(config_path: str = "model_config.json") -> dict:
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

def prepare_data(data_path: str):
    """Load and prepare data for all models.

    IMPORTANT: Centering and scaling MUST match model_config.json exactly!
    The simulator uses these values from the config's interaction specifications.
    If you change these, parameter recovery will be biased.

    Config specifies: (variable - center) / scale
    """
    df = pd.read_csv(data_path)

    # Scale fees by 10,000 (standard across all configs and models)
    FEE_SCALE = 10000.0
    df['fee1_10k'] = df['fee1'] / FEE_SCALE
    df['fee2_10k'] = df['fee2'] / FEE_SCALE
    df['fee3_10k'] = df['fee3'] / FEE_SCALE

    # Center demographics - MUST match model_config.json interactions!
    # Format: (variable - center) / scale
    df['age_c'] = (df['age_idx'] - 2) / 2           # center=2.0, scale=2.0
    df['edu_c'] = (df['edu_idx'] - 3) / 2           # center=3.0, scale=2.0
    df['inc_c'] = (df['income_indiv_idx'] - 3) / 2  # center=3.0, scale=2.0
    df['inc_hh_c'] = (df['income_house_idx'] - 3) / 2  # center=3.0, scale=2.0
    df['marital_c'] = (df['marital_idx'] - 0.5) / 0.5  # center=0.5, scale=0.5

    # Create Likert proxies (standardized)
    lv_items = {
        'pat_blind': ['pat_blind_1', 'pat_blind_2', 'pat_blind_3', 'pat_blind_4'],
        'pat_const': ['pat_constructive_1', 'pat_constructive_2', 'pat_constructive_3', 'pat_constructive_4'],
        'sec_dl': ['sec_dl_1', 'sec_dl_2', 'sec_dl_3', 'sec_dl_4'],
        'sec_fp': ['sec_fp_1', 'sec_fp_2', 'sec_fp_3', 'sec_fp_4'],
    }

    for lv_name, items in lv_items.items():
        available = [c for c in items if c in df.columns]
        if available:
            proxy = df[available].mean(axis=1)
            df[f'{lv_name}_proxy'] = (proxy - proxy.mean()) / proxy.std()

    # Drop string columns
    string_cols = df.select_dtypes(include=['object']).columns.tolist()
    df_num = df.drop(columns=string_cols)

    print(f"Loaded {len(df):,} obs from {df['ID'].nunique()} respondents")

    # Choice share balance check
    shares = {
        1: df['CHOICE'].eq(1).mean(),
        2: df['CHOICE'].eq(2).mean(),
        3: df['CHOICE'].eq(3).mean()
    }
    print(f"Choice shares: 1={shares[1]:.1%}, 2={shares[2]:.1%}, 3={shares[3]:.1%}")

    # IMBALANCE WARNING: Extreme choice shares can cause identification issues
    for alt, share in shares.items():
        if share > 0.70:
            print(f"  ⚠️  WARNING: Alternative {alt} has {share:.1%} share - may cause identification issues")
        elif share < 0.10:
            print(f"  ⚠️  WARNING: Alternative {alt} has only {share:.1%} share - limited variation for estimation")

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
        print(f"  Panel structure: {n_individuals} individuals, {n_obs_per_person:.1f} obs/person avg")

    return database

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
    """Basic MNL - just ASC, fee, duration."""
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

def model_mnl_demographics(database):
    """MNL with demographic interactions."""
    dur1, dur2, dur3 = Variable('dur1'), Variable('dur2'), Variable('dur3')
    fee1, fee2, fee3 = Variable('fee1_10k'), Variable('fee2_10k'), Variable('fee3_10k')
    CHOICE = Variable('CHOICE')
    age_c, edu_c, inc_c = Variable('age_c'), Variable('edu_c'), Variable('inc_c')
    inc_hh_c, marital_c = Variable('inc_hh_c'), Variable('marital_c')

    ASC_paid = Beta('ASC_paid', 1.0, None, None, 0)
    B_FEE = Beta('B_FEE', -0.5, -10, 0, 0)  # Bounded negative
    B_FEE_AGE = Beta('B_FEE_AGE', 0.05, -2, 2, 0)
    B_FEE_EDU = Beta('B_FEE_EDU', 0.05, -2, 2, 0)
    B_FEE_INC = Beta('B_FEE_INC', 0.10, -2, 2, 0)
    B_FEE_INC_H = Beta('B_FEE_INC_H', 0.05, -2, 2, 0)

    B_DUR = Beta('B_DUR', -0.05, -5, 0, 0)  # Bounded negative
    B_DUR_EDU = Beta('B_DUR_EDU', -0.02, -1, 1, 0)
    B_DUR_INC = Beta('B_DUR_INC', -0.02, -1, 1, 0)
    B_DUR_INC_H = Beta('B_DUR_INC_H', -0.01, -1, 1, 0)
    B_DUR_MARITAL = Beta('B_DUR_MARITAL', -0.02, -1, 1, 0)

    B_FEE_i = B_FEE + B_FEE_AGE*age_c + B_FEE_EDU*edu_c + B_FEE_INC*inc_c + B_FEE_INC_H*inc_hh_c
    B_DUR_i = B_DUR + B_DUR_EDU*edu_c + B_DUR_INC*inc_c + B_DUR_INC_H*inc_hh_c + B_DUR_MARITAL*marital_c

    V1 = ASC_paid + B_FEE_i * fee1 + B_DUR_i * dur1
    V2 = ASC_paid + B_FEE_i * fee2 + B_DUR_i * dur2
    V3 = B_FEE_i * fee3 + B_DUR_i * dur3

    V = {1: V1, 2: V2, 3: V3}
    av = {1: 1, 2: 1, 3: 1}

    return models.loglogit(V, av, CHOICE), 'MNL-Demographics'

def model_mnl_full(database):
    """MNL with demographics + latent proxies."""
    dur1, dur2, dur3 = Variable('dur1'), Variable('dur2'), Variable('dur3')
    fee1, fee2, fee3 = Variable('fee1_10k'), Variable('fee2_10k'), Variable('fee3_10k')
    CHOICE = Variable('CHOICE')
    age_c, edu_c, inc_c = Variable('age_c'), Variable('edu_c'), Variable('inc_c')
    inc_hh_c, marital_c = Variable('inc_hh_c'), Variable('marital_c')
    pat_blind, pat_const = Variable('pat_blind_proxy'), Variable('pat_const_proxy')
    sec_dl, sec_fp = Variable('sec_dl_proxy'), Variable('sec_fp_proxy')

    ASC_paid = Beta('ASC_paid', 1.0, None, None, 0)

    B_FEE = Beta('B_FEE', -0.5, -10, 0, 0)  # Bounded negative
    B_FEE_AGE = Beta('B_FEE_AGE', 0.05, -2, 2, 0)
    B_FEE_EDU = Beta('B_FEE_EDU', 0.05, -2, 2, 0)
    B_FEE_INC = Beta('B_FEE_INC', 0.10, -2, 2, 0)
    B_FEE_INC_H = Beta('B_FEE_INC_H', 0.05, -2, 2, 0)
    B_FEE_PAT_B = Beta('B_FEE_PAT_B', -0.05, -2, 2, 0)
    B_FEE_PAT_C = Beta('B_FEE_PAT_C', -0.02, -2, 2, 0)
    B_FEE_SEC_D = Beta('B_FEE_SEC_D', 0.02, -2, 2, 0)
    B_FEE_SEC_F = Beta('B_FEE_SEC_F', 0.02, -2, 2, 0)

    B_DUR = Beta('B_DUR', -0.05, -5, 0, 0)  # Bounded negative
    B_DUR_EDU = Beta('B_DUR_EDU', -0.02, -1, 1, 0)
    B_DUR_INC = Beta('B_DUR_INC', -0.02, -1, 1, 0)
    B_DUR_INC_H = Beta('B_DUR_INC_H', -0.01, -1, 1, 0)
    B_DUR_MARITAL = Beta('B_DUR_MARITAL', -0.02, -1, 1, 0)
    B_DUR_PAT_B = Beta('B_DUR_PAT_B', 0.02, -1, 1, 0)
    B_DUR_PAT_C = Beta('B_DUR_PAT_C', 0.02, -1, 1, 0)
    B_DUR_SEC_D = Beta('B_DUR_SEC_D', -0.02, -1, 1, 0)
    B_DUR_SEC_F = Beta('B_DUR_SEC_F', -0.02, -1, 1, 0)

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
    B_FEE_SIGMA = Beta('B_FEE_SIGMA', 0.2, 0.001, 5, 0)  # Bounded positive

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
    B_FEE_SIGMA = Beta('B_FEE_SIGMA', 0.2, 0.001, 5, 0)  # Bounded positive
    B_DUR_MU = Beta('B_DUR_MU', -0.05, -5, 0, 0)  # Bounded negative
    B_DUR_SIGMA = Beta('B_DUR_SIGMA', 0.02, 0.001, 2, 0)  # Bounded positive

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

def model_hcm_basic(database):
    """HCM with latent variable proxies (basic - 2 LVs).

    WARNING: Two-stage approach causes attenuation bias in LV effects.
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

def get_warm_start_values(previous_result, current_model_params):
    """Extract starting values from previous model for warm-start.

    Args:
        previous_result: Result dict from a simpler model
        current_model_params: List of parameter names in current model

    Returns:
        Dict of {param_name: starting_value} for matching parameters
    """
    if previous_result is None:
        return {}

    warm_start = {}
    prev_betas = previous_result.get('betas', {})

    for param in current_model_params:
        if param in prev_betas:
            warm_start[param] = prev_betas[param]

    return warm_start


def estimate_model(database, model_func, n_draws=None, output_dir=None, warm_start=None):
    """Estimate a model and return results.

    Args:
        database: Biogeme database
        model_func: Function returning (logprob, name)
        n_draws: Number of draws for MXL models
        output_dir: Directory for output files
        warm_start: Dict of {param_name: starting_value} for warm-start
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

    return {
        'name': name,
        'll': results.final_loglikelihood,
        'k': results.number_of_free_parameters,
        'aic': results.akaike_information_criterion,
        'bic': results.bayesian_information_criterion,
        'betas': betas,
        'std_errs': std_errs,
        'converged': results.algorithm_has_converged
    }

def compare_to_true(result, true_params):
    """Compare estimated parameters to true values."""
    # Parameter name mapping
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

    comparison = []
    for param, est_val in result['betas'].items():
        true_key = mapping.get(param)
        true_val = true_params.get(true_key) if true_key else None
        se = result['std_errs'].get(param, np.nan)
        t_stat = est_val / se if se and se > 0 else np.nan

        if true_val is not None:
            bias = est_val - true_val
            bias_pct = (bias / abs(true_val) * 100) if true_val != 0 else np.nan
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

def run_all_models(data_path: str = "data/test_validation.csv"):
    """Run all models and compare to true parameters."""

    print("="*80)
    print("COMPREHENSIVE MODEL VALIDATION")
    print("="*80)

    # Setup output directory and cleanup
    output_dir = Path("results/all_models")
    cleanup_iter_files()  # Delete .iter files from project root
    cleanup_results_directory(output_dir)  # Clean and recreate output directory
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

    # Define models to run
    mnl_models = [
        (model_mnl_basic, None),
        (model_mnl_demographics, None),
        (model_mnl_full, None),
    ]

    # MXL Configuration
    # Set PRODUCTION_MODE = True for publication-quality results (slower)
    # Set PRODUCTION_MODE = False for development/testing (faster)
    PRODUCTION_MODE = False  # Change to True for final runs

    MXL_DRAWS = 5000 if PRODUCTION_MODE else 1000

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

            result = estimate_model(database, model_func, n_draws, output_dir, warm_start)
            result['rho2'] = 1 - (result['ll'] / null_ll)
            all_results.append(result)
            previous_result = result

            # Save baseline for other model families
            if model_func == model_mnl_basic:
                baseline_result = result

            # Print summary
            print(f"  LL: {result['ll']:.2f} | K: {result['k']} | AIC: {result['aic']:.2f} | ρ²: {result['rho2']:.4f} | Conv: {result['converged']}")

            # Compare key params to true
            comp = compare_to_true(result, true_params)
            key_params = ['ASC_paid', 'B_FEE', 'B_DUR']
            key_comp = comp[comp['Parameter'].isin(key_params)]
            if len(key_comp) > 0 and key_comp['Bias%'].notna().any():
                mean_bias = key_comp['Bias%'].abs().mean()
                print(f"  Key params mean |Bias%|: {mean_bias:.1f}%")
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

            result = estimate_model(mxl_database, model_func, n_draws, output_dir, warm_start)
            result['rho2'] = 1 - (result['ll'] / null_ll)
            all_results.append(result)
            previous_mxl_result = result

            print(f"  LL: {result['ll']:.2f} | K: {result['k']} | AIC: {result['aic']:.2f} | ρ²: {result['rho2']:.4f} | Conv: {result['converged']}")

            comp = compare_to_true(result, true_params)
            key_params = ['ASC_paid', 'B_FEE_MU', 'B_DUR', 'B_DUR_MU']
            key_comp = comp[comp['Parameter'].isin(key_params)]
            if len(key_comp) > 0 and key_comp['Bias%'].notna().any():
                mean_bias = key_comp['Bias%'].abs().mean()
                print(f"  Key params mean |Bias%|: {mean_bias:.1f}%")
        except Exception as e:
            print(f"  ERROR: {e}")

    # Run HCM models
    print("\n" + "="*80)
    print("HCM (HYBRID CHOICE) MODELS")
    print("="*80)

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

            result = estimate_model(database, model_func, n_draws, output_dir, warm_start)
            result['rho2'] = 1 - (result['ll'] / null_ll)
            all_results.append(result)
            previous_hcm_result = result

            print(f"  LL: {result['ll']:.2f} | K: {result['k']} | AIC: {result['aic']:.2f} | ρ²: {result['rho2']:.4f} | Conv: {result['converged']}")

            comp = compare_to_true(result, true_params)
            key_params = ['ASC_paid', 'B_FEE', 'B_DUR']
            key_comp = comp[comp['Parameter'].isin(key_params)]
            if len(key_comp) > 0 and key_comp['Bias%'].notna().any():
                mean_bias = key_comp['Bias%'].abs().mean()
                print(f"  Key params mean |Bias%|: {mean_bias:.1f}%")
        except Exception as e:
            print(f"  ERROR: {e}")

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

    return all_results, summary_df

if __name__ == '__main__':
    run_all_models()
