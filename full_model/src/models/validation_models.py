"""
Validation Models for Simulation Studies
=========================================

WARNING: These models require TRUE latent variable values which are
ONLY available in simulated data. Do NOT use with real survey data.

Purpose: Benchmark estimation quality by comparing against true values.

Contents:
- validate_latent_estimation(): Compare estimated vs true LV correlations
- hcm_with_true_lv(): HCM using true LV values (upper bound benchmark)

Usage:
    # Only use with simulated data that has *_true columns
    from src.models.validation_models import hcm_with_true_lv, validate_latent_estimation

    # Check if true values exist before using
    if 'LV_pat_blind_true' in df.columns:
        results = validate_latent_estimation(df)

Authors: Hakan Mülayim, Giray Girengir, Ataol Azeritürk
"""

import numpy as np
import pandas as pd
from typing import Dict

# Biogeme imports
import biogeme.database as db
from biogeme import models
from biogeme.expressions import Beta, Variable


def has_true_latent_values(df: pd.DataFrame) -> bool:
    """Check if dataframe contains true latent variable columns."""
    true_cols = ['LV_pat_blind_true', 'LV_pat_constructive_true',
                 'LV_sec_dl_true', 'LV_sec_fp_true']
    return all(col in df.columns for col in true_cols)


def validate_latent_estimation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare estimated vs true latent values.

    WARNING: Requires simulated data with *_true columns.

    Args:
        df: DataFrame with both estimated (cfa/mean) and true LV columns

    Returns:
        DataFrame with correlation and RMSE for each estimated-true pair

    Raises:
        ValueError: If true LV columns are not found
    """
    if not has_true_latent_values(df):
        raise ValueError(
            "True latent value columns not found. "
            "This function is for simulation validation only."
        )

    print("\n" + "-" * 40)
    print("LATENT VARIABLE VALIDATION (Simulation)")
    print("-" * 40)

    individuals = df.groupby('ID').first().reset_index()

    # Mapping estimated -> true
    mappings = [
        ('LV_pat_blind_cfa', 'LV_pat_blind_true'),
        ('LV_pat_blind_mean', 'LV_pat_blind_true'),
        ('LV_pat_constructive_cfa', 'LV_pat_constructive_true'),
        ('LV_pat_constructive_mean', 'LV_pat_constructive_true'),
        ('LV_sec_dl_cfa', 'LV_sec_dl_true'),
        ('LV_sec_dl_mean', 'LV_sec_dl_true'),
        ('LV_sec_fp_cfa', 'LV_sec_fp_true'),
        ('LV_sec_fp_mean', 'LV_sec_fp_true'),
    ]

    results = []
    for est_col, true_col in mappings:
        if est_col in individuals.columns and true_col in individuals.columns:
            corr = individuals[est_col].corr(individuals[true_col])
            rmse = np.sqrt(((individuals[est_col] - individuals[true_col])**2).mean())
            print(f"  {est_col:30s} vs true: r={corr:.3f}, RMSE={rmse:.3f}")
            results.append({
                'Estimated': est_col,
                'True': true_col,
                'Correlation': corr,
                'RMSE': rmse
            })

    return pd.DataFrame(results)


def get_choice_vars() -> Dict:
    """Get common choice variables for HCM models."""
    return {
        'dur1': Variable('dur1'),
        'dur2': Variable('dur2'),
        'dur3': Variable('dur3'),
        'fee1_10k': Variable('fee1_10k'),
        'fee2_10k': Variable('fee2_10k'),
        'fee3_10k': Variable('fee3_10k'),
        'CHOICE': Variable('CHOICE'),
    }


def hcm_with_true_lv(database: db.Database):
    """
    HCM using TRUE latent variables.

    WARNING: SIMULATION-ONLY BENCHMARK
    This model requires true latent variable values which are only
    available in simulated data. It establishes the upper bound on
    model performance - the best possible fit with perfect LV knowledge.

    For real data applications, use:
    - hcm_with_cfa_lv() from hcm_model_improved.py
    - Models from hcm_split_latents.py

    Raises:
        ValueError: If true LV columns are not found in database
    """
    # Check for true LV columns
    required_cols = ['LV_pat_blind_true', 'LV_pat_constructive_true',
                     'LV_sec_dl_true', 'LV_sec_fp_true']
    missing = [c for c in required_cols if c not in database.dataframe.columns]
    if missing:
        raise ValueError(
            f"True LV columns not found: {missing}. "
            "This model is for simulation validation only."
        )

    v = get_choice_vars()

    # TRUE latent variables
    LV_pat_blind = Variable('LV_pat_blind_true')
    LV_pat_const = Variable('LV_pat_constructive_true')
    LV_sec_dl = Variable('LV_sec_dl_true')
    LV_sec_fp = Variable('LV_sec_fp_true')

    # Base parameters with bounds
    ASC_paid = Beta('ASC_paid', 0, -10, 10, 0)
    B_FEE = Beta('B_FEE', -0.6, -10, 0, 0)
    B_DUR = Beta('B_DUR', -0.05, -5, 0, 0)

    # LV interactions with fee (bounded)
    B_FEE_PAT_B = Beta('B_FEE_PAT_B', 0, -2, 2, 0)
    B_FEE_PAT_C = Beta('B_FEE_PAT_C', 0, -2, 2, 0)
    B_FEE_SEC_DL = Beta('B_FEE_SEC_DL', 0, -2, 2, 0)
    B_FEE_SEC_FP = Beta('B_FEE_SEC_FP', 0, -2, 2, 0)

    # LV interactions with duration (bounded)
    B_DUR_PAT_B = Beta('B_DUR_PAT_B', 0, -1, 1, 0)
    B_DUR_SEC_DL = Beta('B_DUR_SEC_DL', 0, -1, 1, 0)

    # Individual-specific coefficients
    B_FEE_i = (B_FEE +
               B_FEE_PAT_B * LV_pat_blind + B_FEE_PAT_C * LV_pat_const +
               B_FEE_SEC_DL * LV_sec_dl + B_FEE_SEC_FP * LV_sec_fp)

    B_DUR_i = B_DUR + B_DUR_PAT_B * LV_pat_blind + B_DUR_SEC_DL * LV_sec_dl

    # Utilities
    V1 = ASC_paid + B_FEE_i * v['fee1_10k'] + B_DUR_i * v['dur1']
    V2 = ASC_paid + B_FEE_i * v['fee2_10k'] + B_DUR_i * v['dur2']
    V3 = B_FEE_i * v['fee3_10k'] + B_DUR_i * v['dur3']

    V = {1: V1, 2: V2, 3: V3}
    av = {1: 1, 2: 1, 3: 1}

    logprob = models.loglogit(V, av, v['CHOICE'])
    return logprob, 'HCM-TRUE: True Latent Values (Simulation Benchmark)'
