"""
MNL Extended Models
===================

Extended Multinomial Logit specifications with various functional forms
and interaction patterns.

Model Specifications:
    M1: Basic MNL (baseline)
    M2: Log-transformed fee
    M3: Quadratic fee effect (diminishing sensitivity)
    M4: Piecewise linear fee (threshold effect)
    M5: Full demographic interactions
    M6: Cross-demographic interactions (age×income, etc.)
    M7: Log fee with demographics
    M8: Box-Cox transformed fee (estimated lambda)

Usage:
    python src/models/mnl_extended.py --data data/simulated/fresh_simulation.csv

Author: DCM Research Team
"""

import argparse
import json
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta, Variable, log, exp, bioMin, bioMax


# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_data(filepath: str, fee_scale: float = 10000.0) -> pd.DataFrame:
    """
    Load and prepare data with various transformations.
    """
    df = pd.read_csv(filepath)

    # Scale fees (in 10,000 TL units)
    for alt in [1, 2, 3]:
        df[f'fee{alt}_10k'] = df[f'fee{alt}'] / fee_scale

        # Log transformation (add small constant to handle zeros)
        df[f'fee{alt}_log'] = np.log(df[f'fee{alt}'] + 1000) / 10  # Scaled log

        # Squared term for quadratic effects
        df[f'fee{alt}_sq'] = (df[f'fee{alt}_10k']) ** 2

        # Piecewise: indicator for high fee (above median)
        median_fee = df[f'fee{alt}'].median()
        df[f'fee{alt}_high'] = (df[f'fee{alt}'] > median_fee).astype(float)

    # Center demographic variables
    df['age_c'] = (df['age_idx'] - df['age_idx'].mean()) / df['age_idx'].std()
    df['edu_c'] = (df['edu_idx'] - df['edu_idx'].mean()) / df['edu_idx'].std()
    df['inc_c'] = (df['income_indiv_idx'] - df['income_indiv_idx'].mean()) / df['income_indiv_idx'].std()

    # Cross-demographic interactions
    df['age_x_inc'] = df['age_c'] * df['inc_c']
    df['age_x_edu'] = df['age_c'] * df['edu_c']
    df['edu_x_inc'] = df['edu_c'] * df['inc_c']

    # Keep only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    return df[numeric_cols].copy()


# =============================================================================
# MODEL SPECIFICATIONS
# =============================================================================

def get_base_vars():
    """Get common variables."""
    return {
        'CHOICE': Variable('CHOICE'),
        'fee1': Variable('fee1_10k'),
        'fee2': Variable('fee2_10k'),
        'fee3': Variable('fee3_10k'),
        'dur1': Variable('dur1'),
        'dur2': Variable('dur2'),
        'dur3': Variable('dur3'),
    }


def model_1_basic(database: db.Database):
    """M1: Basic MNL - Linear fee and duration."""
    v = get_base_vars()

    ASC_paid = Beta('ASC_paid', 0, None, None, 0)
    B_FEE = Beta('B_FEE', -0.5, None, None, 0)
    B_DUR = Beta('B_DUR', -0.05, None, None, 0)

    V1 = ASC_paid + B_FEE * v['fee1'] + B_DUR * v['dur1']
    V2 = ASC_paid + B_FEE * v['fee2'] + B_DUR * v['dur2']
    V3 = B_FEE * v['fee3'] + B_DUR * v['dur3']

    return models.loglogit({1: V1, 2: V2, 3: V3}, {1: 1, 2: 1, 3: 1}, v['CHOICE']), 'M1: Basic MNL'


def model_2_log_fee(database: db.Database):
    """M2: Log-transformed fee - Diminishing marginal disutility."""
    v = get_base_vars()

    # Log-transformed fee variables
    fee1_log = Variable('fee1_log')
    fee2_log = Variable('fee2_log')
    fee3_log = Variable('fee3_log')

    ASC_paid = Beta('ASC_paid', 0, None, None, 0)
    B_FEE_LOG = Beta('B_FEE_LOG', -0.5, None, None, 0)
    B_DUR = Beta('B_DUR', -0.05, None, None, 0)

    V1 = ASC_paid + B_FEE_LOG * fee1_log + B_DUR * v['dur1']
    V2 = ASC_paid + B_FEE_LOG * fee2_log + B_DUR * v['dur2']
    V3 = B_FEE_LOG * fee3_log + B_DUR * v['dur3']

    return models.loglogit({1: V1, 2: V2, 3: V3}, {1: 1, 2: 1, 3: 1}, v['CHOICE']), 'M2: Log Fee'


def model_3_quadratic_fee(database: db.Database):
    """M3: Quadratic fee - Non-linear sensitivity (U-shaped or inverted-U)."""
    v = get_base_vars()

    fee1_sq = Variable('fee1_sq')
    fee2_sq = Variable('fee2_sq')
    fee3_sq = Variable('fee3_sq')

    ASC_paid = Beta('ASC_paid', 0, None, None, 0)
    B_FEE = Beta('B_FEE', -0.5, None, None, 0)
    B_FEE_SQ = Beta('B_FEE_SQ', 0, None, None, 0)  # Quadratic term
    B_DUR = Beta('B_DUR', -0.05, None, None, 0)

    V1 = ASC_paid + B_FEE * v['fee1'] + B_FEE_SQ * fee1_sq + B_DUR * v['dur1']
    V2 = ASC_paid + B_FEE * v['fee2'] + B_FEE_SQ * fee2_sq + B_DUR * v['dur2']
    V3 = B_FEE * v['fee3'] + B_FEE_SQ * fee3_sq + B_DUR * v['dur3']

    return models.loglogit({1: V1, 2: V2, 3: V3}, {1: 1, 2: 1, 3: 1}, v['CHOICE']), 'M3: Quadratic Fee'


def model_4_piecewise_fee(database: db.Database):
    """M4: Piecewise linear fee - Threshold effect at median."""
    v = get_base_vars()

    fee1_high = Variable('fee1_high')
    fee2_high = Variable('fee2_high')
    fee3_high = Variable('fee3_high')

    ASC_paid = Beta('ASC_paid', 0, None, None, 0)
    B_FEE = Beta('B_FEE', -0.5, None, None, 0)
    B_FEE_HIGH = Beta('B_FEE_HIGH', 0, None, None, 0)  # Additional effect for high fee
    B_DUR = Beta('B_DUR', -0.05, None, None, 0)

    # Piecewise: base effect + additional slope for high values
    V1 = ASC_paid + B_FEE * v['fee1'] + B_FEE_HIGH * fee1_high * v['fee1'] + B_DUR * v['dur1']
    V2 = ASC_paid + B_FEE * v['fee2'] + B_FEE_HIGH * fee2_high * v['fee2'] + B_DUR * v['dur2']
    V3 = B_FEE * v['fee3'] + B_FEE_HIGH * fee3_high * v['fee3'] + B_DUR * v['dur3']

    return models.loglogit({1: V1, 2: V2, 3: V3}, {1: 1, 2: 1, 3: 1}, v['CHOICE']), 'M4: Piecewise Fee'


def model_5_full_demo(database: db.Database):
    """M5: Full demographic interactions on fee and duration."""
    v = get_base_vars()

    age_c = Variable('age_c')
    edu_c = Variable('edu_c')
    inc_c = Variable('inc_c')

    ASC_paid = Beta('ASC_paid', 0, None, None, 0)

    # Base effects
    B_FEE = Beta('B_FEE', -0.5, None, None, 0)
    B_DUR = Beta('B_DUR', -0.05, None, None, 0)

    # Fee interactions
    B_FEE_AGE = Beta('B_FEE_AGE', 0, None, None, 0)
    B_FEE_EDU = Beta('B_FEE_EDU', 0, None, None, 0)
    B_FEE_INC = Beta('B_FEE_INC', 0, None, None, 0)

    # Duration interactions
    B_DUR_AGE = Beta('B_DUR_AGE', 0, None, None, 0)
    B_DUR_EDU = Beta('B_DUR_EDU', 0, None, None, 0)
    B_DUR_INC = Beta('B_DUR_INC', 0, None, None, 0)

    # Individual-specific coefficients
    B_FEE_i = B_FEE + B_FEE_AGE * age_c + B_FEE_EDU * edu_c + B_FEE_INC * inc_c
    B_DUR_i = B_DUR + B_DUR_AGE * age_c + B_DUR_EDU * edu_c + B_DUR_INC * inc_c

    V1 = ASC_paid + B_FEE_i * v['fee1'] + B_DUR_i * v['dur1']
    V2 = ASC_paid + B_FEE_i * v['fee2'] + B_DUR_i * v['dur2']
    V3 = B_FEE_i * v['fee3'] + B_DUR_i * v['dur3']

    return models.loglogit({1: V1, 2: V2, 3: V3}, {1: 1, 2: 1, 3: 1}, v['CHOICE']), 'M5: Full Demographics'


def model_6_cross_demo(database: db.Database):
    """M6: Cross-demographic interactions (age×income, etc.)."""
    v = get_base_vars()

    age_c = Variable('age_c')
    edu_c = Variable('edu_c')
    inc_c = Variable('inc_c')
    age_x_inc = Variable('age_x_inc')
    age_x_edu = Variable('age_x_edu')
    edu_x_inc = Variable('edu_x_inc')

    ASC_paid = Beta('ASC_paid', 0, None, None, 0)
    B_FEE = Beta('B_FEE', -0.5, None, None, 0)
    B_DUR = Beta('B_DUR', -0.05, None, None, 0)

    # Main demographic effects on fee
    B_FEE_AGE = Beta('B_FEE_AGE', 0, None, None, 0)
    B_FEE_INC = Beta('B_FEE_INC', 0, None, None, 0)

    # Cross interactions on fee
    B_FEE_AGExINC = Beta('B_FEE_AGExINC', 0, None, None, 0)
    B_FEE_AGExEDU = Beta('B_FEE_AGExEDU', 0, None, None, 0)
    B_FEE_EDUxINC = Beta('B_FEE_EDUxINC', 0, None, None, 0)

    B_FEE_i = (B_FEE + B_FEE_AGE * age_c + B_FEE_INC * inc_c +
               B_FEE_AGExINC * age_x_inc + B_FEE_AGExEDU * age_x_edu +
               B_FEE_EDUxINC * edu_x_inc)

    V1 = ASC_paid + B_FEE_i * v['fee1'] + B_DUR * v['dur1']
    V2 = ASC_paid + B_FEE_i * v['fee2'] + B_DUR * v['dur2']
    V3 = B_FEE_i * v['fee3'] + B_DUR * v['dur3']

    return models.loglogit({1: V1, 2: V2, 3: V3}, {1: 1, 2: 1, 3: 1}, v['CHOICE']), 'M6: Cross Demographics'


def model_7_log_fee_demo(database: db.Database):
    """M7: Log fee with demographic interactions."""
    fee1_log = Variable('fee1_log')
    fee2_log = Variable('fee2_log')
    fee3_log = Variable('fee3_log')
    dur1 = Variable('dur1')
    dur2 = Variable('dur2')
    dur3 = Variable('dur3')
    CHOICE = Variable('CHOICE')

    age_c = Variable('age_c')
    edu_c = Variable('edu_c')
    inc_c = Variable('inc_c')

    ASC_paid = Beta('ASC_paid', 0, None, None, 0)
    B_FEE_LOG = Beta('B_FEE_LOG', -0.5, None, None, 0)
    B_DUR = Beta('B_DUR', -0.05, None, None, 0)

    B_FEE_AGE = Beta('B_FEE_AGE', 0, None, None, 0)
    B_FEE_INC = Beta('B_FEE_INC', 0, None, None, 0)

    B_FEE_i = B_FEE_LOG + B_FEE_AGE * age_c + B_FEE_INC * inc_c

    V1 = ASC_paid + B_FEE_i * fee1_log + B_DUR * dur1
    V2 = ASC_paid + B_FEE_i * fee2_log + B_DUR * dur2
    V3 = B_FEE_i * fee3_log + B_DUR * dur3

    return models.loglogit({1: V1, 2: V2, 3: V3}, {1: 1, 2: 1, 3: 1}, CHOICE), 'M7: Log Fee + Demo'


def model_8_asc_demo(database: db.Database):
    """M8: Demographics affecting ASC (preference for paid service)."""
    v = get_base_vars()

    age_c = Variable('age_c')
    edu_c = Variable('edu_c')
    inc_c = Variable('inc_c')

    ASC_paid = Beta('ASC_paid', 0, None, None, 0)
    B_FEE = Beta('B_FEE', -0.5, None, None, 0)
    B_DUR = Beta('B_DUR', -0.05, None, None, 0)

    # Demographics affect preference for paid service
    B_ASC_AGE = Beta('B_ASC_AGE', 0, None, None, 0)
    B_ASC_EDU = Beta('B_ASC_EDU', 0, None, None, 0)
    B_ASC_INC = Beta('B_ASC_INC', 0, None, None, 0)

    ASC_i = ASC_paid + B_ASC_AGE * age_c + B_ASC_EDU * edu_c + B_ASC_INC * inc_c

    V1 = ASC_i + B_FEE * v['fee1'] + B_DUR * v['dur1']
    V2 = ASC_i + B_FEE * v['fee2'] + B_DUR * v['dur2']
    V3 = B_FEE * v['fee3'] + B_DUR * v['dur3']

    return models.loglogit({1: V1, 2: V2, 3: V3}, {1: 1, 2: 1, 3: 1}, v['CHOICE']), 'M8: ASC Demographics'


# =============================================================================
# ESTIMATION
# =============================================================================

@dataclass
class ModelResult:
    name: str
    ll: float
    k: int
    aic: float
    bic: float
    rho2: float
    params: Dict[str, float]
    t_stats: Dict[str, float]
    converged: bool


def estimate_model(database: db.Database, model_func, null_ll: float) -> Optional[ModelResult]:
    """Estimate a single model."""
    try:
        logprob, name = model_func(database)

        biogeme = bio.BIOGEME(database, logprob)
        biogeme.model_name = name.replace(' ', '_').replace(':', '')

        results = biogeme.estimate()

        ll = results.final_loglikelihood
        k = results.number_of_free_parameters
        n = len(database.dataframe)

        aic = 2 * k - 2 * ll
        bic = k * np.log(n) - 2 * ll
        rho2 = 1 - (ll / null_ll)

        betas = results.get_beta_values()
        t_stats = {}
        for p in betas:
            try:
                se = results.get_parameter_std_err(p)
                t_stats[p] = betas[p] / se if se > 0 else np.nan
            except:
                t_stats[p] = np.nan

        return ModelResult(
            name=name, ll=ll, k=k, aic=aic, bic=bic, rho2=rho2,
            params=betas, t_stats=t_stats, converged=results.algorithm_has_converged
        )
    except Exception as e:
        print(f"  ERROR in {model_func.__name__}: {e}")
        return None


def run_mnl_extended(data_path: str, output_dir: str = 'results/mnl_extended') -> Tuple[List[ModelResult], pd.DataFrame]:
    """
    Run extended MNL model comparison.
    """
    print("="*70)
    print("EXTENDED MNL MODEL COMPARISON")
    print("="*70)

    # Prepare data
    df = prepare_data(data_path)
    n_obs = len(df)
    n_ind = df['ID'].nunique() if 'ID' in df.columns else n_obs

    print(f"\nData: {n_obs} observations from {n_ind} individuals")

    # Create database
    database = db.Database('mnl_extended', df)
    null_ll = n_obs * np.log(1/3)

    # Model list
    model_funcs = [
        model_1_basic,
        model_2_log_fee,
        model_3_quadratic_fee,
        model_4_piecewise_fee,
        model_5_full_demo,
        model_6_cross_demo,
        model_7_log_fee_demo,
        model_8_asc_demo,
    ]

    # Estimate all models
    results = []
    print("\nEstimating models...")
    print("-"*70)

    for model_func in model_funcs:
        print(f"\n{model_func.__name__}...")
        r = estimate_model(database, model_func, null_ll)
        if r:
            results.append(r)
            print(f"  LL: {r.ll:.2f} | K: {r.k} | AIC: {r.aic:.2f} | Conv: {r.converged}")

    # Create comparison table
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)

    comp_data = []
    for r in results:
        comp_data.append({
            'Model': r.name,
            'LL': r.ll,
            'K': r.k,
            'AIC': r.aic,
            'BIC': r.bic,
            'rho2': r.rho2,
            'Conv': 'Yes' if r.converged else 'No'
        })

    comp_df = pd.DataFrame(comp_data).sort_values('AIC')
    comp_df['Rank'] = range(1, len(comp_df) + 1)
    print("\n" + comp_df.to_string(index=False))

    # Best model
    best = comp_df.iloc[0]
    print(f"\n*** Best by AIC: {best['Model']} (AIC={best['AIC']:.2f}) ***")

    # Print parameter estimates for top models
    print("\n" + "="*70)
    print("PARAMETER ESTIMATES (Top 3 Models)")
    print("="*70)

    for r in sorted(results, key=lambda x: x.aic)[:3]:
        print(f"\n{r.name}:")
        print(f"{'Parameter':<20} {'Estimate':>12} {'t-stat':>10}")
        print("-"*44)
        for p, v in r.params.items():
            t = r.t_stats.get(p, np.nan)
            sig = '***' if abs(t) > 2.576 else '**' if abs(t) > 1.96 else '*' if abs(t) > 1.645 else ''
            print(f"{p:<20} {v:>12.4f} {t:>10.2f} {sig}")

    # Save results
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        comp_df.to_csv(output_path / 'model_comparison.csv', index=False)

        # Parameter estimates
        param_rows = []
        for r in results:
            for p, v in r.params.items():
                param_rows.append({
                    'Model': r.name,
                    'Parameter': p,
                    'Estimate': v,
                    't_stat': r.t_stats.get(p, np.nan)
                })
        pd.DataFrame(param_rows).to_csv(output_path / 'parameters.csv', index=False)

        print(f"\nResults saved to: {output_path}")

    return results, comp_df


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extended MNL Model Comparison')
    parser.add_argument('--data', type=str, required=True, help='Path to data CSV')
    parser.add_argument('--output', type=str, default='results/mnl_extended',
                        help='Output directory')

    args = parser.parse_args()
    run_mnl_extended(args.data, args.output)
