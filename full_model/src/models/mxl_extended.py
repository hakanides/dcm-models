"""
MXL Extended Models
===================

Extended Mixed Logit specifications with various distributions,
correlation structures, and functional forms.

Model Specifications:
    M1: Normal random fee (baseline)
    M2: Normal random fee + duration
    M3: Lognormal fee (ensures negative)
    M4: Lognormal fee + duration
    M5: Triangular distribution
    M6: Random ASC + random fee
    M7: Log-transformed fee with random coefficient
    M8: Random coefficients with demographic shifters

Usage:
    python src/models/mxl_extended.py --data data/simulated/fresh_simulation.csv --draws 500

Authors: Hakan Mülayim, Giray Girengir, Ataol Azeritürk
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
from biogeme.expressions import Beta, Variable, log, exp, MonteCarlo

# Use Draws instead of deprecated bioDraws
try:
    from biogeme.expressions import Draws
except ImportError:
    from biogeme.expressions import bioDraws as Draws


# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_data(filepath: str, fee_scale: float = 10000.0) -> pd.DataFrame:
    """Load and prepare data with transformations."""
    df = pd.read_csv(filepath)

    # Scale fees
    for alt in [1, 2, 3]:
        df[f'fee{alt}_10k'] = df[f'fee{alt}'] / fee_scale
        df[f'fee{alt}_log'] = np.log(df[f'fee{alt}'] + 1000) / 10

    # Center demographics
    df['age_c'] = (df['age_idx'] - df['age_idx'].mean()) / df['age_idx'].std()
    df['edu_c'] = (df['edu_idx'] - df['edu_idx'].mean()) / df['edu_idx'].std()
    df['inc_c'] = (df['income_indiv_idx'] - df['income_indiv_idx'].mean()) / df['income_indiv_idx'].std()

    # Numeric only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return df[numeric_cols].copy()


# =============================================================================
# MODEL SPECIFICATIONS
# =============================================================================

def get_vars():
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


def model_1_normal_fee(database: db.Database):
    """M1: Normal random fee coefficient (baseline MXL)."""
    v = get_vars()

    ASC_paid = Beta('ASC_paid', 0, None, None, 0)
    B_DUR = Beta('B_DUR', -0.05, None, None, 0)

    # Random fee: B_FEE ~ N(mu, sigma^2)
    B_FEE_MU = Beta('B_FEE_MU', -0.5, None, None, 0)
    B_FEE_SIGMA = Beta('B_FEE_SIGMA', 0.1, 0.001, 5, 0)

    B_FEE_RND = B_FEE_MU + B_FEE_SIGMA * Draws('B_FEE_RND', 'NORMAL')

    V1 = ASC_paid + B_FEE_RND * v['fee1'] + B_DUR * v['dur1']
    V2 = ASC_paid + B_FEE_RND * v['fee2'] + B_DUR * v['dur2']
    V3 = B_FEE_RND * v['fee3'] + B_DUR * v['dur3']

    prob = models.logit({1: V1, 2: V2, 3: V3}, {1: 1, 2: 1, 3: 1}, v['CHOICE'])
    logprob = log(MonteCarlo(prob))

    return logprob, 'M1: Normal Fee'


def model_2_normal_both(database: db.Database):
    """M2: Normal random fee and duration."""
    v = get_vars()

    ASC_paid = Beta('ASC_paid', 0, None, None, 0)

    B_FEE_MU = Beta('B_FEE_MU', -0.5, None, None, 0)
    B_FEE_SIGMA = Beta('B_FEE_SIGMA', 0.1, 0.001, 5, 0)

    B_DUR_MU = Beta('B_DUR_MU', -0.05, None, None, 0)
    B_DUR_SIGMA = Beta('B_DUR_SIGMA', 0.02, 0.001, 2, 0)

    B_FEE_RND = B_FEE_MU + B_FEE_SIGMA * Draws('B_FEE_RND', 'NORMAL')
    B_DUR_RND = B_DUR_MU + B_DUR_SIGMA * Draws('B_DUR_RND', 'NORMAL')

    V1 = ASC_paid + B_FEE_RND * v['fee1'] + B_DUR_RND * v['dur1']
    V2 = ASC_paid + B_FEE_RND * v['fee2'] + B_DUR_RND * v['dur2']
    V3 = B_FEE_RND * v['fee3'] + B_DUR_RND * v['dur3']

    prob = models.logit({1: V1, 2: V2, 3: V3}, {1: 1, 2: 1, 3: 1}, v['CHOICE'])
    logprob = log(MonteCarlo(prob))

    return logprob, 'M2: Normal Fee+Dur'


def model_3_lognormal_fee(database: db.Database):
    """M3: Lognormal fee - Ensures coefficient is always negative."""
    v = get_vars()

    ASC_paid = Beta('ASC_paid', 0, None, None, 0)
    B_DUR = Beta('B_DUR', -0.05, None, None, 0)

    # Lognormal: B_FEE = -exp(mu + sigma * z), z ~ N(0,1)
    # This ensures B_FEE is always negative
    B_FEE_MU_LOG = Beta('B_FEE_MU_LOG', -1.0, None, None, 0)  # Mean of underlying normal
    B_FEE_SIGMA_LOG = Beta('B_FEE_SIGMA_LOG', 0.5, 0.001, 3, 0)  # Std of underlying normal

    # Negative lognormal: always negative
    B_FEE_RND = -exp(B_FEE_MU_LOG + B_FEE_SIGMA_LOG * Draws('B_FEE_RND', 'NORMAL'))

    V1 = ASC_paid + B_FEE_RND * v['fee1'] + B_DUR * v['dur1']
    V2 = ASC_paid + B_FEE_RND * v['fee2'] + B_DUR * v['dur2']
    V3 = B_FEE_RND * v['fee3'] + B_DUR * v['dur3']

    prob = models.logit({1: V1, 2: V2, 3: V3}, {1: 1, 2: 1, 3: 1}, v['CHOICE'])
    logprob = log(MonteCarlo(prob))

    return logprob, 'M3: Lognormal Fee'


def model_4_lognormal_both(database: db.Database):
    """M4: Lognormal fee and duration - Both always negative."""
    v = get_vars()

    ASC_paid = Beta('ASC_paid', 0, None, None, 0)

    # Lognormal fee (negative)
    B_FEE_MU_LOG = Beta('B_FEE_MU_LOG', -1.0, None, None, 0)
    B_FEE_SIGMA_LOG = Beta('B_FEE_SIGMA_LOG', 0.5, 0.001, 3, 0)

    # Lognormal duration (negative)
    B_DUR_MU_LOG = Beta('B_DUR_MU_LOG', -3.0, None, None, 0)
    B_DUR_SIGMA_LOG = Beta('B_DUR_SIGMA_LOG', 0.5, 0.001, 3, 0)

    B_FEE_RND = -exp(B_FEE_MU_LOG + B_FEE_SIGMA_LOG * Draws('B_FEE_RND', 'NORMAL'))
    B_DUR_RND = -exp(B_DUR_MU_LOG + B_DUR_SIGMA_LOG * Draws('B_DUR_RND', 'NORMAL'))

    V1 = ASC_paid + B_FEE_RND * v['fee1'] + B_DUR_RND * v['dur1']
    V2 = ASC_paid + B_FEE_RND * v['fee2'] + B_DUR_RND * v['dur2']
    V3 = B_FEE_RND * v['fee3'] + B_DUR_RND * v['dur3']

    prob = models.logit({1: V1, 2: V2, 3: V3}, {1: 1, 2: 1, 3: 1}, v['CHOICE'])
    logprob = log(MonteCarlo(prob))

    return logprob, 'M4: Lognormal Fee+Dur'


def model_5_uniform_fee(database: db.Database):
    """M5: Uniform distribution for fee coefficient."""
    v = get_vars()

    ASC_paid = Beta('ASC_paid', 0, None, None, 0)
    B_DUR = Beta('B_DUR', -0.05, None, None, 0)

    # Uniform: B_FEE ~ U(lower, upper)
    # Using: B_FEE = lower + (upper - lower) * U, where U ~ Uniform(0,1)
    B_FEE_LOWER = Beta('B_FEE_LOWER', -1.0, -5, 0, 0)
    B_FEE_UPPER = Beta('B_FEE_UPPER', -0.1, -5, 0, 0)

    B_FEE_RND = B_FEE_LOWER + (B_FEE_UPPER - B_FEE_LOWER) * Draws('B_FEE_RND', 'UNIFORM')

    V1 = ASC_paid + B_FEE_RND * v['fee1'] + B_DUR * v['dur1']
    V2 = ASC_paid + B_FEE_RND * v['fee2'] + B_DUR * v['dur2']
    V3 = B_FEE_RND * v['fee3'] + B_DUR * v['dur3']

    prob = models.logit({1: V1, 2: V2, 3: V3}, {1: 1, 2: 1, 3: 1}, v['CHOICE'])
    logprob = log(MonteCarlo(prob))

    return logprob, 'M5: Uniform Fee'


def model_6_random_asc(database: db.Database):
    """M6: Random ASC + random fee - Captures unobserved preference for paid service."""
    v = get_vars()

    B_DUR = Beta('B_DUR', -0.05, None, None, 0)

    # Random ASC
    ASC_MU = Beta('ASC_MU', 0, None, None, 0)
    ASC_SIGMA = Beta('ASC_SIGMA', 1.0, 0.001, 10, 0)

    # Random fee
    B_FEE_MU = Beta('B_FEE_MU', -0.5, None, None, 0)
    B_FEE_SIGMA = Beta('B_FEE_SIGMA', 0.1, 0.001, 5, 0)

    ASC_RND = ASC_MU + ASC_SIGMA * Draws('ASC_RND', 'NORMAL')
    B_FEE_RND = B_FEE_MU + B_FEE_SIGMA * Draws('B_FEE_RND', 'NORMAL')

    V1 = ASC_RND + B_FEE_RND * v['fee1'] + B_DUR * v['dur1']
    V2 = ASC_RND + B_FEE_RND * v['fee2'] + B_DUR * v['dur2']
    V3 = B_FEE_RND * v['fee3'] + B_DUR * v['dur3']

    prob = models.logit({1: V1, 2: V2, 3: V3}, {1: 1, 2: 1, 3: 1}, v['CHOICE'])
    logprob = log(MonteCarlo(prob))

    return logprob, 'M6: Random ASC+Fee'


def model_7_log_fee_random(database: db.Database):
    """M7: Log-transformed fee with random coefficient."""
    fee1_log = Variable('fee1_log')
    fee2_log = Variable('fee2_log')
    fee3_log = Variable('fee3_log')
    dur1 = Variable('dur1')
    dur2 = Variable('dur2')
    dur3 = Variable('dur3')
    CHOICE = Variable('CHOICE')

    ASC_paid = Beta('ASC_paid', 0, None, None, 0)
    B_DUR = Beta('B_DUR', -0.05, None, None, 0)

    B_FEE_LOG_MU = Beta('B_FEE_LOG_MU', -0.5, None, None, 0)
    B_FEE_LOG_SIGMA = Beta('B_FEE_LOG_SIGMA', 0.1, 0.001, 5, 0)

    B_FEE_LOG_RND = B_FEE_LOG_MU + B_FEE_LOG_SIGMA * Draws('B_FEE_LOG_RND', 'NORMAL')

    V1 = ASC_paid + B_FEE_LOG_RND * fee1_log + B_DUR * dur1
    V2 = ASC_paid + B_FEE_LOG_RND * fee2_log + B_DUR * dur2
    V3 = B_FEE_LOG_RND * fee3_log + B_DUR * dur3

    prob = models.logit({1: V1, 2: V2, 3: V3}, {1: 1, 2: 1, 3: 1}, CHOICE)
    logprob = log(MonteCarlo(prob))

    return logprob, 'M7: Log Fee Random'


def model_8_demo_shifters(database: db.Database):
    """M8: Random coefficients with demographic shifters on mean."""
    v = get_vars()

    age_c = Variable('age_c')
    inc_c = Variable('inc_c')

    ASC_paid = Beta('ASC_paid', 0, None, None, 0)
    B_DUR = Beta('B_DUR', -0.05, None, None, 0)

    # Mean of fee coefficient depends on demographics
    B_FEE_MU_BASE = Beta('B_FEE_MU_BASE', -0.5, None, None, 0)
    B_FEE_MU_AGE = Beta('B_FEE_MU_AGE', 0, None, None, 0)  # How age shifts mean
    B_FEE_MU_INC = Beta('B_FEE_MU_INC', 0, None, None, 0)  # How income shifts mean
    B_FEE_SIGMA = Beta('B_FEE_SIGMA', 0.1, 0.001, 5, 0)

    # Individual mean depends on demographics
    B_FEE_MU_i = B_FEE_MU_BASE + B_FEE_MU_AGE * age_c + B_FEE_MU_INC * inc_c

    # Random coefficient around individual mean
    B_FEE_RND = B_FEE_MU_i + B_FEE_SIGMA * Draws('B_FEE_RND', 'NORMAL')

    V1 = ASC_paid + B_FEE_RND * v['fee1'] + B_DUR * v['dur1']
    V2 = ASC_paid + B_FEE_RND * v['fee2'] + B_DUR * v['dur2']
    V3 = B_FEE_RND * v['fee3'] + B_DUR * v['dur3']

    prob = models.logit({1: V1, 2: V2, 3: V3}, {1: 1, 2: 1, 3: 1}, v['CHOICE'])
    logprob = log(MonteCarlo(prob))

    return logprob, 'M8: Demo Shifters'


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


def estimate_model(database: db.Database, model_func, null_ll: float, n_draws: int) -> Optional[ModelResult]:
    """Estimate a single MXL model."""
    try:
        logprob, name = model_func(database)

        biogeme = bio.BIOGEME(database, logprob, number_of_draws=n_draws)
        biogeme.model_name = name.replace(' ', '_').replace(':', '').replace('+', '_')

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


def run_mxl_extended(data_path: str, output_dir: str = 'results/mxl_extended',
                     n_draws: int = 500) -> Tuple[List[ModelResult], pd.DataFrame]:
    """Run extended MXL model comparison."""
    print("="*70)
    print("EXTENDED MXL MODEL COMPARISON")
    print("="*70)

    # Prepare data
    df = prepare_data(data_path)
    n_obs = len(df)

    # Validate panel structure exists (required for MXL with repeated choices)
    if 'ID' not in df.columns:
        raise ValueError(
            "MXL requires panel data with 'ID' column for correct standard errors. "
            "Each individual should have multiple choice observations."
        )
    n_ind = df['ID'].nunique()

    print(f"\nData: {n_obs} observations from {n_ind} individuals")
    print(f"Draws: {n_draws}")

    # Create database with panel structure for cluster-robust inference
    database = db.Database('mxl_extended', df)
    database.panel('ID')  # Declare panel structure for consistent random draws
    null_ll = n_obs * np.log(1/3)

    # Model list
    model_funcs = [
        model_1_normal_fee,
        model_2_normal_both,
        model_3_lognormal_fee,
        model_4_lognormal_both,
        model_5_uniform_fee,
        model_6_random_asc,
        model_7_log_fee_random,
        model_8_demo_shifters,
    ]

    # Estimate all models
    results = []
    print("\nEstimating models (this may take several minutes)...")
    print("-"*70)

    for i, model_func in enumerate(model_funcs):
        print(f"\n[{i+1}/{len(model_funcs)}] {model_func.__name__}...")
        r = estimate_model(database, model_func, null_ll, n_draws)
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
    parser = argparse.ArgumentParser(description='Extended MXL Model Comparison')
    parser.add_argument('--data', type=str, required=True, help='Path to data CSV')
    parser.add_argument('--output', type=str, default='results/mxl_extended',
                        help='Output directory')
    parser.add_argument('--draws', type=int, default=500,
                        help='Number of simulation draws')

    args = parser.parse_args()
    run_mxl_extended(args.data, args.output, args.draws)
