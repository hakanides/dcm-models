"""
MNL Model Comparison Framework v2
=================================

Fixed version without sign constraints to improve convergence.

Models:
1. Model 1: Basic MNL (ASC + attributes)
2. Model 2: MNL + demographic interactions on fee
3. Model 3: MNL + demographic interactions on all attributes
4. Model 4: MNL + Likert proxies for latent variables
5. Model 5: Full MNL (demographics + Likert)
6. Model 6: MNL with alternative-specific attributes

Author: DCM Research Team
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings
# Selective warning suppression - allow important warnings through
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*overflow.*')
warnings.filterwarnings('ignore', message='.*divide by zero.*')

# Biogeme imports
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta, Variable


# =============================================================================
# DATA PREPARATION
# =============================================================================

def load_and_prepare_data(data_path: str) -> Tuple[pd.DataFrame, db.Database]:
    """Load data and prepare for Biogeme."""
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} observations from {df['ID'].nunique()} respondents")

    df = df.copy()

    # Scale fees
    df['fee1_10k'] = df['fee1'] / 10000.0
    df['fee2_10k'] = df['fee2'] / 10000.0
    df['fee3_10k'] = df['fee3'] / 10000.0

    # Center demographics
    df['age_c'] = (df['age_idx'] - 2) / 2
    df['edu_c'] = (df['edu_idx'] - 3) / 2
    df['inc_c'] = (df['income_indiv_idx'] - 3) / 2
    df['inc_hh_c'] = (df['income_house_idx'] - 3) / 2
    df['marital_c'] = (df['marital_idx'] - 0.5) / 0.5

    # Create Likert indices
    pat_blind_cols = [c for c in df.columns if c.startswith('pat_blind_') and not c.endswith('_cont')]
    if pat_blind_cols:
        df['pat_blind_idx'] = df[pat_blind_cols].mean(axis=1)
        df['pat_blind_idx_c'] = (df['pat_blind_idx'] - 3) / 2

    pat_const_cols = [c for c in df.columns if c.startswith('pat_constructive_') and not c.endswith('_cont')]
    if pat_const_cols:
        df['pat_const_idx'] = df[pat_const_cols].mean(axis=1)
        df['pat_const_idx_c'] = (df['pat_const_idx'] - 3) / 2

    sec_dl_cols = [c for c in df.columns if c.startswith('sec_dl_') and not c.endswith('_cont')]
    if sec_dl_cols:
        df['sec_dl_idx'] = df[sec_dl_cols].mean(axis=1)
        df['sec_dl_idx_c'] = (df['sec_dl_idx'] - 3) / 2

    sec_fp_cols = [c for c in df.columns if c.startswith('sec_fp_') and not c.endswith('_cont')]
    if sec_fp_cols:
        df['sec_fp_idx'] = df[sec_fp_cols].mean(axis=1)
        df['sec_fp_idx_c'] = (df['sec_fp_idx'] - 3) / 2

    # Drop string columns
    string_cols = df.select_dtypes(include=['object']).columns.tolist()
    df_numeric = df.drop(columns=string_cols)

    database = db.Database('dcm_data', df_numeric)

    return df, database


# =============================================================================
# MODEL SPECIFICATIONS (No sign constraints)
# =============================================================================

def get_variables():
    """Define variables."""
    return {
        'dur1': Variable('dur1'),
        'dur2': Variable('dur2'),
        'dur3': Variable('dur3'),
        'fee1_10k': Variable('fee1_10k'),
        'fee2_10k': Variable('fee2_10k'),
        'fee3_10k': Variable('fee3_10k'),
        'CHOICE': Variable('CHOICE'),
        'age_c': Variable('age_c'),
        'edu_c': Variable('edu_c'),
        'inc_c': Variable('inc_c'),
        'inc_hh_c': Variable('inc_hh_c'),
        'marital_c': Variable('marital_c'),
        'pat_blind_idx_c': Variable('pat_blind_idx_c'),
        'pat_const_idx_c': Variable('pat_const_idx_c'),
        'sec_dl_idx_c': Variable('sec_dl_idx_c'),
        'sec_fp_idx_c': Variable('sec_fp_idx_c'),
    }


def model_1_basic(database: db.Database):
    """Model 1: Basic MNL - No constraints"""
    v = get_variables()

    # Parameters - NO SIGN CONSTRAINTS
    ASC_paid = Beta('ASC_paid', 0, None, None, 0)
    B_FEE = Beta('B_FEE', -0.5, None, None, 0)
    B_DUR = Beta('B_DUR', -0.05, None, None, 0)

    V1 = ASC_paid + B_FEE * v['fee1_10k'] + B_DUR * v['dur1']
    V2 = ASC_paid + B_FEE * v['fee2_10k'] + B_DUR * v['dur2']
    V3 = B_FEE * v['fee3_10k'] + B_DUR * v['dur3']

    V = {1: V1, 2: V2, 3: V3}
    av = {1: 1, 2: 1, 3: 1}

    return models.loglogit(V, av, v['CHOICE']), 'M1: Basic MNL'


def model_2_demo_fee(database: db.Database):
    """Model 2: MNL + demographic interactions on fee"""
    v = get_variables()

    ASC_paid = Beta('ASC_paid', 0, None, None, 0)
    B_FEE = Beta('B_FEE', -0.5, None, None, 0)
    B_DUR = Beta('B_DUR', -0.05, None, None, 0)

    # Fee interactions
    B_FEE_AGE = Beta('B_FEE_AGE', 0, None, None, 0)
    B_FEE_EDU = Beta('B_FEE_EDU', 0, None, None, 0)
    B_FEE_INC = Beta('B_FEE_INC', 0, None, None, 0)

    B_FEE_i = B_FEE + B_FEE_AGE * v['age_c'] + B_FEE_EDU * v['edu_c'] + B_FEE_INC * v['inc_c']

    V1 = ASC_paid + B_FEE_i * v['fee1_10k'] + B_DUR * v['dur1']
    V2 = ASC_paid + B_FEE_i * v['fee2_10k'] + B_DUR * v['dur2']
    V3 = B_FEE_i * v['fee3_10k'] + B_DUR * v['dur3']

    V = {1: V1, 2: V2, 3: V3}
    av = {1: 1, 2: 1, 3: 1}

    return models.loglogit(V, av, v['CHOICE']), 'M2: Demo-Fee'


def model_3_demo_all(database: db.Database):
    """Model 3: MNL + demographic interactions on all attributes"""
    v = get_variables()

    ASC_paid = Beta('ASC_paid', 0, None, None, 0)
    B_FEE = Beta('B_FEE', -0.5, None, None, 0)
    B_DUR = Beta('B_DUR', -0.05, None, None, 0)

    # Fee interactions
    B_FEE_AGE = Beta('B_FEE_AGE', 0, None, None, 0)
    B_FEE_EDU = Beta('B_FEE_EDU', 0, None, None, 0)
    B_FEE_INC = Beta('B_FEE_INC', 0, None, None, 0)

    # Duration interactions
    B_DUR_EDU = Beta('B_DUR_EDU', 0, None, None, 0)
    B_DUR_INC = Beta('B_DUR_INC', 0, None, None, 0)
    B_DUR_MARITAL = Beta('B_DUR_MARITAL', 0, None, None, 0)

    B_FEE_i = B_FEE + B_FEE_AGE * v['age_c'] + B_FEE_EDU * v['edu_c'] + B_FEE_INC * v['inc_c']
    B_DUR_i = B_DUR + B_DUR_EDU * v['edu_c'] + B_DUR_INC * v['inc_c'] + B_DUR_MARITAL * v['marital_c']

    V1 = ASC_paid + B_FEE_i * v['fee1_10k'] + B_DUR_i * v['dur1']
    V2 = ASC_paid + B_FEE_i * v['fee2_10k'] + B_DUR_i * v['dur2']
    V3 = B_FEE_i * v['fee3_10k'] + B_DUR_i * v['dur3']

    V = {1: V1, 2: V2, 3: V3}
    av = {1: 1, 2: 1, 3: 1}

    return models.loglogit(V, av, v['CHOICE']), 'M3: Demo-All'


def model_4_likert(database: db.Database):
    """Model 4: MNL + Likert proxies"""
    v = get_variables()

    ASC_paid = Beta('ASC_paid', 0, None, None, 0)
    B_FEE = Beta('B_FEE', -0.5, None, None, 0)
    B_DUR = Beta('B_DUR', -0.05, None, None, 0)

    # Likert interactions on fee
    B_FEE_PAT_B = Beta('B_FEE_PAT_B', 0, None, None, 0)
    B_FEE_PAT_C = Beta('B_FEE_PAT_C', 0, None, None, 0)
    B_FEE_SEC_DL = Beta('B_FEE_SEC_DL', 0, None, None, 0)
    B_FEE_SEC_FP = Beta('B_FEE_SEC_FP', 0, None, None, 0)

    # Likert interactions on duration
    B_DUR_PAT_B = Beta('B_DUR_PAT_B', 0, None, None, 0)
    B_DUR_SEC_DL = Beta('B_DUR_SEC_DL', 0, None, None, 0)

    B_FEE_i = (B_FEE +
               B_FEE_PAT_B * v['pat_blind_idx_c'] +
               B_FEE_PAT_C * v['pat_const_idx_c'] +
               B_FEE_SEC_DL * v['sec_dl_idx_c'] +
               B_FEE_SEC_FP * v['sec_fp_idx_c'])

    B_DUR_i = (B_DUR +
               B_DUR_PAT_B * v['pat_blind_idx_c'] +
               B_DUR_SEC_DL * v['sec_dl_idx_c'])

    V1 = ASC_paid + B_FEE_i * v['fee1_10k'] + B_DUR_i * v['dur1']
    V2 = ASC_paid + B_FEE_i * v['fee2_10k'] + B_DUR_i * v['dur2']
    V3 = B_FEE_i * v['fee3_10k'] + B_DUR_i * v['dur3']

    V = {1: V1, 2: V2, 3: V3}
    av = {1: 1, 2: 1, 3: 1}

    return models.loglogit(V, av, v['CHOICE']), 'M4: Likert'


def model_5_full(database: db.Database):
    """Model 5: Full MNL (demographics + Likert)"""
    v = get_variables()

    ASC_paid = Beta('ASC_paid', 0, None, None, 0)
    B_FEE = Beta('B_FEE', -0.5, None, None, 0)
    B_DUR = Beta('B_DUR', -0.05, None, None, 0)

    # Demographics on fee
    B_FEE_AGE = Beta('B_FEE_AGE', 0, None, None, 0)
    B_FEE_EDU = Beta('B_FEE_EDU', 0, None, None, 0)
    B_FEE_INC = Beta('B_FEE_INC', 0, None, None, 0)

    # Likert on fee
    B_FEE_PAT_B = Beta('B_FEE_PAT_B', 0, None, None, 0)
    B_FEE_SEC_DL = Beta('B_FEE_SEC_DL', 0, None, None, 0)

    # Demographics on duration
    B_DUR_EDU = Beta('B_DUR_EDU', 0, None, None, 0)
    B_DUR_INC = Beta('B_DUR_INC', 0, None, None, 0)

    # Likert on duration
    B_DUR_PAT_B = Beta('B_DUR_PAT_B', 0, None, None, 0)
    B_DUR_SEC_DL = Beta('B_DUR_SEC_DL', 0, None, None, 0)

    B_FEE_i = (B_FEE +
               B_FEE_AGE * v['age_c'] + B_FEE_EDU * v['edu_c'] + B_FEE_INC * v['inc_c'] +
               B_FEE_PAT_B * v['pat_blind_idx_c'] + B_FEE_SEC_DL * v['sec_dl_idx_c'])

    B_DUR_i = (B_DUR +
               B_DUR_EDU * v['edu_c'] + B_DUR_INC * v['inc_c'] +
               B_DUR_PAT_B * v['pat_blind_idx_c'] + B_DUR_SEC_DL * v['sec_dl_idx_c'])

    V1 = ASC_paid + B_FEE_i * v['fee1_10k'] + B_DUR_i * v['dur1']
    V2 = ASC_paid + B_FEE_i * v['fee2_10k'] + B_DUR_i * v['dur2']
    V3 = B_FEE_i * v['fee3_10k'] + B_DUR_i * v['dur3']

    V = {1: V1, 2: V2, 3: V3}
    av = {1: 1, 2: 1, 3: 1}

    return models.loglogit(V, av, v['CHOICE']), 'M5: Full'


def model_6_alt_specific(database: db.Database):
    """Model 6: Alternative-specific constants and fee coefficients"""
    v = get_variables()

    # Alternative-specific constants
    ASC_paid1 = Beta('ASC_paid1', 0, None, None, 0)
    ASC_paid2 = Beta('ASC_paid2', 0, None, None, 0)
    # ASC_std = 0 (base)

    # Alternative-specific fee coefficients
    B_FEE_PAID = Beta('B_FEE_PAID', -0.5, None, None, 0)

    # Common coefficients
    B_DUR = Beta('B_DUR', -0.05, None, None, 0)

    V1 = ASC_paid1 + B_FEE_PAID * v['fee1_10k'] + B_DUR * v['dur1']
    V2 = ASC_paid2 + B_FEE_PAID * v['fee2_10k'] + B_DUR * v['dur2']
    V3 = B_DUR * v['dur3']  # Standard: no fee

    V = {1: V1, 2: V2, 3: V3}
    av = {1: 1, 2: 1, 3: 1}

    return models.loglogit(V, av, v['CHOICE']), 'M6: Alt-Specific'


# =============================================================================
# MODEL ESTIMATION
# =============================================================================

@dataclass
class ModelResults:
    name: str
    log_likelihood: float
    null_log_likelihood: float
    n_parameters: int
    n_observations: int
    aic: float
    bic: float
    rho_squared: float
    adj_rho_squared: float
    parameters: Dict[str, float]
    std_errors: Dict[str, float]
    t_stats: Dict[str, float]
    converged: bool


def estimate_model(database: db.Database, model_func, null_ll: float) -> ModelResults:
    """Estimate a model."""
    logprob, model_name = model_func(database)

    print(f"\n{model_name}")
    print("-" * 40)

    biogeme_model = bio.BIOGEME(database, logprob)
    results = biogeme_model.estimate()

    ll = results.final_loglikelihood
    n_params = results.number_of_free_parameters
    n_obs = len(database.dataframe)

    aic = results.akaike_information_criterion
    bic = results.bayesian_information_criterion
    rho_sq = 1 - (ll / null_ll)
    adj_rho_sq = 1 - ((ll - n_params) / null_ll)

    betas = results.get_beta_values()
    std_errs = {}
    t_stats = {}

    for param in betas.keys():
        try:
            std_errs[param] = results.get_parameter_std_err(param)
            t_stats[param] = results.get_parameter_t_test(param)
        except:
            std_errs[param] = np.nan
            t_stats[param] = np.nan

    converged = results.algorithm_has_converged

    print(f"  LL: {ll:.2f} | K: {n_params} | AIC: {aic:.2f} | BIC: {bic:.2f}")
    print(f"  ρ²: {rho_sq:.4f} | Converged: {converged}")

    return ModelResults(
        name=model_name,
        log_likelihood=ll,
        null_log_likelihood=null_ll,
        n_parameters=n_params,
        n_observations=n_obs,
        aic=aic,
        bic=bic,
        rho_squared=rho_sq,
        adj_rho_squared=adj_rho_sq,
        parameters=betas,
        std_errors=std_errs,
        t_stats=t_stats,
        converged=converged
    )


def likelihood_ratio_test(ll_r: float, ll_u: float, df: int) -> Tuple[float, float]:
    """LR test."""
    if ll_u < ll_r:
        return np.nan, np.nan
    lr = -2 * (ll_r - ll_u)
    p = 1 - stats.chi2.cdf(lr, df)
    return lr, p


# =============================================================================
# COMPARISON AND OUTPUT
# =============================================================================

def create_comparison_table(results: List[ModelResults]) -> pd.DataFrame:
    """Create comparison table."""
    data = []
    for r in results:
        data.append({
            'Model': r.name,
            'LL': r.log_likelihood,
            'K': r.n_parameters,
            'AIC': r.aic,
            'BIC': r.bic,
            'ρ²': r.rho_squared,
            'Adj.ρ²': r.adj_rho_squared,
            'Conv': 'Yes' if r.converged else 'No'
        })
    df = pd.DataFrame(data)
    df['AIC_Rank'] = df['AIC'].rank().astype(int)
    df['BIC_Rank'] = df['BIC'].rank().astype(int)
    return df


def create_parameter_table(results: List[ModelResults]) -> pd.DataFrame:
    """Create parameter comparison table."""
    all_params = set()
    for r in results:
        all_params.update(r.parameters.keys())

    rows = []
    for param in sorted(all_params):
        row = {'Parameter': param}
        for r in results:
            if param in r.parameters:
                est = r.parameters[param]
                t = r.t_stats.get(param, np.nan)
                if not np.isnan(t):
                    row[r.name] = f"{est:.4f} ({t:.2f})"
                else:
                    row[r.name] = f"{est:.4f}"
            else:
                row[r.name] = "-"
        rows.append(row)

    return pd.DataFrame(rows)


def plot_comparison(comp_df: pd.DataFrame, output_path: str = None):
    """Plot model comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    models = comp_df['Model'].values
    x = np.arange(len(models))

    # LL
    ax = axes[0, 0]
    ax.bar(x, comp_df['LL'], color='steelblue', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Log-Likelihood')
    ax.set_title('Log-Likelihood (higher = better)')

    # AIC
    ax = axes[0, 1]
    colors = ['green' if r == 1 else 'steelblue' for r in comp_df['AIC_Rank']]
    ax.bar(x, comp_df['AIC'], color=colors, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('AIC')
    ax.set_title('AIC (lower = better)')

    # BIC
    ax = axes[1, 0]
    colors = ['green' if r == 1 else 'steelblue' for r in comp_df['BIC_Rank']]
    ax.bar(x, comp_df['BIC'], color=colors, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('BIC')
    ax.set_title('BIC (lower = better)')

    # Rho-squared
    ax = axes[1, 1]
    width = 0.35
    ax.bar(x - width/2, comp_df['ρ²'], width, label='ρ²', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, comp_df['Adj.ρ²'], width, label='Adj.ρ²', color='coral', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Rho-squared')
    ax.set_title('Goodness of Fit')
    ax.legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {output_path}")

    plt.show()


def run_comparison(data_path: str, output_dir: str = None):
    """Run full model comparison."""
    print("=" * 60)
    print("MNL MODEL COMPARISON v2 (No Sign Constraints)")
    print("=" * 60)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

    # Load data
    df, database = load_and_prepare_data(data_path)

    # Null LL
    n_obs = len(database.dataframe)
    null_ll = n_obs * np.log(1.0 / 3)
    print(f"\nNull LL (equal probs): {null_ll:.2f}")

    # Models
    model_funcs = [
        model_1_basic,
        model_2_demo_fee,
        model_3_demo_all,
        model_4_likert,
        model_5_full,
        model_6_alt_specific,
    ]

    results = []
    for func in model_funcs:
        try:
            res = estimate_model(database, func, null_ll)
            results.append(res)
        except Exception as e:
            print(f"  ERROR: {e}")

    # Comparison table
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)

    comp_df = create_comparison_table(results)
    print("\n" + comp_df.to_string(index=False))

    # Best models
    best_aic = comp_df.loc[comp_df['AIC'].idxmin()]
    best_bic = comp_df.loc[comp_df['BIC'].idxmin()]

    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    print(f"Best by AIC: {best_aic['Model']} (AIC={best_aic['AIC']:.2f})")
    print(f"Best by BIC: {best_bic['Model']} (BIC={best_bic['BIC']:.2f})")

    # LR Tests
    print("\n" + "=" * 60)
    print("LIKELIHOOD RATIO TESTS (vs Basic Model)")
    print("=" * 60)

    base_ll = results[0].log_likelihood
    base_k = results[0].n_parameters

    for r in results[1:]:
        df_test = r.n_parameters - base_k
        if df_test > 0 and r.log_likelihood > base_ll:
            lr, p = likelihood_ratio_test(base_ll, r.log_likelihood, df_test)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"{r.name}: LR={lr:.2f}, df={df_test}, p={p:.4f} {sig}")
        else:
            print(f"{r.name}: Not applicable (worse fit or fewer params)")

    # Save outputs
    if output_dir:
        comp_df.to_csv(output_dir / 'model_comparison.csv', index=False)

        param_df = create_parameter_table(results)
        param_df.to_csv(output_dir / 'parameters.csv', index=False)

        plot_comparison(comp_df, str(output_dir / 'comparison_plot.png'))

        print(f"\nOutputs saved to: {output_dir}")

    return results, comp_df


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', default='mnl_v2_results')
    args = parser.parse_args()

    run_comparison(args.data, args.output)


if __name__ == '__main__':
    main()
