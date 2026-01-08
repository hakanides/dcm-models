"""
HCM Extended Models
===================

Extended Hybrid Choice Model specifications with various latent variable
configurations and interaction patterns.

Model Specifications:
    M1: Single LV on fee (Blind Patriotism)
    M2: Single LV on fee (Daily Secularism)
    M3: Single LV on duration
    M4: LV on ASC (preference for paid service)
    M5: Quadratic LV effect
    M6: LV × Demographic interactions
    M7: Multiple LVs with domain separation
    M8: Full model with all interactions

Usage:
    python src/models/hcm_extended.py --data data/simulated/fresh_simulation.csv

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
from biogeme.expressions import Beta, Variable


# =============================================================================
# LATENT VARIABLE ESTIMATION
# =============================================================================

def estimate_latent_cfa(df: pd.DataFrame, items: list, name: str) -> pd.Series:
    """Estimate LV using weighted average (CFA-style)."""
    X = df[items].values

    # Item-total correlation weights
    total = X.sum(axis=1)
    weights = []
    for i in range(X.shape[1]):
        corr = np.corrcoef(X[:, i], total)[0, 1]
        weights.append(max(0.1, corr))
    weights = np.array(weights) / sum(weights)

    # Weighted standardized score
    score = (X * weights).sum(axis=1)
    score = (score - score.mean()) / score.std()

    return pd.Series(score, index=df.index, name=f'LV_{name}')


def compute_cronbach_alpha(X: np.ndarray) -> float:
    """Compute Cronbach's alpha."""
    n_items = X.shape[1]
    item_var = X.var(axis=0, ddof=1).sum()
    total_var = X.sum(axis=1).var(ddof=1)
    return (n_items / (n_items - 1)) * (1 - item_var / total_var)


# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_data(filepath: str, fee_scale: float = 10000.0) -> Tuple[pd.DataFrame, Dict]:
    """Load and prepare data with LV estimation."""
    df = pd.read_csv(filepath)

    # Scale fees
    for alt in [1, 2, 3]:
        df[f'fee{alt}_10k'] = df[f'fee{alt}'] / fee_scale

    # Center demographics
    df['age_c'] = (df['age_idx'] - df['age_idx'].mean()) / df['age_idx'].std()
    df['edu_c'] = (df['edu_idx'] - df['edu_idx'].mean()) / df['edu_idx'].std()
    df['inc_c'] = (df['income_indiv_idx'] - df['income_indiv_idx'].mean()) / df['income_indiv_idx'].std()

    # Define constructs
    constructs = {
        'pat_blind': 'pat_blind_',
        'pat_const': 'pat_constructive_',
        'sec_dl': 'sec_dl_',
        'sec_fp': 'sec_fp_',
    }

    # Get unique individuals
    individuals = df.groupby('ID').first().reset_index()

    print("\nLatent Variable Estimation (CFA):")
    print("-"*50)

    lv_stats = {}
    for lv_name, item_prefix in constructs.items():
        items = [c for c in df.columns if c.startswith(item_prefix) and c[-1].isdigit()]

        if not items:
            continue

        lv_scores = estimate_latent_cfa(individuals, items, lv_name)
        individuals[f'LV_{lv_name}'] = lv_scores.values

        # Squared LV for quadratic effects
        individuals[f'LV_{lv_name}_sq'] = lv_scores.values ** 2

        X = individuals[items].values
        alpha = compute_cronbach_alpha(X)

        lv_stats[lv_name] = {'n_items': len(items), 'alpha': alpha}
        print(f"  {lv_name}: {len(items)} items, alpha={alpha:.3f}")

    # Merge LVs back
    lv_cols = [c for c in individuals.columns if c.startswith('LV_')]
    df = df.merge(individuals[['ID'] + lv_cols], on='ID', how='left')

    # LV × demographic interactions
    for lv in ['pat_blind', 'pat_const', 'sec_dl', 'sec_fp']:
        if f'LV_{lv}' in df.columns:
            df[f'LV_{lv}_x_age'] = df[f'LV_{lv}'] * df['age_c']
            df[f'LV_{lv}_x_inc'] = df[f'LV_{lv}'] * df['inc_c']

    # Numeric only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    return df[numeric_cols].copy(), lv_stats


# =============================================================================
# MODEL SPECIFICATIONS
# =============================================================================

def get_base_vars():
    """Get base choice variables."""
    return {
        'CHOICE': Variable('CHOICE'),
        'fee1': Variable('fee1_10k'),
        'fee2': Variable('fee2_10k'),
        'fee3': Variable('fee3_10k'),
        'dur1': Variable('dur1'),
        'dur2': Variable('dur2'),
        'dur3': Variable('dur3'),
    }


def model_1_pb_fee(database: db.Database):
    """M1: Blind Patriotism on fee sensitivity."""
    v = get_base_vars()
    LV_pb = Variable('LV_pat_blind')

    ASC_paid = Beta('ASC_paid', 0, -10, 10, 0)
    B_FEE = Beta('B_FEE', -0.5, -10, 0, 0)
    B_DUR = Beta('B_DUR', -0.05, -5, 0, 0)
    B_FEE_PB = Beta('B_FEE_PB', 0, -2, 2, 0)

    B_FEE_i = B_FEE + B_FEE_PB * LV_pb

    V1 = ASC_paid + B_FEE_i * v['fee1'] + B_DUR * v['dur1']
    V2 = ASC_paid + B_FEE_i * v['fee2'] + B_DUR * v['dur2']
    V3 = B_FEE_i * v['fee3'] + B_DUR * v['dur3']

    return models.loglogit({1: V1, 2: V2, 3: V3}, {1: 1, 2: 1, 3: 1}, v['CHOICE']), 'M1: PatBlind on Fee'


def model_2_sec_fee(database: db.Database):
    """M2: Daily Secularism on fee sensitivity."""
    v = get_base_vars()
    LV_sec = Variable('LV_sec_dl')

    ASC_paid = Beta('ASC_paid', 0, -10, 10, 0)
    B_FEE = Beta('B_FEE', -0.5, -10, 0, 0)
    B_DUR = Beta('B_DUR', -0.05, -5, 0, 0)
    B_FEE_SEC = Beta('B_FEE_SEC', 0, -2, 2, 0)

    B_FEE_i = B_FEE + B_FEE_SEC * LV_sec

    V1 = ASC_paid + B_FEE_i * v['fee1'] + B_DUR * v['dur1']
    V2 = ASC_paid + B_FEE_i * v['fee2'] + B_DUR * v['dur2']
    V3 = B_FEE_i * v['fee3'] + B_DUR * v['dur3']

    return models.loglogit({1: V1, 2: V2, 3: V3}, {1: 1, 2: 1, 3: 1}, v['CHOICE']), 'M2: SecDL on Fee'


def model_3_lv_duration(database: db.Database):
    """M3: LVs on duration sensitivity."""
    v = get_base_vars()
    LV_pb = Variable('LV_pat_blind')
    LV_sec = Variable('LV_sec_dl')

    ASC_paid = Beta('ASC_paid', 0, -10, 10, 0)
    B_FEE = Beta('B_FEE', -0.5, -10, 0, 0)
    B_DUR = Beta('B_DUR', -0.05, -5, 0, 0)
    B_DUR_PB = Beta('B_DUR_PB', 0, -1, 1, 0)
    B_DUR_SEC = Beta('B_DUR_SEC', 0, -1, 1, 0)

    B_DUR_i = B_DUR + B_DUR_PB * LV_pb + B_DUR_SEC * LV_sec

    V1 = ASC_paid + B_FEE * v['fee1'] + B_DUR_i * v['dur1']
    V2 = ASC_paid + B_FEE * v['fee2'] + B_DUR_i * v['dur2']
    V3 = B_FEE * v['fee3'] + B_DUR_i * v['dur3']

    return models.loglogit({1: V1, 2: V2, 3: V3}, {1: 1, 2: 1, 3: 1}, v['CHOICE']), 'M3: LVs on Duration'


def model_4_lv_asc(database: db.Database):
    """M4: LVs on ASC (preference for paid service)."""
    v = get_base_vars()
    LV_pb = Variable('LV_pat_blind')
    LV_sec = Variable('LV_sec_dl')

    ASC_paid = Beta('ASC_paid', 0, -10, 10, 0)
    B_FEE = Beta('B_FEE', -0.5, -10, 0, 0)
    B_DUR = Beta('B_DUR', -0.05, -5, 0, 0)
    B_ASC_PB = Beta('B_ASC_PB', 0, -5, 5, 0)
    B_ASC_SEC = Beta('B_ASC_SEC', 0, -5, 5, 0)

    ASC_i = ASC_paid + B_ASC_PB * LV_pb + B_ASC_SEC * LV_sec

    V1 = ASC_i + B_FEE * v['fee1'] + B_DUR * v['dur1']
    V2 = ASC_i + B_FEE * v['fee2'] + B_DUR * v['dur2']
    V3 = B_FEE * v['fee3'] + B_DUR * v['dur3']

    return models.loglogit({1: V1, 2: V2, 3: V3}, {1: 1, 2: 1, 3: 1}, v['CHOICE']), 'M4: LVs on ASC'


def model_5_quadratic_lv(database: db.Database):
    """M5: Quadratic LV effect - Diminishing or increasing returns."""
    v = get_base_vars()
    LV_pb = Variable('LV_pat_blind')
    LV_pb_sq = Variable('LV_pat_blind_sq')

    ASC_paid = Beta('ASC_paid', 0, -10, 10, 0)
    B_FEE = Beta('B_FEE', -0.5, -10, 0, 0)
    B_DUR = Beta('B_DUR', -0.05, -5, 0, 0)
    B_FEE_PB = Beta('B_FEE_PB', 0, -2, 2, 0)
    B_FEE_PB_SQ = Beta('B_FEE_PB_SQ', 0, -1, 1, 0)  # Quadratic effect

    B_FEE_i = B_FEE + B_FEE_PB * LV_pb + B_FEE_PB_SQ * LV_pb_sq

    V1 = ASC_paid + B_FEE_i * v['fee1'] + B_DUR * v['dur1']
    V2 = ASC_paid + B_FEE_i * v['fee2'] + B_DUR * v['dur2']
    V3 = B_FEE_i * v['fee3'] + B_DUR * v['dur3']

    return models.loglogit({1: V1, 2: V2, 3: V3}, {1: 1, 2: 1, 3: 1}, v['CHOICE']), 'M5: Quadratic LV'


def model_6_lv_demo_interaction(database: db.Database):
    """M6: LV × Demographic interactions."""
    v = get_base_vars()
    LV_pb = Variable('LV_pat_blind')
    LV_pb_x_age = Variable('LV_pat_blind_x_age')
    LV_pb_x_inc = Variable('LV_pat_blind_x_inc')

    ASC_paid = Beta('ASC_paid', 0, -10, 10, 0)
    B_FEE = Beta('B_FEE', -0.5, -10, 0, 0)
    B_DUR = Beta('B_DUR', -0.05, -5, 0, 0)

    # LV main effect
    B_FEE_PB = Beta('B_FEE_PB', 0, -2, 2, 0)
    # LV × demographic interactions
    B_FEE_PBxAGE = Beta('B_FEE_PBxAGE', 0, -1, 1, 0)
    B_FEE_PBxINC = Beta('B_FEE_PBxINC', 0, -1, 1, 0)

    B_FEE_i = B_FEE + B_FEE_PB * LV_pb + B_FEE_PBxAGE * LV_pb_x_age + B_FEE_PBxINC * LV_pb_x_inc

    V1 = ASC_paid + B_FEE_i * v['fee1'] + B_DUR * v['dur1']
    V2 = ASC_paid + B_FEE_i * v['fee2'] + B_DUR * v['dur2']
    V3 = B_FEE_i * v['fee3'] + B_DUR * v['dur3']

    return models.loglogit({1: V1, 2: V2, 3: V3}, {1: 1, 2: 1, 3: 1}, v['CHOICE']), 'M6: LV x Demo'


def model_7_domain_separation(database: db.Database):
    """M7: Patriotism on fee, Secularism on duration (domain separation)."""
    v = get_base_vars()
    LV_pb = Variable('LV_pat_blind')
    LV_pc = Variable('LV_pat_const')
    LV_sdl = Variable('LV_sec_dl')
    LV_sfp = Variable('LV_sec_fp')

    ASC_paid = Beta('ASC_paid', 0, -10, 10, 0)
    B_FEE = Beta('B_FEE', -0.5, -10, 0, 0)
    B_DUR = Beta('B_DUR', -0.05, -5, 0, 0)

    # Patriotism on fee
    B_FEE_PB = Beta('B_FEE_PB', 0, -2, 2, 0)
    B_FEE_PC = Beta('B_FEE_PC', 0, -2, 2, 0)

    # Secularism on duration
    B_DUR_SDL = Beta('B_DUR_SDL', 0, -1, 1, 0)
    B_DUR_SFP = Beta('B_DUR_SFP', 0, -1, 1, 0)

    B_FEE_i = B_FEE + B_FEE_PB * LV_pb + B_FEE_PC * LV_pc
    B_DUR_i = B_DUR + B_DUR_SDL * LV_sdl + B_DUR_SFP * LV_sfp

    V1 = ASC_paid + B_FEE_i * v['fee1'] + B_DUR_i * v['dur1']
    V2 = ASC_paid + B_FEE_i * v['fee2'] + B_DUR_i * v['dur2']
    V3 = B_FEE_i * v['fee3'] + B_DUR_i * v['dur3']

    return models.loglogit({1: V1, 2: V2, 3: V3}, {1: 1, 2: 1, 3: 1}, v['CHOICE']), 'M7: Domain Separation'


def model_8_full_specification(database: db.Database):
    """M8: Full specification - All 4 LVs on fee, duration, and ASC."""
    v = get_base_vars()
    LV_pb = Variable('LV_pat_blind')
    LV_pc = Variable('LV_pat_const')
    LV_sdl = Variable('LV_sec_dl')
    LV_sfp = Variable('LV_sec_fp')

    ASC_paid = Beta('ASC_paid', 0, -10, 10, 0)
    B_FEE = Beta('B_FEE', -0.5, -10, 0, 0)
    B_DUR = Beta('B_DUR', -0.05, -5, 0, 0)

    # All 4 LVs on fee
    B_FEE_PB = Beta('B_FEE_PB', 0, -2, 2, 0)
    B_FEE_PC = Beta('B_FEE_PC', 0, -2, 2, 0)
    B_FEE_SDL = Beta('B_FEE_SDL', 0, -2, 2, 0)
    B_FEE_SFP = Beta('B_FEE_SFP', 0, -2, 2, 0)

    # All 4 LVs on duration
    B_DUR_PB = Beta('B_DUR_PB', 0, -1, 1, 0)
    B_DUR_PC = Beta('B_DUR_PC', 0, -1, 1, 0)
    B_DUR_SDL = Beta('B_DUR_SDL', 0, -1, 1, 0)
    B_DUR_SFP = Beta('B_DUR_SFP', 0, -1, 1, 0)

    # 2 LVs on ASC
    B_ASC_PB = Beta('B_ASC_PB', 0, -3, 3, 0)
    B_ASC_SDL = Beta('B_ASC_SDL', 0, -3, 3, 0)

    ASC_i = ASC_paid + B_ASC_PB * LV_pb + B_ASC_SDL * LV_sdl
    B_FEE_i = B_FEE + B_FEE_PB * LV_pb + B_FEE_PC * LV_pc + B_FEE_SDL * LV_sdl + B_FEE_SFP * LV_sfp
    B_DUR_i = B_DUR + B_DUR_PB * LV_pb + B_DUR_PC * LV_pc + B_DUR_SDL * LV_sdl + B_DUR_SFP * LV_sfp

    V1 = ASC_i + B_FEE_i * v['fee1'] + B_DUR_i * v['dur1']
    V2 = ASC_i + B_FEE_i * v['fee2'] + B_DUR_i * v['dur2']
    V3 = B_FEE_i * v['fee3'] + B_DUR_i * v['dur3']

    return models.loglogit({1: V1, 2: V2, 3: V3}, {1: 1, 2: 1, 3: 1}, v['CHOICE']), 'M8: Full Specification'


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
    """Estimate a single HCM model."""
    try:
        logprob, name = model_func(database)

        biogeme = bio.BIOGEME(database, logprob)
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


def run_hcm_extended(data_path: str, output_dir: str = 'results/hcm_extended') -> Tuple[List[ModelResult], pd.DataFrame]:
    """Run extended HCM model comparison."""
    print("="*70)
    print("EXTENDED HCM MODEL COMPARISON")
    print("="*70)

    # Prepare data with LV estimation
    df, lv_stats = prepare_data(data_path)
    n_obs = len(df)
    n_ind = df['ID'].nunique() if 'ID' in df.columns else n_obs

    print(f"\nData: {n_obs} observations from {n_ind} individuals")

    # Create database
    database = db.Database('hcm_extended', df)
    null_ll = n_obs * np.log(1/3)

    # Model list
    model_funcs = [
        model_1_pb_fee,
        model_2_sec_fee,
        model_3_lv_duration,
        model_4_lv_asc,
        model_5_quadratic_lv,
        model_6_lv_demo_interaction,
        model_7_domain_separation,
        model_8_full_specification,
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

    # LR tests
    print("\n" + "="*70)
    print("LIKELIHOOD RATIO TESTS (vs Simplest Model)")
    print("="*70)

    baseline = min(results, key=lambda x: x.k)
    for r in sorted(results, key=lambda x: x.k):
        if r.name == baseline.name:
            continue
        if r.ll > baseline.ll:
            lr = 2 * (r.ll - baseline.ll)
            df_diff = r.k - baseline.k
            if df_diff > 0:
                p_val = 1 - stats.chi2.cdf(lr, df_diff)
                sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
                print(f"  {r.name}: LR={lr:.2f}, df={df_diff}, p={p_val:.4f} {sig}")

    # Parameter summary
    print("\n" + "="*70)
    print("SIGNIFICANT LV EFFECTS (|t| > 1.96)")
    print("="*70)

    for r in sorted(results, key=lambda x: x.aic)[:3]:
        sig_params = [(p, v, t) for p, v in r.params.items()
                      for t in [r.t_stats.get(p, 0)] if abs(t) > 1.96 and 'LV' in p or 'PB' in p or 'SEC' in p or 'ASC_P' in p]
        if sig_params:
            print(f"\n{r.name}:")
            for p, v, t in sig_params:
                print(f"  {p}: {v:.4f} (t={t:.2f})")

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

        # Save LV statistics
        pd.DataFrame([
            {'Construct': k, **v} for k, v in lv_stats.items()
        ]).to_csv(output_path / 'lv_statistics.csv', index=False)

        print(f"\nResults saved to: {output_path}")

    return results, comp_df


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extended HCM Model Comparison')
    parser.add_argument('--data', type=str, required=True, help='Path to data CSV')
    parser.add_argument('--output', type=str, default='results/hcm_extended',
                        help='Output directory')

    args = parser.parse_args()
    run_hcm_extended(args.data, args.output)
