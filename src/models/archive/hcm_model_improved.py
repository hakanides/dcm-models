"""
Improved Hybrid Choice Model (HCM)
==================================

STATUS: Intermediate implementation
RECOMMENDATION: Use hcm_split_latents.py for production runs (has robust bounds)

Key improvements over original hcm_model.py:
1. Confirmatory Factor Analysis (respects item groupings)
2. Better standardization of estimated latents
3. Multiple model specifications
4. MIMIC model structure

NOTE: For simulation validation with TRUE latent values (benchmarking),
see validation_models.py. This file contains only real-data-applicable models.

METHODOLOGICAL LIMITATION - MEASUREMENT ERROR
=============================================
This is a TWO-STAGE approach that treats estimated LVs as error-free.
This causes ATTENUATION BIAS: LV effects are biased toward zero.

Interpretation:
- Significant effects are likely real (conservative test)
- Non-significant effects may still exist (attenuated)
- Effect magnitudes are underestimated

For unbiased estimates, full ICLV with simultaneous estimation is needed.

Author: DCM Research Team
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from scipy import stats
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings
# Selective warning suppression - allow important warnings through
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*overflow.*')
warnings.filterwarnings('ignore', message='.*divide by zero.*')

import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta, Variable


# =============================================================================
# LATENT VARIABLE ESTIMATION - IMPROVED
# =============================================================================

def estimate_latent_cfa(df: pd.DataFrame) -> pd.DataFrame:
    """
    Confirmatory Factor Analysis approach.
    Groups items by their theoretical construct.
    """
    print("\n" + "=" * 60)
    print("LATENT VARIABLE ESTIMATION - CFA APPROACH")
    print("=" * 60)

    df = df.copy()

    # Define item groups by construct
    constructs = {
        'pat_blind': [c for c in df.columns if c.startswith('pat_blind_') and c[-1].isdigit()],
        'pat_constructive': [c for c in df.columns if c.startswith('pat_constructive_') and c[-1].isdigit()],
        'sec_dl': [c for c in df.columns if c.startswith('sec_dl_') and c[-1].isdigit()],
        'sec_fp': [c for c in df.columns if c.startswith('sec_fp_') and c[-1].isdigit()],
    }

    print("\nItems per construct:")
    for name, items in constructs.items():
        print(f"  {name}: {len(items)} items")

    # Get unique individuals
    individuals = df.groupby('ID').first().reset_index()
    n_ind = len(individuals)
    print(f"\nEstimating for {n_ind} individuals...")

    # Method 1: Weighted sum scores (item-total correlations as weights)
    for construct, items in constructs.items():
        if not items:
            continue

        X = individuals[items].values

        # Calculate item-total correlations
        total = X.sum(axis=1)
        weights = []
        for i in range(X.shape[1]):
            corr = np.corrcoef(X[:, i], total)[0, 1]
            weights.append(max(0.1, corr))  # Minimum weight of 0.1

        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize

        # Weighted sum
        weighted_score = (X * weights).sum(axis=1)

        # Standardize to N(0,1)
        weighted_score = (weighted_score - weighted_score.mean()) / weighted_score.std()

        individuals[f'LV_{construct}_cfa'] = weighted_score
        print(f"  {construct}: weights = {np.round(weights, 3)}")

    # Method 2: Simple standardized means (baseline)
    for construct, items in constructs.items():
        if not items:
            continue
        mean_score = individuals[items].mean(axis=1)
        mean_score = (mean_score - mean_score.mean()) / mean_score.std()
        individuals[f'LV_{construct}_mean'] = mean_score

    # Merge back - only the newly created CFA/mean columns
    new_lv_cols = [c for c in individuals.columns if c.startswith('LV_') and ('_cfa' in c or '_mean' in c)]
    df_with_lv = df.merge(individuals[['ID'] + new_lv_cols], on='ID', how='left')

    # Keep original true LV columns from df if they exist
    # They should already be in df_with_lv since we merged

    return df_with_lv


# NOTE: validate_latent_estimation() moved to validation_models.py
# It requires TRUE latent values (simulation-only)


# =============================================================================
# HCM CHOICE MODELS
# =============================================================================

def get_choice_vars():
    """Get common choice variables."""
    return {
        'dur1': Variable('dur1'),
        'dur2': Variable('dur2'),
        'dur3': Variable('dur3'),
        'fee1_10k': Variable('fee1_10k'),
        'fee2_10k': Variable('fee2_10k'),
        'fee3_10k': Variable('fee3_10k'),
        'CHOICE': Variable('CHOICE'),
    }


# NOTE: hcm_with_true_lv() moved to validation_models.py
# It requires TRUE latent values (simulation-only benchmark)


def hcm_with_cfa_lv(database: db.Database):
    """HCM using CFA-estimated latent variables."""
    v = get_choice_vars()

    # CFA-estimated latent variables
    LV_pat_blind = Variable('LV_pat_blind_cfa')
    LV_sec_dl = Variable('LV_sec_dl_cfa')

    # Parameters with bounds
    ASC_paid = Beta('ASC_paid', 0, -10, 10, 0)
    B_FEE = Beta('B_FEE', -0.6, -10, 0, 0)
    B_DUR = Beta('B_DUR', -0.05, -5, 0, 0)

    # LV interactions (bounded)
    B_FEE_PAT = Beta('B_FEE_PAT', 0, -2, 2, 0)
    B_DUR_PAT = Beta('B_DUR_PAT', 0, -1, 1, 0)
    B_FEE_SEC = Beta('B_FEE_SEC', 0, -2, 2, 0)
    B_DUR_SEC = Beta('B_DUR_SEC', 0, -1, 1, 0)

    B_FEE_i = B_FEE + B_FEE_PAT * LV_pat_blind + B_FEE_SEC * LV_sec_dl
    B_DUR_i = B_DUR + B_DUR_PAT * LV_pat_blind + B_DUR_SEC * LV_sec_dl

    # Utilities
    V1 = ASC_paid + B_FEE_i * v['fee1_10k'] + B_DUR_i * v['dur1']
    V2 = ASC_paid + B_FEE_i * v['fee2_10k'] + B_DUR_i * v['dur2']
    V3 = B_FEE_i * v['fee3_10k'] + B_DUR_i * v['dur3']

    V = {1: V1, 2: V2, 3: V3}
    av = {1: 1, 2: 1, 3: 1}

    logprob = models.loglogit(V, av, v['CHOICE'])
    return logprob, 'HCM-CFA: CFA Latent Variables'


def hcm_full_cfa(database: db.Database):
    """HCM with all 4 CFA latent variables."""
    v = get_choice_vars()

    # All 4 CFA-estimated latent variables
    LV_pat_blind = Variable('LV_pat_blind_cfa')
    LV_pat_const = Variable('LV_pat_constructive_cfa')
    LV_sec_dl = Variable('LV_sec_dl_cfa')
    LV_sec_fp = Variable('LV_sec_fp_cfa')

    # Parameters with bounds
    ASC_paid = Beta('ASC_paid', 0, -10, 10, 0)
    B_FEE = Beta('B_FEE', -0.6, -10, 0, 0)
    B_DUR = Beta('B_DUR', -0.05, -5, 0, 0)

    # Fee interactions with all LVs (bounded)
    B_FEE_PAT_B = Beta('B_FEE_PAT_B', 0, -2, 2, 0)
    B_FEE_PAT_C = Beta('B_FEE_PAT_C', 0, -2, 2, 0)
    B_FEE_SEC_DL = Beta('B_FEE_SEC_DL', 0, -2, 2, 0)
    B_FEE_SEC_FP = Beta('B_FEE_SEC_FP', 0, -2, 2, 0)

    # Duration interactions (bounded)
    B_DUR_PAT_B = Beta('B_DUR_PAT_B', 0, -1, 1, 0)
    B_DUR_SEC_DL = Beta('B_DUR_SEC_DL', 0, -1, 1, 0)

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
    return logprob, 'HCM-CFA-Full: All CFA Latent Variables'


def hcm_mimic(database: db.Database):
    """
    MIMIC-style model: Demographics -> Latent -> Choice
    Uses demographics as instruments for latent variables.
    """
    v = get_choice_vars()

    # Demographics
    age = Variable('age_idx')
    edu = Variable('edu_idx')
    income = Variable('income_indiv_idx')
    marital = Variable('marital_idx')

    # CFA latent variables
    LV_pat_blind = Variable('LV_pat_blind_cfa')
    LV_sec_dl = Variable('LV_sec_dl_cfa')

    # Parameters with bounds
    ASC_paid = Beta('ASC_paid', 0, -10, 10, 0)
    B_FEE = Beta('B_FEE', -0.6, -10, 0, 0)
    B_DUR = Beta('B_DUR', -0.05, -5, 0, 0)

    # Demographics on fee (bounded)
    B_FEE_AGE = Beta('B_FEE_AGE', 0, -1, 1, 0)
    B_FEE_EDU = Beta('B_FEE_EDU', 0, -1, 1, 0)
    B_FEE_INC = Beta('B_FEE_INC', 0, -1, 1, 0)

    # Demographics on duration (bounded)
    B_DUR_MARITAL = Beta('B_DUR_MARITAL', 0, -0.5, 0.5, 0)

    # LV on fee (bounded)
    B_FEE_PAT = Beta('B_FEE_PAT', 0, -2, 2, 0)
    B_FEE_SEC = Beta('B_FEE_SEC', 0, -2, 2, 0)

    # Combined coefficient
    B_FEE_i = (B_FEE +
               B_FEE_AGE * age + B_FEE_EDU * edu + B_FEE_INC * income +
               B_FEE_PAT * LV_pat_blind + B_FEE_SEC * LV_sec_dl)

    B_DUR_i = B_DUR + B_DUR_MARITAL * marital

    # Utilities
    V1 = ASC_paid + B_FEE_i * v['fee1_10k'] + B_DUR_i * v['dur1']
    V2 = ASC_paid + B_FEE_i * v['fee2_10k'] + B_DUR_i * v['dur2']
    V3 = B_FEE_i * v['fee3_10k'] + B_DUR_i * v['dur3']

    V = {1: V1, 2: V2, 3: V3}
    av = {1: 1, 2: 1, 3: 1}

    logprob = models.loglogit(V, av, v['CHOICE'])
    return logprob, 'HCM-MIMIC: Demographics + Latent'


def mnl_baseline(database: db.Database):
    """
    Basic MNL without latent variables.
    This is the baseline to beat.
    """
    v = get_choice_vars()

    ASC_paid = Beta('ASC_paid', 0, -10, 10, 0)
    B_FEE = Beta('B_FEE', -0.6, -10, 0, 0)
    B_DUR = Beta('B_DUR', -0.05, -5, 0, 0)

    V1 = ASC_paid + B_FEE * v['fee1_10k'] + B_DUR * v['dur1']
    V2 = ASC_paid + B_FEE * v['fee2_10k'] + B_DUR * v['dur2']
    V3 = B_FEE * v['fee3_10k'] + B_DUR * v['dur3']

    V = {1: V1, 2: V2, 3: V3}
    av = {1: 1, 2: 1, 3: 1}

    logprob = models.loglogit(V, av, v['CHOICE'])
    return logprob, 'MNL-Baseline'


def hcm_simple_lv(database: db.Database):
    """
    Simple HCM: Just patriotism effect on fee.
    Minimal specification to test if LV has any effect.
    """
    v = get_choice_vars()

    LV_pat_blind = Variable('LV_pat_blind_cfa')

    ASC_paid = Beta('ASC_paid', 0, -10, 10, 0)
    B_FEE = Beta('B_FEE', -0.6, -10, 0, 0)
    B_DUR = Beta('B_DUR', -0.05, -5, 0, 0)

    B_FEE_PAT = Beta('B_FEE_PAT', 0, -2, 2, 0)

    B_FEE_i = B_FEE + B_FEE_PAT * LV_pat_blind

    V1 = ASC_paid + B_FEE_i * v['fee1_10k'] + B_DUR * v['dur1']
    V2 = ASC_paid + B_FEE_i * v['fee2_10k'] + B_DUR * v['dur2']
    V3 = B_FEE_i * v['fee3_10k'] + B_DUR * v['dur3']

    V = {1: V1, 2: V2, 3: V3}
    av = {1: 1, 2: 1, 3: 1}

    logprob = models.loglogit(V, av, v['CHOICE'])
    return logprob, 'HCM-Simple: Patriotism on Fee'


# =============================================================================
# ESTIMATION
# =============================================================================

@dataclass
class HCMResults:
    name: str
    log_likelihood: float
    n_parameters: int
    aic: float
    bic: float
    rho_sq: float
    parameters: Dict[str, float]
    std_errors: Dict[str, float]
    converged: bool


def prepare_database(df: pd.DataFrame) -> db.Database:
    """Prepare Biogeme database - matching working MNL setup exactly."""
    df = df.copy()

    # Scale fees
    df['fee1_10k'] = df['fee1'] / 10000.0
    df['fee2_10k'] = df['fee2'] / 10000.0
    df['fee3_10k'] = df['fee3'] / 10000.0

    # Center demographics (same as MNL)
    df['age_c'] = (df['age_idx'] - 2) / 2
    df['edu_c'] = (df['edu_idx'] - 3) / 2
    df['inc_c'] = (df['income_indiv_idx'] - 3) / 2
    df['inc_hh_c'] = (df['income_house_idx'] - 3) / 2
    df['marital_c'] = (df['marital_idx'] - 0.5) / 0.5

    # Create Likert indices (same as MNL)
    pat_blind_cols = [c for c in df.columns if c.startswith('pat_blind_') and c[-1].isdigit()]
    if pat_blind_cols:
        df['pat_blind_idx'] = df[pat_blind_cols].mean(axis=1)
        df['pat_blind_idx_c'] = (df['pat_blind_idx'] - 3) / 2

    pat_const_cols = [c for c in df.columns if c.startswith('pat_constructive_') and c[-1].isdigit()]
    if pat_const_cols:
        df['pat_const_idx'] = df[pat_const_cols].mean(axis=1)
        df['pat_const_idx_c'] = (df['pat_const_idx'] - 3) / 2

    sec_dl_cols = [c for c in df.columns if c.startswith('sec_dl_') and c[-1].isdigit()]
    if sec_dl_cols:
        df['sec_dl_idx'] = df[sec_dl_cols].mean(axis=1)
        df['sec_dl_idx_c'] = (df['sec_dl_idx'] - 3) / 2

    sec_fp_cols = [c for c in df.columns if c.startswith('sec_fp_') and c[-1].isdigit()]
    if sec_fp_cols:
        df['sec_fp_idx'] = df[sec_fp_cols].mean(axis=1)
        df['sec_fp_idx_c'] = (df['sec_fp_idx'] - 3) / 2

    # Drop string columns
    string_cols = df.select_dtypes(include=['object']).columns.tolist()
    df_num = df.drop(columns=string_cols)

    print(f"\nDatabase prepared: {len(df_num)} rows, {len(df_num.columns)} columns")
    print(f"LV columns: {[c for c in df_num.columns if c.startswith('LV_')]}")

    return db.Database('hcm_improved', df_num)


def estimate_model(database: db.Database, model_func, null_ll: float) -> Optional[HCMResults]:
    """Estimate single HCM model."""
    try:
        logprob, name = model_func(database)

        print(f"\n{name}")
        print("-" * 50)

        biogeme_model = bio.BIOGEME(database, logprob)
        results = biogeme_model.estimate()

        ll = results.final_loglikelihood
        n_params = results.number_of_free_parameters
        n_obs = len(database.dataframe)

        aic = 2 * n_params - 2 * ll
        bic = n_params * np.log(n_obs) - 2 * ll
        rho_sq = 1 - (ll / null_ll)

        betas = results.get_beta_values()
        std_errs = {}
        for p in betas:
            try:
                std_errs[p] = results.get_parameter_std_err(p)
            except:
                std_errs[p] = np.nan

        converged = results.algorithm_has_converged

        print(f"  LL: {ll:.2f} | K: {n_params} | AIC: {aic:.2f} | rho2: {rho_sq:.4f}")

        for p, val in betas.items():
            se = std_errs.get(p, np.nan)
            t = val / se if se > 0 else np.nan
            sig = '***' if abs(t) > 2.576 else '**' if abs(t) > 1.96 else '*' if abs(t) > 1.645 else ''
            print(f"  {p:20s}: {val:8.4f} (t={t:6.2f}) {sig}")

        return HCMResults(
            name=name,
            log_likelihood=ll,
            n_parameters=n_params,
            aic=aic,
            bic=bic,
            rho_sq=rho_sq,
            parameters=betas,
            std_errors=std_errs,
            converged=converged
        )

    except Exception as e:
        print(f"  ERROR: {e}")
        return None


# =============================================================================
# MAIN
# =============================================================================

def run_improved_hcm(data_path: str, output_dir: str = None):
    """Run improved HCM estimation."""
    print("=" * 60)
    print("IMPROVED HYBRID CHOICE MODEL (HCM)")
    print("=" * 60)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

    # Load data
    df = pd.read_csv(data_path)
    n_obs = len(df)
    n_ind = df['ID'].nunique()
    print(f"\nLoaded {n_obs:,} observations from {n_ind} respondents")

    # Null LL
    null_ll = n_obs * np.log(1/3)
    print(f"Null Log-Likelihood: {null_ll:.2f}")

    # Stage 1: Estimate latent variables
    df_with_lv = estimate_latent_cfa(df)

    # NOTE: For validation against true LVs (simulation only), use:
    # from src.models.validation_models import validate_latent_estimation
    validation_df = None

    # Stage 2: Choice models
    print("\n" + "=" * 60)
    print("STAGE 2: CHOICE MODEL ESTIMATION")
    print("=" * 60)

    database = prepare_database(df_with_lv)

    # Models to estimate (real-data applicable)
    # NOTE: For simulation benchmarking with true LVs, see validation_models.py
    model_funcs = [
        mnl_baseline,          # Baseline MNL (no LVs)
        hcm_simple_lv,         # Simple: 1 LV effect
        hcm_with_cfa_lv,       # CFA: 2 LV effects
        hcm_full_cfa,          # CFA: All 4 LV effects
        hcm_mimic,             # MIMIC: Demo + LV
    ]

    results = []
    for model_func in model_funcs:
        res = estimate_model(database, model_func, null_ll)
        if res:
            results.append(res)

    # Summary comparison
    print("\n" + "=" * 60)
    print("HCM MODEL COMPARISON")
    print("=" * 60)

    data = []
    for r in results:
        data.append({
            'Model': r.name,
            'LL': r.log_likelihood,
            'K': r.n_parameters,
            'AIC': r.aic,
            'BIC': r.bic,
            'rho2': r.rho_sq,
            'Conv': 'Yes' if r.converged else 'No'
        })

    comp_df = pd.DataFrame(data).sort_values('AIC')
    print("\n" + comp_df.to_string(index=False))

    # Best model
    if len(comp_df) > 0:
        best = comp_df.iloc[0]
        print(f"\nBest by AIC: {best['Model']}")
        print(f"  AIC = {best['AIC']:.2f}, rho2 = {best['rho2']:.4f}")

    # Save outputs
    if output_dir:
        comp_df.to_csv(output_dir / 'hcm_comparison.csv', index=False)

        if validation_df is not None:
            validation_df.to_csv(output_dir / 'latent_validation.csv', index=False)

        # Parameters
        param_rows = []
        for r in results:
            for p, v in r.parameters.items():
                se = r.std_errors.get(p, np.nan)
                param_rows.append({
                    'Model': r.name,
                    'Parameter': p,
                    'Estimate': v,
                    'Std.Err': se,
                    't-stat': v/se if se > 0 else np.nan
                })
        pd.DataFrame(param_rows).to_csv(output_dir / 'hcm_parameters.csv', index=False)

        print(f"\nOutputs saved to: {output_dir}")

    return results, comp_df


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', default='hcm_improved_results')
    args = parser.parse_args()

    run_improved_hcm(args.data, args.output)


if __name__ == '__main__':
    main()
