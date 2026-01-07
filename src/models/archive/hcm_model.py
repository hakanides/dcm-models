"""
Hybrid Choice Model (HCM) / ICLV Model
======================================

STATUS: Basic implementation (SUPERSEDED by hcm_split_latents.py)
RECOMMENDATION: Use hcm_split_latents.py for production runs

Implements an Integrated Choice and Latent Variable (ICLV) model
using a two-stage estimation approach:

Stage 1: Estimate latent variables using factor analysis on Likert items
Stage 2: Use estimated latent variables in choice model

Model Structure:
    Demographics ──> Latent Variables ──> Choice
                            │
                            └──> Likert Indicators

METHODOLOGICAL LIMITATION - MEASUREMENT ERROR
=============================================
WARNING: This is a TWO-STAGE approach, NOT true simultaneous ICLV.

The two-stage approach treats estimated LVs as ERROR-FREE, which causes
ATTENUATION BIAS: LV effect estimates are systematically biased toward zero.

This means:
- Significant LV effects found here are likely real (conservative)
- Non-significant effects may actually exist but are attenuated
- Effect MAGNITUDES are underestimated

For unbiased estimates, full ICLV with simultaneous estimation of the
measurement model and choice model is required. This needs specialized
software (e.g., Apollo in R, or custom likelihood in Python).

Author: DCM Research Team
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil
from pathlib import Path
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
from scipy import stats
from dataclasses import dataclass
from typing import Dict, List, Tuple
import warnings
# Selective warning suppression - allow important warnings through
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*overflow.*')
warnings.filterwarnings('ignore', message='.*divide by zero.*')


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

import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta, Variable


# =============================================================================
# STAGE 1: LATENT VARIABLE ESTIMATION
# =============================================================================

def estimate_latent_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate latent variables using factor analysis.

    Uses Likert items to estimate latent factors:
    - Patriotism (Blind + Constructive)
    - Secularism (Daily Life + Faith/Prayer)
    """
    print("\n" + "=" * 60)
    print("STAGE 1: LATENT VARIABLE ESTIMATION")
    print("=" * 60)

    df = df.copy()

    # Define item groups
    pat_blind_items = [c for c in df.columns if c.startswith('pat_blind_') and not c.endswith('_cont')]
    pat_const_items = [c for c in df.columns if c.startswith('pat_constructive_') and not c.endswith('_cont')]
    sec_dl_items = [c for c in df.columns if c.startswith('sec_dl_') and not c.endswith('_cont')]
    sec_fp_items = [c for c in df.columns if c.startswith('sec_fp_') and not c.endswith('_cont')]

    print(f"Likert items found:")
    print(f"  Patriotism-Blind: {len(pat_blind_items)}")
    print(f"  Patriotism-Constructive: {len(pat_const_items)}")
    print(f"  Secularism-DailyLife: {len(sec_dl_items)}")
    print(f"  Secularism-Faith: {len(sec_fp_items)}")

    # Get unique individuals (latent vars are constant within person)
    # Use first row per individual
    id_vars = ['ID', 'ID_STR'] if 'ID_STR' in df.columns else ['ID']
    individuals = df.groupby('ID').first().reset_index()

    print(f"\nEstimating latent variables for {len(individuals)} individuals...")

    # Method 1: Simple averages (as before)
    if pat_blind_items:
        individuals['LV_pat_blind_est'] = individuals[pat_blind_items].mean(axis=1)
        individuals['LV_pat_blind_est'] = (individuals['LV_pat_blind_est'] - 3) / 2  # Center

    if pat_const_items:
        individuals['LV_pat_const_est'] = individuals[pat_const_items].mean(axis=1)
        individuals['LV_pat_const_est'] = (individuals['LV_pat_const_est'] - 3) / 2

    if sec_dl_items:
        individuals['LV_sec_dl_est'] = individuals[sec_dl_items].mean(axis=1)
        individuals['LV_sec_dl_est'] = (individuals['LV_sec_dl_est'] - 3) / 2

    if sec_fp_items:
        individuals['LV_sec_fp_est'] = individuals[sec_fp_items].mean(axis=1)
        individuals['LV_sec_fp_est'] = (individuals['LV_sec_fp_est'] - 3) / 2

    # Method 2: Factor Analysis (more sophisticated)
    all_items = pat_blind_items + pat_const_items + sec_dl_items + sec_fp_items
    if len(all_items) >= 4:
        X = individuals[all_items].values
        X = StandardScaler().fit_transform(X)

        # 4 factors for 4 latent variables
        fa = FactorAnalysis(n_components=4, random_state=42)
        factor_scores = fa.fit_transform(X)

        individuals['LV_factor_1'] = factor_scores[:, 0]
        individuals['LV_factor_2'] = factor_scores[:, 1]
        individuals['LV_factor_3'] = factor_scores[:, 2]
        individuals['LV_factor_4'] = factor_scores[:, 3]

        print("\nFactor Analysis Results:")
        print(f"  Variance explained: {fa.noise_variance_.mean():.4f} (noise)")

    # Print summary stats
    lv_cols = [c for c in individuals.columns if c.startswith('LV_') and '_est' in c]
    print("\nEstimated Latent Variables Summary:")
    for col in lv_cols:
        vals = individuals[col]
        print(f"  {col}: mean={vals.mean():.3f}, std={vals.std():.3f}")

    # Merge back to full panel data
    lv_cols_to_merge = [c for c in individuals.columns if c.startswith('LV_')]
    df_with_lv = df.merge(individuals[['ID'] + lv_cols_to_merge], on='ID', how='left')

    return df_with_lv


def validate_latent_estimation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare estimated vs true latent values.

    WARNING: SIMULATION DATA ONLY
    This function requires true latent value columns (*_true) which are
    only available in simulated data. For comprehensive validation,
    see validation_models.py.
    """
    true_cols = [c for c in df.columns if c.startswith('LV_') and '_true' in c]
    est_cols = [c for c in df.columns if c.startswith('LV_') and '_est' in c]

    if not true_cols or not est_cols:
        return None

    print("\n" + "-" * 40)
    print("Latent Variable Validation (SIMULATION DATA)")
    print("-" * 40)

    # Get unique individuals
    individuals = df.groupby('ID').first().reset_index()

    validation = []
    mapping = {
        'LV_pat_blind_est': 'LV_pat_blind_true',
        'LV_pat_const_est': 'LV_pat_constructive_true',
        'LV_sec_dl_est': 'LV_sec_dl_true',
        'LV_sec_fp_est': 'LV_sec_fp_true',
    }

    for est_col, true_col in mapping.items():
        if est_col in individuals.columns and true_col in individuals.columns:
            corr = individuals[est_col].corr(individuals[true_col])
            rmse = np.sqrt(((individuals[est_col] - individuals[true_col])**2).mean())
            print(f"  {est_col.replace('_est', '')}: Corr={corr:.3f}, RMSE={rmse:.3f}")
            validation.append({
                'Latent': est_col.replace('LV_', '').replace('_est', ''),
                'Correlation': corr,
                'RMSE': rmse
            })

    return pd.DataFrame(validation)


# =============================================================================
# STAGE 2: CHOICE MODEL WITH LATENT VARIABLES
# =============================================================================

def prepare_hcm_data(df: pd.DataFrame) -> db.Database:
    """Prepare data for HCM choice model."""
    df = df.copy()

    # Scale fees
    df['fee1_10k'] = df['fee1'] / 10000.0
    df['fee2_10k'] = df['fee2'] / 10000.0
    df['fee3_10k'] = df['fee3'] / 10000.0

    # Drop strings
    string_cols = df.select_dtypes(include=['object']).columns.tolist()
    df_num = df.drop(columns=string_cols)

    return db.Database('hcm_data', df_num)


def hcm_choice_model(database: db.Database):
    """
    HCM Choice Model using estimated latent variables.

    Utility = ASC + B_FEE*fee + B_DUR*dur + B_FEE_PAT*LV_pat + B_DUR_SEC*LV_sec + ...
    """
    # Variables
    dur1 = Variable('dur1')
    dur2 = Variable('dur2')
    dur3 = Variable('dur3')
    fee1_10k = Variable('fee1_10k')
    fee2_10k = Variable('fee2_10k')
    fee3_10k = Variable('fee3_10k')
    CHOICE = Variable('CHOICE')

    # Estimated latent variables
    LV_pat_blind = Variable('LV_pat_blind_est')
    LV_sec_dl = Variable('LV_sec_dl_est')

    # Parameters
    ASC_paid = Beta('ASC_paid', 0, None, None, 0)
    B_FEE = Beta('B_FEE', -0.5, -10, 0, 0)  # Bounded negative
    B_DUR = Beta('B_DUR', -0.05, -5, 0, 0)  # Bounded negative

    # Latent variable interactions
    B_FEE_PAT = Beta('B_FEE_PAT', 0, -2, 2, 0)  # Patriotism effect on fee sensitivity
    B_DUR_PAT = Beta('B_DUR_PAT', 0, -1, 1, 0)  # Patriotism effect on duration
    B_FEE_SEC = Beta('B_FEE_SEC', 0, -2, 2, 0)  # Secularism effect on fee
    B_DUR_SEC = Beta('B_DUR_SEC', 0, -1, 1, 0)  # Secularism effect on duration

    # Individual-specific coefficients
    B_FEE_i = B_FEE + B_FEE_PAT * LV_pat_blind + B_FEE_SEC * LV_sec_dl
    B_DUR_i = B_DUR + B_DUR_PAT * LV_pat_blind + B_DUR_SEC * LV_sec_dl

    # Utilities
    V1 = ASC_paid + B_FEE_i * fee1_10k + B_DUR_i * dur1
    V2 = ASC_paid + B_FEE_i * fee2_10k + B_DUR_i * dur2
    V3 = B_FEE_i * fee3_10k + B_DUR_i * dur3

    V = {1: V1, 2: V2, 3: V3}
    av = {1: 1, 2: 1, 3: 1}

    logprob = models.loglogit(V, av, CHOICE)

    return logprob, 'HCM: Latent Variable Model'


def hcm_full_model(database: db.Database):
    """
    Full HCM with all latent variables and demographics.
    """
    # Variables
    dur1 = Variable('dur1')
    dur2 = Variable('dur2')
    dur3 = Variable('dur3')
    fee1_10k = Variable('fee1_10k')
    fee2_10k = Variable('fee2_10k')
    fee3_10k = Variable('fee3_10k')
    CHOICE = Variable('CHOICE')

    # Latent variables
    LV_pat_blind = Variable('LV_pat_blind_est')
    LV_pat_const = Variable('LV_pat_const_est')
    LV_sec_dl = Variable('LV_sec_dl_est')
    LV_sec_fp = Variable('LV_sec_fp_est')

    # Parameters
    ASC_paid = Beta('ASC_paid', 0, None, None, 0)
    B_FEE = Beta('B_FEE', -0.5, -10, 0, 0)  # Bounded negative
    B_DUR = Beta('B_DUR', -0.05, -5, 0, 0)  # Bounded negative

    # Latent variable effects on fee
    B_FEE_PAT_B = Beta('B_FEE_PAT_B', 0, -2, 2, 0)
    B_FEE_PAT_C = Beta('B_FEE_PAT_C', 0, -2, 2, 0)
    B_FEE_SEC_DL = Beta('B_FEE_SEC_DL', 0, -2, 2, 0)
    B_FEE_SEC_FP = Beta('B_FEE_SEC_FP', 0, -2, 2, 0)

    # Latent variable effects on duration
    B_DUR_PAT_B = Beta('B_DUR_PAT_B', 0, -1, 1, 0)
    B_DUR_SEC_DL = Beta('B_DUR_SEC_DL', 0, -1, 1, 0)

    # Coefficients with LV effects
    B_FEE_i = (B_FEE +
               B_FEE_PAT_B * LV_pat_blind + B_FEE_PAT_C * LV_pat_const +
               B_FEE_SEC_DL * LV_sec_dl + B_FEE_SEC_FP * LV_sec_fp)

    B_DUR_i = (B_DUR +
               B_DUR_PAT_B * LV_pat_blind + B_DUR_SEC_DL * LV_sec_dl)

    # Utilities
    V1 = ASC_paid + B_FEE_i * fee1_10k + B_DUR_i * dur1
    V2 = ASC_paid + B_FEE_i * fee2_10k + B_DUR_i * dur2
    V3 = B_FEE_i * fee3_10k + B_DUR_i * dur3

    V = {1: V1, 2: V2, 3: V3}
    av = {1: 1, 2: 1, 3: 1}

    logprob = models.loglogit(V, av, CHOICE)

    return logprob, 'HCM-Full: All Latent Variables'


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
    parameters: Dict[str, float]
    std_errors: Dict[str, float]
    converged: bool


def estimate_hcm(database: db.Database, model_func, output_dir: Path = None) -> HCMResults:
    """Estimate HCM choice model."""
    logprob, name = model_func(database)

    print(f"\n{name}")
    print("-" * 40)

    biogeme_model = bio.BIOGEME(database, logprob)

    # Set model name with output directory path for HTML files
    safe_name = name.replace(':', '').replace(' ', '_').replace('-', '_')
    if output_dir:
        biogeme_model.model_name = str(output_dir / safe_name)
    else:
        biogeme_model.model_name = safe_name

    results = biogeme_model.estimate()

    ll = results.final_loglikelihood
    n_params = results.number_of_free_parameters
    n_obs = len(database.dataframe)

    aic = 2 * n_params - 2 * ll
    bic = n_params * np.log(n_obs) - 2 * ll

    betas = results.get_beta_values()
    std_errs = {}
    for p in betas:
        try:
            std_errs[p] = results.get_parameter_std_err(p)
        except:
            std_errs[p] = np.nan

    converged = results.algorithm_has_converged

    print(f"  LL: {ll:.2f} | K: {n_params} | AIC: {aic:.2f} | Conv: {converged}")

    for p, val in betas.items():
        se = std_errs.get(p, np.nan)
        t = val / se if se > 0 else np.nan
        print(f"  {p}: {val:.4f} (t={t:.2f})")

    return HCMResults(
        name=name,
        log_likelihood=ll,
        n_parameters=n_params,
        aic=aic,
        bic=bic,
        parameters=betas,
        std_errors=std_errs,
        converged=converged
    )


# =============================================================================
# MAIN
# =============================================================================

def run_hcm_estimation(data_path: str, output_dir: str = None):
    """Run full HCM estimation pipeline."""
    print("=" * 60)
    print("HYBRID CHOICE MODEL (HCM) ESTIMATION")
    print("=" * 60)

    # Setup and cleanup
    cleanup_iter_files()  # Delete .iter files from project root

    if output_dir:
        output_dir = Path(output_dir)
        cleanup_results_directory(output_dir)  # Clean and recreate output directory
    else:
        output_dir = None

    # Load data
    df = pd.read_csv(data_path)
    print(f"\nLoaded {len(df):,} observations from {df['ID'].nunique()} respondents")

    # Stage 1: Estimate latent variables
    df_with_lv = estimate_latent_variables(df)

    # Validate if true values exist
    validation_df = validate_latent_estimation(df_with_lv)

    # Stage 2: Choice model
    print("\n" + "=" * 60)
    print("STAGE 2: CHOICE MODEL ESTIMATION")
    print("=" * 60)

    database = prepare_hcm_data(df_with_lv)

    # Null LL
    n_obs = len(database.dataframe)
    null_ll = n_obs * np.log(1/3)

    # Estimate models
    results = []

    try:
        res1 = estimate_hcm(database, hcm_choice_model, output_dir)
        results.append(res1)
    except Exception as e:
        print(f"Error in HCM: {e}")

    try:
        res2 = estimate_hcm(database, hcm_full_model, output_dir)
        results.append(res2)
    except Exception as e:
        print(f"Error in HCM-Full: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("HCM RESULTS SUMMARY")
    print("=" * 60)

    data = []
    for r in results:
        rho_sq = 1 - (r.log_likelihood / null_ll)
        data.append({
            'Model': r.name,
            'LL': r.log_likelihood,
            'K': r.n_parameters,
            'AIC': r.aic,
            'BIC': r.bic,
            'ρ²': rho_sq,
            'Conv': 'Yes' if r.converged else 'No'
        })

    comp_df = pd.DataFrame(data)
    print("\n" + comp_df.to_string(index=False))

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
    parser.add_argument('--output', default='hcm_results')
    args = parser.parse_args()

    run_hcm_estimation(args.data, args.output)


if __name__ == '__main__':
    main()
