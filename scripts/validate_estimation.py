"""
Comprehensive Validation: Compare Estimated vs True Parameters
==============================================================

This script validates the DCM simulation by:
1. Loading simulated data with known true parameters
2. Estimating choice model using Biogeme
3. Comparing estimated vs true parameters (with proper scaling)
4. Computing validation metrics (bias, RMSE, t-statistics)
5. Testing statistical significance

Key insight: The simulation uses fee_scale=1,000,000 but estimation uses fee/10k.
So we need to convert: estimated_beta * 100 = true_beta_per_million
"""

import numpy as np
import pandas as pd
import json
import os
import sys
from pathlib import Path
import warnings

# =============================================================================
# PROJECT ROOT SETUP
# =============================================================================
# Determine project root relative to this script's location (scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

# Selective warning suppression - allow important warnings through
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*overflow.*')
warnings.filterwarnings('ignore', message='.*divide by zero.*')

import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta, Variable


def load_config(config_path: str = "config/model_config.json") -> dict:
    """Load the true parameter configuration."""
    with open(config_path, 'r') as f:
        return json.load(f)


def get_true_parameters(config: dict) -> dict:
    """Extract true parameter values from config."""
    true_params = {}

    # ASC for paid alternatives
    for term in config['choice_model']['base_terms']:
        if 'paid1' in term.get('apply_to', []) or 'paid2' in term.get('apply_to', []):
            true_params['ASC_paid'] = term['coef']
            break

    # Attribute coefficients
    for term in config['choice_model']['attribute_terms']:
        name = term['name']
        true_params[f'{name}_base'] = term['base_coef']

        # Interactions
        for interaction in term.get('interactions', []):
            int_name = f"{name}_x_{interaction['with']}"
            true_params[int_name] = interaction['coef']

    # Fee scale for conversion
    true_params['_fee_scale'] = config['choice_model'].get('fee_scale', 10000)

    return true_params


def prepare_data(data_path: str) -> tuple:
    """Load and prepare data for estimation."""
    df = pd.read_csv(data_path)

    # Scale fees (divide by 10k)
    df['fee1_10k'] = df['fee1'] / 10000.0
    df['fee2_10k'] = df['fee2'] / 10000.0
    df['fee3_10k'] = df['fee3'] / 10000.0

    # Center demographics (matching simulation)
    df['age_c'] = (df['age_idx'] - 2) / 2
    df['edu_c'] = (df['edu_idx'] - 3) / 2
    df['inc_c'] = (df['income_indiv_idx'] - 3) / 2
    df['inc_house_c'] = (df['income_house_idx'] - 3) / 2
    df['marital_c'] = (df['marital_idx'] - 0.5) / 0.5

    # Create latent variable proxies from Likert items (factor means)
    lv_items = {
        'pat_blind': ['pat_blind_1', 'pat_blind_2', 'pat_blind_3', 'pat_blind_4'],
        'pat_constructive': ['pat_constructive_1', 'pat_constructive_2', 'pat_constructive_3', 'pat_constructive_4'],
        'sec_dl': ['sec_dl_1', 'sec_dl_2', 'sec_dl_3', 'sec_dl_4'],
        'sec_fp': ['sec_fp_1', 'sec_fp_2', 'sec_fp_3', 'sec_fp_4'],
    }

    for lv_name, items in lv_items.items():
        available = [c for c in items if c in df.columns]
        if available:
            # Standardize: mean=0, sd=1
            proxy = df[available].mean(axis=1)
            df[f'{lv_name}_proxy'] = (proxy - proxy.mean()) / proxy.std()

    # Drop string columns for Biogeme
    string_cols = df.select_dtypes(include=['object']).columns.tolist()
    if string_cols:
        df = df.drop(columns=string_cols)

    print(f"Loaded {len(df):,} observations from {df['ID'].nunique()} respondents")
    print(f"Choice distribution:")
    for c, pct in df['CHOICE'].value_counts(normalize=True).sort_index().items():
        print(f"  Alternative {c}: {pct:.1%}")

    return df, db.Database('dcm_validation', df)


def estimate_basic_mnl(database: db.Database) -> tuple:
    """Estimate basic MNL model."""
    # Variables
    dur1 = Variable('dur1')
    dur2 = Variable('dur2')
    dur3 = Variable('dur3')
    fee1_10k = Variable('fee1_10k')
    fee2_10k = Variable('fee2_10k')
    fee3_10k = Variable('fee3_10k')
    CHOICE = Variable('CHOICE')

    # Parameters
    ASC_paid = Beta('ASC_paid', 1.0, None, None, 0)
    B_FEE = Beta('B_FEE', -0.5, None, 0, 0)
    B_DUR = Beta('B_DUR', -0.05, None, 0, 0)

    # Utilities
    V1 = ASC_paid + B_FEE * fee1_10k + B_DUR * dur1
    V2 = ASC_paid + B_FEE * fee2_10k + B_DUR * dur2
    V3 = B_FEE * fee3_10k + B_DUR * dur3

    V = {1: V1, 2: V2, 3: V3}
    av = {1: 1, 2: 1, 3: 1}

    logprob = models.loglogit(V, av, CHOICE)

    biogeme_obj = bio.BIOGEME(database, logprob)
    biogeme_obj.modelName = "basic_mnl"

    print("\n" + "="*60)
    print("Estimating Basic MNL Model...")
    print("="*60)

    results = biogeme_obj.estimate()

    return results, ['ASC_paid', 'B_FEE', 'B_DUR']


def estimate_mnl_with_demographics(database: db.Database) -> tuple:
    """Estimate MNL with demographic interactions."""
    # Variables
    dur1 = Variable('dur1')
    dur2 = Variable('dur2')
    dur3 = Variable('dur3')
    fee1_10k = Variable('fee1_10k')
    fee2_10k = Variable('fee2_10k')
    fee3_10k = Variable('fee3_10k')
    CHOICE = Variable('CHOICE')

    # Centered demographics
    age_c = Variable('age_c')
    edu_c = Variable('edu_c')
    inc_c = Variable('inc_c')
    inc_house_c = Variable('inc_house_c')
    marital_c = Variable('marital_c')

    # Parameters
    ASC_paid = Beta('ASC_paid', 1.0, None, None, 0)

    # Fee coefficients
    B_FEE = Beta('B_FEE', -0.5, None, 0, 0)
    B_FEE_AGE = Beta('B_FEE_AGE', 0.05, None, None, 0)
    B_FEE_EDU = Beta('B_FEE_EDU', 0.05, None, None, 0)
    B_FEE_INC = Beta('B_FEE_INC', 0.10, None, None, 0)
    B_FEE_INC_H = Beta('B_FEE_INC_H', 0.05, None, None, 0)

    # Duration coefficients
    B_DUR = Beta('B_DUR', -0.05, None, 0, 0)
    B_DUR_EDU = Beta('B_DUR_EDU', -0.02, None, None, 0)
    B_DUR_INC = Beta('B_DUR_INC', -0.02, None, None, 0)
    B_DUR_INC_H = Beta('B_DUR_INC_H', -0.01, None, None, 0)
    B_DUR_MARITAL = Beta('B_DUR_MARITAL', -0.02, None, None, 0)

    # Individual-specific coefficients
    B_FEE_i = B_FEE + B_FEE_AGE*age_c + B_FEE_EDU*edu_c + B_FEE_INC*inc_c + B_FEE_INC_H*inc_house_c
    B_DUR_i = B_DUR + B_DUR_EDU*edu_c + B_DUR_INC*inc_c + B_DUR_INC_H*inc_house_c + B_DUR_MARITAL*marital_c

    # Utilities
    V1 = ASC_paid + B_FEE_i * fee1_10k + B_DUR_i * dur1
    V2 = ASC_paid + B_FEE_i * fee2_10k + B_DUR_i * dur2
    V3 = B_FEE_i * fee3_10k + B_DUR_i * dur3

    V = {1: V1, 2: V2, 3: V3}
    av = {1: 1, 2: 1, 3: 1}

    logprob = models.loglogit(V, av, CHOICE)

    biogeme_obj = bio.BIOGEME(database, logprob)
    biogeme_obj.modelName = "mnl_demographics"

    print("\n" + "="*60)
    print("Estimating MNL with Demographic Interactions...")
    print("="*60)

    results = biogeme_obj.estimate()

    param_names = ['ASC_paid', 'B_FEE', 'B_FEE_AGE', 'B_FEE_EDU', 'B_FEE_INC', 'B_FEE_INC_H',
                   'B_DUR', 'B_DUR_EDU', 'B_DUR_INC', 'B_DUR_INC_H', 'B_DUR_MARITAL']

    return results, param_names


def compare_parameters(results, true_params: dict, model_name: str) -> pd.DataFrame:
    """Compare estimated vs true parameters with proper scaling."""

    # Mapping from estimation names to true config names
    param_mapping = {
        'ASC_paid': 'ASC_paid',
        'B_FEE': 'b_fee_scaled_base',
        'B_FEE_AGE': 'b_fee_scaled_x_age_idx',
        'B_FEE_EDU': 'b_fee_scaled_x_edu_idx',
        'B_FEE_INC': 'b_fee_scaled_x_income_indiv_idx',
        'B_FEE_INC_H': 'b_fee_scaled_x_income_house_idx',
        'B_FEE_PAT_B': 'b_fee_scaled_x_pat_blind',
        'B_FEE_PAT_C': 'b_fee_scaled_x_pat_constructive',
        'B_FEE_SEC_D': 'b_fee_scaled_x_sec_dl',
        'B_FEE_SEC_F': 'b_fee_scaled_x_sec_fp',
        'B_DUR': 'b_dur_base',
        'B_DUR_EDU': 'b_dur_x_edu_idx',
        'B_DUR_INC': 'b_dur_x_income_indiv_idx',
        'B_DUR_INC_H': 'b_dur_x_income_house_idx',
        'B_DUR_MARITAL': 'b_dur_x_marital_idx',
        'B_DUR_PAT_B': 'b_dur_x_pat_blind',
        'B_DUR_PAT_C': 'b_dur_x_pat_constructive',
        'B_DUR_SEC_D': 'b_dur_x_sec_dl',
        'B_DUR_SEC_F': 'b_dur_x_sec_fp',
    }

    betas = results.get_beta_values()

    comparison = []

    # Get standard errors using correct method
    std_errs = {}
    for param in betas.keys():
        try:
            std_errs[param] = results.get_parameter_std_err(param)
        except:
            std_errs[param] = np.nan

    for est_name, est_val in betas.items():
        robust_se = std_errs.get(est_name, np.nan)

        t_stat = est_val / robust_se if robust_se and robust_se > 0 else np.nan

        # Get true value
        config_name = param_mapping.get(est_name)
        true_val = true_params.get(config_name, np.nan)

        # Calculate bias
        if not np.isnan(true_val):
            bias = est_val - true_val
            bias_pct = (bias / abs(true_val) * 100) if true_val != 0 else np.nan

            # 95% CI coverage
            ci_lower = est_val - 1.96 * robust_se
            ci_upper = est_val + 1.96 * robust_se
            covered = ci_lower <= true_val <= ci_upper
        else:
            bias = bias_pct = np.nan
            ci_lower = ci_upper = np.nan
            covered = None

        comparison.append({
            'Parameter': est_name,
            'True': true_val,
            'Estimated': est_val,
            'Std.Error': robust_se,
            't-stat': t_stat,
            'Signif': '*' if abs(t_stat) > 1.96 else '' if not np.isnan(t_stat) else '-',
            'Bias': bias,
            'Bias%': bias_pct,
            '95% CI': f"[{ci_lower:.4f}, {ci_upper:.4f}]" if not np.isnan(ci_lower) else '-',
            'Covered': covered,
        })

    return pd.DataFrame(comparison)


def print_validation_report(comparison_df: pd.DataFrame, results, model_name: str):
    """Print comprehensive validation report."""

    print("\n" + "="*80)
    print(f"VALIDATION REPORT: {model_name}")
    print("="*80)

    # Model fit statistics
    stats = results.get_general_statistics()

    # Handle different format possibilities
    try:
        ll_val = stats['Final log likelihood']
        ll = float(ll_val[0]) if isinstance(ll_val, (list, tuple)) else float(ll_val)

        n_params_val = stats['Number of estimated parameters']
        n_params = int(n_params_val[0]) if isinstance(n_params_val, (list, tuple)) else int(n_params_val)

        n_obs_val = stats['Sample size']
        n_obs = int(n_obs_val[0]) if isinstance(n_obs_val, (list, tuple)) else int(n_obs_val)
    except (ValueError, TypeError, KeyError) as e:
        print(f"\n  ERROR: Could not parse statistics: {e}")
        print(f"  Stats: {stats}")
        return

    # Null log-likelihood for 3 alternatives
    null_ll = float(n_obs) * np.log(1/3)
    rho2 = 1 - (ll / null_ll)
    adj_rho2 = 1 - ((ll - n_params) / null_ll)

    print(f"\nModel Fit:")
    print(f"  Log-likelihood:     {ll:,.4f}")
    print(f"  Null log-likelihood: {null_ll:,.4f}")
    print(f"  Parameters:         {n_params}")
    print(f"  Observations:       {n_obs:,}")
    print(f"  Rho-squared:        {rho2:.4f}")
    print(f"  Adj. Rho-squared:   {adj_rho2:.4f}")

    print("\n" + "-"*80)
    print("PARAMETER COMPARISON: Estimated vs True")
    print("-"*80)
    print(f"{'Parameter':<15} {'True':>10} {'Estimated':>10} {'SE':>10} {'t-stat':>8} {'Bias%':>8} {'Covered':>8}")
    print("-"*80)

    for _, row in comparison_df.iterrows():
        true_str = f"{row['True']:.4f}" if not pd.isna(row['True']) else "N/A"
        est_str = f"{row['Estimated']:.4f}"
        se_str = f"{row['Std.Error']:.4f}" if not pd.isna(row['Std.Error']) else "N/A"
        t_str = f"{row['t-stat']:.2f}{row['Signif']}" if not pd.isna(row['t-stat']) else "N/A"
        bias_str = f"{row['Bias%']:+.1f}%" if not pd.isna(row['Bias%']) else "N/A"
        cov_str = "Yes" if row['Covered'] == True else "No" if row['Covered'] == False else "N/A"

        print(f"{row['Parameter']:<15} {true_str:>10} {est_str:>10} {se_str:>10} {t_str:>8} {bias_str:>8} {cov_str:>8}")

    # Summary statistics
    valid_rows = comparison_df[comparison_df['True'].notna()]
    if len(valid_rows) > 0:
        print("\n" + "-"*80)
        print("VALIDATION SUMMARY")
        print("-"*80)
        print(f"  Parameters with known true values: {len(valid_rows)}")
        print(f"  Mean absolute bias: {valid_rows['Bias'].abs().mean():.4f}")
        print(f"  Mean |Bias%|: {valid_rows['Bias%'].abs().mean():.1f}%")
        coverage = valid_rows['Covered'].mean() if valid_rows['Covered'].notna().any() else np.nan
        print(f"  95% CI coverage rate: {coverage*100:.1f}% (should be ~95%)")

        # Check convergence quality
        all_significant = (valid_rows['t-stat'].abs() > 1.96).all()
        low_bias = (valid_rows['Bias%'].abs() < 20).all()
        good_coverage = coverage > 0.8 if not np.isnan(coverage) else False

        print("\n  QUALITY CHECKS:")
        print(f"    All key params significant: {'PASS' if all_significant else 'FAIL'}")
        print(f"    All biases < 20%:           {'PASS' if low_bias else 'FAIL'}")
        print(f"    Coverage > 80%:             {'PASS' if good_coverage else 'FAIL'}")


def estimate_mnl_with_latent_proxies(database: db.Database) -> tuple:
    """Estimate MNL with demographic AND latent variable proxy interactions."""
    # Variables
    dur1 = Variable('dur1')
    dur2 = Variable('dur2')
    dur3 = Variable('dur3')
    fee1_10k = Variable('fee1_10k')
    fee2_10k = Variable('fee2_10k')
    fee3_10k = Variable('fee3_10k')
    CHOICE = Variable('CHOICE')

    # Centered demographics
    age_c = Variable('age_c')
    edu_c = Variable('edu_c')
    inc_c = Variable('inc_c')
    inc_house_c = Variable('inc_house_c')
    marital_c = Variable('marital_c')

    # Latent variable proxies (standardized factor means)
    pat_blind = Variable('pat_blind_proxy')
    pat_const = Variable('pat_constructive_proxy')
    sec_dl = Variable('sec_dl_proxy')
    sec_fp = Variable('sec_fp_proxy')

    # Parameters
    ASC_paid = Beta('ASC_paid', 1.0, None, None, 0)

    # Fee coefficients - base + demographic interactions + latent interactions
    B_FEE = Beta('B_FEE', -0.5, None, 0, 0)
    B_FEE_AGE = Beta('B_FEE_AGE', 0.05, None, None, 0)
    B_FEE_EDU = Beta('B_FEE_EDU', 0.05, None, None, 0)
    B_FEE_INC = Beta('B_FEE_INC', 0.10, None, None, 0)
    B_FEE_INC_H = Beta('B_FEE_INC_H', 0.05, None, None, 0)
    B_FEE_PAT_B = Beta('B_FEE_PAT_B', -0.05, None, None, 0)
    B_FEE_PAT_C = Beta('B_FEE_PAT_C', -0.02, None, None, 0)
    B_FEE_SEC_D = Beta('B_FEE_SEC_D', 0.02, None, None, 0)
    B_FEE_SEC_F = Beta('B_FEE_SEC_F', 0.02, None, None, 0)

    # Duration coefficients
    B_DUR = Beta('B_DUR', -0.05, None, 0, 0)
    B_DUR_EDU = Beta('B_DUR_EDU', -0.02, None, None, 0)
    B_DUR_INC = Beta('B_DUR_INC', -0.02, None, None, 0)
    B_DUR_INC_H = Beta('B_DUR_INC_H', -0.01, None, None, 0)
    B_DUR_MARITAL = Beta('B_DUR_MARITAL', -0.02, None, None, 0)
    B_DUR_PAT_B = Beta('B_DUR_PAT_B', 0.02, None, None, 0)
    B_DUR_PAT_C = Beta('B_DUR_PAT_C', 0.02, None, None, 0)
    B_DUR_SEC_D = Beta('B_DUR_SEC_D', -0.02, None, None, 0)
    B_DUR_SEC_F = Beta('B_DUR_SEC_F', -0.02, None, None, 0)

    # Individual-specific coefficients (full model)
    B_FEE_i = (B_FEE +
               B_FEE_AGE*age_c + B_FEE_EDU*edu_c + B_FEE_INC*inc_c + B_FEE_INC_H*inc_house_c +
               B_FEE_PAT_B*pat_blind + B_FEE_PAT_C*pat_const + B_FEE_SEC_D*sec_dl + B_FEE_SEC_F*sec_fp)

    B_DUR_i = (B_DUR +
               B_DUR_EDU*edu_c + B_DUR_INC*inc_c + B_DUR_INC_H*inc_house_c + B_DUR_MARITAL*marital_c +
               B_DUR_PAT_B*pat_blind + B_DUR_PAT_C*pat_const + B_DUR_SEC_D*sec_dl + B_DUR_SEC_F*sec_fp)

    # Utilities
    V1 = ASC_paid + B_FEE_i * fee1_10k + B_DUR_i * dur1
    V2 = ASC_paid + B_FEE_i * fee2_10k + B_DUR_i * dur2
    V3 = B_FEE_i * fee3_10k + B_DUR_i * dur3

    V = {1: V1, 2: V2, 3: V3}
    av = {1: 1, 2: 1, 3: 1}

    logprob = models.loglogit(V, av, CHOICE)

    biogeme_obj = bio.BIOGEME(database, logprob)
    biogeme_obj.modelName = "mnl_full"

    print("\n" + "="*60)
    print("Estimating MNL with Demographics + Latent Proxies...")
    print("="*60)

    results = biogeme_obj.estimate()

    param_names = ['ASC_paid',
                   'B_FEE', 'B_FEE_AGE', 'B_FEE_EDU', 'B_FEE_INC', 'B_FEE_INC_H',
                   'B_FEE_PAT_B', 'B_FEE_PAT_C', 'B_FEE_SEC_D', 'B_FEE_SEC_F',
                   'B_DUR', 'B_DUR_EDU', 'B_DUR_INC', 'B_DUR_INC_H', 'B_DUR_MARITAL',
                   'B_DUR_PAT_B', 'B_DUR_PAT_C', 'B_DUR_SEC_D', 'B_DUR_SEC_F']

    return results, param_names


def run_validation(
    data_path: str = "data/test_validation.csv",
    config_path: str = "config/model_config.json"
):
    """Run complete validation pipeline."""

    print("="*80)
    print("DCM SIMULATION VALIDATION")
    print("Verifying Estimation Recovers True Parameters")
    print("="*80)

    # Load true parameters
    config = load_config(config_path)
    true_params = get_true_parameters(config)

    print("\nTrue Parameter Values (from model_config.json):")
    for k, v in true_params.items():
        if not k.startswith('_'):
            print(f"  {k}: {v}")

    # Load and prepare data
    df, database = prepare_data(data_path)

    # Model 1: Basic MNL
    results1, params1 = estimate_basic_mnl(database)
    comp1 = compare_parameters(results1, true_params, "Basic MNL")
    print_validation_report(comp1, results1, "Model 1: Basic MNL")

    # Model 2: MNL with Demographics only
    results2, params2 = estimate_mnl_with_demographics(database)
    comp2 = compare_parameters(results2, true_params, "MNL with Demographics")
    print_validation_report(comp2, results2, "Model 2: MNL + Demographics")

    # Model 3: MNL with Demographics + Latent Proxies (Full Model)
    results3, params3 = estimate_mnl_with_latent_proxies(database)
    comp3 = compare_parameters(results3, true_params, "MNL Full")
    print_validation_report(comp3, results3, "Model 3: MNL + Demo + Latent Proxies")

    # Save results
    output_dir = Path("results/validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    comp1.to_csv(output_dir / "basic_mnl_comparison.csv", index=False)
    comp2.to_csv(output_dir / "demo_mnl_comparison.csv", index=False)
    comp3.to_csv(output_dir / "full_mnl_comparison.csv", index=False)

    print("\n" + "="*80)
    print("OVERALL VALIDATION CONCLUSION")
    print("="*80)

    # Check if key parameters are recovered
    key_params_1 = ['ASC_paid', 'B_FEE', 'B_DUR']
    bias_1 = comp1[comp1['Parameter'].isin(key_params_1)]['Bias%'].abs().mean()
    bias_3 = comp3[comp3['Parameter'].isin(key_params_1)]['Bias%'].abs().mean()

    print(f"\nBasic MNL - Mean |Bias%| for key params: {bias_1:.1f}%")
    print(f"Full MNL  - Mean |Bias%| for key params: {bias_3:.1f}%")

    if bias_1 < 20 and bias_3 < 20:
        print("\nRESULT: Estimation successfully recovers true parameters")
        print("The simulation and estimation pipeline is working correctly.")
        print("\nNote: Some demographic/latent interaction biases are expected due to:")
        print("  1. Measurement error in latent proxies (vs true latent variables)")
        print("  2. Correlation between demographics and latent variables")
    else:
        print("WARNING: Large bias detected - investigation needed")

    return results1, results2, results3, comp1, comp2, comp3


if __name__ == '__main__':
    run_validation()
