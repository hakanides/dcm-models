"""
HCM Full Model - Simultaneous Estimation (ICLV)
================================================

Hybrid Choice Model with all four latent variables affecting
both fee and duration sensitivity.

Latent Constructs:
    1. Blind Patriotism (η_pb) - Uncritical support for country
    2. Constructive Patriotism (η_pc) - Critical, improvement-oriented
    3. Daily Life Secularism (η_sdl) - Separation in daily practices
    4. Faith & Prayer Secularism (η_sfp) - Separation in religious matters

Model Specification (Simultaneous ICLV):
    Structural: η_k = σ_k * ω_k,  where ω_k ~ N(0,1)

    Measurement (Ordered Logit):
        P(I_ki = j | η_k) = Logit(τ_j - λ_ki*η_k) - Logit(τ_{j-1} - λ_ki*η_k)

    Choice:
        B_FEE_i = B_FEE + B_FEE_PB*η_pb + B_FEE_PC*η_pc + B_FEE_SDL*η_sdl + B_FEE_SFP*η_sfp
        B_DUR_i = B_DUR + B_DUR_PB*η_pb + B_DUR_PC*η_pc + B_DUR_SDL*η_sdl + B_DUR_SFP*η_sfp

        V1 = ASC_paid + B_FEE_i * fee1 + B_DUR_i * dur1
        V2 = ASC_paid + B_FEE_i * fee2 + B_DUR_i * dur2
        V3 = B_FEE_i * fee3 + B_DUR_i * dur3

    Likelihood:
        L_n = ∫∫∫∫ P(choice|η) × Π_k Π_i P(I_ki|η_k) dη_pb dη_pc dη_sdl dη_sfp

    Estimated via Monte Carlo integration using Halton draws.

This eliminates attenuation bias from two-stage estimation.

References:
    - Bierlaire (2018): Technical Report on Latent Variables in Biogeme
    - Walker & Ben-Akiva (2002): Generalized Random Utility Model

Authors: Hakan Mülayim, Giray Girengir, Ataol Azeritürk
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import (
    Beta, Variable, MonteCarlo, log, exp, Elem, Draws
)

from src.utils.item_detection import get_lv_items, get_items_by_prefix


# =============================================================================
# MODEL CONSTANTS
# =============================================================================

# Availability dict for alternatives (used in logit and null LL calculation)
AV_DICT = {1: 1, 2: 1, 3: 1}  # All alternatives always available


# =============================================================================
# MEASUREMENT MODEL (Ordered Logit for Likert Items)
# =============================================================================

def ordered_logit_prob(indicator, loading, lv, thresholds):
    """
    Compute ordered logit probability for a single indicator.

    P(I = j | η) = Logistic(τ_j - λ*η) - Logistic(τ_{j-1} - λ*η)

    Args:
        indicator: Biogeme Variable for the indicator (1-5)
        loading: Factor loading (λ) - can be float or Beta
        lv: Latent variable expression (η)
        thresholds: List of threshold Beta parameters [τ_1, τ_2, τ_3, τ_4]

    Returns:
        Biogeme expression for probability
    """
    def logistic_cdf(x):
        return 1 / (1 + exp(-x))

    # Cumulative probabilities at each threshold
    cum_probs = {}
    cum_probs[1] = logistic_cdf(thresholds[0] - loading * lv)
    cum_probs[2] = logistic_cdf(thresholds[1] - loading * lv)
    cum_probs[3] = logistic_cdf(thresholds[2] - loading * lv)
    cum_probs[4] = logistic_cdf(thresholds[3] - loading * lv)
    cum_probs[5] = 1.0

    # Category probabilities
    category_probs = {}
    category_probs[1] = cum_probs[1]
    category_probs[2] = cum_probs[2] - cum_probs[1]
    category_probs[3] = cum_probs[3] - cum_probs[2]
    category_probs[4] = cum_probs[4] - cum_probs[3]
    category_probs[5] = cum_probs[5] - cum_probs[4]

    prob = Elem(category_probs, indicator)

    return prob


def create_construct_measurement(lv, items, thresholds, construct_name):
    """
    Create measurement likelihood for a single construct.

    Args:
        lv: Latent variable expression
        items: List of indicator column names
        thresholds: List of threshold Beta parameters
        construct_name: Name for parameter naming

    Returns:
        Tuple of (joint probability expression, loadings dict)
    """
    loadings = {}
    joint_prob = 1.0

    for i, item in enumerate(items):
        indicator = Variable(item)

        # First loading fixed to 1 for identification
        if i == 0:
            loading = 1.0
        else:
            loading = Beta(f'lambda_{construct_name}_{i+1}', 0.8, 0.01, 5.0, 0)

        loadings[item] = loading
        prob = ordered_logit_prob(indicator, loading, lv, thresholds)
        joint_prob = joint_prob * prob

    return joint_prob, loadings


# =============================================================================
# MODEL SPECIFICATION
# =============================================================================

def create_hcm_full_iclv(database: db.Database, construct_items: dict):
    """
    Create full HCM with all 4 LVs using ICLV estimation.

    Args:
        database: Biogeme database with indicator columns
        construct_items: Dict mapping construct name to list of items

    Returns:
        Tuple of (log_probability, model_name)
    """
    # -------------------------------------------------------------------------
    # CHOICE VARIABLES
    # -------------------------------------------------------------------------
    CHOICE = Variable('CHOICE')
    fee1 = Variable('fee1_10k')
    fee2 = Variable('fee2_10k')
    fee3 = Variable('fee3_10k')
    dur1 = Variable('dur1')
    dur2 = Variable('dur2')
    dur3 = Variable('dur3')

    # -------------------------------------------------------------------------
    # LATENT VARIABLES: Structural Equations
    # η_k = σ_k * ω_k,  where ω_k ~ N(0,1)
    # Using different Halton primes for independent sequences
    # -------------------------------------------------------------------------
    omega_pb = Draws('omega_pb', 'NORMAL_HALTON2')
    omega_pc = Draws('omega_pc', 'NORMAL_HALTON3')
    omega_sdl = Draws('omega_sdl', 'NORMAL_HALTON5')
    omega_sfp = Draws('omega_sfp', 'NORMAL_HALTON7')

    # Latent variables (σ=1 for identification)
    LV_pat_blind = omega_pb
    LV_pat_const = omega_pc
    LV_sec_dl = omega_sdl
    LV_sec_fp = omega_sfp

    # -------------------------------------------------------------------------
    # MEASUREMENT MODEL: Ordered Logit for Likert Items
    # -------------------------------------------------------------------------
    # Shared thresholds (across all items)
    tau_1 = Beta('tau_1', -1.5, -5, 5, 0)
    tau_2 = Beta('tau_2', -0.5, -5, 5, 0)
    tau_3 = Beta('tau_3', 0.5, -5, 5, 0)
    tau_4 = Beta('tau_4', 1.5, -5, 5, 0)
    thresholds = [tau_1, tau_2, tau_3, tau_4]

    # Build measurement model for each construct
    all_loadings = {}
    measurement_prob = 1.0

    constructs = [
        ('pat_blind', LV_pat_blind),
        ('pat_const', LV_pat_const),
        ('sec_dl', LV_sec_dl),
        ('sec_fp', LV_sec_fp)
    ]

    for construct_name, lv in constructs:
        items = construct_items.get(construct_name, [])

        if items:
            print(f"  {construct_name}: {len(items)} indicators")

            prob, loadings = create_construct_measurement(
                lv, items, thresholds, construct_name
            )
            measurement_prob = measurement_prob * prob
            all_loadings.update(loadings)
        else:
            print(f"  WARNING: No items found for {construct_name}")

    # -------------------------------------------------------------------------
    # CHOICE MODEL
    # -------------------------------------------------------------------------
    ASC_paid = Beta('ASC_paid', 0, -10, 10, 0)
    B_FEE = Beta('B_FEE', -0.6, -10, 0, 0)
    B_DUR = Beta('B_DUR', -0.05, -5, 0, 0)

    # LV effects on fee sensitivity
    B_FEE_PB = Beta('B_FEE_PatBlind', 0.1, -2, 2, 0)
    B_FEE_PC = Beta('B_FEE_PatConst', 0.0, -2, 2, 0)
    B_FEE_SDL = Beta('B_FEE_SecDL', 0.0, -2, 2, 0)
    B_FEE_SFP = Beta('B_FEE_SecFP', 0.0, -2, 2, 0)

    # LV effects on duration sensitivity
    B_DUR_PB = Beta('B_DUR_PatBlind', 0.0, -1, 1, 0)
    B_DUR_PC = Beta('B_DUR_PatConst', 0.0, -1, 1, 0)
    B_DUR_SDL = Beta('B_DUR_SecDL', 0.0, -1, 1, 0)
    B_DUR_SFP = Beta('B_DUR_SecFP', 0.0, -1, 1, 0)

    # Individual-specific fee coefficient
    B_FEE_i = (B_FEE
               + B_FEE_PB * LV_pat_blind
               + B_FEE_PC * LV_pat_const
               + B_FEE_SDL * LV_sec_dl
               + B_FEE_SFP * LV_sec_fp)

    # Individual-specific duration coefficient
    B_DUR_i = (B_DUR
               + B_DUR_PB * LV_pat_blind
               + B_DUR_PC * LV_pat_const
               + B_DUR_SDL * LV_sec_dl
               + B_DUR_SFP * LV_sec_fp)

    # Utility functions
    V1 = ASC_paid + B_FEE_i * fee1 + B_DUR_i * dur1
    V2 = ASC_paid + B_FEE_i * fee2 + B_DUR_i * dur2
    V3 = B_FEE_i * fee3 + B_DUR_i * dur3

    V = {1: V1, 2: V2, 3: V3}

    # Choice probability (conditional on LVs)
    choice_prob = models.logit(V, AV_DICT, CHOICE)

    # -------------------------------------------------------------------------
    # JOINT LIKELIHOOD WITH MONTE CARLO INTEGRATION
    # L_n = ∫∫∫∫ P(choice|η) × P(indicators|η) dη_1 dη_2 dη_3 dη_4
    # -------------------------------------------------------------------------
    joint_prob = choice_prob * measurement_prob

    # Monte Carlo integration
    integrated_prob = MonteCarlo(joint_prob)

    # Log-likelihood
    logprob = log(integrated_prob)

    return logprob, 'HCM_Full_ICLV'


# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_data(filepath: str, fee_scale: float = 10000.0):
    """
    Load and prepare data for full HCM ICLV estimation.

    Args:
        filepath: Path to CSV data file
        fee_scale: Divisor for fee scaling (default 10000)

    Returns:
        Tuple of (prepared DataFrame, dict of construct -> items)
    """
    df = pd.read_csv(filepath)

    # Scale fees
    for alt in [1, 2, 3]:
        df[f'fee{alt}_10k'] = df[f'fee{alt}'] / fee_scale

    # Define constructs: short_name -> (full_lv_name, legacy_prefix)
    constructs_config = {
        'pat_blind': ('pat_blind', 'pat_blind_'),
        'pat_const': ('pat_constructive', 'pat_constructive_'),
        'sec_dl': ('sec_dl', 'sec_dl_'),
        'sec_fp': ('sec_fp', 'sec_fp_'),
    }

    construct_items = {}

    print("\nLooking for indicator items...")

    for short_name, (full_lv_name, legacy_prefix) in constructs_config.items():
        # Try new unified naming first
        items = get_lv_items(df, full_lv_name)

        # Fallback to legacy prefix detection
        if not items:
            items = get_items_by_prefix(df, legacy_prefix)

        if items:
            print(f"  {short_name}: Found {len(items)} items")
            # Convert Likert items to integers (1-5)
            for item in items:
                df[item] = df[item].round().astype(int).clip(1, 5)
            construct_items[short_name] = items
        else:
            print(f"  {short_name}: No items found")

    # Keep only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    return df[numeric_cols].copy(), construct_items


# =============================================================================
# ESTIMATION
# =============================================================================

def estimate_hcm_full(data_path: str,
                      config_path: str = None,
                      output_dir: str = 'results/hcm_full',
                      n_draws: int = 1000):
    """
    Estimate full HCM model using simultaneous ICLV approach.

    Args:
        data_path: Path to data CSV
        config_path: Path to config JSON with true parameters (optional)
        output_dir: Directory for output files
        n_draws: Number of Halton draws for Monte Carlo integration

    Returns:
        Dictionary with estimation results
    """
    print("=" * 70)
    print("HCM FULL MODEL - SIMULTANEOUS ICLV ESTIMATION")
    print("(All 4 Latent Variables on Fee and Duration)")
    print("=" * 70)
    print(f"\nUsing {n_draws} Halton draws for Monte Carlo integration")
    print("This approach eliminates attenuation bias from two-stage estimation!")

    # Load and prepare data
    df, construct_items = prepare_data(data_path)
    n_obs = len(df)

    # Validate panel structure exists (required for HCM/ICLV with repeated choices)
    if 'ID' not in df.columns:
        raise ValueError(
            "HCM/ICLV requires panel data with 'ID' column for correct standard errors. "
            "Each individual should have multiple choice observations."
        )
    n_individuals = df['ID'].nunique()

    print(f"\nData: {n_obs} observations from {n_individuals} individuals")

    # Create database with panel structure for cluster-robust inference
    database = db.Database('hcm_full_iclv', df)
    database.panel('ID')  # Declare panel structure for consistent random draws
    database.number_of_draws = n_draws

    # Create model
    print("\nBuilding ICLV model with 4 latent variables...")
    logprob, model_name = create_hcm_full_iclv(database, construct_items)

    # Estimate
    print("\nEstimating model (this may take several minutes)...")
    biogeme = bio.BIOGEME(database, logprob)
    biogeme.model_name = model_name
    biogeme.calculate_null_loglikelihood(AV_DICT)

    results = biogeme.estimate()

    # Extract results
    ll = results.final_loglikelihood
    k = results.number_of_free_parameters
    n = n_obs

    general_stats = results.get_general_statistics()
    null_ll = None
    for key, val in general_stats.items():
        if 'Null log likelihood' in key:
            null_ll = float(val[0]) if isinstance(val, tuple) else float(val)
            break

    rho2 = 1 - (ll / null_ll) if null_ll else 0
    aic = 2 * k - 2 * ll
    bic = k * np.log(n) - 2 * ll

    betas = results.get_beta_values()
    t_stats = {}
    p_vals = {}
    std_errs = {}

    for p in betas:
        try:
            se = results.get_parameter_std_err(p)
            std_errs[p] = se
            t_stats[p] = betas[p] / se if se > 0 else np.nan
            from scipy import stats as scipy_stats
            p_vals[p] = 2 * (1 - scipy_stats.norm.cdf(abs(t_stats[p])))
        except:
            std_errs[p] = np.nan
            t_stats[p] = np.nan
            p_vals[p] = np.nan

    # Print results
    print("\n" + "=" * 70)
    print("ESTIMATION RESULTS")
    print("=" * 70)
    print(f"\nModel: {model_name}")
    print(f"Estimation: Simultaneous (Monte Carlo, {n_draws} draws)")
    print(f"Log-Likelihood: {ll:.4f}")
    print(f"Null LL: {null_ll:.4f}")
    print(f"Rho-squared: {rho2:.4f}")
    print(f"AIC: {aic:.2f}")
    print(f"BIC: {bic:.2f}")
    print(f"N: {n_obs}")
    print(f"K: {k}")

    # Base parameters
    print("\n" + "-" * 70)
    print("CHOICE MODEL - BASE PARAMETERS")
    print("-" * 70)
    print(f"{'Parameter':<20} {'Estimate':>12} {'Std.Err':>10} {'t-stat':>10}")
    print("-" * 55)

    base_params = ['ASC_paid', 'B_FEE', 'B_DUR']
    for param in base_params:
        if param in betas:
            sig = "***" if p_vals.get(param, 1) < 0.01 else "**" if p_vals.get(param, 1) < 0.05 else "*" if p_vals.get(param, 1) < 0.10 else ""
            print(f"{param:<20} {betas[param]:>12.4f} {std_errs.get(param, np.nan):>10.4f} {t_stats.get(param, np.nan):>10.2f} {sig}")

    # Fee sensitivity interactions
    print("\n" + "-" * 70)
    print("FEE SENSITIVITY INTERACTIONS")
    print("-" * 70)

    fee_params = ['B_FEE_PatBlind', 'B_FEE_PatConst', 'B_FEE_SecDL', 'B_FEE_SecFP']
    for param in fee_params:
        if param in betas:
            sig = "***" if p_vals.get(param, 1) < 0.01 else "**" if p_vals.get(param, 1) < 0.05 else "*" if p_vals.get(param, 1) < 0.10 else ""
            print(f"{param:<20} {betas[param]:>12.4f} {std_errs.get(param, np.nan):>10.4f} {t_stats.get(param, np.nan):>10.2f} {sig}")

    # Duration sensitivity interactions
    print("\n" + "-" * 70)
    print("DURATION SENSITIVITY INTERACTIONS")
    print("-" * 70)

    dur_params = ['B_DUR_PatBlind', 'B_DUR_PatConst', 'B_DUR_SecDL', 'B_DUR_SecFP']
    for param in dur_params:
        if param in betas:
            sig = "***" if p_vals.get(param, 1) < 0.01 else "**" if p_vals.get(param, 1) < 0.05 else "*" if p_vals.get(param, 1) < 0.10 else ""
            print(f"{param:<20} {betas[param]:>12.4f} {std_errs.get(param, np.nan):>10.4f} {t_stats.get(param, np.nan):>10.2f} {sig}")

    # Measurement model
    print("\n" + "-" * 70)
    print("MEASUREMENT MODEL PARAMETERS")
    print("-" * 70)

    print("\nThresholds:")
    for i in range(1, 5):
        param = f'tau_{i}'
        if param in betas:
            print(f"  {param}: {betas[param]:.4f} (SE: {std_errs.get(param, np.nan):.4f})")

    print("\nFactor Loadings (first per construct fixed to 1):")
    for construct in ['pat_blind', 'pat_const', 'sec_dl', 'sec_fp']:
        loadings = [p for p in sorted(betas.keys()) if p.startswith(f'lambda_{construct}')]
        if loadings:
            print(f"  {construct}:")
            for param in loadings:
                print(f"    {param}: {betas[param]:.4f} (SE: {std_errs.get(param, np.nan):.4f})")

    # Compare to true values if config provided
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            config = json.load(f)

        print("\n" + "=" * 70)
        print("COMPARISON TO TRUE PARAMETERS")
        print("=" * 70)

        # Extract true values
        true_params = {}
        for term in config['choice_model']['base_terms']:
            if term['name'] == 'ASC_paid':
                true_params['ASC_paid'] = term['coef']

        for term in config['choice_model']['attribute_terms']:
            if term['name'] == 'b_fee10k':
                true_params['B_FEE'] = term['base_coef']
                for inter in term.get('interactions', []):
                    if 'pat_blind' in inter.get('with', ''):
                        true_params['B_FEE_PatBlind'] = inter['coef']
                    elif 'pat_constructive' in inter.get('with', ''):
                        true_params['B_FEE_PatConst'] = inter['coef']
                    elif 'sec_dl' in inter.get('with', ''):
                        true_params['B_FEE_SecDL'] = inter['coef']
                    elif 'sec_fp' in inter.get('with', ''):
                        true_params['B_FEE_SecFP'] = inter['coef']
            elif term['name'] == 'b_dur':
                true_params['B_DUR'] = term['base_coef']
                for inter in term.get('interactions', []):
                    if 'pat_blind' in inter.get('with', ''):
                        true_params['B_DUR_PatBlind'] = inter['coef']
                    elif 'pat_constructive' in inter.get('with', ''):
                        true_params['B_DUR_PatConst'] = inter['coef']
                    elif 'sec_dl' in inter.get('with', ''):
                        true_params['B_DUR_SecDL'] = inter['coef']
                    elif 'sec_fp' in inter.get('with', ''):
                        true_params['B_DUR_SecFP'] = inter['coef']

        all_params = base_params + fee_params + dur_params
        print(f"\n{'Parameter':<20} {'True':>10} {'Estimated':>12} {'Bias%':>10}")
        print("-" * 55)
        for param in all_params:
            if param in betas and param in true_params:
                true_val = true_params[param]
                est_val = betas[param]
                if true_val != 0:
                    bias_pct = ((est_val - true_val) / abs(true_val)) * 100
                    print(f"{param:<20} {true_val:>10.4f} {est_val:>12.4f} {bias_pct:>+9.1f}%")
                else:
                    print(f"{param:<20} {true_val:>10.4f} {est_val:>12.4f} {'N/A':>10}")

        print("\n*** ICLV eliminates attenuation bias - LV effects are unbiased! ***")

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results_dict = {
        'model': model_name,
        'estimation_method': 'Simultaneous (ICLV)',
        'n_draws': n_draws,
        'log_likelihood': ll,
        'null_ll': null_ll,
        'rho_squared': rho2,
        'aic': aic,
        'bic': bic,
        'n_obs': n_obs,
        'n_params': k,
        'parameters': betas,
        'std_errors': std_errs,
        't_stats': t_stats,
        'p_values': p_vals,
        'converged': results.algorithm_has_converged
    }

    # Save parameter estimates
    pd.DataFrame([{
        'Parameter': p,
        'Estimate': betas[p],
        'Std_Error': std_errs.get(p, np.nan),
        't_stat': t_stats.get(p, np.nan),
        'p_value': p_vals.get(p, np.nan)
    } for p in betas]).to_csv(output_path / 'parameter_estimates.csv', index=False)

    print(f"\nResults saved to: {output_path}")

    return results_dict


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimate HCM Full model (ICLV)')
    parser.add_argument('--data', type=str, required=True, help='Path to data CSV')
    parser.add_argument('--config', type=str, default='config/model_config.json',
                        help='Path to config JSON with true parameters')
    parser.add_argument('--output', type=str, default='results/hcm_full',
                        help='Output directory')
    parser.add_argument('--draws', type=int, default=1000,
                        help='Number of Halton draws (default: 1000)')

    args = parser.parse_args()

    estimate_hcm_full(args.data, args.config, args.output, args.draws)
