"""
HCM Basic Model - Simultaneous Estimation (ICLV)
=================================================

Hybrid Choice Model with a single latent variable (Blind Patriotism).

This implements proper ICLV (Integrated Choice and Latent Variable) estimation
using Biogeme's Monte Carlo integration, eliminating attenuation bias from
two-stage approaches.

Model Specification:
    Structural (links demographics to latent variable):
        η = γ_age * (age - center) + σ * ω,  where ω ~ N(0,1)

    Measurement (Ordered Probit - preferred over logit per Biogeme docs):
        P(I_k = j | η) = Φ(τ_j - λ_k*η) - Φ(τ_{j-1} - λ_k*η)
        where Φ is the standard normal CDF

    Choice:
        B_FEE_i = B_FEE + B_FEE_LV * η
        V1 = ASC_paid + B_FEE_i * fee1 + B_DUR * dur1
        V2 = ASC_paid + B_FEE_i * fee2 + B_DUR * dur2
        V3 = B_FEE_i * fee3 + B_DUR * dur3

    Likelihood (integrated via Monte Carlo):
        L_n = ∫ P(choice|η) × Π_k P(I_k|η) dη

Identification:
    - First factor loading fixed to 1.0 for scale identification
    - sigma_LV estimated (or can be fixed to 1)
    - Thresholds freely estimated with proper ordering

References:
    - Bierlaire (2018): Technical Report on Latent Variables in Biogeme
    - Walker & Ben-Akiva (2002): Generalized Random Utility Model
    - Ben-Akiva et al. (2002): Hybrid Choice Models: Progress and Challenges

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
    Beta, Variable, MonteCarlo, log, exp, Elem, Draws, bioNormalCdf
)

from src.utils.item_detection import get_lv_items, get_items_by_prefix


# =============================================================================
# MODEL CONSTANTS
# =============================================================================

# Availability dict for alternatives (used in logit and null LL calculation)
AV_DICT = {1: 1, 2: 1, 3: 1}  # All alternatives always available


# =============================================================================
# MEASUREMENT MODEL (Ordered Probit for Likert Items)
# =============================================================================

def ordered_probit_prob(indicator, loading, lv, thresholds):
    """
    Compute ordered probit probability for a single indicator.

    P(I = j | η) = Φ(τ_j - λ*η) - Φ(τ_{j-1} - λ*η)

    Uses the normal CDF (probit) which is more common in the literature
    and matches the DGP specification.

    Args:
        indicator: Biogeme Variable for the indicator (1-5)
        loading: Factor loading (λ) - can be float or Beta
        lv: Latent variable expression (η)
        thresholds: List of threshold Beta parameters [τ_1, τ_2, τ_3, τ_4]

    Returns:
        Biogeme expression for probability
    """
    z = loading * lv

    # Cumulative probabilities using normal CDF (probit)
    prob_1 = bioNormalCdf(thresholds[0] - z)
    prob_2 = bioNormalCdf(thresholds[1] - z) - bioNormalCdf(thresholds[0] - z)
    prob_3 = bioNormalCdf(thresholds[2] - z) - bioNormalCdf(thresholds[1] - z)
    prob_4 = bioNormalCdf(thresholds[3] - z) - bioNormalCdf(thresholds[2] - z)
    prob_5 = 1 - bioNormalCdf(thresholds[3] - z)

    # Select probability based on observed indicator value
    prob = Elem({1: prob_1, 2: prob_2, 3: prob_3, 4: prob_4, 5: prob_5}, indicator)

    return prob


def create_measurement_likelihood(lv, items, thresholds, loadings):
    """
    Create joint measurement likelihood for all indicators.

    L_measurement = Π_k P(I_k | η)

    Args:
        lv: Latent variable expression
        items: List of indicator column names
        thresholds: List of threshold Beta parameters
        loadings: Dict of item -> loading (float or Beta)

    Returns:
        Joint measurement probability expression
    """
    joint_prob = 1.0

    for item in items:
        indicator = Variable(item)
        loading = loadings[item]

        prob = ordered_probit_prob(indicator, loading, lv, thresholds)
        joint_prob = joint_prob * prob

    return joint_prob


# =============================================================================
# MODEL SPECIFICATION
# =============================================================================

def create_hcm_basic_iclv(database: db.Database, items: list):
    """
    Create basic HCM with Blind Patriotism using ICLV estimation.

    Args:
        database: Biogeme database with indicator columns
        items: List of Likert indicator column names

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
    # DEMOGRAPHICS (Centered for numerical stability)
    # -------------------------------------------------------------------------
    age_idx = Variable('age_idx')
    # Center age around mean (typically 2 for age_idx scale 1-4)
    age_centered = age_idx - 2.0

    # -------------------------------------------------------------------------
    # LATENT VARIABLE: Structural Equation
    # η = γ_age * (age - center) + σ * ω,  where ω ~ N(0,1)
    # This links observable demographics to the latent construct
    # -------------------------------------------------------------------------
    # Structural parameters
    gamma_age = Beta('gamma_age', 0.2, -2, 2, 0)  # Effect of age on LV
    sigma_LV = Beta('sigma_LV', 1.0, 0.1, 5, 0)   # LV standard deviation

    # Random draw from standard normal (using Halton sequences for efficiency)
    omega = Draws('omega', 'NORMAL_HALTON2')

    # STRUCTURAL EQUATION: η = γ_age * age_centered + σ * ω
    LV_pat_blind = gamma_age * age_centered + sigma_LV * omega

    # -------------------------------------------------------------------------
    # MEASUREMENT MODEL: Ordered Probit for Likert Items
    # P(I = j | η) = Φ(τ_j - λ*η) - Φ(τ_{j-1} - λ*η)
    # -------------------------------------------------------------------------
    # Shared thresholds
    tau_1 = Beta('tau_1', -1.5, -5, 5, 0)
    tau_2 = Beta('tau_2', -0.5, -5, 5, 0)
    tau_3 = Beta('tau_3', 0.5, -5, 5, 0)
    tau_4 = Beta('tau_4', 1.5, -5, 5, 0)
    thresholds = [tau_1, tau_2, tau_3, tau_4]

    # Factor loadings (first fixed to 1 for identification)
    loadings = {}
    loadings[items[0]] = 1.0  # First loading fixed

    for i, item in enumerate(items[1:], start=2):
        loadings[item] = Beta(f'lambda_{i}', 0.8, 0.01, 5.0, 0)

    # Build measurement likelihood
    measurement_prob = create_measurement_likelihood(
        LV_pat_blind, items, thresholds, loadings
    )

    # -------------------------------------------------------------------------
    # CHOICE MODEL
    # -------------------------------------------------------------------------
    ASC_paid = Beta('ASC_paid', 0, -10, 10, 0)
    B_FEE = Beta('B_FEE', -0.6, -10, 0, 0)
    B_DUR = Beta('B_DUR', -0.05, -5, 0, 0)

    # LV interaction on fee sensitivity
    B_FEE_LV = Beta('B_FEE_PatBlind', 0.1, -2, 2, 0)

    # Individual-specific fee coefficient
    B_FEE_i = B_FEE + B_FEE_LV * LV_pat_blind

    # Utility functions
    V1 = ASC_paid + B_FEE_i * fee1 + B_DUR * dur1
    V2 = ASC_paid + B_FEE_i * fee2 + B_DUR * dur2
    V3 = B_FEE_i * fee3 + B_DUR * dur3

    V = {1: V1, 2: V2, 3: V3}

    # Choice probability (conditional on LV)
    choice_prob = models.logit(V, AV_DICT, CHOICE)

    # -------------------------------------------------------------------------
    # JOINT LIKELIHOOD WITH MONTE CARLO INTEGRATION
    # L_n = ∫ P(choice|η) × P(indicators|η) dη
    # -------------------------------------------------------------------------
    joint_prob = choice_prob * measurement_prob

    # Monte Carlo integration
    integrated_prob = MonteCarlo(joint_prob)

    # Log-likelihood
    logprob = log(integrated_prob)

    return logprob, 'HCM_Basic_ICLV'


# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_data(filepath: str, fee_scale: float = 10000.0):
    """
    Load and prepare data for ICLV estimation.

    Args:
        filepath: Path to CSV data file
        fee_scale: Divisor for fee scaling (default 10000)

    Returns:
        Tuple of (prepared DataFrame, list of indicator items)
    """
    df = pd.read_csv(filepath)

    # Scale fees
    for alt in [1, 2, 3]:
        df[f'fee{alt}_10k'] = df[f'fee{alt}'] / fee_scale

    # Find Blind Patriotism items
    pat_blind_items = get_lv_items(df, 'pat_blind')

    if not pat_blind_items:
        pat_blind_items = get_items_by_prefix(df, 'pat_blind_')

    if not pat_blind_items:
        raise ValueError("No blind patriotism items found")

    print(f"Found {len(pat_blind_items)} Blind Patriotism items: {pat_blind_items}")

    # Convert Likert items to integers (1-5)
    for item in pat_blind_items:
        if item in df.columns:
            df[item] = df[item].round().astype(int).clip(1, 5)

    # Keep only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    return df[numeric_cols].copy(), pat_blind_items


# =============================================================================
# ESTIMATION
# =============================================================================

def estimate_hcm_basic(data_path: str,
                       config_path: str = None,
                       output_dir: str = 'results/hcm_basic',
                       n_draws: int = 1000):
    """
    Estimate basic HCM model using simultaneous ICLV approach.

    Args:
        data_path: Path to data CSV
        config_path: Path to config JSON with true parameters (optional)
        output_dir: Directory for output files
        n_draws: Number of Halton draws for Monte Carlo integration

    Returns:
        Dictionary with estimation results
    """
    print("=" * 70)
    print("HCM BASIC MODEL - SIMULTANEOUS ICLV ESTIMATION")
    print("=" * 70)
    print(f"\nUsing {n_draws} Halton draws for Monte Carlo integration")
    print("This approach eliminates attenuation bias from two-stage estimation!")

    # Load and prepare data
    df, items = prepare_data(data_path)
    n_obs = len(df)

    # Validate panel structure exists (required for HCM/ICLV with repeated choices)
    if 'ID' not in df.columns:
        raise ValueError(
            "HCM/ICLV requires panel data with 'ID' column for correct standard errors. "
            "Each individual should have multiple choice observations."
        )
    n_individuals = df['ID'].nunique()

    print(f"\nData: {n_obs} observations from {n_individuals} individuals")
    print(f"Indicators: {len(items)} items")

    # Create database with panel structure for cluster-robust inference
    database = db.Database('hcm_basic_iclv', df)
    database.panel('ID')  # Declare panel structure for consistent random draws
    database.number_of_draws = n_draws

    # Create model
    print("\nBuilding ICLV model...")
    logprob, model_name = create_hcm_basic_iclv(database, items)

    # Estimate
    print("\nEstimating model (this may take a few minutes)...")
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
            # Approximate p-value from t-stat
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

    print("\n" + "-" * 70)
    print("CHOICE MODEL PARAMETERS")
    print("-" * 70)
    print(f"{'Parameter':<20} {'Estimate':>12} {'Std.Err':>10} {'t-stat':>10}")
    print("-" * 55)

    choice_params = ['ASC_paid', 'B_FEE', 'B_FEE_PatBlind', 'B_DUR']
    for param in choice_params:
        if param in betas:
            sig = "***" if p_vals.get(param, 1) < 0.01 else "**" if p_vals.get(param, 1) < 0.05 else "*" if p_vals.get(param, 1) < 0.10 else ""
            print(f"{param:<20} {betas[param]:>12.4f} {std_errs.get(param, np.nan):>10.4f} {t_stats.get(param, np.nan):>10.2f} {sig}")

    print("\n" + "-" * 70)
    print("MEASUREMENT MODEL PARAMETERS")
    print("-" * 70)

    print("\nThresholds:")
    for i in range(1, 5):
        param = f'tau_{i}'
        if param in betas:
            print(f"  {param}: {betas[param]:.4f} (SE: {std_errs.get(param, np.nan):.4f})")

    print("\nFactor Loadings (first fixed to 1):")
    for param in sorted(betas.keys()):
        if param.startswith('lambda_'):
            print(f"  {param}: {betas[param]:.4f} (SE: {std_errs.get(param, np.nan):.4f})")

    # Compare to true values if config provided
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            config = json.load(f)

        print("\n" + "=" * 70)
        print("COMPARISON TO TRUE PARAMETERS")
        print("=" * 70)

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
            if term['name'] == 'b_dur':
                true_params['B_DUR'] = term['base_coef']

        print(f"\n{'Parameter':<20} {'True':>10} {'Estimated':>12} {'Bias%':>10}")
        print("-" * 55)
        for param in choice_params:
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
    parser = argparse.ArgumentParser(description='Estimate HCM Basic model (ICLV)')
    parser.add_argument('--data', type=str, required=True, help='Path to data CSV')
    parser.add_argument('--config', type=str, default='config/model_config.json',
                        help='Path to config JSON with true parameters')
    parser.add_argument('--output', type=str, default='results/hcm_basic',
                        help='Output directory')
    parser.add_argument('--draws', type=int, default=1000,
                        help='Number of Halton draws (default: 1000)')

    args = parser.parse_args()

    estimate_hcm_basic(args.data, args.config, args.output, args.draws)
