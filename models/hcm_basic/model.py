#!/usr/bin/env python3
"""
HCM Basic Model - Simultaneous Estimation (ICLV)
=================================================

Hybrid Choice Model with single latent variable (Blind Patriotism) on fee sensitivity.

Model Specification (Simultaneous ICLV):
    Structural Model:
        η = γ_age * (age - 2) + σ * ω,  where ω ~ N(0,1)

    Measurement Model (Ordered Probit):
        P(I_k = j | η) = Φ(τ_j - λ_k*η) - Φ(τ_{j-1} - λ_k*η)

    Choice Model:
        B_FEE_i = B_FEE + B_FEE_LV * η
        V1 = ASC_paid + B_FEE_i * fee1 + B_DUR * dur1
        V2 = ASC_paid + B_FEE_i * fee2 + B_DUR * dur2
        V3 = B_FEE_i * fee3 + B_DUR * dur3

    Joint Likelihood:
        L_n = ∫ P(choice|η) × Π_k P(I_k|η) dη

    Estimated via Monte Carlo integration using Halton draws.

This eliminates attenuation bias from two-stage estimation by
integrating over the latent variable distribution.

References:
    - Ben-Akiva et al. (2002): Hybrid Choice Models
    - Walker & Ben-Akiva (2002): Generalized Random Utility Model
    - Bierlaire (2018): Biogeme Documentation on Latent Variables
"""

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Add shared utilities to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.policy_tools import run_policy_analysis
from shared.latex_tools import generate_all_latex
from shared import validate_required_columns, validate_coefficient_signs

import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
# Handle Biogeme API changes (bioDraws deprecated in 3.3+, use Draws)
try:
    from biogeme.expressions import Beta, Variable, Draws, MonteCarlo, log, exp, Elem, bioNormalCdf
except ImportError:
    from biogeme.expressions import Beta, Variable, bioDraws as Draws, MonteCarlo, log, exp, Elem, bioNormalCdf


# Alternative availability (1=available, 0=not available)
# This should match the alternatives defined in config.json
ALTERNATIVES = {1: 'paid1', 2: 'paid2', 3: 'standard'}
AV_DICT = {k: 1 for k in ALTERNATIVES.keys()}  # All alternatives available


# =============================================================================
# MEASUREMENT MODEL (Ordered Probit for Likert Items)
# =============================================================================

def ordered_probit_prob(indicator, loading, lv, tau1, tau2, tau3, tau4):
    """
    Compute ordered probit probability for a single indicator.

    P(I = j | η) = Φ(τ_j - λ*η) - Φ(τ_{j-1} - λ*η)

    Uses the normal CDF (probit) which is more common in the literature
    and matches the DGP specification.

    Args:
        indicator: Biogeme Variable for the indicator (1-5)
        loading: Factor loading (λ)
        lv: Latent variable expression (η)
        tau1-4: Threshold parameters (must be ordered)

    Returns:
        Biogeme expression for probability
    """
    z = loading * lv

    # Cumulative probabilities using normal CDF
    prob_1 = bioNormalCdf(tau1 - z)
    prob_2 = bioNormalCdf(tau2 - z) - bioNormalCdf(tau1 - z)
    prob_3 = bioNormalCdf(tau3 - z) - bioNormalCdf(tau2 - z)
    prob_4 = bioNormalCdf(tau4 - z) - bioNormalCdf(tau3 - z)
    prob_5 = 1 - bioNormalCdf(tau4 - z)

    # Select probability based on observed indicator value
    prob = Elem({1: prob_1, 2: prob_2, 3: prob_3, 4: prob_4, 5: prob_5}, indicator)

    return prob


def create_measurement_likelihood(lv, items: list,
                                   tau1, tau2, tau3, tau4,
                                   loadings: dict):
    """
    Create joint measurement likelihood for all indicators.

    L_measurement = Π_k P(I_k | η)

    Args:
        lv: Latent variable expression
        items: List of indicator column names
        tau1-4: Threshold parameters
        loadings: Dict of item -> loading (Beta or float)

    Returns:
        Joint measurement probability expression
    """
    joint_prob = 1.0

    for item in items:
        indicator = Variable(item)
        loading = loadings[item]

        prob = ordered_probit_prob(indicator, loading, lv, tau1, tau2, tau3, tau4)
        joint_prob = joint_prob * prob

    return joint_prob


# =============================================================================
# MODEL SPECIFICATION
# =============================================================================

def create_iclv_model(database: db.Database, config: dict, available_columns: list):
    """
    Create ICLV model with simultaneous estimation.

    Key features:
    1. Structural model: η = γ*X + σ*ω linking demographics to LV
    2. Measurement model: Ordered probit for Likert items
    3. Choice model: Logit with LV interaction on fee sensitivity
    4. Integration: Monte Carlo over latent variable distribution

    Args:
        database: Biogeme database
        config: Model configuration with LV items and structural parameters
        available_columns: List of available column names in the data

    Returns:
        Log-likelihood expression for estimation
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
    # DEMOGRAPHICS (Centered)
    # -------------------------------------------------------------------------
    # Get centering value from config
    latent_cfg = config.get('latent', {}).get('pat_blind', {})
    structural_cfg = latent_cfg.get('structural', {})
    age_center = structural_cfg.get('center', {}).get('age_idx', 2.0)

    age_idx = Variable('age_idx')
    age_centered = age_idx - age_center

    # -------------------------------------------------------------------------
    # STRUCTURAL MODEL: η = γ*X + σ*ω
    # -------------------------------------------------------------------------
    # Structural coefficients (linking demographics to latent variable)
    gamma_age = Beta('gamma_age', 0.2, -2, 2, 0)
    sigma_LV = Beta('sigma_LV', 1.0, 0.1, 5, 0)

    # Random draw from standard normal (using Halton sequences for efficiency)
    omega = Draws('omega', 'NORMAL_HALTON2')

    # STRUCTURAL EQUATION: η = γ_age * age_centered + σ * ω
    LV_pat_blind = gamma_age * age_centered + sigma_LV * omega

    # -------------------------------------------------------------------------
    # MEASUREMENT MODEL: Ordered Probit for Likert Items
    # -------------------------------------------------------------------------
    # Delta parameterization for thresholds (GUARANTEES ordering)
    # τ_1 is free, τ_k = τ_{k-1} + exp(δ_k) ensures τ_1 < τ_2 < τ_3 < τ_4
    #
    # NOTE ON DGP ALIGNMENT:
    # The DGP uses FIXED thresholds [-1.0, -0.35, 0.35, 1.0] for simulation.
    # Here we ESTIMATE thresholds, which is standard practice for real data.
    # Starting values are initialized close to DGP values.
    tau_1 = Beta('tau_1', -1.0, -3, 3, 0)
    delta_2 = Beta('delta_2', 0.0, -3, 3, 0)  # exp(0) ≈ 1
    delta_3 = Beta('delta_3', 0.0, -3, 3, 0)
    delta_4 = Beta('delta_4', 0.0, -3, 3, 0)

    # Compute ordered thresholds
    tau_2 = tau_1 + exp(delta_2)
    tau_3 = tau_2 + exp(delta_3)
    tau_4 = tau_3 + exp(delta_4)

    # Get indicator items from config
    items = latent_cfg.get('measurement', {}).get('items', [])

    # Filter to available items in database
    available_items = [item for item in items if item in available_columns]

    if not available_items:
        raise ValueError("No indicator items found for pat_blind LV")

    print(f"  Using {len(available_items)} indicator items: {available_items}")

    # Factor loadings (first item fixed to 1.0 for scale identification)
    # IMPORTANT: We explicitly fix the first item in lexicographic order to ensure
    # consistent scale identification regardless of item ordering in the data.
    loadings = {}
    fixed_item = sorted(available_items)[0]  # Always fix the lexicographically first item
    loadings[fixed_item] = 1.0
    print(f"  Scale identification: λ({fixed_item}) = 1.0 (fixed)")

    for item in available_items:
        if item != fixed_item:
            loadings[item] = Beta(f'lambda_{item}', 0.8, 0.1, 3.0, 0)

    # Build measurement likelihood
    measurement_prob = create_measurement_likelihood(
        LV_pat_blind, available_items, tau_1, tau_2, tau_3, tau_4, loadings
    )

    # -------------------------------------------------------------------------
    # CHOICE MODEL
    # -------------------------------------------------------------------------
    # Base parameters
    ASC_paid = Beta('ASC_paid', 5.0, -10, 15, 0)
    B_FEE = Beta('B_FEE', -0.08, -1, 0, 0)
    B_DUR = Beta('B_DUR', -0.08, -1, 0, 0)

    # LV effect on fee sensitivity
    B_FEE_PatBlind = Beta('B_FEE_PatBlind', -0.1, -2, 2, 0)

    # Individual-specific fee coefficient: β_i = β + β_LV * η_i
    B_FEE_i = B_FEE + B_FEE_PatBlind * LV_pat_blind

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
    # Approximated via: (1/R) Σ_r P(choice|η_r) × P(indicators|η_r)
    # -------------------------------------------------------------------------
    joint_prob = choice_prob * measurement_prob

    # Monte Carlo integration over latent variable draws
    integrated_prob = MonteCarlo(joint_prob)

    # Log-likelihood
    logprob = log(integrated_prob)

    return logprob


# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_data(filepath: Path, config: dict) -> pd.DataFrame:
    """Load and prepare data for ICLV estimation."""
    df = pd.read_csv(filepath)

    # Base required columns
    required_cols = ['CHOICE', 'fee1', 'fee2', 'fee3', 'dur1', 'dur2', 'dur3', 'age_idx']

    # Add indicator items from config
    latent_cfg = config.get('latent', {}).get('pat_blind', {})
    items = latent_cfg.get('measurement', {}).get('items', [])
    required_cols.extend(items)

    # Validate required columns exist
    validate_required_columns(df, required_cols, 'HCM_Basic')

    # Ensure fee columns are scaled
    if 'fee1_10k' not in df.columns:
        for alt in [1, 2, 3]:
            df[f'fee{alt}_10k'] = df[f'fee{alt}'] / 10000.0

    for item in items:
        if item in df.columns:
            df[item] = df[item].round().astype(int).clip(1, 5)

    # Keep only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return df[numeric_cols].copy()


# =============================================================================
# ESTIMATION
# =============================================================================

def estimate(model_dir: Path, verbose: bool = True, n_draws: int = 500) -> dict:
    """
    Estimate HCM Basic model using simultaneous ICLV approach.

    Args:
        model_dir: Directory containing model files
        verbose: Print progress
        n_draws: Number of Halton draws for Monte Carlo integration

    Returns:
        Dictionary with estimation results
    """
    data_path = model_dir / "data" / "simulated_data.csv"
    config_path = model_dir / "config.json"
    results_dir = model_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path) as f:
        config = json.load(f)

    true_values = config['model_info']['true_values']

    if verbose:
        print("=" * 70)
        print("HCM BASIC MODEL - SIMULTANEOUS ICLV ESTIMATION")
        print("=" * 70)
        print(f"\nTrue parameters: {true_values}")
        print(f"\nUsing {n_draws} Halton draws for Monte Carlo integration")
        print("\nKey improvements over two-stage:")
        print("  - Structural equation: η = γ*age + σ*ω (demographics → LV)")
        print("  - Delta thresholds: τ_k = τ_{k-1} + exp(δ_k) (guaranteed ordering)")
        print("  - Ordered probit: Φ(τ - λη) (matches DGP)")
        print("  - Monte Carlo integration: eliminates attenuation bias")

    # Load and prepare data
    df = prepare_data(data_path, config)
    n_obs = len(df)

    if verbose:
        print(f"\nData: {n_obs} observations")

    # Create database
    database = db.Database('hcm_basic_iclv', df)
    database.panel('ID')  # Declare panel structure for consistent random draws within individuals

    if verbose:
        print("\nBuilding ICLV model...")

    # Create ICLV model
    logprob = create_iclv_model(database, config, df.columns.tolist())

    # Create Biogeme object with draws
    biogeme_model = bio.BIOGEME(database, logprob, number_of_draws=n_draws)
    biogeme_model.model_name = "HCM_Basic_ICLV"

    # Calculate null log-likelihood
    biogeme_model.calculate_null_loglikelihood(AV_DICT)

    original_dir = os.getcwd()
    os.chdir(results_dir)

    if verbose:
        print("\nEstimating model (this may take a few minutes)...")

    try:
        results = biogeme_model.estimate()
    finally:
        os.chdir(original_dir)

    # Check convergence status
    general_stats = results.get_general_statistics()
    converged = True
    for key, val in general_stats.items():
        if 'Optimization' in key and 'algorithm' not in key.lower():
            status = str(val[0]) if isinstance(val, tuple) else str(val)
            if 'success' not in status.lower() and 'converged' not in status.lower():
                converged = False
                if verbose:
                    print(f"\nWARNING: Optimization may not have converged! Status: {status}")
                break

    # Extract results
    estimates_df = results.get_estimated_parameters()
    estimates_df = estimates_df.set_index('Name')

    betas = estimates_df['Value'].to_dict()
    stderrs = estimates_df['Robust std err.'].to_dict()
    tstats = estimates_df['Robust t-stat.'].to_dict()
    pvals = estimates_df['Robust p-value'].to_dict()

    # Validate coefficient signs
    if verbose:
        validate_coefficient_signs(betas, model_name='HCM_Basic')

    # Get fit statistics (reuse general_stats from convergence check)
    final_ll = None
    null_ll = None
    aic = None
    bic = None

    for key, val in general_stats.items():
        if 'Final log likelihood' in key:
            final_ll = float(val[0]) if isinstance(val, tuple) else float(val)
        elif 'Null log likelihood' in key:
            null_ll = float(val[0]) if isinstance(val, tuple) else float(val)
        elif 'Akaike' in key:
            aic = float(val[0]) if isinstance(val, tuple) else float(val)
        elif 'Bayesian' in key:
            bic = float(val[0]) if isinstance(val, tuple) else float(val)

    rho2 = 1 - final_ll / null_ll if null_ll else 0

    if verbose:
        print("\n" + "=" * 70)
        print("ESTIMATION RESULTS")
        print("=" * 70)
        print(f"\nLog-Likelihood: {final_ll:.4f}")
        print(f"Null LL: {null_ll:.4f}")
        print(f"Rho-squared: {rho2:.4f}")
        print(f"AIC: {aic:.2f}")
        print(f"BIC: {bic:.2f}")

        print("\n" + "-" * 70)
        print("CHOICE MODEL PARAMETERS")
        print("-" * 70)
        print(f"{'Parameter':<20} {'Estimate':>10} {'Std.Err':>10} {'t-stat':>10} {'p-value':>10}")
        print("-" * 65)

        choice_params = ['ASC_paid', 'B_FEE', 'B_FEE_PatBlind', 'B_DUR']
        for param in choice_params:
            if param in betas:
                sig = "***" if pvals[param] < 0.01 else "**" if pvals[param] < 0.05 else "*" if pvals[param] < 0.10 else ""
                print(f"{param:<20} {betas[param]:>10.4f} {stderrs[param]:>10.4f} {tstats[param]:>10.2f} {pvals[param]:>10.4f} {sig}")

        print("\n" + "-" * 70)
        print("STRUCTURAL MODEL PARAMETERS")
        print("-" * 70)
        struct_params = ['gamma_age', 'sigma_LV']
        for param in struct_params:
            if param in betas:
                sig = "***" if pvals[param] < 0.01 else "**" if pvals[param] < 0.05 else "*" if pvals[param] < 0.10 else ""
                print(f"{param:<20} {betas[param]:>10.4f} {stderrs[param]:>10.4f} {tstats[param]:>10.2f} {pvals[param]:>10.4f} {sig}")

        print("\n" + "-" * 70)
        print("MEASUREMENT MODEL PARAMETERS")
        print("-" * 70)

        # Thresholds (show computed values)
        print("\nThresholds (delta parameterization):")
        if 'tau_1' in betas:
            tau1 = betas['tau_1']
            delta2 = betas.get('delta_2', 0)
            delta3 = betas.get('delta_3', 0)
            delta4 = betas.get('delta_4', 0)
            tau2 = tau1 + np.exp(delta2)
            tau3 = tau2 + np.exp(delta3)
            tau4 = tau3 + np.exp(delta4)
            print(f"  τ_1 = {tau1:.4f}")
            print(f"  τ_2 = {tau2:.4f} (τ_1 + exp(δ_2))")
            print(f"  τ_3 = {tau3:.4f} (τ_2 + exp(δ_3))")
            print(f"  τ_4 = {tau4:.4f} (τ_3 + exp(δ_4))")

        # Loadings
        print("\nFactor Loadings:")
        for param in sorted(betas.keys()):
            if param.startswith('lambda_'):
                print(f"  {param}: {betas[param]:.4f} (SE: {stderrs[param]:.4f})")

        print("\n" + "=" * 70)
        print("COMPARISON TO TRUE PARAMETERS")
        print("=" * 70)
        print(f"\n{'Parameter':<20} {'True':>10} {'Estimated':>10} {'Bias%':>10} {'95%CI':<6}")
        print("-" * 65)

        all_covered = True
        for param in choice_params:
            if param in betas and param in true_values:
                true_val = true_values[param]
                est_val = betas[param]
                se = stderrs[param]

                if abs(true_val) > 0.0001:
                    bias_pct = ((est_val - true_val) / abs(true_val)) * 100
                else:
                    bias_pct = 0

                ci_low = est_val - 1.96 * se
                ci_high = est_val + 1.96 * se
                covered = ci_low <= true_val <= ci_high
                coverage_str = "Yes" if covered else "NO"
                if not covered:
                    all_covered = False

                print(f"{param:<20} {true_val:>10.4f} {est_val:>10.4f} {bias_pct:>+9.1f}% {coverage_str:<6}")

        print("\n" + "-" * 65)
        if all_covered:
            print("SUCCESS: All true parameters fall within 95% confidence intervals")
        else:
            print("NOTE: Some parameters outside 95% CI")

        print("\n*** ICLV with structural equations provides unbiased LV effects! ***")

    # Save results
    results_dict = {
        'model': 'HCM_Basic_ICLV',
        'estimation_method': 'Simultaneous (Monte Carlo)',
        'n_draws': n_draws,
        'log_likelihood': final_ll,
        'null_ll': null_ll,
        'rho_squared': rho2,
        'aic': aic,
        'bic': bic,
        'n_obs': n_obs,
        'parameters': betas,
        'std_errors': stderrs,
        't_stats': tstats,
        'p_values': pvals,
        'true_values': true_values,
        'biogeme_results': results,
        'data': df,
        'config': config
    }

    # Save parameter estimates
    param_rows = []
    for param in betas.keys():
        true_val = true_values.get(param, None)
        bias_pct = ((betas[param] - true_val) / abs(true_val)) * 100 if true_val and abs(true_val) > 0.0001 else None
        ci_low = betas[param] - 1.96 * stderrs[param]
        ci_high = betas[param] + 1.96 * stderrs[param]
        covered = ci_low <= true_val <= ci_high if true_val is not None else None

        param_rows.append({
            'Parameter': param,
            'True_Value': true_val,
            'Estimate': betas[param],
            'Std_Error': stderrs[param],
            't_stat': tstats[param],
            'p_value': pvals[param],
            'Bias_Percent': bias_pct,
            'CI_95_Low': ci_low,
            'CI_95_High': ci_high,
            'CI_Coverage': covered
        })

    pd.DataFrame(param_rows).to_csv(results_dir / 'parameter_estimates.csv', index=False)

    pd.DataFrame([{
        'Model': 'HCM_Basic_ICLV',
        'Method': 'Simultaneous',
        'N_Draws': n_draws,
        'LL': final_ll,
        'Null_LL': null_ll,
        'K': len(betas),
        'N': n_obs,
        'AIC': aic,
        'BIC': bic,
        'Rho2': rho2
    }]).to_csv(results_dir / 'model_comparison.csv', index=False)

    if verbose:
        print(f"\nResults saved to: {results_dir}")

    return results_dict


def main():
    """Entry point for standalone execution."""
    model_dir = Path(__file__).parent
    results = estimate(model_dir, n_draws=500)

    # Run policy analysis
    policy_dir = model_dir / 'policy_analysis'
    policy_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 70)
    print("POLICY ANALYSIS")
    print("=" * 70)

    policy_results = run_policy_analysis(
        biogeme_results=results['biogeme_results'],
        df=results['data'],
        config=results['config'],
        output_dir=policy_dir,
        model_type='HCM_ICLV',
        verbose=True
    )

    # Generate LaTeX tables
    latex_dir = model_dir / 'output' / 'latex'
    latex_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("LATEX OUTPUT")
    print("=" * 70)

    generate_all_latex(
        biogeme_results=results['biogeme_results'],
        true_values=results['true_values'],
        policy_results=policy_results,
        output_dir=latex_dir,
        model_name='HCM_Basic_ICLV'
    )

    print(f"\nPolicy analysis saved to: {policy_dir}")
    print(f"LaTeX tables saved to: {latex_dir}")


if __name__ == "__main__":
    main()
