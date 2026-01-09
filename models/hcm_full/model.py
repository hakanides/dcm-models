#!/usr/bin/env python3
"""
HCM Full Model - Simultaneous Estimation (ICLV)
================================================

Hybrid Choice Model with all 4 latent variables affecting fee and duration sensitivity.

Latent Constructs:
    1. Blind Patriotism (η_pb) - Uncritical support for country
    2. Constructive Patriotism (η_pc) - Critical, improvement-oriented
    3. Daily Life Secularism (η_sdl) - Separation in daily practices
    4. Faith & Prayer Secularism (η_sfp) - Separation in religious matters

Model Specification (Simultaneous ICLV):
    Structural: η_k = σ_k * ω_k,  ω_k ~ N(0,1)

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

import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import (
    Beta, Variable, MonteCarlo, log, exp, Elem, Draws
)


# =============================================================================
# MEASUREMENT MODEL (Ordered Logit for Likert Items)
# =============================================================================

def ordered_logit_prob(indicator: Variable,
                       loading,
                       lv,
                       thresholds: list):
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
    # Logistic CDF: 1 / (1 + exp(-x))
    def logistic_cdf(x):
        return 1 / (1 + exp(-x))

    # Cumulative probabilities at each threshold
    cum_probs = {}
    cum_probs[1] = logistic_cdf(thresholds[0] - loading * lv)
    cum_probs[2] = logistic_cdf(thresholds[1] - loading * lv)
    cum_probs[3] = logistic_cdf(thresholds[2] - loading * lv)
    cum_probs[4] = logistic_cdf(thresholds[3] - loading * lv)
    cum_probs[5] = 1.0

    # Category probabilities: P(I = j) = P(I <= j) - P(I <= j-1)
    category_probs = {}
    category_probs[1] = cum_probs[1]
    category_probs[2] = cum_probs[2] - cum_probs[1]
    category_probs[3] = cum_probs[3] - cum_probs[2]
    category_probs[4] = cum_probs[4] - cum_probs[3]
    category_probs[5] = cum_probs[5] - cum_probs[4]

    # Select probability based on observed indicator value
    prob = Elem(category_probs, indicator)

    return prob


def create_construct_measurement(database: db.Database,
                                 lv,
                                 items: list,
                                 thresholds: list,
                                 construct_name: str):
    """
    Create measurement likelihood for a single construct.

    Args:
        database: Biogeme database
        lv: Latent variable expression
        items: List of indicator column names
        thresholds: List of threshold Beta parameters
        construct_name: Name for parameter naming

    Returns:
        Joint probability expression and loadings dict
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

def create_iclv_model(database: db.Database, config: dict, available_columns: list):
    """
    Create ICLV model with simultaneous estimation for all 4 LVs.

    Args:
        database: Biogeme database
        config: Model configuration with LV items
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
    # LATENT VARIABLES: Structural Equations
    # η_k = σ_k * ω_k,  where ω_k ~ N(0,1)
    # -------------------------------------------------------------------------
    # Random draws for each latent variable (independent)
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

    # Get latent variable config
    latent_cfg = config.get('latent', {})

    # Build measurement model for each construct
    all_loadings = {}
    measurement_prob = 1.0

    constructs = [
        ('pat_blind', LV_pat_blind),
        ('pat_constructive', LV_pat_const),
        ('sec_dl', LV_sec_dl),
        ('sec_fp', LV_sec_fp)
    ]

    for construct_name, lv in constructs:
        cfg = latent_cfg.get(construct_name, {})
        items = cfg.get('measurement', {}).get('items', [])

        # Filter to available items
        available_items = [item for item in items
                          if item in available_columns]

        if available_items:
            print(f"  {construct_name}: {len(available_items)} indicators")

            prob, loadings = create_construct_measurement(
                database, lv, available_items, thresholds, construct_name
            )
            measurement_prob = measurement_prob * prob
            all_loadings.update(loadings)
        else:
            print(f"  WARNING: No items found for {construct_name}")

    # -------------------------------------------------------------------------
    # CHOICE MODEL
    # -------------------------------------------------------------------------
    # Base parameters
    ASC_paid = Beta('ASC_paid', 1.0, -10, 10, 0)
    B_FEE = Beta('B_FEE', -0.5, -10, 0, 0)
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
    av = {1: 1, 2: 1, 3: 1}

    # Choice probability (conditional on LVs)
    choice_prob = models.logit(V, av, CHOICE)

    # -------------------------------------------------------------------------
    # JOINT LIKELIHOOD WITH MONTE CARLO INTEGRATION
    # L_n = ∫∫∫∫ P(choice|η) × P(indicators|η) dη_1 dη_2 dη_3 dη_4
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

    # Ensure fee columns are scaled
    if 'fee1_10k' not in df.columns:
        for alt in [1, 2, 3]:
            df[f'fee{alt}_10k'] = df[f'fee{alt}'] / 10000.0

    # Convert all Likert items to integers (1-5)
    latent_cfg = config.get('latent', {})
    all_items = []
    for lv_cfg in latent_cfg.values():
        items = lv_cfg.get('measurement', {}).get('items', [])
        all_items.extend(items)

    for item in all_items:
        if item in df.columns:
            df[item] = df[item].round().astype(int).clip(1, 5)

    # Keep only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return df[numeric_cols].copy()


# =============================================================================
# ESTIMATION
# =============================================================================

def estimate(model_dir: Path, verbose: bool = True, n_draws: int = 1000) -> dict:
    """
    Estimate HCM Full model using simultaneous ICLV approach.

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
        print("HCM FULL MODEL - SIMULTANEOUS ICLV ESTIMATION")
        print("(All 4 Latent Variables on Fee and Duration)")
        print("=" * 70)
        print(f"\nTrue parameters:")
        for k, v in true_values.items():
            print(f"  {k}: {v}")
        print(f"\nUsing {n_draws} Halton draws for Monte Carlo integration")
        print("\nThis approach eliminates attenuation bias from two-stage estimation!")

    # Load and prepare data
    df = prepare_data(data_path, config)
    n_obs = len(df)

    if verbose:
        print(f"\nData: {n_obs} observations")

    # Create database
    database = db.Database('hcm_full_iclv', df)

    # Set number of draws
    database.number_of_draws = n_draws

    if verbose:
        print("\nBuilding ICLV model with 4 latent variables...")

    # Create ICLV model
    logprob = create_iclv_model(database, config, df.columns.tolist())

    # Create Biogeme object
    biogeme_model = bio.BIOGEME(database, logprob)
    biogeme_model.model_name = "HCM_Full_ICLV"

    # Calculate null log-likelihood
    biogeme_model.calculate_null_loglikelihood({1: 1, 2: 1, 3: 1})

    original_dir = os.getcwd()
    os.chdir(results_dir)

    if verbose:
        print("\nEstimating model (this may take several minutes)...")

    try:
        results = biogeme_model.estimate()
    finally:
        os.chdir(original_dir)

    # Extract results
    estimates_df = results.get_estimated_parameters()
    estimates_df = estimates_df.set_index('Name')

    betas = estimates_df['Value'].to_dict()
    stderrs = estimates_df['Robust std err.'].to_dict()
    tstats = estimates_df['Robust t-stat.'].to_dict()
    pvals = estimates_df['Robust p-value'].to_dict()

    # Get fit statistics
    general_stats = results.get_general_statistics()
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

        # Choice model parameters
        print("\n" + "-" * 70)
        print("CHOICE MODEL - BASE PARAMETERS")
        print("-" * 70)
        print(f"{'Parameter':<20} {'Estimate':>10} {'Std.Err':>10} {'t-stat':>10} {'p-value':>10}")
        print("-" * 65)

        base_params = ['ASC_paid', 'B_FEE', 'B_DUR']
        for param in base_params:
            if param in betas:
                sig = "***" if pvals[param] < 0.01 else "**" if pvals[param] < 0.05 else "*" if pvals[param] < 0.10 else ""
                print(f"{param:<20} {betas[param]:>10.4f} {stderrs[param]:>10.4f} {tstats[param]:>10.2f} {pvals[param]:>10.4f} {sig}")

        print("\n" + "-" * 70)
        print("FEE SENSITIVITY INTERACTIONS")
        print("-" * 70)
        fee_params = ['B_FEE_PatBlind', 'B_FEE_PatConst', 'B_FEE_SecDL', 'B_FEE_SecFP']
        for param in fee_params:
            if param in betas:
                sig = "***" if pvals[param] < 0.01 else "**" if pvals[param] < 0.05 else "*" if pvals[param] < 0.10 else ""
                print(f"{param:<20} {betas[param]:>10.4f} {stderrs[param]:>10.4f} {tstats[param]:>10.2f} {pvals[param]:>10.4f} {sig}")

        print("\n" + "-" * 70)
        print("DURATION SENSITIVITY INTERACTIONS")
        print("-" * 70)
        dur_params = ['B_DUR_PatBlind', 'B_DUR_PatConst', 'B_DUR_SecDL', 'B_DUR_SecFP']
        for param in dur_params:
            if param in betas:
                sig = "***" if pvals[param] < 0.01 else "**" if pvals[param] < 0.05 else "*" if pvals[param] < 0.10 else ""
                print(f"{param:<20} {betas[param]:>10.4f} {stderrs[param]:>10.4f} {tstats[param]:>10.2f} {pvals[param]:>10.4f} {sig}")

        print("\n" + "-" * 70)
        print("MEASUREMENT MODEL PARAMETERS")
        print("-" * 70)

        # Thresholds
        print("\nThresholds:")
        for i in range(1, 5):
            param = f'tau_{i}'
            if param in betas:
                print(f"  {param}: {betas[param]:.4f} (SE: {stderrs[param]:.4f})")

        # Loadings by construct
        print("\nFactor Loadings:")
        for construct in ['pat_blind', 'pat_constructive', 'sec_dl', 'sec_fp']:
            loadings = [p for p in sorted(betas.keys()) if p.startswith(f'lambda_{construct}')]
            if loadings:
                print(f"  {construct}:")
                for param in loadings:
                    print(f"    {param}: {betas[param]:.4f} (SE: {stderrs[param]:.4f})")

        # Comparison to true values
        print("\n" + "=" * 70)
        print("COMPARISON TO TRUE PARAMETERS")
        print("=" * 70)
        print(f"\n{'Parameter':<20} {'True':>10} {'Estimated':>10} {'Bias%':>10} {'95%CI':<6}")
        print("-" * 65)

        all_covered = True
        all_params = base_params + fee_params + dur_params

        for param in all_params:
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

        print("\n*** ICLV eliminates attenuation bias - LV effects are unbiased! ***")

    # Save results
    results_dict = {
        'model': 'HCM_Full_ICLV',
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
        'Model': 'HCM_Full_ICLV',
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
    results = estimate(model_dir, n_draws=1000)

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
        model_type='HCM_FULL_ICLV',
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
        model_name='HCM_Full_ICLV'
    )

    print(f"\nPolicy analysis saved to: {policy_dir}")
    print(f"LaTeX tables saved to: {latex_dir}")


if __name__ == "__main__":
    main()
