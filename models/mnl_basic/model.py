#!/usr/bin/env python3
"""
MNL Basic Model Estimation
==========================

Estimates a basic Multinomial Logit model with no interactions.

Model Specification:
    V1 = ASC_paid + B_FEE * fee1 + B_DUR * dur1
    V2 = ASC_paid + B_FEE * fee2 + B_DUR * dur2
    V3 = B_FEE * fee3 + B_DUR * dur3  (reference: ASC = 0)

True Parameters (from config):
    ASC_paid = 5.0
    B_FEE = -0.08
    B_DUR = -0.12
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta, Variable

# Add shared utilities to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.policy_tools import run_policy_analysis
from shared.latex_tools import generate_all_latex
from shared import validate_required_columns, validate_coefficient_signs

# Required columns for MNL Basic
REQUIRED_COLUMNS = ['CHOICE', 'fee1', 'fee2', 'fee3', 'dur1', 'dur2', 'dur3']

# Alternative availability (1=available, 0=not available)
# This should match the alternatives defined in config.json
ALTERNATIVES = {1: 'paid1', 2: 'paid2', 3: 'standard'}
AV_DICT = {k: 1 for k in ALTERNATIVES.keys()}  # All alternatives available


def prepare_data(filepath: Path) -> pd.DataFrame:
    """Load and prepare data for estimation."""
    df = pd.read_csv(filepath)

    # Validate required columns exist
    validate_required_columns(df, REQUIRED_COLUMNS, 'MNL_Basic')

    # Validate panel structure exists (required for cluster-robust standard errors)
    if 'ID' not in df.columns:
        raise ValueError(
            "Panel data with 'ID' column required for cluster-robust standard errors. "
            "Each individual should have multiple choice observations."
        )

    # Ensure fee columns are scaled
    if 'fee1_10k' not in df.columns:
        for alt in [1, 2, 3]:
            df[f'fee{alt}_10k'] = df[f'fee{alt}'] / 10000.0

    # Keep only numeric columns for Biogeme
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return df[numeric_cols].copy()


def create_model(database: db.Database):
    """Create MNL Basic model specification."""
    # Variables
    CHOICE = Variable('CHOICE')
    fee1 = Variable('fee1_10k')
    fee2 = Variable('fee2_10k')
    fee3 = Variable('fee3_10k')
    dur1 = Variable('dur1')
    dur2 = Variable('dur2')
    dur3 = Variable('dur3')

    # Parameters with bounds
    # ASC NORMALIZATION EXPLAINED:
    # - Alternative 3 is the REFERENCE (ASC_3 = 0, normalized to zero)
    # - ASC_paid appears in BOTH V1 and V2 because they share the same
    #   alternative-specific constant relative to the reference
    # - This captures the baseline preference for "paid" options over "free"
    # - If paid alternatives should differ, add ASC_1 and ASC_2 separately
    #
    # Current specification: ASC_paid = preference for any paid vs free
    # Alternative: ASC_1, ASC_2 = separate constants for each paid option
    # Starting values aligned with DGP (config.json true_values)
    # Bounds: [-5, 10] for ASC, narrower for coefficients
    ASC_paid = Beta('ASC_paid', 5.0, -5, 10, 0)  # True: 5.0
    B_FEE = Beta('B_FEE', -0.08, -1, 0, 0)       # True: -0.08
    B_DUR = Beta('B_DUR', -0.12, -1, 0, 0)       # True: -0.12

    # Utility functions
    # V1, V2: Paid alternatives with shared ASC
    # V3: Free alternative (reference, ASC = 0)
    V1 = ASC_paid + B_FEE * fee1 + B_DUR * dur1
    V2 = ASC_paid + B_FEE * fee2 + B_DUR * dur2
    V3 = B_FEE * fee3 + B_DUR * dur3  # Reference (ASC = 0)

    V = {1: V1, 2: V2, 3: V3}

    logprob = models.loglogit(V, AV_DICT, CHOICE)
    return logprob


def estimate(model_dir: Path, verbose: bool = True) -> dict:
    """
    Estimate MNL Basic model and compare to true parameters.

    Returns:
        Dictionary with estimation results
    """
    # Paths
    data_path = model_dir / "data" / "simulated_data.csv"
    config_path = model_dir / "config.json"
    results_dir = model_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    with open(config_path) as f:
        config = json.load(f)

    true_values = config['model_info']['true_values']

    if verbose:
        print("=" * 70)
        print("MNL BASIC MODEL ESTIMATION")
        print("=" * 70)
        print(f"\nTrue parameters: {true_values}")

    # Load and prepare data
    df = prepare_data(data_path)
    n_obs = len(df)
    n_individuals = df['ID'].nunique()

    if verbose:
        print(f"Data: {n_obs} observations from {n_individuals} individuals")

    # Create database
    # Note: In Biogeme 3.3.1, database.panel() causes CHOICE variable lookup failure
    # due to internal data reshaping. Robust standard errors are still computed but
    # are observation-level, not cluster-robust. For cluster-robust SEs, use
    # bootstrap or upgrade to Biogeme 3.3.2+.
    database = db.Database('mnl_basic', df)

    # Create and estimate model
    logprob = create_model(database)

    biogeme_model = bio.BIOGEME(database, logprob)
    biogeme_model.model_name = "MNL_Basic"
    biogeme_model.calculate_null_loglikelihood(AV_DICT)

    # Change to results dir for output files
    import os
    original_dir = os.getcwd()
    os.chdir(results_dir)

    if verbose:
        print("\nEstimating model...")

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
        validate_coefficient_signs(betas, model_name='MNL_Basic')

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

    # Print results
    if verbose:
        print("\n" + "=" * 70)
        print("ESTIMATION RESULTS")
        print("=" * 70)
        print(f"\nLog-Likelihood: {final_ll:.4f}")
        print(f"Null LL: {null_ll:.4f}")
        print(f"Rho-squared: {rho2:.4f}")
        print(f"AIC: {aic:.2f}")
        print(f"BIC: {bic:.2f}")

        print("\nParameter Estimates:")
        print(f"{'Parameter':<12} {'Estimate':>10} {'Std.Err':>10} {'t-stat':>10} {'p-value':>10}")
        print("-" * 56)
        for param in ['ASC_paid', 'B_FEE', 'B_DUR']:
            sig = "***" if pvals[param] < 0.01 else "**" if pvals[param] < 0.05 else "*" if pvals[param] < 0.10 else ""
            print(f"{param:<12} {betas[param]:>10.4f} {stderrs[param]:>10.4f} {tstats[param]:>10.2f} {pvals[param]:>10.4f} {sig}")

        # Compare to true parameters
        print("\n" + "=" * 70)
        print("COMPARISON TO TRUE PARAMETERS")
        print("=" * 70)
        print(f"\n{'Parameter':<12} {'True':>10} {'Estimated':>10} {'Bias%':>10} {'95% CI Coverage':<15}")
        print("-" * 60)

        all_covered = True
        for param in ['ASC_paid', 'B_FEE', 'B_DUR']:
            true_val = true_values[param]
            est_val = betas[param]
            se = stderrs[param]

            # Bias
            bias_pct = ((est_val - true_val) / abs(true_val)) * 100

            # 95% CI coverage
            ci_low = est_val - 1.96 * se
            ci_high = est_val + 1.96 * se
            covered = ci_low <= true_val <= ci_high
            coverage_str = "Yes" if covered else "NO"
            if not covered:
                all_covered = False

            print(f"{param:<12} {true_val:>10.4f} {est_val:>10.4f} {bias_pct:>+9.1f}% {coverage_str:<15}")

        print("\n" + "-" * 60)
        if all_covered:
            print("SUCCESS: All true parameters fall within 95% confidence intervals")
        else:
            print("WARNING: Some true parameters outside 95% CI - check for issues")

    # Save results
    results_dict = {
        'model': 'MNL_Basic',
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
        'true_values': true_values
    }

    # Save parameter estimates CSV
    param_rows = []
    for param in betas.keys():
        true_val = true_values.get(param, None)
        bias_pct = ((betas[param] - true_val) / abs(true_val)) * 100 if true_val else None
        ci_low = betas[param] - 1.96 * stderrs[param]
        ci_high = betas[param] + 1.96 * stderrs[param]
        covered = ci_low <= true_val <= ci_high if true_val else None

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

    # Save model comparison CSV
    pd.DataFrame([{
        'Model': 'MNL_Basic',
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

    # Return both dict and biogeme results for policy analysis
    results_dict['biogeme_results'] = results
    results_dict['data'] = df
    results_dict['config'] = config

    return results_dict


def main():
    """Entry point for standalone execution."""
    model_dir = Path(__file__).parent

    # Run estimation
    results = estimate(model_dir)

    # Run policy analysis
    policy_dir = model_dir / "policy_analysis"
    policy_results = run_policy_analysis(
        biogeme_results=results['biogeme_results'],
        df=results['data'],
        config=results['config'],
        output_dir=policy_dir,
        model_type='MNL'
    )

    # Generate LaTeX tables
    latex_dir = model_dir / "output" / "latex"
    generate_all_latex(
        biogeme_results=results['biogeme_results'],
        true_values=results['true_values'],
        policy_results=policy_results,
        output_dir=latex_dir,
        model_name='MNL_Basic'
    )


if __name__ == "__main__":
    main()
