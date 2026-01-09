"""
MNL Demographics Model
======================

Multinomial Logit with demographic interactions on fee and duration sensitivity.

Model Specification:
    B_FEE_i = B_FEE + B_FEE_AGE*(age-mean) + B_FEE_EDU*(edu-mean) + B_FEE_INC*(inc-mean)
    B_DUR_i = B_DUR + B_DUR_EDU*(edu-mean) + B_DUR_INC*(inc-mean)

    V1 = ASC_paid + B_FEE_i * fee1 + B_DUR_i * dur1
    V2 = ASC_paid + B_FEE_i * fee2 + B_DUR_i * dur2
    V3 = B_FEE_i * fee3 + B_DUR_i * dur3

This model captures OBSERVED heterogeneity in taste (demographic-based).

Usage:
    python src/models/mnl_demographics.py --data data/simulated/fresh_simulation.csv

Author: DCM Research Team
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
from biogeme.expressions import Beta, Variable


# =============================================================================
# MODEL SPECIFICATION
# =============================================================================

def create_mnl_demographics(database: db.Database):
    """
    Create MNL model with demographic interactions.

    Args:
        database: Biogeme database

    Returns:
        Tuple of (log_probability, model_name)
    """
    # Choice and attribute variables
    CHOICE = Variable('CHOICE')
    fee1 = Variable('fee1_10k')
    fee2 = Variable('fee2_10k')
    fee3 = Variable('fee3_10k')
    dur1 = Variable('dur1')
    dur2 = Variable('dur2')
    dur3 = Variable('dur3')

    # Demographic variables (centered)
    age_c = Variable('age_c')
    edu_c = Variable('edu_c')
    inc_c = Variable('inc_c')

    # Base parameters
    ASC_paid = Beta('ASC_paid', 0, None, None, 0)
    B_FEE = Beta('B_FEE', -0.01, None, None, 0)
    B_DUR = Beta('B_DUR', -0.01, None, None, 0)

    # Demographic interactions on fee sensitivity
    B_FEE_AGE = Beta('B_FEE_AGE', 0, None, None, 0)
    B_FEE_EDU = Beta('B_FEE_EDU', 0, None, None, 0)
    B_FEE_INC = Beta('B_FEE_INC', 0, None, None, 0)

    # Demographic interactions on duration sensitivity
    B_DUR_EDU = Beta('B_DUR_EDU', 0, None, None, 0)
    B_DUR_INC = Beta('B_DUR_INC', 0, None, None, 0)

    # Individual-specific coefficients
    B_FEE_i = B_FEE + B_FEE_AGE * age_c + B_FEE_EDU * edu_c + B_FEE_INC * inc_c
    B_DUR_i = B_DUR + B_DUR_EDU * edu_c + B_DUR_INC * inc_c

    # Utility functions
    V1 = ASC_paid + B_FEE_i * fee1 + B_DUR_i * dur1
    V2 = ASC_paid + B_FEE_i * fee2 + B_DUR_i * dur2
    V3 = B_FEE_i * fee3 + B_DUR_i * dur3

    V = {1: V1, 2: V2, 3: V3}
    av = {1: 1, 2: 1, 3: 1}

    logprob = models.loglogit(V, av, CHOICE)

    return logprob, 'MNL_Demographics'


# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_data(filepath: str, fee_scale: float = 10000.0) -> pd.DataFrame:
    """
    Load and prepare data for estimation.

    Args:
        filepath: Path to CSV data file
        fee_scale: Divisor for fee scaling (default 10000)

    Returns:
        Prepared DataFrame with centered demographics
    """
    df = pd.read_csv(filepath)

    # Scale fees
    for alt in [1, 2, 3]:
        df[f'fee{alt}_10k'] = df[f'fee{alt}'] / fee_scale

    # Center demographic variables
    # These centering values should match the DGP config
    df['age_c'] = (df['age_idx'] - 2.0) / 2.0  # Centered and scaled
    df['edu_c'] = (df['edu_idx'] - 3.0) / 2.0
    df['inc_c'] = (df['income_indiv_idx'] - 3.0) / 2.0

    # Keep only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    return df[numeric_cols].copy()


# =============================================================================
# ESTIMATION
# =============================================================================

def estimate_mnl_demographics(data_path: str,
                              config_path: str = None,
                              output_dir: str = 'results/mnl_demographics') -> dict:
    """
    Estimate MNL model with demographic interactions.

    Args:
        data_path: Path to data CSV
        config_path: Path to config JSON with true parameters (optional)
        output_dir: Directory for output files

    Returns:
        Dictionary with estimation results
    """
    print("="*70)
    print("MNL DEMOGRAPHICS MODEL ESTIMATION")
    print("="*70)

    # Load and prepare data
    df = prepare_data(data_path)
    n_obs = len(df)

    print(f"\nData: {n_obs} observations")
    print(f"Demographics: age_c, edu_c, inc_c (centered and scaled)")

    # Create database
    database = db.Database('mnl_demographics', df)

    # Create model
    logprob, model_name = create_mnl_demographics(database)

    # Estimate
    print("\nEstimating model...")
    biogeme_model = bio.BIOGEME(database, logprob)
    biogeme_model.model_name = model_name
    biogeme_model.calculate_null_loglikelihood({1: 1, 2: 1, 3: 1})

    results = biogeme_model.estimate()

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

    # Print results
    print("\n" + "="*70)
    print("ESTIMATION RESULTS")
    print("="*70)
    print(f"\nModel: {model_name}")
    print(f"Log-Likelihood: {final_ll:.4f}")
    print(f"Null LL: {null_ll:.4f}")
    print(f"Rho-squared: {rho2:.4f}")
    print(f"AIC: {aic:.2f}")
    print(f"BIC: {bic:.2f}")
    print(f"N: {n_obs}")
    print(f"K: {len(betas)}")

    print("\nParameter Estimates:")
    print(f"{'Parameter':<12} {'Estimate':>12} {'Std.Err':>10} {'t-stat':>10} {'p-value':>10}")
    print("-"*56)

    param_order = ['ASC_paid', 'B_FEE', 'B_FEE_AGE', 'B_FEE_EDU', 'B_FEE_INC',
                   'B_DUR', 'B_DUR_EDU', 'B_DUR_INC']

    for param in param_order:
        if param in betas:
            sig = "***" if pvals[param] < 0.01 else "**" if pvals[param] < 0.05 else "*" if pvals[param] < 0.10 else ""
            print(f"{param:<12} {betas[param]:>12.4f} {stderrs[param]:>10.4f} {tstats[param]:>10.2f} {pvals[param]:>10.4f} {sig}")

    # Compare to true if config provided
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            config = json.load(f)

        print("\n" + "="*70)
        print("COMPARISON TO TRUE PARAMETERS")
        print("="*70)

        # Extract true values from config
        true_params = {}
        for term in config['choice_model']['base_terms']:
            if term['name'] == 'ASC_paid':
                true_params['ASC_paid'] = term['coef']

        for term in config['choice_model']['attribute_terms']:
            if term['name'] == 'b_fee10k':
                true_params['B_FEE'] = term['base_coef']
                for inter in term.get('interactions', []):
                    if inter['with'] == 'age_idx':
                        true_params['B_FEE_AGE'] = inter['coef']
                    elif inter['with'] == 'edu_idx':
                        true_params['B_FEE_EDU'] = inter['coef']
                    elif inter['with'] == 'income_indiv_idx':
                        true_params['B_FEE_INC'] = inter['coef']
            elif term['name'] == 'b_dur':
                true_params['B_DUR'] = term['base_coef']
                for inter in term.get('interactions', []):
                    if inter['with'] == 'edu_idx':
                        true_params['B_DUR_EDU'] = inter['coef']
                    elif inter['with'] == 'income_indiv_idx':
                        true_params['B_DUR_INC'] = inter['coef']

        print(f"\n{'Parameter':<12} {'True':>10} {'Estimated':>12} {'Bias%':>10}")
        print("-"*46)
        for param in param_order:
            if param in betas and param in true_params:
                true_val = true_params[param]
                est_val = betas[param]
                if true_val != 0:
                    bias_pct = ((est_val - true_val) / abs(true_val)) * 100
                    print(f"{param:<12} {true_val:>10.4f} {est_val:>12.4f} {bias_pct:>+9.1f}%")
                else:
                    print(f"{param:<12} {true_val:>10.4f} {est_val:>12.4f} {'N/A':>10}")

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results_dict = {
        'model': model_name,
        'log_likelihood': final_ll,
        'null_ll': null_ll,
        'rho_squared': rho2,
        'aic': aic,
        'bic': bic,
        'n_obs': n_obs,
        'n_params': len(betas),
        'parameters': betas,
        'std_errors': stderrs,
        't_stats': tstats,
        'p_values': pvals,
        'converged': True
    }

    pd.DataFrame([{
        'Parameter': k,
        'Estimate': betas[k],
        'Std_Error': stderrs[k],
        't_stat': tstats[k],
        'p_value': pvals[k]
    } for k in betas]).to_csv(output_path / 'parameter_estimates.csv', index=False)

    print(f"\nResults saved to: {output_path}")

    return results_dict


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimate MNL Demographics model')
    parser.add_argument('--data', type=str, required=True, help='Path to data CSV')
    parser.add_argument('--config', type=str, default='config/model_config.json',
                        help='Path to config JSON with true parameters')
    parser.add_argument('--output', type=str, default='results/mnl_demographics',
                        help='Output directory')

    args = parser.parse_args()

    estimate_mnl_demographics(args.data, args.config, args.output)
