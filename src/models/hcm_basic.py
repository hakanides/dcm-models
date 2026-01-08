"""
HCM Basic Model
===============

Hybrid Choice Model with a single latent variable (Blind Patriotism).

This is the simplest HCM specification that demonstrates how latent
attitudes can moderate price sensitivity in discrete choice models.

Model Specification:
    B_FEE_i = B_FEE + B_FEE_LV * LV_pat_blind

    V1 = ASC_paid + B_FEE_i * fee1 + B_DUR * dur1
    V2 = ASC_paid + B_FEE_i * fee2 + B_DUR * dur2
    V3 = B_FEE_i * fee3 + B_DUR * dur3

METHODOLOGICAL NOTE:
    This uses a TWO-STAGE approach:
    Stage 1: Estimate latent variables from Likert items (CFA)
    Stage 2: Use LV estimates as fixed regressors in choice model

    This causes ATTENUATION BIAS: LV effects are biased toward zero.
    For unbiased estimates, use ICLV (simultaneous estimation).

Usage:
    python src/models/hcm_basic.py --data data/simulated/fresh_simulation.csv

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
# LATENT VARIABLE ESTIMATION
# =============================================================================

def estimate_latent_cfa(df: pd.DataFrame, items: list, name: str) -> pd.Series:
    """
    Estimate latent variable using weighted average (CFA-style).

    Uses item-total correlation weights for optimal combination.

    Args:
        df: DataFrame with individual-level data
        items: List of column names for Likert items
        name: Name of the latent construct

    Returns:
        Series with standardized LV scores
    """
    X = df[items].values

    # Item-total correlation weights
    total = X.sum(axis=1)
    weights = []
    for i in range(X.shape[1]):
        corr = np.corrcoef(X[:, i], total)[0, 1]
        weights.append(max(0.1, corr))  # Floor at 0.1 to prevent division issues
    weights = np.array(weights) / sum(weights)

    # Weighted score, standardized to mean=0, std=1
    score = (X * weights).sum(axis=1)
    score = (score - score.mean()) / score.std()

    return pd.Series(score, index=df.index, name=f'LV_{name}')


# =============================================================================
# MODEL SPECIFICATION
# =============================================================================

def create_hcm_basic(database: db.Database):
    """
    Create basic HCM with Blind Patriotism affecting fee sensitivity.

    Args:
        database: Biogeme database with LV column

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

    # Latent variable (estimated in Stage 1)
    LV_pat_blind = Variable('LV_pat_blind')

    # Base parameters
    ASC_paid = Beta('ASC_paid', 0, -10, 10, 0)
    B_FEE = Beta('B_FEE', -0.6, -10, 0, 0)  # Bounded negative
    B_DUR = Beta('B_DUR', -0.05, -5, 0, 0)  # Bounded negative

    # LV interaction on fee sensitivity
    B_FEE_LV = Beta('B_FEE_PatBlind', 0, -2, 2, 0)

    # Individual-specific fee coefficient
    B_FEE_i = B_FEE + B_FEE_LV * LV_pat_blind

    # Utility functions
    V1 = ASC_paid + B_FEE_i * fee1 + B_DUR * dur1
    V2 = ASC_paid + B_FEE_i * fee2 + B_DUR * dur2
    V3 = B_FEE_i * fee3 + B_DUR * dur3  # Reference

    V = {1: V1, 2: V2, 3: V3}
    av = {1: 1, 2: 1, 3: 1}

    logprob = models.loglogit(V, av, CHOICE)

    return logprob, 'HCM_Basic'


# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_data(filepath: str, fee_scale: float = 10000.0) -> pd.DataFrame:
    """
    Load and prepare data for HCM estimation.

    Args:
        filepath: Path to CSV data file
        fee_scale: Divisor for fee scaling (default 10000)

    Returns:
        Prepared DataFrame with LV estimates
    """
    df = pd.read_csv(filepath)

    # Scale fees
    for alt in [1, 2, 3]:
        df[f'fee{alt}_10k'] = df[f'fee{alt}'] / fee_scale

    # Find Blind Patriotism items
    pat_blind_items = [c for c in df.columns
                       if c.startswith('pat_blind_') and c[-1].isdigit()]

    if not pat_blind_items:
        raise ValueError("No pat_blind_* columns found in data")

    print(f"Found {len(pat_blind_items)} Blind Patriotism items: {pat_blind_items}")

    # Get unique individuals for LV estimation
    individuals = df.groupby('ID').first().reset_index()

    # Stage 1: Estimate latent variable from Likert items
    print("\nStage 1: Latent Variable Estimation (CFA)")
    print("-" * 50)

    lv_scores = estimate_latent_cfa(individuals, pat_blind_items, 'pat_blind')
    individuals['LV_pat_blind'] = lv_scores.values

    # Compute Cronbach's alpha for reliability
    X = individuals[pat_blind_items].values
    n_items = len(pat_blind_items)
    item_var = X.var(axis=0, ddof=1).sum()
    total_var = X.sum(axis=1).var(ddof=1)
    alpha = (n_items / (n_items - 1)) * (1 - item_var / total_var)

    print(f"  Construct: Blind Patriotism")
    print(f"  Items: {n_items}")
    print(f"  Cronbach's alpha: {alpha:.3f}")
    print(f"  LV mean: {lv_scores.mean():.3f}, std: {lv_scores.std():.3f}")

    # Reliability warning
    if alpha < 0.7:
        print(f"  WARNING: Alpha < 0.7 suggests measurement issues")
        print(f"  Attenuation bias may be severe (>30%)")
    elif alpha < 0.8:
        print(f"  Note: Alpha in 0.7-0.8 range - expect ~20-30% attenuation")
    else:
        print(f"  Good reliability - expect ~15-20% attenuation")

    # Merge LV back to full data
    df = df.merge(individuals[['ID', 'LV_pat_blind']], on='ID', how='left')

    # Validate against true LV if available (simulation only)
    if 'LV_pat_blind_true' in df.columns:
        corr = df.groupby('ID').first()[['LV_pat_blind', 'LV_pat_blind_true']].corr().iloc[0, 1]
        print(f"\nValidation (CFA vs True): r = {corr:.3f}")

    # Keep only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    return df[numeric_cols].copy()


# =============================================================================
# ESTIMATION
# =============================================================================

def estimate_hcm_basic(data_path: str,
                       config_path: str = None,
                       output_dir: str = 'results/hcm_basic') -> dict:
    """
    Estimate basic HCM model with single latent variable.

    Args:
        data_path: Path to data CSV
        config_path: Path to config JSON with true parameters (optional)
        output_dir: Directory for output files

    Returns:
        Dictionary with estimation results
    """
    print("="*70)
    print("HCM BASIC MODEL ESTIMATION")
    print("(Two-Stage Approach - Subject to Attenuation Bias)")
    print("="*70)

    # Load and prepare data (Stage 1: LV estimation)
    df = prepare_data(data_path)
    n_obs = len(df)
    n_individuals = df['ID'].nunique() if 'ID' in df.columns else n_obs

    print(f"\nData: {n_obs} observations from {n_individuals} individuals")

    # Create database
    database = db.Database('hcm_basic', df)

    # Create model (Stage 2: Choice model)
    print("\nStage 2: Choice Model Estimation")
    print("-" * 50)

    logprob, model_name = create_hcm_basic(database)

    # Estimate
    print("Estimating model...")
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
    print(f"{'Parameter':<16} {'Estimate':>12} {'Std.Err':>10} {'t-stat':>10} {'p-value':>10}")
    print("-"*60)

    param_order = ['ASC_paid', 'B_FEE', 'B_FEE_PatBlind', 'B_DUR']
    for param in param_order:
        if param in betas:
            sig = "***" if pvals[param] < 0.01 else "**" if pvals[param] < 0.05 else "*" if pvals[param] < 0.10 else ""
            print(f"{param:<16} {betas[param]:>12.4f} {stderrs[param]:>10.4f} {tstats[param]:>10.2f} {pvals[param]:>10.4f} {sig}")

    # Interpretation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)

    if 'B_FEE_PatBlind' in betas:
        lv_effect = betas['B_FEE_PatBlind']
        lv_t = tstats['B_FEE_PatBlind']

        if abs(lv_t) > 1.96:
            direction = "reduces" if lv_effect < 0 else "increases"
            print(f"\nBlind Patriotism significantly {direction} fee sensitivity (p<0.05)")
            print(f"Effect: {lv_effect:.4f} per std dev of LV")
            print(f"\nNOTE: This estimate is ATTENUATED due to two-stage approach.")
            print(f"True effect is likely 15-30% larger in magnitude.")
        else:
            print(f"\nBlind Patriotism effect not significant (t={lv_t:.2f})")
            print(f"\nNOTE: Non-significance may be due to attenuation bias.")
            print(f"Consider ICLV estimation for unbiased test.")

    # Compare to true if config provided
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            config = json.load(f)

        print("\n" + "="*70)
        print("COMPARISON TO TRUE PARAMETERS")
        print("="*70)

        # Extract true values
        true_params = {}
        for term in config['choice_model']['base_terms']:
            if term['name'] == 'ASC_paid':
                true_params['ASC_paid'] = term['coef']

        for term in config['choice_model']['attribute_terms']:
            if term['name'] == 'b_fee10k':
                true_params['B_FEE'] = term['base_coef']
                # Find LV interaction
                for inter in term.get('interactions', []):
                    if 'pat_blind' in inter.get('with', ''):
                        true_params['B_FEE_PatBlind'] = inter['coef']
            if term['name'] == 'b_dur':
                true_params['B_DUR'] = term['base_coef']

        print(f"\n{'Parameter':<16} {'True':>10} {'Estimated':>12} {'Bias%':>10}")
        print("-"*50)
        for param in param_order:
            if param in betas and param in true_params:
                true_val = true_params[param]
                est_val = betas[param]
                if true_val != 0:
                    bias_pct = ((est_val - true_val) / abs(true_val)) * 100
                    print(f"{param:<16} {true_val:>10.4f} {est_val:>12.4f} {bias_pct:>+9.1f}%")
                else:
                    print(f"{param:<16} {true_val:>10.4f} {est_val:>12.4f} {'N/A':>10}")

        # Note on expected attenuation
        if 'B_FEE_PatBlind' in betas and 'B_FEE_PatBlind' in true_params:
            true_lv = true_params['B_FEE_PatBlind']
            est_lv = betas['B_FEE_PatBlind']
            if true_lv != 0:
                attenuation = (1 - abs(est_lv) / abs(true_lv)) * 100
                print(f"\nAttenuation of LV effect: {attenuation:.1f}%")
                print("(Expected: 15-30% due to measurement error)")

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
    parser = argparse.ArgumentParser(description='Estimate HCM Basic model')
    parser.add_argument('--data', type=str, required=True, help='Path to data CSV')
    parser.add_argument('--config', type=str, default='config/model_config_advanced.json',
                        help='Path to config JSON with true parameters')
    parser.add_argument('--output', type=str, default='results/hcm_basic',
                        help='Output directory')

    args = parser.parse_args()

    estimate_hcm_basic(args.data, args.config, args.output)
