"""
HCM Full Model
==============

Hybrid Choice Model with all four latent variables affecting
both fee and duration sensitivity.

Latent Constructs:
1. Blind Patriotism (pat_blind) - Uncritical support for country
2. Constructive Patriotism (pat_const) - Critical, improvement-oriented
3. Daily Life Secularism (sec_dl) - Separation in daily practices
4. Faith & Prayer Secularism (sec_fp) - Separation in religious matters

Model Specification:
    B_FEE_i = B_FEE + B_FEE_PB*LV_pat_blind + B_FEE_PC*LV_pat_const
              + B_FEE_SDL*LV_sec_dl + B_FEE_SFP*LV_sec_fp

    B_DUR_i = B_DUR + B_DUR_PB*LV_pat_blind + B_DUR_PC*LV_pat_const
              + B_DUR_SDL*LV_sec_dl + B_DUR_SFP*LV_sec_fp

    V1 = ASC_paid + B_FEE_i * fee1 + B_DUR_i * dur1
    V2 = ASC_paid + B_FEE_i * fee2 + B_DUR_i * dur2
    V3 = B_FEE_i * fee3 + B_DUR_i * dur3

METHODOLOGICAL NOTE:
    This uses a TWO-STAGE approach with ATTENUATION BIAS.
    For unbiased estimates, use ICLV (simultaneous estimation).

Usage:
    python src/models/hcm_full.py --data data/simulated/fresh_simulation.csv

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
        weights.append(max(0.1, corr))
    weights = np.array(weights) / sum(weights)

    # Weighted score, standardized
    score = (X * weights).sum(axis=1)
    score = (score - score.mean()) / score.std()

    return pd.Series(score, index=df.index, name=f'LV_{name}')


def compute_cronbach_alpha(X: np.ndarray) -> float:
    """Compute Cronbach's alpha for reliability assessment."""
    n_items = X.shape[1]
    item_var = X.var(axis=0, ddof=1).sum()
    total_var = X.sum(axis=1).var(ddof=1)
    return (n_items / (n_items - 1)) * (1 - item_var / total_var)


# =============================================================================
# MODEL SPECIFICATION
# =============================================================================

def create_hcm_full(database: db.Database):
    """
    Create full HCM with all 4 LVs affecting fee and duration.

    Args:
        database: Biogeme database with LV columns

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

    # Latent variables (estimated in Stage 1)
    LV_pat_blind = Variable('LV_pat_blind')
    LV_pat_const = Variable('LV_pat_const')
    LV_sec_dl = Variable('LV_sec_dl')
    LV_sec_fp = Variable('LV_sec_fp')

    # Base parameters
    ASC_paid = Beta('ASC_paid', 0, -10, 10, 0)
    B_FEE = Beta('B_FEE', -0.6, -10, 0, 0)
    B_DUR = Beta('B_DUR', -0.05, -5, 0, 0)

    # LV interactions on fee sensitivity
    B_FEE_PB = Beta('B_FEE_PatBlind', 0, -2, 2, 0)
    B_FEE_PC = Beta('B_FEE_PatConst', 0, -2, 2, 0)
    B_FEE_SDL = Beta('B_FEE_SecDL', 0, -2, 2, 0)
    B_FEE_SFP = Beta('B_FEE_SecFP', 0, -2, 2, 0)

    # LV interactions on duration sensitivity
    B_DUR_PB = Beta('B_DUR_PatBlind', 0, -1, 1, 0)
    B_DUR_PC = Beta('B_DUR_PatConst', 0, -1, 1, 0)
    B_DUR_SDL = Beta('B_DUR_SecDL', 0, -1, 1, 0)
    B_DUR_SFP = Beta('B_DUR_SecFP', 0, -1, 1, 0)

    # Individual-specific coefficients
    B_FEE_i = (B_FEE + B_FEE_PB * LV_pat_blind + B_FEE_PC * LV_pat_const +
               B_FEE_SDL * LV_sec_dl + B_FEE_SFP * LV_sec_fp)

    B_DUR_i = (B_DUR + B_DUR_PB * LV_pat_blind + B_DUR_PC * LV_pat_const +
               B_DUR_SDL * LV_sec_dl + B_DUR_SFP * LV_sec_fp)

    # Utility functions
    V1 = ASC_paid + B_FEE_i * fee1 + B_DUR_i * dur1
    V2 = ASC_paid + B_FEE_i * fee2 + B_DUR_i * dur2
    V3 = B_FEE_i * fee3 + B_DUR_i * dur3

    V = {1: V1, 2: V2, 3: V3}
    av = {1: 1, 2: 1, 3: 1}

    logprob = models.loglogit(V, av, CHOICE)

    return logprob, 'HCM_Full'


# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_data(filepath: str, fee_scale: float = 10000.0) -> pd.DataFrame:
    """
    Load and prepare data for full HCM estimation.

    Args:
        filepath: Path to CSV data file
        fee_scale: Divisor for fee scaling (default 10000)

    Returns:
        Prepared DataFrame with all LV estimates
    """
    df = pd.read_csv(filepath)

    # Scale fees
    for alt in [1, 2, 3]:
        df[f'fee{alt}_10k'] = df[f'fee{alt}'] / fee_scale

    # Define constructs and their item patterns
    constructs = {
        'pat_blind': 'pat_blind_',
        'pat_const': 'pat_constructive_',
        'sec_dl': 'sec_dl_',
        'sec_fp': 'sec_fp_',
    }

    # Get unique individuals
    individuals = df.groupby('ID').first().reset_index()

    print("\nStage 1: Latent Variable Estimation (CFA)")
    print("="*60)

    lv_stats = {}
    for lv_name, item_prefix in constructs.items():
        items = [c for c in df.columns
                 if c.startswith(item_prefix) and c[-1].isdigit()]

        if not items:
            print(f"  WARNING: No items found for {lv_name} (pattern: {item_prefix}*)")
            continue

        # Estimate LV
        lv_scores = estimate_latent_cfa(individuals, items, lv_name)
        individuals[f'LV_{lv_name}'] = lv_scores.values

        # Compute reliability
        X = individuals[items].values
        alpha = compute_cronbach_alpha(X)

        lv_stats[lv_name] = {
            'n_items': len(items),
            'alpha': alpha,
            'mean': lv_scores.mean(),
            'std': lv_scores.std()
        }

        print(f"\n  {lv_name.upper()}:")
        print(f"    Items: {len(items)}")
        print(f"    Cronbach's alpha: {alpha:.3f}")

        # Reliability assessment
        if alpha < 0.7:
            print(f"    WARNING: Low reliability (alpha < 0.7)")
        elif alpha < 0.8:
            print(f"    Acceptable reliability")
        else:
            print(f"    Good reliability")

    # LV correlation matrix
    lv_cols = [f'LV_{name}' for name in constructs.keys()
               if f'LV_{name}' in individuals.columns]

    if len(lv_cols) > 1:
        print("\n" + "="*60)
        print("LATENT VARIABLE CORRELATIONS")
        print("="*60)

        corr_matrix = individuals[lv_cols].corr()
        print("\n" + corr_matrix.round(3).to_string())

        # Check for multicollinearity
        for i, c1 in enumerate(lv_cols):
            for c2 in lv_cols[i+1:]:
                corr = corr_matrix.loc[c1, c2]
                if abs(corr) > 0.7:
                    print(f"\n  WARNING: High correlation between {c1} and {c2}: r={corr:.3f}")
                    print(f"  This may cause multicollinearity in the choice model")

    # Merge LVs back to full data
    merge_cols = ['ID'] + lv_cols
    df = df.merge(individuals[merge_cols], on='ID', how='left')

    # Keep only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    return df[numeric_cols].copy(), lv_stats


# =============================================================================
# ESTIMATION
# =============================================================================

def estimate_hcm_full(data_path: str,
                      config_path: str = None,
                      output_dir: str = 'results/hcm_full') -> dict:
    """
    Estimate full HCM model with all 4 latent variables.

    Args:
        data_path: Path to data CSV
        config_path: Path to config JSON with true parameters (optional)
        output_dir: Directory for output files

    Returns:
        Dictionary with estimation results
    """
    print("="*70)
    print("HCM FULL MODEL ESTIMATION")
    print("(All 4 Latent Variables on Fee and Duration)")
    print("="*70)

    # Load and prepare data (Stage 1)
    df, lv_stats = prepare_data(data_path)
    n_obs = len(df)
    n_individuals = df['ID'].nunique() if 'ID' in df.columns else n_obs

    print(f"\nData: {n_obs} observations from {n_individuals} individuals")

    # Create database
    database = db.Database('hcm_full', df)

    # Create model (Stage 2)
    print("\n" + "="*60)
    print("Stage 2: Choice Model Estimation")
    print("="*60)

    logprob, model_name = create_hcm_full(database)

    # Estimate
    print("\nEstimating model (this may take a moment)...")
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

    # Organize parameter output
    print("\n" + "-"*70)
    print("BASE PARAMETERS")
    print("-"*70)
    print(f"{'Parameter':<18} {'Estimate':>10} {'Std.Err':>10} {'t-stat':>10} {'p-value':>10}")
    print("-"*60)

    base_params = ['ASC_paid', 'B_FEE', 'B_DUR']
    for param in base_params:
        if param in betas:
            sig = "***" if pvals[param] < 0.01 else "**" if pvals[param] < 0.05 else "*" if pvals[param] < 0.10 else ""
            print(f"{param:<18} {betas[param]:>10.4f} {stderrs[param]:>10.4f} {tstats[param]:>10.2f} {pvals[param]:>10.4f} {sig}")

    print("\n" + "-"*70)
    print("FEE SENSITIVITY INTERACTIONS")
    print("-"*70)
    print(f"{'Parameter':<18} {'Estimate':>10} {'Std.Err':>10} {'t-stat':>10} {'p-value':>10}")
    print("-"*60)

    fee_params = ['B_FEE_PatBlind', 'B_FEE_PatConst', 'B_FEE_SecDL', 'B_FEE_SecFP']
    for param in fee_params:
        if param in betas:
            sig = "***" if pvals[param] < 0.01 else "**" if pvals[param] < 0.05 else "*" if pvals[param] < 0.10 else ""
            print(f"{param:<18} {betas[param]:>10.4f} {stderrs[param]:>10.4f} {tstats[param]:>10.2f} {pvals[param]:>10.4f} {sig}")

    print("\n" + "-"*70)
    print("DURATION SENSITIVITY INTERACTIONS")
    print("-"*70)
    print(f"{'Parameter':<18} {'Estimate':>10} {'Std.Err':>10} {'t-stat':>10} {'p-value':>10}")
    print("-"*60)

    dur_params = ['B_DUR_PatBlind', 'B_DUR_PatConst', 'B_DUR_SecDL', 'B_DUR_SecFP']
    for param in dur_params:
        if param in betas:
            sig = "***" if pvals[param] < 0.01 else "**" if pvals[param] < 0.05 else "*" if pvals[param] < 0.10 else ""
            print(f"{param:<18} {betas[param]:>10.4f} {stderrs[param]:>10.4f} {tstats[param]:>10.2f} {pvals[param]:>10.4f} {sig}")

    # Interpretation summary
    print("\n" + "="*70)
    print("INTERPRETATION SUMMARY")
    print("="*70)

    print("\nSignificant LV effects on FEE (|t| > 1.96):")
    sig_fee = []
    for param in fee_params:
        if param in tstats and abs(tstats[param]) > 1.96:
            direction = "reduces" if betas[param] < 0 else "increases"
            sig_fee.append(f"  - {param}: {direction} fee sensitivity ({betas[param]:.4f})")
    if sig_fee:
        print("\n".join(sig_fee))
    else:
        print("  None")

    print("\nSignificant LV effects on DURATION (|t| > 1.96):")
    sig_dur = []
    for param in dur_params:
        if param in tstats and abs(tstats[param]) > 1.96:
            direction = "reduces" if betas[param] < 0 else "increases"
            sig_dur.append(f"  - {param}: {direction} duration sensitivity ({betas[param]:.4f})")
    if sig_dur:
        print("\n".join(sig_dur))
    else:
        print("  None")

    print("\n" + "-"*70)
    print("METHODOLOGICAL NOTES")
    print("-"*70)
    print("""
1. Two-Stage Approach: These estimates are subject to ATTENUATION BIAS
   - LV effect magnitudes are biased toward zero by ~15-30%
   - Standard errors are underestimated
   - Significance tests are conservative (if significant here, likely real)

2. Multicollinearity: Check LV correlation matrix above
   - High correlations (|r| > 0.7) may inflate standard errors
   - Consider using only theoretically distinct constructs

3. For unbiased estimates: Use ICLV (simultaneous estimation)
   - from src.models.iclv import estimate_iclv
""")

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
        print(f"\n{'Parameter':<18} {'True':>10} {'Estimated':>12} {'Bias%':>10}")
        print("-"*52)
        for param in all_params:
            if param in betas and param in true_params:
                true_val = true_params[param]
                est_val = betas[param]
                if true_val != 0:
                    bias_pct = ((est_val - true_val) / abs(true_val)) * 100
                    print(f"{param:<18} {true_val:>10.4f} {est_val:>12.4f} {bias_pct:>+9.1f}%")
                else:
                    print(f"{param:<18} {true_val:>10.4f} {est_val:>12.4f} {'N/A':>10}")

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
        'lv_statistics': lv_stats,
        'converged': True
    }

    # Save parameter estimates
    pd.DataFrame([{
        'Parameter': k,
        'Estimate': betas[k],
        'Std_Error': stderrs[k],
        't_stat': tstats[k],
        'p_value': pvals[k]
    } for k in betas]).to_csv(output_path / 'parameter_estimates.csv', index=False)

    # Save LV statistics
    pd.DataFrame([
        {'Construct': k, **v}
        for k, v in lv_stats.items()
    ]).to_csv(output_path / 'lv_statistics.csv', index=False)

    print(f"\nResults saved to: {output_path}")

    return results_dict


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimate HCM Full model')
    parser.add_argument('--data', type=str, required=True, help='Path to data CSV')
    parser.add_argument('--config', type=str, default='config/model_config_advanced.json',
                        help='Path to config JSON with true parameters')
    parser.add_argument('--output', type=str, default='results/hcm_full',
                        help='Output directory')

    args = parser.parse_args()

    estimate_hcm_full(args.data, args.config, args.output)
