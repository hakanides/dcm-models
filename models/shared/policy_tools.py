"""
Policy analysis tools for isolated DCM models.

Provides:
- WTP (Willingness to Pay) calculations
- Elasticity computations
- Market share predictions
- Marginal effects
- Welfare analysis (for complex models)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from scipy import stats


@dataclass
class EstimationResults:
    """Container for estimation results needed by policy tools."""
    betas: Dict[str, float]
    std_errs: Dict[str, float]
    cov_matrix: Optional[np.ndarray] = None
    param_names: Optional[List[str]] = None
    n_obs: int = 0
    log_likelihood: float = 0.0
    model_type: str = 'MNL'


def extract_results(biogeme_results, model_type: str = 'MNL') -> EstimationResults:
    """
    Extract estimation results from Biogeme results object.

    Args:
        biogeme_results: Biogeme estimation results
        model_type: Type of model (MNL, MXL, HCM, ICLV)

    Returns:
        EstimationResults object
    """
    # Get parameter estimates - try different API versions
    try:
        betas = biogeme_results.get_beta_values()
    except AttributeError:
        # Alternative for different Biogeme versions
        estimates_df = biogeme_results.get_estimated_parameters()
        if 'Name' in estimates_df.columns:
            estimates_df = estimates_df.set_index('Name')
        betas = estimates_df['Value'].to_dict()

    # Get standard errors - try different API versions
    try:
        std_errs = biogeme_results.get_stdErr()
    except AttributeError:
        try:
            estimates_df = biogeme_results.get_estimated_parameters()
            if 'Name' in estimates_df.columns:
                estimates_df = estimates_df.set_index('Name')
            # Try robust std errors first
            if 'Robust std err.' in estimates_df.columns:
                std_errs = estimates_df['Robust std err.'].to_dict()
            elif 'Std err' in estimates_df.columns:
                std_errs = estimates_df['Std err'].to_dict()
            else:
                std_errs = {k: 0.0 for k in betas.keys()}
        except Exception:
            std_errs = {k: 0.0 for k in betas.keys()}

    # Get covariance matrix if available
    try:
        cov = biogeme_results.variance_covariance_matrix
        param_names = list(betas.keys())
    except Exception:
        cov = None
        param_names = list(betas.keys())

    # Get general statistics
    try:
        gen_stats = biogeme_results.get_general_statistics()
        n_obs = 0
        ll = 0
        for key, val in gen_stats.items():
            if 'Number of observations' in key:
                n_obs = int(val[0]) if isinstance(val, tuple) else int(val)
            elif 'Final log likelihood' in key:
                ll = float(val[0]) if isinstance(val, tuple) else float(val)
    except Exception:
        n_obs = 0
        ll = 0

    return EstimationResults(
        betas=betas,
        std_errs=std_errs,
        cov_matrix=cov,
        param_names=param_names,
        n_obs=n_obs,
        log_likelihood=ll,
        model_type=model_type
    )


# ============================================================================
# WTP CALCULATIONS
# ============================================================================

def compute_wtp(
    results: EstimationResults,
    config: Dict[str, Any],
    numerator_param: str = 'B_DUR',
    denominator_param: str = 'B_FEE'
) -> pd.DataFrame:
    """
    Compute Willingness to Pay using delta method.

    WTP = -β_numerator / β_denominator

    For duration: WTP = -B_DUR / B_FEE = TL per day saved

    Args:
        results: Estimation results
        config: Model configuration
        numerator_param: Parameter in numerator (e.g., B_DUR)
        denominator_param: Parameter in denominator (e.g., B_FEE)

    Returns:
        DataFrame with WTP estimates and confidence intervals
    """
    betas = results.betas
    std_errs = results.std_errs

    # Get fee scale from config
    fee_scale = config.get('choice_model', {}).get('fee_scale', 10000.0)

    # Check parameters exist
    if numerator_param not in betas or denominator_param not in betas:
        available = list(betas.keys())
        raise ValueError(
            f"Parameters not found. Available: {available}. "
            f"Requested: {numerator_param}, {denominator_param}"
        )

    # Get parameter values
    beta_num = betas[numerator_param]
    beta_den = betas[denominator_param]
    se_num = std_errs[numerator_param]
    se_den = std_errs[denominator_param]

    # Compute WTP (negative ratio for typical utility specification)
    # WTP for duration: -B_DUR / B_FEE
    # If B_FEE is scaled (per 10k TL), need to adjust
    wtp_point = -beta_num / beta_den * fee_scale

    # Delta method for standard error
    # WTP = -β_num / β_den
    # Var(WTP) ≈ (∂WTP/∂β_num)² * Var(β_num) + (∂WTP/∂β_den)² * Var(β_den)
    # ∂WTP/∂β_num = -1/β_den
    # ∂WTP/∂β_den = β_num/β_den²

    grad_num = -1 / beta_den * fee_scale
    grad_den = beta_num / (beta_den ** 2) * fee_scale

    # If covariance available, use full formula
    if results.cov_matrix is not None and results.param_names is not None:
        try:
            idx_num = results.param_names.index(numerator_param)
            idx_den = results.param_names.index(denominator_param)
            var_num = results.cov_matrix[idx_num, idx_num]
            var_den = results.cov_matrix[idx_den, idx_den]
            cov_nd = results.cov_matrix[idx_num, idx_den]

            wtp_var = (grad_num ** 2 * var_num +
                       grad_den ** 2 * var_den +
                       2 * grad_num * grad_den * cov_nd)
        except Exception:
            wtp_var = grad_num ** 2 * se_num ** 2 + grad_den ** 2 * se_den ** 2
    else:
        wtp_var = grad_num ** 2 * se_num ** 2 + grad_den ** 2 * se_den ** 2

    wtp_se = np.sqrt(max(0, wtp_var))

    # 95% confidence interval
    z = 1.96
    ci_lower = wtp_point - z * wtp_se
    ci_upper = wtp_point + z * wtp_se

    # t-statistic
    t_stat = wtp_point / wtp_se if wtp_se > 0 else np.nan

    result_df = pd.DataFrame([{
        'Attribute': numerator_param.replace('B_', '').lower(),
        'WTP_TL': wtp_point,
        'SE': wtp_se,
        't_stat': t_stat,
        'CI_95_Lower': ci_lower,
        'CI_95_Upper': ci_upper,
        'Fee_Scale': fee_scale
    }])

    return result_df


def compute_all_wtp(
    results: EstimationResults,
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Compute WTP for all attributes relative to fee.

    Args:
        results: Estimation results
        config: Model configuration

    Returns:
        DataFrame with WTP for each attribute
    """
    betas = results.betas

    # Handle different model types - MXL uses B_FEE_MU instead of B_FEE
    if 'B_FEE' in betas:
        fee_param = 'B_FEE'
    elif 'B_FEE_MU' in betas:
        fee_param = 'B_FEE_MU'
    else:
        return pd.DataFrame()

    # Find all non-fee coefficients
    wtp_list = []

    # Duration WTP
    if 'B_DUR' in betas:
        wtp_dur = compute_wtp(results, config, 'B_DUR', fee_param)
        wtp_list.append(wtp_dur)

    # LV interaction effects (for HCM/ICLV)
    lv_params = [k for k in betas.keys() if 'B_FEE_' in k and k != fee_param]
    for param in lv_params:
        try:
            # For interaction: WTP change per unit LV
            wtp_lv = compute_wtp(results, config, param, fee_param)
            wtp_lv['Attribute'] = param.replace('B_FEE_', '').lower() + '_effect'
            wtp_list.append(wtp_lv)
        except Exception:
            pass

    if wtp_list:
        return pd.concat(wtp_list, ignore_index=True)
    return pd.DataFrame()


# ============================================================================
# ELASTICITY CALCULATIONS
# ============================================================================

def compute_logit_probabilities(utilities: np.ndarray) -> np.ndarray:
    """Compute multinomial logit probabilities."""
    exp_u = np.exp(utilities - np.max(utilities))  # Numerical stability
    return exp_u / np.sum(exp_u)


def compute_elasticity_matrix(
    results: EstimationResults,
    df: pd.DataFrame,
    config: Dict[str, Any],
    attribute: str = 'fee'
) -> pd.DataFrame:
    """
    Compute own and cross-price elasticity matrix.

    Own-price elasticity: η_jj = β * x_j * (1 - P_j)
    Cross-price elasticity: η_jk = -β * x_k * P_k

    Args:
        results: Estimation results
        df: Data with attribute values
        config: Model configuration
        attribute: Attribute for elasticity ('fee' or 'dur')

    Returns:
        DataFrame with J×J elasticity matrix
    """
    betas = results.betas
    fee_scale = config.get('choice_model', {}).get('fee_scale', 10000.0)

    # Get coefficient - handle MXL which uses B_FEE_MU instead of B_FEE
    if attribute == 'fee':
        # Check for B_FEE or B_FEE_MU (MXL model)
        if 'B_FEE' in betas:
            coef_name = 'B_FEE'
        elif 'B_FEE_MU' in betas:
            coef_name = 'B_FEE_MU'
        else:
            return pd.DataFrame()
        cols = ['fee1', 'fee2', 'fee3']
        # Scale coefficient if fee is in units of fee_scale
        coef = betas.get(coef_name, 0) / fee_scale
    else:
        coef_name = 'B_DUR'
        cols = ['dur1', 'dur2', 'dur3']
        coef = betas.get(coef_name, 0)
        if coef_name not in betas:
            return pd.DataFrame()

    # Get mean attribute values
    mean_attrs = []
    for col in cols:
        if col in df.columns:
            mean_attrs.append(df[col].mean())
        else:
            mean_attrs.append(0)

    mean_attrs = np.array(mean_attrs)
    n_alts = len(mean_attrs)

    # Compute utilities at mean values (simplified)
    asc = betas.get('ASC_paid', 0)
    utilities = np.zeros(n_alts)

    # Simplified utility: V_j = ASC + β*x_j
    for j in range(n_alts):
        if j < 2:  # Paid alternatives
            utilities[j] = asc + coef * mean_attrs[j]
        else:
            utilities[j] = coef * mean_attrs[j]

    # Compute choice probabilities
    probs = compute_logit_probabilities(utilities)

    # Compute elasticity matrix
    elasticity_matrix = np.zeros((n_alts, n_alts))

    for j in range(n_alts):
        for k in range(n_alts):
            if j == k:
                # Own elasticity
                elasticity_matrix[j, k] = coef * mean_attrs[j] * (1 - probs[j])
            else:
                # Cross elasticity
                elasticity_matrix[j, k] = -coef * mean_attrs[k] * probs[k]

    # Create DataFrame
    alt_names = ['Alt_1', 'Alt_2', 'Alt_3'][:n_alts]
    result_df = pd.DataFrame(
        elasticity_matrix,
        index=alt_names,
        columns=[f'd{attribute[0]}{i+1}' for i in range(n_alts)]  # dfee1, dfee2, dfee3
    )
    result_df.index.name = 'Alternative'

    return result_df


# ============================================================================
# MARKET SHARE PREDICTIONS
# ============================================================================

def compute_market_shares(
    results: EstimationResults,
    df: pd.DataFrame,
    config: Dict[str, Any],
    scenario_name: str = 'Sample Mean'
) -> pd.DataFrame:
    """
    Compute predicted market shares at sample mean attribute values.

    Args:
        results: Estimation results
        df: Data with attribute values
        config: Model configuration
        scenario_name: Name for this scenario

    Returns:
        DataFrame with predicted market shares
    """
    betas = results.betas
    fee_scale = config.get('choice_model', {}).get('fee_scale', 10000.0)

    # Get mean attribute values
    fee_means = [df[f'fee{i+1}'].mean() / fee_scale for i in range(3)
                 if f'fee{i+1}' in df.columns]
    dur_means = [df[f'dur{i+1}'].mean() for i in range(3)
                 if f'dur{i+1}' in df.columns]

    n_alts = len(fee_means)

    # Compute utilities - handle MXL which uses B_FEE_MU instead of B_FEE
    asc = betas.get('ASC_paid', 0)
    b_fee = betas.get('B_FEE', betas.get('B_FEE_MU', 0))
    b_dur = betas.get('B_DUR', 0)

    utilities = np.zeros(n_alts)
    for j in range(n_alts):
        if j < 2:  # Paid alternatives
            utilities[j] = asc + b_fee * fee_means[j] + b_dur * dur_means[j]
        else:  # Standard alternative
            utilities[j] = b_dur * dur_means[j]

    # Compute probabilities
    probs = compute_logit_probabilities(utilities)

    # Observed shares
    if 'CHOICE' in df.columns:
        obs_counts = df['CHOICE'].value_counts().sort_index()
        obs_shares = obs_counts / len(df)
        obs_shares_list = [obs_shares.get(i+1, 0) for i in range(n_alts)]
    else:
        obs_shares_list = [np.nan] * n_alts

    # Create result
    rows = []
    for j in range(n_alts):
        rows.append({
            'Alternative': j + 1,
            'Predicted_Share': probs[j],
            'Observed_Share': obs_shares_list[j],
            'Difference': probs[j] - obs_shares_list[j] if not np.isnan(obs_shares_list[j]) else np.nan,
            'Scenario': scenario_name
        })

    return pd.DataFrame(rows)


# ============================================================================
# MARGINAL EFFECTS
# ============================================================================

def compute_marginal_effects(
    results: EstimationResults,
    df: pd.DataFrame,
    config: Dict[str, Any],
    attribute: str = 'fee'
) -> pd.DataFrame:
    """
    Compute marginal effects of attribute on choice probabilities.

    Marginal effect: ∂P_j/∂x_j = β * P_j * (1 - P_j)

    Args:
        results: Estimation results
        df: Data with attribute values
        config: Model configuration
        attribute: Attribute ('fee' or 'dur')

    Returns:
        DataFrame with marginal effects
    """
    betas = results.betas
    fee_scale = config.get('choice_model', {}).get('fee_scale', 10000.0)

    # Get coefficient - handle MXL which uses B_FEE_MU instead of B_FEE
    if attribute == 'fee':
        if 'B_FEE' in betas:
            coef_name = 'B_FEE'
        elif 'B_FEE_MU' in betas:
            coef_name = 'B_FEE_MU'
        else:
            return pd.DataFrame()
        coef = betas.get(coef_name, 0) / fee_scale  # Effect per TL
    else:
        coef_name = 'B_DUR'
        coef = betas.get(coef_name, 0)
        if coef_name not in betas:
            return pd.DataFrame()

    # Get market shares at mean
    shares_df = compute_market_shares(results, df, config)
    probs = shares_df['Predicted_Share'].values

    # Compute marginal effects
    rows = []
    n_alts = len(probs)

    for j in range(n_alts):
        # Own effect: ∂P_j/∂x_j = β * P_j * (1 - P_j)
        own_effect = coef * probs[j] * (1 - probs[j])

        # Cross effects: ∂P_j/∂x_k = -β * P_j * P_k (for k ≠ j)
        cross_effects = [-coef * probs[j] * probs[k] for k in range(n_alts) if k != j]

        rows.append({
            'Alternative': j + 1,
            'Attribute': attribute,
            'Own_Effect': own_effect,
            'Cross_Effects_Sum': sum(cross_effects),
            'Probability': probs[j]
        })

    return pd.DataFrame(rows)


# ============================================================================
# SEGMENT ANALYSIS (HCM/ICLV)
# ============================================================================

def compute_wtp_by_segment(
    results: EstimationResults,
    df: pd.DataFrame,
    config: Dict[str, Any],
    segment_col: str = 'age_idx',
    n_segments: int = 3
) -> pd.DataFrame:
    """
    Compute WTP by demographic or LV segments.

    For models with interactions (e.g., B_FEE_AGE), computes segment-specific WTP.

    Args:
        results: Estimation results
        df: Data with segment variable
        config: Model configuration
        segment_col: Column to segment by
        n_segments: Number of segments (terciles)

    Returns:
        DataFrame with WTP by segment
    """
    betas = results.betas
    fee_scale = config.get('choice_model', {}).get('fee_scale', 10000.0)

    if segment_col not in df.columns:
        return pd.DataFrame()

    # Get base parameters
    b_fee = betas.get('B_FEE', 0)
    b_dur = betas.get('B_DUR', 0)

    # Check for interaction term
    interaction_param = f'B_FEE_{segment_col.upper()}'
    b_interaction = betas.get(interaction_param, 0)

    # Create segments
    df_indiv = df.drop_duplicates(subset=['ID'])
    segment_values = df_indiv[segment_col]

    # Define segment boundaries
    if n_segments == 3:
        low_thresh = segment_values.quantile(0.33)
        high_thresh = segment_values.quantile(0.67)
        segments = ['Low', 'Medium', 'High']
        conditions = [
            segment_values <= low_thresh,
            (segment_values > low_thresh) & (segment_values <= high_thresh),
            segment_values > high_thresh
        ]
    else:
        # Binary split
        median = segment_values.median()
        segments = ['Below Median', 'Above Median']
        conditions = [segment_values <= median, segment_values > median]

    # Compute WTP for each segment
    rows = []
    for i, (condition, seg_name) in enumerate(zip(conditions, segments)):
        seg_mean = segment_values[condition].mean()

        # Effective fee coefficient with interaction
        effective_b_fee = b_fee + b_interaction * seg_mean

        # WTP = -B_DUR / effective_B_FEE
        if effective_b_fee != 0:
            wtp = -b_dur / effective_b_fee * fee_scale
        else:
            wtp = np.nan

        rows.append({
            'Segment': seg_name,
            'Segment_Variable': segment_col,
            'Segment_Mean': seg_mean,
            'N_Individuals': condition.sum(),
            'Effective_B_FEE': effective_b_fee,
            'WTP_Duration_TL': wtp
        })

    return pd.DataFrame(rows)


# ============================================================================
# COMPENSATING VARIATION / CONSUMER SURPLUS
# ============================================================================

def compute_compensating_variation(
    results: EstimationResults,
    df: pd.DataFrame,
    config: Dict[str, Any],
    scenario_changes: Dict[str, float] = None,
    n_bootstrap: int = 1000
) -> pd.DataFrame:
    """
    Compute Compensating Variation (CV) for policy scenarios.

    CV measures the monetary equivalent of a policy change:
    CV = -1/β_cost × ln[Σ exp(V_j^new) / Σ exp(V_j^old)]

    This is the amount of money that would need to be taken from (given to)
    consumers to leave them as well off as before the policy change.

    Args:
        results: Estimation results
        df: Data with attribute values
        config: Model configuration
        scenario_changes: Dict of attribute changes, e.g., {'fee1': -5000, 'dur1': -5}
        n_bootstrap: Number of bootstrap samples for SE estimation

    Returns:
        DataFrame with CV estimates and confidence intervals
    """
    betas = results.betas
    std_errs = results.std_errs
    fee_scale = config.get('choice_model', {}).get('fee_scale', 10000.0)

    # Get cost coefficient - handle MXL
    if 'B_FEE' in betas:
        b_fee = betas['B_FEE']
        se_fee = std_errs.get('B_FEE', 0)
    elif 'B_FEE_MU' in betas:
        b_fee = betas['B_FEE_MU']
        se_fee = std_errs.get('B_FEE_MU', 0)
    else:
        return pd.DataFrame()

    b_dur = betas.get('B_DUR', 0)
    asc = betas.get('ASC_paid', 0)

    # Get mean attribute values
    fee_means = [df[f'fee{i+1}'].mean() for i in range(3) if f'fee{i+1}' in df.columns]
    dur_means = [df[f'dur{i+1}'].mean() for i in range(3) if f'dur{i+1}' in df.columns]
    n_alts = len(fee_means)

    # Default scenario: 10% fee reduction on Alt 1
    if scenario_changes is None:
        scenario_changes = {'fee1': -fee_means[0] * 0.10}

    def compute_logsum(fee_vals, dur_vals, b_fee_val):
        """Compute log-sum (expected max utility)."""
        utilities = np.zeros(n_alts)
        for j in range(n_alts):
            if j < 2:  # Paid alternatives
                utilities[j] = asc + b_fee_val * fee_vals[j] / fee_scale + b_dur * dur_vals[j]
            else:
                utilities[j] = b_dur * dur_vals[j]
        return np.log(np.sum(np.exp(utilities - np.max(utilities)))) + np.max(utilities)

    # Baseline log-sum
    logsum_base = compute_logsum(fee_means, dur_means, b_fee)

    # New scenario values
    fee_new = fee_means.copy()
    dur_new = dur_means.copy()

    for key, change in scenario_changes.items():
        if key.startswith('fee'):
            idx = int(key[3]) - 1
            if idx < len(fee_new):
                fee_new[idx] += change
        elif key.startswith('dur'):
            idx = int(key[3]) - 1
            if idx < len(dur_new):
                dur_new[idx] += change

    # New log-sum
    logsum_new = compute_logsum(fee_new, dur_new, b_fee)

    # Compensating Variation: CV = -(1/β_fee) × (logsum_new - logsum_base) × fee_scale
    # Positive CV = consumer gains (willing to pay this much)
    # Negative CV = consumer loses (needs this much compensation)
    cv_point = -(logsum_new - logsum_base) / b_fee * fee_scale

    # Bootstrap for SE estimation
    rng = np.random.default_rng(42)
    cv_bootstrap = []

    for _ in range(n_bootstrap):
        # Sample coefficient with noise
        b_fee_boot = rng.normal(b_fee, abs(se_fee))
        if b_fee_boot >= 0:  # Skip invalid samples (fee coef must be negative)
            continue

        ls_base_boot = compute_logsum(fee_means, dur_means, b_fee_boot)
        ls_new_boot = compute_logsum(fee_new, dur_new, b_fee_boot)
        cv_boot = -(ls_new_boot - ls_base_boot) / b_fee_boot * fee_scale
        cv_bootstrap.append(cv_boot)

    if len(cv_bootstrap) > 10:
        cv_se = np.std(cv_bootstrap)
        cv_ci_lower = np.percentile(cv_bootstrap, 2.5)
        cv_ci_upper = np.percentile(cv_bootstrap, 97.5)
    else:
        cv_se = np.nan
        cv_ci_lower = np.nan
        cv_ci_upper = np.nan

    # Consumer Surplus change (aggregate over sample)
    n_individuals = df['ID'].nunique() if 'ID' in df.columns else len(df)
    total_cs_change = cv_point * n_individuals

    # Format scenario description
    scenario_desc = ', '.join([f"{k}: {v:+.0f}" for k, v in scenario_changes.items()])

    result_df = pd.DataFrame([{
        'Scenario': scenario_desc,
        'CV_per_person_TL': cv_point,
        'CV_SE': cv_se,
        'CV_CI_Lower': cv_ci_lower,
        'CV_CI_Upper': cv_ci_upper,
        'Total_CS_Change_TL': total_cs_change,
        'N_Individuals': n_individuals,
        'Logsum_Base': logsum_base,
        'Logsum_New': logsum_new
    }])

    return result_df


def compute_consumer_surplus_scenarios(
    results: EstimationResults,
    df: pd.DataFrame,
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Compute CV for multiple standard policy scenarios.

    Scenarios:
    1. 10% fee reduction on Alt 1
    2. 20% fee reduction on Alt 1
    3. 5 day duration reduction on Alt 1
    4. 10 day duration reduction on Alt 1
    5. Combined: 10% fee + 5 day duration on Alt 1

    Args:
        results: Estimation results
        df: Data with attribute values
        config: Model configuration

    Returns:
        DataFrame with CV for each scenario
    """
    # Get baseline values
    fee1_mean = df['fee1'].mean() if 'fee1' in df.columns else 50000

    scenarios = [
        {'fee1': -fee1_mean * 0.10},  # 10% fee reduction
        {'fee1': -fee1_mean * 0.20},  # 20% fee reduction
        {'dur1': -5},                  # 5 day reduction
        {'dur1': -10},                 # 10 day reduction
        {'fee1': -fee1_mean * 0.10, 'dur1': -5},  # Combined
    ]

    results_list = []
    for scenario in scenarios:
        try:
            cv_df = compute_compensating_variation(results, df, config, scenario)
            results_list.append(cv_df)
        except Exception:
            pass

    if results_list:
        return pd.concat(results_list, ignore_index=True)
    return pd.DataFrame()


# ============================================================================
# WTP DISTRIBUTION FOR MXL
# ============================================================================

def compute_wtp_distribution(
    results: EstimationResults,
    config: Dict[str, Any],
    n_draws: int = 10000
) -> pd.DataFrame:
    """
    Compute WTP distribution for mixed logit models with random coefficients.

    For MXL with B_FEE ~ N(μ, σ²):
    - Simulates individual-level coefficients
    - Computes WTP = -B_DUR / B_FEE_i for each individual
    - Reports distribution statistics

    Args:
        results: Estimation results (must include B_FEE_MU, B_FEE_SIGMA)
        config: Model configuration
        n_draws: Number of simulation draws

    Returns:
        DataFrame with WTP distribution statistics
    """
    betas = results.betas
    std_errs = results.std_errs
    fee_scale = config.get('choice_model', {}).get('fee_scale', 10000.0)

    # Check if this is an MXL model
    if 'B_FEE_MU' not in betas or 'B_FEE_SIGMA' not in betas:
        return pd.DataFrame()

    mu_fee = betas['B_FEE_MU']
    sigma_fee = abs(betas['B_FEE_SIGMA'])  # Ensure positive
    b_dur = betas.get('B_DUR', 0)

    if sigma_fee < 0.0001:
        # No heterogeneity - return point estimate
        return pd.DataFrame()

    # Simulate individual coefficients
    rng = np.random.default_rng(42)
    b_fee_draws = rng.normal(mu_fee, sigma_fee, n_draws)

    # Compute WTP for each draw (exclude draws where B_FEE >= 0)
    valid_mask = b_fee_draws < 0
    wtp_draws = np.where(valid_mask, -b_dur / b_fee_draws * fee_scale, np.nan)
    wtp_valid = wtp_draws[~np.isnan(wtp_draws)]

    # Handle cases where WTP is extreme (e.g., B_FEE close to 0)
    wtp_trimmed = wtp_valid[np.abs(wtp_valid) < np.percentile(np.abs(wtp_valid), 99)]

    # Compute statistics
    pct_positive_fee = (b_fee_draws >= 0).mean() * 100  # % with wrong sign
    pct_valid = valid_mask.mean() * 100

    # Percentiles
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    pct_values = np.percentile(wtp_trimmed, percentiles) if len(wtp_trimmed) > 0 else [np.nan] * len(percentiles)

    summary_stats = {
        'Statistic': 'WTP_Duration',
        'Mean_WTP_TL': np.mean(wtp_trimmed) if len(wtp_trimmed) > 0 else np.nan,
        'Std_WTP_TL': np.std(wtp_trimmed) if len(wtp_trimmed) > 0 else np.nan,
        'Median_WTP_TL': np.median(wtp_trimmed) if len(wtp_trimmed) > 0 else np.nan,
        'Pct_5': pct_values[0],
        'Pct_10': pct_values[1],
        'Pct_25': pct_values[2],
        'Pct_50': pct_values[3],
        'Pct_75': pct_values[4],
        'Pct_90': pct_values[5],
        'Pct_95': pct_values[6],
        'Pct_Positive_Fee_Coef': pct_positive_fee,
        'Pct_Valid_WTP': pct_valid,
        'N_Draws': n_draws,
        'B_FEE_MU': mu_fee,
        'B_FEE_SIGMA': sigma_fee,
        'B_DUR': b_dur
    }

    return pd.DataFrame([summary_stats])


def compute_wtp_distribution_detailed(
    results: EstimationResults,
    config: Dict[str, Any],
    n_draws: int = 10000
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Compute detailed WTP distribution with individual draws.

    Returns both summary statistics and the raw draws for plotting.

    Args:
        results: Estimation results
        config: Model configuration
        n_draws: Number of simulation draws

    Returns:
        Tuple of (summary DataFrame, WTP draws array)
    """
    betas = results.betas
    fee_scale = config.get('choice_model', {}).get('fee_scale', 10000.0)

    if 'B_FEE_MU' not in betas or 'B_FEE_SIGMA' not in betas:
        return pd.DataFrame(), np.array([])

    mu_fee = betas['B_FEE_MU']
    sigma_fee = abs(betas['B_FEE_SIGMA'])
    b_dur = betas.get('B_DUR', 0)

    rng = np.random.default_rng(42)
    b_fee_draws = rng.normal(mu_fee, sigma_fee, n_draws)

    # WTP for valid draws only
    valid_mask = b_fee_draws < 0
    wtp_draws = np.where(valid_mask, -b_dur / b_fee_draws * fee_scale, np.nan)

    summary = compute_wtp_distribution(results, config, n_draws)

    return summary, wtp_draws[valid_mask]


# ============================================================================
# ELASTICITY STANDARD ERRORS
# ============================================================================

def compute_elasticity_with_se(
    results: EstimationResults,
    df: pd.DataFrame,
    config: Dict[str, Any],
    attribute: str = 'fee',
    n_bootstrap: int = 1000
) -> pd.DataFrame:
    """
    Compute elasticity matrix with standard errors using delta method and bootstrap.

    Own-price elasticity: η_jj = β × x_j × (1 - P_j)
    Cross-price elasticity: η_jk = -β × x_k × P_k

    SE computed via bootstrap over parameter uncertainty.

    Args:
        results: Estimation results
        df: Data with attribute values
        config: Model configuration
        attribute: Attribute for elasticity ('fee' or 'dur')
        n_bootstrap: Number of bootstrap samples

    Returns:
        DataFrame with elasticity estimates and standard errors
    """
    betas = results.betas
    std_errs = results.std_errs
    fee_scale = config.get('choice_model', {}).get('fee_scale', 10000.0)

    # Get coefficient and SE
    if attribute == 'fee':
        if 'B_FEE' in betas:
            coef_name = 'B_FEE'
        elif 'B_FEE_MU' in betas:
            coef_name = 'B_FEE_MU'
        else:
            return pd.DataFrame()
        coef = betas[coef_name] / fee_scale
        coef_se = std_errs.get(coef_name, 0) / fee_scale
        cols = ['fee1', 'fee2', 'fee3']
    else:
        coef_name = 'B_DUR'
        if coef_name not in betas:
            return pd.DataFrame()
        coef = betas[coef_name]
        coef_se = std_errs.get(coef_name, 0)
        cols = ['dur1', 'dur2', 'dur3']

    # Get mean attribute values
    mean_attrs = np.array([df[col].mean() if col in df.columns else 0 for col in cols])
    n_alts = len(mean_attrs)

    # Get ASC and its SE
    asc = betas.get('ASC_paid', 0)
    asc_se = std_errs.get('ASC_paid', 0)

    def compute_elasticities_for_coef(coef_val, asc_val):
        """Compute elasticity matrix for given coefficient values."""
        # Compute utilities
        utilities = np.zeros(n_alts)
        for j in range(n_alts):
            if j < 2:
                utilities[j] = asc_val + coef_val * mean_attrs[j]
            else:
                utilities[j] = coef_val * mean_attrs[j]

        # Probabilities
        exp_u = np.exp(utilities - np.max(utilities))
        probs = exp_u / np.sum(exp_u)

        # Elasticity matrix
        elast_matrix = np.zeros((n_alts, n_alts))
        for j in range(n_alts):
            for k in range(n_alts):
                if j == k:
                    elast_matrix[j, k] = coef_val * mean_attrs[j] * (1 - probs[j])
                else:
                    elast_matrix[j, k] = -coef_val * mean_attrs[k] * probs[k]

        return elast_matrix, probs

    # Point estimates
    elast_point, probs_point = compute_elasticities_for_coef(coef, asc)

    # Bootstrap for SEs
    rng = np.random.default_rng(42)
    elast_bootstrap = np.zeros((n_bootstrap, n_alts, n_alts))

    for b in range(n_bootstrap):
        coef_boot = rng.normal(coef, abs(coef_se))
        asc_boot = rng.normal(asc, abs(asc_se))
        elast_boot, _ = compute_elasticities_for_coef(coef_boot, asc_boot)
        elast_bootstrap[b] = elast_boot

    # Compute SEs
    elast_se = np.std(elast_bootstrap, axis=0)

    # Compute t-statistics and CIs
    t_stats = elast_point / np.where(elast_se > 0, elast_se, np.nan)
    ci_lower = elast_point - 1.96 * elast_se
    ci_upper = elast_point + 1.96 * elast_se

    # Create results DataFrame
    alt_names = ['Alt_1', 'Alt_2', 'Alt_3'][:n_alts]
    rows = []

    for j in range(n_alts):
        for k in range(n_alts):
            elast_type = 'Own' if j == k else 'Cross'
            rows.append({
                'Alternative': alt_names[j],
                'WRT_Attribute': f'{attribute}{k+1}',
                'Elasticity_Type': elast_type,
                'Elasticity': elast_point[j, k],
                'SE': elast_se[j, k],
                't_stat': t_stats[j, k],
                'CI_95_Lower': ci_lower[j, k],
                'CI_95_Upper': ci_upper[j, k],
                'Probability': probs_point[j]
            })

    return pd.DataFrame(rows)


def compute_aggregate_elasticities(
    results: EstimationResults,
    df: pd.DataFrame,
    config: Dict[str, Any],
    n_bootstrap: int = 1000
) -> pd.DataFrame:
    """
    Compute aggregate (market-level) elasticities with SEs.

    Aggregate elasticity = Σ_i (w_i × η_i)
    where w_i is the probability-weighted contribution of each alternative.

    Args:
        results: Estimation results
        df: Data with attribute values
        config: Model configuration
        n_bootstrap: Number of bootstrap samples

    Returns:
        DataFrame with aggregate elasticity estimates
    """
    # Get detailed elasticities
    elast_df = compute_elasticity_with_se(results, df, config, 'fee', n_bootstrap)

    if elast_df.empty:
        return pd.DataFrame()

    # Aggregate: probability-weighted average of own-price elasticities
    own_elast = elast_df[elast_df['Elasticity_Type'] == 'Own']

    if own_elast.empty:
        return pd.DataFrame()

    # Weighted average by choice probability
    total_prob = own_elast['Probability'].sum()
    agg_elast = (own_elast['Elasticity'] * own_elast['Probability']).sum() / total_prob

    # SE via error propagation (simplified)
    weights = own_elast['Probability'].values / total_prob
    agg_se = np.sqrt(np.sum((weights * own_elast['SE'].values) ** 2))

    result = pd.DataFrame([{
        'Measure': 'Aggregate_Own_Price_Elasticity',
        'Attribute': 'fee',
        'Elasticity': agg_elast,
        'SE': agg_se,
        't_stat': agg_elast / agg_se if agg_se > 0 else np.nan,
        'CI_95_Lower': agg_elast - 1.96 * agg_se,
        'CI_95_Upper': agg_elast + 1.96 * agg_se
    }])

    return result


# ============================================================================
# SCENARIO ANALYSIS
# ============================================================================

def compute_scenario_analysis(
    results: EstimationResults,
    df: pd.DataFrame,
    config: Dict[str, Any],
    scenarios: List[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Compute market share predictions for multiple policy scenarios.

    Each scenario specifies changes to attribute values from baseline.

    Args:
        results: Estimation results
        df: Data with attribute values
        config: Model configuration
        scenarios: List of scenario dicts with 'name' and attribute changes

    Returns:
        DataFrame with market shares for each scenario
    """
    betas = results.betas
    fee_scale = config.get('choice_model', {}).get('fee_scale', 10000.0)

    # Get baseline values
    fee_means = [df[f'fee{i+1}'].mean() for i in range(3) if f'fee{i+1}' in df.columns]
    dur_means = [df[f'dur{i+1}'].mean() for i in range(3) if f'dur{i+1}' in df.columns]
    n_alts = len(fee_means)

    # Get coefficients - handle MXL
    asc = betas.get('ASC_paid', 0)
    b_fee = betas.get('B_FEE', betas.get('B_FEE_MU', 0))
    b_dur = betas.get('B_DUR', 0)

    def compute_shares(fee_vals, dur_vals):
        """Compute market shares for given attribute values."""
        utilities = np.zeros(n_alts)
        for j in range(n_alts):
            if j < 2:  # Paid alternatives
                utilities[j] = asc + b_fee * fee_vals[j] / fee_scale + b_dur * dur_vals[j]
            else:
                utilities[j] = b_dur * dur_vals[j]

        exp_u = np.exp(utilities - np.max(utilities))
        return exp_u / np.sum(exp_u)

    # Default scenarios if none provided
    if scenarios is None:
        scenarios = [
            {'name': 'Baseline', 'changes': {}},
            {'name': 'Alt1: -10% Fee', 'changes': {'fee1': -fee_means[0] * 0.10}},
            {'name': 'Alt1: -20% Fee', 'changes': {'fee1': -fee_means[0] * 0.20}},
            {'name': 'Alt1: -5 Days', 'changes': {'dur1': -5}},
            {'name': 'Alt1: -10% Fee, -5 Days', 'changes': {'fee1': -fee_means[0] * 0.10, 'dur1': -5}},
            {'name': 'Alt2: -10% Fee', 'changes': {'fee2': -fee_means[1] * 0.10}},
            {'name': 'All: -10% Fee', 'changes': {'fee1': -fee_means[0] * 0.10, 'fee2': -fee_means[1] * 0.10}},
        ]

    rows = []

    for scenario in scenarios:
        name = scenario.get('name', 'Unnamed')
        changes = scenario.get('changes', {})

        # Apply changes
        fee_new = fee_means.copy()
        dur_new = dur_means.copy()

        for key, change in changes.items():
            if key.startswith('fee'):
                idx = int(key[3]) - 1
                if idx < len(fee_new):
                    fee_new[idx] += change
            elif key.startswith('dur'):
                idx = int(key[3]) - 1
                if idx < len(dur_new):
                    dur_new[idx] += change

        # Compute shares
        shares = compute_shares(fee_new, dur_new)
        baseline_shares = compute_shares(fee_means, dur_means)

        # Record results
        for j in range(n_alts):
            rows.append({
                'Scenario': name,
                'Alternative': j + 1,
                'Predicted_Share': shares[j],
                'Baseline_Share': baseline_shares[j],
                'Share_Change_pp': (shares[j] - baseline_shares[j]) * 100,
                'Fee_Value': fee_new[j],
                'Dur_Value': dur_new[j]
            })

    return pd.DataFrame(rows)


def compute_sensitivity_analysis(
    results: EstimationResults,
    df: pd.DataFrame,
    config: Dict[str, Any],
    attribute: str = 'fee',
    alternative: int = 1,
    pct_range: List[float] = None
) -> pd.DataFrame:
    """
    Compute sensitivity analysis for one attribute.

    Shows how market shares change as one attribute varies.

    Args:
        results: Estimation results
        df: Data with attribute values
        config: Model configuration
        attribute: 'fee' or 'dur'
        alternative: Which alternative to vary (1, 2, or 3)
        pct_range: Percentage changes to evaluate (default: -50% to +50%)

    Returns:
        DataFrame with shares at each variation level
    """
    betas = results.betas
    fee_scale = config.get('choice_model', {}).get('fee_scale', 10000.0)

    # Get baseline values
    fee_means = [df[f'fee{i+1}'].mean() for i in range(3) if f'fee{i+1}' in df.columns]
    dur_means = [df[f'dur{i+1}'].mean() for i in range(3) if f'dur{i+1}' in df.columns]
    n_alts = len(fee_means)

    # Get coefficients
    asc = betas.get('ASC_paid', 0)
    b_fee = betas.get('B_FEE', betas.get('B_FEE_MU', 0))
    b_dur = betas.get('B_DUR', 0)

    if pct_range is None:
        pct_range = [-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50]

    def compute_shares(fee_vals, dur_vals):
        utilities = np.zeros(n_alts)
        for j in range(n_alts):
            if j < 2:
                utilities[j] = asc + b_fee * fee_vals[j] / fee_scale + b_dur * dur_vals[j]
            else:
                utilities[j] = b_dur * dur_vals[j]
        exp_u = np.exp(utilities - np.max(utilities))
        return exp_u / np.sum(exp_u)

    alt_idx = alternative - 1
    baseline_value = fee_means[alt_idx] if attribute == 'fee' else dur_means[alt_idx]

    rows = []
    for pct in pct_range:
        # Apply percentage change
        fee_new = fee_means.copy()
        dur_new = dur_means.copy()

        if attribute == 'fee':
            fee_new[alt_idx] = baseline_value * (1 + pct / 100)
        else:
            dur_new[alt_idx] = baseline_value * (1 + pct / 100)

        shares = compute_shares(fee_new, dur_new)

        rows.append({
            'Pct_Change': pct,
            'Attribute': attribute,
            'Alternative_Varied': alternative,
            'New_Value': fee_new[alt_idx] if attribute == 'fee' else dur_new[alt_idx],
            'Alt1_Share': shares[0],
            'Alt2_Share': shares[1],
            'Alt3_Share': shares[2] if n_alts > 2 else np.nan
        })

    return pd.DataFrame(rows)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def run_policy_analysis(
    biogeme_results,
    df: pd.DataFrame,
    config: Dict[str, Any],
    output_dir: Path,
    model_type: str = 'MNL',
    verbose: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Run all policy analysis and save results.

    Args:
        biogeme_results: Biogeme estimation results object
        df: Data used for estimation
        config: Model configuration
        output_dir: Directory to save outputs (policy_analysis/)
        model_type: Type of model (MNL, MXL, HCM, ICLV)
        verbose: Whether to print progress

    Returns:
        Dictionary of result DataFrames
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 60)
        print("POLICY ANALYSIS")
        print("=" * 60)

    # Extract results
    results = extract_results(biogeme_results, model_type)

    policy_results = {}

    # 1. WTP Analysis
    try:
        wtp_df = compute_all_wtp(results, config)
        if len(wtp_df) > 0:
            wtp_df.to_csv(output_dir / 'wtp_results.csv', index=False)
            policy_results['wtp'] = wtp_df
            if verbose:
                print(f"  Saved: wtp_results.csv")
                for _, row in wtp_df.iterrows():
                    print(f"    WTP({row['Attribute']}): {row['WTP_TL']:,.0f} TL "
                          f"[{row['CI_95_Lower']:,.0f}, {row['CI_95_Upper']:,.0f}]")
    except Exception as e:
        if verbose:
            print(f"  Warning: WTP calculation failed: {e}")

    # 2. Elasticity Matrix
    try:
        elast_df = compute_elasticity_matrix(results, df, config, 'fee')
        if len(elast_df) > 0:
            elast_df.to_csv(output_dir / 'elasticity_matrix.csv')
            policy_results['elasticities'] = elast_df
            if verbose:
                print(f"  Saved: elasticity_matrix.csv")
    except Exception as e:
        if verbose:
            print(f"  Warning: Elasticity calculation failed: {e}")

    # 3. Market Shares
    try:
        shares_df = compute_market_shares(results, df, config)
        if len(shares_df) > 0:
            shares_df.to_csv(output_dir / 'market_shares.csv', index=False)
            policy_results['market_shares'] = shares_df
            if verbose:
                print(f"  Saved: market_shares.csv")
                for _, row in shares_df.iterrows():
                    pred = row['Predicted_Share']
                    obs = row['Observed_Share']
                    print(f"    Alt {int(row['Alternative'])}: "
                          f"Pred={pred:.1%}, Obs={obs:.1%}")
    except Exception as e:
        if verbose:
            print(f"  Warning: Market share calculation failed: {e}")

    # 4. Marginal Effects
    try:
        me_df = compute_marginal_effects(results, df, config, 'fee')
        if len(me_df) > 0:
            me_df.to_csv(output_dir / 'marginal_effects.csv', index=False)
            policy_results['marginal_effects'] = me_df
            if verbose:
                print(f"  Saved: marginal_effects.csv")
    except Exception as e:
        if verbose:
            print(f"  Warning: Marginal effects calculation failed: {e}")

    # 5. Segment Analysis (for models with demographics or LV)
    if model_type in ['MNL_DEMO', 'HCM', 'ICLV']:
        for seg_col in ['age_idx', 'edu_idx']:
            if seg_col in df.columns:
                try:
                    seg_df = compute_wtp_by_segment(results, df, config, seg_col)
                    if len(seg_df) > 0:
                        filename = f'wtp_by_{seg_col}.csv'
                        seg_df.to_csv(output_dir / filename, index=False)
                        policy_results[f'wtp_{seg_col}'] = seg_df
                        if verbose:
                            print(f"  Saved: {filename}")
                except Exception as e:
                    if verbose:
                        print(f"  Warning: Segment analysis for {seg_col} failed: {e}")

    # 6. Consumer Surplus / Compensating Variation (NEW)
    try:
        cv_df = compute_consumer_surplus_scenarios(results, df, config)
        if len(cv_df) > 0:
            cv_df.to_csv(output_dir / 'compensating_variation.csv', index=False)
            policy_results['compensating_variation'] = cv_df
            if verbose:
                print(f"  Saved: compensating_variation.csv")
                print("  Consumer Surplus Scenarios:")
                for _, row in cv_df.iterrows():
                    cv = row['CV_per_person_TL']
                    se = row.get('CV_SE', np.nan)
                    se_str = f" (SE: {se:,.0f})" if not np.isnan(se) else ""
                    print(f"    {row['Scenario']}: CV = {cv:,.0f} TL{se_str}")
    except Exception as e:
        if verbose:
            print(f"  Warning: Compensating variation calculation failed: {e}")

    # 7. WTP Distribution for MXL (NEW)
    if model_type == 'MXL':
        try:
            wtp_dist_df = compute_wtp_distribution(results, config)
            if len(wtp_dist_df) > 0:
                wtp_dist_df.to_csv(output_dir / 'wtp_distribution.csv', index=False)
                policy_results['wtp_distribution'] = wtp_dist_df
                if verbose:
                    print(f"  Saved: wtp_distribution.csv")
                    row = wtp_dist_df.iloc[0]
                    print(f"  WTP Distribution (MXL heterogeneity):")
                    print(f"    Mean: {row['Mean_WTP_TL']:,.0f} TL")
                    print(f"    Median: {row['Median_WTP_TL']:,.0f} TL")
                    print(f"    Std Dev: {row['Std_WTP_TL']:,.0f} TL")
                    print(f"    [5th, 95th]: [{row['Pct_5']:,.0f}, {row['Pct_95']:,.0f}] TL")
                    print(f"    % with positive fee coef (wrong sign): {row['Pct_Positive_Fee_Coef']:.1f}%")
        except Exception as e:
            if verbose:
                print(f"  Warning: WTP distribution calculation failed: {e}")

    # 8. Elasticity with Standard Errors (NEW)
    try:
        elast_se_df = compute_elasticity_with_se(results, df, config, 'fee')
        if len(elast_se_df) > 0:
            elast_se_df.to_csv(output_dir / 'elasticity_with_se.csv', index=False)
            policy_results['elasticity_with_se'] = elast_se_df
            if verbose:
                print(f"  Saved: elasticity_with_se.csv")
                # Print own-price elasticities with SE
                own_elast = elast_se_df[elast_se_df['Elasticity_Type'] == 'Own']
                print("  Own-Price Elasticities with SE:")
                for _, row in own_elast.iterrows():
                    elast = row['Elasticity']
                    se = row['SE']
                    print(f"    {row['Alternative']}: {elast:.4f} (SE: {se:.4f})")
    except Exception as e:
        if verbose:
            print(f"  Warning: Elasticity with SE calculation failed: {e}")

    # 9. Aggregate Elasticity (NEW)
    try:
        agg_elast_df = compute_aggregate_elasticities(results, df, config)
        if len(agg_elast_df) > 0:
            agg_elast_df.to_csv(output_dir / 'aggregate_elasticity.csv', index=False)
            policy_results['aggregate_elasticity'] = agg_elast_df
            if verbose:
                print(f"  Saved: aggregate_elasticity.csv")
                row = agg_elast_df.iloc[0]
                print(f"  Aggregate Own-Price Elasticity: {row['Elasticity']:.4f} "
                      f"(SE: {row['SE']:.4f})")
    except Exception as e:
        if verbose:
            print(f"  Warning: Aggregate elasticity calculation failed: {e}")

    # 10. Scenario Analysis (NEW)
    try:
        scenario_df = compute_scenario_analysis(results, df, config)
        if len(scenario_df) > 0:
            scenario_df.to_csv(output_dir / 'scenario_analysis.csv', index=False)
            policy_results['scenario_analysis'] = scenario_df
            if verbose:
                print(f"  Saved: scenario_analysis.csv")
                print("  Policy Scenarios (Alt 1 share changes):")
                alt1_scenarios = scenario_df[scenario_df['Alternative'] == 1]
                for _, row in alt1_scenarios.iterrows():
                    change = row['Share_Change_pp']
                    sign = '+' if change >= 0 else ''
                    print(f"    {row['Scenario']}: {sign}{change:.1f}pp")
    except Exception as e:
        if verbose:
            print(f"  Warning: Scenario analysis failed: {e}")

    # 11. Sensitivity Analysis (NEW)
    try:
        sens_df = compute_sensitivity_analysis(results, df, config, 'fee', 1)
        if len(sens_df) > 0:
            sens_df.to_csv(output_dir / 'sensitivity_analysis.csv', index=False)
            policy_results['sensitivity_analysis'] = sens_df
            if verbose:
                print(f"  Saved: sensitivity_analysis.csv")
    except Exception as e:
        if verbose:
            print(f"  Warning: Sensitivity analysis failed: {e}")

    if verbose:
        print(f"\nPolicy analysis saved to: {output_dir}")
        print("=" * 60 + "\n")

    return policy_results
