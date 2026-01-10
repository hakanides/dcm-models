"""
LaTeX table generation for isolated DCM models.

Generates publication-ready LaTeX tables for:
- Parameter estimates with bias and CI coverage
- Model summary statistics
- Policy analysis results (WTP, elasticities)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime


def escape_latex(text: str) -> str:
    """Escape special LaTeX characters."""
    if not isinstance(text, str):
        text = str(text)
    replacements = {
        '_': r'\_',
        '%': r'\%',
        '&': r'\&',
        '#': r'\#',
        '$': r'\$',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def format_number(value: float, decimals: int = 4) -> str:
    """Format number for LaTeX with appropriate precision."""
    if pd.isna(value):
        return '--'
    # Handle string values that may come from Biogeme
    if isinstance(value, str):
        try:
            value = float(value)
        except (ValueError, TypeError):
            return '--'
    try:
        if abs(value) < 0.0001 and value != 0:
            return f'{value:.2e}'
        return f'{value:.{decimals}f}'
    except (TypeError, ValueError):
        return '--'


def significance_stars(p_value: float) -> str:
    """Return significance stars based on p-value."""
    if pd.isna(p_value):
        return ''
    if p_value < 0.001:
        return '$^{***}$'
    elif p_value < 0.01:
        return '$^{**}$'
    elif p_value < 0.05:
        return '$^{*}$'
    return ''


# ============================================================================
# PARAMETER TABLE
# ============================================================================

def generate_parameter_table(
    results_df: pd.DataFrame,
    true_values: Dict[str, float],
    output_dir: Path,
    model_name: str = 'Model'
) -> Path:
    """
    Generate LaTeX table for parameter estimates.

    Args:
        results_df: DataFrame with columns: Parameter, Estimate, Std_Error, t_stat, p_value
        true_values: Dictionary of true parameter values
        output_dir: Directory to save the .tex file
        model_name: Name of the model for caption

    Returns:
        Path to the generated .tex file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = []

    # Header
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(r'\caption{Parameter Estimates: ' + escape_latex(model_name) + '}')
    lines.append(r'\label{tab:' + model_name.lower().replace(' ', '_') + '_params}')
    lines.append(r'\begin{tabular}{lrrrrrrr}')
    lines.append(r'\toprule')
    lines.append(r'Parameter & True & Estimate & SE & $t$-stat & Bias\,\% & 95\,\% CI & Coverage \\')
    lines.append(r'\midrule')

    # Data rows
    for _, row in results_df.iterrows():
        param = row.get('Parameter', row.name if hasattr(row, 'name') else '')
        estimate = row.get('Estimate', row.get('estimate', np.nan))
        se = row.get('Std_Error', row.get('std_err', row.get('SE', np.nan)))
        t_stat = row.get('t_stat', row.get('t-stat', np.nan))
        p_val = row.get('p_value', row.get('p-value', np.nan))

        # Get true value
        true_val = true_values.get(param, np.nan)

        # Calculate bias
        if not pd.isna(true_val) and true_val != 0:
            bias_pct = (estimate - true_val) / abs(true_val) * 100
        else:
            bias_pct = np.nan

        # Calculate CI
        if not pd.isna(se):
            ci_low = estimate - 1.96 * se
            ci_high = estimate + 1.96 * se
            ci_str = f'[{ci_low:.3f}, {ci_high:.3f}]'

            # Check coverage
            if not pd.isna(true_val):
                covered = ci_low <= true_val <= ci_high
                coverage_str = r'\checkmark' if covered else r'$\times$'
            else:
                coverage_str = '--'
        else:
            ci_str = '--'
            coverage_str = '--'

        # Format row
        stars = significance_stars(p_val)
        param_escaped = escape_latex(param)

        line = (f'{param_escaped} & {format_number(true_val, 4)} & '
                f'{format_number(estimate, 4)}{stars} & {format_number(se, 4)} & '
                f'{format_number(t_stat, 2)} & {format_number(bias_pct, 1)} & '
                f'{ci_str} & {coverage_str} \\\\')
        lines.append(line)

    # Footer
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\begin{tablenotes}')
    lines.append(r'\small')
    lines.append(r'\item Note: $^{***}p<0.001$, $^{**}p<0.01$, $^{*}p<0.05$. '
                r'Bias\,\% = (Estimate - True) / $|$True$|$ $\times$ 100.')
    lines.append(r'\end{tablenotes}')
    lines.append(r'\end{table}')

    # Write file
    output_path = output_dir / 'parameter_table.tex'
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    return output_path


# ============================================================================
# MODEL SUMMARY
# ============================================================================

def generate_model_summary(
    results,
    output_dir: Path,
    model_name: str = 'Model'
) -> Path:
    """
    Generate LaTeX table for model fit statistics.

    Args:
        results: Biogeme results object or dict with fit statistics
        output_dir: Directory to save the .tex file
        model_name: Name of the model

    Returns:
        Path to the generated .tex file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Helper to extract value from tuple or scalar
    def get_value(val, default=np.nan):
        if val is None:
            return default
        if isinstance(val, tuple):
            val = val[0] if len(val) > 0 else default
        # Try to convert string to float
        if isinstance(val, str):
            try:
                return float(val)
            except (ValueError, TypeError):
                return default
        return val

    # Extract statistics
    if hasattr(results, 'get_general_statistics'):
        stats = results.get_general_statistics()
        ll = np.nan
        null_ll = np.nan
        n_obs = np.nan
        n_params = np.nan
        aic = np.nan
        bic = np.nan
        rho2 = np.nan
        rho2_adj = np.nan

        for key, val in stats.items():
            if 'Final log likelihood' in key:
                ll = get_value(val)
            elif 'Null log' in key:
                null_ll = get_value(val)
            elif 'Number of observations' in key:
                n_obs = get_value(val)
            elif 'Number of estimated' in key:
                n_params = get_value(val)
            elif 'Akaike' in key:
                aic = get_value(val)
            elif 'Bayesian' in key:
                bic = get_value(val)
            elif key == 'Rho-square':
                rho2 = get_value(val)
            elif 'Rho-square-bar' in key:
                rho2_adj = get_value(val)
    else:
        # Assume dict
        ll = results.get('log_likelihood', np.nan)
        null_ll = results.get('null_ll', np.nan)
        n_obs = results.get('n_obs', np.nan)
        n_params = results.get('n_params', np.nan)
        aic = results.get('aic', np.nan)
        bic = results.get('bic', np.nan)
        rho2 = results.get('rho2', np.nan)
        rho2_adj = results.get('rho2_adj', np.nan)

    # Format integer values safely
    def format_int(val, default='--'):
        if pd.isna(val):
            return default
        return f'{int(val):,}'

    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(r'\caption{Model Fit Statistics: ' + escape_latex(model_name) + '}')
    lines.append(r'\label{tab:' + model_name.lower().replace(' ', '_') + '_fit}')
    lines.append(r'\begin{tabular}{lr}')
    lines.append(r'\toprule')
    lines.append(r'Statistic & Value \\')
    lines.append(r'\midrule')
    lines.append(f'Number of observations & {format_int(n_obs)} \\\\')
    lines.append(f'Number of parameters & {format_int(n_params)} \\\\')
    lines.append(f'Log-likelihood & {format_number(ll, 2)} \\\\')
    lines.append(f'Null log-likelihood & {format_number(null_ll, 2)} \\\\')
    lines.append(f'AIC & {format_number(aic, 2)} \\\\')
    lines.append(f'BIC & {format_number(bic, 2)} \\\\')
    lines.append(f'$\\rho^2$ & {format_number(rho2, 4)} \\\\')
    lines.append(f'Adjusted $\\rho^2$ & {format_number(rho2_adj, 4)} \\\\')
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    output_path = output_dir / 'model_summary.tex'
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    return output_path


# ============================================================================
# POLICY SUMMARY
# ============================================================================

def generate_policy_summary(
    policy_results: Dict[str, pd.DataFrame],
    output_dir: Path,
    model_name: str = 'Model'
) -> Path:
    """
    Generate LaTeX table for policy analysis results.

    Args:
        policy_results: Dictionary with 'wtp', 'elasticities', 'market_shares' DataFrames
        output_dir: Directory to save the .tex file
        model_name: Name of the model

    Returns:
        Path to the generated .tex file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(r'\caption{Policy Analysis: ' + escape_latex(model_name) + '}')
    lines.append(r'\label{tab:' + model_name.lower().replace(' ', '_') + '_policy}')

    # Panel A: WTP
    if 'wtp' in policy_results and len(policy_results['wtp']) > 0:
        wtp_df = policy_results['wtp']
        lines.append(r'\begin{subtable}{\textwidth}')
        lines.append(r'\centering')
        lines.append(r'\caption{Willingness to Pay (TL)}')
        lines.append(r'\begin{tabular}{lrrr}')
        lines.append(r'\toprule')
        lines.append(r'Attribute & WTP & SE & 95\% CI \\')
        lines.append(r'\midrule')

        for _, row in wtp_df.iterrows():
            attr = escape_latex(str(row.get('Attribute', '')))
            wtp = row.get('WTP_TL', np.nan)
            se = row.get('SE', np.nan)
            ci_low = row.get('CI_95_Lower', np.nan)
            ci_high = row.get('CI_95_Upper', np.nan)

            ci_str = f'[{ci_low:,.0f}, {ci_high:,.0f}]' if not pd.isna(ci_low) else '--'
            lines.append(f'{attr} & {wtp:,.0f} & {se:,.0f} & {ci_str} \\\\')

        lines.append(r'\bottomrule')
        lines.append(r'\end{tabular}')
        lines.append(r'\end{subtable}')
        lines.append(r'\vspace{1em}')

    # Panel B: Market Shares
    if 'market_shares' in policy_results and len(policy_results['market_shares']) > 0:
        shares_df = policy_results['market_shares']
        lines.append(r'\begin{subtable}{\textwidth}')
        lines.append(r'\centering')
        lines.append(r'\caption{Predicted vs Observed Market Shares}')
        lines.append(r'\begin{tabular}{lrrr}')
        lines.append(r'\toprule')
        lines.append(r'Alternative & Predicted & Observed & Difference \\')
        lines.append(r'\midrule')

        for _, row in shares_df.iterrows():
            alt = int(row.get('Alternative', 0))
            pred = row.get('Predicted_Share', np.nan) * 100
            obs = row.get('Observed_Share', np.nan) * 100
            diff = row.get('Difference', np.nan) * 100

            obs_str = f'{obs:.1f}\\%' if not pd.isna(obs) else '--'
            diff_str = f'{diff:+.1f}pp' if not pd.isna(diff) else '--'
            lines.append(f'Alt {alt} & {pred:.1f}\\% & {obs_str} & {diff_str} \\\\')

        lines.append(r'\bottomrule')
        lines.append(r'\end{tabular}')
        lines.append(r'\end{subtable}')

    lines.append(r'\end{table}')

    output_path = output_dir / 'policy_summary.tex'
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    return output_path


# ============================================================================
# ELASTICITY MATRIX
# ============================================================================

def generate_elasticity_table(
    elasticity_df: pd.DataFrame,
    output_dir: Path,
    model_name: str = 'Model'
) -> Path:
    """
    Generate LaTeX table for elasticity matrix.

    Args:
        elasticity_df: DataFrame with elasticity matrix
        output_dir: Directory to save the .tex file
        model_name: Name of the model

    Returns:
        Path to the generated .tex file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_alts = len(elasticity_df)

    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(r'\caption{Fee Elasticity Matrix: ' + escape_latex(model_name) + '}')
    lines.append(r'\label{tab:' + model_name.lower().replace(' ', '_') + '_elast}')

    # Column format
    col_fmt = 'l' + 'r' * n_alts
    lines.append(r'\begin{tabular}{' + col_fmt + '}')
    lines.append(r'\toprule')

    # Header
    header = r'$\frac{\partial P_i}{\partial \text{fee}_j}$'
    for j in range(n_alts):
        header += f' & Alt {j+1}'
    header += r' \\'
    lines.append(header)
    lines.append(r'\midrule')

    # Data rows
    for i, (idx, row) in enumerate(elasticity_df.iterrows()):
        line = f'Alt {i+1}'
        for val in row.values:
            line += f' & {format_number(val, 4)}'
        line += r' \\'
        lines.append(line)

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\begin{tablenotes}')
    lines.append(r'\small')
    lines.append(r'\item Note: Diagonal elements are own-price elasticities; '
                r'off-diagonal elements are cross-price elasticities.')
    lines.append(r'\end{tablenotes}')
    lines.append(r'\end{table}')

    output_path = output_dir / 'elasticity_table.tex'
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    return output_path


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def generate_all_latex(
    biogeme_results,
    true_values: Dict[str, float],
    policy_results: Dict[str, pd.DataFrame],
    output_dir: Path,
    model_name: str = 'Model',
    verbose: bool = True
) -> Dict[str, Path]:
    """
    Generate all LaTeX tables.

    Args:
        biogeme_results: Biogeme results object
        true_values: Dictionary of true parameter values
        policy_results: Dictionary of policy analysis DataFrames
        output_dir: Directory to save .tex files
        model_name: Name of the model
        verbose: Whether to print progress

    Returns:
        Dictionary of generated file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 60)
        print("LATEX OUTPUT")
        print("=" * 60)

    generated_files = {}

    # 1. Parameter Table
    try:
        # Extract parameter estimates - handle different Biogeme versions
        try:
            betas = biogeme_results.get_beta_values()
        except AttributeError:
            estimates_df = biogeme_results.get_estimated_parameters()
            if 'Name' in estimates_df.columns:
                estimates_df = estimates_df.set_index('Name')
            betas = estimates_df['Value'].to_dict()

        # Get other statistics from DataFrame
        try:
            estimates_df = biogeme_results.get_estimated_parameters()
            if 'Name' in estimates_df.columns:
                estimates_df = estimates_df.set_index('Name')

            std_errs = estimates_df['Robust std err.'].to_dict() if 'Robust std err.' in estimates_df.columns else {}
            t_stats = estimates_df['Robust t-stat.'].to_dict() if 'Robust t-stat.' in estimates_df.columns else {}
            p_vals = estimates_df['Robust p-value'].to_dict() if 'Robust p-value' in estimates_df.columns else {}
        except Exception:
            std_errs = {}
            t_stats = {}
            p_vals = {}

        rows = []
        for param in betas.keys():
            rows.append({
                'Parameter': param,
                'Estimate': betas[param],
                'Std_Error': std_errs.get(param, np.nan),
                't_stat': t_stats.get(param, np.nan),
                'p_value': p_vals.get(param, np.nan)
            })
        results_df = pd.DataFrame(rows)

        path = generate_parameter_table(results_df, true_values, output_dir, model_name)
        generated_files['parameter_table'] = path
        if verbose:
            print(f"  Saved: parameter_table.tex")
    except Exception as e:
        if verbose:
            print(f"  Warning: Parameter table generation failed: {e}")

    # 2. Model Summary
    try:
        path = generate_model_summary(biogeme_results, output_dir, model_name)
        generated_files['model_summary'] = path
        if verbose:
            print(f"  Saved: model_summary.tex")
    except Exception as e:
        if verbose:
            print(f"  Warning: Model summary generation failed: {e}")

    # 3. Policy Summary
    try:
        if policy_results:
            path = generate_policy_summary(policy_results, output_dir, model_name)
            generated_files['policy_summary'] = path
            if verbose:
                print(f"  Saved: policy_summary.tex")
    except Exception as e:
        if verbose:
            print(f"  Warning: Policy summary generation failed: {e}")

    # 4. Elasticity Table
    try:
        if 'elasticities' in policy_results and len(policy_results['elasticities']) > 0:
            path = generate_elasticity_table(policy_results['elasticities'], output_dir, model_name)
            generated_files['elasticity_table'] = path
            if verbose:
                print(f"  Saved: elasticity_table.tex")
    except Exception as e:
        if verbose:
            print(f"  Warning: Elasticity table generation failed: {e}")

    if verbose:
        print(f"\nLaTeX files saved to: {output_dir}")
        print("=" * 60 + "\n")

    return generated_files
