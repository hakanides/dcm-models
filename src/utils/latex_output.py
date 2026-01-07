"""
LaTeX Output Generation for DCM Models
======================================

Generate LaTeX output files from Biogeme estimation results,
organized by model family (MNL, MXL, HCM).

Usage:
    from src.utils.latex_output import generate_latex_output

    # After model estimation
    results = biogeme_obj.estimate()
    generate_latex_output(results, 'MNL-Basic')

Author: DCM Research Team
"""

from pathlib import Path
import shutil


def generate_latex_output(biogeme_results,
                          model_name: str,
                          output_dir: Path = None) -> Path:
    """
    Generate LaTeX file from Biogeme estimation results.

    Creates organized directory structure by model family and generates
    fragment-only LaTeX files (for \\input{} into your documents).

    Args:
        biogeme_results: Biogeme EstimationResults object (from biogeme.estimate())
        model_name: Model name, e.g., 'MNL-Basic', 'MXL-RandomFee'
        output_dir: Base output directory (default: 'output/latex')

    Returns:
        Path to the generated LaTeX file

    Directory Structure:
        latex_output/
        ├── MNL/
        │   ├── MNL_Basic.tex
        │   └── ...
        ├── MXL/
        │   └── ...
        └── HCM/
            └── ...
    """
    if output_dir is None:
        output_dir = Path("output/latex")

    # Determine model family from name (e.g., 'MNL-Basic' -> 'MNL')
    family = model_name.split('-')[0].upper()

    # Create directory structure
    family_dir = output_dir / family
    family_dir.mkdir(parents=True, exist_ok=True)

    # Generate safe filename
    file_name = model_name.replace('-', '_').replace(' ', '_')
    output_path = family_dir / f"{file_name}.tex"

    # Generate LaTeX using Biogeme's built-in method
    # include_begin_document=False creates fragment-only (no \documentclass)
    temp_latex_path = biogeme_results.write_latex(include_begin_document=False)

    # Move generated file to our organized directory
    if temp_latex_path and Path(temp_latex_path).exists():
        shutil.move(temp_latex_path, output_path)
        print(f"  LaTeX output: {output_path}")
    else:
        # If write_latex didn't work as expected, try alternative approach
        _generate_latex_fallback(biogeme_results, output_path)

    return output_path


def _generate_latex_fallback(biogeme_results, output_path: Path) -> None:
    """
    Fallback LaTeX generation if Biogeme's write_latex doesn't produce expected output.

    Creates a basic LaTeX table with parameter estimates.
    """
    betas = biogeme_results.get_beta_values()
    general_stats = biogeme_results.get_general_statistics()

    lines = [
        f"% LaTeX output for model estimation",
        f"% Generated automatically - fragment for \\input{{}}",
        "",
        "% General Statistics",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Model Statistics}",
        "\\begin{tabular}{lr}",
        "\\toprule",
        "Statistic & Value \\\\",
        "\\midrule",
    ]

    # Add general statistics
    stat_names = {
        'final_log_likelihood': 'Log-likelihood',
        'number_of_free_parameters': 'Parameters',
        'akaike_information_criterion': 'AIC',
        'bayesian_information_criterion': 'BIC',
    }

    for key, label in stat_names.items():
        if key in general_stats:
            val = general_stats[key]
            if isinstance(val, float):
                lines.append(f"{label} & {val:.4f} \\\\")
            else:
                lines.append(f"{label} & {val} \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
        "% Parameter Estimates",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Parameter Estimates}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Parameter & Estimate & Std. Err. & t-stat & p-value \\\\",
        "\\midrule",
    ])

    # Add parameter estimates
    for param, value in betas.items():
        try:
            se = biogeme_results.get_parameter_std_err(param)
            t_stat = value / se if se and se > 0 else float('nan')
            # Two-tailed p-value approximation
            import math
            if not math.isnan(t_stat):
                # Simple approximation using normal distribution
                p_val = 2 * (1 - _normal_cdf(abs(t_stat)))
            else:
                p_val = float('nan')

            lines.append(
                f"{_escape_latex(param)} & {value:.4f} & {se:.4f} & {t_stat:.2f} & {p_val:.4f} \\\\"
            )
        except:
            lines.append(f"{_escape_latex(param)} & {value:.4f} & -- & -- & -- \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"  LaTeX output (fallback): {output_path}")


def _escape_latex(text: str) -> str:
    """Escape special LaTeX characters.

    ISSUE #27 FIX: Added complete set of special characters.

    LaTeX special characters that need escaping:
    - _ (underscore) -> \\_
    - % (percent) -> \\%
    - & (ampersand) -> \\&
    - # (hash) -> \\#
    - $ (dollar) -> \\$
    - { (left brace) -> \\{
    - } (right brace) -> \\}
    - ^ (caret) -> \\^{}
    - ~ (tilde) -> \\~{}
    - \\ (backslash) -> \\textbackslash{}

    Note: Backslash must be handled first to avoid double-escaping.
    """
    # Order matters: backslash first, then others
    replacements = [
        ('\\', '\\textbackslash{}'),  # Must be first!
        ('_', '\\_'),
        ('%', '\\%'),
        ('&', '\\&'),
        ('#', '\\#'),
        ('$', '\\$'),
        ('{', '\\{'),
        ('}', '\\}'),
        ('^', '\\^{}'),
        ('~', '\\~{}'),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def _normal_cdf(x: float) -> float:
    """Approximate standard normal CDF."""
    import math
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def cleanup_latex_output(output_dir: Path = None) -> None:
    """
    Remove all files in the latex_output directory.

    Call this at the start of a new model run to ensure fresh output.
    """
    if output_dir is None:
        output_dir = Path("output/latex")

    if output_dir.exists():
        shutil.rmtree(output_dir)
        print(f"Cleaned up LaTeX output: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)


# =============================================================================
# SIMULATION LATEX OUTPUT FUNCTIONS
# =============================================================================

def generate_simulation_summary_latex(df, output_dir: Path = None) -> Path:
    """
    Generate LaTeX table with simulation summary statistics.

    Creates a table with:
    - Total observations
    - Unique respondents
    - Tasks per respondent
    - Choice shares for each alternative

    Args:
        df: DataFrame with simulation data (must have CHOICE and ID columns)
        output_dir: Base output directory (default: 'output/latex')

    Returns:
        Path to generated LaTeX file
    """
    if output_dir is None:
        output_dir = Path("output/latex")

    sim_dir = output_dir / "simulation"
    sim_dir.mkdir(parents=True, exist_ok=True)
    output_path = sim_dir / "simulation_summary.tex"

    # Compute statistics
    n_obs = len(df)
    n_respondents = df['ID'].nunique() if 'ID' in df.columns else n_obs
    tasks_per_respondent = n_obs / n_respondents if n_respondents > 0 else 0

    # Choice shares
    choice_counts = df['CHOICE'].value_counts().sort_index()
    choice_shares = choice_counts / n_obs * 100

    lines = [
        "% Simulation Summary Statistics",
        "% Generated automatically - fragment for \\input{}",
        "",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Simulation Summary}",
        "\\begin{tabular}{lr}",
        "\\toprule",
        "Statistic & Value \\\\",
        "\\midrule",
        f"Total Observations & {n_obs:,} \\\\",
        f"Unique Respondents & {n_respondents:,} \\\\",
        f"Tasks per Respondent & {tasks_per_respondent:.1f} \\\\",
        "\\midrule",
        "\\multicolumn{2}{l}{\\textit{Choice Shares}} \\\\",
    ]

    alt_names = {1: 'Alternative 1 (paid1)', 2: 'Alternative 2 (paid2)', 3: 'Alternative 3 (standard)'}
    for alt in sorted(choice_shares.index):
        name = alt_names.get(alt, f'Alternative {alt}')
        lines.append(f"{name} & {choice_shares[alt]:.1f}\\% \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"  LaTeX output: {output_path}")
    return output_path


def generate_data_quality_latex(df, output_dir: Path = None) -> Path:
    """
    Generate LaTeX table with data quality metrics.

    Creates a table with:
    - Attribute ranges (min, max)
    - Missing value counts
    - Balance status

    Args:
        df: DataFrame with simulation data
        output_dir: Base output directory (default: 'output/latex')

    Returns:
        Path to generated LaTeX file
    """
    if output_dir is None:
        output_dir = Path("output/latex")

    sim_dir = output_dir / "simulation"
    sim_dir.mkdir(parents=True, exist_ok=True)
    output_path = sim_dir / "data_quality.tex"

    # Attributes to check
    attributes = ['fee1', 'fee2', 'fee3', 'dur1', 'dur2', 'dur3']
    available_attrs = [a for a in attributes if a in df.columns]

    lines = [
        "% Data Quality Metrics",
        "% Generated automatically - fragment for \\input{}",
        "",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Data Quality Metrics}",
        "\\begin{tabular}{lrrrl}",
        "\\toprule",
        "Attribute & Min & Max & Missing & Status \\\\",
        "\\midrule",
    ]

    for attr in available_attrs:
        col = df[attr]
        min_val = col.min()
        max_val = col.max()
        missing = col.isna().sum()
        status = "OK" if missing == 0 and min_val < max_val else "Check"

        # Format numbers appropriately
        if 'fee' in attr:
            lines.append(f"{_escape_latex(attr)} & {min_val:,.0f} & {max_val:,.0f} & {missing} & {status} \\\\")
        else:
            lines.append(f"{_escape_latex(attr)} & {min_val:.1f} & {max_val:.1f} & {missing} & {status} \\\\")

    # Add choice balance info
    lines.append("\\midrule")
    lines.append("\\multicolumn{5}{l}{\\textit{Choice Balance}} \\\\")

    choice_shares = df['CHOICE'].value_counts(normalize=True) * 100
    for alt in sorted(choice_shares.index):
        share = choice_shares[alt]
        status = "OK" if 10 < share < 70 else "Imbalanced"
        lines.append(f"Choice {alt} share & \\multicolumn{{2}}{{c}}{{{share:.1f}\\%}} & -- & {status} \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"  LaTeX output: {output_path}")
    return output_path


def generate_true_parameters_latex(config_path: str = "config/model_config.json",
                                    output_dir: Path = None) -> Path:
    """
    Generate LaTeX table with true parameter values from model config.

    Creates a table with all true parameter values used in simulation,
    including base coefficients and interaction terms.

    Args:
        config_path: Path to config/model_config.json
        output_dir: Base output directory (default: 'output/latex')

    Returns:
        Path to generated LaTeX file
    """
    import json

    if output_dir is None:
        output_dir = Path("output/latex")

    sim_dir = output_dir / "simulation"
    sim_dir.mkdir(parents=True, exist_ok=True)
    output_path = sim_dir / "true_parameters.tex"

    # Load config
    with open(config_path) as f:
        config = json.load(f)

    lines = [
        "% True Parameter Values (from model\\_config.json)",
        "% Generated automatically - fragment for \\input{}",
        "",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{True Parameter Values}",
        "\\begin{tabular}{lr}",
        "\\toprule",
        "Parameter & True Value \\\\",
        "\\midrule",
    ]

    # Extract ASC from base_terms
    for term in config.get('choice_model', {}).get('base_terms', []):
        if 'paid' in str(term.get('apply_to', [])):
            lines.append(f"ASC\\_paid & {term['coef']:.4f} \\\\")

    lines.append("\\midrule")
    lines.append("\\multicolumn{2}{l}{\\textit{Attribute Coefficients}} \\\\")

    # Extract attribute coefficients
    for term in config.get('choice_model', {}).get('attribute_terms', []):
        name = term['name']
        base_coef = term.get('base_coef', 0)

        # Format parameter name for LaTeX
        param_name = f"B\\_{name.upper()}"
        if name == 'b_fee_scaled':
            param_name = "B\\_FEE (scaled)"
        elif name == 'b_dur':
            param_name = "B\\_DUR"

        lines.append(f"{param_name} & {base_coef:.4f} \\\\")

        # Add interactions
        for interaction in term.get('interactions', []):
            int_var = interaction['with']
            int_coef = interaction['coef']
            int_name = f"{param_name} $\\times$ {_escape_latex(int_var)}"
            lines.append(f"{int_name} & {int_coef:.4f} \\\\")

    # Add fee scale note
    fee_scale = config.get('choice_model', {}).get('fee_scale', 10000)
    lines.append("\\midrule")
    lines.append(f"Fee Scale & {fee_scale:,} \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"  LaTeX output: {output_path}")
    return output_path


def generate_all_simulation_latex(df, config_path: str = "config/model_config.json",
                                   output_dir: Path = None) -> None:
    """
    Generate all simulation-related LaTeX tables.

    Convenience function that calls all simulation LaTeX generators.

    Args:
        df: DataFrame with simulation data
        config_path: Path to config/model_config.json
        output_dir: Base output directory
    """
    generate_simulation_summary_latex(df, output_dir)
    generate_data_quality_latex(df, output_dir)
    generate_true_parameters_latex(config_path, output_dir)


# =============================================================================
# POLICY ANALYSIS LATEX OUTPUT FUNCTIONS
# =============================================================================

def generate_wtp_latex(wtp_results: list, output_dir: Path = None) -> Path:
    """
    Generate WTP summary table from WTPResult objects.

    Args:
        wtp_results: List of WTPResult objects (from WTPCalculator.compute_wtp)
        output_dir: Base output directory (default: 'output/latex')

    Returns:
        Path to generated LaTeX file
    """
    if output_dir is None:
        output_dir = Path("output/latex")

    policy_dir = output_dir / "policy"
    policy_dir.mkdir(parents=True, exist_ok=True)
    output_path = policy_dir / "wtp_summary.tex"

    lines = [
        "% Willingness-to-Pay Estimates",
        "% Generated automatically - fragment for \\input{}",
        "",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Willingness-to-Pay Estimates}",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Attribute & WTP & Std. Err. & 95\\% CI & t-stat & p-value \\\\",
        "\\midrule",
    ]

    for wtp in wtp_results:
        # Format attribute name
        attr_name = _escape_latex(wtp.numerator_param.replace('B_', ''))

        # Format WTP value
        wtp_str = f"{wtp.wtp_point:,.0f}"

        # Format SE
        se_str = f"{wtp.wtp_se:,.0f}" if wtp.wtp_se else "--"

        # Format CI
        if wtp.ci_lower is not None and wtp.ci_upper is not None:
            ci_str = f"[{wtp.ci_lower:,.0f}, {wtp.ci_upper:,.0f}]"
        else:
            ci_str = "--"

        # Format t-stat and p-value
        t_str = f"{wtp.t_stat:.2f}" if wtp.t_stat else "--"
        p_str = f"{wtp.p_value:.4f}" if wtp.p_value else "--"

        lines.append(f"{attr_name} & {wtp_str} & {se_str} & {ci_str} & {t_str} & {p_str} \\\\")

    lines.extend([
        "\\bottomrule",
        "\\multicolumn{6}{l}{\\footnotesize Note: WTP in KRW (fee scale = 10,000)} \\\\",
        "\\end{tabular}",
        "\\end{table}",
    ])

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"  LaTeX output: {output_path}")
    return output_path


def generate_elasticity_matrix_latex(matrix, alt_names: list = None,
                                      output_dir: Path = None) -> Path:
    """
    Generate elasticity matrix table.

    Args:
        matrix: J×J numpy array or DataFrame of elasticities (diagonal = own, off-diagonal = cross)
        alt_names: List of alternative names (default: ['Alt 1', 'Alt 2', 'Alt 3'])
        output_dir: Base output directory (default: 'output/latex')

    Returns:
        Path to generated LaTeX file
    """
    import numpy as np
    import pandas as pd

    if output_dir is None:
        output_dir = Path("output/latex")

    # Convert DataFrame to numpy array if needed
    if hasattr(matrix, 'values'):
        matrix = matrix.values

    if alt_names is None:
        alt_names = [f"Alt {i+1}" for i in range(matrix.shape[0])]

    policy_dir = output_dir / "policy"
    policy_dir.mkdir(parents=True, exist_ok=True)
    output_path = policy_dir / "elasticity_matrix.tex"

    n_alts = len(alt_names)

    # Build column spec
    col_spec = "l" + "r" * n_alts

    lines = [
        "% Price Elasticity Matrix",
        "% Generated automatically - fragment for \\input{}",
        "",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Price Elasticity Matrix}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        f"& \\multicolumn{{{n_alts}}}{{c}}{{Fee Change in Alternative}} \\\\",
        f"\\cmidrule(lr){{2-{n_alts+1}}}",
        "Probability of & " + " & ".join(alt_names) + " \\\\",
        "\\midrule",
    ]

    for i, row_name in enumerate(alt_names):
        row_vals = []
        for j in range(n_alts):
            val = matrix[i, j]
            # Mark diagonal (own-price) with asterisk
            if i == j:
                row_vals.append(f"{val:.3f}*")
            else:
                row_vals.append(f"{val:.3f}")
        lines.append(f"Choosing {row_name} & " + " & ".join(row_vals) + " \\\\")

    lines.extend([
        "\\bottomrule",
        f"\\multicolumn{{{n_alts+1}}}{{l}}{{\\footnotesize * Own-price elasticity (diagonal)}} \\\\",
        "\\end{tabular}",
        "\\end{table}",
    ])

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"  LaTeX output: {output_path}")
    return output_path


def generate_marginal_effects_latex(matrix, alt_names: list = None,
                                     output_dir: Path = None) -> Path:
    """
    Generate marginal effects matrix table.

    Args:
        matrix: J×J numpy array or DataFrame of marginal effects
        alt_names: List of alternative names (default: ['Alt 1', 'Alt 2', 'Alt 3'])
        output_dir: Base output directory (default: 'output/latex')

    Returns:
        Path to generated LaTeX file
    """
    import numpy as np
    import pandas as pd

    if output_dir is None:
        output_dir = Path("output/latex")

    # Convert DataFrame to numpy array if needed
    if hasattr(matrix, 'values'):
        matrix = matrix.values

    if alt_names is None:
        alt_names = [f"Alt {i+1}" for i in range(matrix.shape[0])]

    policy_dir = output_dir / "policy"
    policy_dir.mkdir(parents=True, exist_ok=True)
    output_path = policy_dir / "marginal_effects.tex"

    n_alts = len(alt_names)
    col_spec = "l" + "r" * n_alts

    lines = [
        "% Marginal Effects of Fee (per 10,000 KRW)",
        "% Generated automatically - fragment for \\input{}",
        "",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Marginal Effects of Fee (per 10,000 KRW)}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        f"& \\multicolumn{{{n_alts}}}{{c}}{{Fee Change in Alternative}} \\\\",
        f"\\cmidrule(lr){{2-{n_alts+1}}}",
        "$\\Delta$ Probability & " + " & ".join(alt_names) + " \\\\",
        "\\midrule",
    ]

    for i, row_name in enumerate(alt_names):
        row_vals = []
        for j in range(n_alts):
            val = matrix[i, j]
            row_vals.append(f"{val:.4f}")
        lines.append(f"{row_name} & " + " & ".join(row_vals) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"  LaTeX output: {output_path}")
    return output_path


def generate_market_shares_latex(comparison, alt_names: list = None,
                                  output_dir: Path = None) -> Path:
    """
    Generate scenario comparison table for market shares.

    Args:
        comparison: ScenarioComparisonResult object or dict with keys:
                   'baseline_shares', 'policy_shares', 'share_changes', 'percent_changes'
        alt_names: List of alternative names
        output_dir: Base output directory (default: 'output/latex')

    Returns:
        Path to generated LaTeX file
    """
    import numpy as np

    if output_dir is None:
        output_dir = Path("output/latex")

    policy_dir = output_dir / "policy"
    policy_dir.mkdir(parents=True, exist_ok=True)
    output_path = policy_dir / "market_shares.tex"

    # Handle both object and dict input
    if hasattr(comparison, 'baseline_shares'):
        baseline = comparison.baseline_shares
        policy = comparison.policy_shares
        delta = comparison.share_changes
        pct_change = comparison.percent_changes
    else:
        baseline = comparison['baseline_shares']
        policy = comparison['policy_shares']
        delta = comparison['share_changes']
        pct_change = comparison['percent_changes']

    n_alts = len(baseline)
    if alt_names is None:
        alt_names = ['Paid Alt 1', 'Paid Alt 2', 'Standard'] if n_alts == 3 else [f"Alt {i+1}" for i in range(n_alts)]

    lines = [
        "% Scenario Comparison: Policy Impact on Market Shares",
        "% Generated automatically - fragment for \\input{}",
        "",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Scenario Comparison: Policy Impact on Market Shares}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Alternative & Baseline & Policy & $\\Delta$ Share & \\% Change \\\\",
        "\\midrule",
    ]

    for i, name in enumerate(alt_names):
        b = baseline[i] * 100  # Convert to percentage
        p = policy[i] * 100
        d = delta[i] * 100  # pp change
        # percent_changes is already in percentage form (e.g., 17.75 = 17.75%)
        pc = pct_change[i]

        # Format with sign for delta and percent change
        d_str = f"+{d:.1f}pp" if d >= 0 else f"{d:.1f}pp"
        pc_str = f"+{pc:.1f}\\%" if pc >= 0 else f"{pc:.1f}\\%"

        lines.append(f"{name} & {b:.1f}\\% & {p:.1f}\\% & {d_str} & {pc_str} \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"  LaTeX output: {output_path}")
    return output_path


def generate_welfare_latex(welfare_result: dict, output_dir: Path = None) -> Path:
    """
    Generate welfare analysis table (Consumer Surplus and Compensating Variation).

    Args:
        welfare_result: Dict with keys from WelfareAnalyzer methods:
                       'cs_baseline', 'cs_policy', 'cv', 'cv_se', 'ci_lower', 'ci_upper',
                       Optional: 'population', 'total_welfare_change', 'total_ci'
        output_dir: Base output directory (default: 'output/latex')

    Returns:
        Path to generated LaTeX file
    """
    if output_dir is None:
        output_dir = Path("output/latex")

    policy_dir = output_dir / "policy"
    policy_dir.mkdir(parents=True, exist_ok=True)
    output_path = policy_dir / "welfare_analysis.tex"

    lines = [
        "% Welfare Analysis: Compensating Variation",
        "% Generated automatically - fragment for \\input{}",
        "",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Welfare Analysis: Compensating Variation}",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Measure & Value & Std. Err. & 95\\% CI \\\\",
        "\\midrule",
    ]

    # Baseline CS
    cs_b = welfare_result.get('cs_baseline', welfare_result.get('logsum_baseline', 0))
    lines.append(f"Baseline CS (log-sum) & {cs_b:.4f} & -- & -- \\\\")

    # Policy CS
    cs_p = welfare_result.get('cs_policy', welfare_result.get('logsum_policy', 0))
    lines.append(f"Policy CS (log-sum) & {cs_p:.4f} & -- & -- \\\\")

    # Compensating Variation
    cv = welfare_result.get('cv', welfare_result.get('per_person_cv', 0))
    cv_se = welfare_result.get('cv_se', welfare_result.get('per_person_se', None))
    ci_l = welfare_result.get('ci_lower', None)
    ci_u = welfare_result.get('ci_upper', None)

    se_str = f"{cv_se:,.0f}" if cv_se else "--"
    ci_str = f"[{ci_l:,.0f}, {ci_u:,.0f}]" if ci_l is not None else "--"
    lines.append(f"Compensating Variation & {cv:,.0f} KRW & {se_str} & {ci_str} \\\\")

    # Population-level if available
    if 'population' in welfare_result or 'total_welfare_change' in welfare_result:
        pop = welfare_result.get('population', 0)
        total = welfare_result.get('total_welfare_change', 0)
        total_ci = welfare_result.get('total_ci', None)

        lines.append("\\midrule")
        lines.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{Population-level (N={pop:,})}}}} \\\\")

        if total_ci:
            total_ci_str = f"[{total_ci[0]/1e6:.1f}M, {total_ci[1]/1e6:.1f}M]"
        else:
            total_ci_str = "--"

        lines.append(f"Total Welfare Change & {total/1e6:.1f}M KRW & -- & {total_ci_str} \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"  LaTeX output: {output_path}")
    return output_path


def generate_sensitivity_latex(sensitivity_df, output_dir: Path = None) -> Path:
    """
    Generate sensitivity analysis table.

    Args:
        sensitivity_df: DataFrame with columns for fee changes and resulting shares.
                       Expected: 'pct_change' column and share columns for each alternative.
        output_dir: Base output directory (default: 'output/latex')

    Returns:
        Path to generated LaTeX file
    """
    if output_dir is None:
        output_dir = Path("output/latex")

    policy_dir = output_dir / "policy"
    policy_dir.mkdir(parents=True, exist_ok=True)
    output_path = policy_dir / "sensitivity.tex"

    # Identify share columns (anything with 'share' in name or numeric columns after pct_change)
    share_cols = [c for c in sensitivity_df.columns if 'share' in c.lower() or c.startswith('alt_')]
    if not share_cols:
        # Assume columns after 'pct_change' are share columns
        cols = list(sensitivity_df.columns)
        pct_idx = cols.index('pct_change') if 'pct_change' in cols else 0
        share_cols = cols[pct_idx+1:]

    n_alts = len(share_cols)
    col_spec = "r" + "r" * n_alts

    # Format column headers
    alt_headers = []
    for i, col in enumerate(share_cols):
        if 'alt' in col.lower():
            alt_headers.append(f"Alt {i+1} Share")
        else:
            alt_headers.append(_escape_latex(col))

    lines = [
        "% Sensitivity Analysis: Fee Changes",
        "% Generated automatically - fragment for \\input{}",
        "",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Sensitivity Analysis: Fee Changes}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        "Fee Change & " + " & ".join(alt_headers) + " \\\\",
        "\\midrule",
    ]

    for _, row in sensitivity_df.iterrows():
        pct = row.get('pct_change', row.iloc[0])
        pct_str = f"+{pct:.0f}\\%" if pct >= 0 else f"{pct:.0f}\\%"

        share_vals = []
        for col in share_cols:
            val = row[col]
            if isinstance(val, (int, float)):
                share_vals.append(f"{val*100:.1f}\\%")
            else:
                share_vals.append(str(val))

        lines.append(f"{pct_str} & " + " & ".join(share_vals) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"  LaTeX output: {output_path}")
    return output_path


def generate_all_policy_latex(estimation_result, scenario, config=None,
                               output_dir: Path = None) -> None:
    """
    Generate all policy analysis LaTeX tables.

    Convenience function that runs all policy analysis calculators and
    generates corresponding LaTeX tables.

    Args:
        estimation_result: Dict with 'betas' and 'std_errs' or EstimationResult object
        scenario: PolicyScenario object with attribute values
        config: PolicyAnalysisConfig object (optional)
        output_dir: Base output directory (default: 'output/latex')
    """
    import numpy as np

    # Import policy analysis classes
    try:
        from src.policy_analysis import (
            WTPCalculator, ElasticityCalculator, MarginalEffectCalculator,
            DemandForecaster, WelfareAnalyzer
        )
    except ImportError:
        print("  Warning: Could not import policy_analysis module")
        return

    if output_dir is None:
        output_dir = Path("output/latex")

    print("Generating policy analysis LaTeX outputs...")

    # 1. WTP
    try:
        wtp_calc = WTPCalculator(estimation_result)
        wtp_dur = wtp_calc.compute_wtp('B_DUR')
        generate_wtp_latex([wtp_dur], output_dir)
    except Exception as e:
        print(f"  Warning: Could not generate WTP table: {e}")

    # 2. Elasticity Matrix
    try:
        elast_calc = ElasticityCalculator(estimation_result)
        elast_matrix = elast_calc.elasticity_matrix(scenario)
        generate_elasticity_matrix_latex(elast_matrix, output_dir=output_dir)
    except Exception as e:
        print(f"  Warning: Could not generate elasticity matrix: {e}")

    # 3. Marginal Effects Matrix
    try:
        me_calc = MarginalEffectCalculator(estimation_result)
        me_matrix = me_calc.marginal_effect_matrix(scenario)
        generate_marginal_effects_latex(me_matrix, output_dir=output_dir)
    except Exception as e:
        print(f"  Warning: Could not generate marginal effects matrix: {e}")

    # 4. Market Shares (with a policy scenario)
    try:
        forecaster = DemandForecaster(estimation_result)
        # Create a policy scenario with reduced fee
        policy_scenario = scenario.with_modification('fee', 0, scenario.get_attribute('fee', 0) * 0.9)
        comparison = forecaster.compare_scenarios(scenario, policy_scenario)
        generate_market_shares_latex(comparison, output_dir=output_dir)
    except Exception as e:
        print(f"  Warning: Could not generate market shares table: {e}")

    # 5. Welfare Analysis
    try:
        analyzer = WelfareAnalyzer(estimation_result)
        policy_scenario = scenario.with_modification('fee', 0, scenario.get_attribute('fee', 0) * 0.9)
        cv_result = analyzer.compute_compensating_variation(scenario, policy_scenario)
        total = analyzer.total_welfare_change(scenario, policy_scenario, population=5000)

        welfare_dict = {
            'cs_baseline': cv_result.cs_baseline,
            'cs_policy': cv_result.cs_policy,
            'cv': cv_result.cv,
            'cv_se': cv_result.cv_se,
            'ci_lower': cv_result.ci_lower,
            'ci_upper': cv_result.ci_upper,
            'population': total.get('population', 5000),
            'total_welfare_change': total.get('total_welfare_change', 0),
        }
        generate_welfare_latex(welfare_dict, output_dir)
    except Exception as e:
        print(f"  Warning: Could not generate welfare table: {e}")

    # 6. Sensitivity Analysis
    try:
        forecaster = DemandForecaster(estimation_result)
        sensitivity = forecaster.sensitivity_analysis(scenario, 'fee', 0, [-20, -10, 0, 10, 20])
        generate_sensitivity_latex(sensitivity, output_dir)
    except Exception as e:
        print(f"  Warning: Could not generate sensitivity table: {e}")


if __name__ == '__main__':
    print("LaTeX Output Module")
    print("=" * 40)
    print("\nFunctions:")
    print("  generate_latex_output(results, model_name)")
    print("  cleanup_latex_output()")
    print("  generate_all_simulation_latex(df)")
    print("  generate_all_policy_latex(result, scenario)")
    print("\nUsage:")
    print("  from src.utils.latex_output import generate_latex_output")
    print("  generate_latex_output(biogeme_results, 'MNL-Basic')")
