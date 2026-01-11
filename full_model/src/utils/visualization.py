"""
Visualization Functions for DCM/HCM Models
==========================================

Publication-ready visualization tools for discrete choice model results.

Includes:
- Coefficient forest plots
- Marginal effects of latent variables
- Monte Carlo convergence plots
- Model comparison plots

Authors: Hakan Mülayim, Giray Girengir, Ataol Azeritürk
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Conditional imports for matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None


def _check_matplotlib():
    """Check if matplotlib is available."""
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: pip install matplotlib"
        )


def plot_coefficient_forest(results: Dict,
                            params: List[str] = None,
                            figsize: Tuple[int, int] = (10, 6),
                            title: str = "Parameter Estimates",
                            save_path: Path = None):
    """
    Create forest plot of coefficient estimates with confidence intervals.

    Args:
        results: Dict with structure:
                 {'params': {'name': value, ...},
                  'se': {'name': se, ...}} or
                 Biogeme-like result object
        params: List of parameter names to plot (default: all)
        figsize: Figure size tuple
        title: Plot title
        save_path: Path to save figure (optional)

    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()

    # Extract parameters and SEs
    if hasattr(results, 'get_beta_values'):
        # Biogeme result object
        betas = results.get_beta_values()
        try:
            se_dict = {p: results.get_parameter_std_err(p) for p in betas}
        except:
            se_dict = {p: 0 for p in betas}
    else:
        betas = results.get('params', results.get('betas', {}))
        se_dict = results.get('se', results.get('std_errs', {}))

    # Filter parameters
    if params is None:
        params = list(betas.keys())
    else:
        params = [p for p in params if p in betas]

    n_params = len(params)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot each parameter
    y_positions = np.arange(n_params)

    estimates = [betas[p] for p in params]
    ses = [se_dict.get(p, 0) for p in params]

    # 95% CI
    ci_lower = [e - 1.96 * s for e, s in zip(estimates, ses)]
    ci_upper = [e + 1.96 * s for e, s in zip(estimates, ses)]

    # Plot CIs
    for i, (low, high, est) in enumerate(zip(ci_lower, ci_upper, estimates)):
        color = 'tab:blue' if est >= 0 else 'tab:red'
        ax.hlines(i, low, high, color=color, linewidth=2, alpha=0.7)
        ax.plot(est, i, 'o', color=color, markersize=8)

    # Add zero line
    ax.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # Labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(params)
    ax.set_xlabel('Estimate (with 95% CI)')
    ax.set_title(title)

    # Add significance markers
    for i, (est, se) in enumerate(zip(estimates, ses)):
        if se > 0:
            t_stat = abs(est / se)
            if t_stat > 2.576:
                ax.annotate('***', (est, i), textcoords='offset points',
                           xytext=(5, 0), fontsize=10)
            elif t_stat > 1.96:
                ax.annotate('**', (est, i), textcoords='offset points',
                           xytext=(5, 0), fontsize=10)
            elif t_stat > 1.645:
                ax.annotate('*', (est, i), textcoords='offset points',
                           xytext=(5, 0), fontsize=10)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    return fig


def plot_marginal_effects_lv(result: Dict,
                             lv_name: str,
                             attribute: str = 'fee',
                             lv_range: Tuple[float, float] = (-2, 2),
                             n_points: int = 50,
                             figsize: Tuple[int, int] = (10, 6),
                             title: str = None,
                             save_path: Path = None):
    """
    Plot marginal effects of an attribute across latent variable values.

    Shows how the effect of an attribute (e.g., fee) changes as the
    latent variable (e.g., environmental concern) varies.

    Args:
        result: Estimation result with beta coefficients
        lv_name: Name of latent variable (e.g., 'env_concern')
        attribute: Attribute for marginal effect (e.g., 'fee', 'dur')
        lv_range: Range of LV values to plot
        n_points: Number of points
        figsize: Figure size
        title: Plot title
        save_path: Path to save figure

    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()

    # Extract relevant coefficients
    if hasattr(result, 'get_beta_values'):
        betas = result.get_beta_values()
    else:
        betas = result.get('params', result.get('betas', {}))

    # Find base attribute coefficient and interaction
    base_coef = None
    interaction_coef = None

    for param, value in betas.items():
        param_lower = param.lower()
        if attribute.lower() in param_lower and lv_name.lower() not in param_lower:
            if 'int' not in param_lower:  # Not an interaction
                base_coef = value
        if attribute.lower() in param_lower and lv_name.lower() in param_lower:
            interaction_coef = value

    if base_coef is None:
        raise ValueError(f"Could not find coefficient for {attribute}")

    if interaction_coef is None:
        interaction_coef = 0

    # Generate LV values
    lv_values = np.linspace(lv_range[0], lv_range[1], n_points)

    # Marginal effect: base + interaction * LV
    marginal_effects = base_coef + interaction_coef * lv_values

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(lv_values, marginal_effects, 'b-', linewidth=2)
    ax.axhline(base_coef, color='gray', linestyle='--', alpha=0.5,
               label=f'Base effect ({base_coef:.3f})')
    ax.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax.axvline(0, color='black', linestyle='-', alpha=0.3)

    # Shade regions
    ax.fill_between(lv_values, 0, marginal_effects,
                    where=(marginal_effects < 0),
                    alpha=0.2, color='red',
                    label='Negative effect (utility decreasing)')
    ax.fill_between(lv_values, 0, marginal_effects,
                    where=(marginal_effects >= 0),
                    alpha=0.2, color='green',
                    label='Positive effect (utility increasing)')

    ax.set_xlabel(f'Latent Variable: {lv_name}')
    ax.set_ylabel(f'Marginal Effect of {attribute}')
    ax.set_title(title or f'Marginal Effect of {attribute} by {lv_name}')
    ax.legend(loc='best')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    return fig


def plot_monte_carlo_convergence(mc_results,
                                 param: str,
                                 figsize: Tuple[int, int] = (12, 5),
                                 save_path: Path = None):
    """
    Plot Monte Carlo convergence: bias and RMSE across sample sizes.

    Args:
        mc_results: MonteCarloResult object or dict with bias/rmse by sample size
        param: Parameter name to plot
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()

    # Extract data
    if hasattr(mc_results, 'bias'):
        sample_sizes = mc_results.sample_sizes
        bias = [mc_results.bias[n].get(param, np.nan) for n in sample_sizes]
        rmse = [mc_results.rmse[n].get(param, np.nan) for n in sample_sizes]
        coverage = [mc_results.coverage[n].get(param, np.nan) for n in sample_sizes]
        true_val = mc_results.true_values.get(param, 0)
    else:
        sample_sizes = sorted(mc_results['sample_sizes'])
        bias = [mc_results['bias'][n].get(param, np.nan) for n in sample_sizes]
        rmse = [mc_results['rmse'][n].get(param, np.nan) for n in sample_sizes]
        coverage = [mc_results.get('coverage', {}).get(n, {}).get(param, np.nan) for n in sample_sizes]
        true_val = mc_results.get('true_values', {}).get(param, 0)

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Bias plot
    ax1 = axes[0]
    ax1.plot(sample_sizes, bias, 'bo-', linewidth=2, markersize=8)
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Sample Size')
    ax1.set_ylabel('Bias')
    ax1.set_title(f'Bias ({param})')
    ax1.set_xscale('log')

    # RMSE plot
    ax2 = axes[1]
    ax2.plot(sample_sizes, rmse, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Sample Size')
    ax2.set_ylabel('RMSE')
    ax2.set_title(f'RMSE ({param})')
    ax2.set_xscale('log')

    # Coverage plot
    ax3 = axes[2]
    ax3.plot(sample_sizes, coverage, 'go-', linewidth=2, markersize=8)
    ax3.axhline(0.95, color='gray', linestyle='--', alpha=0.5, label='Nominal (95%)')
    ax3.fill_between(sample_sizes, 0.90, 0.98, alpha=0.2, color='green',
                     label='Acceptable range')
    ax3.set_xlabel('Sample Size')
    ax3.set_ylabel('Coverage')
    ax3.set_title(f'95% CI Coverage ({param})')
    ax3.set_ylim(0.8, 1.0)
    ax3.set_xscale('log')
    ax3.legend(loc='lower right')

    plt.suptitle(f'Monte Carlo Convergence: {param} (True value = {true_val})',
                 fontsize=12, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    return fig


def plot_model_comparison(models: Dict[str, Dict],
                          metric: str = 'bic',
                          figsize: Tuple[int, int] = (10, 6),
                          title: str = None,
                          save_path: Path = None):
    """
    Bar chart comparing model fit statistics.

    Args:
        models: Dict mapping model name to result dict with metrics
                {'M0': {'ll': -1000, 'aic': 2100, 'bic': 2200}, ...}
        metric: Metric to compare ('ll', 'aic', 'bic')
        figsize: Figure size
        title: Plot title
        save_path: Path to save figure

    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()

    model_names = list(models.keys())
    values = []

    for name in model_names:
        result = models[name]
        if hasattr(result, 'get_general_statistics'):
            stats = result.get_general_statistics()
            if metric == 'll':
                values.append(stats.get('final_log_likelihood', 0))
            elif metric == 'aic':
                values.append(stats.get('akaike_information_criterion', 0))
            elif metric == 'bic':
                values.append(stats.get('bayesian_information_criterion', 0))
        else:
            values.append(result.get(metric, result.get(metric.upper(), 0)))

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(model_names)))
    bars = ax.bar(model_names, values, color=colors)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{val:.1f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)

    # Highlight best model
    if metric in ['aic', 'bic']:
        best_idx = np.argmin(values)
    else:  # log-likelihood
        best_idx = np.argmax(values)
    bars[best_idx].set_color('green')
    bars[best_idx].set_alpha(0.9)

    metric_labels = {
        'll': 'Log-Likelihood',
        'aic': 'AIC',
        'bic': 'BIC'
    }

    ax.set_ylabel(metric_labels.get(metric, metric.upper()))
    ax.set_title(title or f'Model Comparison: {metric_labels.get(metric, metric.upper())}')
    ax.set_xlabel('Model')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    return fig


def plot_attenuation_comparison(two_stage: Dict,
                                iclv: Dict,
                                true_values: Dict,
                                figsize: Tuple[int, int] = (10, 6),
                                save_path: Path = None):
    """
    Compare two-stage vs ICLV estimates showing attenuation bias.

    Args:
        two_stage: Dict of two-stage estimates {'param': value}
        iclv: Dict of ICLV estimates {'param': value}
        true_values: Dict of true values {'param': value}
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()

    # Common parameters
    params = [p for p in true_values if p in two_stage and p in iclv]
    n_params = len(params)

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(n_params)
    width = 0.25

    true_vals = [true_values[p] for p in params]
    ts_vals = [two_stage[p] for p in params]
    iclv_vals = [iclv[p] for p in params]

    # Plot bars
    bars1 = ax.bar(x - width, true_vals, width, label='True', color='gray', alpha=0.7)
    bars2 = ax.bar(x, ts_vals, width, label='Two-Stage', color='tab:red', alpha=0.7)
    bars3 = ax.bar(x + width, iclv_vals, width, label='ICLV', color='tab:green', alpha=0.7)

    # Labels
    ax.set_ylabel('Parameter Value')
    ax.set_title('Two-Stage vs ICLV: Attenuation Bias Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(params, rotation=45, ha='right')
    ax.legend()

    # Add bias annotations
    for i, (true, ts, iclv_v) in enumerate(zip(true_vals, ts_vals, iclv_vals)):
        if true != 0:
            ts_bias = (ts - true) / true * 100
            iclv_bias = (iclv_v - true) / true * 100

            ax.annotate(f'{ts_bias:+.1f}%',
                       xy=(i, ts),
                       xytext=(0, 5),
                       textcoords='offset points',
                       ha='center', fontsize=8, color='tab:red')
            ax.annotate(f'{iclv_bias:+.1f}%',
                       xy=(i + width, iclv_v),
                       xytext=(0, 5),
                       textcoords='offset points',
                       ha='center', fontsize=8, color='tab:green')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    return fig


def plot_lv_distribution(lv_scores: np.ndarray,
                         lv_name: str,
                         by_choice: np.ndarray = None,
                         choice_labels: List[str] = None,
                         figsize: Tuple[int, int] = (10, 6),
                         save_path: Path = None):
    """
    Plot distribution of latent variable scores.

    Args:
        lv_scores: Array of LV scores for each individual
        lv_name: Name of the latent variable
        by_choice: Optional choice outcomes for grouping
        choice_labels: Labels for choice groups
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)

    if by_choice is not None:
        # Plot distribution by choice
        unique_choices = np.unique(by_choice)
        if choice_labels is None:
            choice_labels = [f'Choice {c}' for c in unique_choices]

        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_choices)))

        for i, choice in enumerate(unique_choices):
            mask = by_choice == choice
            ax.hist(lv_scores[mask], bins=30, alpha=0.5,
                   label=choice_labels[i], color=colors[i], density=True)

        ax.legend()
    else:
        # Single distribution
        ax.hist(lv_scores, bins=50, alpha=0.7, color='steelblue', density=True)

    # Add normal reference
    x_range = np.linspace(lv_scores.min() - 0.5, lv_scores.max() + 0.5, 100)
    from scipy import stats
    normal_pdf = stats.norm.pdf(x_range, lv_scores.mean(), lv_scores.std())
    ax.plot(x_range, normal_pdf, 'r--', linewidth=2, label='Normal fit')

    ax.set_xlabel(f'{lv_name} Score')
    ax.set_ylabel('Density')
    ax.set_title(f'Distribution of {lv_name}')
    ax.legend()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    return fig


if __name__ == '__main__':
    print("Visualization Module")
    print("=" * 40)
    print("\nFunctions:")
    print("  plot_coefficient_forest(results, params)")
    print("  plot_marginal_effects_lv(result, lv_name, attribute)")
    print("  plot_monte_carlo_convergence(mc_results, param)")
    print("  plot_model_comparison(models, metric)")
    print("  plot_attenuation_comparison(two_stage, iclv, true)")
    print("  plot_lv_distribution(lv_scores, lv_name)")
