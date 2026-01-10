"""
Validation Script: Compare Estimated vs True Parameters
=======================================================

This script validates the DCM simulation by:
1. Loading simulated data with known true parameters
2. Estimating choice model using Biogeme
3. Comparing estimated vs true parameters
4. Computing validation metrics (bias, RMSE, coverage)
5. Generating diagnostic plots

This is crucial for:
- Validating the data generating process
- Testing estimation code before real data
- Understanding identification and precision

Authors: Hakan Mülayim, Giray Girengir, Ataol Azeritürk
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import warnings
# Selective warning suppression - allow important warnings through
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*overflow.*')
warnings.filterwarnings('ignore', message='.*divide by zero.*')

# Biogeme imports
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta, Variable, log, exp


# =============================================================================
# DATA PREPARATION
# =============================================================================

def load_simulated_data(data_path: str) -> pd.DataFrame:
    """Load and prepare simulated data for estimation."""
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} observations from {len(df['ID'].unique())} respondents")
    return df


def prepare_biogeme_data(df: pd.DataFrame) -> db.Database:
    """
    Prepare data for Biogeme estimation.

    Creates wide-format data with all necessary variables.
    """
    # Scale fees for numerical stability (divide by 10k)
    df = df.copy()
    df['fee1_10k'] = df['fee1'] / 10000.0
    df['fee2_10k'] = df['fee2'] / 10000.0
    df['fee3_10k'] = df['fee3'] / 10000.0

    # Drop string columns (Biogeme requires numeric only)
    string_cols = df.select_dtypes(include=['object']).columns.tolist()
    if string_cols:
        print(f"Dropping non-numeric columns: {string_cols}")
        df = df.drop(columns=string_cols)

    # Create database
    database = db.Database('simulated_dcm', df)

    return database


# =============================================================================
# MODEL SPECIFICATION
# =============================================================================

def define_mnl_model(database: db.Database):
    """
    Define a Multinomial Logit model for estimation.

    This is a simpler model (no random coefficients) to validate
    the basic structure. For Mixed Logit validation, use define_mixl_model.
    """
    # Define variables
    dur1 = Variable('dur1')
    dur2 = Variable('dur2')
    dur3 = Variable('dur3')
    fee1_10k = Variable('fee1_10k')
    fee2_10k = Variable('fee2_10k')
    fee3_10k = Variable('fee3_10k')
    CHOICE = Variable('CHOICE')

    # Parameters to estimate
    ASC_paid = Beta('ASC_paid', 0, None, None, 0)  # ASC for paid alternatives
    ASC_std = 0  # Fixed to 0 (base alternative normalization)

    B_FEE = Beta('B_FEE', -1, None, 0, 0)          # Fee coefficient (constrained negative)
    B_DUR = Beta('B_DUR', -0.1, None, 0, 0)        # Duration coefficient (constrained negative)

    # Utility functions
    V1 = ASC_paid + B_FEE * fee1_10k + B_DUR * dur1  # paid1
    V2 = ASC_paid + B_FEE * fee2_10k + B_DUR * dur2  # paid2
    V3 = ASC_std + B_FEE * fee3_10k + B_DUR * dur3   # standard

    # Associate utilities with alternatives
    V = {1: V1, 2: V2, 3: V3}

    # Availability (all alternatives always available)
    av = {1: 1, 2: 1, 3: 1}

    # Log-likelihood
    logprob = models.loglogit(V, av, CHOICE)

    return logprob, ['ASC_paid', 'B_FEE', 'B_DUR']


def define_mnl_with_demographics(database: db.Database):
    """
    MNL model with demographic interactions.

    This captures systematic taste heterogeneity.
    """
    # Variables
    dur1 = Variable('dur1')
    dur2 = Variable('dur2')
    dur3 = Variable('dur3')
    fee1_10k = Variable('fee1_10k')
    fee2_10k = Variable('fee2_10k')
    fee3_10k = Variable('fee3_10k')
    CHOICE = Variable('CHOICE')

    # Demographics
    age_idx = Variable('age_idx')
    edu_idx = Variable('edu_idx')
    income_indiv_idx = Variable('income_indiv_idx')

    # Base parameters
    ASC_paid = Beta('ASC_paid', 0, None, None, 0)
    B_FEE = Beta('B_FEE', -1, None, 0, 0)
    B_DUR = Beta('B_DUR', -0.1, None, 0, 0)

    # Interaction parameters
    B_FEE_AGE = Beta('B_FEE_AGE', 0, None, None, 0)
    B_FEE_EDU = Beta('B_FEE_EDU', 0, None, None, 0)
    B_FEE_INC = Beta('B_FEE_INC', 0, None, None, 0)

    # Centered demographics
    age_c = (age_idx - 2) / 2
    edu_c = (edu_idx - 3) / 2
    inc_c = (income_indiv_idx - 3) / 2

    # Individual-specific fee coefficient
    B_FEE_i = B_FEE + B_FEE_AGE * age_c + B_FEE_EDU * edu_c + B_FEE_INC * inc_c

    # Utilities
    V1 = ASC_paid + B_FEE_i * fee1_10k + B_DUR * dur1
    V2 = ASC_paid + B_FEE_i * fee2_10k + B_DUR * dur2
    V3 = B_FEE_i * fee3_10k + B_DUR * dur3

    V = {1: V1, 2: V2, 3: V3}
    av = {1: 1, 2: 1, 3: 1}

    logprob = models.loglogit(V, av, CHOICE)

    param_names = ['ASC_paid', 'B_FEE', 'B_DUR',
                   'B_FEE_AGE', 'B_FEE_EDU', 'B_FEE_INC']

    return logprob, param_names


# =============================================================================
# ESTIMATION
# =============================================================================

def estimate_model(database: db.Database, logprob, model_name: str = "dcm_model"):
    """Run Biogeme estimation."""
    print(f"\nEstimating {model_name}...")
    print("-" * 50)

    biogeme_model = bio.BIOGEME(database, logprob)

    # Estimate
    results = biogeme_model.estimate()

    return results


# =============================================================================
# VALIDATION METRICS
# =============================================================================

def compute_true_parameter_means(df: pd.DataFrame) -> dict:
    """
    Compute mean true parameters from simulated data.

    For the base MNL model, we average the individual-specific betas.
    """
    true_params = {}

    # Get unique individuals (parameters are constant within individual)
    individuals = df.drop_duplicates('ID')

    # ASC - from config (0.25 for paid)
    true_params['ASC_paid'] = 0.25

    # Average betas across individuals
    if 'beta_b_fee10k_true' in individuals.columns:
        true_params['B_FEE'] = individuals['beta_b_fee10k_true'].mean()

    if 'beta_b_dur_true' in individuals.columns:
        true_params['B_DUR'] = individuals['beta_b_dur_true'].mean()

    return true_params


def compare_parameters(results, true_params: dict) -> pd.DataFrame:
    """
    Compare estimated parameters to true values.

    Returns DataFrame with estimates, true values, bias, and coverage.
    """
    comparison = []

    betas = results.get_beta_values()

    for param_name, estimated in betas.items():
        if param_name in true_params:
            true_val = true_params[param_name]
            try:
                se = results.get_parameter_std_err(param_name)
            except:
                se = np.nan

            # Compute metrics
            bias = estimated - true_val
            bias_pct = (bias / abs(true_val) * 100) if true_val != 0 else np.nan

            # 95% CI coverage
            ci_lower = estimated - 1.96 * se
            ci_upper = estimated + 1.96 * se
            covered = ci_lower <= true_val <= ci_upper

            comparison.append({
                'Parameter': param_name,
                'True': true_val,
                'Estimated': estimated,
                'Std.Error': se,
                't-stat': estimated / se if se > 0 else np.nan,
                'Bias': bias,
                'Bias%': bias_pct,
                'CI_Lower': ci_lower,
                'CI_Upper': ci_upper,
                'Covered': covered
            })

    return pd.DataFrame(comparison)


def print_comparison_table(comparison_df: pd.DataFrame):
    """Print formatted comparison table."""
    print("\n" + "=" * 80)
    print("PARAMETER VALIDATION: Estimated vs True")
    print("=" * 80)

    for _, row in comparison_df.iterrows():
        print(f"\n{row['Parameter']}:")
        print(f"  True value:    {row['True']:>10.4f}")
        print(f"  Estimated:     {row['Estimated']:>10.4f} (SE: {row['Std.Error']:.4f})")
        print(f"  Bias:          {row['Bias']:>10.4f} ({row['Bias%']:+.1f}%)")
        print(f"  95% CI:        [{row['CI_Lower']:.4f}, {row['CI_Upper']:.4f}]")
        print(f"  True in CI:    {'Yes' if row['Covered'] else 'NO - POTENTIAL ISSUE'}")

    print("\n" + "-" * 80)
    print("Summary:")
    print(f"  Parameters validated: {len(comparison_df)}")
    print(f"  Coverage rate: {comparison_df['Covered'].mean()*100:.1f}% (should be ~95%)")
    print(f"  Mean absolute bias: {comparison_df['Bias'].abs().mean():.4f}")
    print("=" * 80)


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_parameter_comparison(comparison_df: pd.DataFrame, output_path: str = None):
    """Create visual comparison of estimated vs true parameters."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Point estimates with CI
    ax1 = axes[0]
    params = comparison_df['Parameter'].values
    y_pos = np.arange(len(params))

    ax1.errorbar(comparison_df['Estimated'], y_pos,
                 xerr=1.96*comparison_df['Std.Error'],
                 fmt='o', color='blue', capsize=5, label='Estimated (95% CI)')
    ax1.scatter(comparison_df['True'], y_pos,
               marker='x', color='red', s=100, zorder=5, label='True value')

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(params)
    ax1.set_xlabel('Parameter Value')
    ax1.set_title('Estimated vs True Parameters')
    ax1.legend()
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Bias
    ax2 = axes[1]
    colors = ['green' if c else 'red' for c in comparison_df['Covered']]
    bars = ax2.barh(y_pos, comparison_df['Bias%'], color=colors, alpha=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(params)
    ax2.set_xlabel('Bias (%)')
    ax2.set_title('Estimation Bias')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax2.axvline(x=-10, color='red', linestyle='--', alpha=0.5)
    ax2.axvline(x=10, color='red', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")

    plt.show()


def plot_beta_distributions(df: pd.DataFrame, output_path: str = None):
    """Plot distributions of true individual-specific betas."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Get unique individuals
    individuals = df.drop_duplicates('ID')

    beta_cols = ['beta_b_fee10k_true', 'beta_b_dur_true']
    titles = ['Fee Coefficient (β_fee)', 'Duration Coefficient (β_dur)']

    for ax, col, title in zip(axes, beta_cols, titles):
        if col in individuals.columns:
            values = individuals[col].values
            ax.hist(values, bins=30, density=True, alpha=0.7, edgecolor='black')
            ax.axvline(values.mean(), color='red', linestyle='--',
                      label=f'Mean: {values.mean():.3f}')
            ax.axvline(0, color='gray', linestyle='-', alpha=0.5)
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nDistribution plot saved to: {output_path}")

    plt.show()


def plot_latent_variable_distributions(df: pd.DataFrame, output_path: str = None):
    """Plot distributions of true latent variables."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    individuals = df.drop_duplicates('ID')

    lv_cols = [
        ('LV_pat_blind_true', 'Patriotism: Blind'),
        ('LV_pat_constructive_true', 'Patriotism: Constructive'),
        ('LV_sec_dl_true', 'Secularism: Daily Life'),
        ('LV_sec_fp_true', 'Secularism: Faith & Prayer')
    ]

    for ax, (col, title) in zip(axes.flat, lv_cols):
        if col in individuals.columns:
            values = individuals[col].values
            ax.hist(values, bins=30, density=True, alpha=0.7, edgecolor='black')
            ax.axvline(values.mean(), color='red', linestyle='--',
                      label=f'Mean: {values.mean():.2f}\nSD: {values.std():.2f}')
            ax.axvline(0, color='gray', linestyle='-', alpha=0.5)
            ax.set_xlabel('Latent Variable Value')
            ax.set_ylabel('Density')
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nLatent variable plot saved to: {output_path}")

    plt.show()


# =============================================================================
# MAIN VALIDATION PIPELINE
# =============================================================================

def run_validation(
    data_path: str,
    output_dir: str = None,
    model_type: str = 'simple'
):
    """
    Run complete validation pipeline.

    Args:
        data_path: Path to simulated data CSV
        output_dir: Directory for output files (plots, results)
        model_type: 'simple' for basic MNL, 'demographics' for MNL with interactions
    """
    print("=" * 70)
    print("DCM SIMULATION VALIDATION")
    print("Comparing Estimated vs True Parameters")
    print("=" * 70)

    # Load data
    df = load_simulated_data(data_path)

    # Prepare for Biogeme
    database = prepare_biogeme_data(df)

    # Define and estimate model
    if model_type == 'simple':
        logprob, param_names = define_mnl_model(database)
        model_name = 'mnl_simple'
    else:
        logprob, param_names = define_mnl_with_demographics(database)
        model_name = 'mnl_demographics'

    results = estimate_model(database, logprob, model_name)

    # Print estimation results
    print("\n" + "=" * 70)
    print("ESTIMATION RESULTS")
    print("=" * 70)
    print(results.short_summary())

    # Get true parameters
    true_params = compute_true_parameter_means(df)
    print("\nTrue parameter means (from simulation):")
    for k, v in true_params.items():
        print(f"  {k}: {v:.4f}")

    # Compare
    comparison_df = compare_parameters(results, true_params)
    print_comparison_table(comparison_df)

    # Save comparison to CSV
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        comparison_df.to_csv(output_dir / 'parameter_comparison.csv', index=False)
        print(f"\nComparison saved to: {output_dir / 'parameter_comparison.csv'}")

    # Generate plots
    print("\nGenerating diagnostic plots...")

    if output_dir:
        plot_parameter_comparison(comparison_df, str(output_dir / 'param_comparison.png'))
        plot_beta_distributions(df, str(output_dir / 'beta_distributions.png'))
        plot_latent_variable_distributions(df, str(output_dir / 'latent_distributions.png'))
    else:
        plot_parameter_comparison(comparison_df)
        plot_beta_distributions(df)
        plot_latent_variable_distributions(df)

    return results, comparison_df


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Validate DCM simulation by comparing estimated vs true parameters',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python validate_estimation.py --data test_advanced_full.csv
  python validate_estimation.py --data test_advanced_full.csv --output validation_results
  python validate_estimation.py --data test_advanced_full.csv --model demographics
        """
    )
    parser.add_argument('--data', required=True, help='Path to simulated data CSV')
    parser.add_argument('--output', default=None, help='Output directory for results')
    parser.add_argument('--model', choices=['simple', 'demographics'], default='simple',
                       help='Model type to estimate')

    args = parser.parse_args()

    run_validation(args.data, args.output, args.model)


if __name__ == '__main__':
    main()
