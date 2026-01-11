"""
Final Model Comparison
======================

Compares all estimated models:
- MNL models (5 specifications)
- MXL models (3 specifications)
- HCM models (15 specifications)

Authors: Hakan Mülayim, Giray Girengir, Ataol Azeritürk
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def find_result_file(base_path: str, fallback_names: list = None) -> Path:
    """Find result file, trying fallback names if primary doesn't exist."""
    path = Path(base_path)
    if path.exists():
        return path

    if fallback_names:
        parent = path.parent
        for name in fallback_names:
            fallback = parent / name
            if fallback.exists():
                return fallback

    return path  # Return original even if not found (for error message)


def load_results(mnl_path: str, mxl_path: str, hcm_path: str):
    """Load all model results from specified paths."""
    results = []

    # MNL results - try multiple possible names
    mnl_file = find_result_file(mnl_path, ['model_comparison.csv', 'mnl_comparison.csv'])
    if mnl_file.exists():
        mnl_df = pd.read_csv(mnl_file)
        mnl_df['Type'] = 'MNL'
        # Add converged column if missing
        if 'Converged' not in mnl_df.columns:
            mnl_df['Converged'] = True
        results.append(mnl_df)
        print(f"Loaded {len(mnl_df)} MNL models from {mnl_file}")
    else:
        print(f"MNL results not found: {mnl_path}")

    # MXL results - try multiple possible names
    mxl_file = find_result_file(mxl_path, ['mxl_comparison.csv', 'model_comparison.csv'])
    if mxl_file.exists():
        mxl_df = pd.read_csv(mxl_file)
        mxl_df['Type'] = 'MXL'
        # Add rho-squared if missing (estimate from null LL)
        if 'ρ²' not in mxl_df.columns:
            # Estimate null LL from n_obs (assume ~5000 obs, 3 alts)
            null_ll = 5000 * np.log(1/3)
            mxl_df['ρ²'] = 1 - (mxl_df['LL'] / null_ll)
        # Add converged column if missing (check Conv column)
        if 'Converged' not in mxl_df.columns:
            if 'Conv' in mxl_df.columns:
                mxl_df['Converged'] = mxl_df['Conv'] == 'Yes'
            else:
                mxl_df['Converged'] = True
        results.append(mxl_df)
        print(f"Loaded {len(mxl_df)} MXL models from {mxl_file}")
    else:
        print(f"MXL results not found: {mxl_path}")

    # HCM results - try multiple possible names
    hcm_file = find_result_file(hcm_path, ['hcm_comparison.csv', 'model_comparison.csv'])
    if hcm_file.exists():
        hcm_df = pd.read_csv(hcm_file)
        hcm_df['Type'] = 'HCM'
        # Rename rho2 to ρ² if needed
        if 'rho2' in hcm_df.columns and 'ρ²' not in hcm_df.columns:
            hcm_df['ρ²'] = hcm_df['rho2']
        # Add converged column if missing
        if 'Converged' not in hcm_df.columns:
            hcm_df['Converged'] = True
        # Filter out bad models (negative rho²)
        hcm_df = hcm_df[hcm_df['ρ²'] > 0]
        results.append(hcm_df)
        print(f"Loaded {len(hcm_df)} HCM models from {hcm_file}")
    else:
        print(f"HCM results not found: {hcm_file}")

    if not results:
        print("No results found!")
        return None

    # Combine
    all_results = pd.concat(results, ignore_index=True)

    # Standardize columns
    if 'Adj.ρ²' not in all_results.columns:
        all_results['Adj.ρ²'] = all_results.get('Adj. ρ²', all_results['ρ²'])

    return all_results


def create_comparison_table(df: pd.DataFrame, converged_only: bool = True) -> pd.DataFrame:
    """Create formatted comparison table.

    Args:
        df: DataFrame with model results
        converged_only: If True, only include converged models in ranking
    """
    # Filter by convergence if requested
    if converged_only and 'Converged' in df.columns:
        df_filtered = df[df['Converged'] == True].copy()
        if len(df_filtered) == 0:
            print("Warning: No converged models found, showing all")
            df_filtered = df.copy()
    else:
        df_filtered = df.copy()

    cols = ['Type', 'Model', 'LL', 'K', 'AIC', 'BIC', 'ρ²', 'Converged']
    available_cols = [c for c in cols if c in df_filtered.columns]

    table = df_filtered[available_cols].copy()
    table = table.sort_values('AIC')
    table['Rank'] = range(1, len(table) + 1)

    return table


def plot_comparison(df: pd.DataFrame, output_path: str = None):
    """Create visual comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Sort by AIC
    df = df.sort_values('AIC')

    # Color by type
    colors = {'MNL': 'steelblue', 'MXL': 'coral', 'HCM': 'seagreen'}
    bar_colors = [colors.get(t, 'gray') for t in df['Type']]

    x = np.arange(len(df))
    labels = df['Model'].str[:15]  # Truncate names

    # AIC
    ax = axes[0]
    ax.barh(x, df['AIC'], color=bar_colors, alpha=0.8)
    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('AIC')
    ax.set_title('AIC (lower = better)')
    ax.invert_yaxis()

    # BIC
    ax = axes[1]
    ax.barh(x, df['BIC'], color=bar_colors, alpha=0.8)
    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('BIC')
    ax.set_title('BIC (lower = better)')
    ax.invert_yaxis()

    # Rho-squared
    ax = axes[2]
    ax.barh(x, df['ρ²'], color=bar_colors, alpha=0.8)
    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('ρ²')
    ax.set_title('Rho-squared (higher = better)')
    ax.invert_yaxis()

    # Legend
    handles = [plt.Rectangle((0,0),1,1, color=c, alpha=0.8) for c in colors.values()]
    fig.legend(handles, colors.keys(), loc='upper right', bbox_to_anchor=(0.99, 0.99))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {output_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Compare MNL, MXL, and HCM model results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python final_comparison.py --mnl results/mnl_basic/model_comparison.csv \\
                             --mxl results/mxl_basic/mxl_comparison.csv \\
                             --hcm results/hcm_basic/hcm_comparison.csv

Note: This script compares outputs from individual model scripts.
      For comprehensive analysis, use scripts/run_all_models.py instead.
        """
    )
    parser.add_argument('--mnl', default=None,
                        help='Path to MNL comparison CSV (required)')
    parser.add_argument('--mxl', default=None,
                        help='Path to MXL comparison CSV (required)')
    parser.add_argument('--hcm', default=None,
                        help='Path to HCM comparison CSV (required)')
    parser.add_argument('--output', default='results/final_comparison',
                        help='Output directory')
    parser.add_argument('--all-models', action='store_true',
                        help='Include non-converged models in comparison')

    args = parser.parse_args()

    print("=" * 60)
    print("FINAL MODEL COMPARISON")
    print("MNL vs MXL vs HCM")
    print("=" * 60)

    # Load results
    df = load_results(args.mnl, args.mxl, args.hcm)

    if df is None:
        return

    # Create comparison
    converged_only = not args.all_models
    table = create_comparison_table(df, converged_only=converged_only)

    print("\n" + "=" * 60)
    print("ALL MODELS RANKED BY AIC")
    if converged_only:
        print("(Only converged models)")
    print("=" * 60)
    print("\n" + table.to_string(index=False))

    # Best models by type
    print("\n" + "=" * 60)
    print("BEST MODEL BY TYPE")
    print("=" * 60)

    for model_type in ['MNL', 'MXL', 'HCM']:
        type_df = df[df['Type'] == model_type]
        if converged_only and 'Converged' in type_df.columns:
            type_df = type_df[type_df['Converged'] == True]
        if len(type_df) > 0:
            best = type_df.loc[type_df['AIC'].idxmin()]
            print(f"\n{model_type}: {best['Model']}")
            print(f"  LL: {best['LL']:.2f}")
            print(f"  AIC: {best['AIC']:.2f}")
            print(f"  BIC: {best['BIC']:.2f}")
            print(f"  ρ²: {best['ρ²']:.4f}")

    # Overall best
    print("\n" + "=" * 60)
    print("OVERALL WINNER")
    print("=" * 60)

    best_aic = table.iloc[0]
    best_bic = table.loc[table['BIC'].idxmin()]

    print(f"\nBest by AIC: {best_aic['Model']} ({best_aic['Type']})")
    print(f"  AIC = {best_aic['AIC']:.2f}")

    print(f"\nBest by BIC: {best_bic['Model']} ({best_bic['Type']})")
    print(f"  BIC = {best_bic['BIC']:.2f}")

    # Save
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    table.to_csv(output_dir / 'all_models_comparison.csv', index=False)
    plot_comparison(df, str(output_dir / 'model_comparison_plot.png'))

    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
