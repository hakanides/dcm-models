"""
Sample statistics generation for simulated DCM data.

Generates summary tables and plots for population characteristics after simulation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any

# Conditional matplotlib import
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for saving
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def generate_sample_stats(
    df: pd.DataFrame,
    config: Dict[str, Any],
    output_dir: Path,
    verbose: bool = True
) -> None:
    """
    Generate all sample statistics tables and plots.

    Args:
        df: Simulated choice data
        config: Model configuration dictionary
        output_dir: Directory to save outputs (sample_stats/)
        verbose: Whether to print progress messages
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 60)
        print("SAMPLE STATISTICS")
        print("=" * 60)

    # Generate summary tables
    demographics_summary(df, config, output_dir, verbose)
    choice_distribution(df, config, output_dir, verbose)
    attribute_summary(df, config, output_dir, verbose)

    # Check for Likert items (HCM/ICLV models)
    likert_cols = [c for c in df.columns if any(
        c.startswith(prefix) for prefix in ['pat_blind_', 'pat_constructive_', 'sec_dl_', 'sec_fp_']
    )]
    if likert_cols:
        likert_summary(df, likert_cols, output_dir, verbose)

    # Generate plots if matplotlib is available
    if HAS_MATPLOTLIB:
        plot_demographics(df, config, output_dir, verbose)
        plot_choice_distribution(df, output_dir, verbose)
        plot_attribute_distributions(df, config, output_dir, verbose)
        if likert_cols:
            plot_likert_distributions(df, likert_cols, output_dir, verbose)
    else:
        if verbose:
            print("  Note: matplotlib not available, skipping plots")

    if verbose:
        print(f"\nSample statistics saved to: {output_dir}")
        print("=" * 60 + "\n")


def demographics_summary(
    df: pd.DataFrame,
    config: Dict[str, Any],
    output_dir: Path,
    verbose: bool = True
) -> pd.DataFrame:
    """Generate demographics summary table."""
    # Get unique individuals
    df_indiv = df.drop_duplicates(subset=['ID'])

    summary_rows = []

    # Standard demographic columns
    demo_cols = ['age_idx', 'edu_idx', 'income_idx', 'income_indiv_idx']
    demo_labels = {
        'age_idx': 'Age Category',
        'edu_idx': 'Education Level',
        'income_idx': 'Household Income',
        'income_indiv_idx': 'Individual Income'
    }

    for col in demo_cols:
        if col in df_indiv.columns:
            values = df_indiv[col]
            summary_rows.append({
                'Variable': demo_labels.get(col, col),
                'N': len(values),
                'Mean': values.mean(),
                'Std': values.std(),
                'Min': values.min(),
                'Max': values.max(),
                'Unique': values.nunique()
            })

    # Centered versions
    centered_cols = ['age_c', 'edu_c', 'income_c']
    for col in centered_cols:
        if col in df_indiv.columns:
            values = df_indiv[col]
            summary_rows.append({
                'Variable': f'{col} (centered)',
                'N': len(values),
                'Mean': values.mean(),
                'Std': values.std(),
                'Min': values.min(),
                'Max': values.max(),
                'Unique': values.nunique()
            })

    summary_df = pd.DataFrame(summary_rows)

    if len(summary_df) > 0:
        # Save to CSV
        summary_df.to_csv(output_dir / 'demographics_summary.csv', index=False)
        if verbose:
            print(f"  Saved: demographics_summary.csv ({len(summary_df)} variables)")

    return summary_df


def choice_distribution(
    df: pd.DataFrame,
    config: Dict[str, Any],
    output_dir: Path,
    verbose: bool = True
) -> pd.DataFrame:
    """Generate choice distribution table."""
    if 'CHOICE' not in df.columns:
        return pd.DataFrame()

    # Overall choice distribution
    choice_counts = df['CHOICE'].value_counts().sort_index()
    total = len(df)

    rows = []
    for choice, count in choice_counts.items():
        rows.append({
            'Alternative': choice,
            'Count': count,
            'Percentage': 100 * count / total
        })

    choice_df = pd.DataFrame(rows)
    choice_df.to_csv(output_dir / 'choice_distribution.csv', index=False)

    if verbose:
        print(f"  Saved: choice_distribution.csv")
        for _, row in choice_df.iterrows():
            print(f"    Alt {int(row['Alternative'])}: {int(row['Count'])} ({row['Percentage']:.1f}%)")

    return choice_df


def attribute_summary(
    df: pd.DataFrame,
    config: Dict[str, Any],
    output_dir: Path,
    verbose: bool = True
) -> pd.DataFrame:
    """Generate attribute statistics table."""
    # Find fee and duration columns
    fee_cols = [c for c in df.columns if c.startswith('fee') and c != 'fee_scale']
    dur_cols = [c for c in df.columns if c.startswith('dur')]

    rows = []

    # Fee statistics
    for col in sorted(fee_cols):
        if col in df.columns:
            values = df[col]
            rows.append({
                'Attribute': col,
                'Mean': values.mean(),
                'Std': values.std(),
                'Min': values.min(),
                'Max': values.max(),
                'Median': values.median()
            })

    # Duration statistics
    for col in sorted(dur_cols):
        if col in df.columns:
            values = df[col]
            rows.append({
                'Attribute': col,
                'Mean': values.mean(),
                'Std': values.std(),
                'Min': values.min(),
                'Max': values.max(),
                'Median': values.median()
            })

    attr_df = pd.DataFrame(rows)
    if len(attr_df) > 0:
        attr_df.to_csv(output_dir / 'attribute_summary.csv', index=False)
        if verbose:
            print(f"  Saved: attribute_summary.csv ({len(attr_df)} attributes)")

    return attr_df


def likert_summary(
    df: pd.DataFrame,
    likert_cols: List[str],
    output_dir: Path,
    verbose: bool = True
) -> pd.DataFrame:
    """Generate Likert item summary for HCM/ICLV models."""
    # Get unique individuals
    df_indiv = df.drop_duplicates(subset=['ID'])

    rows = []
    for col in sorted(likert_cols):
        if col in df_indiv.columns:
            values = df_indiv[col]
            # Value distribution
            value_counts = values.value_counts().sort_index()
            dist_str = ', '.join([f"{int(v)}:{int(c)}" for v, c in value_counts.items()])

            rows.append({
                'Item': col,
                'Mean': values.mean(),
                'Std': values.std(),
                'Mode': values.mode().iloc[0] if len(values.mode()) > 0 else np.nan,
                'Distribution': dist_str
            })

    likert_df = pd.DataFrame(rows)
    likert_df.to_csv(output_dir / 'likert_summary.csv', index=False)

    if verbose:
        print(f"  Saved: likert_summary.csv ({len(likert_cols)} items)")

    return likert_df


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_demographics(
    df: pd.DataFrame,
    config: Dict[str, Any],
    output_dir: Path,
    verbose: bool = True
) -> None:
    """Plot demographic distributions."""
    if not HAS_MATPLOTLIB:
        return

    df_indiv = df.drop_duplicates(subset=['ID'])

    # Find available demographic columns
    demo_cols = [c for c in ['age_idx', 'edu_idx', 'income_idx', 'income_indiv_idx']
                 if c in df_indiv.columns]

    if not demo_cols:
        return

    n_cols = len(demo_cols)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]

    labels = {
        'age_idx': 'Age Category',
        'edu_idx': 'Education Level',
        'income_idx': 'Household Income',
        'income_indiv_idx': 'Individual Income'
    }

    for ax, col in zip(axes, demo_cols):
        values = df_indiv[col].value_counts().sort_index()
        ax.bar(values.index, values.values, color='steelblue', edgecolor='black')
        ax.set_xlabel(labels.get(col, col))
        ax.set_ylabel('Count')
        ax.set_title(f'{labels.get(col, col)} Distribution')

    plt.tight_layout()
    plt.savefig(output_dir / 'demographics_plot.png', dpi=150, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"  Saved: demographics_plot.png")


def plot_choice_distribution(
    df: pd.DataFrame,
    output_dir: Path,
    verbose: bool = True
) -> None:
    """Plot choice distribution."""
    if not HAS_MATPLOTLIB:
        return

    if 'CHOICE' not in df.columns:
        return

    choice_counts = df['CHOICE'].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ['#2ecc71', '#3498db', '#e74c3c'][:len(choice_counts)]
    ax.bar(choice_counts.index, choice_counts.values, color=colors, edgecolor='black')
    ax.set_xlabel('Alternative')
    ax.set_ylabel('Count')
    ax.set_title('Choice Distribution')
    ax.set_xticks(choice_counts.index)

    # Add percentage labels
    total = choice_counts.sum()
    for i, (idx, count) in enumerate(choice_counts.items()):
        pct = 100 * count / total
        ax.annotate(f'{pct:.1f}%', (idx, count), ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_dir / 'choice_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"  Saved: choice_distribution.png")


def plot_attribute_distributions(
    df: pd.DataFrame,
    config: Dict[str, Any],
    output_dir: Path,
    verbose: bool = True
) -> None:
    """Plot attribute distributions (fee, duration)."""
    if not HAS_MATPLOTLIB:
        return

    # Find fee and duration columns for alternative 1
    fee_cols = [c for c in ['fee1', 'fee2', 'fee3'] if c in df.columns]
    dur_cols = [c for c in ['dur1', 'dur2', 'dur3'] if c in df.columns]

    n_plots = (1 if fee_cols else 0) + (1 if dur_cols else 0)
    if n_plots == 0:
        return

    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    plot_idx = 0

    # Fee distribution
    if fee_cols:
        ax = axes[plot_idx]
        # Stack all fee values
        all_fees = pd.concat([df[c] for c in fee_cols])
        ax.hist(all_fees, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Fee (TL)')
        ax.set_ylabel('Count')
        ax.set_title('Fee Distribution (All Alternatives)')
        plot_idx += 1

    # Duration distribution
    if dur_cols:
        ax = axes[plot_idx]
        all_durs = pd.concat([df[c] for c in dur_cols])
        ax.hist(all_durs, bins=20, color='coral', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Duration (days)')
        ax.set_ylabel('Count')
        ax.set_title('Duration Distribution (All Alternatives)')

    plt.tight_layout()
    plt.savefig(output_dir / 'attribute_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"  Saved: attribute_distributions.png")


def plot_likert_distributions(
    df: pd.DataFrame,
    likert_cols: List[str],
    output_dir: Path,
    verbose: bool = True
) -> None:
    """Plot Likert item distributions for HCM/ICLV models."""
    if not HAS_MATPLOTLIB:
        return

    df_indiv = df.drop_duplicates(subset=['ID'])

    # Group by construct
    constructs = {}
    for col in likert_cols:
        prefix = '_'.join(col.split('_')[:-1])  # e.g., 'pat_blind' from 'pat_blind_1'
        if prefix not in constructs:
            constructs[prefix] = []
        constructs[prefix].append(col)

    n_constructs = len(constructs)
    if n_constructs == 0:
        return

    # Create subplot for each construct
    fig, axes = plt.subplots(1, n_constructs, figsize=(5 * n_constructs, 4))
    if n_constructs == 1:
        axes = [axes]

    for ax, (construct, cols) in zip(axes, constructs.items()):
        # Compute mean response for each item
        means = [df_indiv[col].mean() for col in sorted(cols)]
        item_nums = range(1, len(cols) + 1)

        ax.bar(item_nums, means, color='mediumpurple', edgecolor='black')
        ax.set_xlabel('Item Number')
        ax.set_ylabel('Mean Response (1-5)')
        ax.set_title(f'{construct.replace("_", " ").title()}')
        ax.set_ylim(1, 5)
        ax.set_xticks(item_nums)
        ax.axhline(y=3, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / 'likert_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"  Saved: likert_distributions.png")


def plot_choice_by_demographics(
    df: pd.DataFrame,
    output_dir: Path,
    verbose: bool = True
) -> None:
    """Plot choice distribution by demographic groups."""
    if not HAS_MATPLOTLIB:
        return

    if 'CHOICE' not in df.columns:
        return

    # Find demographic column
    demo_col = None
    for col in ['age_idx', 'edu_idx', 'income_idx']:
        if col in df.columns:
            demo_col = col
            break

    if demo_col is None:
        return

    # Compute choice shares by demographic group
    crosstab = pd.crosstab(df[demo_col], df['CHOICE'], normalize='index') * 100

    fig, ax = plt.subplots(figsize=(8, 5))

    crosstab.plot(kind='bar', ax=ax, width=0.8, edgecolor='black')
    ax.set_xlabel(demo_col.replace('_', ' ').title())
    ax.set_ylabel('Choice Share (%)')
    ax.set_title(f'Choice Distribution by {demo_col.replace("_", " ").title()}')
    ax.legend(title='Alternative', loc='upper right')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    plt.tight_layout()
    plt.savefig(output_dir / 'choice_by_demographics.png', dpi=150, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"  Saved: choice_by_demographics.png")
