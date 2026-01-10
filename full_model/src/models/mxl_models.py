"""
Mixed Logit (MXL) Models
========================

Estimates Mixed Logit models with random coefficients to capture
unobserved taste heterogeneity.

Models:
1. MXL-1: Random fee coefficient only
2. MXL-2: Random fee + duration coefficients
3. MXL-3: Random all attribute coefficients
4. MXL-4: Correlated random coefficients

IMPORTANT: DATA GENERATION COMPATIBILITY
========================================
MXL models estimate random coefficient heterogeneity (σ parameters).
The estimated σ will reflect what's in the DATA GENERATING PROCESS:

- model_config.json: NO random coefficients defined
  → MXL will estimate σ ≈ 0 (no heterogeneity to find)
  → Only demographic/LV interactions create systematic heterogeneity

- model_config.json: HAS random coefficients
  → MXL will estimate σ > 0 matching the true values
  → b_fee10k: mean=0, std=0.3
  → b_dur: mean=0, std=0.02

If MXL estimates σ ≈ 0 on your data, it's not a bug - it means the data
was generated without random taste heterogeneity. Use model_config.json
to generate data with true random coefficients for MXL validation.

Authors: Hakan Mülayim, Giray Girengir, Ataol Azeritürk
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil
from pathlib import Path
from scipy import stats
from dataclasses import dataclass
from typing import Dict, List, Tuple
import warnings
# Selective warning suppression - allow important warnings through
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*overflow.*')
warnings.filterwarnings('ignore', message='.*divide by zero.*')


def cleanup_results_directory(output_dir: Path):
    """Delete all files in output directory before new run."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
        print(f"Cleaned up: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def cleanup_iter_files(project_root: Path = None):
    """Delete all .iter files from project root for clean estimation."""
    if project_root is None:
        project_root = Path.cwd()
    iter_files = list(project_root.glob("__*.iter"))
    if iter_files:
        for iter_file in iter_files:
            iter_file.unlink()
        print(f"Deleted {len(iter_files)} .iter files from project root")

# Biogeme
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta, Variable, MonteCarlo, log, exp

# Use Draws instead of deprecated bioDraws
try:
    from biogeme.expressions import Draws
except ImportError:
    from biogeme.expressions import bioDraws as Draws


# =============================================================================
# DATA
# =============================================================================

def load_data(data_path: str, use_panel: bool = True) -> Tuple[pd.DataFrame, db.Database]:
    """Load and prepare data for MXL estimation.

    Args:
        data_path: Path to CSV data file
        use_panel: If True (default), specify panel structure using 'ID' column.
                   This ensures same random coefficient draws across all choice
                   tasks for each individual, which is REQUIRED for proper MXL
                   estimation with panel data.

                   Set to False only for cross-sectional data (one obs per person).

    Returns:
        Tuple of (DataFrame, Biogeme Database)

    Note:
        For panel data (T>1 observations per individual), panel structure is
        ESSENTIAL for correct inference. Without it:
        - Random coefficients would be redrawn for each observation
        - Standard errors would be incorrect
        - The model would not capture true taste heterogeneity

        References:
        - Train (2009): Discrete Choice Methods with Simulation, Ch. 6
        - Biogeme documentation: Panel data examples
    """
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} observations")

    df = df.copy()
    df['fee1_10k'] = df['fee1'] / 10000.0
    df['fee2_10k'] = df['fee2'] / 10000.0
    df['fee3_10k'] = df['fee3'] / 10000.0

    # Drop strings
    string_cols = df.select_dtypes(include=['object']).columns.tolist()
    df_num = df.drop(columns=string_cols)

    database = db.Database('mxl_data', df_num)

    # PANEL STRUCTURE: Essential for panel data with multiple obs per individual
    # Ensures same random coefficient draws across all choice tasks for each person
    if use_panel:
        if 'ID' not in df_num.columns:
            raise ValueError(
                "MXL with use_panel=True requires 'ID' column for panel structure. "
                "Either add 'ID' column or set use_panel=False for cross-sectional data."
            )
        database.panel('ID')
        n_individuals = df['ID'].nunique()
        n_obs_per_person = len(df) / n_individuals
        print(f"Panel structure ENABLED: {n_individuals} individuals, {n_obs_per_person:.1f} obs/person avg")
    else:
        if 'ID' not in df.columns:
            raise ValueError(
                "MXL requires 'ID' column to identify individuals. "
                "Each row should have an individual identifier."
            )
        n_individuals = df['ID'].nunique()
        n_obs_per_person = len(df) / n_individuals if n_individuals > 0 else 1

        # Warn if panel data detected but panel structure not used
        if n_obs_per_person > 1.5:
            import warnings
            warnings.warn(
                f"Panel data detected ({n_obs_per_person:.1f} obs/person) but panel structure "
                f"is DISABLED. This will produce incorrect standard errors for MXL. "
                f"Set use_panel=True unless you have cross-sectional data.",
                UserWarning
            )
        print(f"Data: {n_individuals} individuals (panel structure DISABLED)")

    return df, database


# =============================================================================
# MXL MODELS
# =============================================================================

def get_vars():
    """Get common variables for MXL models."""
    return {
        'dur1': Variable('dur1'),
        'dur2': Variable('dur2'),
        'dur3': Variable('dur3'),
        'fee1_10k': Variable('fee1_10k'),
        'fee2_10k': Variable('fee2_10k'),
        'fee3_10k': Variable('fee3_10k'),
        'CHOICE': Variable('CHOICE'),
    }


def mxl_1_random_fee(database: db.Database):
    """MXL with random fee coefficient: B_FEE ~ N(mu, sigma^2)"""
    v = get_vars()

    # Fixed parameters
    ASC_paid = Beta('ASC_paid', 0, None, None, 0)
    B_DUR = Beta('B_DUR', -0.05, -5, 0, 0)  # Bounded negative

    # Random fee: mean and std
    # Lower bound on sigma: 0.0 would allow testing for no heterogeneity,
    # but 0.001 prevents numerical issues. Use LR test (MNL vs MXL) instead.
    B_FEE_MU = Beta('B_FEE_MU', -0.5, -5, 0, 0)  # Bounded negative
    B_FEE_SIGMA = Beta('B_FEE_SIGMA', 0.1, 0.0, 3, 0)  # Allow 0 for testing

    # Random coefficient
    B_FEE_RND = B_FEE_MU + B_FEE_SIGMA * Draws('B_FEE_RND', 'NORMAL')

    # Utilities
    V1 = ASC_paid + B_FEE_RND * v['fee1_10k'] + B_DUR * v['dur1']
    V2 = ASC_paid + B_FEE_RND * v['fee2_10k'] + B_DUR * v['dur2']
    V3 = B_FEE_RND * v['fee3_10k'] + B_DUR * v['dur3']

    V = {1: V1, 2: V2, 3: V3}
    av = {1: 1, 2: 1, 3: 1}

    # Conditional probability
    prob = models.logit(V, av, v['CHOICE'])

    # Simulated log-likelihood (integrate over draws)
    logprob = log(MonteCarlo(prob))

    return logprob, 'MXL-1: Random Fee'


def mxl_2_random_fee_dur(database: db.Database):
    """MXL with random fee and duration coefficients."""
    v = get_vars()

    ASC_paid = Beta('ASC_paid', 0, None, None, 0)

    # Random fee
    B_FEE_MU = Beta('B_FEE_MU', -0.5, -5, 0, 0)  # Bounded negative
    B_FEE_SIGMA = Beta('B_FEE_SIGMA', 0.1, 0.0, 3, 0)  # Allow 0 for testing

    # Random duration
    B_DUR_MU = Beta('B_DUR_MU', -0.05, -2, 0, 0)  # Bounded negative
    B_DUR_SIGMA = Beta('B_DUR_SIGMA', 0.01, 0.0, 1, 0)  # Allow 0 for testing

    B_FEE_RND = B_FEE_MU + B_FEE_SIGMA * Draws('B_FEE_RND', 'NORMAL')
    B_DUR_RND = B_DUR_MU + B_DUR_SIGMA * Draws('B_DUR_RND', 'NORMAL')

    # Utilities
    V1 = ASC_paid + B_FEE_RND * v['fee1_10k'] + B_DUR_RND * v['dur1']
    V2 = ASC_paid + B_FEE_RND * v['fee2_10k'] + B_DUR_RND * v['dur2']
    V3 = B_FEE_RND * v['fee3_10k'] + B_DUR_RND * v['dur3']

    V = {1: V1, 2: V2, 3: V3}
    av = {1: 1, 2: 1, 3: 1}

    prob = models.logit(V, av, v['CHOICE'])
    logprob = log(MonteCarlo(prob))

    return logprob, 'MXL-2: Random Fee+Dur'


def mxl_3_random_all(database: db.Database):
    """MXL with random fee and duration coefficients (both attributes)."""
    v = get_vars()

    ASC_paid = Beta('ASC_paid', 0, None, None, 0)

    # Random fee
    B_FEE_MU = Beta('B_FEE_MU', -0.5, -5, 0, 0)  # Bounded negative
    B_FEE_SIGMA = Beta('B_FEE_SIGMA', 0.1, 0.0, 3, 0)  # Allow 0 for testing

    # Random duration
    B_DUR_MU = Beta('B_DUR_MU', -0.05, -2, 0, 0)  # Bounded negative
    B_DUR_SIGMA = Beta('B_DUR_SIGMA', 0.01, 0.0, 1, 0)  # Allow 0 for testing

    B_FEE_RND = B_FEE_MU + B_FEE_SIGMA * Draws('B_FEE_RND', 'NORMAL')
    B_DUR_RND = B_DUR_MU + B_DUR_SIGMA * Draws('B_DUR_RND', 'NORMAL')

    # Utilities
    V1 = ASC_paid + B_FEE_RND * v['fee1_10k'] + B_DUR_RND * v['dur1']
    V2 = ASC_paid + B_FEE_RND * v['fee2_10k'] + B_DUR_RND * v['dur2']
    V3 = B_FEE_RND * v['fee3_10k'] + B_DUR_RND * v['dur3']

    V = {1: V1, 2: V2, 3: V3}
    av = {1: 1, 2: 1, 3: 1}

    prob = models.logit(V, av, v['CHOICE'])
    logprob = log(MonteCarlo(prob))

    return logprob, 'MXL-3: Random Fee+Dur (Full)'


# =============================================================================
# ESTIMATION
# =============================================================================

@dataclass
class MXLResults:
    name: str
    log_likelihood: float
    n_parameters: int
    aic: float
    bic: float
    parameters: Dict[str, float]
    std_errors: Dict[str, float]
    converged: bool


def estimate_mxl(database: db.Database, model_func, n_draws: int = 5000, output_dir: Path = None) -> MXLResults:
    """Estimate MXL model."""
    logprob, name = model_func(database)

    print(f"\n{name} (draws={n_draws})")
    print("-" * 40)

    biogeme_model = bio.BIOGEME(database, logprob, number_of_draws=n_draws)

    # Set model name with output directory path for HTML files
    safe_name = name.replace(':', '').replace(' ', '_').replace('-', '_')
    if output_dir:
        biogeme_model.model_name = str(output_dir / safe_name)
    else:
        biogeme_model.model_name = safe_name

    results = biogeme_model.estimate()

    ll = results.final_loglikelihood
    n_params = results.number_of_free_parameters
    n_obs = len(database.dataframe)

    aic = 2 * n_params - 2 * ll
    bic = n_params * np.log(n_obs) - 2 * ll

    betas = results.get_beta_values()
    std_errs = {}
    for p in betas:
        try:
            std_errs[p] = results.get_parameter_std_err(p)
        except:
            std_errs[p] = np.nan

    converged = results.algorithm_has_converged

    print(f"  LL: {ll:.2f} | K: {n_params} | AIC: {aic:.2f} | Conv: {converged}")

    # Print parameters
    for p, val in betas.items():
        se = std_errs.get(p, np.nan)
        t = val / se if se > 0 else np.nan
        print(f"  {p}: {val:.4f} (t={t:.2f})")

    return MXLResults(
        name=name,
        log_likelihood=ll,
        n_parameters=n_params,
        aic=aic,
        bic=bic,
        parameters=betas,
        std_errors=std_errs,
        converged=converged
    )


def run_mxl_comparison(data_path: str, output_dir: str = None, n_draws: int = 5000):
    """Run MXL model comparison.

    Args:
        data_path: Path to data CSV
        output_dir: Output directory for results
        n_draws: Number of simulation draws (default: 5000 for stable estimates)
    """
    print("=" * 60)
    print("MIXED LOGIT (MXL) MODEL COMPARISON")
    print("=" * 60)

    # Setup and cleanup
    cleanup_iter_files()  # Delete .iter files from project root

    if output_dir:
        output_dir = Path(output_dir)
        cleanup_results_directory(output_dir)  # Clean and recreate output directory
    else:
        output_dir = None

    df, database = load_data(data_path)

    # Models to estimate
    models_list = [
        mxl_1_random_fee,
        mxl_2_random_fee_dur,
        mxl_3_random_all,
    ]

    results = []
    for model_func in models_list:
        try:
            res = estimate_mxl(database, model_func, n_draws, output_dir)
            results.append(res)
        except Exception as e:
            print(f"  ERROR: {e}")

    # Comparison
    print("\n" + "=" * 60)
    print("MXL COMPARISON SUMMARY")
    print("=" * 60)

    data = []
    for r in results:
        data.append({
            'Model': r.name,
            'LL': r.log_likelihood,
            'K': r.n_parameters,
            'AIC': r.aic,
            'BIC': r.bic,
            'Conv': 'Yes' if r.converged else 'No'
        })

    comp_df = pd.DataFrame(data)
    print("\n" + comp_df.to_string(index=False))

    # Best model
    best = comp_df.loc[comp_df['AIC'].idxmin()]
    print(f"\nBest by AIC: {best['Model']} (AIC={best['AIC']:.2f})")

    if output_dir:
        comp_df.to_csv(output_dir / 'mxl_comparison.csv', index=False)

        # Parameters
        param_data = []
        for r in results:
            for p, v in r.parameters.items():
                se = r.std_errors.get(p, np.nan)
                param_data.append({
                    'Model': r.name,
                    'Parameter': p,
                    'Estimate': v,
                    'Std.Err': se,
                    't-stat': v / se if se > 0 else np.nan
                })
        pd.DataFrame(param_data).to_csv(output_dir / 'mxl_parameters.csv', index=False)

        print(f"\nOutputs saved to: {output_dir}")

    return results, comp_df


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', default='mxl_results')
    parser.add_argument('--draws', type=int, default=5000,
                        help='Number of draws for simulation (default: 5000)')
    args = parser.parse_args()

    run_mxl_comparison(args.data, args.output, args.draws)


if __name__ == '__main__':
    main()
