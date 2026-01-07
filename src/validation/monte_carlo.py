"""
Monte Carlo Validation
======================

Framework for Monte Carlo simulation studies to validate
estimation methods and compare two-stage vs ICLV approaches.

Key metrics:
- Bias: E[θ̂] - θ
- RMSE: sqrt(E[(θ̂ - θ)²])
- Coverage: P(θ ∈ CI)
- Relative efficiency

Author: DCM Research Team
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import time
import warnings


@dataclass
class MonteCarloResult:
    """Container for Monte Carlo study results."""
    # Study parameters
    n_replications: int
    sample_sizes: List[int]
    true_values: Dict[str, float]

    # Results by sample size
    estimates: Dict[int, np.ndarray]  # {sample_size: (n_rep, n_params)}
    std_errors: Dict[int, np.ndarray]
    convergence: Dict[int, np.ndarray]  # Boolean convergence flags

    # Summary statistics
    bias: Dict[int, Dict[str, float]] = field(default_factory=dict)
    rmse: Dict[int, Dict[str, float]] = field(default_factory=dict)
    coverage: Dict[int, Dict[str, float]] = field(default_factory=dict)
    mean_se: Dict[int, Dict[str, float]] = field(default_factory=dict)
    empirical_se: Dict[int, Dict[str, float]] = field(default_factory=dict)

    # Timing
    total_time: float = 0.0
    time_per_rep: float = 0.0

    def summary_table(self) -> pd.DataFrame:
        """Generate summary table of results."""
        records = []

        for sample_size in self.sample_sizes:
            for param, true_val in self.true_values.items():
                records.append({
                    'sample_size': sample_size,
                    'parameter': param,
                    'true_value': true_val,
                    'bias': self.bias.get(sample_size, {}).get(param, np.nan),
                    'bias_pct': self.bias.get(sample_size, {}).get(param, np.nan) / true_val * 100 if true_val != 0 else np.nan,
                    'rmse': self.rmse.get(sample_size, {}).get(param, np.nan),
                    'coverage_95': self.coverage.get(sample_size, {}).get(param, np.nan),
                    'mean_se': self.mean_se.get(sample_size, {}).get(param, np.nan),
                    'empirical_se': self.empirical_se.get(sample_size, {}).get(param, np.nan),
                })

        return pd.DataFrame(records)


def compute_bias(estimates: np.ndarray, true_value: float) -> float:
    """
    Compute bias of estimator.

    Bias = E[θ̂] - θ

    Args:
        estimates: Array of estimates across replications
        true_value: True parameter value

    Returns:
        Bias estimate
    """
    valid_estimates = estimates[~np.isnan(estimates)]
    if len(valid_estimates) == 0:
        return np.nan
    return np.mean(valid_estimates) - true_value


def compute_rmse(estimates: np.ndarray, true_value: float) -> float:
    """
    Compute Root Mean Squared Error.

    RMSE = sqrt(E[(θ̂ - θ)²])

    Args:
        estimates: Array of estimates across replications
        true_value: True parameter value

    Returns:
        RMSE estimate
    """
    valid_estimates = estimates[~np.isnan(estimates)]
    if len(valid_estimates) == 0:
        return np.nan
    return np.sqrt(np.mean((valid_estimates - true_value) ** 2))


def compute_coverage(estimates: np.ndarray,
                     std_errors: np.ndarray,
                     true_value: float,
                     confidence: float = 0.95) -> float:
    """
    Compute confidence interval coverage rate.

    Coverage = P(θ ∈ [θ̂ - z*SE, θ̂ + z*SE])

    Args:
        estimates: Array of estimates
        std_errors: Array of standard errors
        true_value: True parameter value
        confidence: Confidence level (default 95%)

    Returns:
        Coverage rate (0 to 1)
    """
    valid_mask = ~np.isnan(estimates) & ~np.isnan(std_errors) & (std_errors > 0)

    if valid_mask.sum() == 0:
        return np.nan

    z = stats.norm.ppf((1 + confidence) / 2)

    lower = estimates[valid_mask] - z * std_errors[valid_mask]
    upper = estimates[valid_mask] + z * std_errors[valid_mask]

    covered = (lower <= true_value) & (true_value <= upper)
    return covered.mean()


class MonteCarloStudy:
    """
    Monte Carlo simulation study for estimation validation.

    Runs multiple replications of:
    1. Generate data from known DGP
    2. Estimate model
    3. Record estimates and SEs
    4. Compute summary statistics

    Example:
        >>> study = MonteCarloStudy(
        ...     n_replications=100,
        ...     sample_sizes=[500, 1000, 2000]
        ... )
        >>> result = study.run(dgp_func, estimate_func, true_values)
    """

    def __init__(self,
                 n_replications: int = 100,
                 sample_sizes: List[int] = None,
                 seed: int = 42,
                 n_workers: int = 1,
                 verbose: bool = True):
        """
        Initialize Monte Carlo study.

        Args:
            n_replications: Number of replications per sample size
            sample_sizes: List of sample sizes to test
            seed: Base random seed
            n_workers: Number of parallel workers (1 = sequential)
            verbose: Print progress
        """
        self.n_replications = n_replications
        self.sample_sizes = sample_sizes or [500, 1000, 2000, 5000]
        self.seed = seed
        self.n_workers = n_workers
        self.verbose = verbose

    def _single_replication(self,
                            rep: int,
                            sample_size: int,
                            dgp_func: Callable,
                            estimate_func: Callable,
                            param_names: List[str],
                            base_seed: int) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Run single replication."""
        # Set unique seed for this replication
        rep_seed = base_seed + rep + sample_size * 1000

        try:
            # Generate data
            data = dgp_func(n=sample_size, seed=rep_seed)

            # Estimate model
            result = estimate_func(data)

            # Extract estimates and SEs
            estimates = np.array([result.get(p, np.nan) for p in param_names])
            std_errors = np.array([result.get(f'{p}_se', np.nan) for p in param_names])
            converged = result.get('converged', True)

            return estimates, std_errors, converged

        except Exception as e:
            if self.verbose:
                warnings.warn(f"Replication {rep} failed: {e}")
            n_params = len(param_names)
            return np.full(n_params, np.nan), np.full(n_params, np.nan), False

    def run(self,
            dgp_func: Callable,
            estimate_func: Callable,
            true_values: Dict[str, float]) -> MonteCarloResult:
        """
        Run Monte Carlo study.

        Args:
            dgp_func: Function(n, seed) -> data that generates data from DGP
            estimate_func: Function(data) -> dict with estimates and SEs
            true_values: Dict mapping parameter name to true value

        Returns:
            MonteCarloResult with summary statistics
        """
        start_time = time.time()

        param_names = list(true_values.keys())
        n_params = len(param_names)

        # Storage for results
        estimates = {}
        std_errors = {}
        convergence = {}

        for sample_size in self.sample_sizes:
            if self.verbose:
                print(f"\nSample size: {sample_size}")
                print("-" * 40)

            estimates[sample_size] = np.zeros((self.n_replications, n_params))
            std_errors[sample_size] = np.zeros((self.n_replications, n_params))
            convergence[sample_size] = np.zeros(self.n_replications, dtype=bool)

            if self.n_workers > 1:
                # Parallel execution
                with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                    futures = [
                        executor.submit(
                            self._single_replication,
                            rep, sample_size, dgp_func, estimate_func,
                            param_names, self.seed
                        )
                        for rep in range(self.n_replications)
                    ]

                    for rep, future in enumerate(futures):
                        est, se, conv = future.result()
                        estimates[sample_size][rep] = est
                        std_errors[sample_size][rep] = se
                        convergence[sample_size][rep] = conv

                        if self.verbose and (rep + 1) % 10 == 0:
                            print(f"  Completed {rep + 1}/{self.n_replications}")
            else:
                # Sequential execution
                for rep in range(self.n_replications):
                    est, se, conv = self._single_replication(
                        rep, sample_size, dgp_func, estimate_func,
                        param_names, self.seed
                    )
                    estimates[sample_size][rep] = est
                    std_errors[sample_size][rep] = se
                    convergence[sample_size][rep] = conv

                    if self.verbose and (rep + 1) % 10 == 0:
                        conv_rate = convergence[sample_size][:rep+1].mean()
                        print(f"  Rep {rep + 1}/{self.n_replications}, Conv: {conv_rate:.1%}")

        # Compute summary statistics
        result = MonteCarloResult(
            n_replications=self.n_replications,
            sample_sizes=self.sample_sizes,
            true_values=true_values,
            estimates=estimates,
            std_errors=std_errors,
            convergence=convergence
        )

        for sample_size in self.sample_sizes:
            result.bias[sample_size] = {}
            result.rmse[sample_size] = {}
            result.coverage[sample_size] = {}
            result.mean_se[sample_size] = {}
            result.empirical_se[sample_size] = {}

            for i, param in enumerate(param_names):
                est = estimates[sample_size][:, i]
                se = std_errors[sample_size][:, i]
                true_val = true_values[param]

                result.bias[sample_size][param] = compute_bias(est, true_val)
                result.rmse[sample_size][param] = compute_rmse(est, true_val)
                result.coverage[sample_size][param] = compute_coverage(est, se, true_val)
                result.mean_se[sample_size][param] = np.nanmean(se)
                result.empirical_se[sample_size][param] = np.nanstd(est)

        total_time = time.time() - start_time
        result.total_time = total_time
        result.time_per_rep = total_time / (self.n_replications * len(self.sample_sizes))

        if self.verbose:
            print(f"\n{'='*50}")
            print(f"Monte Carlo study complete")
            print(f"Total time: {total_time:.1f}s")
            print(f"Time per replication: {result.time_per_rep:.2f}s")

        return result


def run_monte_carlo_comparison(dgp_func: Callable,
                                two_stage_func: Callable,
                                iclv_func: Callable,
                                true_values: Dict[str, float],
                                n_replications: int = 100,
                                sample_sizes: List[int] = None,
                                seed: int = 42,
                                verbose: bool = True) -> pd.DataFrame:
    """
    Run Monte Carlo comparison of two-stage vs ICLV estimation.

    This is the key validation for demonstrating attenuation bias
    correction in simultaneous estimation.

    Args:
        dgp_func: Data generating process
        two_stage_func: Two-stage estimation function
        iclv_func: ICLV estimation function
        true_values: True parameter values
        n_replications: Number of replications
        sample_sizes: Sample sizes to test
        seed: Random seed
        verbose: Print progress

    Returns:
        DataFrame comparing methods across metrics

    Example output:
        | parameter | method    | N=500 bias | N=1000 bias | N=2000 bias |
        |-----------|-----------|------------|-------------|-------------|
        | beta_env  | two_stage |    -0.25   |    -0.24    |    -0.23    |
        | beta_env  | iclv      |    -0.02   |    -0.01    |     0.00    |
    """
    sample_sizes = sample_sizes or [500, 1000, 2000]

    if verbose:
        print("=" * 60)
        print("Monte Carlo Comparison: Two-Stage vs ICLV")
        print("=" * 60)

    # Run two-stage study
    if verbose:
        print("\n[1/2] Running Two-Stage estimation...")

    study_2s = MonteCarloStudy(
        n_replications=n_replications,
        sample_sizes=sample_sizes,
        seed=seed,
        verbose=verbose
    )
    result_2s = study_2s.run(dgp_func, two_stage_func, true_values)

    # Run ICLV study
    if verbose:
        print("\n[2/2] Running ICLV estimation...")

    study_iclv = MonteCarloStudy(
        n_replications=n_replications,
        sample_sizes=sample_sizes,
        seed=seed,
        verbose=verbose
    )
    result_iclv = study_iclv.run(dgp_func, iclv_func, true_values)

    # Combine results
    records = []

    for param, true_val in true_values.items():
        # Two-stage results
        row_2s = {
            'parameter': param,
            'true_value': true_val,
            'method': 'two_stage'
        }
        for n in sample_sizes:
            row_2s[f'bias_N{n}'] = result_2s.bias[n][param]
            row_2s[f'rmse_N{n}'] = result_2s.rmse[n][param]
            row_2s[f'coverage_N{n}'] = result_2s.coverage[n][param]
        records.append(row_2s)

        # ICLV results
        row_iclv = {
            'parameter': param,
            'true_value': true_val,
            'method': 'iclv'
        }
        for n in sample_sizes:
            row_iclv[f'bias_N{n}'] = result_iclv.bias[n][param]
            row_iclv[f'rmse_N{n}'] = result_iclv.rmse[n][param]
            row_iclv[f'coverage_N{n}'] = result_iclv.coverage[n][param]
        records.append(row_iclv)

    df = pd.DataFrame(records)

    if verbose:
        print("\n" + "=" * 60)
        print("SUMMARY: Attenuation Bias Comparison")
        print("=" * 60)

        # Focus on LV effects
        lv_params = [p for p in true_values.keys() if 'beta' in p.lower()]
        if lv_params:
            print("\nLV Effect Parameters (where attenuation matters most):")
            print("-" * 50)
            for param in lv_params:
                true_val = true_values[param]
                print(f"\n{param} (true = {true_val:.3f}):")
                for n in sample_sizes:
                    bias_2s = result_2s.bias[n][param]
                    bias_iclv = result_iclv.bias[n][param]
                    pct_2s = (bias_2s / true_val * 100) if true_val != 0 else np.nan
                    pct_iclv = (bias_iclv / true_val * 100) if true_val != 0 else np.nan
                    print(f"  N={n}: Two-stage bias={bias_2s:+.3f} ({pct_2s:+.1f}%), "
                          f"ICLV bias={bias_iclv:+.3f} ({pct_iclv:+.1f}%)")

    return df


def generate_latex_mc_table(result: MonteCarloResult,
                            caption: str = "Monte Carlo Simulation Results",
                            label: str = "tab:monte_carlo") -> str:
    """
    Generate LaTeX table from Monte Carlo results.

    Args:
        result: MonteCarloResult object
        caption: Table caption
        label: Table label

    Returns:
        LaTeX table string
    """
    df = result.summary_table()

    # Pivot for better formatting
    params = df['parameter'].unique()
    sample_sizes = df['sample_size'].unique()

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{l" + "rrr" * len(sample_sizes) + "}",
        "\\toprule",
    ]

    # Header
    header = "Parameter"
    for n in sample_sizes:
        header += f" & \\multicolumn{{3}}{{c}}{{N = {n}}}"
    lines.append(header + " \\\\")

    subheader = ""
    for _ in sample_sizes:
        subheader += " & Bias & RMSE & Coverage"
    lines.append(subheader + " \\\\")
    lines.append("\\midrule")

    # Data rows
    for param in params:
        row = f"{param}"
        for n in sample_sizes:
            mask = (df['parameter'] == param) & (df['sample_size'] == n)
            bias = df.loc[mask, 'bias'].values[0]
            rmse = df.loc[mask, 'rmse'].values[0]
            cov = df.loc[mask, 'coverage_95'].values[0]
            row += f" & {bias:.3f} & {rmse:.3f} & {cov:.2f}"
        lines.append(row + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        f"\\\\[0.5em]",
        f"\\footnotesize Note: Based on {result.n_replications} replications. ",
        f"Coverage is for 95\\% confidence intervals.",
        "\\end{table}"
    ])

    return "\n".join(lines)


if __name__ == '__main__':
    print("Monte Carlo Validation Module")
    print("=" * 50)

    # Simple demo with fake DGP and estimator
    def demo_dgp(n, seed):
        np.random.seed(seed)
        x = np.random.randn(n)
        y = 0.5 * x + np.random.randn(n) * 0.5
        return {'x': x, 'y': y}

    def demo_estimator(data):
        x, y = data['x'], data['y']
        beta = np.sum(x * y) / np.sum(x ** 2)
        residuals = y - beta * x
        se = np.sqrt(np.sum(residuals ** 2) / (len(x) - 1) / np.sum(x ** 2))
        return {'beta': beta, 'beta_se': se, 'converged': True}

    print("\nRunning demo Monte Carlo study...")

    study = MonteCarloStudy(
        n_replications=50,
        sample_sizes=[100, 500],
        verbose=True
    )

    result = study.run(
        dgp_func=demo_dgp,
        estimate_func=demo_estimator,
        true_values={'beta': 0.5}
    )

    print("\nResults:")
    print(result.summary_table().to_string())
