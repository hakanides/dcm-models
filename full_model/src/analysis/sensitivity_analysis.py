"""
Sensitivity Analysis for DCM Models
====================================

Assess estimation robustness by varying true parameters and evaluating
recovery rates across multiple simulation replications.

Usage:
    from src.analysis.sensitivity_analysis import SensitivityAnalyzer

    analyzer = SensitivityAnalyzer(base_config='config/model_config.json')
    results = analyzer.run_sensitivity(
        param_name='B_FEE',
        variations=[-0.5, -0.3, -0.1],
        n_replications=10
    )
    analyzer.plot_results(results)

Authors: Hakan Mülayim, Giray Girengir, Ataol Azeritürk
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable, Any
from pathlib import Path
import json
import warnings
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import time


@dataclass
class SensitivityResult:
    """Results from a single sensitivity run."""
    true_value: float
    estimated_values: List[float]
    std_errors: List[float]
    converged: List[bool]
    bias: float
    rmse: float
    coverage_95: float
    mean_se: float


class SensitivityAnalyzer:
    """
    Sensitivity analysis for DCM parameter estimation.

    Varies true parameter values and assesses estimation quality metrics:
    - Bias: Mean(estimated - true)
    - RMSE: Root Mean Squared Error
    - Coverage: Proportion of 95% CIs containing true value
    - Convergence rate: Proportion of successful estimations
    """

    def __init__(self, base_config: str = 'config/model_config.json'):
        """
        Initialize analyzer with base configuration.

        Args:
            base_config: Path to base model configuration JSON
        """
        self.base_config_path = Path(base_config)
        if self.base_config_path.exists():
            with open(self.base_config_path) as f:
                self.base_config = json.load(f)
        else:
            self.base_config = {}

    def _modify_config(self, param_path: str, new_value: float) -> Dict:
        """
        Create modified config with new parameter value.

        Args:
            param_path: Dot-separated path to parameter (e.g., 'choice.attributes.fee.base_coef')
            new_value: New value for the parameter

        Returns:
            Modified config dictionary
        """
        import copy
        config = copy.deepcopy(self.base_config)

        # Navigate to parameter
        parts = param_path.split('.')
        current = config
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[parts[-1]] = new_value
        return config

    def compute_metrics(self, true_value: float,
                        estimates: List[float],
                        std_errors: List[float],
                        converged: List[bool]) -> SensitivityResult:
        """
        Compute sensitivity metrics from estimation results.

        Args:
            true_value: True parameter value used in simulation
            estimates: List of estimated values across replications
            std_errors: List of standard errors
            converged: List of convergence indicators

        Returns:
            SensitivityResult with computed metrics
        """
        # Filter to converged estimates
        valid_idx = [i for i, c in enumerate(converged) if c]
        valid_estimates = [estimates[i] for i in valid_idx]
        valid_ses = [std_errors[i] for i in valid_idx]

        if not valid_estimates:
            return SensitivityResult(
                true_value=true_value,
                estimated_values=estimates,
                std_errors=std_errors,
                converged=converged,
                bias=np.nan,
                rmse=np.nan,
                coverage_95=np.nan,
                mean_se=np.nan
            )

        estimates_arr = np.array(valid_estimates)
        ses_arr = np.array(valid_ses)

        # Bias
        bias = np.mean(estimates_arr - true_value)

        # RMSE
        rmse = np.sqrt(np.mean((estimates_arr - true_value) ** 2))

        # Coverage (95% CI)
        lower = estimates_arr - 1.96 * ses_arr
        upper = estimates_arr + 1.96 * ses_arr
        covered = (lower <= true_value) & (true_value <= upper)
        coverage_95 = np.mean(covered)

        # Mean SE
        mean_se = np.mean(ses_arr)

        return SensitivityResult(
            true_value=true_value,
            estimated_values=estimates,
            std_errors=std_errors,
            converged=converged,
            bias=bias,
            rmse=rmse,
            coverage_95=coverage_95,
            mean_se=mean_se
        )

    def run_single_replication(self,
                               config: Dict,
                               model_func: Callable,
                               param_name: str,
                               n_obs: int = 1000,
                               seed: int = None) -> Tuple[float, float, bool]:
        """
        Run a single simulation and estimation replication.

        Args:
            config: Configuration dictionary
            model_func: Function to create and estimate model
            param_name: Name of parameter to extract
            n_obs: Number of observations to simulate
            seed: Random seed

        Returns:
            Tuple of (estimate, std_error, converged)
        """
        try:
            # This is a placeholder - actual implementation depends on
            # the simulation and estimation pipeline
            import biogeme.database as db
            import biogeme.biogeme as bio

            # Simulate data with config
            # df = simulate_data(config, n_obs, seed)

            # Create database and estimate
            # database = db.Database('sensitivity', df)
            # logprob, name = model_func(database)
            # biogeme_obj = bio.BIOGEME(database, logprob)
            # results = biogeme_obj.estimate()

            # Extract parameter estimate
            # betas = results.get_beta_values()
            # estimate = betas.get(param_name, np.nan)
            # se = results.get_parameter_std_err(param_name)
            # converged = results.algorithm_has_converged

            # Placeholder return
            return np.nan, np.nan, False

        except Exception as e:
            warnings.warn(f"Replication failed: {e}")
            return np.nan, np.nan, False

    def run_sensitivity(self,
                        param_name: str,
                        param_path: str,
                        variations: List[float],
                        model_func: Callable,
                        n_replications: int = 50,
                        n_obs: int = 1000,
                        n_jobs: int = 1,
                        verbose: bool = True) -> pd.DataFrame:
        """
        Run full sensitivity analysis.

        Args:
            param_name: Parameter name in estimation results
            param_path: Path to parameter in config file
            variations: List of true parameter values to test
            model_func: Model specification function
            n_replications: Number of replications per variation
            n_obs: Observations per replication
            n_jobs: Number of parallel jobs
            verbose: Print progress

        Returns:
            DataFrame with sensitivity results
        """
        results = []

        for true_value in variations:
            if verbose:
                print(f"\n{'='*50}")
                print(f"Testing {param_name} = {true_value}")
                print(f"{'='*50}")

            config = self._modify_config(param_path, true_value)
            estimates = []
            std_errors = []
            converged = []

            for rep in range(n_replications):
                if verbose and rep % 10 == 0:
                    print(f"  Replication {rep + 1}/{n_replications}")

                seed = 42 + rep
                est, se, conv = self.run_single_replication(
                    config, model_func, param_name, n_obs, seed
                )
                estimates.append(est)
                std_errors.append(se)
                converged.append(conv)

            # Compute metrics
            metrics = self.compute_metrics(true_value, estimates, std_errors, converged)

            results.append({
                'parameter': param_name,
                'true_value': true_value,
                'mean_estimate': np.nanmean(estimates),
                'std_estimate': np.nanstd(estimates),
                'bias': metrics.bias,
                'rmse': metrics.rmse,
                'coverage_95': metrics.coverage_95,
                'mean_se': metrics.mean_se,
                'convergence_rate': np.mean(converged),
                'n_replications': n_replications
            })

            if verbose:
                print(f"  Bias: {metrics.bias:.4f}")
                print(f"  RMSE: {metrics.rmse:.4f}")
                print(f"  Coverage: {metrics.coverage_95:.1%}")
                print(f"  Convergence: {np.mean(converged):.1%}")

        return pd.DataFrame(results)

    def run_variation_analysis(self,
                               param_name: str,
                               param_path: str,
                               base_value: float,
                               variation_pcts: List[float] = [-20, -10, 0, 10, 20],
                               **kwargs) -> pd.DataFrame:
        """
        Run sensitivity analysis with percentage variations from base value.

        Args:
            param_name: Parameter name
            param_path: Config path
            base_value: Base parameter value
            variation_pcts: Percentage variations to test
            **kwargs: Additional arguments for run_sensitivity

        Returns:
            DataFrame with results
        """
        variations = [base_value * (1 + pct/100) for pct in variation_pcts]
        return self.run_sensitivity(param_name, param_path, variations, **kwargs)


def compute_coverage_statistics(estimates: np.ndarray,
                                std_errors: np.ndarray,
                                true_value: float,
                                confidence_levels: List[float] = [0.90, 0.95, 0.99]) -> Dict[float, float]:
    """
    Compute coverage statistics at multiple confidence levels.

    Args:
        estimates: Array of point estimates
        std_errors: Array of standard errors
        true_value: True parameter value
        confidence_levels: Confidence levels to compute

    Returns:
        Dict mapping confidence level to coverage rate
    """
    from scipy import stats

    coverage = {}
    for level in confidence_levels:
        z = stats.norm.ppf(1 - (1 - level) / 2)
        lower = estimates - z * std_errors
        upper = estimates + z * std_errors
        covered = (lower <= true_value) & (true_value <= upper)
        coverage[level] = np.mean(covered)

    return coverage


def monte_carlo_simulation(true_params: Dict[str, float],
                           simulate_func: Callable,
                           estimate_func: Callable,
                           n_replications: int = 100,
                           n_obs: int = 1000,
                           seed_base: int = 42,
                           verbose: bool = True) -> pd.DataFrame:
    """
    Run Monte Carlo simulation for parameter recovery assessment.

    Args:
        true_params: Dictionary of true parameter values
        simulate_func: Function(params, n_obs, seed) -> DataFrame
        estimate_func: Function(df) -> Dict with 'betas' and 'std_errs'
        n_replications: Number of replications
        n_obs: Observations per replication
        seed_base: Base random seed
        verbose: Print progress

    Returns:
        DataFrame with results for each replication
    """
    results = []

    for rep in range(n_replications):
        if verbose and rep % 10 == 0:
            print(f"Replication {rep + 1}/{n_replications}")

        seed = seed_base + rep

        try:
            # Simulate data
            df = simulate_func(true_params, n_obs, seed)

            # Estimate model
            est_result = estimate_func(df)

            # Record results
            row = {
                'replication': rep + 1,
                'seed': seed,
                'converged': est_result.get('converged', True)
            }

            for param, true_val in true_params.items():
                est_val = est_result.get('betas', {}).get(param, np.nan)
                se_val = est_result.get('std_errs', {}).get(param, np.nan)

                row[f'{param}_true'] = true_val
                row[f'{param}_est'] = est_val
                row[f'{param}_se'] = se_val
                row[f'{param}_bias'] = est_val - true_val
                row[f'{param}_t'] = (est_val - true_val) / se_val if se_val else np.nan

            results.append(row)

        except Exception as e:
            if verbose:
                print(f"  Replication {rep + 1} failed: {e}")
            results.append({
                'replication': rep + 1,
                'seed': seed,
                'converged': False
            })

    return pd.DataFrame(results)


def summarize_monte_carlo(mc_results: pd.DataFrame,
                          param_names: List[str]) -> pd.DataFrame:
    """
    Summarize Monte Carlo simulation results.

    Args:
        mc_results: DataFrame from monte_carlo_simulation
        param_names: List of parameter names to summarize

    Returns:
        Summary DataFrame with bias, RMSE, coverage, etc.
    """
    # Filter to converged replications
    converged = mc_results[mc_results['converged'] == True]
    n_converged = len(converged)
    n_total = len(mc_results)

    summary = []
    for param in param_names:
        true_col = f'{param}_true'
        est_col = f'{param}_est'
        se_col = f'{param}_se'
        bias_col = f'{param}_bias'

        if est_col not in converged.columns:
            continue

        true_val = converged[true_col].iloc[0] if true_col in converged.columns else np.nan
        estimates = converged[est_col].dropna()
        std_errors = converged[se_col].dropna()
        biases = converged[bias_col].dropna()

        # Coverage
        if len(estimates) > 0 and len(std_errors) > 0:
            lower = estimates - 1.96 * std_errors
            upper = estimates + 1.96 * std_errors
            coverage = np.mean((lower <= true_val) & (true_val <= upper))
        else:
            coverage = np.nan

        summary.append({
            'parameter': param,
            'true_value': true_val,
            'mean_estimate': estimates.mean() if len(estimates) > 0 else np.nan,
            'std_estimate': estimates.std() if len(estimates) > 0 else np.nan,
            'mean_bias': biases.mean() if len(biases) > 0 else np.nan,
            'rmse': np.sqrt((biases ** 2).mean()) if len(biases) > 0 else np.nan,
            'mean_se': std_errors.mean() if len(std_errors) > 0 else np.nan,
            'coverage_95': coverage,
            'convergence_rate': n_converged / n_total,
            'n_converged': n_converged,
            'n_total': n_total
        })

    return pd.DataFrame(summary)


if __name__ == '__main__':
    print("Sensitivity Analysis Module")
    print("=" * 40)
    print("\nClasses:")
    print("  SensitivityAnalyzer - Full sensitivity analysis workflow")
    print("\nFunctions:")
    print("  monte_carlo_simulation - Run MC parameter recovery study")
    print("  summarize_monte_carlo - Summarize MC results")
    print("  compute_coverage_statistics - Coverage at multiple levels")
