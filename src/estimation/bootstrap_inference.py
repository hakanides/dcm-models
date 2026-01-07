"""
Bootstrap Inference for DCM Models
===================================

Implements bootstrap methods for robust standard errors and confidence intervals
in discrete choice models.

Methods:
- Nonparametric bootstrap (resample individuals)
- Parametric bootstrap (simulate from estimated model)
- Cluster bootstrap (for panel data)

Usage:
    from src.estimation.bootstrap_inference import BootstrapEstimator

    boot = BootstrapEstimator(model_func, database)
    results = boot.nonparametric_bootstrap(n_bootstrap=500)
    boot.print_summary()

Author: DCM Research Team
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass
import warnings
from scipy import stats


@dataclass
class BootstrapResults:
    """Container for bootstrap inference results."""
    parameter_names: List[str]
    point_estimates: Dict[str, float]
    bootstrap_estimates: np.ndarray  # Shape: (n_bootstrap, n_params)
    bootstrap_se: Dict[str, float]
    ci_percentile: Dict[str, Tuple[float, float]]
    ci_bca: Dict[str, Tuple[float, float]]  # Bias-corrected accelerated
    n_bootstrap: int
    n_failed: int
    method: str


class BootstrapEstimator:
    """
    Bootstrap inference for DCM parameter estimates.

    Provides bootstrap standard errors and confidence intervals that are
    robust to model misspecification and heteroskedasticity.
    """

    def __init__(self,
                 model_func: Callable,
                 df: pd.DataFrame,
                 id_col: str = 'ID',
                 choice_col: str = 'CHOICE'):
        """
        Initialize bootstrap estimator.

        Args:
            model_func: Function(database) -> (logprob, name) for Biogeme
            df: Full dataset as DataFrame
            id_col: Column for individual identifier (for cluster bootstrap)
            choice_col: Column for choice variable
        """
        self.model_func = model_func
        self.df = df
        self.id_col = id_col
        self.choice_col = choice_col
        self.n_obs = len(df)
        self.n_individuals = df[id_col].nunique() if id_col in df.columns else self.n_obs

        # Store original estimates
        self.original_estimates = None
        self.bootstrap_results = None

    def _estimate_model(self, df_sample: pd.DataFrame) -> Optional[Dict[str, float]]:
        """
        Estimate model on a sample and return parameter estimates.

        Args:
            df_sample: Bootstrap sample

        Returns:
            Dictionary of parameter estimates or None if failed
        """
        try:
            import biogeme.database as db
            import biogeme.biogeme as bio

            # Drop string columns
            string_cols = df_sample.select_dtypes(include=['object']).columns.tolist()
            df_num = df_sample.drop(columns=string_cols, errors='ignore')

            database = db.Database('bootstrap', df_num)

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                logprob, name = self.model_func(database)
                biogeme_obj = bio.BIOGEME(database, logprob)
                biogeme_obj.model_name = 'bootstrap_sample'
                results = biogeme_obj.estimate()

            if not results.algorithm_has_converged:
                return None

            return results.get_beta_values()

        except Exception as e:
            return None

    def _get_original_estimates(self) -> Dict[str, float]:
        """Get estimates from full sample."""
        if self.original_estimates is None:
            self.original_estimates = self._estimate_model(self.df)
        return self.original_estimates

    def nonparametric_bootstrap(self,
                                n_bootstrap: int = 500,
                                cluster: bool = True,
                                seed: int = 42,
                                verbose: bool = True) -> BootstrapResults:
        """
        Nonparametric bootstrap (resample with replacement).

        For panel data, resamples individuals (cluster bootstrap) to preserve
        within-individual correlation structure.

        Args:
            n_bootstrap: Number of bootstrap replications
            cluster: If True, resample individuals; if False, resample observations
            seed: Random seed
            verbose: Print progress

        Returns:
            BootstrapResults object
        """
        np.random.seed(seed)

        if verbose:
            print(f"\n{'='*60}")
            print(f"NONPARAMETRIC BOOTSTRAP ({'cluster' if cluster else 'observation'} resampling)")
            print(f"{'='*60}")
            print(f"Bootstrap replications: {n_bootstrap}")

        # Get original estimates
        original = self._get_original_estimates()
        if original is None:
            raise ValueError("Original estimation failed")

        param_names = list(original.keys())
        n_params = len(param_names)

        if verbose:
            print(f"Parameters: {n_params}")
            print(f"{'Individuals' if cluster else 'Observations'}: {self.n_individuals if cluster else self.n_obs}")

        # Bootstrap loop
        bootstrap_estimates = []
        n_failed = 0

        for b in range(n_bootstrap):
            if verbose and (b + 1) % 50 == 0:
                print(f"  Completed {b + 1}/{n_bootstrap} replications ({n_failed} failed)")

            # Create bootstrap sample
            if cluster and self.id_col in self.df.columns:
                # Cluster bootstrap: resample individuals
                unique_ids = self.df[self.id_col].unique()
                sampled_ids = np.random.choice(unique_ids, size=len(unique_ids), replace=True)

                # Get all observations for sampled individuals
                dfs = []
                for i, uid in enumerate(sampled_ids):
                    df_ind = self.df[self.df[self.id_col] == uid].copy()
                    df_ind[self.id_col] = i  # Renumber to avoid duplicate IDs
                    dfs.append(df_ind)
                df_sample = pd.concat(dfs, ignore_index=True)
            else:
                # Observation bootstrap: resample observations
                idx = np.random.choice(self.n_obs, size=self.n_obs, replace=True)
                df_sample = self.df.iloc[idx].reset_index(drop=True)

            # Estimate on bootstrap sample
            estimates = self._estimate_model(df_sample)

            if estimates is not None:
                bootstrap_estimates.append([estimates.get(p, np.nan) for p in param_names])
            else:
                n_failed += 1

        if verbose:
            print(f"\nCompleted: {n_bootstrap - n_failed}/{n_bootstrap} successful")

        # Convert to array
        boot_array = np.array(bootstrap_estimates)

        # Compute bootstrap statistics
        bootstrap_se = {}
        ci_percentile = {}
        ci_bca = {}

        for i, param in enumerate(param_names):
            param_boots = boot_array[:, i]
            valid = param_boots[~np.isnan(param_boots)]

            if len(valid) > 0:
                bootstrap_se[param] = np.std(valid, ddof=1)

                # Percentile CI
                ci_percentile[param] = (
                    np.percentile(valid, 2.5),
                    np.percentile(valid, 97.5)
                )

                # BCa CI (bias-corrected and accelerated)
                ci_bca[param] = self._compute_bca_ci(
                    original[param], valid, self.df, param
                )
            else:
                bootstrap_se[param] = np.nan
                ci_percentile[param] = (np.nan, np.nan)
                ci_bca[param] = (np.nan, np.nan)

        results = BootstrapResults(
            parameter_names=param_names,
            point_estimates=original,
            bootstrap_estimates=boot_array,
            bootstrap_se=bootstrap_se,
            ci_percentile=ci_percentile,
            ci_bca=ci_bca,
            n_bootstrap=n_bootstrap,
            n_failed=n_failed,
            method='nonparametric_cluster' if cluster else 'nonparametric_obs'
        )

        self.bootstrap_results = results
        return results

    def _compute_bca_ci(self,
                        theta_hat: float,
                        theta_boot: np.ndarray,
                        df: pd.DataFrame,
                        param_name: str,
                        alpha: float = 0.05) -> Tuple[float, float]:
        """
        Compute BCa (bias-corrected and accelerated) confidence interval.

        Args:
            theta_hat: Original point estimate
            theta_boot: Bootstrap estimates
            df: Original data
            param_name: Parameter name
            alpha: Significance level

        Returns:
            Tuple of (lower, upper) CI bounds
        """
        n_boot = len(theta_boot)

        # Bias correction factor
        z0 = stats.norm.ppf(np.mean(theta_boot < theta_hat))

        # Acceleration factor (jackknife)
        # Simplified: use 0 if jackknife is too expensive
        a = 0.0  # Could compute from jackknife influence values

        # Adjusted percentiles
        z_alpha_lower = stats.norm.ppf(alpha / 2)
        z_alpha_upper = stats.norm.ppf(1 - alpha / 2)

        # BCa adjustment
        alpha_lower = stats.norm.cdf(z0 + (z0 + z_alpha_lower) / (1 - a * (z0 + z_alpha_lower)))
        alpha_upper = stats.norm.cdf(z0 + (z0 + z_alpha_upper) / (1 - a * (z0 + z_alpha_upper)))

        # Ensure valid percentiles
        alpha_lower = np.clip(alpha_lower, 0.001, 0.999)
        alpha_upper = np.clip(alpha_upper, 0.001, 0.999)

        ci_lower = np.percentile(theta_boot, alpha_lower * 100)
        ci_upper = np.percentile(theta_boot, alpha_upper * 100)

        return (ci_lower, ci_upper)

    def parametric_bootstrap(self,
                             n_bootstrap: int = 500,
                             n_obs: int = None,
                             seed: int = 42,
                             verbose: bool = True) -> BootstrapResults:
        """
        Parametric bootstrap (simulate from estimated model).

        Simulates new datasets from the estimated model and re-estimates.

        Args:
            n_bootstrap: Number of bootstrap replications
            n_obs: Observations per simulated dataset (default: same as original)
            seed: Random seed
            verbose: Print progress

        Returns:
            BootstrapResults object
        """
        # This requires a simulation function that can generate data
        # from the estimated model parameters
        raise NotImplementedError(
            "Parametric bootstrap requires a simulation function. "
            "Use nonparametric_bootstrap() instead, or implement "
            "a custom simulation function for your model."
        )

    def print_summary(self, show_boots: bool = False) -> None:
        """Print summary of bootstrap results."""
        if self.bootstrap_results is None:
            print("No bootstrap results available. Run bootstrap first.")
            return

        results = self.bootstrap_results
        original = self._get_original_estimates()

        print(f"\n{'='*70}")
        print("BOOTSTRAP INFERENCE SUMMARY")
        print(f"{'='*70}")
        print(f"Method: {results.method}")
        print(f"Replications: {results.n_bootstrap} ({results.n_failed} failed)")
        print()

        # Header
        print(f"{'Parameter':<20} {'Estimate':>10} {'Boot SE':>10} {'95% CI (Percentile)':>25}")
        print("-" * 70)

        for param in results.parameter_names:
            est = original.get(param, np.nan)
            se = results.bootstrap_se.get(param, np.nan)
            ci = results.ci_percentile.get(param, (np.nan, np.nan))

            ci_str = f"[{ci[0]:>8.4f}, {ci[1]:>8.4f}]"
            print(f"{param:<20} {est:>10.4f} {se:>10.4f} {ci_str:>25}")

        print("-" * 70)

        # BCa intervals
        print(f"\n{'Parameter':<20} {'95% CI (BCa)':>25}")
        print("-" * 50)
        for param in results.parameter_names:
            ci = results.ci_bca.get(param, (np.nan, np.nan))
            ci_str = f"[{ci[0]:>8.4f}, {ci[1]:>8.4f}]"
            print(f"{param:<20} {ci_str:>25}")

    def to_dataframe(self) -> pd.DataFrame:
        """Export bootstrap results to DataFrame."""
        if self.bootstrap_results is None:
            return pd.DataFrame()

        results = self.bootstrap_results
        original = self._get_original_estimates()

        rows = []
        for param in results.parameter_names:
            rows.append({
                'parameter': param,
                'estimate': original.get(param, np.nan),
                'bootstrap_se': results.bootstrap_se.get(param, np.nan),
                'ci_lower_pct': results.ci_percentile.get(param, (np.nan, np.nan))[0],
                'ci_upper_pct': results.ci_percentile.get(param, (np.nan, np.nan))[1],
                'ci_lower_bca': results.ci_bca.get(param, (np.nan, np.nan))[0],
                'ci_upper_bca': results.ci_bca.get(param, (np.nan, np.nan))[1],
            })

        return pd.DataFrame(rows)


def compare_standard_errors(analytical_se: Dict[str, float],
                            bootstrap_se: Dict[str, float],
                            param_names: List[str] = None) -> pd.DataFrame:
    """
    Compare analytical and bootstrap standard errors.

    Large differences may indicate model misspecification or
    heteroskedasticity.

    Args:
        analytical_se: Standard errors from Hessian
        bootstrap_se: Standard errors from bootstrap
        param_names: Parameters to compare (default: all)

    Returns:
        DataFrame with comparison
    """
    if param_names is None:
        param_names = list(analytical_se.keys())

    rows = []
    for param in param_names:
        ana_se = analytical_se.get(param, np.nan)
        boot_se = bootstrap_se.get(param, np.nan)

        ratio = boot_se / ana_se if ana_se and ana_se != 0 else np.nan

        rows.append({
            'parameter': param,
            'analytical_se': ana_se,
            'bootstrap_se': boot_se,
            'ratio': ratio,
            'pct_diff': (ratio - 1) * 100 if not np.isnan(ratio) else np.nan
        })

    df = pd.DataFrame(rows)

    # Add interpretation
    def interpret(ratio):
        if np.isnan(ratio):
            return 'N/A'
        elif ratio > 1.5:
            return 'Bootstrap much larger (possible misspecification)'
        elif ratio < 0.67:
            return 'Analytical much larger (unusual)'
        else:
            return 'Similar'

    df['interpretation'] = df['ratio'].apply(interpret)

    return df


if __name__ == '__main__':
    print("Bootstrap Inference Module")
    print("=" * 40)
    print("\nClasses:")
    print("  BootstrapEstimator - Main bootstrap inference class")
    print("  BootstrapResults - Container for results")
    print("\nFunctions:")
    print("  compare_standard_errors - Compare analytical vs bootstrap SE")
    print("\nUsage:")
    print("  boot = BootstrapEstimator(model_func, df)")
    print("  results = boot.nonparametric_bootstrap(n_bootstrap=500)")
    print("  boot.print_summary()")
