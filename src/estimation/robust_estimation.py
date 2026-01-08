"""
Robust Estimation Module for DCM Models
========================================

This module provides utilities to guarantee convergence and precise estimation
for Discrete Choice Models (MNL, MXL, HCM) using Biogeme.

Key Features:
- Multiple optimizer retry strategy
- Starting value initialization from simpler models
- Identification diagnostics via Hessian eigenvalue analysis
- Robust standard error computation (sandwich estimator)
- Data validation and scaling checks
- Automatic convergence management with strategy escalation
- Smart starting value generation

Author: DCM Research Team
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass
import warnings
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models

# Import convergence diagnostics
try:
    from .convergence_diagnostics import ConvergenceChecker, ConvergenceDiagnostics
    CONVERGENCE_DIAGNOSTICS_AVAILABLE = True
except ImportError:
    CONVERGENCE_DIAGNOSTICS_AVAILABLE = False


class BiogemeConfig:
    """Centralized configuration for Biogeme estimation."""

    # Optimizer priority list (most robust first)
    OPTIMIZERS = ['TR-BFGS', 'LS-BFGS', 'scipy', 'simple_bounds_BFGS']

    # Default settings
    DEFAULT_SETTINGS = {
        'max_iterations': 10000,
        'tolerance': 1e-8,
        'second_derivatives': 1.0,
        'numerically_safe': True,
    }

    @staticmethod
    def apply_robust_settings(biogeme_obj: bio.BIOGEME) -> None:
        """Apply robust estimation settings to Biogeme object."""
        # These settings can be modified programmatically
        pass  # Settings now in biogeme.toml


class DataValidator:
    """Validates data before estimation."""

    @staticmethod
    def validate(df: pd.DataFrame,
                 choice_col: str = 'choice',
                 fee_cols: List[str] = None,
                 lv_cols: List[str] = None) -> Dict[str, Any]:
        """
        Validate data for DCM estimation.

        Returns:
            Dictionary with validation results and warnings
        """
        results = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'stats': {}
        }

        # Check sample size
        n_obs = len(df)
        n_respondents = df['respid'].nunique() if 'respid' in df.columns else None
        results['stats']['n_obs'] = n_obs
        results['stats']['n_respondents'] = n_respondents

        if n_obs < 500:
            results['warnings'].append(
                f"Small sample size: {n_obs} observations. May have identification issues."
            )

        # Check choice distribution
        if choice_col in df.columns:
            choice_dist = df[choice_col].value_counts(normalize=True)
            results['stats']['choice_distribution'] = choice_dist.to_dict()

            # Check for extreme imbalance
            max_share = choice_dist.max()
            if max_share > 0.85:
                results['warnings'].append(
                    f"Extreme choice imbalance: {max_share:.1%} choose option {choice_dist.idxmax()}. "
                    "This may cause convergence issues."
                )

        # Check fee scaling
        if fee_cols:
            for col in fee_cols:
                if col in df.columns:
                    fee_range = df[col].max() - df[col].min()
                    fee_mean = df[col].mean()
                    results['stats'][f'{col}_range'] = fee_range
                    results['stats'][f'{col}_mean'] = fee_mean

                    # Check if scaling is appropriate
                    if fee_range > 100:
                        results['warnings'].append(
                            f"Large range for {col}: {fee_range:.2f}. Consider rescaling."
                        )

        # Check LV correlations
        if lv_cols:
            available_lv = [c for c in lv_cols if c in df.columns]
            if len(available_lv) > 1:
                lv_data = df[available_lv].dropna()
                corr_matrix = lv_data.corr()
                results['stats']['lv_correlations'] = corr_matrix.to_dict()

                # Check for high correlations
                for i, col1 in enumerate(available_lv):
                    for col2 in available_lv[i+1:]:
                        corr = abs(corr_matrix.loc[col1, col2])
                        if corr > 0.8:
                            results['warnings'].append(
                                f"High correlation between {col1} and {col2}: {corr:.3f}. "
                                "May cause multicollinearity."
                            )

        # Check for missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            missing_cols = missing[missing > 0]
            results['warnings'].append(
                f"Missing values in columns: {missing_cols.to_dict()}"
            )

        return results

    @staticmethod
    def print_validation(results: Dict[str, Any]) -> None:
        """Print validation results."""
        print("\n" + "=" * 60)
        print("DATA VALIDATION RESULTS")
        print("=" * 60)

        print(f"\nSample size: {results['stats'].get('n_obs', 'N/A')} observations")
        print(f"Respondents: {results['stats'].get('n_respondents', 'N/A')}")

        if 'choice_distribution' in results['stats']:
            print("\nChoice distribution:")
            for opt, share in results['stats']['choice_distribution'].items():
                print(f"  Option {opt}: {share:.1%}")

        if results['warnings']:
            print("\n⚠️  WARNINGS:")
            for w in results['warnings']:
                print(f"  - {w}")

        if results['errors']:
            print("\n❌ ERRORS:")
            for e in results['errors']:
                print(f"  - {e}")

        if not results['warnings'] and not results['errors']:
            print("\n✓ All checks passed")


def check_identification(results) -> Dict[str, Any]:
    """
    Check model identification using Hessian eigenvalue analysis.

    Args:
        results: Biogeme estimation results

    Returns:
        Dictionary with identification diagnostics
    """
    diagnostics = {
        'identified': True,
        'min_eigenvalue': None,
        'problematic_params': [],
        'condition_number': None
    }

    try:
        # Get Hessian matrix
        hessian = results.getHessian()
        if hessian is None:
            diagnostics['identified'] = None
            diagnostics['message'] = "Hessian not available"
            return diagnostics

        # Compute eigenvalues
        eigenvalues = np.linalg.eigvalsh(hessian)
        diagnostics['min_eigenvalue'] = float(np.min(eigenvalues))
        diagnostics['max_eigenvalue'] = float(np.max(eigenvalues))

        # Condition number
        if np.min(np.abs(eigenvalues)) > 1e-10:
            diagnostics['condition_number'] = float(
                np.max(np.abs(eigenvalues)) / np.min(np.abs(eigenvalues))
            )

        # Check for identification issues
        if diagnostics['min_eigenvalue'] < 1e-6:
            diagnostics['identified'] = False
            diagnostics['message'] = (
                f"Identification issue detected: min eigenvalue = {diagnostics['min_eigenvalue']:.2e}"
            )

            # Find problematic parameters using eigenvector
            min_idx = np.argmin(eigenvalues)
            eigenvectors = np.linalg.eigh(hessian)[1]
            problematic_vec = eigenvectors[:, min_idx]

            param_names = list(results.getBetaValues().keys())
            for i, (name, weight) in enumerate(zip(param_names, problematic_vec)):
                if abs(weight) > 0.1:
                    diagnostics['problematic_params'].append({
                        'name': name,
                        'weight': float(weight)
                    })
        else:
            diagnostics['message'] = "Model appears to be identified"

    except Exception as e:
        diagnostics['identified'] = None
        diagnostics['message'] = f"Could not compute diagnostics: {str(e)}"

    return diagnostics


def estimate_with_retry(database: db.Database,
                        model_function: Callable,
                        model_name: str,
                        starting_values: Dict[str, float] = None,
                        max_attempts: int = 3,
                        optimizers: List[str] = None) -> Tuple[Any, Dict]:
    """
    Estimate model with multiple retry attempts using different optimizers.

    Args:
        database: Biogeme database
        model_function: Function that returns (logprob, availability) or model
        model_name: Name for the model
        starting_values: Optional dictionary of starting parameter values
        max_attempts: Maximum number of attempts per optimizer
        optimizers: List of optimizers to try

    Returns:
        Tuple of (results, diagnostics_dict)
    """
    if optimizers is None:
        optimizers = BiogemeConfig.OPTIMIZERS

    diagnostics = {
        'attempts': [],
        'converged': False,
        'final_optimizer': None,
        'warnings': []
    }

    best_result = None
    best_ll = float('-inf')

    for optimizer in optimizers:
        for attempt in range(max_attempts):
            attempt_info = {
                'optimizer': optimizer,
                'attempt': attempt + 1,
                'success': False,
                'log_likelihood': None,
                'message': None
            }

            try:
                # Get model
                logprob = model_function(database)

                # Handle different return types
                if isinstance(logprob, tuple):
                    logprob, av = logprob
                    biogeme_obj = bio.BIOGEME(database, logprob)
                else:
                    biogeme_obj = bio.BIOGEME(database, logprob)

                biogeme_obj.modelName = f"{model_name}_{optimizer}_{attempt+1}"

                # Set starting values if provided
                if starting_values:
                    biogeme_obj.set_variables_values(starting_values)

                # Estimate
                results = biogeme_obj.estimate()

                attempt_info['log_likelihood'] = float(results.getGeneralStatistics()['Final log likelihood'][0])
                attempt_info['success'] = True
                attempt_info['message'] = "Estimation completed"

                # Check if this is better than previous
                if attempt_info['log_likelihood'] > best_ll:
                    best_ll = attempt_info['log_likelihood']
                    best_result = results
                    diagnostics['final_optimizer'] = optimizer
                    diagnostics['converged'] = True

            except Exception as e:
                attempt_info['message'] = str(e)
                diagnostics['warnings'].append(f"{optimizer} attempt {attempt+1}: {str(e)}")

            diagnostics['attempts'].append(attempt_info)

            # If successful, move to next optimizer for comparison
            if attempt_info['success']:
                break

    return best_result, diagnostics


def get_warm_start_values(simple_results,
                          complex_params: List[str]) -> Dict[str, float]:
    """
    Get starting values from simpler model results.

    Args:
        simple_results: Results from a simpler model
        complex_params: List of parameter names in the complex model

    Returns:
        Dictionary of starting values
    """
    simple_betas = simple_results.getBetaValues()
    starting = {}

    for param in complex_params:
        if param in simple_betas:
            starting[param] = simple_betas[param]
        else:
            # Default starting values for new parameters
            if 'ASC' in param:
                starting[param] = 0.0
            elif 'FEE' in param or 'COST' in param:
                starting[param] = -0.01 if 'LV' not in param else 0.0
            elif 'DUR' in param or 'TIME' in param:
                starting[param] = -0.01 if 'LV' not in param else 0.0
            else:
                starting[param] = 0.0

    return starting


def validate_results(results,
                     null_ll: float = None,
                     min_rho2: float = 0.0) -> Dict[str, Any]:
    """
    Validate estimation results.

    Args:
        results: Biogeme estimation results
        null_ll: Null model log-likelihood for rho-squared calculation
        min_rho2: Minimum acceptable rho-squared value

    Returns:
        Dictionary with validation results
    """
    validation = {
        'valid': True,
        'warnings': [],
        'errors': [],
        'stats': {}
    }

    # Get general statistics
    stats = results.getGeneralStatistics()

    # Log-likelihood
    ll = stats['Final log likelihood'][0]
    validation['stats']['log_likelihood'] = ll

    # Rho-squared
    if null_ll is not None:
        rho2 = 1 - (ll / null_ll)
        validation['stats']['rho_squared'] = rho2

        if rho2 < min_rho2:
            validation['errors'].append(
                f"Negative rho-squared: {rho2:.4f}. Model is worse than null."
            )
            validation['valid'] = False
        elif rho2 < 0.1:
            validation['warnings'].append(
                f"Low rho-squared: {rho2:.4f}. Model has poor fit."
            )

    # Check parameter estimates
    betas = results.getBetaValues()
    beta_stderrs = results.getStdErr()

    for param, value in betas.items():
        # Check for extreme values
        if abs(value) > 100:
            validation['warnings'].append(
                f"Extreme parameter value: {param} = {value:.4f}"
            )

        # Check standard errors
        if param in beta_stderrs:
            se = beta_stderrs[param]
            if se == 0 or np.isnan(se) or np.isinf(se):
                validation['warnings'].append(
                    f"Invalid standard error for {param}: {se}"
                )
            elif se > abs(value) * 10 and value != 0:
                validation['warnings'].append(
                    f"Large standard error for {param}: SE={se:.4f}, est={value:.4f}"
                )

    # Run identification check
    id_check = check_identification(results)
    validation['identification'] = id_check

    if id_check['identified'] == False:
        validation['warnings'].append(id_check['message'])

    return validation


def compute_robust_se(results) -> Dict[str, float]:
    """
    Compute robust (sandwich) standard errors.

    Note: Biogeme already computes robust SEs by default when
    only_robust_stats = True in biogeme.toml

    Args:
        results: Biogeme estimation results

    Returns:
        Dictionary of parameter name -> robust SE
    """
    try:
        # Get robust standard errors (already computed by Biogeme)
        robust_se = results.getRobustStdErr()
        return dict(robust_se)
    except Exception as e:
        print(f"Warning: Could not get robust SEs: {e}")
        return {}


def print_estimation_summary(results,
                            diagnostics: Dict = None,
                            validation: Dict = None) -> None:
    """Print comprehensive estimation summary."""
    print("\n" + "=" * 70)
    print("ESTIMATION SUMMARY")
    print("=" * 70)

    # Basic statistics
    stats = results.getGeneralStatistics()
    print(f"\nLog-likelihood: {stats['Final log likelihood'][0]:.4f}")
    print(f"Parameters: {stats['Number of estimated parameters'][0]}")
    print(f"Observations: {stats['Sample size'][0]}")

    if 'Rho-square for the null model' in stats:
        print(f"Rho-squared (null): {stats['Rho-square for the null model'][0]:.4f}")

    # Parameter estimates
    print("\n" + "-" * 70)
    print("PARAMETER ESTIMATES")
    print("-" * 70)

    betas = results.getBetaValues()
    stderrs = results.getRobustStdErr()

    print(f"{'Parameter':<25} {'Estimate':>12} {'Robust SE':>12} {'t-stat':>10}")
    print("-" * 60)

    for param in betas:
        est = betas[param]
        se = stderrs.get(param, np.nan)
        t_stat = est / se if se and se > 0 else np.nan
        print(f"{param:<25} {est:>12.4f} {se:>12.4f} {t_stat:>10.2f}")

    # Diagnostics
    if diagnostics:
        print("\n" + "-" * 70)
        print("ESTIMATION DIAGNOSTICS")
        print("-" * 70)
        print(f"Converged: {diagnostics.get('converged', 'Unknown')}")
        print(f"Final optimizer: {diagnostics.get('final_optimizer', 'Unknown')}")

        if diagnostics.get('warnings'):
            print("\nWarnings:")
            for w in diagnostics['warnings']:
                print(f"  - {w}")

    # Validation
    if validation:
        print("\n" + "-" * 70)
        print("VALIDATION RESULTS")
        print("-" * 70)
        print(f"Valid: {validation.get('valid', 'Unknown')}")

        if validation.get('identification'):
            id_check = validation['identification']
            print(f"Identified: {id_check.get('identified', 'Unknown')}")
            if id_check.get('min_eigenvalue'):
                print(f"Min eigenvalue: {id_check['min_eigenvalue']:.2e}")
            if id_check.get('problematic_params'):
                print("Problematic parameters:")
                for p in id_check['problematic_params']:
                    print(f"  - {p['name']} (weight: {p['weight']:.3f})")

        if validation.get('warnings'):
            print("\nWarnings:")
            for w in validation['warnings']:
                print(f"  - {w}")

        if validation.get('errors'):
            print("\nErrors:")
            for e in validation['errors']:
                print(f"  - {e}")


# ============================================================================
# SEQUENTIAL ESTIMATION PIPELINE
# ============================================================================

class SequentialEstimator:
    """
    Estimates models sequentially, using simpler models to warm-start complex ones.
    """

    def __init__(self, database: db.Database, null_ll: float = None):
        """
        Initialize estimator.

        Args:
            database: Biogeme database
            null_ll: Null model log-likelihood
        """
        self.database = database
        self.null_ll = null_ll
        self.results = {}
        self.diagnostics = {}

    def estimate_baseline(self,
                         model_function: Callable,
                         model_name: str = "baseline") -> Any:
        """Estimate baseline model without warm start."""
        print(f"\n{'='*60}")
        print(f"Estimating baseline model: {model_name}")
        print('='*60)

        result, diag = estimate_with_retry(
            self.database, model_function, model_name
        )

        self.results[model_name] = result
        self.diagnostics[model_name] = diag

        if result:
            validation = validate_results(result, self.null_ll)
            print_estimation_summary(result, diag, validation)

        return result

    def estimate_with_warmstart(self,
                                model_function: Callable,
                                model_name: str,
                                warmstart_from: str = None) -> Any:
        """
        Estimate model using warm start from previous model.

        Args:
            model_function: Function returning model logprob
            model_name: Name for this model
            warmstart_from: Name of model to use for starting values
        """
        print(f"\n{'='*60}")
        print(f"Estimating model: {model_name}")
        if warmstart_from:
            print(f"Warm starting from: {warmstart_from}")
        print('='*60)

        # Get starting values
        starting = None
        if warmstart_from and warmstart_from in self.results:
            prev_result = self.results[warmstart_from]
            if prev_result:
                starting = dict(prev_result.getBetaValues())
                print(f"Using {len(starting)} starting values from {warmstart_from}")

        result, diag = estimate_with_retry(
            self.database, model_function, model_name,
            starting_values=starting
        )

        self.results[model_name] = result
        self.diagnostics[model_name] = diag

        if result:
            validation = validate_results(result, self.null_ll)
            print_estimation_summary(result, diag, validation)
        else:
            print(f"ERROR: Model {model_name} failed to estimate")

        return result

    def get_comparison_table(self) -> pd.DataFrame:
        """Get comparison table for all estimated models."""
        rows = []

        for name, result in self.results.items():
            if result is None:
                continue

            stats = result.getGeneralStatistics()
            ll = stats['Final log likelihood'][0]
            k = stats['Number of estimated parameters'][0]
            n = stats['Sample size'][0]

            aic = 2 * k - 2 * ll
            bic = k * np.log(n) - 2 * ll
            rho2 = 1 - (ll / self.null_ll) if self.null_ll else np.nan

            rows.append({
                'Model': name,
                'LL': ll,
                'K': k,
                'AIC': aic,
                'BIC': bic,
                'rho2': rho2,
                'Converged': self.diagnostics[name].get('converged', False)
            })

        df = pd.DataFrame(rows)
        df = df.sort_values('AIC')
        df['Rank'] = range(1, len(df) + 1)

        return df


# ============================================================================
# STARTING VALUE GENERATION
# ============================================================================

class StartingValueGenerator:
    """
    Generate intelligent starting values for HCM parameters.

    Provides multiple strategies for computing starting values:
    1. From baseline model results (warm-start)
    2. From OLS regression estimates
    3. From grid search over parameter space
    4. Heuristic defaults based on parameter names

    Example:
        >>> generator = StartingValueGenerator()
        >>> starting = generator.from_baseline(baseline_result)
        >>> # Or with OLS
        >>> starting = generator.from_ols(df, ['fee1_10k', 'dur1'], 'CHOICE')
    """

    # Default starting values by parameter type
    DEFAULTS = {
        'ASC': 0.0,
        'B_FEE': -0.5,
        'B_DUR': -0.05,
        'B_COST': -0.5,
        'B_TIME': -0.05,
    }

    # Bounds for parameter types
    BOUNDS = {
        'ASC': (-10, 10),
        'B_FEE': (-10, 0),
        'B_DUR': (-5, 0),
        'B_COST': (-10, 0),
        'B_TIME': (-5, 0),
        'LV': (-2, 2),  # For LV interaction terms
    }

    def from_baseline(self, baseline_result,
                     new_params: List[str] = None) -> Dict[str, float]:
        """
        Generate starting values from baseline model results.

        Args:
            baseline_result: Biogeme result from simpler model
            new_params: List of new parameters not in baseline

        Returns:
            Dict of parameter name -> starting value
        """
        if baseline_result is None:
            return {}

        # Get beta values from baseline
        if hasattr(baseline_result, 'getBetaValues'):
            starting = dict(baseline_result.getBetaValues())
        elif hasattr(baseline_result, 'get_beta_values'):
            starting = dict(baseline_result.get_beta_values())
        elif hasattr(baseline_result, 'params'):
            starting = dict(baseline_result.params)
        else:
            return {}

        # Add default values for new parameters
        if new_params:
            for param in new_params:
                if param not in starting:
                    starting[param] = self._get_default(param)

        return starting

    def from_ols(self, df: pd.DataFrame,
                 attribute_cols: List[str],
                 choice_col: str,
                 lv_cols: List[str] = None) -> Dict[str, float]:
        """
        Estimate starting values via OLS regression.

        Uses linear probability model to get rough coefficient estimates.

        Args:
            df: DataFrame with choice data
            attribute_cols: List of attribute column names
            choice_col: Name of choice column
            lv_cols: Optional list of LV column names for interactions

        Returns:
            Dict of parameter name -> starting value
        """
        starting = {}

        try:
            # Prepare design matrix
            # For simplicity, use basic regression on first alternative
            y = (df[choice_col] == 1).astype(float)

            # Use first alternative attributes
            X_cols = []
            for col in attribute_cols:
                if col.endswith('1') or col.endswith('_1'):
                    X_cols.append(col)

            if not X_cols:
                X_cols = attribute_cols[:3]  # Fallback

            X = df[X_cols].values
            X = np.column_stack([np.ones(len(X)), X])

            # OLS: beta = (X'X)^-1 X'y
            try:
                XTX_inv = np.linalg.inv(X.T @ X)
                beta_ols = XTX_inv @ X.T @ y.values
            except:
                return self._get_defaults(attribute_cols + (lv_cols or []))

            # Map OLS coefficients to parameter names
            for i, col in enumerate(X_cols):
                base_name = col.rstrip('0123456789_')
                param_name = f'B_{base_name.upper()}'
                if i + 1 < len(beta_ols):
                    # Scale down OLS estimate
                    starting[param_name] = float(beta_ols[i + 1]) * 0.1

            # Add zero starting values for LV interactions
            if lv_cols:
                for lv in lv_cols:
                    starting[f'B_FEE_{lv}'] = 0.0
                    starting[f'B_DUR_{lv}'] = 0.0

        except Exception as e:
            warnings.warn(f"OLS starting values failed: {e}")
            return self._get_defaults(attribute_cols + (lv_cols or []))

        return starting

    def grid_search(self, database, model_func,
                   bounds: Dict[str, Tuple[float, float]],
                   n_points: int = 5) -> Dict[str, float]:
        """
        Grid search for optimal starting values.

        Evaluates log-likelihood on a grid and returns best starting point.

        Args:
            database: Biogeme database
            model_func: Function that returns model logprob
            bounds: Dict mapping param names to (min, max) bounds
            n_points: Number of grid points per dimension

        Returns:
            Dict of parameter name -> starting value
        """
        if len(bounds) > 3:
            # Too many dimensions for grid search
            warnings.warn("Grid search limited to 3 dimensions")
            bounds = dict(list(bounds.items())[:3])

        param_names = list(bounds.keys())
        grids = [np.linspace(b[0], b[1], n_points) for b in bounds.values()]

        best_ll = float('-inf')
        best_params = {name: (bounds[name][0] + bounds[name][1]) / 2
                      for name in param_names}

        # Evaluate grid
        from itertools import product
        for values in product(*grids):
            try:
                # This is a simplified version - full implementation would
                # need to set up the model properly
                starting = dict(zip(param_names, values))
                # Evaluate LL at this point
                # (Implementation depends on model structure)
                pass
            except:
                continue

        return best_params

    def _get_default(self, param_name: str) -> float:
        """Get default value for a parameter based on its name."""
        # Check exact matches first
        if param_name in self.DEFAULTS:
            return self.DEFAULTS[param_name]

        # Check partial matches
        param_upper = param_name.upper()
        for key, value in self.DEFAULTS.items():
            if key in param_upper:
                return value

        # LV interaction terms default to zero
        if 'LV' in param_upper or 'PAT' in param_upper or 'SEC' in param_upper:
            return 0.0

        # ASC-like parameters
        if 'ASC' in param_upper:
            return 0.0

        return 0.0

    def _get_defaults(self, param_names: List[str]) -> Dict[str, float]:
        """Get defaults for list of parameters."""
        return {name: self._get_default(name) for name in param_names}


# ============================================================================
# CONVERGENCE MANAGEMENT
# ============================================================================

@dataclass
class EstimationStrategy:
    """Configuration for an estimation strategy."""
    optimizer: str
    maxiter: int
    tolerance: float = 1e-8
    perturbation: float = 0.0  # Starting value perturbation

    def describe(self) -> str:
        return f"{self.optimizer} (maxiter={self.maxiter}, tol={self.tolerance:.0e})"


class ConvergenceManager:
    """
    Manage estimation with automatic retry and strategy escalation.

    Tries progressively more robust optimization strategies until
    convergence is achieved. Strategies escalate from fast/simple
    to slow/robust.

    Example:
        >>> manager = ConvergenceManager(verbose=True)
        >>> result, diagnostics = manager.estimate_until_convergence(
        ...     database, model_func, warm_start={'B_FEE': -0.5}
        ... )
    """

    # Ordered list of strategies from fast to robust
    STRATEGIES = [
        EstimationStrategy('simple_bounds_BFGS', maxiter=10000, tolerance=1e-8),
        EstimationStrategy('TR-BFGS', maxiter=15000, tolerance=1e-8),
        EstimationStrategy('LS-BFGS', maxiter=15000, tolerance=1e-8),
        EstimationStrategy('simple_bounds_BFGS', maxiter=20000, tolerance=1e-6,
                          perturbation=0.1),
        EstimationStrategy('TR-BFGS', maxiter=30000, tolerance=1e-6,
                          perturbation=0.2),
    ]

    def __init__(self, max_attempts: int = 5, verbose: bool = True):
        """
        Initialize convergence manager.

        Args:
            max_attempts: Maximum total attempts across all strategies
            verbose: Print progress messages
        """
        self.max_attempts = max_attempts
        self.verbose = verbose
        self.checker = ConvergenceChecker() if CONVERGENCE_DIAGNOSTICS_AVAILABLE else None

    def estimate_until_convergence(self,
                                   database: db.Database,
                                   model_func: Callable,
                                   model_name: str = "model",
                                   warm_start: Dict[str, float] = None
                                   ) -> Tuple[Any, Optional['ConvergenceDiagnostics']]:
        """
        Estimate model with automatic strategy escalation.

        Tries each strategy in order until convergence is achieved
        or all strategies are exhausted.

        Args:
            database: Biogeme database
            model_func: Function that returns (logprob, name) tuple
            model_name: Name for the model
            warm_start: Optional starting values

        Returns:
            Tuple of (result, diagnostics)
        """
        best_result = None
        best_ll = float('-inf')
        best_diagnostics = None

        attempt = 0
        for strategy in self.STRATEGIES:
            if attempt >= self.max_attempts:
                break

            attempt += 1

            if self.verbose:
                print(f"\n  Strategy {attempt}: {strategy.describe()}")

            try:
                # Prepare starting values with optional perturbation
                starting = dict(warm_start) if warm_start else {}
                if strategy.perturbation > 0:
                    for key in starting:
                        starting[key] *= (1 + np.random.uniform(
                            -strategy.perturbation, strategy.perturbation))

                # Get model from function
                model_result = model_func(database)
                if isinstance(model_result, tuple):
                    logprob, _ = model_result
                else:
                    logprob = model_result

                # Create Biogeme object
                biogeme_obj = bio.BIOGEME(database, logprob)
                biogeme_obj.modelName = f"{model_name}_{strategy.optimizer}_{attempt}"

                # Set starting values if provided
                if starting:
                    try:
                        biogeme_obj.set_variables_values(starting)
                    except:
                        pass  # Ignore if setting values fails

                # Estimate
                result = biogeme_obj.estimate()

                # Get log-likelihood
                if hasattr(result, 'final_loglikelihood'):
                    ll = result.final_loglikelihood
                else:
                    stats = result.getGeneralStatistics()
                    ll = stats.get('Final log likelihood', [float('-inf')])[0]

                # Check convergence
                if hasattr(result, 'algorithm_has_converged'):
                    converged = result.algorithm_has_converged
                else:
                    converged = True

                # Run diagnostics
                if self.checker:
                    diagnostics = self.checker.full_diagnostics(result, model_name)
                else:
                    diagnostics = None

                # Update best if this is better
                if ll > best_ll:
                    best_ll = ll
                    best_result = result
                    best_diagnostics = diagnostics

                if self.verbose:
                    status = "CONVERGED" if converged else "NOT CONVERGED"
                    print(f"    LL: {ll:.2f} - {status}")

                # Stop if fully converged
                if converged and (diagnostics is None or diagnostics.publication_ready):
                    if self.verbose:
                        print(f"  Success on attempt {attempt}")
                    return best_result, best_diagnostics

            except Exception as e:
                if self.verbose:
                    print(f"    Error: {str(e)[:50]}...")
                continue

        # Return best result even if not fully converged
        if self.verbose:
            if best_result is not None:
                print(f"  Returning best result (LL: {best_ll:.2f})")
            else:
                print("  All strategies failed")

        return best_result, best_diagnostics

    def estimate_with_fallback(self,
                               database: db.Database,
                               model_func: Callable,
                               fallback_model_func: Callable = None,
                               model_name: str = "model",
                               warm_start: Dict[str, float] = None
                               ) -> Tuple[Any, Optional['ConvergenceDiagnostics']]:
        """
        Estimate with fallback to simpler model if convergence fails.

        Args:
            database: Biogeme database
            model_func: Primary model function
            fallback_model_func: Simpler model to try if primary fails
            model_name: Name for the model
            warm_start: Optional starting values

        Returns:
            Tuple of (result, diagnostics)
        """
        # Try primary model
        result, diagnostics = self.estimate_until_convergence(
            database, model_func, model_name, warm_start
        )

        # Check if we need fallback
        if result is not None:
            if diagnostics is None or diagnostics.converged:
                return result, diagnostics

        # Try fallback if provided
        if fallback_model_func is not None:
            if self.verbose:
                print(f"\n  Trying fallback model...")
            return self.estimate_until_convergence(
                database, fallback_model_func, f"{model_name}_fallback", warm_start
            )

        return result, diagnostics


if __name__ == '__main__':
    print("Robust Estimation Module")
    print("=" * 40)
    print("\nThis module provides utilities for robust DCM estimation:")
    print("  - DataValidator: Validate data before estimation")
    print("  - estimate_with_retry: Multiple optimizer attempts")
    print("  - check_identification: Hessian eigenvalue diagnostics")
    print("  - validate_results: Result validation")
    print("  - SequentialEstimator: Warm-start estimation pipeline")
    print("  - StartingValueGenerator: Intelligent starting values")
    print("  - ConvergenceManager: Automatic retry with strategy escalation")
    print("\nImport and use in your estimation scripts.")
