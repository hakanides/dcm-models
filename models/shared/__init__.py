"""
Shared utilities for isolated DCM model folders.

Provides common functionality for:
- cleanup: Clean results and output files before each run
- sample_stats: Generate population statistics after simulation
- policy_tools: WTP, elasticity, market share calculations
- latex_tools: LaTeX table generation
- validation: Data column validation utilities
"""

from typing import List
import pandas as pd


def validate_required_columns(df: pd.DataFrame, required_columns: List[str],
                              model_name: str = 'Model') -> None:
    """
    Validate that all required columns exist in the dataframe.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        model_name: Name of model for error messages

    Raises:
        ValueError: If any required columns are missing
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        available = sorted(df.columns.tolist())
        raise ValueError(
            f"{model_name}: Missing required columns: {missing}\n"
            f"Available columns: {available}"
        )


def validate_coefficient_signs(betas: dict, expected_signs: dict = None,
                                model_name: str = 'Model', verbose: bool = True) -> List[str]:
    """
    Validate that estimated coefficients have economically sensible signs.

    Standard expectations:
    - B_FEE (or B_FEE_MU): negative (higher costs reduce utility)
    - B_DUR: negative (longer duration reduces utility)

    Args:
        betas: Dictionary of estimated parameter values
        expected_signs: Dict mapping param names to expected sign ('negative', 'positive')
                       If None, uses standard expectations for fee and duration
        model_name: Name of model for warnings
        verbose: Whether to print warnings

    Returns:
        List of parameter names with unexpected signs
    """
    if expected_signs is None:
        expected_signs = {
            'B_FEE': 'negative',
            'B_FEE_MU': 'negative',
            'B_DUR': 'negative',
        }

    issues = []
    for param, expected in expected_signs.items():
        if param in betas:
            value = betas[param]
            if expected == 'negative' and value > 0:
                issues.append(param)
                if verbose:
                    print(f"\nWARNING [{model_name}]: {param} = {value:.4f} (expected negative)")
            elif expected == 'positive' and value < 0:
                issues.append(param)
                if verbose:
                    print(f"\nWARNING [{model_name}]: {param} = {value:.4f} (expected positive)")

    if issues and verbose:
        print("  Sign violations may indicate estimation issues or data problems.")

    return issues


def safe_biogeme_estimate(biogeme_model, max_retries: int = 2,
                           verbose: bool = True):
    """
    Safely estimate a Biogeme model with error handling.

    Handles common Biogeme failures:
    - Singular Hessian (identification issues)
    - Convergence failures
    - API version differences

    Args:
        biogeme_model: Configured Biogeme BIOGEME object
        max_retries: Maximum estimation attempts
        verbose: Print status messages

    Returns:
        Biogeme results object, or None if estimation failed

    Raises:
        RuntimeError: If all retry attempts fail
    """
    import warnings

    last_error = None
    for attempt in range(max_retries):
        try:
            results = biogeme_model.estimate()

            # Check if estimation converged
            if hasattr(results, 'algorithm_has_converged'):
                if not results.algorithm_has_converged:
                    warnings.warn(
                        f"Estimation did not converge (attempt {attempt + 1}/{max_retries})",
                        UserWarning
                    )
                    if attempt < max_retries - 1:
                        continue

            return results

        except Exception as e:
            last_error = e
            error_msg = str(e).lower()

            if 'singular' in error_msg or 'hessian' in error_msg:
                if verbose:
                    print(f"  Identification issue detected (attempt {attempt + 1})")
                    print(f"  Consider fixing parameters or checking data variation")
            elif 'convergence' in error_msg:
                if verbose:
                    print(f"  Convergence failure (attempt {attempt + 1})")
            else:
                if verbose:
                    print(f"  Estimation error: {e}")

            if attempt >= max_retries - 1:
                raise RuntimeError(
                    f"Biogeme estimation failed after {max_retries} attempts. "
                    f"Last error: {last_error}"
                )

    return None


from .cleanup import cleanup_before_run, cleanup_simulation_outputs
from .sample_stats import generate_sample_stats
from .policy_tools import (
    compute_wtp,
    compute_elasticity_matrix,
    compute_market_shares,
    compute_marginal_effects,
    run_policy_analysis
)
from .latex_tools import (
    generate_parameter_table,
    generate_model_summary,
    generate_policy_summary,
    generate_all_latex
)

__all__ = [
    'cleanup_before_run',
    'cleanup_simulation_outputs',
    'generate_sample_stats',
    'validate_required_columns',
    'validate_coefficient_signs',
    'safe_biogeme_estimate',
    'compute_wtp',
    'compute_elasticity_matrix',
    'compute_market_shares',
    'compute_marginal_effects',
    'run_policy_analysis',
    'generate_parameter_table',
    'generate_model_summary',
    'generate_policy_summary',
    'generate_all_latex',
]
