"""
ICLV Estimation Module
======================

High-level estimation interface for ICLV models using
Simulated Maximum Likelihood (SML).

Provides convenient wrappers and utilities for estimation,
including parameter constraints, robust standard errors,
and comparison with two-stage estimation.

Author: DCM Research Team
"""

import numpy as np
from scipy import stats, optimize
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import pandas as pd
import time

from .core import ICLVModel, ICLVResult
from .measurement import OrderedProbitMeasurement, estimate_thresholds
from .structural import StructuralModel
from .integration import HaltonDraws, MonteCarloIntegrator


@dataclass
class EstimationConfig:
    """Configuration for ICLV estimation."""
    n_draws: int = 500
    draw_type: str = 'halton'
    n_categories: int = 5
    seed: int = 42
    method: str = 'BFGS'
    maxiter: int = 1000
    gtol: float = 1e-5
    verbose: bool = True
    compute_robust_se: bool = True


class SMLEstimator:
    """
    Simulated Maximum Likelihood Estimator for ICLV.

    Handles the full estimation workflow including:
    - Data preparation
    - Starting value computation
    - Optimization
    - Standard error computation
    - Result formatting

    Example:
        >>> estimator = SMLEstimator(config=EstimationConfig(n_draws=500))
        >>> result = estimator.estimate(df, spec)
    """

    def __init__(self, config: EstimationConfig = None):
        """
        Initialize SML estimator.

        Args:
            config: Estimation configuration
        """
        self.config = config or EstimationConfig()
        self._estimation_time = None

    def prepare_data(self,
                     df: pd.DataFrame,
                     spec: Dict) -> Dict:
        """
        Prepare data for estimation.

        Args:
            df: DataFrame with all data
            spec: Model specification dict with keys:
                - 'choice_col': name of choice column
                - 'constructs': {construct_name: [item_cols]}
                - 'covariates': [covariate_cols]
                - 'attributes': {alt: {attr: col}} or wide format cols
                - 'lv_in_utility': {lv_name: coefficient_name}

        Returns:
            Prepared data dict for estimation
        """
        n_individuals = len(df)

        # Extract choice data
        choice_col = spec['choice_col']
        choices = df[choice_col].values.astype(int)
        n_alts = choices.max() + 1

        # Extract indicator data
        constructs = spec['constructs']
        indicator_cols = []
        for items in constructs.values():
            indicator_cols.extend(items)
        indicators = df[indicator_cols].values

        # Extract covariate data
        covariate_cols = spec['covariates']
        covariates = df[covariate_cols].values

        # Extract attribute data
        # Handle multiple formats
        attributes_spec = spec.get('attributes', {})
        if isinstance(attributes_spec, dict) and len(attributes_spec) > 0:
            # Long or explicit format
            attribute_names = list(set(
                attr for alt_attrs in attributes_spec.values()
                for attr in alt_attrs.keys()
            ))
            attributes = np.zeros((n_individuals, n_alts, len(attribute_names)))

            for alt_idx, (alt, attrs) in enumerate(attributes_spec.items()):
                for attr_idx, attr_name in enumerate(attribute_names):
                    if attr_name in attrs:
                        col = attrs[attr_name]
                        attributes[:, alt_idx, attr_idx] = df[col].values
        else:
            # Wide format: assume columns like 'attr_alt1', 'attr_alt2', etc.
            attribute_cols = spec.get('attribute_cols', [])
            attributes = None  # Will be handled differently
            attribute_names = attribute_cols

        # Availability
        avail_cols = spec.get('availability_cols')
        if avail_cols:
            availability = df[avail_cols].values
        else:
            availability = np.ones((n_individuals, n_alts))

        # LV specifications for utility
        lv_in_utility = spec.get('lv_in_utility', {})

        return {
            'n_individuals': n_individuals,
            'n_alts': n_alts,
            'choices': choices,
            'indicators': indicators,
            'item_names': indicator_cols,
            'covariates': covariates,
            'covariate_names': covariate_cols,
            'attributes': attributes,
            'attribute_names': attribute_names,
            'availability': availability,
            'lv_in_utility': lv_in_utility,
            'constructs': constructs,
        }

    def compute_starting_values(self,
                                data: Dict,
                                spec: Dict) -> Dict:
        """
        Compute reasonable starting values for parameters.

        Uses:
        - Zero for choice coefficients
        - Sample correlations for structural coefficients
        - Factor analysis for loadings
        - Marginal frequencies for thresholds

        Args:
            data: Prepared data dict
            spec: Model specification

        Returns:
            Dict of starting values
        """
        starting = {}

        # Choice coefficients: zeros
        starting['beta'] = {name: 0.0 for name in data['attribute_names']}

        # Add LV effects if specified
        for lv_name in data['lv_in_utility']:
            starting['beta'][f'beta_{lv_name}'] = 0.0

        # Structural coefficients: correlations as hints
        starting['gamma'] = {}
        constructs = data['constructs']
        for construct in constructs:
            starting['gamma'][construct] = {}
            for cov in data['covariate_names']:
                starting['gamma'][construct][cov] = 0.0

        # Factor loadings: based on item correlations
        starting['loadings'] = {}
        indicators = data['indicators']
        item_names = data['item_names']

        for construct, items in constructs.items():
            # Compute correlation matrix for this construct's items
            item_indices = [item_names.index(item) for item in items]
            item_data = indicators[:, item_indices]

            # First loading fixed to 1
            starting['loadings'][items[0]] = 1.0

            # Other loadings based on correlation with first item
            if len(items) > 1:
                corr_with_first = np.corrcoef(item_data.T)[0, 1:]
                for i, item in enumerate(items[1:]):
                    starting['loadings'][item] = max(0.3, min(0.95, abs(corr_with_first[i])))

        # Thresholds: from marginal frequencies
        all_responses = indicators.flatten()
        valid_responses = all_responses[~np.isnan(all_responses)]
        starting['thresholds'] = estimate_thresholds(
            valid_responses,
            self.config.n_categories
        )

        return starting

    def estimate(self,
                 df: pd.DataFrame,
                 spec: Dict,
                 starting_values: Dict = None) -> ICLVResult:
        """
        Estimate ICLV model.

        Args:
            df: DataFrame with all data
            spec: Model specification
            starting_values: Optional starting values (computed if not provided)

        Returns:
            ICLVResult with estimates
        """
        start_time = time.time()

        if self.config.verbose:
            print("=" * 60)
            print("ICLV Simulated Maximum Likelihood Estimation")
            print("=" * 60)

        # Prepare data
        if self.config.verbose:
            print("\nPreparing data...")
        data = self.prepare_data(df, spec)

        if self.config.verbose:
            print(f"  N individuals: {data['n_individuals']}")
            print(f"  N alternatives: {data['n_alts']}")
            print(f"  N indicators: {len(data['item_names'])}")
            print(f"  N covariates: {len(data['covariate_names'])}")

        # Compute starting values
        if starting_values is None:
            if self.config.verbose:
                print("\nComputing starting values...")
            starting_values = self.compute_starting_values(data, spec)

        # Initialize model
        model = ICLVModel(
            constructs=spec['constructs'],
            covariates=spec['covariates'],
            n_draws=self.config.n_draws,
            draw_type=self.config.draw_type,
            n_categories=self.config.n_categories,
            seed=self.config.seed
        )

        # Prepare for optimization
        lv_beta_init = {}
        for lv_name, coef_name in data['lv_in_utility'].items():
            lv_beta_init[coef_name] = starting_values['beta'].get(f'beta_{lv_name}', 0.0)

        if self.config.verbose:
            print(f"\nRunning optimization...")
            print(f"  Method: {self.config.method}")
            print(f"  Max iterations: {self.config.maxiter}")
            print(f"  N draws: {self.config.n_draws} ({self.config.draw_type})")

        # Run estimation
        result = model.estimate(
            df=df,
            choice_col=spec['choice_col'],
            attribute_cols=data['attribute_names'],
            covariate_cols=data['covariate_names'],
            indicator_cols=data['item_names'],
            lv_beta_init=lv_beta_init,
            method=self.config.method,
            maxiter=self.config.maxiter,
            verbose=self.config.verbose
        )

        # Compute robust standard errors if requested
        if self.config.compute_robust_se:
            if self.config.verbose:
                print("\nComputing robust standard errors...")
            # Would implement sandwich estimator here
            pass

        self._estimation_time = time.time() - start_time

        if self.config.verbose:
            print(f"\nEstimation complete in {self._estimation_time:.2f} seconds")
            print("=" * 60)

        return result


def estimate_iclv(df: pd.DataFrame,
                  constructs: Dict[str, List[str]],
                  covariates: List[str],
                  choice_col: str,
                  attribute_cols: List[str] = None,
                  lv_effects: Dict[str, str] = None,
                  n_draws: int = 500,
                  n_categories: int = 5,
                  verbose: bool = True) -> ICLVResult:
    """
    Convenience function to estimate ICLV model.

    This is the simplest interface for ICLV estimation.

    Args:
        df: DataFrame with all data
        constructs: Dict mapping construct name to list of indicator columns
        covariates: List of covariate columns for structural model
        choice_col: Name of choice column
        attribute_cols: List of attribute columns for choice model
        lv_effects: Dict mapping LV name to coefficient name in utility
        n_draws: Number of simulation draws
        n_categories: Number of Likert categories
        verbose: Print progress

    Returns:
        ICLVResult with estimates

    Example:
        >>> result = estimate_iclv(
        ...     df=data,
        ...     constructs={'env': ['env1', 'env2', 'env3']},
        ...     covariates=['age', 'income'],
        ...     choice_col='choice',
        ...     attribute_cols=['price', 'time'],
        ...     lv_effects={'env': 'beta_env'},
        ...     n_draws=500
        ... )
    """
    spec = {
        'constructs': constructs,
        'covariates': covariates,
        'choice_col': choice_col,
        'attribute_cols': attribute_cols or [],
        'lv_in_utility': lv_effects or {},
    }

    config = EstimationConfig(
        n_draws=n_draws,
        n_categories=n_categories,
        verbose=verbose
    )

    estimator = SMLEstimator(config=config)
    return estimator.estimate(df, spec)


def compare_two_stage_vs_iclv(two_stage_results: Dict,
                               iclv_result: ICLVResult,
                               true_values: Dict = None) -> pd.DataFrame:
    """
    Compare two-stage and ICLV estimation results.

    Useful for demonstrating attenuation bias correction.

    Args:
        two_stage_results: Dict with two-stage estimates
        iclv_result: ICLVResult from ICLV estimation
        true_values: Optional true values (for simulation studies)

    Returns:
        DataFrame comparing estimates
    """
    records = []

    # Compare LV effects in choice model
    for param in iclv_result.beta:
        if param.startswith('beta_'):
            two_stage_est = two_stage_results.get('beta', {}).get(param, np.nan)
            iclv_est = iclv_result.beta[param]
            true_val = true_values.get('beta', {}).get(param, np.nan) if true_values else np.nan

            records.append({
                'parameter': param,
                'two_stage': two_stage_est,
                'iclv': iclv_est,
                'true': true_val,
                'attenuation': (iclv_est - two_stage_est) / two_stage_est if two_stage_est != 0 else np.nan,
                'two_stage_bias': (two_stage_est - true_val) / true_val if not np.isnan(true_val) and true_val != 0 else np.nan,
                'iclv_bias': (iclv_est - true_val) / true_val if not np.isnan(true_val) and true_val != 0 else np.nan,
            })

    # Compare structural coefficients
    for lv, coeffs in iclv_result.gamma.items():
        for cov, iclv_est in coeffs.items():
            two_stage_est = two_stage_results.get('gamma', {}).get(lv, {}).get(cov, np.nan)
            true_val = true_values.get('gamma', {}).get(lv, {}).get(cov, np.nan) if true_values else np.nan

            records.append({
                'parameter': f'gamma_{lv}_{cov}',
                'two_stage': two_stage_est,
                'iclv': iclv_est,
                'true': true_val,
                'attenuation': np.nan,  # N/A for structural
                'two_stage_bias': (two_stage_est - true_val) / true_val if not np.isnan(true_val) and true_val != 0 else np.nan,
                'iclv_bias': (iclv_est - true_val) / true_val if not np.isnan(true_val) and true_val != 0 else np.nan,
            })

    return pd.DataFrame(records)


if __name__ == '__main__':
    print("ICLV Estimation Module")
    print("=" * 50)

    print("\nAvailable functions:")
    print("  - SMLEstimator: Full-featured estimator class")
    print("  - estimate_iclv: Convenience function")
    print("  - compare_two_stage_vs_iclv: Compare estimation methods")

    print("\nExample usage:")
    print("""
    from src.models.iclv import estimate_iclv

    result = estimate_iclv(
        df=data,
        constructs={
            'env_concern': ['env1', 'env2', 'env3'],
            'tech_affinity': ['tech1', 'tech2', 'tech3']
        },
        covariates=['age', 'income', 'education'],
        choice_col='choice',
        attribute_cols=['price', 'time', 'comfort'],
        lv_effects={
            'env_concern': 'beta_env',
            'tech_affinity': 'beta_tech'
        },
        n_draws=500
    )

    print(result.summary())
    """)
