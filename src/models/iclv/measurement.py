"""
Measurement Model for ICLV
==========================

Implements ordered probit measurement model for Likert-scale indicators.

Model:
    y*_k = λ_k * η + ε_k,  ε_k ~ N(0, 1)
    y_k = j  if  τ_{j-1} < y*_k ≤ τ_j

    P(y_k = j | η) = Φ(τ_j - λ_k*η) - Φ(τ_{j-1} - λ_k*η)

Where:
    - y_k = observed Likert response (1, 2, ..., J)
    - y*_k = latent continuous variable
    - λ_k = factor loading for item k
    - η = latent variable
    - τ_j = threshold parameters
    - Φ = standard normal CDF

Author: DCM Research Team
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MeasurementParams:
    """Parameters for measurement model."""
    loadings: Dict[str, float]  # Item -> loading (λ)
    thresholds: np.ndarray  # Threshold values (τ), shape (J-1,)
    n_categories: int = 5  # Number of Likert categories


class OrderedProbitMeasurement:
    """
    Ordered probit measurement model for Likert indicators.

    Computes probability of observing each response category
    given a latent variable value.

    Example:
        >>> measurement = OrderedProbitMeasurement(n_categories=5)
        >>> # For item with loading 0.8 and LV value 1.5
        >>> probs = measurement.response_probabilities(loading=0.8, lv_value=1.5)
        >>> probs  # Probabilities for categories 1-5
    """

    def __init__(self,
                 n_categories: int = 5,
                 thresholds: np.ndarray = None):
        """
        Initialize ordered probit measurement model.

        Args:
            n_categories: Number of response categories (e.g., 5 for 5-point Likert)
            thresholds: Threshold parameters (J-1 values). If None, uses default
                       equally-spaced thresholds.
        """
        self.n_categories = n_categories

        if thresholds is None:
            # Default: equally spaced thresholds assuming standard normal
            # For 5 categories: approximately [-1.5, -0.5, 0.5, 1.5]
            self.thresholds = self._default_thresholds(n_categories)
        else:
            if len(thresholds) != n_categories - 1:
                raise ValueError(f"Need {n_categories-1} thresholds, got {len(thresholds)}")
            self.thresholds = np.array(thresholds)

    def _default_thresholds(self, n_categories: int) -> np.ndarray:
        """Generate default equally-spaced thresholds."""
        # Use quantiles of standard normal
        probs = np.linspace(0, 1, n_categories + 1)[1:-1]
        return stats.norm.ppf(probs)

    def response_probabilities(self,
                               loading: float,
                               lv_value: float) -> np.ndarray:
        """
        Compute response probabilities for all categories.

        P(y = j | η) = Φ(τ_j - λ*η) - Φ(τ_{j-1} - λ*η)

        Args:
            loading: Factor loading (λ)
            lv_value: Latent variable value (η)

        Returns:
            Array of probabilities for each category, shape (n_categories,)
        """
        # Compute cumulative probabilities at each threshold
        # P(y ≤ j) = Φ(τ_j - λ*η)

        # Adjusted thresholds
        adj_thresholds = self.thresholds - loading * lv_value

        # Cumulative probabilities (with 0 and 1 at ends)
        cum_probs = np.zeros(self.n_categories + 1)
        cum_probs[0] = 0.0
        cum_probs[-1] = 1.0
        cum_probs[1:-1] = stats.norm.cdf(adj_thresholds)

        # Category probabilities: P(y=j) = P(y≤j) - P(y≤j-1)
        probs = np.diff(cum_probs)

        # Ensure numerical stability
        probs = np.clip(probs, 1e-10, 1.0)
        probs = probs / probs.sum()  # Renormalize

        return probs

    def log_probability(self,
                        response: int,
                        loading: float,
                        lv_value: float) -> float:
        """
        Compute log probability of a specific response.

        Args:
            response: Observed response (1 to n_categories)
            loading: Factor loading
            lv_value: Latent variable value

        Returns:
            Log probability of the response
        """
        probs = self.response_probabilities(loading, lv_value)
        return np.log(probs[response - 1])  # response is 1-indexed

    def log_probability_batch(self,
                              responses: np.ndarray,
                              loadings: np.ndarray,
                              lv_values: np.ndarray) -> np.ndarray:
        """
        Compute log probabilities for batch of observations.

        Args:
            responses: Observed responses, shape (n_items,) or (n_individuals, n_items)
            loadings: Factor loadings, shape (n_items,)
            lv_values: LV values, shape (n_individuals,) or scalar

        Returns:
            Log probabilities, shape (n_individuals,) or scalar
        """
        responses = np.atleast_2d(responses)
        lv_values = np.atleast_1d(lv_values)

        n_individuals = len(lv_values)
        n_items = responses.shape[-1]

        log_probs = np.zeros(n_individuals)

        for i in range(n_individuals):
            for k in range(n_items):
                resp = int(responses[i, k] if responses.ndim > 1 else responses[0, k])
                log_probs[i] += self.log_probability(resp, loadings[k], lv_values[i])

        return log_probs


class MeasurementLikelihood:
    """
    Full measurement model likelihood for multiple constructs.

    Handles multiple latent variables, each with multiple indicators,
    computing the joint probability of all observed responses.
    """

    def __init__(self,
                 constructs: Dict[str, List[str]],
                 loadings: Dict[str, float],
                 n_categories: int = 5,
                 thresholds: np.ndarray = None):
        """
        Initialize measurement likelihood.

        Args:
            constructs: Dict mapping construct name to list of item names
            loadings: Dict mapping item name to factor loading
            n_categories: Number of Likert categories
            thresholds: Threshold parameters (shared across items)
        """
        self.constructs = constructs
        self.loadings = loadings
        self.n_categories = n_categories

        self.measurement_model = OrderedProbitMeasurement(
            n_categories=n_categories,
            thresholds=thresholds
        )

        # Build construct-item mapping
        self.construct_items = {}
        for construct, items in constructs.items():
            self.construct_items[construct] = items

        self.n_constructs = len(constructs)
        self.construct_names = list(constructs.keys())

    def log_likelihood(self,
                       indicators: np.ndarray,
                       item_names: List[str],
                       lv_values: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute log-likelihood of indicator responses given LV values.

        P(I|η) = Π_c Π_k P(I_ck | η_c)

        Args:
            indicators: Observed responses, shape (n_individuals, n_items)
            item_names: List of item column names matching indicators
            lv_values: Dict mapping construct name to LV values,
                      each shape (n_individuals,) or (n_individuals, n_draws)

        Returns:
            Log-likelihood, shape (n_individuals,) or (n_individuals, n_draws)
        """
        n_individuals = indicators.shape[0]

        # Check if we're doing integration (multiple draws)
        first_lv = list(lv_values.values())[0]
        if first_lv.ndim == 2:
            n_draws = first_lv.shape[1]
            ll = np.zeros((n_individuals, n_draws))
        else:
            n_draws = None
            ll = np.zeros(n_individuals)

        # Build item -> construct mapping
        item_to_construct = {}
        for construct, items in self.constructs.items():
            for item in items:
                item_to_construct[item] = construct

        # Compute likelihood for each item
        for k, item_name in enumerate(item_names):
            if item_name not in item_to_construct:
                continue

            construct = item_to_construct[item_name]
            loading = self.loadings.get(item_name, 0.7)  # Default loading
            lv = lv_values[construct]

            responses = indicators[:, k]

            if n_draws is not None:
                # Multiple draws: compute for each draw
                for r in range(n_draws):
                    for i in range(n_individuals):
                        ll[i, r] += self.measurement_model.log_probability(
                            int(responses[i]),
                            loading,
                            lv[i, r]
                        )
            else:
                # Single LV value per individual
                for i in range(n_individuals):
                    ll[i] += self.measurement_model.log_probability(
                        int(responses[i]),
                        loading,
                        lv[i]
                    )

        return ll

    def log_likelihood_vectorized(self,
                                  indicators: np.ndarray,
                                  item_names: List[str],
                                  lv_draws: np.ndarray,
                                  construct_indices: Dict[str, int]) -> np.ndarray:
        """
        Vectorized log-likelihood computation for simulation.

        Args:
            indicators: Observed responses, shape (n_individuals, n_items)
            item_names: List of item column names
            lv_draws: LV draws, shape (n_individuals, n_draws, n_constructs)
            construct_indices: Dict mapping construct name to index in lv_draws

        Returns:
            Log-likelihood, shape (n_individuals, n_draws)
        """
        n_individuals, n_draws, _ = lv_draws.shape
        ll = np.zeros((n_individuals, n_draws))

        # Build item -> construct mapping
        item_to_construct = {}
        for construct, items in self.constructs.items():
            for item in items:
                item_to_construct[item] = construct

        for k, item_name in enumerate(item_names):
            if item_name not in item_to_construct:
                continue

            construct = item_to_construct[item_name]
            construct_idx = construct_indices[construct]
            loading = self.loadings.get(item_name, 0.7)

            responses = indicators[:, k]  # (n_individuals,)
            lv = lv_draws[:, :, construct_idx]  # (n_individuals, n_draws)

            # Compute for all individuals and draws
            for i in range(n_individuals):
                resp = int(responses[i])
                for r in range(n_draws):
                    ll[i, r] += self.measurement_model.log_probability(
                        resp, loading, lv[i, r]
                    )

        return ll


def estimate_thresholds(responses: np.ndarray,
                        n_categories: int = 5) -> np.ndarray:
    """
    Estimate threshold parameters from data using marginal frequencies.

    Thresholds are set at quantiles matching observed category frequencies.

    Args:
        responses: Observed responses (1 to n_categories)
        n_categories: Number of categories

    Returns:
        Estimated thresholds, shape (n_categories-1,)
    """
    # Compute cumulative proportions
    counts = np.bincount(responses.astype(int), minlength=n_categories + 1)[1:]
    proportions = counts / counts.sum()
    cum_props = np.cumsum(proportions)[:-1]

    # Convert to thresholds via probit
    cum_props = np.clip(cum_props, 0.001, 0.999)
    thresholds = stats.norm.ppf(cum_props)

    return thresholds


if __name__ == '__main__':
    print("ICLV Measurement Model")
    print("=" * 40)

    # Demo
    measurement = OrderedProbitMeasurement(n_categories=5)

    print("\nResponse probabilities for loading=0.8, LV=0:")
    probs = measurement.response_probabilities(loading=0.8, lv_value=0)
    for j, p in enumerate(probs, 1):
        print(f"  P(y={j}) = {p:.4f}")

    print("\nResponse probabilities for loading=0.8, LV=1.5:")
    probs = measurement.response_probabilities(loading=0.8, lv_value=1.5)
    for j, p in enumerate(probs, 1):
        print(f"  P(y={j}) = {p:.4f}")
