"""
Monte Carlo Integration for ICLV
================================

Provides quasi-random number generation and integration methods
for simulated maximum likelihood estimation.

Uses Halton sequences for better coverage than pseudo-random draws,
reducing simulation variance for the same number of draws.

Author: DCM Research Team
"""

import numpy as np
from scipy import stats
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class DrawsResult:
    """Container for simulation draws."""
    draws: np.ndarray  # Shape: (n_individuals, n_draws, n_dimensions)
    n_individuals: int
    n_draws: int
    n_dimensions: int
    draw_type: str


class HaltonDraws:
    """
    Halton sequence generator for quasi-Monte Carlo integration.

    Halton sequences provide more uniform coverage of the unit hypercube
    than pseudo-random numbers, reducing variance in SML estimation.

    Example:
        >>> generator = HaltonDraws(n_draws=500, n_dimensions=4)
        >>> draws = generator.generate(n_individuals=1000)
        >>> draws.shape  # (1000, 500, 4)
    """

    # First 20 primes for Halton bases
    PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
              31, 37, 41, 43, 47, 53, 59, 61, 67, 71]

    def __init__(self,
                 n_draws: int = 500,
                 n_dimensions: int = 1,
                 scramble: bool = True,
                 seed: int = 42):
        """
        Initialize Halton sequence generator.

        Args:
            n_draws: Number of draws per individual
            n_dimensions: Number of latent variable dimensions
            scramble: Whether to apply randomized scrambling
            seed: Random seed for scrambling
        """
        self.n_draws = n_draws
        self.n_dimensions = n_dimensions
        self.scramble = scramble
        self.seed = seed

        if n_dimensions > len(self.PRIMES):
            raise ValueError(f"Maximum {len(self.PRIMES)} dimensions supported")

        self.bases = self.PRIMES[:n_dimensions]

    def _halton_sequence(self, n: int, base: int) -> np.ndarray:
        """
        Generate Halton sequence for a single dimension.

        Args:
            n: Number of points
            base: Prime base for the sequence

        Returns:
            Array of n Halton sequence values in (0, 1)
        """
        sequence = np.zeros(n)

        for i in range(n):
            f = 1.0
            r = 0.0
            index = i + 1  # Start from 1, not 0

            while index > 0:
                f = f / base
                r = r + f * (index % base)
                index = index // base

            sequence[i] = r

        return sequence

    def _scramble_sequence(self, sequence: np.ndarray,
                           rng: np.random.Generator) -> np.ndarray:
        """Apply random scrambling to reduce correlation."""
        shift = rng.uniform(0, 1)
        return (sequence + shift) % 1.0

    def generate_uniform(self, n_individuals: int) -> np.ndarray:
        """
        Generate uniform Halton draws in (0, 1).

        Args:
            n_individuals: Number of individuals

        Returns:
            Array of shape (n_individuals, n_draws, n_dimensions)
        """
        rng = np.random.default_rng(self.seed)

        # Total draws needed
        total_draws = n_individuals * self.n_draws

        # Generate base Halton sequence for each dimension
        draws = np.zeros((total_draws, self.n_dimensions))

        for d, base in enumerate(self.bases):
            seq = self._halton_sequence(total_draws, base)
            if self.scramble:
                seq = self._scramble_sequence(seq, rng)
            draws[:, d] = seq

        # Reshape: (n_individuals, n_draws, n_dimensions)
        draws = draws.reshape(n_individuals, self.n_draws, self.n_dimensions)

        # Shuffle draws across individuals to reduce correlation
        if self.scramble:
            for i in range(n_individuals):
                rng.shuffle(draws[i])

        return draws

    def generate(self, n_individuals: int) -> DrawsResult:
        """
        Generate standard normal Halton draws.

        Transforms uniform Halton draws to standard normal using
        inverse CDF (probit) transformation.

        Args:
            n_individuals: Number of individuals

        Returns:
            DrawsResult with normal draws
        """
        uniform_draws = self.generate_uniform(n_individuals)

        # Clip to avoid infinite values at 0 and 1
        uniform_draws = np.clip(uniform_draws, 1e-6, 1 - 1e-6)

        # Transform to standard normal
        normal_draws = stats.norm.ppf(uniform_draws)

        return DrawsResult(
            draws=normal_draws,
            n_individuals=n_individuals,
            n_draws=self.n_draws,
            n_dimensions=self.n_dimensions,
            draw_type='halton'
        )


class MonteCarloIntegrator:
    """
    Monte Carlo integration for ICLV likelihood.

    Approximates the integral over the latent variable distribution:
        L_n = ∫ P(y|η) P(I|η) f(η) dη ≈ (1/R) Σ_r P(y|η_r) P(I|η_r)

    where η_r are draws from the prior distribution f(η).
    """

    def __init__(self,
                 n_draws: int = 500,
                 draw_type: str = 'halton',
                 seed: int = 42):
        """
        Initialize Monte Carlo integrator.

        Args:
            n_draws: Number of simulation draws
            draw_type: 'halton' (recommended) or 'random'
            seed: Random seed
        """
        self.n_draws = n_draws
        self.draw_type = draw_type
        self.seed = seed

    def generate_draws(self,
                       n_individuals: int,
                       n_dimensions: int) -> DrawsResult:
        """
        Generate simulation draws.

        Args:
            n_individuals: Number of individuals
            n_dimensions: Number of LV dimensions

        Returns:
            DrawsResult with draws
        """
        if self.draw_type == 'halton':
            generator = HaltonDraws(
                n_draws=self.n_draws,
                n_dimensions=n_dimensions,
                seed=self.seed
            )
            return generator.generate(n_individuals)
        else:
            # Pseudo-random draws
            rng = np.random.default_rng(self.seed)
            draws = rng.standard_normal((n_individuals, self.n_draws, n_dimensions))
            return DrawsResult(
                draws=draws,
                n_individuals=n_individuals,
                n_draws=self.n_draws,
                n_dimensions=n_dimensions,
                draw_type='random'
            )

    def integrate(self,
                  log_likelihood_func,
                  draws: np.ndarray) -> np.ndarray:
        """
        Compute Monte Carlo integral of likelihood.

        Uses log-sum-exp trick for numerical stability:
            log(L_n) = log((1/R) Σ_r exp(LL_nr))
                     = -log(R) + log(Σ_r exp(LL_nr))
                     = -log(R) + max(LL_nr) + log(Σ_r exp(LL_nr - max(LL_nr)))

        Args:
            log_likelihood_func: Function(draws) -> log_likelihood per draw
                                Shape: (n_individuals, n_draws)
            draws: Simulation draws, shape (n_individuals, n_draws, n_dimensions)

        Returns:
            Array of integrated log-likelihoods, shape (n_individuals,)
        """
        # Compute log-likelihood for each draw
        # ll shape: (n_individuals, n_draws)
        ll = log_likelihood_func(draws)

        n_individuals, n_draws = ll.shape

        # Log-sum-exp trick for numerical stability
        max_ll = ll.max(axis=1, keepdims=True)
        log_sum_exp = max_ll.squeeze() + np.log(np.exp(ll - max_ll).sum(axis=1))

        # Monte Carlo average: log(1/R * sum) = log(sum) - log(R)
        integrated_ll = log_sum_exp - np.log(n_draws)

        return integrated_ll

    def integrate_with_gradient(self,
                                log_likelihood_func,
                                gradient_func,
                                draws: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute integral and gradient for optimization.

        The gradient of the log-likelihood is:
            ∂log(L)/∂θ = (1/L) * ∂L/∂θ
                       = (1/L) * (1/R) Σ_r L_r * ∂log(L_r)/∂θ
                       = Σ_r w_r * ∂log(L_r)/∂θ

        where w_r = L_r / Σ_s L_s (importance weights)

        Args:
            log_likelihood_func: Function(draws) -> log_likelihood per draw
            gradient_func: Function(draws) -> gradient per draw
            draws: Simulation draws

        Returns:
            Tuple of (integrated_ll, integrated_gradient)
        """
        ll = log_likelihood_func(draws)
        grad = gradient_func(draws)  # Shape: (n_individuals, n_draws, n_params)

        n_individuals, n_draws = ll.shape

        # Compute importance weights (softmax of log-likelihoods)
        max_ll = ll.max(axis=1, keepdims=True)
        exp_ll = np.exp(ll - max_ll)
        weights = exp_ll / exp_ll.sum(axis=1, keepdims=True)  # (n_individuals, n_draws)

        # Integrated log-likelihood
        log_sum_exp = max_ll.squeeze() + np.log(exp_ll.sum(axis=1))
        integrated_ll = log_sum_exp - np.log(n_draws)

        # Integrated gradient: weighted average
        # weights: (n_individuals, n_draws) -> (n_individuals, n_draws, 1)
        weights_expanded = weights[:, :, np.newaxis]
        integrated_grad = (weights_expanded * grad).sum(axis=1)  # (n_individuals, n_params)

        return integrated_ll, integrated_grad


def create_antithetic_draws(draws: np.ndarray) -> np.ndarray:
    """
    Create antithetic draws for variance reduction.

    For each draw η, also use -η. This reduces variance when
    the integrand has some symmetry.

    Args:
        draws: Original draws, shape (n_individuals, n_draws, n_dimensions)

    Returns:
        Extended draws including antithetics, shape (n_individuals, 2*n_draws, n_dimensions)
    """
    antithetic = -draws
    return np.concatenate([draws, antithetic], axis=1)


def generate_halton_draws(
    n_draws_or_individuals: int,
    n_dim_or_draws: int = None,
    n_dimensions: int = None,
    seed: int = 42,
    return_uniform: bool = None
) -> np.ndarray:
    """
    Generate Halton draws for simulation-based estimation.

    This function supports two interfaces:

    1. Simple interface (for basic tests):
       generate_halton_draws(n_draws, n_dimensions)
       Returns: shape (n_draws, n_dimensions) uniform draws in (0,1)

    2. Full interface (for ICLV estimation):
       generate_halton_draws(n_individuals, n_draws=500, n_dimensions=1)
       Returns: shape (n_individuals, n_draws, n_dimensions) standard normal draws

    Args:
        n_draws_or_individuals: Number of draws (simple) or individuals (full)
        n_dim_or_draws: Number of dimensions (simple) or draws per individual (full)
        n_dimensions: Number of dimensions (full interface only)
        seed: Random seed for scrambling (default 42)
        return_uniform: If True, return uniform (0,1); None uses auto-detection

    Returns:
        np.ndarray of quasi-random draws

    Examples:
        # Simple interface (for compatibility with existing tests)
        >>> draws = generate_halton_draws(100, 2)
        >>> draws.shape
        (100, 2)

        # Full interface (for ICLV estimation)
        >>> draws = generate_halton_draws(1000, n_draws=200, n_dimensions=2)
        >>> draws.shape
        (1000, 200, 2)
    """
    # Detect which interface is being used
    if n_dimensions is None and isinstance(n_dim_or_draws, int) and n_dim_or_draws <= 20:
        # Simple interface: generate_halton_draws(n_draws, n_dim)
        # Returns 2D array of uniform draws in (0,1)
        n_draws = n_draws_or_individuals
        n_dim = n_dim_or_draws

        generator = HaltonDraws(
            n_draws=n_draws,
            n_dimensions=n_dim,
            scramble=True,
            seed=seed
        )

        # Return uniform draws reshaped to (n_draws, n_dim)
        uniform = generator.generate_uniform(1)  # Shape: (1, n_draws, n_dim)
        return uniform[0]  # Shape: (n_draws, n_dim)

    else:
        # Full interface: generate_halton_draws(n_individuals, n_draws, n_dimensions)
        n_individuals = n_draws_or_individuals
        n_draws = n_dim_or_draws if n_dim_or_draws is not None else 500
        n_dim = n_dimensions if n_dimensions is not None else 1

        generator = HaltonDraws(
            n_draws=n_draws,
            n_dimensions=n_dim,
            scramble=True,
            seed=seed
        )

        if return_uniform:
            return generator.generate_uniform(n_individuals)
        else:
            result = generator.generate(n_individuals)
            return result.draws


if __name__ == '__main__':
    print("ICLV Integration Module")
    print("=" * 40)

    # Demo
    generator = HaltonDraws(n_draws=100, n_dimensions=2)
    result = generator.generate(n_individuals=10)

    print(f"\nGenerated {result.n_individuals} x {result.n_draws} x {result.n_dimensions} draws")
    print(f"Draw type: {result.draw_type}")
    print(f"Mean: {result.draws.mean():.4f} (should be ~0)")
    print(f"Std: {result.draws.std():.4f} (should be ~1)")
