"""
Structural Model for ICLV
==========================

Models the relationship between observed covariates and latent variables.

Structural Equation:
    η = Γ * X + ζ,  ζ ~ N(0, Ψ)

Where:
    - η = vector of latent variables
    - X = vector of observed covariates (demographics, etc.)
    - Γ = matrix of structural coefficients
    - ζ = structural error term
    - Ψ = covariance matrix of structural errors

For identification, typically Ψ = I (identity matrix) for the base model.

Author: DCM Research Team
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class StructuralParams:
    """Parameters for structural model."""
    gamma: Dict[str, Dict[str, float]]  # LV -> {covariate: coefficient}
    psi: np.ndarray  # Error covariance matrix
    construct_names: List[str]


class StructuralModel:
    """
    Structural model relating covariates to latent variables.

    Models how demographic and other observed variables predict
    the latent constructs (e.g., environmental concern, tech affinity).

    Example:
        >>> structural = StructuralModel(
        ...     constructs=['env_concern', 'tech_affinity'],
        ...     covariates=['age', 'income', 'education']
        ... )
        >>> lv_mean = structural.predict_lv_mean({'age': 0.5, 'income': -0.2})
    """

    def __init__(self,
                 constructs: List[str],
                 covariates: List[str],
                 gamma: Dict[str, Dict[str, float]] = None,
                 psi: np.ndarray = None):
        """
        Initialize structural model.

        Args:
            constructs: List of latent variable names
            covariates: List of covariate names
            gamma: Structural coefficients {lv: {covariate: coefficient}}
            psi: Error covariance matrix (default: identity)
        """
        self.constructs = constructs
        self.covariates = covariates
        self.n_constructs = len(constructs)
        self.n_covariates = len(covariates)

        # Initialize gamma matrix
        if gamma is None:
            self.gamma = {lv: {cov: 0.0 for cov in covariates}
                         for lv in constructs}
        else:
            self.gamma = gamma

        # Initialize error covariance (identity by default for identification)
        if psi is None:
            self.psi = np.eye(self.n_constructs)
        else:
            self.psi = np.array(psi)

        # Build index mappings
        self.construct_idx = {name: i for i, name in enumerate(constructs)}
        self.covariate_idx = {name: i for i, name in enumerate(covariates)}

    def get_gamma_matrix(self) -> np.ndarray:
        """
        Get structural coefficients as matrix.

        Returns:
            Array of shape (n_constructs, n_covariates)
        """
        gamma_matrix = np.zeros((self.n_constructs, self.n_covariates))

        for lv, coeffs in self.gamma.items():
            lv_idx = self.construct_idx[lv]
            for cov, val in coeffs.items():
                if cov in self.covariate_idx:
                    cov_idx = self.covariate_idx[cov]
                    gamma_matrix[lv_idx, cov_idx] = val

        return gamma_matrix

    def set_gamma_matrix(self, gamma_matrix: np.ndarray):
        """Set structural coefficients from matrix."""
        for lv_idx, lv in enumerate(self.constructs):
            for cov_idx, cov in enumerate(self.covariates):
                self.gamma[lv][cov] = gamma_matrix[lv_idx, cov_idx]

    def predict_lv_mean(self,
                        covariate_values: Dict[str, float]) -> np.ndarray:
        """
        Predict latent variable conditional mean given covariates.

        E[η|X] = Γ * X

        Args:
            covariate_values: Dict mapping covariate name to value

        Returns:
            Array of predicted LV means, shape (n_constructs,)
        """
        x = np.array([covariate_values.get(cov, 0.0) for cov in self.covariates])
        gamma_matrix = self.get_gamma_matrix()
        return gamma_matrix @ x

    def predict_lv_mean_batch(self,
                              covariate_data: np.ndarray,
                              covariate_names: List[str]) -> np.ndarray:
        """
        Predict LV means for batch of individuals.

        Args:
            covariate_data: Array of shape (n_individuals, n_covariates)
            covariate_names: List of covariate column names

        Returns:
            Array of shape (n_individuals, n_constructs)
        """
        n_individuals = covariate_data.shape[0]

        # Build covariate matrix with proper ordering
        X = np.zeros((n_individuals, self.n_covariates))
        for i, cov in enumerate(covariate_names):
            if cov in self.covariate_idx:
                cov_idx = self.covariate_idx[cov]
                X[:, cov_idx] = covariate_data[:, i]

        gamma_matrix = self.get_gamma_matrix()
        return X @ gamma_matrix.T  # (n_individuals, n_constructs)

    def generate_lv_draws(self,
                          covariate_values: Dict[str, float],
                          base_draws: np.ndarray) -> np.ndarray:
        """
        Generate LV draws given covariates and standard normal draws.

        η = Γ*X + L*ν, where Ψ = L*L' (Cholesky decomposition)

        Args:
            covariate_values: Dict of covariate values for one individual
            base_draws: Standard normal draws, shape (n_draws, n_constructs)

        Returns:
            LV draws, shape (n_draws, n_constructs)
        """
        mean = self.predict_lv_mean(covariate_values)

        # Cholesky decomposition of error covariance
        L = np.linalg.cholesky(self.psi)

        # Transform draws
        lv_draws = mean + base_draws @ L.T

        return lv_draws

    def generate_lv_draws_batch(self,
                                covariate_data: np.ndarray,
                                covariate_names: List[str],
                                base_draws: np.ndarray) -> np.ndarray:
        """
        Generate LV draws for batch of individuals.

        Args:
            covariate_data: Array of shape (n_individuals, n_covariates)
            covariate_names: List of covariate column names
            base_draws: Standard normal draws, shape (n_individuals, n_draws, n_constructs)

        Returns:
            LV draws, shape (n_individuals, n_draws, n_constructs)
        """
        n_individuals, n_draws, n_constructs = base_draws.shape

        # Predicted means
        means = self.predict_lv_mean_batch(covariate_data, covariate_names)
        # Shape: (n_individuals, n_constructs)

        # Cholesky decomposition
        L = np.linalg.cholesky(self.psi)

        # Transform draws for each individual
        lv_draws = np.zeros((n_individuals, n_draws, n_constructs))

        for i in range(n_individuals):
            # Transform standard normal to correlated
            transformed = base_draws[i] @ L.T  # (n_draws, n_constructs)
            lv_draws[i] = means[i] + transformed

        return lv_draws

    def log_density(self,
                    lv_values: np.ndarray,
                    covariate_values: Dict[str, float]) -> float:
        """
        Compute log density of LV values given covariates.

        log f(η|X) = log N(η; Γ*X, Ψ)

        Args:
            lv_values: LV values, shape (n_constructs,)
            covariate_values: Dict of covariate values

        Returns:
            Log density value
        """
        mean = self.predict_lv_mean(covariate_values)
        return stats.multivariate_normal.logpdf(lv_values, mean=mean, cov=self.psi)

    def log_density_batch(self,
                          lv_values: np.ndarray,
                          covariate_data: np.ndarray,
                          covariate_names: List[str]) -> np.ndarray:
        """
        Compute log densities for batch of individuals.

        Args:
            lv_values: LV values, shape (n_individuals, n_constructs)
            covariate_data: Array of shape (n_individuals, n_covariates)
            covariate_names: List of covariate column names

        Returns:
            Array of log densities, shape (n_individuals,)
        """
        means = self.predict_lv_mean_batch(covariate_data, covariate_names)

        n_individuals = lv_values.shape[0]
        log_densities = np.zeros(n_individuals)

        for i in range(n_individuals):
            log_densities[i] = stats.multivariate_normal.logpdf(
                lv_values[i], mean=means[i], cov=self.psi
            )

        return log_densities


class StructuralModelBuilder:
    """
    Helper class to build structural model from specification.

    Handles conversion from different specification formats.
    """

    @staticmethod
    def from_dataframe_columns(df_columns: List[str],
                               construct_patterns: Dict[str, str]) -> 'StructuralModel':
        """
        Infer covariates from dataframe columns.

        Args:
            df_columns: List of column names in dataframe
            construct_patterns: Dict mapping construct name to column pattern

        Returns:
            Configured StructuralModel
        """
        constructs = list(construct_patterns.keys())

        # Identify potential covariates (exclude indicators and choice vars)
        exclude_patterns = ['_lv', '_ind', 'choice', 'alt_', 'av_']
        covariates = [col for col in df_columns
                     if not any(pat in col.lower() for pat in exclude_patterns)]

        return StructuralModel(constructs=constructs, covariates=covariates)

    @staticmethod
    def from_biogeme_specification(spec: Dict) -> 'StructuralModel':
        """
        Build structural model from Biogeme-style specification.

        Args:
            spec: Specification dict with 'constructs', 'covariates', 'gamma'

        Returns:
            Configured StructuralModel
        """
        return StructuralModel(
            constructs=spec['constructs'],
            covariates=spec['covariates'],
            gamma=spec.get('gamma'),
            psi=spec.get('psi')
        )


if __name__ == '__main__':
    print("ICLV Structural Model")
    print("=" * 40)

    # Demo
    structural = StructuralModel(
        constructs=['env_concern', 'tech_affinity'],
        covariates=['age', 'income', 'education']
    )

    # Set some coefficients
    structural.gamma['env_concern']['age'] = 0.3
    structural.gamma['env_concern']['education'] = 0.4
    structural.gamma['tech_affinity']['age'] = -0.2
    structural.gamma['tech_affinity']['income'] = 0.25

    # Predict LV means
    covs = {'age': 0.5, 'income': 1.0, 'education': 0.5}
    means = structural.predict_lv_mean(covs)

    print(f"\nCovariate values: {covs}")
    print(f"Predicted LV means:")
    for lv, mean in zip(structural.constructs, means):
        print(f"  {lv}: {mean:.4f}")

    # Generate draws
    base_draws = np.random.randn(100, 2)
    lv_draws = structural.generate_lv_draws(covs, base_draws)
    print(f"\nGenerated 100 LV draws:")
    print(f"  env_concern: mean={lv_draws[:, 0].mean():.4f}, std={lv_draws[:, 0].std():.4f}")
    print(f"  tech_affinity: mean={lv_draws[:, 1].mean():.4f}, std={lv_draws[:, 1].std():.4f}")
