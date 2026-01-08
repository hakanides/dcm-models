"""
ICLV Core Model
===============

Integrated Choice and Latent Variable (ICLV) model for simultaneous
estimation of measurement and choice models.

This eliminates attenuation bias from two-stage estimation by
integrating over the latent variable distribution.

Mathematical Formulation:
    L_n = (1/R) Σ_r [ P(y_n|η_r) × Π_k P(I_nk|η_r) ]

Where:
    - y_n = choice outcome for individual n
    - I_nk = Likert indicator k for individual n
    - η_r = draw r from structural equation: η = Γ*X + ζ
    - R = number of simulation draws

Author: DCM Research Team
"""

import numpy as np
from scipy import stats, optimize
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
import pandas as pd

from .measurement import OrderedProbitMeasurement, MeasurementLikelihood
from .structural import StructuralModel
from .integration import HaltonDraws, MonteCarloIntegrator, DrawsResult


@dataclass
class ICLVResult:
    """Container for ICLV estimation results."""
    # Parameter estimates
    beta: Dict[str, float]  # Choice model coefficients
    gamma: Dict[str, Dict[str, float]]  # Structural coefficients
    loadings: Dict[str, float]  # Factor loadings
    thresholds: np.ndarray  # Measurement thresholds

    # Standard errors
    beta_se: Dict[str, float] = field(default_factory=dict)
    gamma_se: Dict[str, Dict[str, float]] = field(default_factory=dict)
    loadings_se: Dict[str, float] = field(default_factory=dict)
    thresholds_se: np.ndarray = None

    # Fit statistics
    log_likelihood: float = 0.0
    n_parameters: int = 0
    n_observations: int = 0
    aic: float = 0.0
    bic: float = 0.0
    convergence: bool = False
    n_iterations: int = 0
    hessian: np.ndarray = None

    # Model info
    n_draws: int = 500
    draw_type: str = 'halton'

    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "=" * 60,
            "ICLV Model Results",
            "=" * 60,
            f"Log-likelihood: {self.log_likelihood:.4f}",
            f"AIC: {self.aic:.4f}",
            f"BIC: {self.bic:.4f}",
            f"N observations: {self.n_observations}",
            f"N parameters: {self.n_parameters}",
            f"N draws: {self.n_draws} ({self.draw_type})",
            f"Converged: {self.convergence}",
            "",
            "Choice Model Coefficients (β):",
            "-" * 40,
        ]

        for param, value in self.beta.items():
            se = self.beta_se.get(param, np.nan)
            t_stat = value / se if se > 0 else np.nan
            lines.append(f"  {param:20s}: {value:8.4f} (SE: {se:.4f}, t: {t_stat:.2f})")

        lines.extend([
            "",
            "Structural Coefficients (Γ):",
            "-" * 40,
        ])

        for lv, coeffs in self.gamma.items():
            lines.append(f"  {lv}:")
            for cov, value in coeffs.items():
                se = self.gamma_se.get(lv, {}).get(cov, np.nan)
                t_stat = value / se if se > 0 else np.nan
                lines.append(f"    {cov:18s}: {value:8.4f} (SE: {se:.4f}, t: {t_stat:.2f})")

        lines.extend([
            "",
            "Factor Loadings (λ):",
            "-" * 40,
        ])

        for item, value in self.loadings.items():
            se = self.loadings_se.get(item, np.nan)
            t_stat = value / se if se > 0 else np.nan
            lines.append(f"  {item:20s}: {value:8.4f} (SE: {se:.4f}, t: {t_stat:.2f})")

        lines.append("=" * 60)

        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame format."""
        records = []

        # Choice coefficients
        for param, value in self.beta.items():
            records.append({
                'type': 'choice',
                'construct': '',
                'parameter': param,
                'estimate': value,
                'se': self.beta_se.get(param, np.nan),
                't_stat': value / self.beta_se.get(param, 1e10)
            })

        # Structural coefficients
        for lv, coeffs in self.gamma.items():
            for cov, value in coeffs.items():
                se = self.gamma_se.get(lv, {}).get(cov, np.nan)
                records.append({
                    'type': 'structural',
                    'construct': lv,
                    'parameter': cov,
                    'estimate': value,
                    'se': se,
                    't_stat': value / se if se > 0 else np.nan
                })

        # Factor loadings
        for item, value in self.loadings.items():
            se = self.loadings_se.get(item, np.nan)
            records.append({
                'type': 'measurement',
                'construct': '',
                'parameter': item,
                'estimate': value,
                'se': se,
                't_stat': value / se if se > 0 else np.nan
            })

        return pd.DataFrame(records)


class ICLVModel:
    """
    Integrated Choice and Latent Variable Model.

    Simultaneously estimates:
    1. Structural model: Demographics → Latent Variables
    2. Measurement model: Latent Variables → Indicators
    3. Choice model: Attributes + Latent Variables → Choice

    Uses Simulated Maximum Likelihood (SML) with Halton draws
    for integration over the latent variable distribution.

    Example:
        >>> model = ICLVModel(
        ...     constructs={'env': ['env1', 'env2', 'env3']},
        ...     covariates=['age', 'income'],
        ...     n_draws=500
        ... )
        >>> result = model.estimate(df, choice_col='choice', ...)
    """

    def __init__(self,
                 constructs: Dict[str, List[str]],
                 covariates: List[str],
                 n_draws: int = 500,
                 draw_type: str = 'halton',
                 n_categories: int = 5,
                 seed: int = 42):
        """
        Initialize ICLV model.

        Args:
            constructs: Dict mapping construct name to list of indicator items
            covariates: List of covariate names for structural model
            n_draws: Number of simulation draws (500-1000 recommended)
            draw_type: 'halton' (recommended) or 'random'
            n_categories: Number of Likert scale categories
            seed: Random seed for reproducibility
        """
        self.constructs = constructs
        self.covariates = covariates
        self.n_draws = n_draws
        self.draw_type = draw_type
        self.n_categories = n_categories
        self.seed = seed

        self.construct_names = list(constructs.keys())
        self.n_constructs = len(self.construct_names)

        # Build construct index mapping
        self.construct_idx = {name: i for i, name in enumerate(self.construct_names)}

        # Initialize sub-models (will be fully configured during estimation)
        self.structural = StructuralModel(
            constructs=self.construct_names,
            covariates=covariates
        )

        # Build flat list of all indicator items
        self.all_items = []
        for items in constructs.values():
            self.all_items.extend(items)

        # Default loadings
        self.loadings = {item: 0.7 for item in self.all_items}

        # Measurement model
        self.measurement = MeasurementLikelihood(
            constructs=constructs,
            loadings=self.loadings,
            n_categories=n_categories
        )

        # Monte Carlo integrator
        self.integrator = MonteCarloIntegrator(
            n_draws=n_draws,
            draw_type=draw_type,
            seed=seed
        )

        # Store draws (generated once for consistency)
        self._draws_cache = None
        self._n_individuals_cache = None

    def _get_draws(self, n_individuals: int) -> DrawsResult:
        """Get or generate simulation draws."""
        if (self._draws_cache is None or
            self._n_individuals_cache != n_individuals):
            self._draws_cache = self.integrator.generate_draws(
                n_individuals=n_individuals,
                n_dimensions=self.n_constructs
            )
            self._n_individuals_cache = n_individuals
        return self._draws_cache

    def _pack_parameters(self,
                         beta: Dict[str, float],
                         gamma: Dict[str, Dict[str, float]],
                         loadings: Dict[str, float],
                         thresholds: np.ndarray) -> np.ndarray:
        """Pack all parameters into single vector for optimization."""
        params = []

        # Choice coefficients
        for key in sorted(beta.keys()):
            params.append(beta[key])

        # Structural coefficients
        for lv in self.construct_names:
            for cov in self.covariates:
                params.append(gamma[lv].get(cov, 0.0))

        # Factor loadings (skip first per construct for identification)
        for construct, items in self.constructs.items():
            for item in items[1:]:  # First loading fixed to 1 for identification
                params.append(loadings[item])

        # Thresholds
        params.extend(thresholds.tolist())

        return np.array(params)

    def _unpack_parameters(self,
                           params: np.ndarray,
                           beta_keys: List[str],
                           free_loading_items: List[str]) -> Tuple:
        """Unpack parameter vector back to structured form."""
        idx = 0

        # Choice coefficients
        beta = {}
        for key in beta_keys:
            beta[key] = params[idx]
            idx += 1

        # Structural coefficients
        gamma = {lv: {} for lv in self.construct_names}
        for lv in self.construct_names:
            for cov in self.covariates:
                gamma[lv][cov] = params[idx]
                idx += 1

        # Factor loadings
        loadings = {}
        # First loading per construct fixed to 1
        for construct, items in self.constructs.items():
            loadings[items[0]] = 1.0

        for item in free_loading_items:
            loadings[item] = params[idx]
            idx += 1

        # Thresholds
        n_thresholds = self.n_categories - 1
        thresholds = params[idx:idx + n_thresholds]

        return beta, gamma, loadings, thresholds

    def compute_choice_utility(self,
                               attributes: np.ndarray,
                               beta: Dict[str, float],
                               attribute_names: List[str],
                               lv_values: np.ndarray,
                               lv_beta: Dict[str, float]) -> np.ndarray:
        """
        Compute choice utility for all alternatives.

        V = Σ β_k * X_k + Σ β_lv * η_lv

        Args:
            attributes: Alternative attributes, shape (n_individuals, n_alts, n_attrs)
            beta: Choice coefficients for attributes
            attribute_names: Names of attribute columns
            lv_values: LV values, shape (n_individuals, n_draws, n_constructs) or
                      (n_individuals, n_constructs)
            lv_beta: Coefficients for LV effects in utility

        Returns:
            Utilities, shape (n_individuals, n_draws, n_alts) or
                      (n_individuals, n_alts)
        """
        n_individuals = attributes.shape[0]
        n_alts = attributes.shape[1]

        # Attribute contribution
        utility = np.zeros((n_individuals, n_alts))
        for i, attr_name in enumerate(attribute_names):
            if attr_name in beta:
                utility += beta[attr_name] * attributes[:, :, i]

        # If multiple draws, expand utility
        if lv_values.ndim == 3:
            n_draws = lv_values.shape[1]
            utility = np.repeat(utility[:, np.newaxis, :], n_draws, axis=1)

            # LV contribution
            for lv_name, coef in lv_beta.items():
                if lv_name in self.construct_idx:
                    lv_idx = self.construct_idx[lv_name]
                    # Add LV effect (same for all alternatives, or specific)
                    utility += coef * lv_values[:, :, lv_idx:lv_idx+1]
        else:
            # Single LV value per individual
            for lv_name, coef in lv_beta.items():
                if lv_name in self.construct_idx:
                    lv_idx = self.construct_idx[lv_name]
                    utility += coef * lv_values[:, lv_idx:lv_idx+1]

        return utility

    def compute_choice_probability(self,
                                   utility: np.ndarray,
                                   availability: np.ndarray = None) -> np.ndarray:
        """
        Compute multinomial logit choice probabilities.

        P(alt j) = exp(V_j) / Σ_k exp(V_k)

        Args:
            utility: Utilities, shape (..., n_alts)
            availability: Availability indicator, shape (n_individuals, n_alts)

        Returns:
            Choice probabilities, same shape as utility
        """
        if availability is not None:
            # Mask unavailable alternatives
            utility = np.where(
                availability[..., np.newaxis, :] if utility.ndim == 3 else availability,
                utility,
                -np.inf
            )

        # Softmax with numerical stability
        max_util = utility.max(axis=-1, keepdims=True)
        exp_util = np.exp(utility - max_util)
        probs = exp_util / exp_util.sum(axis=-1, keepdims=True)

        return probs

    def log_likelihood_individual(self,
                                  params: np.ndarray,
                                  data: Dict) -> np.ndarray:
        """
        Compute integrated log-likelihood for each individual.

        L_n = (1/R) Σ_r [ P(y_n|η_r) × Π_k P(I_nk|η_r) ]

        Args:
            params: Parameter vector
            data: Dict containing all model data

        Returns:
            Array of log-likelihoods, shape (n_individuals,)
        """
        # Unpack parameters
        beta, gamma, loadings, thresholds = self._unpack_parameters(
            params,
            data['beta_keys'],
            data['free_loading_items']
        )

        # Update structural model
        self.structural.gamma = gamma

        # Get base draws
        draws_result = self._get_draws(data['n_individuals'])
        base_draws = draws_result.draws  # (n_individuals, n_draws, n_constructs)

        # Generate LV draws from structural model
        lv_draws = self.structural.generate_lv_draws_batch(
            data['covariates'],
            data['covariate_names'],
            base_draws
        )  # (n_individuals, n_draws, n_constructs)

        n_individuals = data['n_individuals']
        n_draws = self.n_draws

        # Initialize log-likelihoods for each draw
        ll_draws = np.zeros((n_individuals, n_draws))

        # 1. Choice model likelihood
        # Compute utilities for each draw
        utility = self.compute_choice_utility(
            data['attributes'],
            beta,
            data['attribute_names'],
            lv_draws,
            data.get('lv_beta', {})
        )  # (n_individuals, n_draws, n_alts)

        # Choice probabilities
        probs = self.compute_choice_probability(
            utility,
            data.get('availability')
        )  # (n_individuals, n_draws, n_alts)

        # Log probability of chosen alternative
        choices = data['choices']  # (n_individuals,)
        for i in range(n_individuals):
            ll_draws[i, :] += np.log(probs[i, :, choices[i]] + 1e-10)

        # 2. Measurement model likelihood
        # Update measurement model with current loadings and thresholds
        self.measurement.loadings = loadings
        self.measurement.measurement_model.thresholds = thresholds

        indicators = data['indicators']  # (n_individuals, n_items)
        item_names = data['item_names']

        ll_measurement = self.measurement.log_likelihood_vectorized(
            indicators,
            item_names,
            lv_draws,
            self.construct_idx
        )  # (n_individuals, n_draws)

        ll_draws += ll_measurement

        # 3. Integrate over draws using log-sum-exp
        max_ll = ll_draws.max(axis=1, keepdims=True)
        log_sum_exp = max_ll.squeeze() + np.log(np.exp(ll_draws - max_ll).sum(axis=1))
        integrated_ll = log_sum_exp - np.log(n_draws)

        return integrated_ll

    def negative_log_likelihood(self,
                                params: np.ndarray,
                                data: Dict) -> float:
        """
        Compute negative sum of log-likelihoods for optimization.

        Args:
            params: Parameter vector
            data: Dict containing all model data

        Returns:
            Negative log-likelihood (scalar)
        """
        ll = self.log_likelihood_individual(params, data)
        return -ll.sum()

    def estimate(self,
                 df: pd.DataFrame,
                 choice_col: str,
                 attribute_cols: List[str],
                 covariate_cols: List[str],
                 indicator_cols: List[str],
                 lv_beta_init: Dict[str, float] = None,
                 method: str = 'BFGS',
                 maxiter: int = 1000,
                 verbose: bool = True) -> ICLVResult:
        """
        Estimate ICLV model using Simulated Maximum Likelihood.

        Args:
            df: DataFrame with all data
            choice_col: Name of choice column (0-indexed alternative)
            attribute_cols: List of attribute column names
            covariate_cols: List of covariate column names for structural model
            indicator_cols: List of indicator column names
            lv_beta_init: Initial values for LV effects in choice model
            method: Optimization method ('BFGS', 'L-BFGS-B', 'Nelder-Mead')
            maxiter: Maximum iterations
            verbose: Whether to print progress

        Returns:
            ICLVResult with estimates and statistics
        """
        n_individuals = len(df)

        # Prepare data dict
        n_alts = df[choice_col].max() + 1

        # Prepare attributes array from wide-format columns
        # Detect base attribute names (e.g., 'fee' from 'fee1', 'fee2', 'fee3')
        base_attrs = set()
        for col in attribute_cols:
            base = col.rstrip('0123456789')
            if base and base != col:
                base_attrs.add(base)

        if base_attrs:
            # Wide format: columns like fee1, fee2, fee3, dur1, dur2, dur3
            attribute_names = sorted(list(base_attrs))
            attributes = np.zeros((n_individuals, n_alts, len(attribute_names)))

            for attr_idx, base_attr in enumerate(attribute_names):
                for alt in range(n_alts):
                    col_name = f'{base_attr}{alt + 1}'
                    if col_name in df.columns:
                        attributes[:, alt, attr_idx] = df[col_name].values
        else:
            # No alternative-specific attributes found
            attribute_names = attribute_cols
            attributes = np.zeros((n_individuals, n_alts, len(attribute_cols)))
            for attr_idx, col in enumerate(attribute_cols):
                if col in df.columns:
                    attributes[:, :, attr_idx] = df[col].values[:, np.newaxis]

        data = {
            'n_individuals': n_individuals,
            'choices': df[choice_col].values.astype(int),
            'covariates': df[covariate_cols].values,
            'covariate_names': covariate_cols,
            'indicators': df[indicator_cols].values,
            'item_names': indicator_cols,
            'attributes': attributes,
            'attribute_names': attribute_names,
        }

        # Initialize starting values
        beta_init = {col: 0.0 for col in attribute_names}
        gamma_init = {lv: {cov: 0.0 for cov in covariate_cols}
                     for lv in self.construct_names}

        # Free loading items (first per construct fixed to 1)
        free_loading_items = []
        for construct, items in self.constructs.items():
            free_loading_items.extend(items[1:])

        loadings_init = {item: 0.7 for item in self.all_items}

        # Default thresholds
        thresholds_init = stats.norm.ppf(
            np.linspace(0, 1, self.n_categories + 1)[1:-1]
        )

        # Add LV betas if provided
        if lv_beta_init:
            beta_init.update(lv_beta_init)
            data['lv_beta'] = {k: v for k, v in lv_beta_init.items()}
        else:
            data['lv_beta'] = {}

        data['beta_keys'] = sorted(beta_init.keys())
        data['free_loading_items'] = free_loading_items

        # Pack initial parameters
        params_init = self._pack_parameters(
            beta_init, gamma_init, loadings_init, thresholds_init
        )

        n_params = len(params_init)

        if verbose:
            print(f"ICLV Estimation")
            print(f"=" * 50)
            print(f"N individuals: {n_individuals}")
            print(f"N parameters: {n_params}")
            print(f"N draws: {self.n_draws} ({self.draw_type})")
            print(f"Optimizing...")

        # Optimize
        result = optimize.minimize(
            self.negative_log_likelihood,
            params_init,
            args=(data,),
            method=method,
            options={'maxiter': maxiter, 'disp': verbose}
        )

        # Unpack final parameters
        beta_final, gamma_final, loadings_final, thresholds_final = \
            self._unpack_parameters(result.x, data['beta_keys'], free_loading_items)

        # Compute standard errors from Hessian
        try:
            hessian = result.hess_inv if hasattr(result, 'hess_inv') else None
            if hessian is not None:
                if hasattr(hessian, 'todense'):
                    hessian = hessian.todense()
                se = np.sqrt(np.diag(np.abs(hessian)))
            else:
                se = np.full(n_params, np.nan)
        except:
            se = np.full(n_params, np.nan)

        # Unpack standard errors
        idx = 0
        beta_se = {}
        for key in data['beta_keys']:
            beta_se[key] = se[idx]
            idx += 1

        gamma_se = {lv: {} for lv in self.construct_names}
        for lv in self.construct_names:
            for cov in self.covariates:
                gamma_se[lv][cov] = se[idx]
                idx += 1

        loadings_se = {}
        for item in free_loading_items:
            loadings_se[item] = se[idx]
            idx += 1

        thresholds_se = se[idx:idx + self.n_categories - 1]

        # Compute fit statistics
        ll = -result.fun
        aic = 2 * n_params - 2 * ll
        bic = n_params * np.log(n_individuals) - 2 * ll

        if verbose:
            print(f"\nOptimization complete")
            print(f"Log-likelihood: {ll:.4f}")
            print(f"AIC: {aic:.4f}")
            print(f"BIC: {bic:.4f}")

        return ICLVResult(
            beta=beta_final,
            gamma=gamma_final,
            loadings=loadings_final,
            thresholds=thresholds_final,
            beta_se=beta_se,
            gamma_se=gamma_se,
            loadings_se=loadings_se,
            thresholds_se=thresholds_se,
            log_likelihood=ll,
            n_parameters=n_params,
            n_observations=n_individuals,
            aic=aic,
            bic=bic,
            convergence=result.success,
            n_iterations=result.nit if hasattr(result, 'nit') else 0,
            hessian=hessian if hessian is not None else None,
            n_draws=self.n_draws,
            draw_type=self.draw_type
        )


if __name__ == '__main__':
    print("ICLV Core Model")
    print("=" * 50)

    # Demo setup
    model = ICLVModel(
        constructs={
            'env_concern': ['env1', 'env2', 'env3'],
            'tech_affinity': ['tech1', 'tech2', 'tech3']
        },
        covariates=['age', 'income', 'education'],
        n_draws=100
    )

    print(f"\nModel configured:")
    print(f"  Constructs: {model.construct_names}")
    print(f"  N draws: {model.n_draws}")
    print(f"  N items: {len(model.all_items)}")
