"""
Elasticity Calculations for DCM Models
======================================

Compute price elasticities for discrete choice models.

Elasticities measure the percentage change in choice probability
resulting from a percentage change in an attribute.

Formulas (Standard Logit/MNL):
    Own-price elasticity:
        η_jj = β * x_j * (1 - P_j)

    Cross-price elasticity:
        η_jk = -β * x_k * P_k

    Where:
        β = coefficient (e.g., B_FEE)
        x = attribute value (scaled if necessary)
        P = choice probability

For fee elasticities with scale factor:
    η_jj = B_FEE * (fee_j / scale) * (1 - P_j)
    η_jk = -B_FEE * (fee_k / scale) * P_k

IMPORTANT LIMITATION FOR HCM/ICLV MODELS:
=========================================
These formulas are ONLY correct for standard MNL models. For HCM/ICLV models
with latent variable interactions (e.g., B_FEE_i = B_FEE + B_FEE_LV * η),
the elasticity formulas are more complex because:

1. The effective coefficient varies across individuals: β_i = β + β_LV * η_i
2. The derivative ∂P/∂β must account for how η enters the utility
3. Standard errors require the full covariance matrix including LV parameters

For HCM/ICLV, the correct approach is:
- Use simulation-based elasticities (draw η and compute for each individual)
- Or compute average elasticity across the population
- Standard errors should use bootstrap or simulation

References:
- Train (2009): Discrete Choice Methods with Simulation, Ch. 6
- Hensher et al. (2015): Applied Choice Analysis, Ch. 13
- Ben-Akiva et al. (2002): Hybrid Choice Models

Authors: Hakan Mülayim, Giray Girengir, Ataol Azeritürk
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from scipy import stats

from .base import (
    EstimationResult, PolicyScenario, PolicyAnalysisConfig,
    compute_logit_probabilities, compute_utilities,
    delta_method_se, krinsky_robb_draws
)


@dataclass
class ElasticityResult:
    """
    Container for elasticity calculation results.

    Attributes:
        elasticity: Point estimate
        se: Standard error
        ci_lower: Lower confidence bound
        ci_upper: Upper confidence bound
        alternative: Alternative index
        attribute: Attribute name
        elasticity_type: 'own' or 'cross'
        with_respect_to: For cross elasticity, the alternative whose attribute changes
    """
    elasticity: float
    se: float
    ci_lower: float
    ci_upper: float
    alternative: int
    attribute: str
    elasticity_type: str  # 'own' or 'cross'
    with_respect_to: int = None  # For cross elasticity

    def __str__(self) -> str:
        if self.elasticity_type == 'own':
            return f"Own-price elasticity Alt {self.alternative}: {self.elasticity:.3f}"
        else:
            return (f"Cross elasticity Alt {self.alternative} w.r.t. "
                    f"Alt {self.with_respect_to}: {self.elasticity:.3f}")


class ElasticityCalculator:
    """
    Elasticity calculator for DCM models.

    Computes own-price, cross-price, and aggregate elasticities
    with uncertainty quantification.

    Example:
        >>> scenario = PolicyScenario(
        ...     name='Current',
        ...     attributes={'fee': np.array([500000, 600000, 0]),
        ...                 'dur': np.array([10, 8, 15])}
        ... )
        >>> result = EstimationResult(betas={'B_FEE': -0.5, ...}, ...)
        >>> calc = ElasticityCalculator(result)
        >>> eta = calc.own_price_elasticity(scenario, alternative=0)
    """

    def __init__(self,
                 result: Union[EstimationResult, Dict[str, Any]],
                 config: PolicyAnalysisConfig = None):
        """
        Initialize elasticity calculator.

        Args:
            result: Estimation results
            config: Configuration parameters
        """
        if isinstance(result, dict):
            self.result = EstimationResult.from_dict(result)
        else:
            self.result = result

        self.config = config or PolicyAnalysisConfig()

        # Check for latent variable parameters and warn if detected
        lv_params = [p for p in self.result.betas.keys()
                     if any(lv in p.lower() for lv in ['patblind', 'pat_blind', 'secdl', 'sec_dl',
                                                        '_lv', 'latent', 'gamma_', 'sigma_lv'])]
        if lv_params:
            import warnings
            warnings.warn(
                f"HCM/ICLV parameters detected: {lv_params}. "
                f"Elasticity formulas in this module are only correct for standard MNL. "
                f"For HCM/ICLV models, use simulation-based elasticities or bootstrap. "
                f"See module docstring for details.",
                UserWarning
            )

    def _compute_probabilities(self, scenario: PolicyScenario) -> np.ndarray:
        """Compute choice probabilities for a scenario."""
        utilities = compute_utilities(scenario, self.result, self.config)
        return compute_logit_probabilities(utilities)

    def own_price_elasticity(self,
                              scenario: PolicyScenario,
                              alternative: int,
                              attribute: str = 'fee',
                              coefficient_param: str = None) -> ElasticityResult:
        """
        Compute own-price elasticity for an alternative.

        Own-price elasticity: η_jj = β * x_j * (1 - P_j)

        This measures how the probability of choosing alternative j
        changes when its own attribute changes.

        SIGN INTERPRETATION (consistent with economic theory):
        - For "bad" attributes (β < 0, e.g., fee, duration):
          η < 0 means INCREASE in attribute DECREASES choice probability
          This is expected: higher fee → lower demand (correct negative sign)

        - For "good" attributes (β > 0):
          η > 0 means INCREASE in attribute INCREASES choice probability

        NOTE: Unlike WTP which has a `for_improvement` option to flip signs,
        elasticity preserves the raw formula sign. This is intentional because:
        - Elasticity sign directly indicates direction of response
        - Negative own-price elasticity is standard for normal goods
        - Cross-elasticity sign indicates substitute (positive) vs complement (negative)

        Args:
            scenario: Policy scenario with attribute levels
            alternative: Alternative index (0-indexed)
            attribute: Attribute name ('fee', 'dur', etc.)
            coefficient_param: Parameter name (default: B_FEE for fee)

        Returns:
            ElasticityResult with point estimate and CI
        """
        # Determine coefficient parameter
        if coefficient_param is None:
            if attribute == 'fee':
                coefficient_param = self.config.fee_param
            elif attribute == 'dur':
                coefficient_param = self.config.dur_param
            else:
                coefficient_param = f'B_{attribute.upper()}'

        beta = self.result.betas.get(coefficient_param, 0)

        # Get attribute value (apply scaling for fee)
        x = scenario.get_attribute(attribute, alternative)
        if attribute == 'fee':
            x = x / self.config.fee_scale

        # Compute probability
        probs = self._compute_probabilities(scenario)
        p_j = probs[alternative]

        # Own-price elasticity: η = β * x * (1 - P)
        elasticity = beta * x * (1 - p_j)

        # Standard error via delta method
        se = self._elasticity_se_delta(
            scenario, alternative, alternative,
            attribute, coefficient_param, 'own'
        )

        z = self.config.z_score
        ci_lower = elasticity - z * se
        ci_upper = elasticity + z * se

        return ElasticityResult(
            elasticity=elasticity,
            se=se,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            alternative=alternative,
            attribute=attribute,
            elasticity_type='own'
        )

    def cross_price_elasticity(self,
                                scenario: PolicyScenario,
                                alternative: int,
                                with_respect_to: int,
                                attribute: str = 'fee',
                                coefficient_param: str = None) -> ElasticityResult:
        """
        Compute cross-price elasticity between alternatives.

        Cross-price elasticity: η_jk = -β * x_k * P_k

        This measures how the probability of choosing alternative j
        changes when the attribute of alternative k changes.

        Args:
            scenario: Policy scenario
            alternative: Alternative whose probability changes
            with_respect_to: Alternative whose attribute changes
            attribute: Attribute name
            coefficient_param: Coefficient parameter name

        Returns:
            ElasticityResult with point estimate and CI
        """
        if alternative == with_respect_to:
            return self.own_price_elasticity(
                scenario, alternative, attribute, coefficient_param
            )

        # Determine coefficient parameter
        if coefficient_param is None:
            if attribute == 'fee':
                coefficient_param = self.config.fee_param
            elif attribute == 'dur':
                coefficient_param = self.config.dur_param
            else:
                coefficient_param = f'B_{attribute.upper()}'

        beta = self.result.betas.get(coefficient_param, 0)

        # Get attribute value of k (apply scaling for fee)
        x_k = scenario.get_attribute(attribute, with_respect_to)
        if attribute == 'fee':
            x_k = x_k / self.config.fee_scale

        # Compute probability of k
        probs = self._compute_probabilities(scenario)
        p_k = probs[with_respect_to]

        # Cross-price elasticity: η_jk = -β * x_k * P_k
        elasticity = -beta * x_k * p_k

        # Standard error
        se = self._elasticity_se_delta(
            scenario, alternative, with_respect_to,
            attribute, coefficient_param, 'cross'
        )

        z = self.config.z_score
        ci_lower = elasticity - z * se
        ci_upper = elasticity + z * se

        return ElasticityResult(
            elasticity=elasticity,
            se=se,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            alternative=alternative,
            attribute=attribute,
            elasticity_type='cross',
            with_respect_to=with_respect_to
        )

    def elasticity_matrix(self,
                          scenario: PolicyScenario,
                          attribute: str = 'fee') -> pd.DataFrame:
        """
        Compute full elasticity matrix for all alternatives.

        Returns J×J matrix where:
        - Diagonal: own-price elasticities
        - Off-diagonal: cross-price elasticities

        Args:
            scenario: Policy scenario
            attribute: Attribute for elasticity

        Returns:
            DataFrame with elasticity matrix
        """
        n_alts = scenario.n_alternatives

        matrix = np.zeros((n_alts, n_alts))

        for j in range(n_alts):
            for k in range(n_alts):
                if j == k:
                    result = self.own_price_elasticity(scenario, j, attribute)
                else:
                    result = self.cross_price_elasticity(scenario, j, k, attribute)
                matrix[j, k] = result.elasticity

        # Create DataFrame with labeled indices
        labels = [f'Alt {i+1}' for i in range(n_alts)]
        df = pd.DataFrame(matrix, index=labels, columns=labels)
        df.index.name = 'P(choose)'
        df.columns.name = f'{attribute} of'

        return df

    def aggregate_elasticity(self,
                              scenario: PolicyScenario,
                              sample_data: pd.DataFrame = None,
                              alternative: int = 0,
                              attribute: str = 'fee',
                              weights: np.ndarray = None) -> ElasticityResult:
        """
        Compute aggregate (sample-weighted) elasticity.

        When sample_data is provided, computes elasticity for each
        individual and returns weighted average.

        Args:
            scenario: Base scenario
            sample_data: DataFrame with individual attributes (optional)
            alternative: Alternative index
            attribute: Attribute name
            weights: Optional weights for aggregation

        Returns:
            ElasticityResult with aggregate elasticity
        """
        if sample_data is None:
            # Return point elasticity if no sample data
            return self.own_price_elasticity(scenario, alternative, attribute)

        # Compute individual elasticities
        individual_elasticities = []

        for idx, row in sample_data.iterrows():
            # Create individual-specific scenario
            ind_scenario = PolicyScenario(
                name=f'individual_{idx}',
                attributes=scenario.attributes.copy(),
                n_alternatives=scenario.n_alternatives
            )

            result = self.own_price_elasticity(ind_scenario, alternative, attribute)
            individual_elasticities.append(result.elasticity)

        elasticities = np.array(individual_elasticities)

        if weights is None:
            weights = np.ones(len(elasticities)) / len(elasticities)
        else:
            weights = weights / np.sum(weights)

        aggregate = np.sum(weights * elasticities)
        se = np.sqrt(np.sum(weights**2 * np.var(elasticities)))

        z = self.config.z_score

        return ElasticityResult(
            elasticity=aggregate,
            se=se,
            ci_lower=aggregate - z * se,
            ci_upper=aggregate + z * se,
            alternative=alternative,
            attribute=attribute,
            elasticity_type='aggregate'
        )

    def _elasticity_se_delta(self,
                              scenario: PolicyScenario,
                              alt_j: int,
                              alt_k: int,
                              attribute: str,
                              coefficient_param: str,
                              elasticity_type: str) -> float:
        """
        Compute standard error for elasticity using delta method.

        For own-price: η = β * x * (1 - P)
        For cross-price: η = -β * x * P

        Full gradient includes how P varies with β:
        For logit, ∂P_j/∂β = P_j * (1 - P_j) * x_j

        Own-price: ∂η/∂β = x*(1-P) + β*x*(-∂P/∂β)
                        = x*(1-P) - β*x*P*(1-P)*x
                        = x*(1-P)*(1 - β*x*P)

        Cross-price: ∂η/∂β = -x*P + (-β)*x*∂P/∂β
                          = -x*P - β*x*P*(1-P)*x
                          = -x*P*(1 + β*x*(1-P))
        """
        beta = self.result.betas.get(coefficient_param, 0)
        se_beta = self.result.std_errs.get(coefficient_param, 0)

        x = scenario.get_attribute(attribute, alt_k)
        if attribute == 'fee':
            x = x / self.config.fee_scale

        probs = self._compute_probabilities(scenario)
        p_j = probs[alt_j]
        p_k = probs[alt_k]

        if elasticity_type == 'own':
            # η = β * x * (1 - P)
            # Full gradient: ∂η/∂β = x * (1 - P) * (1 - β * x * P)
            grad_beta = x * (1 - p_j) * (1 - beta * x * p_j)
        else:
            # η = -β * x * P_k
            # Full gradient: ∂η/∂β = -x * P_k * (1 + β * x * (1 - P_k))
            grad_beta = -x * p_k * (1 + beta * x * (1 - p_k))

        return abs(grad_beta) * se_beta

    def elasticity_simulation(self,
                               scenario: PolicyScenario,
                               alternative: int,
                               attribute: str = 'fee',
                               n_draws: int = None) -> Dict[str, Any]:
        """
        Compute elasticity distribution via simulation.

        Draws parameters from estimated distribution and computes
        elasticity for each draw.

        Args:
            scenario: Policy scenario
            alternative: Alternative index
            attribute: Attribute name
            n_draws: Number of simulation draws

        Returns:
            Dictionary with distribution statistics and draws
        """
        n = n_draws or self.config.n_simulations

        # Get relevant parameters
        if attribute == 'fee':
            coef_param = self.config.fee_param
        elif attribute == 'dur':
            coef_param = self.config.dur_param
        else:
            coef_param = f'B_{attribute.upper()}'

        # Parameters affecting probabilities
        params = [coef_param, self.config.asc_param, self.config.dur_param]
        params = [p for p in params if p in self.result.betas]

        # Draw parameters
        draws = krinsky_robb_draws(self.result, params, n, self.config.seed)

        # Compute elasticity for each draw
        elasticities = []
        x = scenario.get_attribute(attribute, alternative)
        if attribute == 'fee':
            x = x / self.config.fee_scale

        for i in range(n):
            # Create temporary result with drawn parameters
            temp_betas = self.result.betas.copy()
            for j, p in enumerate(params):
                temp_betas[p] = draws[i, j]

            temp_result = EstimationResult(
                betas=temp_betas,
                std_errs=self.result.std_errs
            )

            # Compute utilities and probabilities with drawn params
            utilities = compute_utilities(scenario, temp_result, self.config)
            probs = compute_logit_probabilities(utilities)

            beta = temp_betas[coef_param]
            eta = beta * x * (1 - probs[alternative])
            elasticities.append(eta)

        elasticities = np.array(elasticities)

        alpha = 1 - self.config.confidence_level
        return {
            'mean': np.mean(elasticities),
            'std': np.std(elasticities),
            'median': np.median(elasticities),
            'ci_lower': np.percentile(elasticities, alpha/2 * 100),
            'ci_upper': np.percentile(elasticities, (1 - alpha/2) * 100),
            'draws': elasticities
        }

    def print_summary(self,
                      scenario: PolicyScenario,
                      attribute: str = 'fee') -> None:
        """Print formatted elasticity summary."""
        print(f"\n{'='*60}")
        print(f"ELASTICITY ANALYSIS: {attribute.upper()}")
        print(f"{'='*60}")
        print(f"Scenario: {scenario.name}")
        print()

        # Choice probabilities
        probs = self._compute_probabilities(scenario)
        print("Choice Probabilities:")
        for j in range(scenario.n_alternatives):
            print(f"  Alternative {j+1}: {probs[j]:.3f}")
        print()

        # Elasticity matrix
        print("Elasticity Matrix:")
        matrix = self.elasticity_matrix(scenario, attribute)
        print(matrix.to_string())
        print()

        # Interpretation
        print("Own-Price Elasticities:")
        for j in range(scenario.n_alternatives):
            eta = matrix.iloc[j, j]
            if abs(eta) < 1:
                interp = "inelastic"
            elif abs(eta) > 1:
                interp = "elastic"
            else:
                interp = "unit elastic"
            print(f"  Alt {j+1}: {eta:.3f} ({interp})")

        print(f"\n{'='*60}")


def compute_hcm_elasticity(result: Union[EstimationResult, Dict[str, Any]],
                            scenario: PolicyScenario,
                            alternative: int = 0,
                            attribute: str = 'fee',
                            lv_interaction_params: Dict[str, str] = None,
                            n_draws: int = 10000,
                            config: PolicyAnalysisConfig = None,
                            seed: int = 42) -> Dict[str, Any]:
    """
    Compute simulation-based elasticity for HCM/ICLV models.

    CORRECT METHOD for models with latent variable heterogeneity.

    For HCM/ICLV, the effective coefficient varies across individuals:
        β_i = β + Σ β_LV_k * η_k

    where η_k ~ N(μ_k, σ_k²) are latent variables.

    This function:
    1. Draws η values from their estimated distributions
    2. Computes individual-specific coefficients β_i
    3. Calculates elasticity for each simulated individual
    4. Returns distribution of elasticities with proper uncertainty

    Args:
        result: Estimation results containing LV parameters
        scenario: Policy scenario with attribute levels
        alternative: Alternative index for own-price elasticity
        attribute: Attribute name ('fee' or 'dur')
        lv_interaction_params: Dict mapping LV names to interaction parameters
            e.g., {'pat_blind': 'B_FEE_PatBlind', 'sec_dl': 'B_FEE_SecDL'}
            If None, attempts to auto-detect from result.betas
        n_draws: Number of simulation draws
        config: Policy analysis configuration
        seed: Random seed for reproducibility

    Returns:
        Dictionary with:
        - mean: Population-average elasticity
        - std: Standard deviation of elasticity distribution
        - median: Median elasticity
        - ci_lower, ci_upper: 95% confidence interval
        - pct_elastic: Percent of population with |η| > 1
        - draws: Raw elasticity draws for further analysis

    Example:
        >>> # For ICLV model with two latent variables
        >>> elas = compute_hcm_elasticity(
        ...     result,
        ...     scenario,
        ...     alternative=0,
        ...     lv_interaction_params={
        ...         'pat_blind': 'B_FEE_PatBlind',
        ...         'sec_dl': 'B_FEE_SecDL'
        ...     }
        ... )
        >>> print(f"Population elasticity: {elas['mean']:.3f}")
    """
    if isinstance(result, dict):
        result = EstimationResult.from_dict(result)

    config = config or PolicyAnalysisConfig()

    # Determine base coefficient parameter
    if attribute == 'fee':
        base_param = config.fee_param
    elif attribute == 'dur':
        base_param = config.dur_param
    else:
        base_param = f'B_{attribute.upper()}'

    beta_base = result.betas.get(base_param, 0)

    # Auto-detect LV interaction parameters if not provided
    # Uses pattern matching on parameter names rather than hardcoded LV names
    if lv_interaction_params is None:
        lv_interaction_params = {}
        base_attr = base_param.lower().replace('b_', '')  # e.g., 'fee' or 'dur'

        for param_name in result.betas.keys():
            param_lower = param_name.lower()

            # Skip the base parameter itself
            if param_name == base_param:
                continue

            # Check if this looks like an LV interaction: B_<attr>_<lvname>
            # Pattern: param contains the attribute name AND is longer (has LV suffix)
            if base_attr in param_lower and len(param_name) > len(base_param):
                # Extract the LV name from the parameter name
                # e.g., 'B_FEE_PAT_BLIND' -> 'pat_blind'
                # or 'B_FEE_PATBLIND' -> 'patblind'
                parts = param_lower.split('_')

                # Find where the attribute name is and take everything after
                try:
                    attr_idx = parts.index(base_attr)
                    lv_parts = parts[attr_idx + 1:]
                    if lv_parts:
                        lv_name = '_'.join(lv_parts)
                        lv_interaction_params[lv_name] = param_name
                except (ValueError, IndexError):
                    # Attribute not found as separate part, try alternative patterns
                    # e.g., 'b_fee_patblind' where 'fee' is part of 'b_fee'
                    if param_lower.startswith(f'b_{base_attr}_'):
                        suffix = param_lower[len(f'b_{base_attr}_'):]
                        if suffix:
                            lv_interaction_params[suffix] = param_name

    # Get LV distribution parameters (sigma values)
    lv_sigmas = {}
    for lv_name in lv_interaction_params.keys():
        # Look for sigma parameter for this LV
        for param_name in result.betas.keys():
            if 'sigma' in param_name.lower() and lv_name in param_name.lower():
                lv_sigmas[lv_name] = abs(result.betas[param_name])
                break
        if lv_name not in lv_sigmas:
            # Default to 1.0 if not found (standard normal)
            lv_sigmas[lv_name] = 1.0

    # Simulate LV draws
    np.random.seed(seed)
    n_lvs = len(lv_interaction_params)

    if n_lvs == 0:
        # No LV interactions found - fall back to standard calculation
        import warnings
        warnings.warn(
            f"No LV interaction parameters found for {base_param}. "
            f"Using standard MNL elasticity formula.",
            UserWarning
        )
        calc = ElasticityCalculator(result, config)
        std_result = calc.own_price_elasticity(scenario, alternative, attribute)
        return {
            'mean': std_result.elasticity,
            'std': std_result.se,
            'median': std_result.elasticity,
            'ci_lower': std_result.ci_lower,
            'ci_upper': std_result.ci_upper,
            'pct_elastic': 100.0 if abs(std_result.elasticity) > 1 else 0.0,
            'draws': np.array([std_result.elasticity])
        }

    # Draw LV values (standard normal, scale by sigma)
    lv_draws = {}
    for lv_name, sigma in lv_sigmas.items():
        lv_draws[lv_name] = np.random.normal(0, sigma, n_draws)

    # Compute individual-specific coefficients
    beta_individuals = np.full(n_draws, beta_base)
    for lv_name, interaction_param in lv_interaction_params.items():
        beta_lv = result.betas.get(interaction_param, 0)
        beta_individuals += beta_lv * lv_draws[lv_name]

    # Get attribute value
    x = scenario.get_attribute(attribute, alternative)
    if attribute == 'fee':
        x = x / config.fee_scale

    # Compute utilities and probabilities for each individual
    elasticities = np.zeros(n_draws)

    # Base utilities (without the random LV effects on the specific attribute)
    base_utilities = compute_utilities(scenario, result, config)
    base_probs = compute_logit_probabilities(base_utilities)

    for i in range(n_draws):
        # Adjust utility for this individual's coefficient
        # For alternative j: V_j = ... + β_i * x_j + ...
        # The difference from base is (β_i - β_base) * x_j
        delta_beta = beta_individuals[i] - beta_base

        # Compute adjusted utilities
        utilities_i = base_utilities.copy()
        for j in range(len(utilities_i)):
            x_j = scenario.get_attribute(attribute, j)
            if attribute == 'fee':
                x_j = x_j / config.fee_scale
            utilities_i[j] += delta_beta * x_j

        probs_i = compute_logit_probabilities(utilities_i)

        # Own-price elasticity: η = β_i * x * (1 - P)
        elasticities[i] = beta_individuals[i] * x * (1 - probs_i[alternative])

    # Compute summary statistics
    alpha = 1 - config.confidence_level

    return {
        'mean': np.mean(elasticities),
        'std': np.std(elasticities),
        'median': np.median(elasticities),
        'ci_lower': np.percentile(elasticities, alpha/2 * 100),
        'ci_upper': np.percentile(elasticities, (1 - alpha/2) * 100),
        'pct_elastic': 100 * np.mean(np.abs(elasticities) > 1),
        'draws': elasticities,
        'beta_distribution': {
            'mean': np.mean(beta_individuals),
            'std': np.std(beta_individuals),
            'min': np.min(beta_individuals),
            'max': np.max(beta_individuals)
        }
    }


def arc_elasticity(p1: float, p2: float, x1: float, x2: float) -> float:
    """
    Compute arc elasticity between two points.

    Arc elasticity = ((P2-P1)/P_avg) / ((X2-X1)/X_avg)

    Args:
        p1, p2: Probabilities at points 1 and 2
        x1, x2: Attribute values at points 1 and 2

    Returns:
        Arc elasticity
    """
    p_avg = (p1 + p2) / 2
    x_avg = (x1 + x2) / 2

    if x_avg == 0 or x1 == x2:
        return np.nan

    return ((p2 - p1) / p_avg) / ((x2 - x1) / x_avg)


if __name__ == '__main__':
    print("Elasticity Calculator Module")
    print("=" * 40)

    # Example
    result = EstimationResult(
        betas={'B_FEE': -0.5, 'B_DUR': -0.08, 'ASC_paid': 1.2},
        std_errs={'B_FEE': 0.05, 'B_DUR': 0.02, 'ASC_paid': 0.15}
    )

    scenario = PolicyScenario(
        name='Example',
        attributes={
            'fee': np.array([500000, 600000, 0]),
            'dur': np.array([10, 8, 15])
        }
    )

    calc = ElasticityCalculator(result)
    calc.print_summary(scenario, 'fee')
