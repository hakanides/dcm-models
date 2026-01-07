"""
Elasticity Calculations for DCM Models
======================================

Compute price elasticities for discrete choice models.

Elasticities measure the percentage change in choice probability
resulting from a percentage change in an attribute.

Formulas (Logit):
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

Author: DCM Research Team
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
