"""
Marginal Effects for DCM Models
===============================

Compute marginal effects showing how choice probabilities change
with unit changes in attributes.

Key Formulas (Logit):
    Own marginal effect:
        ∂P_j/∂x_j = β * P_j * (1 - P_j)

    Cross marginal effect:
        ∂P_j/∂x_k = -β * P_j * P_k

Types of marginal effects:
- MEM: Marginal Effect at the Mean (evaluate at sample means)
- AME: Average Marginal Effect (average individual effects)
- MER: Marginal Effect at Representative values

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
class MarginalEffectResult:
    """
    Container for marginal effect results.

    Attributes:
        me: Marginal effect point estimate
        se: Standard error
        ci_lower: Lower confidence bound
        ci_upper: Upper confidence bound
        alternative: Alternative index
        attribute: Attribute name
        method: 'MEM', 'AME', or 'point'
        probability: Choice probability at evaluation point
    """
    me: float
    se: float
    ci_lower: float
    ci_upper: float
    alternative: int
    attribute: str
    method: str
    probability: float = None

    def __str__(self) -> str:
        return (f"ME({self.attribute}, Alt {self.alternative}): "
                f"{self.me:.6f} (SE: {self.se:.6f})")


class MarginalEffectCalculator:
    """
    Marginal effects calculator for DCM models.

    Computes how choice probabilities change with changes in attributes,
    holding other factors constant.

    Example:
        >>> calc = MarginalEffectCalculator(result)
        >>> me = calc.marginal_effect_at_point(scenario, alternative=0, attribute='fee')
        >>> print(f"A 1-unit increase in fee changes P by {me.me:.4f}")
    """

    def __init__(self,
                 result: Union[EstimationResult, Dict[str, Any]],
                 config: PolicyAnalysisConfig = None):
        """
        Initialize marginal effects calculator.

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

    def _get_coefficient(self, attribute: str) -> Tuple[str, float]:
        """Get coefficient name and value for an attribute."""
        if attribute == 'fee':
            param = self.config.fee_param
        elif attribute == 'dur':
            param = self.config.dur_param
        else:
            param = f'B_{attribute.upper()}'

        beta = self.result.betas.get(param, 0)
        return param, beta

    def marginal_effect_at_point(self,
                                  scenario: PolicyScenario,
                                  alternative: int,
                                  attribute: str = 'fee') -> MarginalEffectResult:
        """
        Compute marginal effect at a specific point.

        For own attribute:
            ∂P_j/∂x_j = β * P_j * (1 - P_j)

        Note: For fee, the coefficient is on scaled fee, so:
            ∂P/∂fee = β * P * (1-P) / scale

        Args:
            scenario: Policy scenario defining the evaluation point
            alternative: Alternative index
            attribute: Attribute name

        Returns:
            MarginalEffectResult with point estimate and CI
        """
        param, beta = self._get_coefficient(attribute)
        probs = self._compute_probabilities(scenario)
        p_j = probs[alternative]

        # Own marginal effect: ∂P/∂x = β * P * (1 - P)
        me = beta * p_j * (1 - p_j)

        # For fee, convert to original units using chain rule:
        # Utility uses fee_scaled = fee / scale, so β is effect per unit fee_scaled
        # ∂P/∂fee = ∂P/∂fee_scaled * ∂fee_scaled/∂fee = β * P * (1-P) * (1/scale)
        # This gives effect per 1 TL (not per 10,000 TL)
        if attribute == 'fee':
            me = me / self.config.fee_scale

        # Standard error via delta method
        se = self._me_se_delta(scenario, alternative, attribute, param)

        z = self.config.z_score

        return MarginalEffectResult(
            me=me,
            se=se,
            ci_lower=me - z * se,
            ci_upper=me + z * se,
            alternative=alternative,
            attribute=attribute,
            method='point',
            probability=p_j
        )

    def cross_marginal_effect(self,
                               scenario: PolicyScenario,
                               alternative: int,
                               with_respect_to: int,
                               attribute: str = 'fee') -> MarginalEffectResult:
        """
        Compute cross marginal effect.

        Cross marginal effect:
            ∂P_j/∂x_k = -β * P_j * P_k

        How probability of j changes when attribute of k changes.

        ISSUE #19 FIX: Now uses delta method for SE consistency with own ME.

        Args:
            scenario: Policy scenario
            alternative: Alternative whose probability changes
            with_respect_to: Alternative whose attribute changes
            attribute: Attribute name

        Returns:
            MarginalEffectResult
        """
        if alternative == with_respect_to:
            return self.marginal_effect_at_point(scenario, alternative, attribute)

        param, beta = self._get_coefficient(attribute)
        probs = self._compute_probabilities(scenario)
        p_j = probs[alternative]
        p_k = probs[with_respect_to]

        # Cross marginal effect: ∂P_j/∂x_k = -β * P_j * P_k
        me = -beta * p_j * p_k

        # For fee, convert to original units (same chain rule as own ME)
        if attribute == 'fee':
            me = me / self.config.fee_scale

        # ISSUE #19 FIX: Use delta method for consistency with own ME
        # Cross ME: ∂P_j/∂x_k = -β * P_j * P_k
        # Gradient w.r.t. β (full, including ∂P/∂β terms):
        # ∂ME/∂β = -P_j * P_k - β * (∂P_j/∂β * P_k + P_j * ∂P_k/∂β)
        # Where ∂P_j/∂β = -x_k * P_j * P_k (for cross alternative)
        # This is complex, so we use simplified form but document it
        #
        # Simplified gradient (treating P as ~constant, consistent with _me_se_delta):
        # ∂ME/∂β = -P_j * P_k
        se_beta = self.result.std_errs.get(param, 0)
        grad = -p_j * p_k  # Gradient w.r.t. beta
        se = abs(grad) * se_beta

        if attribute == 'fee':
            se = se / self.config.fee_scale

        z = self.config.z_score

        return MarginalEffectResult(
            me=me,
            se=se,
            ci_lower=me - z * se,
            ci_upper=me + z * se,
            alternative=alternative,
            attribute=attribute,
            method='cross',
            probability=p_j
        )

    def marginal_effect_matrix(self,
                                scenario: PolicyScenario,
                                attribute: str = 'fee') -> pd.DataFrame:
        """
        Compute full marginal effect matrix.

        Diagonal: own marginal effects
        Off-diagonal: cross marginal effects

        Args:
            scenario: Policy scenario
            attribute: Attribute name

        Returns:
            DataFrame with J×J marginal effect matrix
        """
        n_alts = scenario.n_alternatives
        matrix = np.zeros((n_alts, n_alts))

        for j in range(n_alts):
            for k in range(n_alts):
                if j == k:
                    result = self.marginal_effect_at_point(scenario, j, attribute)
                else:
                    result = self.cross_marginal_effect(scenario, j, k, attribute)
                matrix[j, k] = result.me

        labels = [f'Alt {i+1}' for i in range(n_alts)]
        df = pd.DataFrame(matrix, index=labels, columns=labels)
        df.index.name = 'ΔP for'
        df.columns.name = f'Δ{attribute} of'

        return df

    def average_marginal_effect(self,
                                 sample_data: pd.DataFrame,
                                 alternative: int,
                                 attribute: str = 'fee',
                                 attribute_col: str = None,
                                 weights: np.ndarray = None) -> MarginalEffectResult:
        """
        Compute Average Marginal Effect (AME) across sample.

        AME = (1/N) * Σ_i ME_i

        Where ME_i is the marginal effect for individual i,
        evaluated at their specific attribute levels.

        Args:
            sample_data: DataFrame with individual-level attributes
            alternative: Alternative index
            attribute: Attribute name
            attribute_col: Column name in sample_data for this attribute
            weights: Optional weights for averaging

        Returns:
            MarginalEffectResult with AME
        """
        param, beta = self._get_coefficient(attribute)

        if attribute_col is None:
            attribute_col = attribute

        # Compute ME for each individual
        individual_mes = []

        for idx, row in sample_data.iterrows():
            # Create scenario for this individual
            # Assume sample_data has columns for each alternative's attributes
            # e.g., fee_1, fee_2, fee_3, dur_1, dur_2, dur_3

            attrs = {}
            for attr_name in ['fee', 'dur']:
                vals = []
                for j in range(self.config.n_alternatives):
                    col = f'{attr_name}_{j+1}'
                    if col in row:
                        vals.append(row[col])
                    else:
                        vals.append(0)
                attrs[attr_name] = np.array(vals)

            if len(attrs['fee']) == 0:
                # Fallback: use same attributes for everyone
                continue

            scenario = PolicyScenario(
                name=f'individual_{idx}',
                attributes=attrs,
                n_alternatives=self.config.n_alternatives
            )

            me_result = self.marginal_effect_at_point(scenario, alternative, attribute)
            individual_mes.append(me_result.me)

        if len(individual_mes) == 0:
            # No individual variation; compute at means
            return self.marginal_effect_at_means(sample_data, alternative, attribute)

        mes = np.array(individual_mes)

        if weights is None:
            weights = np.ones(len(mes)) / len(mes)
        else:
            weights = weights / np.sum(weights)

        ame = np.sum(weights * mes)
        se = np.sqrt(np.sum(weights**2 * np.var(mes, ddof=1)))

        z = self.config.z_score

        return MarginalEffectResult(
            me=ame,
            se=se,
            ci_lower=ame - z * se,
            ci_upper=ame + z * se,
            alternative=alternative,
            attribute=attribute,
            method='AME'
        )

    def marginal_effect_at_means(self,
                                  sample_data: pd.DataFrame = None,
                                  alternative: int = 0,
                                  attribute: str = 'fee',
                                  scenario: PolicyScenario = None) -> MarginalEffectResult:
        """
        Compute Marginal Effect at the Mean (MEM).

        Evaluates ME at sample mean values of all attributes.

        Args:
            sample_data: DataFrame with attribute columns
            alternative: Alternative index
            attribute: Attribute name
            scenario: Pre-computed scenario at means (optional)

        Returns:
            MarginalEffectResult with MEM
        """
        if scenario is None and sample_data is not None:
            # Compute means from sample
            attrs = {}
            for attr_name in ['fee', 'dur']:
                vals = []
                for j in range(self.config.n_alternatives):
                    col = f'{attr_name}_{j+1}'
                    if col in sample_data.columns:
                        vals.append(sample_data[col].mean())
                    else:
                        vals.append(0)
                attrs[attr_name] = np.array(vals)

            scenario = PolicyScenario(
                name='at_means',
                attributes=attrs,
                n_alternatives=self.config.n_alternatives
            )
        elif scenario is None:
            raise ValueError("Either sample_data or scenario must be provided")

        result = self.marginal_effect_at_point(scenario, alternative, attribute)
        result.method = 'MEM'
        return result

    def discrete_change_effect(self,
                                scenario: PolicyScenario,
                                alternative: int,
                                attribute: str,
                                from_value: float,
                                to_value: float) -> Dict[str, float]:
        """
        Compute effect of discrete change in attribute.

        Computes ΔP = P(x=to_value) - P(x=from_value)

        Useful for policy analysis with specific changes.

        Args:
            scenario: Base scenario
            alternative: Alternative index
            attribute: Attribute to change
            from_value: Starting value
            to_value: Ending value

        Returns:
            Dict with probability change and percentage change
        """
        # Compute baseline probability
        p_baseline = self._compute_probabilities(scenario)[alternative]

        # Create modified scenario
        modified = scenario.with_modification(attribute, alternative, to_value)
        p_modified = self._compute_probabilities(modified)[alternative]

        delta_p = p_modified - p_baseline
        pct_change = delta_p / p_baseline * 100 if p_baseline > 0 else np.nan

        return {
            'p_baseline': p_baseline,
            'p_modified': p_modified,
            'delta_p': delta_p,
            'percent_change': pct_change,
            'attribute_change': to_value - from_value
        }

    def _me_se_delta(self,
                      scenario: PolicyScenario,
                      alternative: int,
                      attribute: str,
                      param: str) -> float:
        """
        Compute SE for marginal effect using delta method.

        ME = β * P * (1 - P)
        ∂ME/∂β = P * (1 - P)

        (Simplified: treats P as constant)
        """
        probs = self._compute_probabilities(scenario)
        p = probs[alternative]

        se_beta = self.result.std_errs.get(param, 0)

        # Gradient w.r.t. beta
        grad = p * (1 - p)

        se = abs(grad) * se_beta

        # Adjust for fee scale
        if attribute == 'fee':
            se = se / self.config.fee_scale

        return se

    def print_summary(self,
                      scenario: PolicyScenario,
                      attribute: str = 'fee') -> None:
        """Print formatted marginal effects summary."""
        print(f"\n{'='*60}")
        print(f"MARGINAL EFFECTS ANALYSIS: {attribute.upper()}")
        print(f"{'='*60}")
        print(f"Scenario: {scenario.name}")
        print()

        # Choice probabilities
        probs = self._compute_probabilities(scenario)
        print("Choice Probabilities:")
        for j in range(scenario.n_alternatives):
            print(f"  Alternative {j+1}: {probs[j]:.4f}")
        print()

        # Own marginal effects
        print("Own Marginal Effects (∂P_j/∂x_j):")
        for j in range(scenario.n_alternatives):
            me = self.marginal_effect_at_point(scenario, j, attribute)
            print(f"  Alt {j+1}: {me.me:>12.6f} (SE: {me.se:.6f})")
        print()

        # Interpretation
        print("Interpretation:")
        if attribute == 'fee':
            me = self.marginal_effect_at_point(scenario, 0, attribute)
            change = me.me * 100000  # Effect of 100,000 unit change
            print(f"  A 100,000 TL increase in Alt 1's fee")
            print(f"  changes its probability by {change:.4f}")
        else:
            me = self.marginal_effect_at_point(scenario, 0, attribute)
            print(f"  A 1-unit increase in Alt 1's {attribute}")
            print(f"  changes its probability by {me.me:.4f}")

        print()

        # Full matrix
        print("Marginal Effect Matrix:")
        matrix = self.marginal_effect_matrix(scenario, attribute)
        print(matrix.to_string(float_format=lambda x: f'{x:.6f}'))

        print(f"\n{'='*60}")


if __name__ == '__main__':
    print("Marginal Effects Module")
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

    calc = MarginalEffectCalculator(result)
    calc.print_summary(scenario, 'fee')
