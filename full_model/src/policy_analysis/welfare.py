"""
Welfare Analysis for DCM Models
===============================

Compute consumer surplus and compensating/equivalent variation
for welfare evaluation of policies.

Key Concepts:
- Consumer Surplus (CS): Monetary measure of utility
- Compensating Variation (CV): Amount needed to restore original utility after change
- Equivalent Variation (EV): Amount willing to pay to achieve new utility level

Logsum Formula:
    CS = (1/λ) * ln(Σ exp(V_j))

Where λ is the marginal utility of income (cost coefficient).
For our model: λ = -B_FEE * scale (note: B_FEE is negative)

UNCERTAINTY IN λ:
Standard errors for CS and CV are computed via Krinsky-Robb simulation,
which draws B_FEE (and other parameters) from their estimated distribution.
This PROPERLY propagates uncertainty in λ through the welfare calculations.

Welfare Change (CV):
    CV = CS_policy - CS_baseline

Authors: Hakan Mülayim, Giray Girengir, Ataol Azeritürk
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from scipy import stats

from .base import (
    EstimationResult, PolicyScenario, PolicyAnalysisConfig,
    compute_utilities, krinsky_robb_draws
)


@dataclass
class ConsumerSurplusResult:
    """
    Container for consumer surplus calculation results.

    Attributes:
        cs: Consumer surplus (in currency units)
        cs_se: Standard error
        ci_lower: Lower confidence bound
        ci_upper: Upper confidence bound
        logsum: Raw logsum value
        marginal_utility_income: λ (cost coefficient * scale)
        scenario_name: Name of scenario
    """
    cs: float
    cs_se: float
    ci_lower: float
    ci_upper: float
    logsum: float
    marginal_utility_income: float
    scenario_name: str

    def __str__(self) -> str:
        return (f"Consumer Surplus ({self.scenario_name}): "
                f"{self.cs:,.0f} (95% CI: [{self.ci_lower:,.0f}, {self.ci_upper:,.0f}])")


@dataclass
class WelfareChangeResult:
    """
    Results from welfare change analysis (CV/EV).

    Attributes:
        cv: Compensating variation (positive = welfare gain)
        cv_se: Standard error
        ci_lower: Lower confidence bound
        ci_upper: Upper confidence bound
        cs_baseline: Consumer surplus under baseline
        cs_policy: Consumer surplus under policy
        baseline_name: Baseline scenario name
        policy_name: Policy scenario name
        interpretation: Text interpretation
    """
    cv: float
    cv_se: float
    ci_lower: float
    ci_upper: float
    cs_baseline: float
    cs_policy: float
    baseline_name: str
    policy_name: str
    interpretation: str = None

    def __post_init__(self):
        if self.interpretation is None:
            if self.cv > 0:
                self.interpretation = f"Welfare GAIN of {abs(self.cv):,.0f}"
            elif self.cv < 0:
                self.interpretation = f"Welfare LOSS of {abs(self.cv):,.0f}"
            else:
                self.interpretation = "No welfare change"

    def __str__(self) -> str:
        return (f"Compensating Variation: {self.cv:,.0f} "
                f"(95% CI: [{self.ci_lower:,.0f}, {self.ci_upper:,.0f}])\n"
                f"  {self.interpretation}")


class WelfareAnalyzer:
    """
    Welfare analysis for DCM models.

    Computes consumer surplus and welfare changes from policy interventions
    using the logsum formula.

    Example:
        >>> analyzer = WelfareAnalyzer(result)
        >>> cs = analyzer.compute_consumer_surplus(scenario)
        >>> print(f"Consumer surplus: {cs.cs:,.0f}")
        >>>
        >>> cv = analyzer.compute_compensating_variation(baseline, policy)
        >>> print(f"Welfare change: {cv.cv:,.0f}")
    """

    def __init__(self,
                 result: Union[EstimationResult, Dict[str, Any]],
                 config: PolicyAnalysisConfig = None):
        """
        Initialize welfare analyzer.

        Args:
            result: Estimation results
            config: Configuration parameters
        """
        if isinstance(result, dict):
            self.result = EstimationResult.from_dict(result)
        else:
            self.result = result

        self.config = config or PolicyAnalysisConfig()

    def _compute_logsum(self, scenario: PolicyScenario) -> float:
        """
        Compute logsum for a scenario.

        Logsum = ln(Σ exp(V_j))

        Uses numerically stable computation to avoid overflow/underflow.
        """
        utilities = compute_utilities(scenario, self.result, self.config)

        # Numerically stable logsum computation
        # Step 1: Subtract max to prevent overflow in exp()
        max_u = np.max(utilities)

        # Step 2: Clip differences to prevent underflow (exp(-700) ≈ 0)
        # Values below -700 are effectively zero in double precision
        diffs = np.clip(utilities - max_u, -700, 0)

        # Step 3: Compute sum of exponentials and log
        sum_exp = np.sum(np.exp(diffs))

        if sum_exp <= 0:
            # This should never happen with clipping, but guard against it
            raise ValueError("Numerical error in logsum: sum of exp is non-positive")

        logsum = max_u + np.log(sum_exp)
        return logsum

    def _get_marginal_utility_income(self) -> float:
        """
        Get marginal utility of income (λ).

        λ = -B_FEE * scale

        Note: B_FEE is negative, so λ is positive.
        """
        b_fee = self.result.betas.get(self.config.fee_param, 0)
        return -b_fee * self.config.fee_scale

    def compute_consumer_surplus(self,
                                  scenario: PolicyScenario,
                                  with_uncertainty: bool = True,
                                  n_draws: int = None) -> ConsumerSurplusResult:
        """
        Compute consumer surplus for a scenario.

        CS = Logsum / λ = ln(Σ exp(V_j)) / (-B_FEE * scale)

        Args:
            scenario: Policy scenario
            with_uncertainty: Whether to compute confidence intervals
            n_draws: Number of simulation draws

        Returns:
            ConsumerSurplusResult with CS and uncertainty
        """
        logsum = self._compute_logsum(scenario)
        lambda_income = self._get_marginal_utility_income()

        # Check for near-zero marginal utility of income (would produce unreliable CS)
        if abs(lambda_income) < 1e-8:
            raise ValueError(
                f"Marginal utility of income (λ) is too close to zero: {lambda_income:.2e}. "
                f"This produces unreliable consumer surplus estimates. "
                f"Check that {self.config.fee_param} is significantly different from zero."
            )

        cs = logsum / lambda_income

        if not with_uncertainty:
            return ConsumerSurplusResult(
                cs=cs,
                cs_se=0,
                ci_lower=cs,
                ci_upper=cs,
                logsum=logsum,
                marginal_utility_income=lambda_income,
                scenario_name=scenario.name
            )

        # Simulation for uncertainty
        n = n_draws or self.config.n_simulations
        cs_draws = self._simulate_cs(scenario, n)

        cs_se = np.std(cs_draws, ddof=1)
        alpha = 1 - self.config.confidence_level
        ci_lower = np.percentile(cs_draws, alpha/2 * 100)
        ci_upper = np.percentile(cs_draws, (1 - alpha/2) * 100)

        return ConsumerSurplusResult(
            cs=cs,
            cs_se=cs_se,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            logsum=logsum,
            marginal_utility_income=lambda_income,
            scenario_name=scenario.name
        )

    def compute_compensating_variation(self,
                                        baseline: PolicyScenario,
                                        policy: PolicyScenario,
                                        n_draws: int = None) -> WelfareChangeResult:
        """
        Compute compensating variation between scenarios.

        CV = CS_policy - CS_baseline

        Positive CV indicates welfare gain from the policy.

        Args:
            baseline: Baseline scenario
            policy: Policy scenario
            n_draws: Simulation draws for uncertainty

        Returns:
            WelfareChangeResult with CV and interpretation
        """
        n = n_draws or self.config.n_simulations

        # Point estimates
        cs_baseline = self.compute_consumer_surplus(baseline, with_uncertainty=False)
        cs_policy = self.compute_consumer_surplus(policy, with_uncertainty=False)

        cv = cs_policy.cs - cs_baseline.cs

        # Simulation for uncertainty
        cv_draws = self._simulate_cv(baseline, policy, n)

        cv_se = np.std(cv_draws, ddof=1)
        alpha = 1 - self.config.confidence_level
        ci_lower = np.percentile(cv_draws, alpha/2 * 100)
        ci_upper = np.percentile(cv_draws, (1 - alpha/2) * 100)

        return WelfareChangeResult(
            cv=cv,
            cv_se=cv_se,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            cs_baseline=cs_baseline.cs,
            cs_policy=cs_policy.cs,
            baseline_name=baseline.name,
            policy_name=policy.name
        )

    def compute_equivalent_variation(self,
                                      baseline: PolicyScenario,
                                      policy: PolicyScenario,
                                      n_draws: int = None) -> WelfareChangeResult:
        """
        Compute equivalent variation.

        For the logit model, EV = CV (income effects are zero).

        Args:
            baseline: Baseline scenario
            policy: Policy scenario
            n_draws: Simulation draws

        Returns:
            WelfareChangeResult (same as CV for logit)
        """
        # For logit without income effects, EV = CV
        return self.compute_compensating_variation(baseline, policy, n_draws)

    def _simulate_cs(self, scenario: PolicyScenario, n_draws: int) -> np.ndarray:
        """
        Simulate consumer surplus by drawing parameters.
        """
        params = [self.config.fee_param, self.config.dur_param, self.config.asc_param]
        params = [p for p in params if p in self.result.betas]

        draws = krinsky_robb_draws(self.result, params, n_draws, self.config.seed)
        cs_draws = np.zeros(n_draws)

        for i in range(n_draws):
            temp_betas = self.result.betas.copy()
            for j, p in enumerate(params):
                temp_betas[p] = draws[i, j]

            temp_result = EstimationResult(
                betas=temp_betas,
                std_errs=self.result.std_errs
            )

            utilities = compute_utilities(scenario, temp_result, self.config)
            max_u = np.max(utilities)
            logsum = max_u + np.log(np.sum(np.exp(utilities - max_u)))

            lambda_income = -temp_betas[self.config.fee_param] * self.config.fee_scale
            if lambda_income != 0:
                cs_draws[i] = logsum / lambda_income
            else:
                cs_draws[i] = np.nan

        return cs_draws[~np.isnan(cs_draws)]

    def _simulate_cv(self,
                      baseline: PolicyScenario,
                      policy: PolicyScenario,
                      n_draws: int) -> np.ndarray:
        """
        Simulate compensating variation by drawing parameters.
        """
        params = [self.config.fee_param, self.config.dur_param, self.config.asc_param]
        params = [p for p in params if p in self.result.betas]

        draws = krinsky_robb_draws(self.result, params, n_draws, self.config.seed)
        cv_draws = np.zeros(n_draws)

        for i in range(n_draws):
            temp_betas = self.result.betas.copy()
            for j, p in enumerate(params):
                temp_betas[p] = draws[i, j]

            temp_result = EstimationResult(
                betas=temp_betas,
                std_errs=self.result.std_errs
            )

            # Compute logsums
            utils_base = compute_utilities(baseline, temp_result, self.config)
            utils_policy = compute_utilities(policy, temp_result, self.config)

            logsum_base = np.max(utils_base) + np.log(np.sum(np.exp(utils_base - np.max(utils_base))))
            logsum_policy = np.max(utils_policy) + np.log(np.sum(np.exp(utils_policy - np.max(utils_policy))))

            lambda_income = -temp_betas[self.config.fee_param] * self.config.fee_scale
            if lambda_income != 0:
                cv_draws[i] = (logsum_policy - logsum_base) / lambda_income
            else:
                cv_draws[i] = np.nan

        return cv_draws[~np.isnan(cv_draws)]

    def individual_welfare_change(self,
                                   sample_data: pd.DataFrame,
                                   attribute_changes: Dict[str, Dict[int, float]],
                                   aggregate: bool = True,
                                   weights: np.ndarray = None) -> Union[pd.DataFrame, WelfareChangeResult]:
        """
        Compute individual-level welfare changes.

        Args:
            sample_data: DataFrame with individual attributes
            attribute_changes: Changes to apply {attr: {alt: new_value}}
            aggregate: If True, return aggregated welfare change
            weights: Weights for aggregation

        Returns:
            DataFrame (individual) or WelfareChangeResult (aggregate)
        """
        individual_cvs = []

        for idx, row in sample_data.iterrows():
            # Build baseline scenario
            base_attrs = {}
            for attr_name in ['fee', 'dur']:
                vals = []
                for j in range(self.config.n_alternatives):
                    col = f'{attr_name}_{j+1}'
                    if col in row:
                        vals.append(row[col])
                    else:
                        vals.append(0)
                base_attrs[attr_name] = np.array(vals)

            baseline = PolicyScenario(
                name=f'baseline_{idx}',
                attributes=base_attrs,
                n_alternatives=self.config.n_alternatives
            )

            # Apply changes for policy
            policy_attrs = {k: v.copy() for k, v in base_attrs.items()}
            for attr, alt_changes in attribute_changes.items():
                for alt, new_val in alt_changes.items():
                    policy_attrs[attr][alt] = new_val

            policy = PolicyScenario(
                name=f'policy_{idx}',
                attributes=policy_attrs,
                n_alternatives=self.config.n_alternatives
            )

            cv = self.compute_compensating_variation(baseline, policy, n_draws=100)
            individual_cvs.append(cv.cv)

        cvs = np.array(individual_cvs)

        if not aggregate:
            df = sample_data.copy()
            df['cv'] = cvs
            return df

        # Aggregate
        if weights is None:
            weights = np.ones(len(cvs)) / len(cvs)
        else:
            weights = weights / np.sum(weights)

        agg_cv = np.sum(weights * cvs)
        cv_se = np.sqrt(np.sum(weights**2 * np.var(cvs, ddof=1)))

        z = self.config.z_score

        return WelfareChangeResult(
            cv=agg_cv,
            cv_se=cv_se,
            ci_lower=agg_cv - z * cv_se,
            ci_upper=agg_cv + z * cv_se,
            cs_baseline=np.nan,  # Not computed for aggregate
            cs_policy=np.nan,
            baseline_name='aggregate_baseline',
            policy_name='aggregate_policy'
        )

    def total_welfare_change(self,
                              baseline: PolicyScenario,
                              policy: PolicyScenario,
                              population: int,
                              n_draws: int = None) -> Dict[str, float]:
        """
        Compute total welfare change for a population.

        Args:
            baseline: Baseline scenario
            policy: Policy scenario
            population: Number of affected individuals
            n_draws: Simulation draws

        Returns:
            Dict with per-person and total welfare changes
        """
        cv_result = self.compute_compensating_variation(baseline, policy, n_draws)

        total_welfare = cv_result.cv * population
        total_ci_lower = cv_result.ci_lower * population
        total_ci_upper = cv_result.ci_upper * population

        return {
            'per_person_cv': cv_result.cv,
            'per_person_se': cv_result.cv_se,
            'per_person_ci': (cv_result.ci_lower, cv_result.ci_upper),
            'population': population,
            'total_welfare_change': total_welfare,
            'total_ci': (total_ci_lower, total_ci_upper),
            'interpretation': cv_result.interpretation
        }

    def print_summary(self,
                      baseline: PolicyScenario,
                      policy: PolicyScenario,
                      population: int = None) -> None:
        """Print formatted welfare analysis summary."""
        print(f"\n{'='*60}")
        print("WELFARE ANALYSIS")
        print(f"{'='*60}")

        # Consumer surplus for each scenario
        cs_base = self.compute_consumer_surplus(baseline)
        cs_policy = self.compute_consumer_surplus(policy)

        print(f"\nConsumer Surplus:")
        print(f"  {baseline.name}: {cs_base.cs:>12,.0f} "
              f"(95% CI: [{cs_base.ci_lower:,.0f}, {cs_base.ci_upper:,.0f}])")
        print(f"  {policy.name}:  {cs_policy.cs:>12,.0f} "
              f"(95% CI: [{cs_policy.ci_lower:,.0f}, {cs_policy.ci_upper:,.0f}])")

        # Compensating variation
        cv = self.compute_compensating_variation(baseline, policy)

        print(f"\nCompensating Variation (Welfare Change):")
        print(f"  CV = CS_policy - CS_baseline")
        print(f"  CV = {cv.cv:>12,.0f}")
        print(f"  95% CI: [{cv.ci_lower:,.0f}, {cv.ci_upper:,.0f}]")
        print(f"\n  {cv.interpretation}")

        # Statistical significance
        if cv.cv_se > 0:
            t_stat = cv.cv / cv.cv_se
            p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
            sig = "Yes" if p_value < 0.05 else "No"
            print(f"\n  t-statistic: {t_stat:.2f}")
            print(f"  p-value: {p_value:.4f}")
            print(f"  Statistically significant at 5%: {sig}")

        # Total welfare if population given
        if population is not None:
            total = self.total_welfare_change(baseline, policy, population)
            print(f"\nTotal Welfare Impact (Population: {population:,}):")
            print(f"  Total change: {total['total_welfare_change']:>15,.0f}")
            print(f"  95% CI: [{total['total_ci'][0]:,.0f}, {total['total_ci'][1]:,.0f}]")

        print(f"\n{'='*60}")


if __name__ == '__main__':
    print("Welfare Analysis Module")
    print("=" * 40)

    # Example
    result = EstimationResult(
        betas={'B_FEE': -0.5, 'B_DUR': -0.08, 'ASC_paid': 1.2},
        std_errs={'B_FEE': 0.05, 'B_DUR': 0.02, 'ASC_paid': 0.15}
    )

    baseline = PolicyScenario(
        name='Current',
        attributes={
            'fee': np.array([500000, 600000, 0]),
            'dur': np.array([10, 8, 15])
        }
    )

    policy = PolicyScenario(
        name='Fee Cut',
        attributes={
            'fee': np.array([400000, 600000, 0]),
            'dur': np.array([10, 8, 15])
        }
    )

    analyzer = WelfareAnalyzer(result)
    analyzer.print_summary(baseline, policy, population=10000)
