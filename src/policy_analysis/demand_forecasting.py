"""
Demand Forecasting and Market Share Prediction
==============================================

Predict market shares under different policy scenarios and
compare baseline vs. policy outcomes.

Key Capabilities:
- Predict choice probabilities (market shares)
- Compare scenarios (baseline vs policy)
- Simulate share uncertainty via parameter draws
- Individual-level predictions with aggregation

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
    krinsky_robb_draws
)


@dataclass
class MarketShareResult:
    """
    Container for market share prediction results.

    Attributes:
        shares: Array of predicted market shares
        share_se: Standard errors for each share
        share_ci_lower: Lower confidence bounds
        share_ci_upper: Upper confidence bounds
        scenario_name: Name of the scenario
        n_alternatives: Number of alternatives
    """
    shares: np.ndarray
    share_se: np.ndarray
    share_ci_lower: np.ndarray
    share_ci_upper: np.ndarray
    scenario_name: str
    n_alternatives: int

    def __str__(self) -> str:
        lines = [f"Market Shares ({self.scenario_name}):"]
        for j in range(self.n_alternatives):
            lines.append(
                f"  Alt {j+1}: {self.shares[j]:.1%} "
                f"[{self.share_ci_lower[j]:.1%}, {self.share_ci_upper[j]:.1%}]"
            )
        return '\n'.join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame."""
        return pd.DataFrame({
            'alternative': [f'Alt {j+1}' for j in range(self.n_alternatives)],
            'share': self.shares,
            'se': self.share_se,
            'ci_lower': self.share_ci_lower,
            'ci_upper': self.share_ci_upper
        })


@dataclass
class ScenarioComparisonResult:
    """
    Results from comparing baseline vs policy scenarios.

    Attributes:
        baseline_shares: Market shares under baseline
        policy_shares: Market shares under policy
        share_changes: Absolute change (policy - baseline)
        percent_changes: Percentage change
        baseline_name: Baseline scenario name
        policy_name: Policy scenario name
    """
    baseline_shares: np.ndarray
    policy_shares: np.ndarray
    share_changes: np.ndarray
    percent_changes: np.ndarray
    change_se: np.ndarray
    change_ci_lower: np.ndarray
    change_ci_upper: np.ndarray
    baseline_name: str
    policy_name: str

    def __str__(self) -> str:
        n = len(self.baseline_shares)
        lines = [
            f"Scenario Comparison: {self.baseline_name} → {self.policy_name}",
            "-" * 50
        ]
        for j in range(n):
            lines.append(
                f"Alt {j+1}: {self.baseline_shares[j]:.1%} → {self.policy_shares[j]:.1%} "
                f"({self.share_changes[j]:+.1%}, {self.percent_changes[j]:+.1f}%)"
            )
        return '\n'.join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame."""
        n = len(self.baseline_shares)
        return pd.DataFrame({
            'alternative': [f'Alt {j+1}' for j in range(n)],
            'baseline_share': self.baseline_shares,
            'policy_share': self.policy_shares,
            'change': self.share_changes,
            'percent_change': self.percent_changes,
            'change_se': self.change_se,
            'change_ci_lower': self.change_ci_lower,
            'change_ci_upper': self.change_ci_upper
        })


class DemandForecaster:
    """
    Demand forecasting and market share prediction for DCM models.

    Predicts choice probabilities under different scenarios and
    quantifies uncertainty in predictions.

    Example:
        >>> forecaster = DemandForecaster(result)
        >>> baseline = PolicyScenario(name='Current', attributes={...})
        >>> policy = PolicyScenario(name='Fee Cut', attributes={...})
        >>> comparison = forecaster.compare_scenarios(baseline, policy)
        >>> print(comparison)
    """

    def __init__(self,
                 result: Union[EstimationResult, Dict[str, Any]],
                 config: PolicyAnalysisConfig = None):
        """
        Initialize demand forecaster.

        Args:
            result: Estimation results
            config: Configuration parameters
        """
        if isinstance(result, dict):
            self.result = EstimationResult.from_dict(result)
        else:
            self.result = result

        self.config = config or PolicyAnalysisConfig()

    def predict_market_shares(self,
                               scenario: PolicyScenario,
                               with_uncertainty: bool = True,
                               n_draws: int = None) -> MarketShareResult:
        """
        Predict market shares for a scenario.

        Computes choice probabilities using the estimated model.
        Uncertainty is quantified via simulation.

        Args:
            scenario: Policy scenario with attribute levels
            with_uncertainty: Whether to compute confidence intervals
            n_draws: Number of simulation draws for uncertainty

        Returns:
            MarketShareResult with predicted shares and CIs
        """
        # Point estimate
        utilities = compute_utilities(scenario, self.result, self.config)
        shares = compute_logit_probabilities(utilities)

        n_alts = scenario.n_alternatives

        if not with_uncertainty:
            return MarketShareResult(
                shares=shares,
                share_se=np.zeros(n_alts),
                share_ci_lower=shares,
                share_ci_upper=shares,
                scenario_name=scenario.name,
                n_alternatives=n_alts
            )

        # Simulation for uncertainty
        n = n_draws or self.config.n_simulations
        simulated_shares = self._simulate_shares(scenario, n)

        share_se = np.std(simulated_shares, axis=0, ddof=1)
        alpha = 1 - self.config.confidence_level
        share_ci_lower = np.percentile(simulated_shares, alpha/2 * 100, axis=0)
        share_ci_upper = np.percentile(simulated_shares, (1 - alpha/2) * 100, axis=0)

        return MarketShareResult(
            shares=shares,
            share_se=share_se,
            share_ci_lower=share_ci_lower,
            share_ci_upper=share_ci_upper,
            scenario_name=scenario.name,
            n_alternatives=n_alts
        )

    def compare_scenarios(self,
                          baseline: PolicyScenario,
                          policy: PolicyScenario,
                          n_draws: int = None) -> ScenarioComparisonResult:
        """
        Compare market shares between baseline and policy scenarios.

        Computes:
        - Share changes (absolute)
        - Percent changes
        - Uncertainty in changes

        Args:
            baseline: Baseline scenario
            policy: Policy scenario
            n_draws: Simulation draws for uncertainty

        Returns:
            ScenarioComparisonResult with comparison metrics
        """
        n = n_draws or self.config.n_simulations

        # Point estimates
        baseline_shares = self.predict_market_shares(baseline, with_uncertainty=False).shares
        policy_shares = self.predict_market_shares(policy, with_uncertainty=False).shares

        share_changes = policy_shares - baseline_shares
        percent_changes = np.where(
            baseline_shares > 0,
            share_changes / baseline_shares * 100,
            np.nan
        )

        # Simulate changes for uncertainty
        baseline_sims = self._simulate_shares(baseline, n)
        policy_sims = self._simulate_shares(policy, n)
        change_sims = policy_sims - baseline_sims

        change_se = np.std(change_sims, axis=0, ddof=1)
        alpha = 1 - self.config.confidence_level
        change_ci_lower = np.percentile(change_sims, alpha/2 * 100, axis=0)
        change_ci_upper = np.percentile(change_sims, (1 - alpha/2) * 100, axis=0)

        return ScenarioComparisonResult(
            baseline_shares=baseline_shares,
            policy_shares=policy_shares,
            share_changes=share_changes,
            percent_changes=percent_changes,
            change_se=change_se,
            change_ci_lower=change_ci_lower,
            change_ci_upper=change_ci_upper,
            baseline_name=baseline.name,
            policy_name=policy.name
        )

    def _simulate_shares(self,
                          scenario: PolicyScenario,
                          n_draws: int) -> np.ndarray:
        """
        Simulate market shares by drawing parameters.

        Args:
            scenario: Policy scenario
            n_draws: Number of draws

        Returns:
            Array of shape (n_draws, n_alternatives)
        """
        # Get parameters that affect utilities
        params = [self.config.fee_param, self.config.dur_param, self.config.asc_param]
        params = [p for p in params if p in self.result.betas]

        draws = krinsky_robb_draws(self.result, params, n_draws, self.config.seed)

        simulated = np.zeros((n_draws, scenario.n_alternatives))

        for i in range(n_draws):
            # Create result with drawn parameters
            temp_betas = self.result.betas.copy()
            for j, p in enumerate(params):
                temp_betas[p] = draws[i, j]

            temp_result = EstimationResult(
                betas=temp_betas,
                std_errs=self.result.std_errs
            )

            utilities = compute_utilities(scenario, temp_result, self.config)
            simulated[i] = compute_logit_probabilities(utilities)

        return simulated

    def predict_individual_shares(self,
                                   sample_data: pd.DataFrame,
                                   aggregate: bool = True,
                                   weights: np.ndarray = None) -> Union[pd.DataFrame, MarketShareResult]:
        """
        Predict market shares for each individual in sample.

        Args:
            sample_data: DataFrame with individual-level attributes
                        Expected columns: fee_1, fee_2, fee_3, dur_1, dur_2, dur_3, etc.
            aggregate: If True, return aggregated shares; if False, return individual
            weights: Optional weights for aggregation

        Returns:
            MarketShareResult (aggregate) or DataFrame (individual)
        """
        individual_shares = []

        for idx, row in sample_data.iterrows():
            # Build scenario for this individual
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

            scenario = PolicyScenario(
                name=f'individual_{idx}',
                attributes=attrs,
                n_alternatives=self.config.n_alternatives
            )

            utilities = compute_utilities(scenario, self.result, self.config)
            shares = compute_logit_probabilities(utilities)
            individual_shares.append(shares)

        shares_array = np.array(individual_shares)

        if not aggregate:
            df = sample_data.copy()
            for j in range(self.config.n_alternatives):
                df[f'predicted_share_{j+1}'] = shares_array[:, j]
            return df

        # Aggregate
        if weights is None:
            weights = np.ones(len(shares_array)) / len(shares_array)
        else:
            weights = weights / np.sum(weights)

        agg_shares = np.sum(weights[:, np.newaxis] * shares_array, axis=0)
        share_se = np.sqrt(np.sum(
            weights[:, np.newaxis]**2 * np.var(shares_array, axis=0),
            axis=0
        ))

        z = self.config.z_score

        return MarketShareResult(
            shares=agg_shares,
            share_se=share_se,
            share_ci_lower=agg_shares - z * share_se,
            share_ci_upper=agg_shares + z * share_se,
            scenario_name='aggregate',
            n_alternatives=self.config.n_alternatives
        )

    def sensitivity_analysis(self,
                              base_scenario: PolicyScenario,
                              attribute: str,
                              alternative: int,
                              pct_changes: List[float] = [-20, -10, 0, 10, 20]) -> pd.DataFrame:
        """
        Analyze sensitivity of market shares to attribute changes.

        Args:
            base_scenario: Base scenario
            attribute: Attribute to vary
            alternative: Alternative to modify
            pct_changes: Percentage changes to test

        Returns:
            DataFrame with shares at each change level
        """
        results = []
        base_value = base_scenario.get_attribute(attribute, alternative)

        for pct in pct_changes:
            new_value = base_value * (1 + pct / 100)
            modified = base_scenario.with_modification(attribute, alternative, new_value)
            shares = self.predict_market_shares(modified, with_uncertainty=False).shares

            row = {
                'pct_change': pct,
                f'{attribute}_value': new_value
            }
            for j in range(len(shares)):
                row[f'share_alt_{j+1}'] = shares[j]

            results.append(row)

        return pd.DataFrame(results)

    def what_if_analysis(self,
                          base_scenario: PolicyScenario,
                          changes: Dict[str, Dict[int, float]]) -> ScenarioComparisonResult:
        """
        Perform what-if analysis with multiple simultaneous changes.

        Args:
            base_scenario: Base scenario
            changes: Dict mapping attribute -> {alternative: new_value}
                    e.g., {'fee': {0: 400000}, 'dur': {1: 6}}

        Returns:
            ScenarioComparisonResult comparing base to modified scenario
        """
        # Apply all changes
        modified_attrs = {k: v.copy() for k, v in base_scenario.attributes.items()}

        for attr, alt_changes in changes.items():
            for alt, new_val in alt_changes.items():
                modified_attrs[attr][alt] = new_val

        policy = PolicyScenario(
            name='what_if',
            attributes=modified_attrs,
            n_alternatives=base_scenario.n_alternatives
        )

        return self.compare_scenarios(base_scenario, policy)

    def print_summary(self,
                      scenario: PolicyScenario,
                      title: str = None) -> None:
        """Print formatted market share summary."""
        result = self.predict_market_shares(scenario)

        print(f"\n{'='*60}")
        print(f"MARKET SHARE FORECAST")
        if title:
            print(f"  {title}")
        print(f"{'='*60}")
        print(f"Scenario: {scenario.name}")
        print()

        # Show attribute levels
        print("Attribute Levels:")
        for attr, values in scenario.attributes.items():
            vals_str = ', '.join([f'{v:,.0f}' for v in values])
            print(f"  {attr}: [{vals_str}]")
        print()

        # Show shares
        print("Predicted Market Shares:")
        conf_pct = int(self.config.confidence_level * 100)
        print(f"{'Alt':>8} {'Share':>10} {'SE':>10} {f'{conf_pct}% CI':>20}")
        print("-" * 50)
        for j in range(result.n_alternatives):
            ci = f"[{result.share_ci_lower[j]:.1%}, {result.share_ci_upper[j]:.1%}]"
            print(f"{'Alt '+str(j+1):>8} {result.shares[j]:>10.1%} "
                  f"{result.share_se[j]:>10.3f} {ci:>20}")

        print(f"\n{'='*60}")

    def print_comparison(self,
                         baseline: PolicyScenario,
                         policy: PolicyScenario) -> None:
        """Print formatted scenario comparison."""
        result = self.compare_scenarios(baseline, policy)

        print(f"\n{'='*60}")
        print("SCENARIO COMPARISON")
        print(f"{'='*60}")
        print(f"Baseline: {baseline.name}")
        print(f"Policy:   {policy.name}")
        print()

        print("Changes in Attribute Levels:")
        for attr in baseline.attributes.keys():
            base_vals = baseline.attributes[attr]
            policy_vals = policy.attributes[attr]
            for j in range(len(base_vals)):
                if base_vals[j] != policy_vals[j]:
                    print(f"  {attr} Alt {j+1}: {base_vals[j]:,.0f} → {policy_vals[j]:,.0f}")
        print()

        print("Market Share Changes:")
        conf_pct = int(self.config.confidence_level * 100)
        print(f"{'Alt':>8} {'Baseline':>10} {'Policy':>10} {'Change':>10} {'%Change':>10} {f'{conf_pct}% CI':>20}")
        print("-" * 70)

        n = len(result.baseline_shares)
        for j in range(n):
            ci = f"[{result.change_ci_lower[j]:+.1%}, {result.change_ci_upper[j]:+.1%}]"
            print(f"{'Alt '+str(j+1):>8} {result.baseline_shares[j]:>10.1%} "
                  f"{result.policy_shares[j]:>10.1%} "
                  f"{result.share_changes[j]:>+10.1%} "
                  f"{result.percent_changes[j]:>+10.1f}% {ci:>20}")

        print(f"\n{'='*60}")


if __name__ == '__main__':
    print("Demand Forecasting Module")
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
            'fee': np.array([400000, 600000, 0]),  # 20% cut on Alt 1
            'dur': np.array([10, 8, 15])
        }
    )

    forecaster = DemandForecaster(result)
    forecaster.print_summary(baseline)
    forecaster.print_comparison(baseline, policy)
