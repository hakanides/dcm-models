"""
Willingness-to-Pay Calculations
===============================

Compute WTP estimates with uncertainty quantification for DCM models.

WTP represents the marginal rate of substitution between attributes -
how much money a decision-maker is willing to pay to improve an attribute.

Key Formula:
    WTP = -β_attribute / β_cost * scale_factor

For duration (days saved):
    WTP_duration = -B_DUR / B_FEE * fee_scale
                 = -B_DUR * 10000 / B_FEE  (in currency units per day)

Methods:
- Delta method: Analytical variance using gradient and covariance
- Krinsky-Robb: Simulation-based confidence intervals
- Individual WTP: For models with heterogeneity (demographics, latent variables)
- MXL distributions: For random coefficient models

Author: DCM Research Team
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from scipy import stats

from .base import (
    EstimationResult, PolicyAnalysisConfig,
    delta_method_se, krinsky_robb_draws
)


@dataclass
class WTPResult:
    """
    Container for WTP calculation results.

    Attributes:
        wtp_point: Point estimate of WTP
        wtp_se: Standard error (delta method)
        ci_lower: Lower bound of confidence interval
        ci_upper: Upper bound of confidence interval
        numerator_param: Name of numerator parameter
        denominator_param: Name of denominator (cost) parameter
        method: Calculation method used
        scale_factor: Scale factor applied
        t_stat: t-statistic for WTP
        p_value: Two-sided p-value
    """
    wtp_point: float
    wtp_se: float
    ci_lower: float
    ci_upper: float
    numerator_param: str
    denominator_param: str
    method: str = 'delta'
    scale_factor: float = 10000.0
    t_stat: float = None
    p_value: float = None

    def __post_init__(self):
        if self.t_stat is None and self.wtp_se and self.wtp_se > 0:
            self.t_stat = self.wtp_point / self.wtp_se
        if self.p_value is None and self.t_stat is not None:
            self.p_value = 2 * (1 - stats.norm.cdf(abs(self.t_stat)))

    def __str__(self) -> str:
        return (
            f"WTP = {self.wtp_point:,.0f} "
            f"(SE: {self.wtp_se:,.0f}, "
            f"95% CI: [{self.ci_lower:,.0f}, {self.ci_upper:,.0f}])"
        )


@dataclass
class WTPDistributionResult:
    """Results for WTP distribution (MXL models)."""
    wtp_mean: float
    wtp_std: float
    wtp_median: float
    wtp_percentiles: Dict[int, float]  # {5: val, 25: val, 50: val, 75: val, 95: val}
    positive_proportion: float  # Proportion with WTP > 0
    draws: np.ndarray  # Raw WTP draws for further analysis


class WTPCalculator:
    """
    Willingness-to-Pay calculator for DCM models.

    Computes WTP as ratio of coefficients with proper uncertainty
    quantification using delta method or simulation.

    Example:
        >>> result = EstimationResult(
        ...     betas={'B_FEE': -0.5, 'B_DUR': -0.08},
        ...     std_errs={'B_FEE': 0.05, 'B_DUR': 0.02}
        ... )
        >>> calc = WTPCalculator(result)
        >>> wtp = calc.compute_wtp('B_DUR')
        >>> print(f"WTP: {wtp.wtp_point:,.0f} TL/day")
    """

    def __init__(self,
                 result: Union[EstimationResult, Dict[str, Any]],
                 config: PolicyAnalysisConfig = None):
        """
        Initialize WTP calculator.

        Args:
            result: Estimation results (EstimationResult or dict)
            config: Configuration (uses defaults if not provided)
        """
        if isinstance(result, dict):
            self.result = EstimationResult.from_dict(result)
        else:
            self.result = result

        self.config = config or PolicyAnalysisConfig()

    def compute_wtp(self,
                    numerator_param: str = None,
                    denominator_param: str = None,
                    scale_factor: float = None,
                    for_improvement: bool = True) -> WTPResult:
        """
        Compute WTP using delta method for standard errors.

        Standard formula: WTP = -β_num / β_denom * scale

        This gives WTP for a one-unit INCREASE in the attribute.
        For "bad" attributes (negative β, like duration where more is worse),
        this produces negative WTP (people wouldn't pay for more duration).

        When for_improvement=True (default), the sign is adjusted so that
        WTP is POSITIVE for beneficial changes:
        - For bad attributes (β < 0): WTP for REDUCTION (e.g., days saved)
        - For good attributes (β > 0): WTP for INCREASE

        The delta method computes variance as:
        Var(WTP) = g' * Cov(β) * g

        where g = [∂WTP/∂β_num, ∂WTP/∂β_denom]
              = [-scale/β_denom, β_num*scale/β_denom²]

        Args:
            numerator_param: Parameter for attribute (default: B_DUR)
            denominator_param: Cost parameter (default: B_FEE)
            scale_factor: Scale to apply (default: fee_scale from config)
            for_improvement: If True (default), returns WTP for attribute
                           improvement (positive WTP for beneficial changes).
                           If False, returns raw MRS formula result.

        Returns:
            WTPResult with point estimate and confidence interval
        """
        num_param = numerator_param or self.config.dur_param
        denom_param = denominator_param or self.config.fee_param
        scale = scale_factor if scale_factor is not None else self.config.fee_scale

        # Get coefficients
        beta_num = self.result.betas.get(num_param, 0)
        beta_denom = self.result.betas.get(denom_param, 0)

        if beta_denom == 0:
            raise ValueError(f"Denominator parameter {denom_param} is zero")

        # Base formula: WTP = -β_num / β_denom * scale
        # This is WTP for a one-unit INCREASE in the attribute
        wtp = -beta_num / beta_denom * scale

        # For "bad" attributes (negative β), flip sign to get WTP for improvement
        # E.g., B_DUR < 0 means more duration is bad, so WTP for reduction is -wtp
        if for_improvement and beta_num < 0:
            wtp = -wtp

        # Gradient for delta method (before sign adjustment)
        # g = [∂WTP/∂β_num, ∂WTP/∂β_denom]
        grad = np.array([
            -scale / beta_denom,                    # ∂WTP/∂β_num
            beta_num * scale / (beta_denom ** 2)    # ∂WTP/∂β_denom
        ])

        # Get covariance matrix
        params = [num_param, denom_param]
        cov = self.result.get_covariance(params)

        # Standard error via delta method (SE is always positive)
        se = delta_method_se(wtp, grad, cov)

        # Confidence interval
        z = self.config.z_score
        ci_lower = wtp - z * se
        ci_upper = wtp + z * se

        return WTPResult(
            wtp_point=wtp,
            wtp_se=se,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            numerator_param=num_param,
            denominator_param=denom_param,
            method='delta',
            scale_factor=scale
        )

    def compute_wtp_fieller(self,
                            numerator_param: str = None,
                            denominator_param: str = None,
                            scale_factor: float = None,
                            for_improvement: bool = True) -> WTPResult:
        """
        Compute WTP using Fieller's method for ratio confidence intervals.

        ISSUE #18 FIX: Normal approximation (delta method) is inappropriate for
        ratio estimators like WTP because the ratio of two normals follows a
        Cauchy-like distribution with heavy tails.

        Fieller's method provides exact confidence intervals by solving:
            (β_num - θ*β_denom)² = t²_α * Var(β_num - θ*β_denom)

        This yields a quadratic in θ with two solutions giving CI bounds.

        Advantages over delta method:
        - Handles cases where denominator is near zero
        - Provides asymmetric intervals reflecting true ratio distribution
        - Can detect unbounded CIs (when denominator is insignificant)

        Args:
            numerator_param: Parameter for attribute
            denominator_param: Cost parameter
            scale_factor: Scale to apply
            for_improvement: If True, adjust sign for beneficial changes

        Returns:
            WTPResult with Fieller confidence interval
        """
        num_param = numerator_param or self.config.dur_param
        denom_param = denominator_param or self.config.fee_param
        scale = scale_factor if scale_factor is not None else self.config.fee_scale

        # Get coefficients and standard errors
        beta_num = self.result.betas.get(num_param, 0)
        beta_denom = self.result.betas.get(denom_param, 0)
        se_num = self.result.std_errs.get(num_param, 0)
        se_denom = self.result.std_errs.get(denom_param, 0)

        if beta_denom == 0:
            raise ValueError(f"Denominator parameter {denom_param} is zero")

        # Get covariance
        params = [num_param, denom_param]
        cov_matrix = self.result.get_covariance(params)
        cov_num_denom = cov_matrix[0, 1] if cov_matrix.size > 1 else 0

        # Point estimate
        wtp_raw = -beta_num / beta_denom * scale

        # Fieller's method: solve quadratic for θ where
        # (β_num - θ*β_denom)² = t² * (σ²_num - 2θ*σ_num_denom + θ²*σ²_denom)
        # Rearrange: (β²_denom - t²*σ²_denom)*θ² - 2*(β_num*β_denom - t²*σ_num_denom)*θ
        #            + (β²_num - t²*σ²_num) = 0

        t_crit = self.config.z_score  # Using z for large samples
        var_num = se_num ** 2
        var_denom = se_denom ** 2

        # Quadratic coefficients: a*θ² + b*θ + c = 0
        a = beta_denom**2 - t_crit**2 * var_denom
        b = -2 * (beta_num * beta_denom - t_crit**2 * cov_num_denom)
        c = beta_num**2 - t_crit**2 * var_num

        # Check if denominator is significantly different from zero
        # If a ≤ 0, the CI is unbounded
        discriminant = b**2 - 4*a*c

        if a <= 0:
            # Denominator not significantly different from zero
            # CI is unbounded - fall back to delta method with warning
            print(f"  ⚠️  Fieller: denominator coefficient not significantly different from zero")
            print(f"     CI may be unbounded. Using delta method instead.")
            return self.compute_wtp(num_param, denom_param, scale_factor, for_improvement)

        if discriminant < 0:
            # No real solutions - use delta method
            return self.compute_wtp(num_param, denom_param, scale_factor, for_improvement)

        # Solve quadratic
        sqrt_disc = np.sqrt(discriminant)
        theta_lower = (-b - sqrt_disc) / (2 * a)
        theta_upper = (-b + sqrt_disc) / (2 * a)

        # Apply scale (the quadratic was for unscaled β_num/β_denom)
        # Actually, we need to redo with scaled version
        # WTP = -β_num/β_denom * scale, so θ = -β_num/β_denom
        # CI bounds are for θ, multiply by -scale to get WTP

        ci_lower = -theta_upper * scale  # Note: signs flip due to negative
        ci_upper = -theta_lower * scale

        # Ensure lower < upper
        if ci_lower > ci_upper:
            ci_lower, ci_upper = ci_upper, ci_lower

        wtp = wtp_raw
        if for_improvement and beta_num < 0:
            wtp = -wtp
            # Also flip CI
            ci_lower, ci_upper = -ci_upper, -ci_lower

        # Approximate SE from CI width
        se = (ci_upper - ci_lower) / (2 * self.config.z_score)

        return WTPResult(
            wtp_point=wtp,
            wtp_se=se,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            numerator_param=num_param,
            denominator_param=denom_param,
            method='fieller',
            scale_factor=scale
        )

    def compute_wtp_krinsky_robb(self,
                                  numerator_param: str = None,
                                  denominator_param: str = None,
                                  scale_factor: float = None,
                                  n_draws: int = None,
                                  for_improvement: bool = True) -> WTPResult:
        """
        Compute WTP using Krinsky-Robb simulation method.

        Draws parameter vectors from multivariate normal distribution
        and computes WTP for each draw to build empirical distribution.

        More robust than delta method when:
        - WTP distribution is skewed
        - Denominator coefficient is close to zero
        - Sample size is small

        Args:
            numerator_param: Parameter for attribute
            denominator_param: Cost parameter
            scale_factor: Scale to apply
            n_draws: Number of simulation draws
            for_improvement: If True (default), returns WTP for attribute
                           improvement (positive WTP for beneficial changes).

        Returns:
            WTPResult with simulation-based confidence interval
        """
        num_param = numerator_param or self.config.dur_param
        denom_param = denominator_param or self.config.fee_param
        scale = scale_factor if scale_factor is not None else self.config.fee_scale
        n = n_draws or self.config.n_simulations

        # Get parameter draws
        params = [num_param, denom_param]
        draws = krinsky_robb_draws(self.result, params, n, self.config.seed)

        # Compute WTP for each draw
        # WTP = -β_num / β_denom * scale
        beta_num_draws = draws[:, 0]
        beta_denom_draws = draws[:, 1]

        # Filter out draws where denominator is zero or near-zero
        valid_mask = np.abs(beta_denom_draws) > 1e-10
        wtp_draws = np.where(
            valid_mask,
            -beta_num_draws / beta_denom_draws * scale,
            np.nan
        )
        wtp_draws = wtp_draws[~np.isnan(wtp_draws)]

        if len(wtp_draws) == 0:
            raise ValueError("All simulation draws resulted in invalid WTP")

        # For "bad" attributes (mean β < 0), flip sign for improvement WTP
        beta_num_mean = self.result.betas.get(num_param, 0)
        if for_improvement and beta_num_mean < 0:
            wtp_draws = -wtp_draws

        # Statistics from empirical distribution
        wtp_point = np.mean(wtp_draws)
        wtp_se = np.std(wtp_draws, ddof=1)

        # Percentile-based confidence interval
        alpha = 1 - self.config.confidence_level
        ci_lower = np.percentile(wtp_draws, alpha/2 * 100)
        ci_upper = np.percentile(wtp_draws, (1 - alpha/2) * 100)

        return WTPResult(
            wtp_point=wtp_point,
            wtp_se=wtp_se,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            numerator_param=num_param,
            denominator_param=denom_param,
            method='krinsky_robb',
            scale_factor=scale
        )

    def compute_all_wtp(self,
                        attribute_params: List[str] = None,
                        cost_param: str = None) -> pd.DataFrame:
        """
        Compute WTP for multiple attributes.

        Args:
            attribute_params: List of attribute parameters
            cost_param: Cost parameter (default: B_FEE)

        Returns:
            DataFrame with WTP results for each attribute
        """
        if attribute_params is None:
            # Default: find all beta parameters except cost
            cost = cost_param or self.config.fee_param
            attribute_params = [
                p for p in self.result.betas.keys()
                if p != cost and not p.startswith('ASC')
            ]

        results = []
        for param in attribute_params:
            try:
                wtp = self.compute_wtp(numerator_param=param, denominator_param=cost_param)
                results.append({
                    'attribute': param,
                    'wtp': wtp.wtp_point,
                    'se': wtp.wtp_se,
                    'ci_lower': wtp.ci_lower,
                    'ci_upper': wtp.ci_upper,
                    't_stat': wtp.t_stat,
                    'p_value': wtp.p_value
                })
            except Exception as e:
                results.append({
                    'attribute': param,
                    'wtp': np.nan,
                    'se': np.nan,
                    'ci_lower': np.nan,
                    'ci_upper': np.nan,
                    't_stat': np.nan,
                    'p_value': np.nan
                })

        return pd.DataFrame(results)

    def compute_individual_wtp(self,
                                demographics: pd.DataFrame,
                                interaction_params: Dict[str, str],
                                base_num_param: str = None,
                                cost_param: str = None) -> pd.DataFrame:
        """
        Compute individual-specific WTP for models with demographic interactions.

        For models like: β_dur = β_dur_base + β_dur_income * income

        Individual WTP = -(β_dur_base + β_dur_income * income_i) / β_fee * scale

        Args:
            demographics: DataFrame with individual demographics
            interaction_params: Dict mapping demographic column to interaction parameter
                              e.g., {'income': 'B_DUR_INCOME', 'age': 'B_DUR_AGE'}
            base_num_param: Base numerator parameter
            cost_param: Cost parameter

        Returns:
            DataFrame with individual WTP values
        """
        base_param = base_num_param or self.config.dur_param
        denom_param = cost_param or self.config.fee_param
        scale = self.config.fee_scale

        beta_base = self.result.betas.get(base_param, 0)
        beta_denom = self.result.betas.get(denom_param, 0)

        if beta_denom == 0:
            raise ValueError("Cost coefficient is zero")

        # Compute individual-specific numerator
        individual_betas = np.full(len(demographics), beta_base)

        for demo_col, param_name in interaction_params.items():
            if demo_col in demographics.columns and param_name in self.result.betas:
                beta_interaction = self.result.betas[param_name]
                individual_betas += beta_interaction * demographics[demo_col].values

        # Compute individual WTP
        individual_wtp = -individual_betas / beta_denom * scale

        result_df = demographics.copy()
        result_df['wtp'] = individual_wtp
        result_df['beta_individual'] = individual_betas

        return result_df

    def compute_mxl_wtp_distribution(self,
                                      num_mean_param: str = None,
                                      num_std_param: str = None,
                                      denom_mean_param: str = None,
                                      denom_std_param: str = None,
                                      n_draws: int = None,
                                      correlation: float = 0.0) -> WTPDistributionResult:
        """
        Compute WTP distribution for Mixed Logit models.

        For MXL with random coefficients:
        β_dur ~ N(μ_dur, σ_dur²)
        β_fee ~ N(μ_fee, σ_fee²)  [or lognormal]

        Simulates from these distributions to get WTP distribution.

        Args:
            num_mean_param: Mean of numerator random coefficient
            num_std_param: Std dev of numerator random coefficient
            denom_mean_param: Mean of denominator random coefficient
            denom_std_param: Std dev of denominator random coefficient
            n_draws: Number of simulation draws
            correlation: Correlation between coefficients (default 0)

        Returns:
            WTPDistributionResult with distribution statistics
        """
        # Get parameter names
        num_mean = num_mean_param or self.config.dur_mean_param
        num_std = num_std_param or self.config.dur_std_param
        denom_mean = denom_mean_param or self.config.fee_mean_param
        denom_std = denom_std_param or self.config.fee_std_param
        n = n_draws or self.config.n_simulations
        scale = self.config.fee_scale

        # Get estimated distribution parameters
        mu_num = self.result.betas.get(num_mean, 0)
        sigma_num = abs(self.result.betas.get(num_std, 0.01))
        mu_denom = self.result.betas.get(denom_mean, 0)
        sigma_denom = abs(self.result.betas.get(denom_std, 0.01))

        # Set up correlation matrix
        cov_matrix = np.array([
            [sigma_num**2, correlation * sigma_num * sigma_denom],
            [correlation * sigma_num * sigma_denom, sigma_denom**2]
        ])

        # Draw from bivariate normal
        np.random.seed(self.config.seed)
        draws = np.random.multivariate_normal(
            [mu_num, mu_denom], cov_matrix, n
        )

        beta_num_draws = draws[:, 0]
        beta_denom_draws = draws[:, 1]

        # Compute WTP for each draw (filter invalid)
        valid_mask = np.abs(beta_denom_draws) > 1e-10
        wtp_draws = np.where(
            valid_mask,
            -beta_num_draws / beta_denom_draws * scale,
            np.nan
        )
        wtp_draws = wtp_draws[~np.isnan(wtp_draws)]

        # Trim extreme values (beyond 1st/99th percentile)
        p1, p99 = np.percentile(wtp_draws, [1, 99])
        wtp_trimmed = wtp_draws[(wtp_draws >= p1) & (wtp_draws <= p99)]

        # Compute statistics
        percentiles = {
            5: np.percentile(wtp_draws, 5),
            25: np.percentile(wtp_draws, 25),
            50: np.percentile(wtp_draws, 50),
            75: np.percentile(wtp_draws, 75),
            95: np.percentile(wtp_draws, 95)
        }

        return WTPDistributionResult(
            wtp_mean=np.mean(wtp_draws),
            wtp_std=np.std(wtp_draws),
            wtp_median=np.median(wtp_draws),
            wtp_percentiles=percentiles,
            positive_proportion=np.mean(wtp_draws > 0),
            draws=wtp_draws
        )

    def print_summary(self, wtp_result: WTPResult = None) -> None:
        """Print formatted WTP summary."""
        if wtp_result is None:
            wtp_result = self.compute_wtp()

        print(f"\n{'='*60}")
        print("WILLINGNESS-TO-PAY ANALYSIS")
        print(f"{'='*60}")
        print(f"Method: {wtp_result.method}")
        print(f"Scale factor: {wtp_result.scale_factor:,.0f}")
        print()
        print(f"Numerator ({wtp_result.numerator_param}):   "
              f"{self.result.betas.get(wtp_result.numerator_param, 0):.4f}")
        print(f"Denominator ({wtp_result.denominator_param}): "
              f"{self.result.betas.get(wtp_result.denominator_param, 0):.4f}")
        print()
        print(f"WTP Point Estimate: {wtp_result.wtp_point:>12,.2f}")
        print(f"Standard Error:     {wtp_result.wtp_se:>12,.2f}")
        print(f"t-statistic:        {wtp_result.t_stat:>12.3f}")
        print(f"p-value:            {wtp_result.p_value:>12.4f}")
        print()
        conf_pct = int(self.config.confidence_level * 100)
        print(f"{conf_pct}% Confidence Interval: "
              f"[{wtp_result.ci_lower:,.2f}, {wtp_result.ci_upper:,.2f}]")
        print(f"{'='*60}")


def compute_wtp_quick(betas: Dict[str, float],
                      std_errs: Dict[str, float] = None,
                      numerator: str = 'B_DUR',
                      denominator: str = 'B_FEE',
                      scale: float = 10000.0,
                      for_improvement: bool = True) -> float:
    """
    Quick WTP calculation without full result object.

    Args:
        betas: Dictionary of coefficient estimates
        std_errs: Optional standard errors
        numerator: Numerator parameter name
        denominator: Denominator parameter name
        scale: Scale factor
        for_improvement: If True (default), returns positive WTP for
                        beneficial changes (reduction for bad attributes)

    Returns:
        WTP point estimate
    """
    beta_num = betas.get(numerator, 0)
    beta_denom = betas.get(denominator, 0)

    if beta_denom == 0:
        return np.nan

    wtp = -beta_num / beta_denom * scale

    # For "bad" attributes (negative β), flip sign for improvement WTP
    if for_improvement and beta_num < 0:
        wtp = -wtp

    return wtp


if __name__ == '__main__':
    print("WTP Calculator Module")
    print("=" * 40)

    # Example with dummy data
    result = EstimationResult(
        betas={'B_FEE': -0.5, 'B_DUR': -0.08, 'ASC_paid': 1.2},
        std_errs={'B_FEE': 0.05, 'B_DUR': 0.02, 'ASC_paid': 0.15}
    )

    calc = WTPCalculator(result)

    # Default: WTP for improvement (positive for duration reduction)
    wtp = calc.compute_wtp('B_DUR')
    print(f"\nExample Calculation (for_improvement=True, default):")
    print(f"  B_FEE = -0.5, B_DUR = -0.08")
    print(f"  Since B_DUR < 0, duration reduction is desirable")
    print(f"  WTP per day saved = {wtp.wtp_point:,.0f} TL (positive)")

    # Raw MRS (WTP for duration INCREASE - would be negative)
    wtp_raw = calc.compute_wtp('B_DUR', for_improvement=False)
    print(f"\nRaw MRS (for_improvement=False):")
    print(f"  WTP per day added = {wtp_raw.wtp_point:,.0f} TL (negative - people don't want more days)")

    calc.print_summary(wtp)
