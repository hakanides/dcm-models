"""
Policy Analysis Module for DCM Models
=====================================

Comprehensive policy analysis tools for discrete choice models, including:
- Willingness-to-Pay (WTP) calculations
- Price elasticities (own and cross)
- Marginal effects
- Demand forecasting and market share prediction
- Consumer surplus and welfare analysis

Usage:
    from src.policy_analysis import (
        WTPCalculator,
        ElasticityCalculator,
        MarginalEffectCalculator,
        DemandForecaster,
        WelfareAnalyzer,
        PolicyScenario
    )

    # Create scenario
    scenario = PolicyScenario(
        name='Current',
        attributes={'fee': [500000, 600000, 0], 'dur': [10, 8, 15]}
    )

    # After model estimation
    result = {'betas': {...}, 'std_errs': {...}}

    # WTP analysis (defaults to Fieller method for statistically valid CIs)
    wtp = WTPCalculator(result).compute()  # or .compute_wtp_robust()

    # Elasticity analysis
    eta = ElasticityCalculator(result).own_price_elasticity(scenario, alternative=0)

    # Market shares
    shares = DemandForecaster(result).predict_market_shares(scenario)

    # Welfare analysis
    cv = WelfareAnalyzer(result).compute_compensating_variation(baseline, policy)

Authors: Hakan Mülayim, Giray Girengir, Ataol Azeritürk
"""

# Base classes and utilities
from .base import (
    EstimationResult,
    PolicyScenario,
    PolicyAnalysisConfig,
    compute_logit_probabilities,
    compute_utilities,
    delta_method_se,
    krinsky_robb_draws
)

# WTP calculations
from .wtp import (
    WTPCalculator,
    WTPResult,
    WTPDistributionResult,
    compute_wtp_quick
)

# Elasticity calculations
from .elasticity import (
    ElasticityCalculator,
    ElasticityResult,
    arc_elasticity,
    compute_hcm_elasticity  # Simulation-based elasticity for HCM/ICLV models
)

# Marginal effects
from .marginal_effects import (
    MarginalEffectCalculator,
    MarginalEffectResult
)

# Demand forecasting
from .demand_forecasting import (
    DemandForecaster,
    MarketShareResult,
    ScenarioComparisonResult
)

# Welfare analysis
from .welfare import (
    WelfareAnalyzer,
    ConsumerSurplusResult,
    WelfareChangeResult
)

__all__ = [
    # Base
    'EstimationResult',
    'PolicyScenario',
    'PolicyAnalysisConfig',
    'compute_logit_probabilities',
    'compute_utilities',
    'delta_method_se',
    'krinsky_robb_draws',

    # WTP
    'WTPCalculator',
    'WTPResult',
    'WTPDistributionResult',
    'compute_wtp_quick',

    # Elasticity
    'ElasticityCalculator',
    'ElasticityResult',
    'arc_elasticity',
    'compute_hcm_elasticity',

    # Marginal Effects
    'MarginalEffectCalculator',
    'MarginalEffectResult',

    # Demand Forecasting
    'DemandForecaster',
    'MarketShareResult',
    'ScenarioComparisonResult',

    # Welfare
    'WelfareAnalyzer',
    'ConsumerSurplusResult',
    'WelfareChangeResult',
]

__version__ = '1.0.0'
