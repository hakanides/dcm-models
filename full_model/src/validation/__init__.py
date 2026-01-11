"""
Validation Module
=================

Monte Carlo simulation and validation tools for DCM/HCM models.

Components:
- monte_carlo.py: Monte Carlo study framework for parameter recovery
- Compare two-stage vs ICLV estimation
- Bias, RMSE, and coverage analysis

Authors: Hakan Mülayim, Giray Girengir, Ataol Azeritürk
"""

from .monte_carlo import (
    MonteCarloStudy,
    MonteCarloResult,
    run_monte_carlo_comparison,
    compute_bias,
    compute_rmse,
    compute_coverage
)

__all__ = [
    'MonteCarloStudy',
    'MonteCarloResult',
    'run_monte_carlo_comparison',
    'compute_bias',
    'compute_rmse',
    'compute_coverage'
]
