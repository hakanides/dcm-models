"""
ICLV: Integrated Choice and Latent Variable Model
==================================================

Full simultaneous estimation of measurement and choice models
using Simulated Maximum Likelihood (SML).

This eliminates the attenuation bias from two-stage estimation
by integrating over the latent variable distribution.

Key Components:
- core.py: ICLVModel main class
- measurement.py: Ordered probit measurement model
- structural.py: Structural model (demographics → LVs)
- integration.py: Monte Carlo integration with Halton draws
- estimation.py: SML estimation wrapper

Mathematical Formulation:
    L_n = (1/R) Σ_r [ P(y_n|η_r) × Π_k P(I_nk|η_r) ]

    Where:
    - y_n = choice outcome for individual n
    - I_nk = Likert indicator k for individual n
    - η_r = draw r from latent variable distribution
    - R = number of simulation draws (typically 500-1000)

References:
- Ben-Akiva et al. (2002). Hybrid Choice Models: Progress and Challenges
- Walker & Ben-Akiva (2002). Generalized random utility model
- Vij & Walker (2016). How, when and why integrated choice and latent variable
  models are latently useful

Author: DCM Research Team
"""

from .core import ICLVModel, ICLVResult
from .measurement import OrderedProbitMeasurement, MeasurementLikelihood
from .structural import StructuralModel
from .integration import HaltonDraws, MonteCarloIntegrator
from .estimation import (
    SMLEstimator,
    estimate_iclv,
    EstimationConfig,
    compare_two_stage_vs_iclv,
    summarize_attenuation_bias,
    compute_two_stage_starting_values,
    auto_scale_attributes,
    ScalingInfo,
)

__all__ = [
    # Core
    'ICLVModel',
    'ICLVResult',
    # Measurement
    'OrderedProbitMeasurement',
    'MeasurementLikelihood',
    # Structural
    'StructuralModel',
    # Integration
    'HaltonDraws',
    'MonteCarloIntegrator',
    # Estimation
    'SMLEstimator',
    'estimate_iclv',
    'EstimationConfig',
    # Comparison tools
    'compare_two_stage_vs_iclv',
    'summarize_attenuation_bias',
    # Utilities
    'compute_two_stage_starting_values',
    'auto_scale_attributes',
    'ScalingInfo',
]
