"""
Models module for DCM research.

This module provides isolated estimation files for each model type,
progressing from simple to complex:

Basic Models:
    - mnl_basic: Basic Multinomial Logit (ASC + fee + duration)
    - mnl_demographics: MNL with demographic interactions

Mixed Logit:
    - mxl_basic: Basic MXL with random fee coefficient

Hybrid Choice Models (Two-Stage):
    - hcm_basic: HCM with single latent variable (Blind Patriotism)
    - hcm_full: HCM with all 4 latent variables

Integrated Choice and Latent Variable (ICLV):
    - iclv: Simultaneous estimation (unbiased, in separate subpackage)

Compound Model Files (multiple models):
    - mnl_model_comparison: MNL model progression
    - mxl_models: Multiple MXL specifications
    - hcm_split_latents: Comprehensive HCM analysis with 15+ models

Usage:
    # Individual model estimation
    from src.models.mnl_basic import estimate_mnl_basic
    from src.models.mxl_basic import estimate_mxl_basic
    from src.models.hcm_basic import estimate_hcm_basic

    # Run estimation
    result = estimate_mnl_basic("path/to/data.csv")

    # Or run the full pipeline
    python scripts/run_all_models.py --data data/simulated/data.csv
"""

# Model factory for registry pattern
from .model_factory import ModelFactory

# Basic MNL models
from .mnl_basic import estimate_mnl_basic, create_mnl_basic
from .mnl_demographics import estimate_mnl_demographics, create_mnl_demographics

# Mixed Logit models
from .mxl_basic import estimate_mxl_basic, create_mxl_basic

# Hybrid Choice Models (two-stage)
from .hcm_basic import estimate_hcm_basic, create_hcm_basic
from .hcm_full import estimate_hcm_full, create_hcm_full

# Compound model files (optional - these have many dependencies)
try:
    from .mnl_model_comparison import run_model_comparison as run_mnl_comparison
except ImportError:
    run_mnl_comparison = None

try:
    from .mxl_models import run_mxl_comparison
except ImportError:
    run_mxl_comparison = None

try:
    from .hcm_split_latents import run_split_lv_analysis
except ImportError:
    run_split_lv_analysis = None

# ICLV (optional - may have additional dependencies)
try:
    from .iclv import ICLVModel, estimate_iclv
    ICLV_AVAILABLE = True
except ImportError:
    ICLV_AVAILABLE = False

# Validation models (simulation only)
try:
    from .validation_models import (
        has_true_latent_values,
        validate_latent_estimation
    )
except ImportError:
    pass

__all__ = [
    # Factory
    'ModelFactory',

    # Basic MNL
    'estimate_mnl_basic',
    'create_mnl_basic',
    'estimate_mnl_demographics',
    'create_mnl_demographics',

    # MXL
    'estimate_mxl_basic',
    'create_mxl_basic',

    # HCM (two-stage)
    'estimate_hcm_basic',
    'create_hcm_basic',
    'estimate_hcm_full',
    'create_hcm_full',

    # Compound runners
    'run_mnl_comparison',
    'run_mxl_comparison',
    'run_split_lv_analysis',

    # ICLV
    'ICLV_AVAILABLE',
]

# Add ICLV exports if available
if ICLV_AVAILABLE:
    __all__.extend(['ICLVModel', 'estimate_iclv'])
