"""
Centralized Constants for DCM Models
=====================================

This module defines all magic numbers and constants used across the DCM project.
Import from here to ensure consistency and avoid hardcoded values.

Usage:
    from models.shared.constants import FEE_SCALE, THRESHOLDS_DEFAULT
    # or
    import models.shared.constants as C
    fee_scaled = fee / C.FEE_SCALE

Authors: Hakan Mülayim, Giray Girengir, Ataol Azeritürk
"""

# =============================================================================
# SCALING CONSTANTS
# =============================================================================

# Fee scaling factor: raw fee values are divided by this for numerical stability
# Example: 15000 TL -> 1.5 in utility equation (keeps coefficients reasonable)
FEE_SCALE = 10000.0

# Duration scaling (if needed)
DUR_SCALE = 1.0


# =============================================================================
# DEMOGRAPHIC CENTERING
# =============================================================================

# Default centering values for demographic variables
# Centering improves numerical stability and makes coefficients interpretable
# Interpretation: coefficient represents effect at the centered value

DEMOGRAPHIC_CENTERS = {
    'age_idx': 2.0,          # Age group: 0-4, centered at middle group
    'edu_idx': 3.0,          # Education: 0-5, centered at most common (university)
    'income_indiv_idx': 3.0, # Income: 0-7, centered at median
}

# Default scaling values for demographic variables
# Scale=2.0 means coefficient represents effect per 2 category change
DEMOGRAPHIC_SCALES = {
    'age_idx': 2.0,
    'edu_idx': 2.0,
    'income_indiv_idx': 2.0,
}


# =============================================================================
# ORDERED PROBIT THRESHOLDS
# =============================================================================

# Default thresholds for 5-point Likert scale (4 thresholds for 5 categories)
# Assumed standard normal latent distribution, symmetric around 0
THRESHOLDS_DEFAULT = [-1.0, -0.35, 0.35, 1.0]

# Threshold bounds for estimation (prevents degenerate solutions)
THRESHOLD_LOWER_BOUND = -3.0
THRESHOLD_UPPER_BOUND = 3.0


# =============================================================================
# PARAMETER BOUNDS
# =============================================================================

# Reasonable bounds for coefficient estimation
COEF_BOUNDS = {
    'asc': (-10.0, 20.0),           # Alternative-specific constants
    'fee': (-2.0, 0.0),              # Fee coefficient (negative expected)
    'dur': (-2.0, 0.0),              # Duration coefficient (negative expected)
    'demographic': (-0.5, 0.5),      # Demographic interactions
    'lv_effect': (-0.5, 0.5),        # Latent variable effects
    'sigma': (0.01, 3.0),            # Standard deviations (positive)
    'loading': (0.3, 1.5),           # Factor loadings
    'gamma': (-1.0, 1.0),            # Structural coefficients
}


# =============================================================================
# ESTIMATION DEFAULTS
# =============================================================================

# Monte Carlo integration draws for mixed models
N_DRAWS_DEFAULT = 500          # Standard for estimation
N_DRAWS_QUICK = 100            # For quick tests
N_DRAWS_ROBUST = 1000          # For final results

# Draw type for simulation draws
DRAW_TYPE_DEFAULT = 'HALTON2'  # Halton sequences (low-discrepancy)

# Random seed for reproducibility
SEED_DEFAULT = 42

# Convergence tolerance
CONVERGENCE_TOL = 1e-6


# =============================================================================
# MEASUREMENT MODEL THRESHOLDS
# =============================================================================

# Psychometric validation thresholds (Hair et al., 2019)
CRONBACH_ALPHA_THRESHOLD = 0.70
COMPOSITE_RELIABILITY_THRESHOLD = 0.70
AVE_THRESHOLD = 0.50
LOADING_THRESHOLD = 0.50
HTMT_THRESHOLD = 0.85  # Henseler et al., 2015


# =============================================================================
# BOOTSTRAP DEFAULTS
# =============================================================================

N_BOOTSTRAP_DEFAULT = 200     # Standard bootstrap replicates
N_BOOTSTRAP_QUICK = 50        # For quick tests
N_BOOTSTRAP_ROBUST = 500      # For publication


# =============================================================================
# ALTERNATIVE DEFINITIONS
# =============================================================================

# Standard 3-alternative choice setup
ALTERNATIVES_DEFAULT = {
    1: {'name': 'paid1', 'available': True},
    2: {'name': 'paid2', 'available': True},
    3: {'name': 'standard', 'available': True},
}

# Alternative names for reference
ALT_PAID1 = 'paid1'
ALT_PAID2 = 'paid2'
ALT_STANDARD = 'standard'


# =============================================================================
# FILE PATHS (relative to project root)
# =============================================================================

# Shared data location
SCENARIOS_PATH = 'models/shared/data/scenarios.csv'

# Output directories
OUTPUT_DIR = 'output'
RESULTS_DIR = 'results'
LATEX_DIR = 'output/latex'


# =============================================================================
# VALIDATION
# =============================================================================

def validate_fee_scale(scale: float) -> bool:
    """Check if fee scale is reasonable."""
    return 1000.0 <= scale <= 100000.0


def validate_threshold_ordering(thresholds: list) -> bool:
    """Check if thresholds are monotonically increasing."""
    return all(thresholds[i] < thresholds[i+1] for i in range(len(thresholds)-1))


def transform_demographic(value: float, center: float, scale: float = 1.0) -> float:
    """
    Transform demographic variable for interaction terms.

    Formula: x_transformed = (x - center) / scale

    Args:
        value: Raw demographic value
        center: Centering value (mean or reference point)
        scale: Scaling factor (default 1.0 = centering only)

    Returns:
        Transformed value

    Interpretation:
        - scale=1.0: Coefficient = effect per 1-unit change in original
        - scale>1.0: Coefficient = effect per 'scale'-unit change
                     Allows comparing effects across different scales

    Example:
        age_idx=4 with center=2.0, scale=2.0
        -> transform_demographic(4, 2.0, 2.0) = (4-2)/2 = 1.0
        -> If coefficient is 0.06, this means 0.06 effect per 2-category age increase
    """
    return (value - center) / scale


def interpret_scaled_coefficient(coef: float, scale: float) -> str:
    """
    Generate interpretation string for a scaled coefficient.

    Args:
        coef: Estimated coefficient
        scale: Scale factor used in transformation

    Returns:
        Human-readable interpretation
    """
    if scale == 1.0:
        return f"Effect of {coef:.4f} per 1-unit change"
    else:
        return f"Effect of {coef:.4f} per {scale:.0f}-unit change (or {coef/scale:.4f} per 1-unit)"
