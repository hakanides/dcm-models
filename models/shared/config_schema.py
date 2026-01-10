"""
Unified Configuration Schema for DCM Models
============================================

This module defines the unified configuration format for all DCM models.
It provides:
1. Schema definition with validation
2. Converters for legacy formats
3. Default values and constants

All model configurations should follow this schema for consistency.

Configuration Structure:
------------------------
{
    "model_info": {
        "name": str,           # Model name (e.g., "MNL Basic")
        "type": str,           # One of: "mnl", "mxl", "hcm", "iclv"
        "description": str,    # Brief description
        "true_values": dict    # True parameter values for validation
    },
    "population": {
        "N": int,              # Number of individuals
        "T": int,              # Choice tasks per individual
        "seed": int            # Random seed for reproducibility
    },
    "demographics": {
        "<name>": {            # Demographic variable specification
            "values": list,    # Possible values
            "probs": list,     # Probabilities for each value
            "center": float,   # Centering value for interactions
            "scale": float     # Scaling factor (default 1.0)
            # NOTE ON CENTERING VS STANDARDIZATION:
            # The transformation applied is: x_transformed = (x - center) / scale
            #
            # If scale = 1.0 (default): CENTERING only
            #   - Coefficient interpretation: effect per 1-unit change in original variable
            #   - At center value, the interaction contribution is zero
            #
            # If scale != 1.0: STANDARDIZATION
            #   - Coefficient interpretation: effect per 'scale'-unit change
            #   - E.g., scale=2.0 means coefficient is effect per 2-category change
            #   - This allows comparing effect sizes across different demographic scales
            #
            # Example: age_idx with center=2.0, scale=2.0
            #   - age_idx=0 -> (0-2)/2 = -1.0
            #   - age_idx=2 -> (2-2)/2 = 0.0 (at center, zero contribution)
            #   - age_idx=4 -> (4-2)/2 = 1.0
            #   - Coefficient of 0.06 means: 0.06 change per 2-category age increase
        }
    },
    "design": {
        "path": str            # Path to scenarios CSV file
    },
    "choice_model": {
        "fee_scale": float,    # Scale factor for fee (e.g., 10000)
        "alternatives": {
            "<idx>": {
                "name": str,   # Alternative name
                "available": bool
            }
        },
        "base_terms": [        # ASC and other constants
            {
                "name": str,
                "coef": float,
                "apply_to": list  # Alternative names
            }
        ],
        "attribute_terms": [   # Attribute coefficients
            {
                "attribute": str,   # e.g., "fee", "dur"
                "base_coef": float, # Base coefficient
                "apply_to": list,   # Alternative names
                "interactions": [   # Demographic interactions
                    {
                        "demographic": str,
                        "coef": float,
                        "center": float
                    }
                ],
                "lv_effects": {     # Latent variable effects (HCM/ICLV)
                    "<lv_name>": float
                },
                "sigma": float      # Random coefficient std (MXL)
            }
        ]
    },
    "latent": {                # For HCM/ICLV models
        "<lv_name>": {
            "structural": {
                "betas": {
                    "<demo>": {"value": float, "center": float}
                },
                "sigma": float
            },
            "measurement": {
                "items": list,      # Indicator names
                "loadings": list,   # Factor loadings (first fixed to 1)
                "thresholds": list  # Ordered probit thresholds
            }
        }
    }
}

Authors: Hakan Mulayim, Giray Girengir, Ataol Azeritürk
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import warnings

# Import centralized constants
from models.shared.constants import (
    FEE_SCALE,
    THRESHOLDS_DEFAULT,
)

# =============================================================================
# CONSTANTS
# =============================================================================

# Use centralized constants
DEFAULT_FEE_SCALE = FEE_SCALE
DEFAULT_THRESHOLDS = THRESHOLDS_DEFAULT
VALID_MODEL_TYPES = ['mnl', 'mxl', 'hcm', 'iclv']
VALID_DISTRIBUTIONS = ['normal', 'lognormal', 'triangular', 'uniform']


# =============================================================================
# SCHEMA VALIDATION
# =============================================================================

@dataclass
class ValidationResult:
    """Result of configuration validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def validate_config(config: Dict) -> ValidationResult:
    """
    Validate configuration against schema.

    Args:
        config: Configuration dictionary

    Returns:
        ValidationResult with validity status and any errors/warnings
    """
    errors = []
    warnings_list = []

    # Required top-level keys
    required_keys = ['model_info', 'population', 'choice_model']
    for key in required_keys:
        if key not in config:
            errors.append(f"Missing required key: {key}")

    if errors:
        return ValidationResult(False, errors, warnings_list)

    # Validate model_info
    model_info = config.get('model_info', {})
    if 'name' not in model_info:
        errors.append("model_info.name is required")
    if 'true_values' not in model_info:
        warnings_list.append("model_info.true_values not specified (needed for validation)")

    # Validate population
    pop = config.get('population', {})
    if 'N' not in pop:
        errors.append("population.N is required")
    if 'T' not in pop:
        errors.append("population.T is required")
    if 'seed' not in pop:
        warnings_list.append("population.seed not specified, using default")

    # Validate choice_model
    choice = config.get('choice_model', {})
    if 'alternatives' not in choice:
        errors.append("choice_model.alternatives is required")
    if 'attribute_terms' not in choice:
        warnings_list.append("choice_model.attribute_terms not specified")

    # Validate fee_scale
    if 'fee_scale' not in choice:
        warnings_list.append(f"choice_model.fee_scale not specified, using default {DEFAULT_FEE_SCALE}")

    # Validate model type consistency
    model_type = model_info.get('type', 'mnl').lower()
    if model_type not in VALID_MODEL_TYPES:
        errors.append(f"Invalid model type: {model_type}. Must be one of {VALID_MODEL_TYPES}")

    # HCM/ICLV models require latent section
    if model_type in ['hcm', 'iclv'] and 'latent' not in config:
        errors.append(f"Model type {model_type} requires 'latent' configuration")

    # MXL models should have sigma in attribute_terms
    if model_type == 'mxl':
        has_sigma = False
        for term in choice.get('attribute_terms', []):
            if 'sigma' in term and term['sigma'] > 0:
                has_sigma = True
                break
        if not has_sigma:
            warnings_list.append("MXL model has no random coefficients (sigma > 0)")

    # Validate latent section if present
    if 'latent' in config:
        for lv_name, lv_spec in config['latent'].items():
            if 'structural' not in lv_spec:
                warnings_list.append(f"latent.{lv_name}.structural not specified")
            if 'measurement' not in lv_spec:
                warnings_list.append(f"latent.{lv_name}.measurement not specified")
            else:
                meas = lv_spec['measurement']
                if 'items' not in meas:
                    errors.append(f"latent.{lv_name}.measurement.items is required")
                if 'loadings' in meas:
                    if len(meas['loadings']) != len(meas.get('items', [])):
                        errors.append(f"latent.{lv_name}: loadings count must match items count")
                if 'thresholds' not in meas:
                    warnings_list.append(f"latent.{lv_name}.measurement.thresholds not specified, using default")

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings_list
    )


# =============================================================================
# CONFIG LOADING AND CONVERSION
# =============================================================================

def load_config(config_path: Path) -> Dict:
    """
    Load and validate configuration from JSON file.

    Args:
        config_path: Path to config.json

    Returns:
        Validated configuration dictionary

    Raises:
        ValueError: If configuration is invalid
    """
    with open(config_path) as f:
        config = json.load(f)

    # Validate
    result = validate_config(config)

    if result.warnings:
        for w in result.warnings:
            warnings.warn(w, UserWarning)

    if not result.is_valid:
        raise ValueError(f"Invalid configuration:\n" + "\n".join(result.errors))

    # Apply defaults
    config = apply_defaults(config)

    return config


def apply_defaults(config: Dict) -> Dict:
    """
    Apply default values to configuration.

    Args:
        config: Raw configuration dictionary

    Returns:
        Configuration with defaults applied
    """
    # Population defaults
    if 'seed' not in config.get('population', {}):
        config['population']['seed'] = 42

    # Choice model defaults
    choice = config.get('choice_model', {})
    if 'fee_scale' not in choice:
        choice['fee_scale'] = DEFAULT_FEE_SCALE

    # Latent defaults
    if 'latent' in config:
        for lv_name, lv_spec in config['latent'].items():
            if 'measurement' in lv_spec:
                meas = lv_spec['measurement']
                if 'thresholds' not in meas:
                    meas['thresholds'] = DEFAULT_THRESHOLDS.copy()
                if 'loadings' not in meas:
                    # Default: first loading fixed to 1, others estimated
                    n_items = len(meas.get('items', []))
                    meas['loadings'] = [1.0] + [0.8] * (n_items - 1)

    return config


def convert_legacy_config(config: Dict, model_type: str = 'mnl') -> Dict:
    """
    Convert legacy configuration format to unified schema.

    Handles configurations from both models/ and full_model/ directories.

    Args:
        config: Legacy configuration dictionary
        model_type: Model type hint if not specified in config

    Returns:
        Configuration in unified format
    """
    unified = {}

    # Model info
    unified['model_info'] = {
        'name': config.get('model_info', {}).get('name', 'Unknown'),
        'type': config.get('model_info', {}).get('type', model_type),
        'description': config.get('model_info', {}).get('description', ''),
        'true_values': config.get('model_info', {}).get('true_values', {})
    }

    # Population
    unified['population'] = {
        'N': config.get('population', {}).get('N', 500),
        'T': config.get('population', {}).get('T', 10),
        'seed': config.get('population', {}).get('seed', 42)
    }

    # Demographics
    unified['demographics'] = config.get('demographics', {})

    # Design
    unified['design'] = config.get('design', {'path': 'data/scenarios.csv'})

    # Choice model
    if 'choice_model' in config:
        unified['choice_model'] = config['choice_model']
    else:
        # Convert from old format if needed
        unified['choice_model'] = {
            'fee_scale': DEFAULT_FEE_SCALE,
            'alternatives': {
                '1': {'name': 'Alt1', 'available': True},
                '2': {'name': 'Alt2', 'available': True},
                '3': {'name': 'Alt3', 'available': True}
            },
            'base_terms': [],
            'attribute_terms': []
        }

    # Latent (if present)
    if 'latent' in config:
        unified['latent'] = config['latent']

    return unified


# =============================================================================
# PARAMETER EXTRACTION
# =============================================================================

def get_true_parameter(config: Dict, param_name: str) -> Optional[float]:
    """
    Get true parameter value from configuration.

    Args:
        config: Configuration dictionary
        param_name: Parameter name

    Returns:
        True value if found, None otherwise
    """
    return config.get('model_info', {}).get('true_values', {}).get(param_name)


def get_all_true_parameters(config: Dict) -> Dict[str, float]:
    """
    Get all true parameter values from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary of parameter name -> true value
    """
    return config.get('model_info', {}).get('true_values', {})


def get_demographic_center(config: Dict, demo_name: str) -> float:
    """
    Get centering value for a demographic variable.

    Args:
        config: Configuration dictionary
        demo_name: Demographic variable name

    Returns:
        Center value (default 0)
    """
    return config.get('demographics', {}).get(demo_name, {}).get('center', 0)


# =============================================================================
# NAMING CONVENTIONS
# =============================================================================

# Standard parameter naming convention
PARAM_NAMING = {
    'asc': 'ASC_{alt}',           # Alternative-specific constant
    'fee_coef': 'B_FEE',          # Fee coefficient
    'dur_coef': 'B_DUR',          # Duration coefficient
    'fee_demo': 'B_FEE_{demo}',   # Fee-demographic interaction
    'dur_demo': 'B_DUR_{demo}',   # Duration-demographic interaction
    'fee_lv': 'B_FEE_{lv}',       # Fee-LV interaction
    'dur_lv': 'B_DUR_{lv}',       # Duration-LV interaction
    'lv_gamma': 'gamma_{lv}_{demo}',  # LV structural coefficient
    'lv_sigma': 'sigma_{lv}',     # LV standard deviation
    'lv_lambda': 'lambda_{lv}_{item}',  # LV factor loading
    'lv_tau': 'tau_{lv}_{k}',     # LV threshold
    'random_sigma': 'SIGMA_{param}'  # Random coefficient std (MXL)
}


def standardize_param_name(param_type: str, **kwargs) -> str:
    """
    Generate standardized parameter name.

    Args:
        param_type: Type of parameter (key in PARAM_NAMING)
        **kwargs: Variables to substitute (alt, demo, lv, item, k, param)

    Returns:
        Standardized parameter name
    """
    template = PARAM_NAMING.get(param_type, param_type)
    return template.format(**kwargs)
