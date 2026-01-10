"""
Shared Data Generating Process (DGP) Utilities
===============================================

Common functions for simulating discrete choice data across all model types.
This module eliminates code duplication across simulate_full_data.py files.

Functions:
- softmax: Compute choice probabilities
- draw_gumbel_errors: Generate Gumbel errors for RUM
- draw_categorical: Sample from categorical distributions
- get_attribute_value: Extract attribute values from scenarios
- simulate_choice: Core choice simulation with random utility
- compute_base_utility: Compute utility from base terms
- apply_demographic_interactions: Add demographic effects to coefficients
- apply_latent_interactions: Add latent variable effects to coefficients
- generate_latent_values: Generate latent constructs from structural model

Usage:
    from shared.dgp import (
        softmax, draw_gumbel_errors, draw_categorical,
        simulate_choice, compute_base_utility
    )

Authors: Hakan Mulayim, Giray Girengir, Ataol Azeritürk
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable, Union


# =============================================================================
# CORE PROBABILITY FUNCTIONS
# =============================================================================

def softmax(utilities: np.ndarray) -> np.ndarray:
    """
    Compute choice probabilities using softmax (logit formula).

    P(j) = exp(V_j) / Σ_k exp(V_k)

    Args:
        utilities: Array of systematic utilities V_j

    Returns:
        Array of choice probabilities summing to 1
    """
    u = utilities - np.max(utilities)  # Numerical stability
    exp_u = np.exp(u)
    return exp_u / exp_u.sum()


def draw_gumbel_errors(rng: np.random.Generator, n: int) -> np.ndarray:
    """
    Draw Gumbel(0,1) errors for random utility model.

    The Gumbel distribution is derived from uniform: ε = -ln(-ln(U))
    where U ~ Uniform(0,1).

    This explicit error structure makes the RUM foundation clear:
    U_ij = V_ij + ε_ij, where ε_ij ~ iid Gumbel(0,1)

    Note: Drawing from Gumbel and choosing argmax is mathematically
    equivalent to multinomial sampling from softmax probabilities,
    but makes the error structure explicit for pedagogical purposes.

    Args:
        rng: NumPy random generator
        n: Number of errors to draw

    Returns:
        Array of n Gumbel(0,1) random variates
    """
    u = rng.uniform(1e-10, 1 - 1e-10, size=n)  # Avoid log(0)
    return -np.log(-np.log(u))


def draw_categorical(rng: np.random.Generator, spec: dict) -> int:
    """
    Sample from a categorical distribution.

    Args:
        rng: NumPy random generator
        spec: Dict with 'values' (list) and 'probs' (list) keys

    Returns:
        Sampled value from the categorical distribution
    """
    return rng.choice(spec['values'], p=spec['probs'])


# =============================================================================
# ATTRIBUTE EXTRACTION
# =============================================================================

def get_attribute_value(scenario: pd.Series, alt_idx: str,
                        attribute: str, fee_scale: float = 10000.0) -> float:
    """
    Extract attribute value for an alternative from scenario.

    Args:
        scenario: Pandas Series with scenario attributes
        alt_idx: Alternative index (e.g., '1', '2', '3')
        attribute: Attribute name (e.g., 'fee', 'dur')
        fee_scale: Scale factor for fee values

    Returns:
        Attribute value (scaled if fee)
    """
    col_name = f"{attribute}{alt_idx}"
    value = scenario[col_name]

    # Scale fee values
    if attribute == 'fee':
        value = value / fee_scale

    return float(value)


def extract_scenario_attributes(scenario: pd.Series,
                                 alternatives: Dict,
                                 fee_scale: float = 10000.0) -> Dict[str, Dict[str, float]]:
    """
    Extract all attributes for all alternatives from a scenario.

    Args:
        scenario: Pandas Series with scenario attributes
        alternatives: Dict of alternative configurations
        fee_scale: Scale factor for fee values

    Returns:
        Dict[alt_idx][attribute] = value
    """
    attributes = {}
    for alt_idx in alternatives.keys():
        attributes[alt_idx] = {}
        for attr in ['fee', 'dur']:
            attributes[alt_idx][attr] = get_attribute_value(
                scenario, alt_idx, attr, fee_scale
            )
    return attributes


# =============================================================================
# UTILITY COMPUTATION
# =============================================================================

def compute_base_utility(choice_cfg: dict, scenario: pd.Series,
                         alt_idx: str, alt_info: dict,
                         individual_coefficients: Dict[str, float] = None) -> float:
    """
    Compute systematic utility for one alternative.

    V = ASC (if applicable) + Σ β_k * x_k

    For models with heterogeneity (MXL, HCM), pass individual_coefficients
    to use individual-specific β values instead of base coefficients.

    Args:
        choice_cfg: Choice model configuration dict
        scenario: Pandas Series with scenario attributes
        alt_idx: Alternative index (e.g., '1', '2', '3')
        alt_info: Dict with alternative info (name, etc.)
        individual_coefficients: Optional dict of individual-specific coefficients
                                 Keys should match attribute names (e.g., 'fee', 'dur')

    Returns:
        Systematic utility V_j
    """
    V = 0.0
    fee_scale = choice_cfg.get('fee_scale', 10000.0)
    alt_name = alt_info['name']

    # Add ASC if this alternative has one
    for term in choice_cfg.get('base_terms', []):
        if alt_name in term.get('apply_to', []):
            V += term['coef']

    # Add attribute effects
    for term in choice_cfg.get('attribute_terms', []):
        if alt_name not in term.get('apply_to', []):
            continue

        attr = term['attribute']
        attr_value = get_attribute_value(scenario, alt_idx, attr, fee_scale)

        # Use individual coefficient if provided, else base coefficient
        if individual_coefficients and attr in individual_coefficients:
            coef = individual_coefficients[attr]
        else:
            coef = term['base_coef']

        V += coef * attr_value

    return V


def compute_individual_coefficient(base_coef: float,
                                    demographics: Dict[str, int],
                                    interactions: List[Dict],
                                    latent_values: Dict[str, float] = None,
                                    random_draw: float = None,
                                    sigma: float = None) -> float:
    """
    Compute individual-specific coefficient with all heterogeneity sources.

    β_i = β_base
          + Σ β_demo_k * (demo_k - center_k)  (demographic interactions)
          + Σ β_LV_k * η_k                     (latent variable effects)
          + σ * ω                              (random coefficient)

    Args:
        base_coef: Base coefficient value
        demographics: Dict of demographic values for this individual
        interactions: List of interaction specifications from config
        latent_values: Dict of latent variable values for this individual
        random_draw: Standard normal draw for random coefficient (if MXL)
        sigma: Standard deviation for random coefficient

    Returns:
        Individual-specific coefficient value
    """
    coef = base_coef

    # Demographic interactions
    for interaction in interactions:
        demo_name = interaction['demographic']
        if demo_name in demographics:
            demo_value = demographics[demo_name]
            center = interaction.get('center', 0)
            interaction_coef = interaction['coef']
            coef += interaction_coef * (demo_value - center)

    # Latent variable interactions
    if latent_values:
        for interaction in interactions:
            if 'latent' in interaction:
                lv_name = interaction['latent']
                if lv_name in latent_values:
                    lv_coef = interaction.get('lv_coef', 0)
                    coef += lv_coef * latent_values[lv_name]

    # Random coefficient (MXL)
    if random_draw is not None and sigma is not None:
        coef += sigma * random_draw

    return coef


# =============================================================================
# LATENT VARIABLE GENERATION
# =============================================================================

def generate_latent_value(demographics: Dict[str, int],
                          structural_params: Dict,
                          rng: np.random.Generator) -> float:
    """
    Generate latent variable value from structural model.

    η = Σ γ_k * (x_k - center_k) + σ * ω

    where ω ~ N(0,1)

    Args:
        demographics: Dict of demographic values
        structural_params: Dict with 'betas' (demographic effects) and 'sigma'
        rng: NumPy random generator

    Returns:
        Latent variable value for this individual
    """
    eta = 0.0

    # Add demographic effects
    for demo_name, gamma_spec in structural_params.get('betas', {}).items():
        if demo_name in demographics:
            gamma = gamma_spec['value']
            center = gamma_spec.get('center', 0)
            eta += gamma * (demographics[demo_name] - center)

    # Add random component
    sigma = structural_params.get('sigma', 1.0)
    omega = rng.standard_normal()
    eta += sigma * omega

    return eta


def generate_all_latent_values(demographics: Dict[str, int],
                                latent_config: Dict,
                                rng: np.random.Generator) -> Dict[str, float]:
    """
    Generate all latent variable values for an individual.

    Args:
        demographics: Dict of demographic values
        latent_config: Config dict with specifications for each LV
        rng: NumPy random generator

    Returns:
        Dict mapping latent variable names to values
    """
    latent_values = {}

    for lv_name, lv_spec in latent_config.items():
        structural = lv_spec.get('structural', {})
        latent_values[lv_name] = generate_latent_value(
            demographics, structural, rng
        )

    return latent_values


# =============================================================================
# MEASUREMENT MODEL
# =============================================================================

def generate_ordinal_indicator(latent_value: float,
                                loading: float,
                                thresholds: List[float],
                                rng: np.random.Generator) -> int:
    """
    Generate ordinal indicator from latent variable (ordered probit).

    y* = λ * η + ε, where ε ~ N(0,1)
    y = k if τ_{k-1} < y* ≤ τ_k

    Args:
        latent_value: Value of the latent variable η
        loading: Factor loading λ
        thresholds: List of threshold values [τ_1, τ_2, ..., τ_K-1]
        rng: NumPy random generator

    Returns:
        Ordinal response (1 to K)
    """
    y_star = loading * latent_value + rng.standard_normal()

    # Determine category based on thresholds
    for k, tau in enumerate(thresholds):
        if y_star <= tau:
            return k + 1

    return len(thresholds) + 1


def generate_measurement_indicators(latent_values: Dict[str, float],
                                     latent_config: Dict,
                                     rng: np.random.Generator) -> Dict[str, int]:
    """
    Generate all measurement indicators for an individual.

    Args:
        latent_values: Dict of latent variable values
        latent_config: Config with measurement specifications
        rng: NumPy random generator

    Returns:
        Dict mapping indicator names to ordinal responses
    """
    indicators = {}

    for lv_name, lv_spec in latent_config.items():
        measurement = lv_spec.get('measurement', {})
        items = measurement.get('items', [])
        loadings = measurement.get('loadings', [1.0] * len(items))
        thresholds = measurement.get('thresholds', [-1.0, -0.35, 0.35, 1.0])

        eta = latent_values.get(lv_name, 0)

        for i, item_name in enumerate(items):
            loading = loadings[i] if i < len(loadings) else 1.0
            indicators[item_name] = generate_ordinal_indicator(
                eta, loading, thresholds, rng
            )

    return indicators


# =============================================================================
# CHOICE SIMULATION
# =============================================================================

def simulate_choice(utilities: np.ndarray,
                    rng: np.random.Generator,
                    alt_indices: List[str]) -> int:
    """
    Simulate a choice given systematic utilities.

    Uses random utility maximization:
    U_j = V_j + ε_j, choose argmax_j U_j

    Args:
        utilities: Array of systematic utilities V_j
        rng: NumPy random generator
        alt_indices: List of alternative indices

    Returns:
        Chosen alternative index (as integer)
    """
    gumbel_errors = draw_gumbel_errors(rng, len(utilities))
    total_utilities = utilities + gumbel_errors
    choice_idx = np.argmax(total_utilities)
    return int(alt_indices[choice_idx])


def simulate_choice_task(choice_cfg: Dict,
                          scenario: pd.Series,
                          alternatives: Dict,
                          rng: np.random.Generator,
                          individual_coefficients: Dict[str, float] = None) -> int:
    """
    Simulate a single choice task for an individual.

    Args:
        choice_cfg: Choice model configuration
        scenario: Pandas Series with scenario attributes
        alternatives: Dict of alternative configurations
        rng: NumPy random generator
        individual_coefficients: Optional individual-specific coefficients

    Returns:
        Chosen alternative (as integer)
    """
    alt_indices = list(alternatives.keys())
    utilities = []

    for alt_idx in alt_indices:
        alt_info = alternatives[alt_idx]
        V = compute_base_utility(
            choice_cfg, scenario, alt_idx, alt_info,
            individual_coefficients
        )
        utilities.append(V)

    return simulate_choice(np.array(utilities), rng, alt_indices)


# =============================================================================
# FULL SIMULATION PIPELINE
# =============================================================================

def simulate_panel_data(config: Dict,
                         scenarios: pd.DataFrame,
                         rng: np.random.Generator,
                         compute_individual_coefs: Callable = None,
                         generate_latent: bool = False,
                         verbose: bool = True) -> pd.DataFrame:
    """
    Simulate panel data for discrete choice models.

    This is the main simulation function that handles all model types:
    - MNL: No heterogeneity
    - MNL with demographics: Demographic interactions
    - MXL: Random coefficients
    - HCM/ICLV: Latent variables

    Args:
        config: Full configuration dict
        scenarios: DataFrame of choice scenarios
        rng: NumPy random generator
        compute_individual_coefs: Optional function(demographics, latent_values, config, rng)
                                  -> Dict[str, float] for individual coefficients
        generate_latent: If True, generate latent variables and indicators
        verbose: Print progress

    Returns:
        DataFrame with simulated panel data
    """
    pop_cfg = config['population']
    N = pop_cfg['N']
    T = pop_cfg['T']

    alternatives = config['choice_model']['alternatives']
    alt_indices = list(alternatives.keys())
    n_scenarios = len(scenarios)

    if verbose:
        print(f"Simulating {N} individuals x {T} tasks = {N*T} observations")

    records = []
    latent_records = []  # For storing true latent values

    for i in range(N):
        # Draw demographics
        demographics = {}
        for demo_name, demo_spec in config.get('demographics', {}).items():
            demographics[demo_name] = draw_categorical(rng, demo_spec)

        # Generate latent values if needed
        latent_values = {}
        indicators = {}
        if generate_latent and 'latent' in config:
            latent_values = generate_all_latent_values(
                demographics, config['latent'], rng
            )
            indicators = generate_measurement_indicators(
                latent_values, config['latent'], rng
            )
            # Store true latent values
            latent_records.append({'ID': i, **latent_values})

        # Compute individual coefficients
        individual_coefs = None
        if compute_individual_coefs:
            individual_coefs = compute_individual_coefs(
                demographics, latent_values, config, rng
            )

        # Simulate T choice tasks
        for t in range(T):
            scenario_idx = rng.integers(0, n_scenarios)
            scenario = scenarios.iloc[scenario_idx]

            # Simulate choice
            choice = simulate_choice_task(
                config['choice_model'], scenario, alternatives,
                rng, individual_coefs
            )

            # Build record
            record = {
                'ID': i,
                'task': t,
                'CHOICE': choice,
                'scenario_id': scenario.get('scenario_id', scenario_idx),
                **demographics,
                **indicators,
                **{col: scenario[col] for col in scenarios.columns}
            }

            # Add scaled fee columns
            fee_scale = config['choice_model'].get('fee_scale', 10000.0)
            for alt_idx in alt_indices:
                record[f'fee{alt_idx}_10k'] = scenario[f'fee{alt_idx}'] / fee_scale

            records.append(record)

    df = pd.DataFrame(records)

    # Optionally return latent values DataFrame
    if generate_latent and latent_records:
        df_latent = pd.DataFrame(latent_records)
        return df, df_latent

    return df


# =============================================================================
# CONVENIENCE FUNCTIONS FOR SPECIFIC MODEL TYPES
# =============================================================================

def mnl_coefficient_function(demographics: Dict, latent_values: Dict,
                              config: Dict, rng: np.random.Generator) -> Dict[str, float]:
    """
    Coefficient function for MNL Basic - no heterogeneity.
    Returns None to use base coefficients.
    """
    return None


def mnl_demographics_coefficient_function(demographics: Dict, latent_values: Dict,
                                           config: Dict, rng: np.random.Generator) -> Dict[str, float]:
    """
    Coefficient function for MNL with demographics.
    Applies demographic interactions to base coefficients.

    Handles config format where interactions use 'with' key and
    centering/scaling comes from demographics config section.
    """
    coefs = {}
    choice_cfg = config['choice_model']
    demo_specs = config.get('demographics', {})

    for term in choice_cfg.get('attribute_terms', []):
        attr = term['attribute']
        base = term['base_coef']

        # Build interactions list with centering from demographics config
        interactions = []
        for inter in term.get('interactions', []):
            demo_name = inter.get('with', inter.get('demographic'))
            if demo_name and demo_name in demo_specs:
                demo_spec = demo_specs[demo_name]
                center = demo_spec.get('center', 0)
                scale = demo_spec.get('scale', 1)
                interactions.append({
                    'demographic': demo_name,
                    'coef': inter['coef'],
                    'center': center,
                    'scale': scale
                })

        coefs[attr] = compute_individual_coefficient_scaled(
            base, demographics, interactions
        )

    return coefs


def compute_individual_coefficient_scaled(base_coef: float,
                                          demographics: Dict[str, int],
                                          interactions: List[Dict]) -> float:
    """
    Compute individual coefficient with centering AND scaling.

    β_i = β_base + Σ β_k * (demo_k - center_k) / scale_k

    This version supports both centering and scaling, matching the config format.
    """
    coef = base_coef

    for interaction in interactions:
        demo_name = interaction['demographic']
        if demo_name in demographics:
            demo_value = demographics[demo_name]
            center = interaction.get('center', 0)
            scale = interaction.get('scale', 1)
            interaction_coef = interaction['coef']
            coef += interaction_coef * (demo_value - center) / scale

    return coef


def mxl_coefficient_function(demographics: Dict, latent_values: Dict,
                              config: Dict, rng: np.random.Generator) -> Dict[str, float]:
    """
    Coefficient function for MXL - random coefficients.

    Supports two config formats:
    1. random_coefficients section: {coef_name: {distribution, mean, std}}
    2. attribute_terms with sigma: [{attribute, base_coef, sigma}]
    """
    coefs = {}
    choice_cfg = config['choice_model']

    # Check for random_coefficients section (common format)
    random_coefs = config.get('random_coefficients', {})
    if random_coefs:
        # Use random_coefficients section
        for coef_name, coef_spec in random_coefs.items():
            if coef_spec.get('distribution') == 'normal':
                mu = coef_spec['mean']
                sigma = coef_spec['std']
                coefs[coef_name] = rng.normal(mu, sigma)
            elif coef_spec.get('distribution') == 'lognormal':
                mu = coef_spec['mean']
                sigma = coef_spec['std']
                coefs[coef_name] = -np.exp(rng.normal(np.log(-mu), sigma))
            else:
                # Default to normal
                mu = coef_spec.get('mean', 0)
                sigma = coef_spec.get('std', 0)
                coefs[coef_name] = rng.normal(mu, sigma) if sigma > 0 else mu

        # Also need non-random coefficients from attribute_terms
        for term in choice_cfg.get('attribute_terms', []):
            coef_name = term.get('name', term['attribute'].upper())
            if coef_name not in coefs and 'B_' + term['attribute'].upper() not in coefs:
                coefs[term['attribute']] = term['base_coef']
    else:
        # Use attribute_terms with sigma
        for term in choice_cfg.get('attribute_terms', []):
            attr = term['attribute']
            base = term['base_coef']
            sigma = term.get('sigma', 0)

            # Draw random component
            if sigma > 0:
                omega = rng.standard_normal()
                coefs[attr] = base + sigma * omega
            else:
                coefs[attr] = base

    return coefs


def hcm_coefficient_function(demographics: Dict, latent_values: Dict,
                              config: Dict, rng: np.random.Generator) -> Dict[str, float]:
    """
    Coefficient function for HCM/ICLV - latent variable effects.

    Supports two config formats:
    1. lv_effects dict: {lv_name: coef}
    2. interactions list with type='latent': [{type: 'latent', with: lv_name, coef: value}]
    """
    coefs = {}
    choice_cfg = config['choice_model']

    for term in choice_cfg.get('attribute_terms', []):
        attr = term['attribute']
        base = term['base_coef']
        coef = base

        # Check for lv_effects dict (format 1)
        lv_effects = term.get('lv_effects', {})
        for lv_name, lv_coef in lv_effects.items():
            if lv_name in latent_values:
                coef += lv_coef * latent_values[lv_name]

        # Check for interactions with type='latent' (format 2)
        for inter in term.get('interactions', []):
            if inter.get('type') == 'latent':
                lv_name = inter.get('with')
                if lv_name and lv_name in latent_values:
                    coef += inter['coef'] * latent_values[lv_name]

        coefs[attr] = coef

    return coefs


def generate_latent_variable_hcm(rng: np.random.Generator, demographics: Dict,
                                  lv_config: Dict, demo_centers: Dict = None) -> float:
    """
    Generate latent variable from HCM structural model config format.

    Handles config format:
    lv_config = {
        'structural': {
            'intercept': 0,
            'betas': {demo_name: beta_value},
            'center': {demo_name: center_value},  # or as separate dict
            'sigma': 1.0
        }
    }

    Args:
        rng: NumPy random generator
        demographics: Dict of demographic values for this individual
        lv_config: Latent variable config with structural section
        demo_centers: Optional dict of demographic centering values

    Returns:
        Latent variable value
    """
    struct = lv_config['structural']

    # Compute systematic component
    lv = struct.get('intercept', 0.0)
    centers = struct.get('center', {})

    # Merge with demo_centers if provided
    if demo_centers:
        for k, v in demo_centers.items():
            if k not in centers:
                centers[k] = v

    for demo_name, beta in struct.get('betas', {}).items():
        demo_val = demographics.get(demo_name, 0)
        center = centers.get(demo_name, 0)
        lv += beta * (demo_val - center)

    # Add random component
    sigma = struct.get('sigma', 1.0)
    lv += rng.normal(0, sigma)

    return lv


def generate_likert_items_hcm(rng: np.random.Generator, lv_value: float,
                               measurement_config: Dict) -> Dict[str, int]:
    """
    Generate Likert items from HCM measurement model config format.

    Handles config format:
    measurement_config = {
        'items': ['item1', 'item2', ...],
        'loadings': [0.85, 0.83, ...],
        'thresholds': [-1.0, -0.35, 0.35, 1.0]
    }

    Args:
        rng: NumPy random generator
        lv_value: Latent variable value
        measurement_config: Dict with items, loadings, thresholds

    Returns:
        Dict mapping item names to ordinal responses (1-5)
    """
    items = measurement_config['items']
    loadings = measurement_config['loadings']
    thresholds = measurement_config['thresholds']

    responses = {}
    for item_name, loading in zip(items, loadings):
        # Continuous latent response: y* = lambda * eta + epsilon
        y_star = loading * lv_value + rng.normal(0, 1)

        # Convert to ordinal using thresholds
        response = 1
        for thresh in thresholds:
            if y_star > thresh:
                response += 1
        responses[item_name] = min(response, 5)

    return responses


def generate_all_latent_hcm(rng: np.random.Generator, demographics: Dict,
                            latent_config: Dict, demo_specs: Dict = None) -> Tuple[Dict[str, float], Dict[str, int]]:
    """
    Generate all latent variables and measurement indicators for HCM/ICLV.

    Args:
        rng: NumPy random generator
        demographics: Dict of demographic values
        latent_config: Config dict with all latent variable specs
        demo_specs: Demographics config with centering values

    Returns:
        Tuple of (latent_values dict, likert_responses dict)
    """
    # Get centering values from demographics spec
    demo_centers = {}
    if demo_specs:
        for demo_name, spec in demo_specs.items():
            demo_centers[demo_name] = spec.get('center', 0)

    latent_values = {}
    likert_responses = {}

    for lv_name, lv_config in latent_config.items():
        # Generate latent value
        lv_value = generate_latent_variable_hcm(rng, demographics, lv_config, demo_centers)
        latent_values[lv_name] = lv_value

        # Generate Likert items
        if 'measurement' in lv_config:
            items = generate_likert_items_hcm(rng, lv_value, lv_config['measurement'])
            likert_responses.update(items)

    return latent_values, likert_responses
