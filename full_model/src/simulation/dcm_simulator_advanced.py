"""
Advanced Agent-Based Discrete Choice Model (DCM) Simulator
==========================================================

Extended version with support for:
1. Mixed Logit (Random Coefficients) with various distributions
2. Correlated random parameters (Cholesky decomposition)
3. Hybrid Choice Model (HCM) / ICLV structure
4. Panel data with consistent individual-level draws
5. Multiple measurement model types (ordinal, continuous, binary)

This implements a full Integrated Choice and Latent Variable (ICLV) model:

    ┌─────────────────────────────────────────────────────────────────┐
    │                    ICLV MODEL STRUCTURE                         │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  STRUCTURAL MODEL                 MEASUREMENT MODEL             │
    │  ════════════════                 ═════════════════             │
    │                                                                 │
    │  Demographics ──┬──> Latent      Latent ──> Indicators          │
    │                 │    Variables   Variables   (Likert, etc.)     │
    │                 │       │                                       │
    │                 │       │        CHOICE MODEL                   │
    │                 │       │        ════════════                   │
    │                 │       ▼                                       │
    │                 └──> Taste ────> Utility ──> Choice             │
    │                      Parameters     │                           │
    │                      (random)       │                           │
    │                         │           │                           │
    │                         └───────────┘                           │
    │                      Mixed Logit                                │
    └─────────────────────────────────────────────────────────────────┘

Authors: Hakan Mülayim, Giray Girengir, Ataol Azeritürk
"""

import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from enum import Enum
from scipy import stats


# =============================================================================
# ENUMS AND TYPES
# =============================================================================

class Distribution(Enum):
    """Supported distributions for random coefficients."""
    FIXED = "fixed"           # No randomness
    NORMAL = "normal"         # β ~ N(μ, σ²)
    LOGNORMAL = "lognormal"   # β ~ LogN(μ, σ²) - always positive
    TRIANGULAR = "triangular" # β ~ Tri(a, b, c)
    UNIFORM = "uniform"       # β ~ U(a, b)
    TRUNCATED_NORMAL = "truncated_normal"  # Truncated at bounds


class IndicatorType(Enum):
    """Types of measurement indicators."""
    ORDINAL = "ordinal"       # Likert scales (ordered categorical)
    CONTINUOUS = "continuous" # Continuous indicators
    BINARY = "binary"         # Binary (0/1) indicators


# =============================================================================
# RANDOM COEFFICIENT SPECIFICATION
# =============================================================================

@dataclass
class RandomCoeffSpec:
    """
    Specification for a random coefficient.

    Attributes:
        name: Coefficient name (e.g., 'b_fee')
        distribution: Type of distribution
        mean: Mean (μ) for normal/lognormal
        std: Standard deviation (σ) for normal/lognormal
        lower: Lower bound for truncated/triangular/uniform
        upper: Upper bound for truncated/triangular/uniform
        mode: Mode for triangular distribution
        correlated_with: List of other coefficients this is correlated with
    """
    name: str
    distribution: Distribution = Distribution.FIXED
    mean: float = 0.0
    std: float = 0.0
    lower: Optional[float] = None
    upper: Optional[float] = None
    mode: Optional[float] = None

    def draw(self, rng: np.random.Generator, n: int = 1) -> np.ndarray:
        """Draw n values from the distribution."""
        if self.distribution == Distribution.FIXED:
            return np.full(n, self.mean)

        elif self.distribution == Distribution.NORMAL:
            return rng.normal(self.mean, self.std, n)

        elif self.distribution == Distribution.LOGNORMAL:
            # Parameterized so that mean and std are of the resulting distribution
            # Convert to underlying normal parameters
            var = self.std ** 2
            mu_underlying = np.log(self.mean ** 2 / np.sqrt(var + self.mean ** 2))
            sigma_underlying = np.sqrt(np.log(1 + var / self.mean ** 2))
            return rng.lognormal(mu_underlying, sigma_underlying, n)

        elif self.distribution == Distribution.TRIANGULAR:
            left = self.lower if self.lower is not None else self.mean - self.std * np.sqrt(6)
            right = self.upper if self.upper is not None else self.mean + self.std * np.sqrt(6)
            mode = self.mode if self.mode is not None else self.mean
            return rng.triangular(left, mode, right, n)

        elif self.distribution == Distribution.UNIFORM:
            left = self.lower if self.lower is not None else self.mean - self.std * np.sqrt(3)
            right = self.upper if self.upper is not None else self.mean + self.std * np.sqrt(3)
            return rng.uniform(left, right, n)

        elif self.distribution == Distribution.TRUNCATED_NORMAL:
            lower = self.lower if self.lower is not None else -np.inf
            upper = self.upper if self.upper is not None else np.inf
            a = (lower - self.mean) / self.std
            b = (upper - self.mean) / self.std
            return stats.truncnorm.rvs(a, b, loc=self.mean, scale=self.std,
                                       size=n, random_state=rng)

        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")


# =============================================================================
# AGENT CLASS (Enhanced)
# =============================================================================

@dataclass
class Agent:
    """
    Enhanced agent with random coefficients for Mixed Logit.

    Attributes:
        agent_id: Unique identifier
        demographics: Observable characteristics
        latent_variables: Unobserved attitudes/preferences
        systematic_betas: Systematic part of taste parameters
        random_draws: Individual-specific random draws (for panel consistency)
        taste_parameters: Final individual-specific coefficients
        likert_responses: Observed indicator responses
        continuous_indicators: Continuous indicator values
    """
    agent_id: int
    agent_id_str: str
    demographics: Dict[str, int]
    latent_variables: Dict[str, float]
    systematic_betas: Dict[str, float]
    random_draws: Dict[str, float]  # η_i for each random coefficient
    taste_parameters: Dict[str, float]  # Final β_i = systematic + random
    likert_responses: Dict[str, int] = field(default_factory=dict)
    continuous_indicators: Dict[str, float] = field(default_factory=dict)

    def get_covariate(self, name: str) -> float:
        """Get demographic or latent variable value."""
        if name in self.demographics:
            return float(self.demographics[name])
        elif name in self.latent_variables:
            return float(self.latent_variables[name])
        raise KeyError(f"Covariate '{name}' not found")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        result = {
            'ID': self.agent_id,
            'ID_STR': self.agent_id_str,
        }
        result.update(self.demographics)
        result.update(self.likert_responses)
        result.update(self.continuous_indicators)
        return result

    def get_true_latents(self) -> Dict[str, float]:
        """Return true latent values for validation."""
        return {f'LV_{k}_true': v for k, v in self.latent_variables.items()}

    def get_true_betas(self) -> Dict[str, float]:
        """Return true taste parameters for validation."""
        result = {}
        for k, v in self.taste_parameters.items():
            result[f'beta_{k}_true'] = v
        for k, v in self.random_draws.items():
            result[f'eta_{k}_true'] = v  # Random component
        for k, v in self.systematic_betas.items():
            result[f'beta_sys_{k}_true'] = v  # Systematic component
        return result


# =============================================================================
# MEASUREMENT MODEL CLASS
# =============================================================================

class MeasurementModel:
    """
    Handles the measurement/indicator part of the HCM.

    Supports:
    - Ordinal indicators (Likert scales via ordered probit)
    - Continuous indicators (linear factor model)
    - Binary indicators (probit)
    """

    def __init__(self, config: Dict, rng: np.random.Generator):
        self.config = config.get('measurement', {})
        self.rng = rng

        # ISSUE #20 FIX: Improved default thresholds for better response distribution
        # Old thresholds [-1.5, -0.5, 0.5, 1.5] with LV~N(0,1) give:
        #   P(y=1) = 7%, P(y=5) = 7% - extremes are rare, weak LV signal
        # New thresholds [-1.0, -0.35, 0.35, 1.0] give more balanced:
        #   P(y=1) ≈ 16%, P(y=5) ≈ 16% - stronger signal for LV estimation
        # These can be overridden in config via 'measurement.thresholds'
        default_thresholds = [-1.0, -0.35, 0.35, 1.0]
        self.thresholds = tuple(self.config.get('thresholds', default_thresholds))

        # ISSUE #22 FIX: Configurable Likert scale for reverse coding
        # Default is 5-point scale, but can be changed via config
        self.likert_scale = int(self.config.get('likert_scale', 5))

        # Load items configuration
        items_path = self.config.get('items_path')
        if items_path and Path(items_path).exists():
            self.items_df = pd.read_csv(items_path)
        else:
            self.items_df = None

    def generate_indicators(
        self,
        latent_vars: Dict[str, float],
        demographics: Dict[str, int] = None
    ) -> Tuple[Dict[str, int], Dict[str, float]]:
        """
        Generate all indicator responses for an agent.

        Implements MIMIC (Multiple Indicators Multiple Causes) model where
        demographics can have direct effects on indicators in addition to
        their indirect effect through latent variables.

        Args:
            latent_vars: Dict of latent variable values {lv_name: value}
            demographics: Dict of demographic values for direct effects (optional)
                         Supported keys: age_idx, edu_idx, income_indiv_idx

        Returns:
            Tuple of (ordinal_responses, continuous_responses)
        """
        ordinal = {}
        continuous = {}
        demographics = demographics or {}

        if self.items_df is None:
            return ordinal, continuous

        # Define mapping from demographic variables to config columns
        demo_columns = [
            ('age_idx', 'direct_age'),
            ('edu_idx', 'direct_edu'),
            ('income_indiv_idx', 'direct_income')
        ]

        for _, item in self.items_df.iterrows():
            lv_name = self._factor_to_lv_name(item['factor'])
            lv_value = latent_vars.get(lv_name, 0.0)
            loading = float(item['loading'])

            # Calculate direct demographic effects (MIMIC component)
            direct_effect = 0.0
            for demo_var, config_col in demo_columns:
                if config_col in item.index:
                    coef = item[config_col]
                    if pd.notna(coef) and coef != 0:
                        demo_val = float(demographics.get(demo_var, 0))
                        direct_effect += float(coef) * demo_val

            # Determine indicator type
            indicator_type = item.get('type', 'ordinal')
            if pd.isna(indicator_type):
                indicator_type = 'ordinal'

            item_name = str(item['item_name'])

            if indicator_type == 'continuous':
                # Continuous: y = λ*η + γ*demo + ε (MIMIC)
                error_var = float(item.get('error_var', 1.0))
                y = loading * lv_value + direct_effect + np.sqrt(error_var) * self.rng.standard_normal()
                continuous[item_name] = y

            elif indicator_type == 'binary':
                # Binary probit: P(y=1) = Φ(λ*η + γ*demo) (MIMIC)
                y_star = loading * lv_value + direct_effect + self.rng.standard_normal()
                ordinal[item_name] = 1 if y_star > 0 else 0

            else:  # ordinal (default)
                # Ordinal probit: y* = λ*η + γ*demo + ε (MIMIC)
                y_star = loading * lv_value + direct_effect + self.rng.standard_normal()
                response = self._ordinal_from_continuous(y_star)

                # ISSUE #22 FIX: Configurable reverse coding
                # Uses self.likert_scale (default 5) instead of hardcoded 6
                if int(item.get('reverse', 0)) == 1:
                    response = (self.likert_scale + 1) - response

                ordinal[item_name] = response

        return ordinal, continuous

    def _ordinal_from_continuous(self, y_star: float) -> int:
        """Convert continuous latent response to ordinal scale."""
        t1, t2, t3, t4 = self.thresholds
        if y_star <= t1: return 1
        if y_star <= t2: return 2
        if y_star <= t3: return 3
        if y_star <= t4: return 4
        return 5

    @staticmethod
    def _factor_to_lv_name(factor: str) -> str:
        """Map factor name to latent variable name.

        Supports both short names (blind, constructive, dl, fp) and
        full LV names (pat_blind, pat_constructive, sec_dl, sec_fp).
        """
        fac = str(factor).strip().lower()
        mapping = {
            # Short names
            'blind': 'pat_blind',
            'constructive': 'pat_constructive',
            'daily': 'sec_dl',
            'dl': 'sec_dl',
            'dailylife': 'sec_dl',
            'faith': 'sec_fp',
            'fp': 'sec_fp',
            'faithandprayer': 'sec_fp',
            'faith_prayer': 'sec_fp',
            # Full LV names (passthrough)
            'pat_blind': 'pat_blind',
            'pat_constructive': 'pat_constructive',
            'sec_dl': 'sec_dl',
            'sec_fp': 'sec_fp',
        }
        return mapping.get(fac, fac)


# =============================================================================
# STRUCTURAL MODEL CLASS
# =============================================================================

class StructuralModel:
    """
    Handles the structural equations for latent variables.

    LV = α + Σ(γ * demographics) + ζ
    where ζ ~ N(0, σ²)

    Supports:
    - Multiple latent variables
    - Correlation between latent variable errors
    """

    def __init__(self, config: Dict, rng: np.random.Generator):
        self.config = config.get('latent', {})
        self.rng = rng
        self.lv_names = self.config.get('names', [])
        self.structural = self.config.get('structural', {})

        # Build correlation matrix for latent variable errors
        self.correlation_matrix = self._build_correlation_matrix()
        self.cholesky = np.linalg.cholesky(self.correlation_matrix) if len(self.lv_names) > 1 else None

    def _build_correlation_matrix(self) -> np.ndarray:
        """Build correlation matrix for latent variable errors."""
        n = len(self.lv_names)
        if n == 0:
            return np.array([[1.0]])

        # Check if correlation structure is specified
        corr_spec = self.config.get('correlations', {})

        # Start with identity matrix
        corr = np.eye(n)

        # Fill in specified correlations
        for i, name_i in enumerate(self.lv_names):
            for j, name_j in enumerate(self.lv_names):
                if i < j:
                    key = f"{name_i}_{name_j}"
                    alt_key = f"{name_j}_{name_i}"
                    rho = corr_spec.get(key, corr_spec.get(alt_key, 0.0))
                    corr[i, j] = rho
                    corr[j, i] = rho

        return corr

    def compute_latent_variables(self, demographics: Dict[str, int]) -> Dict[str, float]:
        """
        Compute latent variables using structural equations with correlated errors.
        """
        n = len(self.lv_names)
        if n == 0:
            return {}

        # Draw correlated standard normal errors
        if self.cholesky is not None:
            z = self.rng.standard_normal(n)
            correlated_errors = self.cholesky @ z
        else:
            correlated_errors = self.rng.standard_normal(n)

        latent_vars = {}

        for idx, lv_name in enumerate(self.lv_names):
            spec = self.structural.get(lv_name, {})

            # Intercept
            mu = float(spec.get('intercept', 0.0))

            # Demographic effects
            for demo_var, gamma in spec.get('betas', {}).items():
                if demo_var in demographics:
                    mu += float(gamma) * float(demographics[demo_var])

            # Add scaled error
            sigma = float(spec.get('sigma', 1.0))
            error = correlated_errors[idx] if isinstance(correlated_errors, np.ndarray) else correlated_errors

            latent_vars[lv_name] = mu + sigma * error

        return latent_vars


# =============================================================================
# RANDOM COEFFICIENTS HANDLER
# =============================================================================

class RandomCoefficientsHandler:
    """
    Manages random coefficient draws with optional correlation.

    Supports:
    - Independent random coefficients
    - Correlated random coefficients (via Cholesky)
    - Various distributions per coefficient
    """

    def __init__(self, config: Dict, rng: np.random.Generator):
        self.rng = rng
        self.specs: Dict[str, RandomCoeffSpec] = {}
        self.correlation_matrix: Optional[np.ndarray] = None
        self.cholesky: Optional[np.ndarray] = None
        self.coef_names: List[str] = []

        # Parse random coefficient specifications
        rc_config = config.get('random_coefficients', {})

        for name, spec in rc_config.get('coefficients', {}).items():
            dist_name = spec.get('distribution', 'fixed')
            self.specs[name] = RandomCoeffSpec(
                name=name,
                distribution=Distribution(dist_name),
                mean=float(spec.get('mean', 0.0)),
                std=float(spec.get('std', 0.0)),
                lower=spec.get('lower'),
                upper=spec.get('upper'),
                mode=spec.get('mode')
            )
            self.coef_names.append(name)

        # Build correlation structure
        if self.coef_names:
            self._build_correlation_structure(rc_config.get('correlations', {}))

    def _build_correlation_structure(self, corr_spec: Dict) -> None:
        """Build Cholesky decomposition for correlated draws."""
        n = len(self.coef_names)
        if n == 0:
            return

        # Start with identity
        corr = np.eye(n)

        # Fill correlations
        for i, name_i in enumerate(self.coef_names):
            for j, name_j in enumerate(self.coef_names):
                if i < j:
                    key = f"{name_i}_{name_j}"
                    alt_key = f"{name_j}_{name_i}"
                    rho = corr_spec.get(key, corr_spec.get(alt_key, 0.0))
                    corr[i, j] = rho
                    corr[j, i] = rho

        self.correlation_matrix = corr

        # Cholesky decomposition for correlated draws
        try:
            self.cholesky = np.linalg.cholesky(corr)
        except np.linalg.LinAlgError:
            print("Warning: Correlation matrix not positive definite, using independent draws")
            self.cholesky = np.eye(n)

    def draw_random_components(self) -> Dict[str, float]:
        """
        Draw random components (η) for all random coefficients.

        These are the deviations from the systematic mean, drawn once per agent
        and used consistently across all choice tasks (panel effect).
        """
        if not self.coef_names:
            return {}

        n = len(self.coef_names)

        # Draw correlated standard normals
        z = self.rng.standard_normal(n)
        if self.cholesky is not None:
            correlated_z = self.cholesky @ z
        else:
            correlated_z = z

        # Transform to desired distributions
        random_draws = {}
        for idx, name in enumerate(self.coef_names):
            spec = self.specs[name]

            if spec.distribution == Distribution.FIXED:
                random_draws[name] = 0.0

            elif spec.distribution == Distribution.NORMAL:
                # η ~ N(0, σ²), already have correlated z
                random_draws[name] = spec.std * correlated_z[idx]

            elif spec.distribution == Distribution.LOGNORMAL:
                # Lognormal random coefficient
                # We parameterize by the desired mean and std of the coefficient distribution
                #
                # CRITICAL: For consistency with other distributions, we return a
                # ZERO-MEAN random component η, not the full coefficient.
                # The systematic mean is added separately in _combine_taste_parameters():
                #   β_i = β_systematic + η_i, where E[η_i] = 0
                #
                # For lognormal: X ~ LogN(μ_u, σ_u²) has E[X] = exp(μ_u + σ_u²/2)
                # So centered: η = X - E[X]
                var = spec.std ** 2
                if spec.mean > 0:
                    # Standard lognormal for positive coefficients
                    mu_u = np.log(spec.mean ** 2 / np.sqrt(var + spec.mean ** 2))
                    sigma_u = np.sqrt(np.log(1 + var / spec.mean ** 2))
                    # Raw lognormal draw
                    raw_lognormal = np.exp(mu_u + sigma_u * correlated_z[idx])
                    # Center at zero: η = X - E[X] where E[X] = exp(μ_u + σ_u²/2)
                    expected_value = np.exp(mu_u + sigma_u ** 2 / 2)
                    random_draws[name] = raw_lognormal - expected_value
                elif spec.mean < 0:
                    # NEGATIVE lognormal: -exp(normal) is always negative
                    # Parameterize using |mean| and std
                    abs_mean = abs(spec.mean)
                    mu_u = np.log(abs_mean ** 2 / np.sqrt(var + abs_mean ** 2))
                    sigma_u = np.sqrt(np.log(1 + var / abs_mean ** 2))
                    # Raw negative lognormal draw
                    raw_neg_lognormal = -np.exp(mu_u + sigma_u * correlated_z[idx])
                    # Center at zero: η = X - E[X] where E[X] = -exp(μ_u + σ_u²/2)
                    expected_value = -np.exp(mu_u + sigma_u ** 2 / 2)
                    random_draws[name] = raw_neg_lognormal - expected_value
                else:
                    # mean = 0: use normal instead (lognormal undefined for mean 0)
                    random_draws[name] = spec.std * correlated_z[idx]

            elif spec.distribution == Distribution.TRIANGULAR:
                # Use inverse CDF transform
                p = stats.norm.cdf(correlated_z[idx])
                left = spec.lower if spec.lower is not None else spec.mean - spec.std * np.sqrt(6)
                right = spec.upper if spec.upper is not None else spec.mean + spec.std * np.sqrt(6)
                mode = spec.mode if spec.mode is not None else spec.mean
                value = stats.triang.ppf(p, c=(mode - left) / (right - left),
                                         loc=left, scale=right - left)
                random_draws[name] = value - spec.mean

            elif spec.distribution == Distribution.UNIFORM:
                p = stats.norm.cdf(correlated_z[idx])
                left = spec.lower if spec.lower is not None else spec.mean - spec.std * np.sqrt(3)
                right = spec.upper if spec.upper is not None else spec.mean + spec.std * np.sqrt(3)
                value = left + p * (right - left)
                random_draws[name] = value - spec.mean

            elif spec.distribution == Distribution.TRUNCATED_NORMAL:
                lower = spec.lower if spec.lower is not None else -np.inf
                upper = spec.upper if spec.upper is not None else np.inf
                a = (lower - spec.mean) / spec.std if spec.std > 0 else -np.inf
                b = (upper - spec.mean) / spec.std if spec.std > 0 else np.inf
                p = stats.norm.cdf(correlated_z[idx])
                # Truncate p to valid range
                p_lower = stats.norm.cdf(a)
                p_upper = stats.norm.cdf(b)
                p_adjusted = p_lower + p * (p_upper - p_lower)
                value = stats.norm.ppf(p_adjusted) * spec.std + spec.mean
                random_draws[name] = value - spec.mean

            else:
                random_draws[name] = 0.0

        return random_draws

    def get_distribution_mean(self, name: str) -> float:
        """Get the mean of a random coefficient's distribution."""
        if name in self.specs:
            return self.specs[name].mean
        return 0.0


# =============================================================================
# ENHANCED POPULATION CLASS
# =============================================================================

class Population:
    """
    Enhanced population manager with random coefficients support.
    """

    def __init__(self, config: Dict, rng: np.random.Generator):
        self.config = config
        self.rng = rng
        self.agents: List[Agent] = []
        self.n_agents = config['population']['N']

        # Initialize sub-models
        self.structural_model = StructuralModel(config, rng)
        self.measurement_model = MeasurementModel(config, rng)
        self.random_coef_handler = RandomCoefficientsHandler(config, rng)

    def create_agents(self) -> None:
        """Generate all agents."""
        print(f"Creating {self.n_agents} agents with random coefficients...")

        for i in range(1, self.n_agents + 1):
            agent = self._create_single_agent(i)
            self.agents.append(agent)

        print(f"Created {len(self.agents)} agents successfully.")
        self._print_coefficient_summary()

    def _create_single_agent(self, agent_id: int) -> Agent:
        """Create a single agent with all components."""

        # 1. Draw demographics
        demographics = self._draw_demographics()

        # 2. Compute latent variables (structural model)
        latent_vars = self.structural_model.compute_latent_variables(demographics)

        # 3. Compute systematic part of taste parameters
        systematic_betas = self._compute_systematic_betas(demographics, latent_vars)

        # 4. Draw random components (for mixed logit)
        random_draws = self.random_coef_handler.draw_random_components()

        # 5. Combine systematic + random to get final taste parameters
        taste_params = self._combine_taste_parameters(systematic_betas, random_draws)

        # 6. Generate indicator responses (measurement model with MIMIC direct effects)
        likert_responses, continuous_indicators = self.measurement_model.generate_indicators(
            latent_vars=latent_vars,
            demographics=demographics  # Pass demographics for MIMIC direct effects
        )

        return Agent(
            agent_id=agent_id,
            agent_id_str=f"SYN_{agent_id:06d}",
            demographics=demographics,
            latent_variables=latent_vars,
            systematic_betas=systematic_betas,
            random_draws=random_draws,
            taste_parameters=taste_params,
            likert_responses=likert_responses,
            continuous_indicators=continuous_indicators
        )

    def _draw_demographics(self) -> Dict[str, int]:
        """Draw demographics from population distributions.

        ISSUE #21 FIX: Now supports demographic correlations.

        Correlations can be specified in config under 'demographic_correlations':
        {
            "age_idx_income_indiv_idx": 0.3,  # Older people tend to earn more
            "edu_idx_income_indiv_idx": 0.4,  # Education correlates with income
        }

        Implementation uses a Gaussian copula approach:
        1. Draw correlated normal random variables
        2. Transform to uniform via CDF
        3. Map to discrete categories via inverse CDF
        """
        demographics = {}
        var_names = list(self.config['demographics'].keys())
        n_vars = len(var_names)

        # Check for correlation specification
        corr_spec = self.config.get('demographic_correlations', {})

        if not corr_spec or n_vars <= 1:
            # No correlations - use independent draws (original behavior)
            for var_name, spec in self.config['demographics'].items():
                if spec['type'] == 'categorical':
                    values = np.array(spec['values'])
                    probs = np.array(spec['probs'], dtype=float)
                    probs = probs / probs.sum()
                    demographics[var_name] = int(self.rng.choice(values, p=probs))
            return demographics

        # Build correlation matrix for Gaussian copula
        corr_matrix = np.eye(n_vars)
        for i, name_i in enumerate(var_names):
            for j, name_j in enumerate(var_names):
                if i < j:
                    key = f"{name_i}_{name_j}"
                    alt_key = f"{name_j}_{name_i}"
                    rho = corr_spec.get(key, corr_spec.get(alt_key, 0.0))
                    corr_matrix[i, j] = rho
                    corr_matrix[j, i] = rho

        # Cholesky decomposition for correlated draws
        try:
            chol = np.linalg.cholesky(corr_matrix)
        except np.linalg.LinAlgError:
            # Correlation matrix not positive definite - fall back to independent
            print("Warning: Demographic correlation matrix not positive definite, using independent draws")
            for var_name, spec in self.config['demographics'].items():
                if spec['type'] == 'categorical':
                    values = np.array(spec['values'])
                    probs = np.array(spec['probs'], dtype=float)
                    probs = probs / probs.sum()
                    demographics[var_name] = int(self.rng.choice(values, p=probs))
            return demographics

        # Draw correlated standard normals
        z = self.rng.standard_normal(n_vars)
        correlated_z = chol @ z

        # Transform to uniforms and then to categories
        uniforms = stats.norm.cdf(correlated_z)

        for idx, var_name in enumerate(var_names):
            spec = self.config['demographics'][var_name]
            if spec['type'] == 'categorical':
                values = np.array(spec['values'])
                probs = np.array(spec['probs'], dtype=float)
                probs = probs / probs.sum()

                # Inverse CDF: find category based on uniform value
                u = uniforms[idx]
                cumprob = np.cumsum(probs)
                category_idx = np.searchsorted(cumprob, u)
                category_idx = min(category_idx, len(values) - 1)
                demographics[var_name] = int(values[category_idx])

        return demographics

    def _compute_systematic_betas(
        self,
        demographics: Dict[str, int],
        latent_vars: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Compute systematic (deterministic) part of taste parameters.

        β_systematic = base + Σ(interaction_coef * covariate)
        """
        systematic = {}
        choice_cfg = self.config['choice_model']

        for term_spec in choice_cfg.get('attribute_terms', []):
            term_name = term_spec['name']

            # Start with base coefficient
            beta = float(term_spec.get('base_coef', 0.0))

            # Add interaction effects
            for interaction in term_spec.get('interactions', []):
                cov_name = interaction['with']
                coef = float(interaction.get('coef', 0.0))

                # Sign enforcement on interaction coefficient
                if interaction.get('enforce_sign') == 'positive':
                    coef = abs(coef)
                elif interaction.get('enforce_sign') == 'negative':
                    coef = -abs(coef)

                # Get covariate value
                if cov_name in demographics:
                    z = float(demographics[cov_name])
                elif cov_name in latent_vars:
                    z = float(latent_vars[cov_name])
                else:
                    continue

                # Centering and scaling
                if 'center' in interaction and interaction['center'] is not None:
                    z -= float(interaction['center'])
                if 'scale' in interaction and interaction['scale'] not in (None, 0):
                    z /= float(interaction['scale'])

                beta += coef * z

            # Sign enforcement on systematic beta
            # NOTE: This ensures the SYSTEMATIC component has the correct sign.
            # Individual coefficients may cross zero if random variation is large.
            #
            # BEST PRACTICE for strictly-signed coefficients:
            # - Use Distribution.LOGNORMAL for positive means (always positive)
            # - Use Distribution.LOGNORMAL for negative means (always negative, -exp())
            # - Or use 'enforce_sign_final': true to enforce after random term (truncates)
            #
            # References:
            # - Train (2009): Discrete Choice Methods with Simulation, Ch. 6
            # - Hensher et al. (2015): Applied Choice Analysis, 2nd ed.
            enforce = term_spec.get('enforce_sign')
            if enforce == 'positive':
                beta = abs(beta)
            elif enforce == 'negative':
                beta = -abs(beta)

            systematic[term_name] = beta

        return systematic

    def _combine_taste_parameters(
        self,
        systematic: Dict[str, float],
        random_draws: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Combine systematic and random components.

        β_i = β_systematic + η_i

        For coefficients with sign constraints, we apply the constraint
        to the final combined value.
        """
        combined = {}
        choice_cfg = self.config['choice_model']

        for term_spec in choice_cfg.get('attribute_terms', []):
            term_name = term_spec['name']

            # Get systematic part
            beta_sys = systematic.get(term_name, 0.0)

            # Add random part if specified
            eta = random_draws.get(term_name, 0.0)

            # Combine: β_i = β_systematic + η_i
            beta = beta_sys + eta

            # Optional final sign enforcement (truncates distribution)
            # Enable with 'enforce_sign_final': true in term_spec
            # WARNING: This truncates the distribution and may bias variance estimates
            # Prefer using bounded distributions (lognormal, truncated_normal) instead
            if term_spec.get('enforce_sign_final', False):
                enforce = term_spec.get('enforce_sign')
                if enforce == 'positive':
                    beta = max(0.0, beta)  # Truncate at 0
                elif enforce == 'negative':
                    beta = min(0.0, beta)  # Truncate at 0

            combined[term_name] = beta

        return combined

    def _print_coefficient_summary(self) -> None:
        """Print summary statistics of taste parameters."""
        if not self.agents:
            return

        print("\nTaste Parameter Summary:")
        print("-" * 60)

        # Collect all beta values
        beta_names = list(self.agents[0].taste_parameters.keys())

        for name in beta_names:
            values = [a.taste_parameters[name] for a in self.agents]
            print(f"  {name}:")
            print(f"    Mean: {np.mean(values):.4f}, Std: {np.std(values):.4f}")
            print(f"    Min: {np.min(values):.4f}, Max: {np.max(values):.4f}")

    def get_agent(self, agent_id: int) -> Agent:
        return self.agents[agent_id - 1]

    def __len__(self) -> int:
        return len(self.agents)

    def __iter__(self):
        return iter(self.agents)


# =============================================================================
# CHOICE MODEL CLASS
# =============================================================================

class ChoiceModel:
    """
    Discrete choice model for utility calculation and choice simulation.
    """

    def __init__(self, config: Dict, rng: np.random.Generator):
        self.config = config['choice_model']
        self.rng = rng
        self.fee_scale = float(self.config.get('fee_scale', 10000.0))
        self.alternatives = self.config['alts']

    def compute_utility(
        self,
        agent: Agent,
        scenario: pd.Series,
        alternative: str
    ) -> float:
        """Compute deterministic utility for an alternative."""
        U = 0.0

        # Base terms (ASCs)
        for bt in self.config.get('base_terms', []):
            if alternative in bt['apply_to']:
                coef = float(bt['coef'])
                x = self._get_term_value(bt['term'], scenario, alternative)
                U += coef * x

        # Attribute terms with individual-specific betas
        for at in self.config.get('attribute_terms', []):
            if alternative not in at['apply_to']:
                continue

            term_name = at['name']
            x = self._get_term_value(at['term'], scenario, alternative)
            beta = agent.taste_parameters.get(term_name, 0.0)
            U += beta * x

        return U

    def _get_term_value(self, term: str, scenario: pd.Series, alternative: str) -> float:
        """Get attribute value for a term."""
        alt_map = {'paid1': '1', 'paid2': '2', 'standard': '3'}
        suffix = alt_map.get(alternative, '3')

        if term == 'const':
            return 1.0

        if term.startswith('fee'):
            fee = float(scenario[f'fee{suffix}'])
            if term == 'fee10k':
                return fee / 10000.0
            elif term == 'fee100k':
                return fee / 100000.0
            return fee / self.fee_scale

        if term == 'dur':
            return float(scenario[f'dur{suffix}'])

        raise ValueError(f"Unknown term: {term}")

    def get_choice_probabilities(self, utilities: List[float]) -> np.ndarray:
        """Compute softmax probabilities."""
        u = np.array(utilities)
        u = u - np.max(u)
        exp_u = np.exp(u)
        return exp_u / exp_u.sum()

    def simulate_choice(self, agent: Agent, scenario: pd.Series) -> Tuple[int, np.ndarray, List[float]]:
        """
        Simulate a choice.

        Returns:
            Tuple of (choice, probabilities, utilities)
        """
        utilities = []
        for alt_code in ['1', '2', '3']:
            alt_name = self.alternatives[alt_code]
            U = self.compute_utility(agent, scenario, alt_name)
            utilities.append(U)

        probs = self.get_choice_probabilities(utilities)
        choice = int(self.rng.choice([1, 2, 3], p=probs))

        return choice, probs, utilities


# =============================================================================
# SIMULATOR CLASS
# =============================================================================

class DCMSimulatorAdvanced:
    """
    Advanced simulator with Mixed Logit and HCM support.
    """

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config(config_path)

        seed = int(self.config['population']['seed'])
        self.rng = np.random.default_rng(seed)

        self.population = Population(self.config, self.rng)
        self.choice_model = ChoiceModel(self.config, self.rng)
        self.design = self._load_design()

        self.results: List[Dict] = []
        self.n_tasks = int(self.config['population']['T'])

    def _load_config(self, path: str) -> Dict:
        """Load configuration file."""
        with open(path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        required = ['population', 'design', 'demographics', 'latent', 'choice_model']
        missing = [k for k in required if k not in config]
        if missing:
            raise ValueError(f"Config missing: {missing}")

        return config

    def _load_design(self) -> pd.DataFrame:
        """Load experimental design."""
        design_path = self.config['design']['path']
        if not Path(design_path).exists():
            raise FileNotFoundError(f"Design file not found: {design_path}")

        df = pd.read_csv(design_path)
        required_cols = ['scenario_id', 'dur1', 'fee1', 'dur2', 'fee2', 'dur3', 'fee3']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Design missing columns: {missing}")

        design = df.drop_duplicates('scenario_id').reset_index(drop=True)
        print(f"Loaded {len(design)} unique scenarios")
        return design

    def run(self, keep_latent: bool = False, keep_utilities: bool = False) -> pd.DataFrame:
        """Run the simulation."""
        print("\n" + "=" * 60)
        print("ADVANCED AGENT-BASED DCM SIMULATION")
        print("Mixed Logit + Hybrid Choice Model")
        print("=" * 60)

        self.population.create_agents()

        if len(self.design) < self.n_tasks:
            raise ValueError(f"Need {self.n_tasks} scenarios, only {len(self.design)} available")

        print(f"\nSimulating {self.n_tasks} choice tasks per agent...")

        sample_replace = not self.config['design'].get('sample_without_replacement', True)

        for agent in self.population:
            self._simulate_agent_choices(agent, keep_latent, keep_utilities, sample_replace)

        results_df = pd.DataFrame(self.results)
        self._print_summary(results_df)

        return results_df

    def _simulate_agent_choices(
        self,
        agent: Agent,
        keep_latent: bool,
        keep_utilities: bool,
        replace: bool
    ) -> None:
        """Simulate choices for an agent."""
        sampled = self.design.sample(
            n=self.n_tasks,
            replace=replace,
            random_state=int(self.rng.integers(1, 1_000_000_000))
        )

        for task, (_, scenario) in enumerate(sampled.iterrows(), start=1):
            choice, probs, utilities = self.choice_model.simulate_choice(agent, scenario)
            record = self._build_record(agent, scenario, task, choice, probs, utilities,
                                       keep_latent, keep_utilities)
            self.results.append(record)

    def _build_record(
        self,
        agent: Agent,
        scenario: pd.Series,
        task: int,
        choice: int,
        probs: np.ndarray,
        utilities: List[float],
        keep_latent: bool,
        keep_utilities: bool
    ) -> Dict:
        """Build output record."""
        record = {
            'ID_STR': agent.agent_id_str,
            'ID': agent.agent_id,
            'task': task,
            'scenario_id': int(scenario['scenario_id']),
            'dur1': float(scenario['dur1']),
            'fee1': float(scenario['fee1']),
            'dur2': float(scenario['dur2']),
            'fee2': float(scenario['fee2']),
            'dur3': float(scenario['dur3']),
            'fee3': float(scenario['fee3']),
            'CHOICE': choice,
        }

        # Demographics
        record.update(agent.demographics)

        # Likert responses
        record.update(agent.likert_responses)

        # Continuous indicators
        record.update(agent.continuous_indicators)

        # Optional: true values
        if keep_latent:
            record.update(agent.get_true_latents())
            record.update(agent.get_true_betas())

        # Optional: utilities and probabilities
        if keep_utilities:
            for i, (u, p) in enumerate(zip(utilities, probs), start=1):
                record[f'U_{i}'] = u
                record[f'P_{i}'] = p

        return record

    def _print_summary(self, df: pd.DataFrame) -> None:
        """Print simulation summary."""
        print("\n" + "=" * 60)
        print("SIMULATION COMPLETE")
        print("=" * 60)
        print(f"Total observations: {len(df):,}")
        print(f"Unique respondents: {df['ID'].nunique():,}")
        print(f"Tasks per respondent: {self.n_tasks}")
        print(f"\nChoice shares:")
        shares = df['CHOICE'].value_counts(normalize=True).sort_index()
        for choice, share in shares.items():
            alt_name = self.choice_model.alternatives[str(choice)]
            print(f"  Alternative {choice} ({alt_name}): {share:.1%}")

    def export(self, output_path: str, keep_latent: bool = False, keep_utilities: bool = False) -> None:
        """Run and export to CSV."""
        df = self.run(keep_latent=keep_latent, keep_utilities=keep_utilities)
        df.to_csv(output_path, index=False)
        print(f"\nExported to: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Advanced Agent-Based DCM Simulator (Mixed Logit + HCM)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--config', required=True, help='Path to JSON config')
    parser.add_argument('--out', default='synthetic_dcm_advanced.csv', help='Output CSV')
    parser.add_argument('--keep_latent', action='store_true', help='Include true latent values')
    parser.add_argument('--keep_utilities', action='store_true', help='Include utilities and probabilities')

    args = parser.parse_args()

    simulator = DCMSimulatorAdvanced(args.config)
    simulator.export(args.out, keep_latent=args.keep_latent, keep_utilities=args.keep_utilities)


if __name__ == '__main__':
    main()
