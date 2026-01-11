"""
Base Classes for Policy Analysis
=================================

Core data structures and configuration for DCM policy analysis.

Classes:
- EstimationResult: Standardized container for model results
- PolicyScenario: Defines attribute levels for policy scenarios
- PolicyAnalysisConfig: Configuration parameters

Authors: Hakan Mülayim, Giray Girengir, Ataol Azeritürk
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from scipy import stats


@dataclass
class PolicyAnalysisConfig:
    """
    Configuration for policy analysis calculations.

    Attributes:
        fee_scale: Scaling factor for fee (default 10000, fees divided by this)
        confidence_level: Confidence level for intervals (default 0.95)
        n_simulations: Number of simulations for Monte Carlo methods
        seed: Random seed for reproducibility

        Parameter names (matching estimation output):
        - fee_param: Name of fee coefficient
        - dur_param: Name of duration coefficient
        - asc_param: Name of ASC for paid alternatives
    """
    fee_scale: float = 10000.0
    confidence_level: float = 0.95
    n_simulations: int = 1000
    seed: int = 42

    # Parameter naming conventions
    fee_param: str = 'B_FEE'
    dur_param: str = 'B_DUR'
    asc_param: str = 'ASC_paid'

    # MXL parameter names
    fee_mean_param: str = 'B_FEE_MU'
    fee_std_param: str = 'B_FEE_SIGMA'
    dur_mean_param: str = 'B_DUR_MU'
    dur_std_param: str = 'B_DUR_SIGMA'

    # Alternative configuration
    # NOTE: reference_alternative is 0-indexed
    # For 3-alternative model: 0=Alt1, 1=Alt2, 2=Alt3 (typically free/standard)
    # This should match the model specification where ASC=0 for reference
    n_alternatives: int = 3
    reference_alternative: int = 2  # 0-indexed, alt 3 is reference (no ASC)

    @classmethod
    def from_config_json(cls, config_path: str) -> 'PolicyAnalysisConfig':
        """
        Load configuration from a model's config.json file.

        This ensures policy analysis uses the same configuration
        as the estimation model (fee_scale, reference alternative, etc.).

        Args:
            config_path: Path to config.json

        Returns:
            PolicyAnalysisConfig instance
        """
        import json
        from pathlib import Path

        with open(Path(config_path)) as f:
            config = json.load(f)

        choice_cfg = config.get('choice_model', {})
        alts_cfg = choice_cfg.get('alternatives', {})

        # Find reference alternative (the one with asc=false)
        ref_alt = 2  # default
        for alt_key, alt_val in alts_cfg.items():
            if not alt_val.get('asc', True):
                ref_alt = int(alt_key) - 1  # Convert to 0-indexed

        return cls(
            fee_scale=choice_cfg.get('fee_scale', 10000.0),
            n_alternatives=len(alts_cfg) if alts_cfg else 3,
            reference_alternative=ref_alt
        )

    @property
    def z_score(self) -> float:
        """Z-score for confidence intervals."""
        return stats.norm.ppf(1 - (1 - self.confidence_level) / 2)


@dataclass
class EstimationResult:
    """
    Standardized container for estimation results.

    Converts various result formats to a common interface for policy analysis.

    Attributes:
        betas: Dictionary of parameter estimates
        std_errs: Dictionary of standard errors
        covariance_matrix: Full covariance matrix (optional)
        param_names: Ordered list of parameter names
        converged: Whether estimation converged
        n_obs: Number of observations
        ll: Log-likelihood
        model_type: Type of model ('mnl', 'mxl', 'hcm')
    """
    betas: Dict[str, float]
    std_errs: Dict[str, float]
    covariance_matrix: Optional[np.ndarray] = None
    param_names: List[str] = field(default_factory=list)
    converged: bool = True
    n_obs: int = 0
    ll: float = 0.0
    model_type: str = 'mnl'

    # For MXL: random coefficient specifications
    random_params: Dict[str, Dict[str, str]] = field(default_factory=dict)

    def __post_init__(self):
        if not self.param_names:
            self.param_names = list(self.betas.keys())

    @classmethod
    def from_dict(cls, result_dict: Dict[str, Any]) -> 'EstimationResult':
        """Create from standard result dictionary format."""
        return cls(
            betas=result_dict.get('betas', {}),
            std_errs=result_dict.get('std_errs', {}),
            converged=result_dict.get('converged', True),
            n_obs=result_dict.get('n_obs', 0),
            ll=result_dict.get('ll', 0.0),
            model_type=result_dict.get('model_type', 'mnl')
        )

    def get_covariance(self, params: List[str],
                        warn_on_diagonal: bool = True) -> np.ndarray:
        """
        Extract covariance submatrix for specified parameters.

        If full covariance not available, builds diagonal matrix from std_errs.
        This assumes zero parameter correlations, which typically
        UNDERSTATES standard errors for derived quantities like WTP.

        Args:
            params: List of parameter names
            warn_on_diagonal: If True (default), warn when falling back to diagonal

        Returns:
            Covariance submatrix

        Note:
            For accurate policy analysis (WTP, elasticities), the full
            covariance matrix should be extracted from Biogeme results:
                result.getVarCovar()  # or result.varCovar in newer versions
        """
        n = len(params)

        if self.covariance_matrix is not None:
            # Extract submatrix
            indices = []
            for p in params:
                if p in self.param_names:
                    indices.append(self.param_names.index(p))
                else:
                    return self._diagonal_covariance(params, warn_on_diagonal)
            return self.covariance_matrix[np.ix_(indices, indices)]
        else:
            return self._diagonal_covariance(params, warn_on_diagonal)

    def _diagonal_covariance(self, params: List[str],
                             warn: bool = True) -> np.ndarray:
        """
        Build diagonal covariance from standard errors.

        WARNING: Assumes zero parameter correlations. This typically
        UNDERSTATES standard errors for ratios/derived quantities because:
        - Numerator and denominator are often negatively correlated
        - Ignoring this correlation inflates variance estimates for WTP

        For accurate SEs, provide full covariance matrix from estimation.
        """
        import warnings
        if warn:
            warnings.warn(
                "Full covariance matrix not available. Using diagonal matrix "
                "(zero correlations). Standard errors for derived quantities "
                "(WTP, elasticities) may be UNDERSTATED. For accurate SEs, "
                "extract full covariance from Biogeme: result.getVarCovar()",
                UserWarning
            )

        n = len(params)
        cov = np.zeros((n, n))
        for i, p in enumerate(params):
            se = self.std_errs.get(p, 0)
            cov[i, i] = se ** 2
        return cov

    def get_t_stat(self, param: str) -> float:
        """Compute t-statistic for a parameter."""
        beta = self.betas.get(param, 0)
        se = self.std_errs.get(param, np.nan)
        if se and se > 0:
            return beta / se
        return np.nan

    def is_significant(self, param: str, alpha: float = 0.05) -> bool:
        """Check if parameter is statistically significant."""
        t_stat = self.get_t_stat(param)
        if np.isnan(t_stat):
            return False
        critical = stats.norm.ppf(1 - alpha / 2)
        return abs(t_stat) > critical


@dataclass
class PolicyScenario:
    """
    Defines a policy scenario with attribute levels.

    Attributes:
        name: Descriptive name for the scenario
        attributes: Dictionary mapping attribute names to arrays of values
                   e.g., {'fee': [500000, 600000, 0], 'dur': [10, 8, 15]}
        n_alternatives: Number of choice alternatives
        demographics: Optional DataFrame with individual demographics
        weights: Optional weights for aggregation
    """
    name: str
    attributes: Dict[str, np.ndarray]
    n_alternatives: int = 3
    demographics: Optional[pd.DataFrame] = None
    weights: Optional[np.ndarray] = None

    def __post_init__(self):
        # Convert lists to arrays
        for key, value in self.attributes.items():
            if isinstance(value, list):
                self.attributes[key] = np.array(value)

    def get_attribute(self, name: str, alternative: int) -> float:
        """Get attribute value for a specific alternative."""
        if name in self.attributes:
            return self.attributes[name][alternative]
        return 0.0

    def with_modification(self,
                          attribute: str,
                          alternative: int,
                          new_value: float) -> 'PolicyScenario':
        """Create new scenario with one attribute modified."""
        new_attrs = {k: v.copy() for k, v in self.attributes.items()}
        new_attrs[attribute][alternative] = new_value
        return PolicyScenario(
            name=f"{self.name}_modified",
            attributes=new_attrs,
            n_alternatives=self.n_alternatives,
            demographics=self.demographics,
            weights=self.weights
        )

    def with_percentage_change(self,
                               attribute: str,
                               alternative: int,
                               pct_change: float) -> 'PolicyScenario':
        """Create scenario with percentage change to attribute."""
        current = self.get_attribute(attribute, alternative)
        new_value = current * (1 + pct_change / 100)
        return self.with_modification(attribute, alternative, new_value)


def compute_logit_probabilities(utilities: np.ndarray) -> np.ndarray:
    """
    Compute logit choice probabilities from utilities.

    P_j = exp(V_j) / sum_k exp(V_k)

    Uses max subtraction for numerical stability.
    """
    V_shifted = utilities - np.max(utilities)
    exp_V = np.exp(V_shifted)
    return exp_V / np.sum(exp_V)


def compute_utilities(scenario: PolicyScenario,
                      result: EstimationResult,
                      config: PolicyAnalysisConfig) -> np.ndarray:
    """
    Compute deterministic utilities for a scenario.

    V_j = ASC_j + B_FEE * fee_j/scale + B_DUR * dur_j

    Reference alternative (last) has no ASC.
    """
    n_alts = scenario.n_alternatives

    asc = result.betas.get(config.asc_param, 0)
    b_fee = result.betas.get(config.fee_param, 0)
    b_dur = result.betas.get(config.dur_param, 0)

    fees = scenario.attributes.get('fee', np.zeros(n_alts))
    fees_scaled = fees / config.fee_scale
    durs = scenario.attributes.get('dur', np.zeros(n_alts))

    V = np.zeros(n_alts)
    for j in range(n_alts):
        if j != config.reference_alternative:
            V[j] = asc + b_fee * fees_scaled[j] + b_dur * durs[j]
        else:
            V[j] = b_fee * fees_scaled[j] + b_dur * durs[j]

    return V


def delta_method_se(func_value: float,
                    gradient: np.ndarray,
                    covariance: np.ndarray) -> float:
    """
    Compute standard error using delta method.

    SE = sqrt(g' * Cov * g)

    Args:
        func_value: Value of the function (not used, for compatibility)
        gradient: Gradient vector of function w.r.t. parameters
        covariance: Covariance matrix of parameters

    Returns:
        Standard error
    """
    var = gradient @ covariance @ gradient
    return np.sqrt(max(var, 0))


def krinsky_robb_draws(result: EstimationResult,
                       params: List[str],
                       n_draws: int,
                       seed: int = 42) -> np.ndarray:
    """
    Generate Krinsky-Robb parameter draws from multivariate normal.

    Returns:
        Array of shape (n_draws, n_params)
    """
    mean = np.array([result.betas.get(p, 0) for p in params])
    cov = result.get_covariance(params)

    np.random.seed(seed)
    return np.random.multivariate_normal(mean, cov, n_draws)
