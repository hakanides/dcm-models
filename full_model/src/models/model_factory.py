"""
Model Factory for DCM Estimation
=================================

Provides a unified interface for creating and running DCM models.

Usage:
    from src.models.model_factory import ModelFactory

    # List available models
    ModelFactory.list_models()

    # Create a specific model
    logprob, name = ModelFactory.create('mnl_basic', database)

    # Run estimation
    results = ModelFactory.estimate('mnl_basic', database)

    # Run multiple models
    all_results = ModelFactory.run_comparison(['mnl_basic', 'mnl_demo'], database)

Authors: Hakan Mülayim, Giray Girengir, Ataol Azeritürk
"""

import warnings
from typing import Dict, List, Tuple, Optional, Callable
import pandas as pd
import numpy as np

import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta, Variable


class ModelFactory:
    """Factory class for creating and managing DCM models."""

    # Registry of available models
    _registry: Dict[str, Callable] = {}
    _descriptions: Dict[str, str] = {}

    @classmethod
    def register(cls, name: str, description: str = ""):
        """Decorator to register a model function."""
        def decorator(func: Callable):
            cls._registry[name] = func
            cls._descriptions[name] = description or func.__doc__ or ""
            return func
        return decorator

    @classmethod
    def list_models(cls) -> pd.DataFrame:
        """List all available models with descriptions."""
        data = [
            {'Model': name, 'Description': cls._descriptions.get(name, '')}
            for name in sorted(cls._registry.keys())
        ]
        return pd.DataFrame(data)

    @classmethod
    def create(cls, name: str, database: db.Database) -> Tuple:
        """
        Create a model specification.

        Args:
            name: Model name (use list_models() to see available)
            database: Biogeme database

        Returns:
            Tuple of (logprob, model_name)

        Raises:
            ValueError: If model name not found
        """
        if name not in cls._registry:
            available = ', '.join(sorted(cls._registry.keys()))
            raise ValueError(f"Model '{name}' not found. Available: {available}")

        return cls._registry[name](database)

    @classmethod
    def estimate(cls, name: str, database: db.Database,
                 n_draws: Optional[int] = None,
                 warm_start: Optional[Dict] = None) -> Dict:
        """
        Create and estimate a model.

        Args:
            name: Model name
            database: Biogeme database
            n_draws: Number of draws for simulation (MXL only)
            warm_start: Initial parameter values

        Returns:
            Dict with estimation results
        """
        logprob, model_name = cls.create(name, database)

        if n_draws:
            biogeme_obj = bio.BIOGEME(database, logprob, number_of_draws=n_draws)
        else:
            biogeme_obj = bio.BIOGEME(database, logprob)

        if warm_start:
            biogeme_obj.change_init_values(warm_start)

        biogeme_obj.model_name = model_name.replace(' ', '_').replace('-', '_')

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            results = biogeme_obj.estimate()

        betas = results.get_beta_values()
        std_errs = {}
        for p in betas:
            try:
                std_errs[p] = results.get_parameter_std_err(p)
            except:
                std_errs[p] = np.nan

        return {
            'name': model_name,
            'll': results.final_loglikelihood,
            'k': results.number_of_free_parameters,
            'aic': results.akaike_information_criterion,
            'bic': results.bayesian_information_criterion,
            'betas': betas,
            'std_errs': std_errs,
            'converged': results.algorithm_has_converged,
            'biogeme_results': results
        }

    @classmethod
    def run_comparison(cls, model_names: List[str], database: db.Database,
                       null_ll: Optional[float] = None,
                       warm_start_chain: bool = True) -> pd.DataFrame:
        """
        Run multiple models and compare results.

        Args:
            model_names: List of model names to compare
            database: Biogeme database
            null_ll: Null log-likelihood for rho-squared
            warm_start_chain: Use previous estimates as starting values

        Returns:
            DataFrame with comparison results
        """
        results = []
        previous_betas = None

        for name in model_names:
            print(f"\nEstimating {name}...")

            warm_start = previous_betas if warm_start_chain else None
            result = cls.estimate(name, database, warm_start=warm_start)

            if null_ll:
                result['rho2'] = 1 - (result['ll'] / null_ll)
            else:
                result['rho2'] = np.nan

            results.append({
                'Model': result['name'],
                'LL': result['ll'],
                'K': result['k'],
                'AIC': result['aic'],
                'BIC': result['bic'],
                'rho2': result['rho2'],
                'Converged': result['converged']
            })

            previous_betas = result['betas']

            print(f"  LL: {result['ll']:.2f} | K: {result['k']} | AIC: {result['aic']:.2f}")

        return pd.DataFrame(results)


# =============================================================================
# REGISTER STANDARD MODELS
# =============================================================================

# Helper function to get common variables
def _get_vars():
    return {
        'dur1': Variable('dur1'),
        'dur2': Variable('dur2'),
        'dur3': Variable('dur3'),
        'fee1_10k': Variable('fee1_10k'),
        'fee2_10k': Variable('fee2_10k'),
        'fee3_10k': Variable('fee3_10k'),
        'CHOICE': Variable('CHOICE'),
    }


@ModelFactory.register('mnl_basic', 'Basic MNL with fee and duration')
def model_mnl_basic(database: db.Database):
    """Basic MNL: ASC + fee + duration only."""
    v = _get_vars()

    ASC_paid = Beta('ASC_paid', 1.0, None, None, 0)
    B_FEE = Beta('B_FEE', -0.5, -10, 0, 0)
    B_DUR = Beta('B_DUR', -0.05, -5, 0, 0)

    V1 = ASC_paid + B_FEE * v['fee1_10k'] + B_DUR * v['dur1']
    V2 = ASC_paid + B_FEE * v['fee2_10k'] + B_DUR * v['dur2']
    V3 = B_FEE * v['fee3_10k'] + B_DUR * v['dur3']

    V = {1: V1, 2: V2, 3: V3}
    av = {1: 1, 2: 1, 3: 1}

    return models.loglogit(V, av, v['CHOICE']), 'MNL-Basic'


@ModelFactory.register('mnl_demo', 'MNL with demographic interactions')
def model_mnl_demographics(database: db.Database):
    """MNL with demographic interactions on fee sensitivity."""
    v = _get_vars()

    # Demographics
    age_c = Variable('age_c')
    edu_c = Variable('edu_c')
    inc_c = Variable('inc_c')
    inc_hh_c = Variable('inc_hh_c')

    # Base parameters
    ASC_paid = Beta('ASC_paid', 1.0, None, None, 0)
    B_FEE = Beta('B_FEE', -0.5, -10, 0, 0)
    B_FEE_AGE = Beta('B_FEE_AGE', 0.05, -2, 2, 0)
    B_FEE_EDU = Beta('B_FEE_EDU', 0.05, -2, 2, 0)
    B_FEE_INC = Beta('B_FEE_INC', 0.10, -2, 2, 0)
    B_FEE_INC_H = Beta('B_FEE_INC_H', 0.05, -2, 2, 0)
    B_DUR = Beta('B_DUR', -0.05, -5, 0, 0)

    # Individual-specific fee coefficient
    B_FEE_i = B_FEE + B_FEE_AGE * age_c + B_FEE_EDU * edu_c + B_FEE_INC * inc_c + B_FEE_INC_H * inc_hh_c

    V1 = ASC_paid + B_FEE_i * v['fee1_10k'] + B_DUR * v['dur1']
    V2 = ASC_paid + B_FEE_i * v['fee2_10k'] + B_DUR * v['dur2']
    V3 = B_FEE_i * v['fee3_10k'] + B_DUR * v['dur3']

    V = {1: V1, 2: V2, 3: V3}
    av = {1: 1, 2: 1, 3: 1}

    return models.loglogit(V, av, v['CHOICE']), 'MNL-Demographics'


@ModelFactory.register('hcm_basic', 'HCM with LV proxies (patriotism + secularism)')
def model_hcm_basic(database: db.Database):
    """HCM with latent variable proxies affecting fee and duration."""
    v = _get_vars()

    # LV proxies (must be pre-computed in data)
    pat_proxy = Variable('pat_blind_proxy')
    sec_proxy = Variable('sec_dl_proxy')

    # Parameters
    ASC_paid = Beta('ASC_paid', 1.0, None, None, 0)
    B_FEE = Beta('B_FEE', -0.5, -10, 0, 0)
    B_DUR = Beta('B_DUR', -0.05, -5, 0, 0)

    B_FEE_PAT = Beta('B_FEE_PAT', -0.05, -2, 2, 0)
    B_FEE_SEC = Beta('B_FEE_SEC', 0.02, -2, 2, 0)
    B_DUR_PAT = Beta('B_DUR_PAT', 0.02, -1, 1, 0)
    B_DUR_SEC = Beta('B_DUR_SEC', -0.02, -1, 1, 0)

    # Individual-specific coefficients
    B_FEE_i = B_FEE + B_FEE_PAT * pat_proxy + B_FEE_SEC * sec_proxy
    B_DUR_i = B_DUR + B_DUR_PAT * pat_proxy + B_DUR_SEC * sec_proxy

    V1 = ASC_paid + B_FEE_i * v['fee1_10k'] + B_DUR_i * v['dur1']
    V2 = ASC_paid + B_FEE_i * v['fee2_10k'] + B_DUR_i * v['dur2']
    V3 = B_FEE_i * v['fee3_10k'] + B_DUR_i * v['dur3']

    V = {1: V1, 2: V2, 3: V3}
    av = {1: 1, 2: 1, 3: 1}

    return models.loglogit(V, av, v['CHOICE']), 'HCM-Basic'


# Print available models on import
if __name__ == '__main__':
    print("Available models:")
    print(ModelFactory.list_models().to_string(index=False))
