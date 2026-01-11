"""
MNL Model Comparison Framework
==============================

This script estimates and compares multiple MNL model specifications:

1. Model 1: Basic MNL (ASC + main attributes only)
2. Model 2: MNL with demographic interactions on fee
3. Model 3: MNL with demographic interactions on all attributes
4. Model 4: MNL with Likert items as proxies for latent variables
5. Model 5: MNL with demographics + Likert proxies (full specification)

Comparison metrics:
- Log-likelihood (LL)
- Null log-likelihood (LL0)
- Rho-squared and adjusted rho-squared
- AIC (Akaike Information Criterion)
- BIC (Bayesian Information Criterion)
- Likelihood Ratio Tests (for nested models)

Authors: Hakan Mülayim, Giray Girengir, Ataol Azeritürk
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil
from pathlib import Path
from scipy import stats
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import warnings
# Selective warning suppression - allow important warnings through
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*overflow.*')
warnings.filterwarnings('ignore', message='.*divide by zero.*')


def cleanup_results_directory(output_dir: Path):
    """Delete all files in output directory before new run."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
        print(f"Cleaned up: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def cleanup_iter_files(project_root: Path = None):
    """Delete all .iter files from project root for clean estimation."""
    if project_root is None:
        project_root = Path.cwd()
    iter_files = list(project_root.glob("__*.iter"))
    if iter_files:
        for iter_file in iter_files:
            iter_file.unlink()
        print(f"Deleted {len(iter_files)} .iter files from project root")

# Biogeme imports
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
# Handle Biogeme API changes (bioDraws deprecated in 3.3+, use Draws)
try:
    from biogeme.expressions import Beta, Variable, Draws, MonteCarlo, log, exp
except ImportError:
    from biogeme.expressions import Beta, Variable, bioDraws as Draws, MonteCarlo, log, exp


# =============================================================================
# DATA PREPARATION
# =============================================================================

def load_and_prepare_data(data_path: str) -> Tuple[pd.DataFrame, db.Database]:
    """Load data and prepare for Biogeme."""
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} observations from {df['ID'].nunique()} respondents")

    # Create derived variables
    df = df.copy()

    # Scale fees (divide by 10k)
    df['fee1_10k'] = df['fee1'] / 10000.0
    df['fee2_10k'] = df['fee2'] / 10000.0
    df['fee3_10k'] = df['fee3'] / 10000.0

    # Center demographics for interactions
    df['age_c'] = (df['age_idx'] - 2) / 2
    df['edu_c'] = (df['edu_idx'] - 3) / 2
    df['inc_c'] = (df['income_indiv_idx'] - 3) / 2
    df['inc_hh_c'] = (df['income_house_idx'] - 3) / 2
    df['marital_c'] = (df['marital_idx'] - 0.5) / 0.5

    # Create Likert scale indices (average of items per construct)
    # Use unified naming with fallback to old naming
    def get_construct_items(df, domain, start, end, fallback_prefix):
        """Get items for construct with fallback to old naming."""
        unified = [f'{domain}_{i}' for i in range(start, end + 1)
                   if f'{domain}_{i}' in df.columns and not f'{domain}_{i}'.endswith('_cont')]
        if unified:
            return unified
        return [c for c in df.columns if c.startswith(fallback_prefix) and not c.endswith('_cont')]

    # Patriotism - Blind (patriotism_1-10 or pat_blind_*)
    pat_blind_cols = get_construct_items(df, 'patriotism', 1, 10, 'pat_blind_')
    if pat_blind_cols:
        df['pat_blind_idx'] = df[pat_blind_cols].mean(axis=1)
        df['pat_blind_idx_c'] = (df['pat_blind_idx'] - 3) / 2  # Center around midpoint

    # Patriotism - Constructive (patriotism_11-20 or pat_constructive_*)
    pat_const_cols = get_construct_items(df, 'patriotism', 11, 20, 'pat_constructive_')
    if pat_const_cols:
        df['pat_const_idx'] = df[pat_const_cols].mean(axis=1)
        df['pat_const_idx_c'] = (df['pat_const_idx'] - 3) / 2

    # Secularism - Daily Life (secularism_1-15 or sec_dl_*)
    sec_dl_cols = get_construct_items(df, 'secularism', 1, 15, 'sec_dl_')
    if sec_dl_cols:
        df['sec_dl_idx'] = df[sec_dl_cols].mean(axis=1)
        df['sec_dl_idx_c'] = (df['sec_dl_idx'] - 3) / 2

    # Secularism - Faith & Prayer (secularism_16-25 or sec_fp_*)
    sec_fp_cols = get_construct_items(df, 'secularism', 16, 25, 'sec_fp_')
    if sec_fp_cols:
        df['sec_fp_idx'] = df[sec_fp_cols].mean(axis=1)
        df['sec_fp_idx_c'] = (df['sec_fp_idx'] - 3) / 2

    # Drop string columns for Biogeme
    string_cols = df.select_dtypes(include=['object']).columns.tolist()
    df_numeric = df.drop(columns=string_cols)

    # Create database
    database = db.Database('dcm_data', df_numeric)

    return df, database


# =============================================================================
# MODEL SPECIFICATIONS
# =============================================================================

def get_common_variables():
    """Define common variables used across models."""
    return {
        'dur1': Variable('dur1'),
        'dur2': Variable('dur2'),
        'dur3': Variable('dur3'),
        'fee1_10k': Variable('fee1_10k'),
        'fee2_10k': Variable('fee2_10k'),
        'fee3_10k': Variable('fee3_10k'),
        'CHOICE': Variable('CHOICE'),
        # Centered demographics
        'age_c': Variable('age_c'),
        'edu_c': Variable('edu_c'),
        'inc_c': Variable('inc_c'),
        'inc_hh_c': Variable('inc_hh_c'),
        'marital_c': Variable('marital_c'),
        # Likert indices
        'pat_blind_idx_c': Variable('pat_blind_idx_c'),
        'pat_const_idx_c': Variable('pat_const_idx_c'),
        'sec_dl_idx_c': Variable('sec_dl_idx_c'),
        'sec_fp_idx_c': Variable('sec_fp_idx_c'),
    }


def model_1_basic(database: db.Database):
    """
    Model 1: Basic MNL

    Only ASCs and main attribute effects (fee, duration).
    No interactions or heterogeneity.
    NOTE: Exemption removed - constant per alternative, absorbed into ASC_paid.
    """
    v = get_common_variables()

    # Parameters with bounds for stable convergence
    # Note: fee is scaled by 10k, so B_FEE ~ -0.03 to -0.08
    ASC_paid = Beta('ASC_paid', 1.0, -10, 10, 0)
    B_FEE = Beta('B_FEE', -0.05, -1, 0, 0)      # Bounded negative, realistic start
    B_DUR = Beta('B_DUR', -0.05, -1, 0, 0)      # Bounded negative, realistic start

    # Utilities
    V1 = ASC_paid + B_FEE * v['fee1_10k'] + B_DUR * v['dur1']
    V2 = ASC_paid + B_FEE * v['fee2_10k'] + B_DUR * v['dur2']
    V3 = B_FEE * v['fee3_10k'] + B_DUR * v['dur3']

    V = {1: V1, 2: V2, 3: V3}
    av = {1: 1, 2: 1, 3: 1}

    logprob = models.loglogit(V, av, v['CHOICE'])

    return logprob, 'Model 1: Basic MNL'


def model_2_demo_fee(database: db.Database):
    """
    Model 2: MNL with demographic interactions on fee coefficient

    Fee sensitivity varies by age, education, and income.
    """
    v = get_common_variables()

    # Base parameters with bounds for stable convergence
    ASC_paid = Beta('ASC_paid', 1.0, -10, 10, 0)
    B_FEE = Beta('B_FEE', -0.05, -1, 0, 0)
    B_DUR = Beta('B_DUR', -0.05, -1, 0, 0)

    # Fee interaction parameters
    B_FEE_AGE = Beta('B_FEE_AGE', 0, None, None, 0)
    B_FEE_EDU = Beta('B_FEE_EDU', 0, None, None, 0)
    B_FEE_INC = Beta('B_FEE_INC', 0, None, None, 0)

    # Individual-specific fee coefficient
    B_FEE_i = B_FEE + B_FEE_AGE * v['age_c'] + B_FEE_EDU * v['edu_c'] + B_FEE_INC * v['inc_c']

    # Utilities
    V1 = ASC_paid + B_FEE_i * v['fee1_10k'] + B_DUR * v['dur1']
    V2 = ASC_paid + B_FEE_i * v['fee2_10k'] + B_DUR * v['dur2']
    V3 = B_FEE_i * v['fee3_10k'] + B_DUR * v['dur3']

    V = {1: V1, 2: V2, 3: V3}
    av = {1: 1, 2: 1, 3: 1}

    logprob = models.loglogit(V, av, v['CHOICE'])

    return logprob, 'Model 2: MNL + Demo interactions (fee)'


def model_3_demo_all(database: db.Database):
    """
    Model 3: MNL with demographic interactions on fee and duration

    Fee and duration coefficients vary by demographics.
    """
    v = get_common_variables()

    # Base parameters with bounds for stable convergence
    ASC_paid = Beta('ASC_paid', 1.0, -10, 10, 0)
    B_FEE = Beta('B_FEE', -0.05, -1, 0, 0)
    B_DUR = Beta('B_DUR', -0.05, -1, 0, 0)

    # Fee interactions
    B_FEE_AGE = Beta('B_FEE_AGE', 0, None, None, 0)
    B_FEE_EDU = Beta('B_FEE_EDU', 0, None, None, 0)
    B_FEE_INC = Beta('B_FEE_INC', 0, None, None, 0)

    # Duration interactions
    B_DUR_EDU = Beta('B_DUR_EDU', 0, None, None, 0)
    B_DUR_INC = Beta('B_DUR_INC', 0, None, None, 0)
    B_DUR_MARITAL = Beta('B_DUR_MARITAL', 0, None, None, 0)

    # Individual-specific coefficients
    B_FEE_i = B_FEE + B_FEE_AGE * v['age_c'] + B_FEE_EDU * v['edu_c'] + B_FEE_INC * v['inc_c']
    B_DUR_i = B_DUR + B_DUR_EDU * v['edu_c'] + B_DUR_INC * v['inc_c'] + B_DUR_MARITAL * v['marital_c']

    # Utilities
    V1 = ASC_paid + B_FEE_i * v['fee1_10k'] + B_DUR_i * v['dur1']
    V2 = ASC_paid + B_FEE_i * v['fee2_10k'] + B_DUR_i * v['dur2']
    V3 = B_FEE_i * v['fee3_10k'] + B_DUR_i * v['dur3']

    V = {1: V1, 2: V2, 3: V3}
    av = {1: 1, 2: 1, 3: 1}

    logprob = models.loglogit(V, av, v['CHOICE'])

    return logprob, 'Model 3: MNL + Demo interactions (all)'


def model_4_likert_proxy(database: db.Database):
    """
    Model 4: MNL with Likert items as proxies for latent variables

    Uses averaged Likert scales as direct predictors (proxy approach).
    This is a "shortcut" that ignores measurement error.
    """
    v = get_common_variables()

    # Base parameters with bounds for stable convergence
    ASC_paid = Beta('ASC_paid', 1.0, -10, 10, 0)
    B_FEE = Beta('B_FEE', -0.05, -1, 0, 0)
    B_DUR = Beta('B_DUR', -0.05, -1, 0, 0)

    # Likert proxy interactions with fee
    B_FEE_PAT_BLIND = Beta('B_FEE_PAT_BLIND', 0, None, None, 0)
    B_FEE_PAT_CONST = Beta('B_FEE_PAT_CONST', 0, None, None, 0)
    B_FEE_SEC_DL = Beta('B_FEE_SEC_DL', 0, None, None, 0)
    B_FEE_SEC_FP = Beta('B_FEE_SEC_FP', 0, None, None, 0)

    # Likert proxy interactions with duration
    B_DUR_PAT_BLIND = Beta('B_DUR_PAT_BLIND', 0, None, None, 0)
    B_DUR_SEC_DL = Beta('B_DUR_SEC_DL', 0, None, None, 0)

    # Individual-specific coefficients
    B_FEE_i = (B_FEE +
               B_FEE_PAT_BLIND * v['pat_blind_idx_c'] +
               B_FEE_PAT_CONST * v['pat_const_idx_c'] +
               B_FEE_SEC_DL * v['sec_dl_idx_c'] +
               B_FEE_SEC_FP * v['sec_fp_idx_c'])

    B_DUR_i = (B_DUR +
               B_DUR_PAT_BLIND * v['pat_blind_idx_c'] +
               B_DUR_SEC_DL * v['sec_dl_idx_c'])

    # Utilities
    V1 = ASC_paid + B_FEE_i * v['fee1_10k'] + B_DUR_i * v['dur1']
    V2 = ASC_paid + B_FEE_i * v['fee2_10k'] + B_DUR_i * v['dur2']
    V3 = B_FEE_i * v['fee3_10k'] + B_DUR_i * v['dur3']

    V = {1: V1, 2: V2, 3: V3}
    av = {1: 1, 2: 1, 3: 1}

    logprob = models.loglogit(V, av, v['CHOICE'])

    return logprob, 'Model 4: MNL + Likert proxies'


def model_5_full_mnl(database: db.Database):
    """
    Model 5: Full MNL with demographics + Likert proxies

    Combines demographic interactions with Likert scale proxies.
    Most complex MNL specification before moving to mixed logit.
    """
    v = get_common_variables()

    # Base parameters with bounds for stable convergence
    ASC_paid = Beta('ASC_paid', 1.0, -10, 10, 0)
    B_FEE = Beta('B_FEE', -0.05, -1, 0, 0)
    B_DUR = Beta('B_DUR', -0.05, -1, 0, 0)

    # Fee - demographic interactions
    B_FEE_AGE = Beta('B_FEE_AGE', 0, None, None, 0)
    B_FEE_EDU = Beta('B_FEE_EDU', 0, None, None, 0)
    B_FEE_INC = Beta('B_FEE_INC', 0, None, None, 0)

    # Fee - Likert interactions
    B_FEE_PAT_BLIND = Beta('B_FEE_PAT_BLIND', 0, None, None, 0)
    B_FEE_SEC_DL = Beta('B_FEE_SEC_DL', 0, None, None, 0)

    # Duration - demographic interactions
    B_DUR_EDU = Beta('B_DUR_EDU', 0, None, None, 0)
    B_DUR_INC = Beta('B_DUR_INC', 0, None, None, 0)

    # Duration - Likert interactions
    B_DUR_PAT_BLIND = Beta('B_DUR_PAT_BLIND', 0, None, None, 0)
    B_DUR_SEC_DL = Beta('B_DUR_SEC_DL', 0, None, None, 0)

    # Individual-specific coefficients
    B_FEE_i = (B_FEE +
               B_FEE_AGE * v['age_c'] + B_FEE_EDU * v['edu_c'] + B_FEE_INC * v['inc_c'] +
               B_FEE_PAT_BLIND * v['pat_blind_idx_c'] + B_FEE_SEC_DL * v['sec_dl_idx_c'])

    B_DUR_i = (B_DUR +
               B_DUR_EDU * v['edu_c'] + B_DUR_INC * v['inc_c'] +
               B_DUR_PAT_BLIND * v['pat_blind_idx_c'] + B_DUR_SEC_DL * v['sec_dl_idx_c'])

    # Utilities
    V1 = ASC_paid + B_FEE_i * v['fee1_10k'] + B_DUR_i * v['dur1']
    V2 = ASC_paid + B_FEE_i * v['fee2_10k'] + B_DUR_i * v['dur2']
    V3 = B_FEE_i * v['fee3_10k'] + B_DUR_i * v['dur3']

    V = {1: V1, 2: V2, 3: V3}
    av = {1: 1, 2: 1, 3: 1}

    logprob = models.loglogit(V, av, v['CHOICE'])

    return logprob, 'Model 5: Full MNL (demo + Likert)'


# =============================================================================
# MODEL ESTIMATION
# =============================================================================

@dataclass
class ModelResults:
    """Container for model estimation results."""
    name: str
    log_likelihood: float
    null_log_likelihood: float
    n_parameters: int
    n_observations: int
    aic: float
    bic: float
    rho_squared: float
    adj_rho_squared: float
    parameters: Dict[str, float]
    std_errors: Dict[str, float]
    t_stats: Dict[str, float]
    biogeme_results: any
    converged: bool = True  # Track convergence status


def estimate_null_model(database: db.Database) -> float:
    """
    Calculate null model (equal probabilities) log-likelihood.

    For J alternatives with equal probabilities, LL0 = N * ln(1/J)
    """
    n_obs = len(database.dataframe)
    n_alts = 3  # Three alternatives
    null_ll = n_obs * np.log(1.0 / n_alts)
    return null_ll


def estimate_model(database: db.Database, model_func, null_ll: float, output_dir: Path = None) -> ModelResults:
    """Estimate a single model and compute fit statistics."""
    logprob, model_name = model_func(database)

    print(f"\nEstimating: {model_name}")
    print("-" * 50)

    biogeme_model = bio.BIOGEME(database, logprob)

    # Set model name with output directory path for HTML files
    safe_name = model_name.replace(':', '').replace(' ', '_').replace('-', '_')
    if output_dir:
        biogeme_model.model_name = str(output_dir / safe_name)
    else:
        biogeme_model.model_name = safe_name

    results = biogeme_model.estimate()

    # Check convergence status
    converged = results.algorithm_has_converged
    if not converged:
        print(f"  WARNING: Algorithm did not converge!")

    # Extract results
    ll = results.final_loglikelihood
    n_params = results.number_of_free_parameters
    n_obs = len(database.dataframe)

    # Fit statistics
    aic = results.akaike_information_criterion
    bic = results.bayesian_information_criterion
    rho_sq = 1 - (ll / null_ll)
    adj_rho_sq = 1 - ((ll - n_params) / null_ll)

    # Parameters
    betas = results.get_beta_values()
    std_errs = {}
    t_stats = {}

    for param_name in betas.keys():
        try:
            std_errs[param_name] = results.get_parameter_std_err(param_name)
            t_stats[param_name] = results.get_parameter_t_test(param_name)
        except:
            std_errs[param_name] = np.nan
            t_stats[param_name] = np.nan

    print(f"  Log-likelihood: {ll:.2f}")
    print(f"  Parameters: {n_params}")
    print(f"  AIC: {aic:.2f}")
    print(f"  BIC: {bic:.2f}")
    print(f"  Rho-squared: {rho_sq:.4f}")
    print(f"  Converged: {converged}")

    return ModelResults(
        name=model_name,
        log_likelihood=ll,
        null_log_likelihood=null_ll,
        n_parameters=n_params,
        n_observations=n_obs,
        aic=aic,
        bic=bic,
        rho_squared=rho_sq,
        adj_rho_squared=adj_rho_sq,
        parameters=betas,
        std_errors=std_errs,
        t_stats=t_stats,
        biogeme_results=results,
        converged=converged
    )


# =============================================================================
# MODEL COMPARISON
# =============================================================================

def likelihood_ratio_test(ll_restricted: float, ll_unrestricted: float,
                          df: int, validate: bool = True) -> Tuple[float, float]:
    """
    Perform likelihood ratio test.

    H0: Restricted model is adequate
    H1: Unrestricted model is better

    Args:
        ll_restricted: Log-likelihood of the restricted (simpler) model
        ll_unrestricted: Log-likelihood of the unrestricted (full) model
        df: Degrees of freedom (difference in number of parameters)
        validate: If True, return NaN for invalid tests

    Returns:
        Tuple of (LR statistic, p-value)

    Notes:
        - LR statistic should be non-negative for properly nested models
        - Negative values indicate: wrong nesting order, convergence issues,
          or models that are not truly nested
        - Very small LR (<0.01) may indicate parameters are not identified
    """
    # Input validation
    if df <= 0:
        return np.nan, np.nan

    if np.isnan(ll_restricted) or np.isnan(ll_unrestricted):
        return np.nan, np.nan

    lr_stat = -2 * (ll_restricted - ll_unrestricted)

    if validate:
        # Guard 1: LR should be positive for properly nested models
        if lr_stat < 0:
            # Possible causes:
            # 1. Models compared in wrong order (restricted/unrestricted swapped)
            # 2. Models not truly nested
            # 3. One or both models didn't converge properly
            # 4. Local optimum issues
            return np.nan, np.nan

        # Guard 2: Extremely large LR may indicate numerical issues
        if lr_stat > 10000:
            # This could indicate one model failed to converge
            return np.nan, np.nan

    p_value = 1 - stats.chi2.cdf(lr_stat, df)
    return lr_stat, p_value


def diagnose_lr_test(ll_restricted: float, ll_unrestricted: float,
                     k_restricted: int, k_unrestricted: int,
                     model_restricted: str = "restricted",
                     model_unrestricted: str = "unrestricted") -> Dict[str, Any]:
    """
    Diagnose potential issues with a likelihood ratio test.

    Returns detailed information about why an LR test might be invalid.
    """
    diagnosis = {
        'valid': True,
        'issues': [],
        'lr_stat': None,
        'df': None,
        'recommendation': None
    }

    df = k_unrestricted - k_restricted

    # Check nesting
    if df <= 0:
        diagnosis['valid'] = False
        diagnosis['issues'].append(
            f"Unrestricted model has fewer/equal parameters ({k_unrestricted}) "
            f"than restricted ({k_restricted})"
        )
        diagnosis['recommendation'] = "Swap restricted and unrestricted models"
        return diagnosis

    diagnosis['df'] = df

    # Check LL values
    if np.isnan(ll_restricted) or np.isnan(ll_unrestricted):
        diagnosis['valid'] = False
        diagnosis['issues'].append("One or both log-likelihoods are NaN")
        diagnosis['recommendation'] = "Check model convergence"
        return diagnosis

    lr_stat = -2 * (ll_restricted - ll_unrestricted)
    diagnosis['lr_stat'] = lr_stat

    if lr_stat < 0:
        diagnosis['valid'] = False
        diagnosis['issues'].append(
            f"Negative LR statistic ({lr_stat:.4f}): unrestricted model "
            f"has worse fit than restricted"
        )
        diagnosis['issues'].append(
            f"  LL({model_restricted}) = {ll_restricted:.2f}"
        )
        diagnosis['issues'].append(
            f"  LL({model_unrestricted}) = {ll_unrestricted:.2f}"
        )
        diagnosis['recommendation'] = (
            "Check that: (1) models are properly nested, "
            "(2) both converged to global optimum, "
            "(3) unrestricted model includes all restricted parameters"
        )
    elif lr_stat < 0.01:
        diagnosis['issues'].append(
            f"Very small LR statistic ({lr_stat:.4f}): additional parameters "
            "may not be identified or have minimal effect"
        )
    elif lr_stat > 1000:
        diagnosis['issues'].append(
            f"Very large LR statistic ({lr_stat:.2f}): may indicate "
            "convergence issues or model misspecification"
        )

    return diagnosis


def create_comparison_table(results_list: List[ModelResults]) -> pd.DataFrame:
    """Create comparison table for all models."""
    data = []

    for r in results_list:
        data.append({
            'Model': r.name,
            'LL': r.log_likelihood,
            'K': r.n_parameters,
            'AIC': r.aic,
            'BIC': r.bic,
            'ρ²': r.rho_squared,
            'Adj. ρ²': r.adj_rho_squared,
            'Converged': r.converged,
        })

    df = pd.DataFrame(data)

    # Add rankings (only for converged models)
    df['AIC Rank'] = df['AIC'].rank().astype(int)
    df['BIC Rank'] = df['BIC'].rank().astype(int)

    return df


def create_parameter_table(results_list: List[ModelResults]) -> pd.DataFrame:
    """Create table comparing parameter estimates across models."""
    # Get all unique parameters
    all_params = set()
    for r in results_list:
        all_params.update(r.parameters.keys())

    all_params = sorted(all_params)

    # Build table
    rows = []
    for param in all_params:
        row = {'Parameter': param}
        for r in results_list:
            if param in r.parameters:
                est = r.parameters[param]
                se = r.std_errors.get(param, np.nan)
                t = r.t_stats.get(param, np.nan)

                # Format as estimate (t-stat)
                if not np.isnan(t):
                    row[r.name] = f"{est:.4f} ({t:.2f})"
                else:
                    row[r.name] = f"{est:.4f}"
            else:
                row[r.name] = "-"
        rows.append(row)

    return pd.DataFrame(rows)


def run_lr_tests(results_list: List[ModelResults]) -> pd.DataFrame:
    """Run likelihood ratio tests between nested models.

    Only compares converged models. Returns NA for non-converged comparisons.
    """
    tests = []

    # Define nested model pairs (restricted, unrestricted)
    nested_pairs = [
        (0, 1, "Model 1 vs Model 2"),  # Basic vs Demo-Fee
        (0, 2, "Model 1 vs Model 3"),  # Basic vs Demo-All
        (1, 2, "Model 2 vs Model 3"),  # Demo-Fee vs Demo-All
        (0, 3, "Model 1 vs Model 4"),  # Basic vs Likert
        (0, 4, "Model 1 vs Model 5"),  # Basic vs Full
        (2, 4, "Model 3 vs Model 5"),  # Demo-All vs Full
        (3, 4, "Model 4 vs Model 5"),  # Likert vs Full
    ]

    for i_r, i_u, test_name in nested_pairs:
        if i_r < len(results_list) and i_u < len(results_list):
            r_res = results_list[i_r]
            u_res = results_list[i_u]

            df = u_res.n_parameters - r_res.n_parameters

            # Check convergence status
            both_converged = r_res.converged and u_res.converged

            if df > 0:
                if both_converged:
                    lr_stat, p_val = likelihood_ratio_test(
                        r_res.log_likelihood, u_res.log_likelihood, df
                    )
                    significant = 'Yes' if (not np.isnan(p_val) and p_val < 0.05) else 'No'
                    note = ''
                    if np.isnan(lr_stat):
                        note = 'Invalid (LL issue)'
                        significant = 'N/A'
                else:
                    lr_stat = np.nan
                    p_val = np.nan
                    significant = 'N/A'
                    note = 'Non-converged model(s)'

                tests.append({
                    'Test': test_name,
                    'Restricted LL': r_res.log_likelihood,
                    'Unrestricted LL': u_res.log_likelihood,
                    'df': df,
                    'LR Statistic': lr_stat,
                    'p-value': p_val,
                    'Significant': significant,
                    'Note': note
                })

    return pd.DataFrame(tests)


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_model_comparison(comparison_df: pd.DataFrame, output_path: str = None):
    """Create visual comparison of models."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    models = comparison_df['Model'].str.replace('Model ', 'M').str.replace(': ', '\n')
    x = np.arange(len(models))

    # Plot 1: Log-likelihood
    ax = axes[0, 0]
    bars = ax.bar(x, comparison_df['LL'], color='steelblue', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Log-Likelihood')
    ax.set_title('Log-Likelihood (higher is better)')
    ax.axhline(y=comparison_df['LL'].max(), color='red', linestyle='--', alpha=0.5)

    # Plot 2: AIC
    ax = axes[0, 1]
    colors = ['green' if r == 1 else 'steelblue' for r in comparison_df['AIC Rank']]
    bars = ax.bar(x, comparison_df['AIC'], color=colors, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('AIC')
    ax.set_title('AIC (lower is better)')

    # Plot 3: BIC
    ax = axes[1, 0]
    colors = ['green' if r == 1 else 'steelblue' for r in comparison_df['BIC Rank']]
    bars = ax.bar(x, comparison_df['BIC'], color=colors, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('BIC')
    ax.set_title('BIC (lower is better)')

    # Plot 4: Rho-squared
    ax = axes[1, 1]
    width = 0.35
    ax.bar(x - width/2, comparison_df['ρ²'], width, label='ρ²', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, comparison_df['Adj. ρ²'], width, label='Adj. ρ²', color='coral', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Rho-squared')
    ax.set_title('Goodness of Fit')
    ax.legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")

    plt.show()


def plot_parameter_comparison(results_list: List[ModelResults],
                             params_to_plot: List[str],
                             output_path: str = None):
    """Plot parameter estimates across models with confidence intervals."""
    n_params = len(params_to_plot)
    fig, axes = plt.subplots(1, n_params, figsize=(5*n_params, 5))

    if n_params == 1:
        axes = [axes]

    model_names = [r.name.replace('Model ', 'M').split(':')[0] for r in results_list]
    x = np.arange(len(model_names))

    for ax, param in zip(axes, params_to_plot):
        estimates = []
        errors = []

        for r in results_list:
            if param in r.parameters:
                estimates.append(r.parameters[param])
                errors.append(1.96 * r.std_errors.get(param, 0))
            else:
                estimates.append(np.nan)
                errors.append(0)

        ax.errorbar(x, estimates, yerr=errors, fmt='o', capsize=5,
                   markersize=8, color='steelblue')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_ylabel('Estimate')
        ax.set_title(param)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nParameter plot saved to: {output_path}")

    plt.show()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_model_comparison(data_path: str, output_dir: str = None):
    """
    Run complete model comparison.

    Estimates all MNL specifications and generates comparison outputs.
    """
    print("=" * 70)
    print("MNL MODEL COMPARISON FRAMEWORK")
    print("=" * 70)

    # Setup and cleanup
    cleanup_iter_files()  # Delete .iter files from project root

    if output_dir:
        output_dir = Path(output_dir)
        cleanup_results_directory(output_dir)  # Clean and recreate output directory
    else:
        output_dir = None

    # Load data
    df, database = load_and_prepare_data(data_path)

    # Estimate null model
    print("\nEstimating null model (equal probabilities)...")
    null_ll = estimate_null_model(database)
    print(f"  Null LL: {null_ll:.2f}")

    # Define models to estimate
    model_functions = [
        model_1_basic,
        model_2_demo_fee,
        model_3_demo_all,
        model_4_likert_proxy,
        model_5_full_mnl,
    ]

    # Estimate all models
    results_list = []
    for model_func in model_functions:
        try:
            result = estimate_model(database, model_func, null_ll, output_dir)
            results_list.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")

    # Create comparison table
    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)

    comparison_df = create_comparison_table(results_list)
    print("\n" + comparison_df.to_string(index=False))

    # Likelihood ratio tests
    print("\n" + "=" * 70)
    print("LIKELIHOOD RATIO TESTS")
    print("=" * 70)

    lr_tests_df = run_lr_tests(results_list)
    print("\n" + lr_tests_df.to_string(index=False))

    # Best model recommendation
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    best_aic = comparison_df.loc[comparison_df['AIC'].idxmin()]
    best_bic = comparison_df.loc[comparison_df['BIC'].idxmin()]

    print(f"\nBest by AIC: {best_aic['Model']} (AIC = {best_aic['AIC']:.2f})")
    print(f"Best by BIC: {best_bic['Model']} (BIC = {best_bic['BIC']:.2f})")

    if best_aic['Model'] == best_bic['Model']:
        print(f"\n*** Both criteria agree: {best_aic['Model']} is the best MNL specification ***")
    else:
        print(f"\n*** Criteria disagree: AIC prefers more complex, BIC prefers simpler ***")

    # Save outputs
    if output_dir:
        comparison_df.to_csv(output_dir / 'model_comparison.csv', index=False)
        lr_tests_df.to_csv(output_dir / 'likelihood_ratio_tests.csv', index=False)

        # Parameter table
        param_df = create_parameter_table(results_list)
        param_df.to_csv(output_dir / 'parameter_estimates.csv', index=False)

        # Plots
        plot_model_comparison(comparison_df, str(output_dir / 'model_comparison.png'))
        plot_parameter_comparison(
            results_list,
            ['B_FEE', 'B_DUR', 'ASC_paid'],  # B_EXEMPT removed - not identifiable
            str(output_dir / 'parameter_estimates.png')
        )

        print(f"\nAll outputs saved to: {output_dir}")

    return results_list, comparison_df, lr_tests_df


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Compare multiple MNL model specifications',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python mnl_model_comparison.py --data test_advanced_full.csv
  python mnl_model_comparison.py --data test_advanced_full.csv --output mnl_results
        """
    )
    parser.add_argument('--data', required=True, help='Path to data CSV')
    parser.add_argument('--output', default='mnl_comparison', help='Output directory')

    args = parser.parse_args()

    run_model_comparison(args.data, args.output)


if __name__ == '__main__':
    main()
