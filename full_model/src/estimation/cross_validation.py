"""
Cross-Validation for DCM Models
================================

Implements k-fold cross-validation for out-of-sample prediction assessment.

IMPORTANT: For panel data, folds are split by INDIVIDUAL (ID), not by observation.
This ensures all observations from the same person are in the same fold.

MODEL TYPE CONSIDERATIONS:
--------------------------
1. **MNL (Multinomial Logit)**: Standard CV works correctly since observations
   are conditionally independent given parameters.

2. **MXL (Mixed Logit)**: This module provides proper simulation-based test LL
   that integrates over the random coefficient distribution:
       LL = Σ_n log( (1/R) Σ_r Π_t P(y_nt | x_nt, β_n^r) )

   For MXL CV to work correctly, you MUST either:
   - Provide random_params dict specifying which parameters are random
   - Use naming convention B_*_MU / B_*_SIGMA for auto-detection

3. **HCM/ICLV (Hybrid Choice Models)**: WARNING - This module does NOT properly
   support ICLV models because it cannot integrate over latent variable
   distributions estimated in the measurement model. The test LL for ICLV
   is APPROXIMATE and should be interpreted with caution.

   For rigorous ICLV CV, you would need:
   - Access to the latent variable structural model
   - Integration over latent variable distributions conditional on indicators
   - This requires re-implementing the full ICLV likelihood on test data

IMPORTANT: If random parameter auto-detection fails, this module will RAISE
an error rather than silently falling back to point estimates, which would
give misleading CV results.

References:
- Train (2009): Discrete Choice Methods with Simulation, Ch. 6 & 11
- McFadden & Train (2000): "Mixed MNL Models for Discrete Response"

Usage:
    from src.estimation.cross_validation import cross_validate_model

    results = cross_validate_model(
        df=data,
        model_func=model_mnl_basic,
        n_folds=5,
        id_col='ID'
    )

Authors: Hakan Mülayim, Giray Girengir, Ataol Azeritürk
"""

import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Tuple, Optional
import warnings

try:
    from sklearn.model_selection import KFold, StratifiedKFold
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models


def split_by_individual(df: pd.DataFrame, n_folds: int = 5,
                        id_col: str = 'ID', seed: int = 42,
                        stratify: bool = True,
                        choice_col: str = 'CHOICE') -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create k-fold splits by individual ID (not by observation).

    This ensures all observations from the same person are in the same fold,
    which is critical for panel data to avoid data leakage.

    Args:
        df: DataFrame with panel data
        n_folds: Number of folds
        id_col: Column name for individual identifier
        seed: Random seed for reproducibility
        stratify: If True (default), use StratifiedKFold based on each
                  individual's modal choice. This ensures balanced choice
                  distributions across folds, which is important for:
                  - Avoiding bias in test LL estimates
                  - Ensuring rare choice alternatives appear in all folds
                  - More stable CV estimates across folds
        choice_col: Column name for choice variable (used if stratify=True)

    Returns:
        List of (train_indices, test_indices) tuples for each fold

    Note:
        Stratification is based on each individual's MODAL (most frequent) choice
        across their panel observations. If an individual has ties, the lowest
        alternative number is used.
    """
    if not HAS_SKLEARN:
        raise ImportError("sklearn is required for cross-validation. Install with: pip install scikit-learn")

    unique_ids = df[id_col].unique()

    if stratify and choice_col in df.columns:
        # Compute modal choice for each individual (for stratification)
        # This ensures balanced choice distribution across folds
        individual_modal_choice = df.groupby(id_col)[choice_col].agg(
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]
        )
        # Align with unique_ids order
        strat_labels = individual_modal_choice.loc[unique_ids].values

        # Check if we have enough samples per stratum
        unique_strata, counts = np.unique(strat_labels, return_counts=True)
        min_count = counts.min()

        if min_count < n_folds:
            warnings.warn(
                f"Choice alternative {unique_strata[counts.argmin()]} has only {min_count} "
                f"individuals, less than n_folds={n_folds}. Falling back to non-stratified KFold.",
                UserWarning
            )
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
            split_iterator = kf.split(unique_ids)
        else:
            kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            split_iterator = kf.split(unique_ids, strat_labels)
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        split_iterator = kf.split(unique_ids)

    folds = []
    for train_id_idx, test_id_idx in split_iterator:
        train_ids = unique_ids[train_id_idx]
        test_ids = unique_ids[test_id_idx]

        train_mask = df[id_col].isin(train_ids)
        test_mask = df[id_col].isin(test_ids)

        train_indices = df.index[train_mask].values
        test_indices = df.index[test_mask].values

        folds.append((train_indices, test_indices))

    return folds


def compute_log_likelihood(betas: Dict, df_test: pd.DataFrame,
                           utility_func: Callable,
                           id_col: str = 'ID',
                           model_type: str = 'mnl') -> float:
    """
    Compute log-likelihood on test data using estimated parameters.

    Args:
        betas: Dictionary of estimated parameter values
        df_test: Test data DataFrame
        utility_func: Function that computes utilities given betas and data
        id_col: Column name for individual identifier (for panel-aware reporting)
        model_type: One of 'mnl', 'mxl', 'hcm', 'iclv'. Used for warnings.
                    Default is 'mnl'.

    Returns:
        Log-likelihood on test data

    Note:
        For MNL models, this computes the exact conditional log-likelihood:
            LL = Σ_n Σ_t log(P(y_nt | x_nt, β))

        For MXL/HCM/ICLV models, this is an APPROXIMATION because:
        - MXL: Uses mean of random coefficients, ignoring heterogeneity
        - HCM/ICLV: Uses expected values of latent variables

        The true log-likelihood for these models requires:
            LL = Σ_n log( ∫ Π_t P(y_nt | x_nt, β_n) f(β_n) dβ_n )

        This integral cannot be evaluated without re-simulating on test data.
        For rigorous MXL/ICLV CV, use simulation-based log-likelihood.
    """
    # Warn about approximation for non-MNL models
    if model_type.lower() in ['mxl', 'hcm', 'iclv', 'mixed']:
        # Check if parameters suggest a mixed model
        has_sigma = any('sigma' in p.lower() or 'std' in p.lower() for p in betas.keys())
        has_lv = any('lv' in p.lower() or 'latent' in p.lower() or 'gamma' in p.lower()
                     for p in betas.keys())

        if has_sigma or has_lv:
            warnings.warn(
                f"Computing test LL for {model_type.upper()} model using mean coefficients. "
                f"This is an approximation that ignores taste heterogeneity. "
                f"True out-of-sample LL would require simulation integration.",
                UserWarning
            )

    # Compute utilities for each alternative
    V = utility_func(betas, df_test)

    # Compute choice probabilities (softmax)
    exp_V = {k: np.exp(v - np.max(list(V.values()), axis=0)) for k, v in V.items()}
    sum_exp = sum(exp_V.values())
    probs = {k: v / sum_exp for k, v in exp_V.items()}

    # Get probability of chosen alternative
    choice = df_test['CHOICE'].values
    prob_chosen = np.zeros(len(choice))
    for alt, prob in probs.items():
        mask = choice == alt
        prob_chosen[mask] = prob[mask]

    # Avoid log(0)
    prob_chosen = np.clip(prob_chosen, 1e-10, 1.0)

    return np.sum(np.log(prob_chosen))


def cross_validate_model(df: pd.DataFrame,
                         model_func: Callable,
                         utility_func: Callable,
                         n_folds: int = 5,
                         id_col: str = 'ID',
                         seed: int = 42,
                         verbose: bool = True,
                         model_type: str = 'mnl',
                         random_params: Dict = None,
                         n_sim_draws: int = 500) -> Dict:
    """
    Perform k-fold cross-validation for a DCM model.

    Args:
        df: Full dataset
        model_func: Function returning (logprob, name) for Biogeme
        utility_func: Function(betas, df) -> {alt: utility_array}
        n_folds: Number of cross-validation folds
        id_col: Column for individual identifier
        seed: Random seed
        verbose: Print progress
        model_type: One of 'mnl', 'mxl', 'hcm', 'iclv'. For MXL/HCM/ICLV,
                   uses simulation-based test LL (slower but unbiased).
                   Default is 'mnl' for speed.
        random_params: For MXL/ICLV models, dict specifying random parameters.
                      e.g., {'B_FEE': {'mean': 'B_FEE_MU', 'sigma': 'B_FEE_SIGMA', 'dist': 'normal'}}
                      If None and model_type is MXL/ICLV, will attempt to auto-detect.
        n_sim_draws: Number of simulation draws for MXL/ICLV test LL (default 500)

    Returns:
        Dict with:
        - 'train_ll': List of training log-likelihoods
        - 'test_ll': List of test log-likelihoods
        - 'train_ll_mean': Mean training LL
        - 'test_ll_mean': Mean test LL
        - 'overfit_ratio': train_ll / test_ll (>1 indicates overfitting)
        - 'fold_results': Detailed results per fold
        - 'test_ll_method': 'point_estimate' or 'simulated'
    """
    # Determine test LL computation method
    use_simulated_ll = model_type.lower() in ['mxl', 'hcm', 'iclv', 'mixed']
    test_ll_method = 'simulated' if use_simulated_ll else 'point_estimate'

    if verbose:
        print(f"\n{'='*60}")
        print(f"CROSS-VALIDATION ({n_folds}-fold, split by {id_col})")
        print(f"{'='*60}")
        print(f"Model type: {model_type.upper()}")
        print(f"Test LL method: {test_ll_method}")
        if use_simulated_ll:
            print(f"Simulation draws: {n_sim_draws}")

    # Warn about ICLV/HCM limitations
    if model_type.lower() in ['hcm', 'iclv']:
        warnings.warn(
            f"IMPORTANT: Cross-validation for {model_type.upper()} models is APPROXIMATE. "
            f"The test LL computed here does NOT properly integrate over latent variable "
            f"distributions. It only accounts for random coefficients, not the full "
            f"measurement model structure. Results should be interpreted with caution. "
            f"For rigorous ICLV CV, consider model-specific validation approaches.",
            UserWarning
        )

    folds = split_by_individual(df, n_folds, id_col, seed)

    train_lls = []
    test_lls = []
    fold_results = []

    # Drop string columns for Biogeme
    string_cols = df.select_dtypes(include=['object']).columns.tolist()
    df_num = df.drop(columns=string_cols)

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        if verbose:
            print(f"\nFold {fold_idx + 1}/{n_folds}:")

        # Split data
        df_train = df_num.iloc[train_idx].reset_index(drop=True)
        df_test = df_num.iloc[test_idx].reset_index(drop=True)

        n_train = len(df_train)
        n_test = len(df_test)
        n_train_ids = df_train[id_col].nunique()
        n_test_ids = df_test[id_col].nunique()

        if verbose:
            print(f"  Train: {n_train:,} obs from {n_train_ids} individuals")
            print(f"  Test:  {n_test:,} obs from {n_test_ids} individuals")

        # Create Biogeme database and estimate on training data
        database_train = db.Database('train', df_train)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            logprob, name = model_func(database_train)
            biogeme_obj = bio.BIOGEME(database_train, logprob)
            biogeme_obj.model_name = f'cv_fold_{fold_idx}'
            results = biogeme_obj.estimate()

        betas = results.get_beta_values()
        train_ll = results.final_loglikelihood

        # Compute test log-likelihood
        if use_simulated_ll:
            # Auto-detect random parameters if not provided
            fold_random_params = random_params
            if fold_random_params is None:
                fold_random_params = _auto_detect_random_params(betas)

            if fold_random_params:
                test_ll = compute_simulated_log_likelihood_mxl(
                    betas, df_test, utility_func,
                    random_params=fold_random_params,
                    n_draws=n_sim_draws,
                    id_col=id_col,
                    seed=seed + fold_idx  # Different seed per fold
                )
            else:
                # No random params detected - this is an ERROR for MXL/ICLV models
                # because point estimate LL would give misleading CV results
                raise ValueError(
                    f"Model type is {model_type.upper()} but no random parameters detected. "
                    f"Found parameters: {list(betas.keys())}. "
                    f"For MXL/ICLV CV to be valid, you must either:\n"
                    f"  1. Provide random_params dict specifying which params are random\n"
                    f"  2. Use naming convention B_*_MU / B_*_SIGMA for auto-detection\n"
                    f"If you want point-estimate CV (not recommended for mixed models), "
                    f"set model_type='mnl' explicitly."
                )
        else:
            test_ll = compute_log_likelihood(betas, df_test, utility_func, id_col, model_type)

        train_lls.append(train_ll)
        test_lls.append(test_ll)

        # Normalize by number of observations for comparison
        train_ll_per_obs = train_ll / n_train
        test_ll_per_obs = test_ll / n_test

        fold_results.append({
            'fold': fold_idx + 1,
            'train_ll': train_ll,
            'test_ll': test_ll,
            'train_ll_per_obs': train_ll_per_obs,
            'test_ll_per_obs': test_ll_per_obs,
            'n_train': n_train,
            'n_test': n_test,
            'betas': betas
        })

        if verbose:
            print(f"  Train LL: {train_ll:.2f} ({train_ll_per_obs:.4f}/obs)")
            print(f"  Test LL:  {test_ll:.2f} ({test_ll_per_obs:.4f}/obs)")

    # Summary statistics
    train_ll_mean = np.mean(train_lls)
    test_ll_mean = np.mean(test_lls)

    # Normalize by average observations
    avg_train_obs = np.mean([f['n_train'] for f in fold_results])
    avg_test_obs = np.mean([f['n_test'] for f in fold_results])

    train_ll_per_obs_mean = np.mean([f['train_ll_per_obs'] for f in fold_results])
    test_ll_per_obs_mean = np.mean([f['test_ll_per_obs'] for f in fold_results])

    # Generalization gap: difference between train and test LL per observation
    # Since LLs are negative: train - test > 0 means train is better (less negative)
    # A positive gap indicates overfitting (train fits better than test)
    generalization_gap = train_ll_per_obs_mean - test_ll_per_obs_mean

    # Overfit ratio: test/train (with absolute values for negative LLs)
    # Ratio > 1 means test LL is more negative (worse) than train → overfitting
    # Ratio ≈ 1 means similar performance → good generalization
    if abs(train_ll_per_obs_mean) > 1e-10:
        overfit_ratio = abs(test_ll_per_obs_mean) / abs(train_ll_per_obs_mean)
    else:
        overfit_ratio = np.nan

    if verbose:
        print(f"\n{'='*60}")
        print("CROSS-VALIDATION SUMMARY")
        print(f"{'='*60}")
        print(f"Mean Train LL/obs: {train_ll_per_obs_mean:.4f}")
        print(f"Mean Test LL/obs:  {test_ll_per_obs_mean:.4f}")
        print(f"Generalization gap: {generalization_gap:.4f}")
        print(f"Overfit ratio:     {overfit_ratio:.4f}")

        if overfit_ratio > 1.1:
            print("  WARNING: Potential overfitting detected (test LL worse than train)")
        elif overfit_ratio < 1.05:
            print("  GOOD: Similar train/test performance (good generalization)")

    return {
        'train_ll': train_lls,
        'test_ll': test_lls,
        'train_ll_mean': train_ll_mean,
        'test_ll_mean': test_ll_mean,
        'train_ll_per_obs_mean': train_ll_per_obs_mean,
        'test_ll_per_obs_mean': test_ll_per_obs_mean,
        'generalization_gap': generalization_gap,
        'overfit_ratio': overfit_ratio,
        'fold_results': fold_results,
        'n_folds': n_folds,
        'test_ll_method': test_ll_method,
        'model_type': model_type
    }


def _auto_detect_random_params(betas: Dict) -> Dict:
    """
    Auto-detect random parameters from estimated betas.

    Looks for patterns like B_FEE_MU/B_FEE_SIGMA or SIGMA_* patterns.

    Args:
        betas: Dictionary of estimated parameter values

    Returns:
        Dictionary of detected random parameter specifications
    """
    random_params = {}

    # Look for _MU/_SIGMA patterns
    for param_name in betas.keys():
        if '_MU' in param_name.upper():
            base_name = param_name.upper().replace('_MU', '')
            sigma_name = param_name.replace('_MU', '_SIGMA').replace('_mu', '_sigma')

            # Check if corresponding sigma exists
            for possible_sigma in betas.keys():
                if possible_sigma.upper() == sigma_name.upper():
                    random_params[base_name] = {
                        'mean': param_name,
                        'sigma': possible_sigma,
                        'dist': 'normal'
                    }
                    break

    # Look for SIGMA_* patterns (e.g., SIGMA_FEE)
    for param_name in betas.keys():
        if param_name.upper().startswith('SIGMA_'):
            base_name = param_name.upper().replace('SIGMA_', 'B_')
            if base_name not in random_params:
                # Check if mean parameter exists
                for possible_mean in betas.keys():
                    if possible_mean.upper() == base_name:
                        random_params[base_name] = {
                            'mean': possible_mean,
                            'sigma': param_name,
                            'dist': 'normal'
                        }
                        break

    return random_params


def compare_models_cv(df: pd.DataFrame,
                      models: List[Tuple[Callable, Callable, str]],
                      n_folds: int = 5,
                      id_col: str = 'ID',
                      seed: int = 42) -> pd.DataFrame:
    """
    Compare multiple models using cross-validation.

    Args:
        df: Full dataset
        models: List of (model_func, utility_func, name) tuples
        n_folds: Number of folds
        id_col: Individual identifier column
        seed: Random seed

    Returns:
        DataFrame with CV results for each model
    """
    results = []

    for model_func, utility_func, name in models:
        print(f"\n{'#'*60}")
        print(f"Model: {name}")
        print(f"{'#'*60}")

        cv_result = cross_validate_model(
            df, model_func, utility_func,
            n_folds=n_folds, id_col=id_col, seed=seed
        )

        results.append({
            'Model': name,
            'Train_LL_per_obs': cv_result['train_ll_per_obs_mean'],
            'Test_LL_per_obs': cv_result['test_ll_per_obs_mean'],
            'Overfit_Ratio': cv_result['overfit_ratio'],
            'N_Folds': n_folds
        })

    return pd.DataFrame(results).sort_values('Test_LL_per_obs', ascending=False)


def compute_simulated_log_likelihood_mxl(
    betas: Dict,
    df_test: pd.DataFrame,
    utility_func: Callable,
    random_params: Dict[str, Dict[str, str]],
    n_draws: int = 500,
    id_col: str = 'ID',
    seed: int = 42
) -> float:
    """
    Compute SIMULATED log-likelihood for MXL/ICLV models on test data.

    This properly integrates over the random coefficient distribution:
        LL = Σ_n log( (1/R) Σ_r Π_t P(y_nt | x_nt, β_n^r) )

    where β_n^r is the r-th draw from the random coefficient distribution.

    Args:
        betas: Dictionary of estimated parameter values including:
               - Mean parameters (e.g., 'B_FEE_MU')
               - Std dev parameters (e.g., 'B_FEE_SIGMA')
        df_test: Test data DataFrame
        utility_func: Function(betas_draw, df) -> {alt: utility_array}
        random_params: Dict specifying which params are random and their distribution
                      e.g., {'B_FEE': {'mean': 'B_FEE_MU', 'sigma': 'B_FEE_SIGMA', 'dist': 'normal'}}
        n_draws: Number of simulation draws per individual
        id_col: Column name for individual identifier
        seed: Random seed for reproducibility

    Returns:
        Simulated log-likelihood on test data

    Note:
        This is slower than point-estimate LL but provides UNBIASED test LL
        for mixed logit models. For ICLV models, you would also need to
        integrate over latent variable distributions.
    """
    np.random.seed(seed)

    unique_ids = df_test[id_col].unique()
    n_individuals = len(unique_ids)

    # Pre-generate all random draws for efficiency
    # Shape: (n_draws, n_random_params)
    random_param_names = list(random_params.keys())
    n_random = len(random_param_names)

    if n_random == 0:
        # No random parameters - use standard LL
        warnings.warn("No random parameters specified, using point estimate LL")
        return compute_log_likelihood(betas, df_test, utility_func, id_col, 'mnl')

    # Draw random coefficients
    draws = {}
    for param_name, param_spec in random_params.items():
        mean_param = param_spec.get('mean', f'{param_name}_MU')
        sigma_param = param_spec.get('sigma', f'{param_name}_SIGMA')
        dist = param_spec.get('dist', 'normal')

        mu = betas.get(mean_param, 0)
        sigma = betas.get(sigma_param, 0.1)

        if dist == 'normal':
            draws[param_name] = np.random.normal(mu, sigma, size=(n_draws,))
        elif dist == 'lognormal':
            # For lognormal: mean of underlying normal is mu, std is sigma
            draws[param_name] = np.random.lognormal(mu, sigma, size=(n_draws,))
        elif dist == 'triangular':
            # Symmetric triangular around mu with spread sigma
            draws[param_name] = np.random.triangular(mu - sigma, mu, mu + sigma, size=(n_draws,))
        else:
            raise ValueError(f"Unknown distribution: {dist}")

    # Compute simulated LL for each individual
    total_ll = 0.0

    for uid in unique_ids:
        df_ind = df_test[df_test[id_col] == uid]

        # For each draw, compute sequence probability
        prob_sequence_draws = np.zeros(n_draws)

        for r in range(n_draws):
            # Create betas with this draw's random coefficients
            betas_draw = betas.copy()
            for param_name in random_param_names:
                betas_draw[param_name] = draws[param_name][r]

            # Compute utilities
            V = utility_func(betas_draw, df_ind)

            # Compute choice probabilities
            exp_V = {k: np.exp(v - np.max(list(V.values()), axis=0)) for k, v in V.items()}
            sum_exp = sum(exp_V.values())
            probs = {k: v / sum_exp for k, v in exp_V.items()}

            # Get probability of chosen alternatives
            choice = df_ind['CHOICE'].values
            prob_chosen = np.ones(len(choice))
            for alt, prob in probs.items():
                mask = choice == alt
                prob_chosen[mask] = prob[mask]

            # Sequence probability (product over choice occasions)
            prob_chosen = np.clip(prob_chosen, 1e-10, 1.0)
            prob_sequence_draws[r] = np.prod(prob_chosen)

        # Average over draws and take log
        avg_prob = np.mean(prob_sequence_draws)
        if avg_prob > 0:
            total_ll += np.log(avg_prob)
        else:
            total_ll += np.log(1e-10)  # Floor to prevent -inf

    return total_ll


# Example utility functions for common models
def utility_mnl_basic(betas: Dict, df: pd.DataFrame) -> Dict[int, np.ndarray]:
    """Basic MNL utility function."""
    ASC = betas.get('ASC_paid', 0)
    B_FEE = betas.get('B_FEE', -0.5)
    B_DUR = betas.get('B_DUR', -0.05)

    V1 = ASC + B_FEE * df['fee1_10k'].values + B_DUR * df['dur1'].values
    V2 = ASC + B_FEE * df['fee2_10k'].values + B_DUR * df['dur2'].values
    V3 = B_FEE * df['fee3_10k'].values + B_DUR * df['dur3'].values

    return {1: V1, 2: V2, 3: V3}


def utility_mxl_basic(betas: Dict, df: pd.DataFrame) -> Dict[int, np.ndarray]:
    """
    Basic MXL utility function.

    Note: For simulation-based CV, this function should use the parameter
    name directly (e.g., 'B_FEE') which will be set to the draw value.
    """
    ASC = betas.get('ASC_paid', 0)
    B_FEE = betas.get('B_FEE', betas.get('B_FEE_MU', -0.5))  # Use draw or mean
    B_DUR = betas.get('B_DUR', -0.05)

    V1 = ASC + B_FEE * df['fee1_10k'].values + B_DUR * df['dur1'].values
    V2 = ASC + B_FEE * df['fee2_10k'].values + B_DUR * df['dur2'].values
    V3 = B_FEE * df['fee3_10k'].values + B_DUR * df['dur3'].values

    return {1: V1, 2: V2, 3: V3}


# Example random parameter specification for MXL
MXL_RANDOM_PARAMS = {
    'B_FEE': {
        'mean': 'B_FEE_MU',
        'sigma': 'B_FEE_SIGMA',
        'dist': 'normal'
    }
}
