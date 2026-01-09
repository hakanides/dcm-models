"""
Cross-Validation for DCM Models
================================

Implements k-fold cross-validation for out-of-sample prediction assessment.

IMPORTANT: For panel data, folds are split by INDIVIDUAL (ID), not by observation.
This ensures all observations from the same person are in the same fold.

Usage:
    from src.estimation.cross_validation import cross_validate_model

    results = cross_validate_model(
        df=data,
        model_func=model_mnl_basic,
        n_folds=5,
        id_col='ID'
    )

Author: DCM Research Team
"""

import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Tuple, Optional
import warnings

try:
    from sklearn.model_selection import KFold
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models


def split_by_individual(df: pd.DataFrame, n_folds: int = 5,
                        id_col: str = 'ID', seed: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create k-fold splits by individual ID (not by observation).

    This ensures all observations from the same person are in the same fold,
    which is critical for panel data to avoid data leakage.

    Args:
        df: DataFrame with panel data
        n_folds: Number of folds
        id_col: Column name for individual identifier
        seed: Random seed for reproducibility

    Returns:
        List of (train_indices, test_indices) tuples for each fold
    """
    if not HAS_SKLEARN:
        raise ImportError("sklearn is required for cross-validation. Install with: pip install scikit-learn")

    unique_ids = df[id_col].unique()
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    folds = []
    for train_id_idx, test_id_idx in kf.split(unique_ids):
        train_ids = unique_ids[train_id_idx]
        test_ids = unique_ids[test_id_idx]

        train_mask = df[id_col].isin(train_ids)
        test_mask = df[id_col].isin(test_ids)

        train_indices = df.index[train_mask].values
        test_indices = df.index[test_mask].values

        folds.append((train_indices, test_indices))

    return folds


def compute_log_likelihood(betas: Dict, df_test: pd.DataFrame,
                           utility_func: Callable) -> float:
    """
    Compute log-likelihood on test data using estimated parameters.

    Args:
        betas: Dictionary of estimated parameter values
        df_test: Test data DataFrame
        utility_func: Function that computes utilities given betas and data

    Returns:
        Log-likelihood on test data
    """
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
                         verbose: bool = True) -> Dict:
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

    Returns:
        Dict with:
        - 'train_ll': List of training log-likelihoods
        - 'test_ll': List of test log-likelihoods
        - 'train_ll_mean': Mean training LL
        - 'test_ll_mean': Mean test LL
        - 'overfit_ratio': train_ll / test_ll (>1 indicates overfitting)
        - 'fold_results': Detailed results per fold
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"CROSS-VALIDATION ({n_folds}-fold, split by {id_col})")
        print(f"{'='*60}")

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
        test_ll = compute_log_likelihood(betas, df_test, utility_func)

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

    # Overfit ratio (per observation to make comparable)
    overfit_ratio = train_ll_per_obs_mean / test_ll_per_obs_mean if test_ll_per_obs_mean != 0 else np.nan

    if verbose:
        print(f"\n{'='*60}")
        print("CROSS-VALIDATION SUMMARY")
        print(f"{'='*60}")
        print(f"Mean Train LL/obs: {train_ll_per_obs_mean:.4f}")
        print(f"Mean Test LL/obs:  {test_ll_per_obs_mean:.4f}")
        print(f"Overfit ratio:     {overfit_ratio:.4f}")

        if overfit_ratio > 1.1:
            print("  WARNING: Potential overfitting detected")
        elif overfit_ratio < 0.95:
            print("  NOTE: Test performance similar to train (good generalization)")

    return {
        'train_ll': train_lls,
        'test_ll': test_lls,
        'train_ll_mean': train_ll_mean,
        'test_ll_mean': test_ll_mean,
        'train_ll_per_obs_mean': train_ll_per_obs_mean,
        'test_ll_per_obs_mean': test_ll_per_obs_mean,
        'overfit_ratio': overfit_ratio,
        'fold_results': fold_results,
        'n_folds': n_folds
    }


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
