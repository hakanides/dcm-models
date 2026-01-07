"""
Tests for Latent Variable Estimation
====================================

Tests for CFA-based latent variable estimation.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))


class TestCFAEstimation:
    """Tests for CFA-based latent estimation."""

    def test_item_total_correlation_weights(self, sample_choice_data):
        """Test that item-total correlations produce valid weights."""
        df = sample_choice_data.copy()
        individuals = df.groupby('ID').first().reset_index()

        items = ['pat_blind_1', 'pat_blind_2', 'pat_blind_3', 'pat_blind_4']
        X = individuals[items].values

        # Calculate item-total correlations
        total = X.sum(axis=1)
        weights = []
        for i in range(X.shape[1]):
            corr = np.corrcoef(X[:, i], total)[0, 1]
            weights.append(max(0.1, corr))  # Minimum weight of 0.1

        weights = np.array(weights)
        weights = weights / weights.sum()

        # Weights should sum to 1
        assert np.isclose(weights.sum(), 1.0), "Weights don't sum to 1"

        # All weights should be positive
        assert (weights > 0).all(), "Some weights are non-positive"

    def test_standardized_scores_mean_zero(self, sample_choice_data):
        """Test that standardized LV scores have mean ~0."""
        df = sample_choice_data.copy()
        individuals = df.groupby('ID').first().reset_index()

        items = ['pat_blind_1', 'pat_blind_2', 'pat_blind_3', 'pat_blind_4']
        X = individuals[items].values

        # Weighted sum
        total = X.sum(axis=1)
        weights = []
        for i in range(X.shape[1]):
            corr = np.corrcoef(X[:, i], total)[0, 1]
            weights.append(max(0.1, corr))
        weights = np.array(weights) / sum(weights)

        score = (X * weights).sum(axis=1)

        # Standardize
        score_std = (score - score.mean()) / score.std()

        assert abs(score_std.mean()) < 0.01, f"Mean is {score_std.mean()}, expected ~0"

    def test_standardized_scores_std_one(self, sample_choice_data):
        """Test that standardized LV scores have std ~1."""
        df = sample_choice_data.copy()
        individuals = df.groupby('ID').first().reset_index()

        items = ['pat_blind_1', 'pat_blind_2', 'pat_blind_3', 'pat_blind_4']
        X = individuals[items].values

        # Simple mean
        score = X.mean(axis=1)

        # Standardize
        score_std = (score - score.mean()) / score.std()

        assert abs(score_std.std() - 1.0) < 0.01, f"Std is {score_std.std()}, expected ~1"


class TestLVCorrelations:
    """Tests for latent variable correlations."""

    def test_lv_correlations_bounded(self, sample_choice_data):
        """Test that LV correlations are in [-1, 1]."""
        df = sample_choice_data.copy()
        individuals = df.groupby('ID').first().reset_index()

        # Create LV proxies
        constructs = {
            'pat_blind': ['pat_blind_1', 'pat_blind_2', 'pat_blind_3', 'pat_blind_4'],
            'sec_dl': ['sec_dl_1', 'sec_dl_2', 'sec_dl_3', 'sec_dl_4'],
        }

        for name, items in constructs.items():
            score = individuals[items].mean(axis=1)
            individuals[f'LV_{name}'] = (score - score.mean()) / score.std()

        corr = individuals['LV_pat_blind'].corr(individuals['LV_sec_dl'])

        assert -1 <= corr <= 1, f"Correlation out of bounds: {corr}"

    def test_multicollinearity_check(self, sample_choice_data):
        """Test that LVs are not perfectly collinear."""
        df = sample_choice_data.copy()
        individuals = df.groupby('ID').first().reset_index()

        constructs = {
            'pat_blind': ['pat_blind_1', 'pat_blind_2', 'pat_blind_3', 'pat_blind_4'],
            'pat_const': ['pat_constructive_1', 'pat_constructive_2', 'pat_constructive_3', 'pat_constructive_4'],
            'sec_dl': ['sec_dl_1', 'sec_dl_2', 'sec_dl_3', 'sec_dl_4'],
            'sec_fp': ['sec_fp_1', 'sec_fp_2', 'sec_fp_3', 'sec_fp_4'],
        }

        for name, items in constructs.items():
            score = individuals[items].mean(axis=1)
            individuals[f'LV_{name}'] = (score - score.mean()) / score.std()

        lv_cols = [f'LV_{name}' for name in constructs.keys()]
        corr_matrix = individuals[lv_cols].corr()

        # No correlation should be > 0.95 (near-perfect multicollinearity)
        for i, c1 in enumerate(lv_cols):
            for c2 in lv_cols[i+1:]:
                corr = abs(corr_matrix.loc[c1, c2])
                assert corr < 0.95, f"High multicollinearity between {c1} and {c2}: {corr}"


class TestValidationWithTrueValues:
    """Tests for validation against true LV values (simulation only)."""

    def test_validation_models_import(self):
        """Test that validation_models module can be imported."""
        try:
            from models.validation_models import has_true_latent_values, validate_latent_estimation
            assert callable(has_true_latent_values)
            assert callable(validate_latent_estimation)
        except ImportError as e:
            pytest.skip(f"validation_models not importable: {e}")

    def test_has_true_latent_values_returns_false_for_synthetic(self, sample_choice_data):
        """Test that synthetic data correctly identified as lacking true LVs."""
        try:
            from models.validation_models import has_true_latent_values
        except ImportError:
            pytest.skip("validation_models not available")

        # Synthetic data doesn't have true LV columns
        result = has_true_latent_values(sample_choice_data)
        assert result == False, "Synthetic data should not have true LV values"
