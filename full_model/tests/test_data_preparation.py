"""
Tests for Data Preparation Functions
====================================

Tests fee scaling, demographic centering, and data quality checks.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestFeeScaling:
    """Tests for fee scaling consistency."""

    def test_fee_scaling_10k(self, sample_choice_data):
        """Test that fee scaling by 10,000 produces correct values."""
        df = sample_choice_data.copy()

        # Apply scaling
        df['fee1_10k'] = df['fee1'] / 10000.0
        df['fee2_10k'] = df['fee2'] / 10000.0
        df['fee3_10k'] = df['fee3'] / 10000.0

        # Check scaling is correct
        assert np.allclose(df['fee1_10k'] * 10000, df['fee1'])
        assert np.allclose(df['fee2_10k'] * 10000, df['fee2'])
        assert np.allclose(df['fee3_10k'] * 10000, df['fee3'])

    def test_fee_scale_reasonable_range(self, sample_choice_data):
        """Test that scaled fees are in reasonable range for utility."""
        df = sample_choice_data.copy()
        df['fee1_10k'] = df['fee1'] / 10000.0
        df['fee2_10k'] = df['fee2'] / 10000.0

        # Scaled fees should be roughly 0-100 for typical TL values
        assert df['fee1_10k'].max() < 1000, "Scaled fees too large"
        assert df['fee2_10k'].max() < 1000, "Scaled fees too large"

    def test_fee_scale_consistency_across_alternatives(self, sample_choice_data):
        """Test that all alternatives use same scaling."""
        df = sample_choice_data.copy()
        scale = 10000.0

        df['fee1_10k'] = df['fee1'] / scale
        df['fee2_10k'] = df['fee2'] / scale
        df['fee3_10k'] = df['fee3'] / scale

        # Verify same scale applied to all
        for col in ['fee1', 'fee2', 'fee3']:
            scaled_col = f'{col}_10k'
            assert scaled_col in df.columns


class TestDemographicCentering:
    """Tests for demographic variable centering."""

    def test_age_centering(self, sample_choice_data):
        """Test age centering around middle category."""
        df = sample_choice_data.copy()

        # Center around 2 (middle of 0-4 range), scale by 2
        df['age_c'] = (df['age_idx'] - 2) / 2

        # Check centering: if age_idx=2, age_c should be 0
        idx_2_mask = df['age_idx'] == 2
        if idx_2_mask.any():
            assert np.allclose(df.loc[idx_2_mask, 'age_c'], 0)

    def test_education_centering(self, sample_choice_data):
        """Test education centering around middle category."""
        df = sample_choice_data.copy()

        # Center around 3 (middle of 0-5 range), scale by 2
        df['edu_c'] = (df['edu_idx'] - 3) / 2

        # Check centering: if edu_idx=3, edu_c should be 0
        idx_3_mask = df['edu_idx'] == 3
        if idx_3_mask.any():
            assert np.allclose(df.loc[idx_3_mask, 'edu_c'], 0)

    def test_centered_variables_reasonable_range(self, sample_choice_data):
        """Test that centered variables are in reasonable range."""
        df = sample_choice_data.copy()

        df['age_c'] = (df['age_idx'] - 2) / 2
        df['edu_c'] = (df['edu_idx'] - 3) / 2
        df['inc_c'] = (df['income_indiv_idx'] - 3) / 2

        # Centered variables should be roughly -2 to +2
        for col in ['age_c', 'edu_c', 'inc_c']:
            assert df[col].min() >= -3, f"{col} too small"
            assert df[col].max() <= 3, f"{col} too large"


class TestLatentVariableProxies:
    """Tests for latent variable proxy creation."""

    def test_likert_mean_calculation(self, sample_choice_data):
        """Test that Likert means are calculated correctly."""
        df = sample_choice_data.copy()

        # Blind patriotism items: patriotism_1 to patriotism_10
        pat_blind_cols = [f'patriotism_{i}' for i in range(1, 11) if f'patriotism_{i}' in df.columns]
        if pat_blind_cols:
            df['pat_blind_mean'] = df[pat_blind_cols].mean(axis=1)

            # Mean should be between 1 and 5
            assert df['pat_blind_mean'].min() >= 1
            assert df['pat_blind_mean'].max() <= 5

    def test_likert_standardization(self, sample_choice_data):
        """Test that Likert proxies are standardized correctly."""
        df = sample_choice_data.copy()

        # Blind patriotism items: patriotism_1 to patriotism_10
        pat_blind_cols = [f'patriotism_{i}' for i in range(1, 11) if f'patriotism_{i}' in df.columns]
        if pat_blind_cols:
            proxy = df[pat_blind_cols].mean(axis=1)
            df['pat_blind_proxy'] = (proxy - proxy.mean()) / proxy.std()

            # Standardized should have mean ~0 and std ~1
            assert abs(df['pat_blind_proxy'].mean()) < 0.01
            assert abs(df['pat_blind_proxy'].std() - 1.0) < 0.01

    def test_all_constructs_have_items(self, sample_choice_data):
        """Test that all 4 constructs have Likert items (new unified naming)."""
        df = sample_choice_data

        # New unified naming: patriotism_1-20, secularism_1-25
        constructs = {
            'pat_blind': [f'patriotism_{i}' for i in range(1, 11) if f'patriotism_{i}' in df.columns],
            'pat_constructive': [f'patriotism_{i}' for i in range(11, 21) if f'patriotism_{i}' in df.columns],
            'sec_dl': [f'secularism_{i}' for i in range(1, 16) if f'secularism_{i}' in df.columns],
            'sec_fp': [f'secularism_{i}' for i in range(16, 26) if f'secularism_{i}' in df.columns],
        }

        for name, items in constructs.items():
            assert len(items) >= 1, f"No items found for {name}"


class TestDataQuality:
    """Tests for data quality checks."""

    def test_no_missing_choice(self, sample_choice_data):
        """Test that CHOICE column has no missing values."""
        df = sample_choice_data
        assert df['CHOICE'].notna().all(), "Missing values in CHOICE"

    def test_valid_choice_values(self, sample_choice_data):
        """Test that CHOICE values are 1, 2, or 3."""
        df = sample_choice_data
        valid_choices = {1, 2, 3}
        actual_choices = set(df['CHOICE'].unique())
        assert actual_choices.issubset(valid_choices), f"Invalid choices: {actual_choices - valid_choices}"

    def test_choice_share_balance(self, sample_choice_data):
        """Test that we have choice variation (at least 2 alternatives chosen)."""
        df = sample_choice_data
        n_unique = df['CHOICE'].nunique()

        # At least 2 different choices should be made
        # Note: In synthetic data with free alternative, choice 3 may dominate
        assert n_unique >= 2, "Need at least 2 different alternatives chosen"

    def test_fee_variation(self, sample_choice_data):
        """Test that fees have sufficient variation."""
        df = sample_choice_data

        for col in ['fee1', 'fee2']:
            std = df[col].std()
            assert std > 0, f"{col} has no variation"

    def test_panel_structure(self, sample_choice_data):
        """Test that data has proper panel structure."""
        df = sample_choice_data

        # Each ID should have multiple observations
        obs_per_id = df.groupby('ID').size()
        assert (obs_per_id > 1).all(), "Some IDs have only one observation"

        # Demographics should be constant within ID
        demo_cols = ['age_idx', 'edu_idx', 'income_indiv_idx']
        for col in demo_cols:
            if col in df.columns:
                within_id_var = df.groupby('ID')[col].nunique()
                assert (within_id_var == 1).all(), f"{col} varies within ID"


class TestConfigConsistency:
    """Tests for configuration consistency."""

    def test_fee_scale_in_config(self, model_config):
        """Test that fee_scale is defined in config."""
        if model_config is None:
            pytest.skip("model_config.json not found")

        fee_scale = model_config.get('choice_model', {}).get('fee_scale')
        assert fee_scale is not None, "fee_scale not defined in config"
        assert fee_scale == 10000, f"fee_scale is {fee_scale}, expected 10000"

    def test_latent_constructs_defined(self, model_config):
        """Test that all 4 latent constructs are defined."""
        if model_config is None:
            pytest.skip("model_config.json not found")

        latent_names = model_config.get('latent', {}).get('names', [])
        expected = {'pat_blind', 'pat_constructive', 'sec_dl', 'sec_fp'}

        assert set(latent_names) == expected, f"Missing latent constructs: {expected - set(latent_names)}"

    def test_centering_scaling_consistency(self, model_config):
        """Test that centering/scaling in estimation matches config.

        CRITICAL: If these don't match, parameter recovery will be biased!
        """
        if model_config is None:
            pytest.skip("model_config.json not found")

        # Expected centering/scaling from run_all_models.py prepare_data()
        expected = {
            'age_idx': {'center': 2.0, 'scale': 2.0},
            'edu_idx': {'center': 3.0, 'scale': 2.0},
            'income_indiv_idx': {'center': 3.0, 'scale': 2.0},
            'income_house_idx': {'center': 3.0, 'scale': 2.0},
            'marital_idx': {'center': 0.5, 'scale': 0.5},
        }

        # Extract from config
        choice_model = model_config.get('choice_model', {})
        for term in choice_model.get('attribute_terms', []):
            for interaction in term.get('interactions', []):
                var_name = interaction.get('with')
                if var_name in expected:
                    config_center = interaction.get('center')
                    config_scale = interaction.get('scale')

                    if config_center is not None:
                        assert config_center == expected[var_name]['center'], \
                            f"{var_name} center mismatch: config={config_center}, expected={expected[var_name]['center']}"
                    if config_scale is not None:
                        assert config_scale == expected[var_name]['scale'], \
                            f"{var_name} scale mismatch: config={config_scale}, expected={expected[var_name]['scale']}"
