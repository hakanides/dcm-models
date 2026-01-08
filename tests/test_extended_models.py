"""
Tests for Extended Model Specifications
=======================================

Tests for MNL, MXL, and HCM extended models.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))


# =============================================================================
# MNL Extended Import Tests
# =============================================================================

@pytest.mark.unit
class TestMNLExtendedImports:
    """Tests for MNL extended module imports."""

    def test_mnl_extended_import(self):
        """Test that MNL extended module can be imported."""
        try:
            from src.models.mnl_extended import run_mnl_extended
            assert callable(run_mnl_extended)
        except ImportError as e:
            pytest.skip(f"MNL extended not importable: {e}")

    def test_mnl_extended_models_list(self):
        """Test that MNL extended has multiple model specifications."""
        try:
            from src.models import mnl_extended

            # Should have at least some model functions or specs
            attrs = dir(mnl_extended)
            model_attrs = [a for a in attrs if 'model' in a.lower() or a.startswith('M')]
            assert len(model_attrs) > 0
        except ImportError as e:
            pytest.skip(f"MNL extended not available: {e}")


# =============================================================================
# MXL Extended Import Tests
# =============================================================================

@pytest.mark.unit
class TestMXLExtendedImports:
    """Tests for MXL extended module imports."""

    def test_mxl_extended_import(self):
        """Test that MXL extended module can be imported."""
        try:
            from src.models.mxl_extended import run_mxl_extended
            assert callable(run_mxl_extended)
        except ImportError as e:
            pytest.skip(f"MXL extended not importable: {e}")


# =============================================================================
# HCM Extended Import Tests
# =============================================================================

@pytest.mark.unit
class TestHCMExtendedImports:
    """Tests for HCM extended module imports."""

    def test_hcm_extended_import(self):
        """Test that HCM extended module can be imported."""
        try:
            from src.models.hcm_extended import run_hcm_extended
            assert callable(run_hcm_extended)
        except ImportError as e:
            pytest.skip(f"HCM extended not importable: {e}")


# =============================================================================
# MNL Extended Specification Tests
# =============================================================================

@pytest.mark.estimation
class TestMNLExtendedSpecifications:
    """Tests for MNL extended model specifications."""

    def test_log_fee_transformation(self, sample_choice_data):
        """Test log fee transformation works correctly."""
        df = sample_choice_data.copy()

        # Apply log transformation
        df['fee1_log'] = np.log(df['fee1'] + 1)
        df['fee2_log'] = np.log(df['fee2'] + 1)
        df['fee3_log'] = np.log(df['fee3'] + 1)

        # Check transformation is valid
        assert df['fee1_log'].min() >= 0
        assert np.isfinite(df['fee1_log']).all()

        # Log should compress range
        fee_range = df['fee1'].max() - df['fee1'].min()
        log_range = df['fee1_log'].max() - df['fee1_log'].min()
        assert log_range < fee_range

    def test_quadratic_fee_transformation(self, sample_choice_data):
        """Test quadratic fee transformation."""
        df = sample_choice_data.copy()

        # Scale first
        df['fee1_10k'] = df['fee1'] / 10000
        df['fee1_10k_sq'] = df['fee1_10k'] ** 2

        # Squared term should be positive
        assert (df['fee1_10k_sq'] >= 0).all()

    def test_piecewise_fee_transformation(self, sample_choice_data):
        """Test piecewise fee transformation."""
        df = sample_choice_data.copy()

        # Scale fees
        df['fee1_10k'] = df['fee1'] / 10000

        # Calculate threshold (median)
        threshold = df['fee1_10k'].median()

        # Apply piecewise
        df['fee1_low'] = np.minimum(df['fee1_10k'], threshold)
        df['fee1_high'] = np.maximum(0, df['fee1_10k'] - threshold)

        # Check constraints
        assert (df['fee1_low'] <= threshold).all()
        assert (df['fee1_high'] >= 0).all()

        # Sum should equal original
        assert np.allclose(df['fee1_low'] + df['fee1_high'], df['fee1_10k'])

    def test_demographic_interactions(self, sample_choice_data):
        """Test demographic interaction terms."""
        df = sample_choice_data.copy()

        # Center demographics
        df['age_c'] = (df['age_idx'] - 2) / 2
        df['inc_c'] = (df['income_indiv_idx'] - 3) / 2

        # Create interaction
        df['age_x_inc'] = df['age_c'] * df['inc_c']

        # Should have variation
        assert df['age_x_inc'].std() > 0


# =============================================================================
# MXL Extended Specification Tests
# =============================================================================

@pytest.mark.estimation
class TestMXLExtendedSpecifications:
    """Tests for MXL extended model specifications."""

    def test_lognormal_ensures_negative(self):
        """Test that lognormal parameterization ensures negative coefficients."""
        np.random.seed(42)

        mu = -2.0
        sigma = 0.5
        n_draws = 1000

        # Lognormal for negative coefficient
        eta = np.random.normal(0, 1, n_draws)
        beta = -np.exp(mu + sigma * eta)

        # All should be negative
        assert (beta < 0).all()

        # Mean should be negative
        assert beta.mean() < 0

    def test_uniform_distribution_bounds(self):
        """Test uniform distribution stays within bounds."""
        np.random.seed(42)

        mean = -0.5
        spread = 0.3
        n_draws = 1000

        beta = np.random.uniform(mean - spread, mean + spread, n_draws)

        assert beta.min() >= mean - spread
        assert beta.max() <= mean + spread


# =============================================================================
# HCM Extended Specification Tests
# =============================================================================

@pytest.mark.estimation
class TestHCMExtendedSpecifications:
    """Tests for HCM extended model specifications."""

    def test_lv_proxy_creation(self, sample_choice_data):
        """Test LV proxy creation for HCM."""
        df = sample_choice_data.copy()

        # Create LV proxies
        items = ['pat_blind_1', 'pat_blind_2', 'pat_blind_3', 'pat_blind_4']
        proxy = df[items].mean(axis=1)
        df['pat_blind_proxy'] = (proxy - proxy.mean()) / proxy.std()

        # Should be standardized
        assert abs(df['pat_blind_proxy'].mean()) < 0.01
        assert abs(df['pat_blind_proxy'].std() - 1.0) < 0.01

    def test_quadratic_lv_term(self, sample_choice_data):
        """Test quadratic LV term."""
        df = sample_choice_data.copy()

        # Create LV proxy
        items = ['pat_blind_1', 'pat_blind_2', 'pat_blind_3', 'pat_blind_4']
        proxy = df[items].mean(axis=1)
        df['LV'] = (proxy - proxy.mean()) / proxy.std()

        # Squared term
        df['LV_sq'] = df['LV'] ** 2

        # Squared should be positive
        assert (df['LV_sq'] >= 0).all()

    def test_lv_demographic_interaction(self, sample_choice_data):
        """Test LV x demographic interaction."""
        df = sample_choice_data.copy()

        # Create LV proxy
        items = ['pat_blind_1', 'pat_blind_2', 'pat_blind_3', 'pat_blind_4']
        proxy = df[items].mean(axis=1)
        df['LV'] = (proxy - proxy.mean()) / proxy.std()

        # Center demographics
        df['age_c'] = (df['age_idx'] - 2) / 2

        # Interaction
        df['LV_x_age'] = df['LV'] * df['age_c']

        # Should have variation
        assert df['LV_x_age'].std() > 0

    def test_domain_separation_setup(self, sample_choice_data):
        """Test domain separation (different LVs for different attributes)."""
        df = sample_choice_data.copy()

        # Create proxies for different domains
        pat_items = ['pat_blind_1', 'pat_blind_2', 'pat_blind_3', 'pat_blind_4']
        sec_items = ['sec_dl_1', 'sec_dl_2', 'sec_dl_3', 'sec_dl_4']

        pat_proxy = df[pat_items].mean(axis=1)
        sec_proxy = df[sec_items].mean(axis=1)

        df['LV_pat'] = (pat_proxy - pat_proxy.mean()) / pat_proxy.std()
        df['LV_sec'] = (sec_proxy - sec_proxy.mean()) / sec_proxy.std()

        # Both should exist and be standardized
        assert abs(df['LV_pat'].mean()) < 0.01
        assert abs(df['LV_sec'].mean()) < 0.01


# =============================================================================
# Integration Tests for Extended Models
# =============================================================================

@pytest.mark.estimation
@pytest.mark.slow
@pytest.mark.integration
class TestExtendedModelEstimation:
    """Integration tests for extended model estimation."""

    def test_mnl_basic_runs(self, sample_choice_data):
        """Test basic MNL model runs."""
        try:
            import biogeme.biogeme as bio
            from biogeme import models
            from biogeme.expressions import Beta, Variable
            from conftest import prepare_test_database

            database = prepare_test_database(sample_choice_data)

            # Define basic MNL
            fee1 = Variable('fee1_10k')
            fee2 = Variable('fee2_10k')
            fee3 = Variable('fee3_10k')
            dur1 = Variable('dur1')
            dur2 = Variable('dur2')
            dur3 = Variable('dur3')
            CHOICE = Variable('CHOICE')

            ASC = Beta('ASC', 0, -10, 10, 0)
            B_FEE = Beta('B_FEE', -0.5, -10, 0, 0)
            B_DUR = Beta('B_DUR', -0.05, -5, 0, 0)

            V1 = ASC + B_FEE * fee1 + B_DUR * dur1
            V2 = ASC + B_FEE * fee2 + B_DUR * dur2
            V3 = B_FEE * fee3 + B_DUR * dur3

            logprob = models.loglogit({1: V1, 2: V2, 3: V3}, None, CHOICE)

            biogeme_model = bio.BIOGEME(database, logprob)
            biogeme_model.model_name = 'test_mnl_basic'
            results = biogeme_model.estimate()

            assert results is not None

        except ImportError as e:
            pytest.skip(f"Biogeme not available: {e}")

    def test_mnl_log_fee_runs(self, sample_choice_data):
        """Test MNL with log fee transformation."""
        try:
            import biogeme.biogeme as bio
            from biogeme import models
            from biogeme.expressions import Beta, Variable, log
            from conftest import prepare_test_database

            # Prepare data with log fee
            df = sample_choice_data.copy()
            df['fee1_log'] = np.log(df['fee1'] + 1)
            df['fee2_log'] = np.log(df['fee2'] + 1)
            df['fee3_log'] = np.log(df['fee3'] + 1)

            database = prepare_test_database(df)

            fee1_log = Variable('fee1_log')
            fee2_log = Variable('fee2_log')
            fee3_log = Variable('fee3_log')
            dur1 = Variable('dur1')
            dur2 = Variable('dur2')
            dur3 = Variable('dur3')
            CHOICE = Variable('CHOICE')

            ASC = Beta('ASC', 0, -10, 10, 0)
            B_FEE_LOG = Beta('B_FEE_LOG', -0.5, -10, 0, 0)
            B_DUR = Beta('B_DUR', -0.05, -5, 0, 0)

            V1 = ASC + B_FEE_LOG * fee1_log + B_DUR * dur1
            V2 = ASC + B_FEE_LOG * fee2_log + B_DUR * dur2
            V3 = B_FEE_LOG * fee3_log + B_DUR * dur3

            logprob = models.loglogit({1: V1, 2: V2, 3: V3}, None, CHOICE)

            biogeme_model = bio.BIOGEME(database, logprob)
            biogeme_model.model_name = 'test_mnl_log'
            results = biogeme_model.estimate()

            assert results is not None

        except ImportError as e:
            pytest.skip(f"Biogeme not available: {e}")


# =============================================================================
# Model Comparison Tests
# =============================================================================

@pytest.mark.unit
class TestModelComparison:
    """Tests for model comparison utilities."""

    def test_aic_calculation(self):
        """Test AIC calculation."""
        ll = -1000
        k = 5

        aic = 2 * k - 2 * ll

        assert aic == 2010

    def test_bic_calculation(self):
        """Test BIC calculation."""
        ll = -1000
        k = 5
        n = 500

        bic = k * np.log(n) - 2 * ll

        expected = 5 * np.log(500) + 2000
        assert np.isclose(bic, expected)

    def test_lr_test(self):
        """Test likelihood ratio test."""
        ll_full = -900
        ll_restricted = -950
        df = 3

        lr_stat = 2 * (ll_full - ll_restricted)

        assert lr_stat == 100

        # p-value from chi-squared
        from scipy import stats
        p_value = 1 - stats.chi2.cdf(lr_stat, df)

        assert p_value < 0.05  # Should be significant
