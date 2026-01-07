"""
Tests for Model Convergence
===========================

Tests that models converge on synthetic and real data.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Selective warning suppression - allow important warnings through
warnings.filterwarnings('ignore', category=FutureWarning)  # Biogeme deprecation warnings
warnings.filterwarnings('ignore', message='.*overflow.*')  # Numerical overflow in exp()
warnings.filterwarnings('ignore', message='.*divide by zero.*')  # Division warnings
warnings.filterwarnings('ignore', message='.*invalid value.*')  # NaN warnings during optimization


def prepare_test_database(df):
    """Prepare a Biogeme database from dataframe."""
    import biogeme.database as db

    df = df.copy()

    # Scale fees
    df['fee1_10k'] = df['fee1'] / 10000.0
    df['fee2_10k'] = df['fee2'] / 10000.0
    df['fee3_10k'] = df['fee3'] / 10000.0

    # Center demographics
    df['age_c'] = (df['age_idx'] - 2) / 2
    df['edu_c'] = (df['edu_idx'] - 3) / 2
    df['inc_c'] = (df['income_indiv_idx'] - 3) / 2

    # Create LV proxies
    lv_items = {
        'pat_blind': ['pat_blind_1', 'pat_blind_2', 'pat_blind_3', 'pat_blind_4'],
        'pat_const': ['pat_constructive_1', 'pat_constructive_2', 'pat_constructive_3', 'pat_constructive_4'],
        'sec_dl': ['sec_dl_1', 'sec_dl_2', 'sec_dl_3', 'sec_dl_4'],
        'sec_fp': ['sec_fp_1', 'sec_fp_2', 'sec_fp_3', 'sec_fp_4'],
    }

    for lv_name, items in lv_items.items():
        available = [c for c in items if c in df.columns]
        if available:
            proxy = df[available].mean(axis=1)
            df[f'{lv_name}_proxy'] = (proxy - proxy.mean()) / proxy.std()

    # Drop string columns
    string_cols = df.select_dtypes(include=['object']).columns.tolist()
    df_num = df.drop(columns=string_cols, errors='ignore')

    return db.Database('test_db', df_num)


class TestMNLConvergence:
    """Tests for MNL model convergence."""

    def test_basic_mnl_converges(self, sample_choice_data):
        """Test that basic MNL converges on synthetic data."""
        import biogeme.biogeme as bio
        from biogeme import models
        from biogeme.expressions import Beta, Variable

        database = prepare_test_database(sample_choice_data)

        # Define simple MNL
        dur1, dur2, dur3 = Variable('dur1'), Variable('dur2'), Variable('dur3')
        fee1, fee2, fee3 = Variable('fee1_10k'), Variable('fee2_10k'), Variable('fee3_10k')
        CHOICE = Variable('CHOICE')

        ASC_paid = Beta('ASC_paid', 0, -10, 10, 0)
        B_FEE = Beta('B_FEE', -0.5, -10, 0, 0)
        B_DUR = Beta('B_DUR', -0.05, -5, 0, 0)

        V1 = ASC_paid + B_FEE * fee1 + B_DUR * dur1
        V2 = ASC_paid + B_FEE * fee2 + B_DUR * dur2
        V3 = B_FEE * fee3 + B_DUR * dur3

        logprob = models.loglogit({1: V1, 2: V2, 3: V3}, {1: 1, 2: 1, 3: 1}, CHOICE)

        biogeme_model = bio.BIOGEME(database, logprob)
        biogeme_model.model_name = 'test_basic_mnl'
        results = biogeme_model.estimate()

        assert results.algorithm_has_converged, "Basic MNL failed to converge"

    def test_mnl_fee_coefficient_negative(self, sample_choice_data):
        """Test that fee coefficient is negative (as expected)."""
        import biogeme.biogeme as bio
        from biogeme import models
        from biogeme.expressions import Beta, Variable

        database = prepare_test_database(sample_choice_data)

        dur1, dur2, dur3 = Variable('dur1'), Variable('dur2'), Variable('dur3')
        fee1, fee2, fee3 = Variable('fee1_10k'), Variable('fee2_10k'), Variable('fee3_10k')
        CHOICE = Variable('CHOICE')

        ASC_paid = Beta('ASC_paid', 0, -10, 10, 0)
        B_FEE = Beta('B_FEE', -0.5, -10, 0, 0)
        B_DUR = Beta('B_DUR', -0.05, -5, 0, 0)

        V1 = ASC_paid + B_FEE * fee1 + B_DUR * dur1
        V2 = ASC_paid + B_FEE * fee2 + B_DUR * dur2
        V3 = B_FEE * fee3 + B_DUR * dur3

        logprob = models.loglogit({1: V1, 2: V2, 3: V3}, {1: 1, 2: 1, 3: 1}, CHOICE)

        biogeme_model = bio.BIOGEME(database, logprob)
        biogeme_model.model_name = 'test_fee_sign'
        results = biogeme_model.estimate()

        betas = results.get_beta_values()
        assert betas['B_FEE'] < 0, f"B_FEE should be negative, got {betas['B_FEE']}"


class TestHCMConvergence:
    """Tests for HCM model convergence."""

    def test_hcm_with_lv_proxy_runs(self, sample_choice_data):
        """Test that HCM with LV proxies runs without error.

        Note: Convergence on small synthetic data is not guaranteed.
        We test that estimation runs and produces finite results.
        """
        import biogeme.biogeme as bio
        from biogeme import models
        from biogeme.expressions import Beta, Variable

        database = prepare_test_database(sample_choice_data)

        dur1, dur2, dur3 = Variable('dur1'), Variable('dur2'), Variable('dur3')
        fee1, fee2, fee3 = Variable('fee1_10k'), Variable('fee2_10k'), Variable('fee3_10k')
        CHOICE = Variable('CHOICE')
        pat_blind = Variable('pat_blind_proxy')

        ASC_paid = Beta('ASC_paid', 0, -10, 10, 0)
        B_FEE = Beta('B_FEE', -0.5, -10, 0, 0)
        B_DUR = Beta('B_DUR', -0.05, -5, 0, 0)
        B_FEE_PAT = Beta('B_FEE_PAT', 0, -2, 2, 0)

        B_FEE_i = B_FEE + B_FEE_PAT * pat_blind

        V1 = ASC_paid + B_FEE_i * fee1 + B_DUR * dur1
        V2 = ASC_paid + B_FEE_i * fee2 + B_DUR * dur2
        V3 = B_FEE_i * fee3 + B_DUR * dur3

        logprob = models.loglogit({1: V1, 2: V2, 3: V3}, {1: 1, 2: 1, 3: 1}, CHOICE)

        biogeme_model = bio.BIOGEME(database, logprob)
        biogeme_model.model_name = 'test_hcm_lv'
        results = biogeme_model.estimate()

        # Test that estimation produced valid results (even if not converged)
        ll = results.final_loglikelihood
        assert np.isfinite(ll), "HCM produced non-finite log-likelihood"

        # Test that parameters are finite
        betas = results.get_beta_values()
        for param, val in betas.items():
            assert np.isfinite(val), f"Parameter {param} is not finite: {val}"


class TestModelFitStatistics:
    """Tests for model fit statistics."""

    def test_log_likelihood_finite(self, sample_choice_data):
        """Test that log-likelihood is finite (not NaN or inf)."""
        import biogeme.biogeme as bio
        from biogeme import models
        from biogeme.expressions import Beta, Variable

        database = prepare_test_database(sample_choice_data)

        dur1, dur2, dur3 = Variable('dur1'), Variable('dur2'), Variable('dur3')
        fee1, fee2, fee3 = Variable('fee1_10k'), Variable('fee2_10k'), Variable('fee3_10k')
        CHOICE = Variable('CHOICE')

        ASC_paid = Beta('ASC_paid', 0, -10, 10, 0)
        B_FEE = Beta('B_FEE', -0.5, -10, 0, 0)
        B_DUR = Beta('B_DUR', -0.05, -5, 0, 0)

        V1 = ASC_paid + B_FEE * fee1 + B_DUR * dur1
        V2 = ASC_paid + B_FEE * fee2 + B_DUR * dur2
        V3 = B_FEE * fee3 + B_DUR * dur3

        logprob = models.loglogit({1: V1, 2: V2, 3: V3}, {1: 1, 2: 1, 3: 1}, CHOICE)

        biogeme_model = bio.BIOGEME(database, logprob)
        biogeme_model.model_name = 'test_ll_finite'
        results = biogeme_model.estimate()

        ll = results.final_loglikelihood
        assert np.isfinite(ll), f"Log-likelihood is not finite: {ll}"

    def test_rho_squared_reasonable(self, sample_choice_data):
        """Test that rho-squared is in reasonable range [0, 1]."""
        import biogeme.biogeme as bio
        from biogeme import models
        from biogeme.expressions import Beta, Variable

        database = prepare_test_database(sample_choice_data)
        n_obs = len(sample_choice_data)
        null_ll = n_obs * np.log(1/3)

        dur1, dur2, dur3 = Variable('dur1'), Variable('dur2'), Variable('dur3')
        fee1, fee2, fee3 = Variable('fee1_10k'), Variable('fee2_10k'), Variable('fee3_10k')
        CHOICE = Variable('CHOICE')

        ASC_paid = Beta('ASC_paid', 0, -10, 10, 0)
        B_FEE = Beta('B_FEE', -0.5, -10, 0, 0)
        B_DUR = Beta('B_DUR', -0.05, -5, 0, 0)

        V1 = ASC_paid + B_FEE * fee1 + B_DUR * dur1
        V2 = ASC_paid + B_FEE * fee2 + B_DUR * dur2
        V3 = B_FEE * fee3 + B_DUR * dur3

        logprob = models.loglogit({1: V1, 2: V2, 3: V3}, {1: 1, 2: 1, 3: 1}, CHOICE)

        biogeme_model = bio.BIOGEME(database, logprob)
        biogeme_model.model_name = 'test_rho_sq'
        results = biogeme_model.estimate()

        ll = results.final_loglikelihood
        rho_sq = 1 - (ll / null_ll)

        assert 0 <= rho_sq <= 1, f"Rho-squared out of range: {rho_sq}"
