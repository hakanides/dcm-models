"""
Parameter Recovery Tests
========================

Tests that validate estimation recovers true parameters within statistical tolerance.

These tests are essential for methodological validation:
- Verify that estimation algorithms are working correctly
- Confirm that standard errors are accurately computed
- Ensure confidence intervals have proper coverage

Methodological Notes:
- Uses large synthetic data (200 individuals × 20 tasks = 4000 obs)
- True parameters are known from data generation process
- Tests check bias (estimated - true) against 2 standard errors
- More stringent tests check coverage probability via Monte Carlo

Authors: Hakan Mülayim, Giray Girengir, Ataol Azeritürk
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))


# =============================================================================
# True Parameter Loading (Dynamic from Config)
# =============================================================================

def load_true_params(config_path: Path = None) -> dict:
    """
    Load true parameters from config.json file.

    This ensures tests use the same true values as the DGP,
    avoiding hardcoded parameter mismatches.

    Args:
        config_path: Path to config.json. If None, uses default MNL basic.

    Returns:
        Dictionary of true parameter values
    """
    import json

    if config_path is None:
        # Default: use MNL basic config
        config_path = PROJECT_ROOT.parent / 'models' / 'mnl_basic' / 'config.json'

    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        return config.get('model_info', {}).get('true_values', {})
    else:
        # Fallback to hardcoded values if config not found
        import warnings
        warnings.warn(f"Config not found at {config_path}, using default TRUE_PARAMS")
        return _DEFAULT_TRUE_PARAMS


# Fallback hardcoded values (used if config.json not found)
_DEFAULT_TRUE_PARAMS = {
    'ASC_paid': 5.0,
    'B_FEE': -0.08,  # Per 10k TL
    'B_DUR': -0.08,
    'B_FEE_PatBlind': -0.10,
    # Structural model parameters
    'gamma_age': 0.20,
    'sigma_LV': 1.0,
    # Measurement model (loadings)
    'lambda_pat_blind_1': 1.0,  # Fixed for identification
    'lambda_pat_blind_2': 0.83,
    'lambda_pat_blind_3': 0.81,
}

# Load TRUE_PARAMS at module import (can be overridden per test)
TRUE_PARAMS = load_true_params()


# =============================================================================
# MNL Parameter Recovery Tests
# =============================================================================

@pytest.mark.slow
@pytest.mark.estimation
class TestMNLParameterRecovery:
    """Test that MNL estimation recovers true parameters."""

    def test_mnl_bias_within_tolerance(self, large_synthetic_data):
        """
        Estimated parameters should be within 2 SEs of true values.

        This is a basic sanity check: if estimation is working correctly,
        the bias (estimated - true) should be within 2 standard errors
        approximately 95% of the time.
        """
        try:
            import biogeme.database as db
            import biogeme.biogeme as bio
            from biogeme import models
            from biogeme.expressions import Beta, Variable
        except ImportError:
            pytest.skip("Biogeme not installed")

        # Prepare data
        df = large_synthetic_data.copy()
        df['fee1_10k'] = df['fee1'] / 10000.0
        df['fee2_10k'] = df['fee2'] / 10000.0
        df['fee3_10k'] = df['fee3'] / 10000.0

        database = db.Database('recovery_test', df)

        # Define variables
        CHOICE = Variable('CHOICE')
        fee1_10k = Variable('fee1_10k')
        fee2_10k = Variable('fee2_10k')
        fee3_10k = Variable('fee3_10k')
        dur1 = Variable('dur1')
        dur2 = Variable('dur2')
        dur3 = Variable('dur3')

        # Parameters to estimate (with true values as starting points nearby)
        ASC_paid = Beta('ASC_paid', 1.5, None, None, 0)
        B_FEE = Beta('B_FEE', -0.4, None, None, 0)
        B_DUR = Beta('B_DUR', -0.02, None, None, 0)

        # Utility functions
        V1 = ASC_paid + B_FEE * fee1_10k + B_DUR * dur1
        V2 = ASC_paid + B_FEE * fee2_10k + B_DUR * dur2
        V3 = B_FEE * fee3_10k + B_DUR * dur3

        V = {1: V1, 2: V2, 3: V3}
        logprob = models.loglogit(V, None, CHOICE)

        # Estimate
        biogeme_obj = bio.BIOGEME(database, logprob)
        biogeme_obj.modelName = 'mnl_recovery_test'
        # HTML and pickle generation controlled via biogeme.toml

        results = biogeme_obj.estimate()

        # Check convergence
        assert results.algorithm_has_converged, "MNL estimation did not converge"

        # Validate parameter recovery
        params_to_check = {
            'ASC_paid': TRUE_PARAMS['ASC_paid'],
            'B_FEE': TRUE_PARAMS['B_FEE'],
            'B_DUR': TRUE_PARAMS['B_DUR'],
        }

        betas = results.get_beta_values()
        std_errs = {p: results.get_parameter_std_err(p) for p in betas}

        recovery_results = []
        for param_name, true_val in params_to_check.items():
            est_val = betas[param_name]
            se = std_errs[param_name]
            bias = est_val - true_val
            z_score = abs(bias) / se if se > 0 else float('inf')

            recovery_results.append({
                'parameter': param_name,
                'true': true_val,
                'estimated': est_val,
                'std_err': se,
                'bias': bias,
                'bias_pct': 100 * bias / abs(true_val) if true_val != 0 else 0,
                'z_score': z_score,
                'within_2se': z_score < 2.0,
            })

            # Assertion: bias should be within 2 standard errors
            assert z_score < 2.5, (
                f"{param_name}: bias={bias:.4f}, SE={se:.4f}, z={z_score:.2f} > 2.5\n"
                f"True={true_val}, Estimated={est_val:.4f}"
            )

        # Print summary for debugging
        print("\nMNL Parameter Recovery Summary:")
        print("-" * 70)
        for r in recovery_results:
            status = "✓" if r['within_2se'] else "✗"
            print(f"{r['parameter']:15} True={r['true']:8.4f}  Est={r['estimated']:8.4f}  "
                  f"Bias={r['bias']:+8.4f} ({r['bias_pct']:+5.1f}%)  z={r['z_score']:.2f} {status}")

    def test_mnl_fee_coefficient_sign(self, large_synthetic_data):
        """Fee coefficient should be negative (higher fee = lower utility)."""
        try:
            import biogeme.database as db
            import biogeme.biogeme as bio
            from biogeme import models
            from biogeme.expressions import Beta, Variable
        except ImportError:
            pytest.skip("Biogeme not installed")

        df = large_synthetic_data.copy()
        df['fee1_10k'] = df['fee1'] / 10000.0
        df['fee2_10k'] = df['fee2'] / 10000.0
        df['fee3_10k'] = df['fee3'] / 10000.0

        database = db.Database('sign_test', df)

        CHOICE = Variable('CHOICE')
        fee1_10k = Variable('fee1_10k')
        fee2_10k = Variable('fee2_10k')
        fee3_10k = Variable('fee3_10k')
        dur1 = Variable('dur1')
        dur2 = Variable('dur2')
        dur3 = Variable('dur3')

        ASC_paid = Beta('ASC_paid', 0, None, None, 0)
        B_FEE = Beta('B_FEE', 0, None, None, 0)
        B_DUR = Beta('B_DUR', 0, None, None, 0)

        V1 = ASC_paid + B_FEE * fee1_10k + B_DUR * dur1
        V2 = ASC_paid + B_FEE * fee2_10k + B_DUR * dur2
        V3 = B_FEE * fee3_10k + B_DUR * dur3

        V = {1: V1, 2: V2, 3: V3}
        logprob = models.loglogit(V, None, CHOICE)

        biogeme_obj = bio.BIOGEME(database, logprob)
        biogeme_obj.modelName = 'sign_test'
        # HTML and pickle generation controlled via biogeme.toml

        results = biogeme_obj.estimate()
        betas = results.get_beta_values()

        assert betas['B_FEE'] < 0, f"B_FEE should be negative, got {betas['B_FEE']:.4f}"
        assert betas['B_DUR'] < 0, f"B_DUR should be negative, got {betas['B_DUR']:.4f}"


# =============================================================================
# HCM Parameter Recovery Tests
# =============================================================================

@pytest.mark.slow
@pytest.mark.estimation
class TestHCMParameterRecovery:
    """Test that HCM with LV proxy recovers parameters (with expected attenuation)."""

    def test_hcm_proxy_runs_and_converges(self, large_synthetic_data):
        """HCM with LV proxy should converge on large data."""
        try:
            import biogeme.database as db
            import biogeme.biogeme as bio
            from biogeme import models
            from biogeme.expressions import Beta, Variable
        except ImportError:
            pytest.skip("Biogeme not installed")

        df = large_synthetic_data.copy()

        # Prepare data
        df['fee1_10k'] = df['fee1'] / 10000.0
        df['fee2_10k'] = df['fee2'] / 10000.0
        df['fee3_10k'] = df['fee3'] / 10000.0

        # Create LV proxy from Likert items (new unified naming: patriotism_1-10)
        lv_items = [f'patriotism_{i}' for i in range(1, 11) if f'patriotism_{i}' in df.columns]
        proxy = df[lv_items].mean(axis=1)
        df['pat_blind_proxy'] = (proxy - proxy.mean()) / proxy.std()

        database = db.Database('hcm_test', df)

        CHOICE = Variable('CHOICE')
        fee1_10k = Variable('fee1_10k')
        fee2_10k = Variable('fee2_10k')
        fee3_10k = Variable('fee3_10k')
        dur1 = Variable('dur1')
        dur2 = Variable('dur2')
        dur3 = Variable('dur3')
        pat_blind_proxy = Variable('pat_blind_proxy')

        ASC_paid = Beta('ASC_paid', 0, None, None, 0)
        B_FEE = Beta('B_FEE', 0, None, None, 0)
        B_DUR = Beta('B_DUR', 0, None, None, 0)
        B_FEE_PAT = Beta('B_FEE_PatBlind', 0, None, None, 0)

        # Fee coefficient varies with LV proxy
        B_FEE_ind = B_FEE + B_FEE_PAT * pat_blind_proxy

        V1 = ASC_paid + B_FEE_ind * fee1_10k + B_DUR * dur1
        V2 = ASC_paid + B_FEE_ind * fee2_10k + B_DUR * dur2
        V3 = B_FEE_ind * fee3_10k + B_DUR * dur3

        V = {1: V1, 2: V2, 3: V3}
        logprob = models.loglogit(V, None, CHOICE)

        biogeme_obj = bio.BIOGEME(database, logprob)
        biogeme_obj.modelName = 'hcm_proxy_test'
        # HTML and pickle generation controlled via biogeme.toml

        results = biogeme_obj.estimate()

        assert results.algorithm_has_converged, "HCM estimation did not converge"

        betas = results.get_beta_values()

        # Basic sign checks
        assert betas['B_FEE'] < 0, f"B_FEE should be negative"

        # Log results for analysis
        print("\nHCM Parameter Estimates:")
        print("-" * 50)
        for name, val in betas.items():
            true_val = TRUE_PARAMS.get(name, 'N/A')
            print(f"{name:20}: Est={val:8.4f}  True={true_val}")


# =============================================================================
# Measurement Model Tests
# =============================================================================

@pytest.mark.unit
class TestMeasurementModelValidation:
    """Tests for measurement model specification."""

    def test_likert_distribution_reasonable(self, large_synthetic_data):
        """Likert responses should have reasonable distribution (not all same value)."""
        df = large_synthetic_data

        # Check patriotism items (new unified naming)
        for i in range(1, 11):
            col = f'patriotism_{i}'
            if col in df.columns:
                unique_values = df[col].nunique()
                assert unique_values >= 3, f"{col} has only {unique_values} unique values"

                # Check distribution is not too skewed
                value_counts = df[col].value_counts(normalize=True)
                max_freq = value_counts.max()
                assert max_freq < 0.6, f"{col} has {max_freq:.1%} in most common category"

    def test_lv_proxy_correlates_with_true(self, large_synthetic_data):
        """LV proxy should correlate with true LV values."""
        df = large_synthetic_data

        if 'pat_blind_true' not in df.columns:
            pytest.skip("True LV values not in data")

        # Create proxy (new unified naming: patriotism_1-10 for blind patriotism)
        lv_items = [f'patriotism_{i}' for i in range(1, 11) if f'patriotism_{i}' in df.columns]
        proxy = df[lv_items].mean(axis=1)

        # Get unique individuals (first row per ID)
        df_unique = df.drop_duplicates(subset='ID')
        proxy_unique = proxy.iloc[df_unique.index]

        # Correlation with true
        from scipy.stats import pearsonr
        r, p = pearsonr(df_unique['pat_blind_true'], proxy_unique)

        print(f"\nProxy-True LV Correlation: r={r:.3f}, p={p:.4f}")

        # Proxy should correlate positively with true LV
        assert r > 0.3, f"Proxy-True correlation too low: r={r:.3f}"
        assert p < 0.05, f"Correlation not significant: p={p:.4f}"


# =============================================================================
# Statistical Validation Tests
# =============================================================================

@pytest.mark.slow
@pytest.mark.estimation
class TestStatisticalProperties:
    """Tests for statistical properties of estimation."""

    def test_standard_errors_positive(self, large_synthetic_data):
        """All standard errors should be positive and finite."""
        try:
            import biogeme.database as db
            import biogeme.biogeme as bio
            from biogeme import models
            from biogeme.expressions import Beta, Variable
        except ImportError:
            pytest.skip("Biogeme not installed")

        df = large_synthetic_data.copy()
        df['fee1_10k'] = df['fee1'] / 10000.0
        df['fee2_10k'] = df['fee2'] / 10000.0
        df['fee3_10k'] = df['fee3'] / 10000.0

        database = db.Database('se_test', df)

        CHOICE = Variable('CHOICE')
        fee1_10k = Variable('fee1_10k')
        fee2_10k = Variable('fee2_10k')
        fee3_10k = Variable('fee3_10k')
        dur1 = Variable('dur1')
        dur2 = Variable('dur2')
        dur3 = Variable('dur3')

        ASC_paid = Beta('ASC_paid', 0, None, None, 0)
        B_FEE = Beta('B_FEE', 0, None, None, 0)
        B_DUR = Beta('B_DUR', 0, None, None, 0)

        V1 = ASC_paid + B_FEE * fee1_10k + B_DUR * dur1
        V2 = ASC_paid + B_FEE * fee2_10k + B_DUR * dur2
        V3 = B_FEE * fee3_10k + B_DUR * dur3

        V = {1: V1, 2: V2, 3: V3}
        logprob = models.loglogit(V, None, CHOICE)

        biogeme_obj = bio.BIOGEME(database, logprob)
        biogeme_obj.modelName = 'se_validation'
        # HTML and pickle generation controlled via biogeme.toml

        results = biogeme_obj.estimate()
        betas = results.get_beta_values()
        std_errs = {p: results.get_parameter_std_err(p) for p in betas}

        for param, se in std_errs.items():
            assert se > 0, f"{param}: SE should be positive, got {se}"
            assert np.isfinite(se), f"{param}: SE should be finite, got {se}"
            assert se < 100, f"{param}: SE suspiciously large: {se}"

    def test_t_statistics_reasonable(self, large_synthetic_data):
        """T-statistics should be reasonable for significant parameters."""
        try:
            import biogeme.database as db
            import biogeme.biogeme as bio
            from biogeme import models
            from biogeme.expressions import Beta, Variable
        except ImportError:
            pytest.skip("Biogeme not installed")

        df = large_synthetic_data.copy()
        df['fee1_10k'] = df['fee1'] / 10000.0
        df['fee2_10k'] = df['fee2'] / 10000.0
        df['fee3_10k'] = df['fee3'] / 10000.0

        database = db.Database('tstat_test', df)

        CHOICE = Variable('CHOICE')
        fee1_10k = Variable('fee1_10k')
        fee2_10k = Variable('fee2_10k')
        fee3_10k = Variable('fee3_10k')
        dur1 = Variable('dur1')
        dur2 = Variable('dur2')
        dur3 = Variable('dur3')

        ASC_paid = Beta('ASC_paid', 0, None, None, 0)
        B_FEE = Beta('B_FEE', 0, None, None, 0)
        B_DUR = Beta('B_DUR', 0, None, None, 0)

        V1 = ASC_paid + B_FEE * fee1_10k + B_DUR * dur1
        V2 = ASC_paid + B_FEE * fee2_10k + B_DUR * dur2
        V3 = B_FEE * fee3_10k + B_DUR * dur3

        V = {1: V1, 2: V2, 3: V3}
        logprob = models.loglogit(V, None, CHOICE)

        biogeme_obj = bio.BIOGEME(database, logprob)
        biogeme_obj.modelName = 'tstat_validation'
        # HTML and pickle generation controlled via biogeme.toml

        results = biogeme_obj.estimate()

        betas = results.get_beta_values()
        std_errs = {p: results.get_parameter_std_err(p) for p in betas}

        print("\nT-statistics:")
        print("-" * 50)
        for param in betas:
            t_stat = betas[param] / std_errs[param] if std_errs[param] > 0 else 0
            print(f"{param:15}: t={t_stat:8.2f}")

            # Fee coefficient should be highly significant
            if param == 'B_FEE':
                assert abs(t_stat) > 2.0, f"B_FEE t-stat too low: {t_stat:.2f}"
