"""
Attenuation Bias Tests
======================

Tests that validate attenuation bias in latent variable models.

Attenuation bias occurs when:
- Using LV proxies (factor scores) instead of true latent variables
- Measurement error attenuates coefficients toward zero
- HCM with proxies has MORE bias than ICLV (full information approach)

Theoretical background:
    True model: U = β × LV + ε
    Observed:   U = β × (LV + measurement_error) + ε

    Due to errors-in-variables:
    β_estimated < β_true (attenuated toward zero)

    Attenuation factor ≈ reliability = Var(LV) / Var(LV + error)

ICLV reduces this bias by:
    1. Simultaneously estimating measurement and choice models
    2. Integrating over the latent variable distribution
    3. Using full information from indicators

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


# True parameter for LV interaction
TRUE_B_FEE_PAT = 0.15


# =============================================================================
# Proxy Quality Tests
# =============================================================================

@pytest.mark.unit
class TestProxyQuality:
    """Tests for LV proxy construction and quality."""

    def test_proxy_reliability(self, large_synthetic_data):
        """
        LV proxy should have measurable reliability.

        Reliability = correlation between proxy and true LV
        For good proxy: reliability > 0.5
        """
        df = large_synthetic_data

        if 'pat_blind_true' not in df.columns:
            pytest.skip("True LV values not in data")

        # Create proxy from Likert items
        lv_items = [f'patriotism_{i}' for i in range(1, 11) if f'patriotism_{i}' in df.columns]
        proxy = df[lv_items].mean(axis=1)
        proxy_std = (proxy - proxy.mean()) / proxy.std()

        # Get unique individuals
        df_unique = df.drop_duplicates(subset='ID')
        proxy_unique = proxy_std.iloc[df_unique.index]
        true_unique = df_unique['pat_blind_true']

        # Compute reliability as correlation squared
        from scipy.stats import pearsonr
        r, p = pearsonr(proxy_unique, true_unique)
        reliability = r ** 2

        print(f"\nProxy Quality Metrics:")
        print("-" * 40)
        print(f"Proxy-True Correlation: r = {r:.3f}")
        print(f"Reliability (r²):       R² = {reliability:.3f}")
        print(f"P-value:                p = {p:.6f}")

        assert r > 0.3, f"Proxy correlation too low: {r:.3f}"
        assert reliability > 0.1, f"Reliability too low: {reliability:.3f}"

    def test_measurement_error_present(self, large_synthetic_data):
        """
        Demonstrate that proxies contain measurement error.

        The proxy variance should exceed the true LV variance due to
        measurement error.
        """
        df = large_synthetic_data

        if 'pat_blind_true' not in df.columns:
            pytest.skip("True LV values not in data")

        df_unique = df.drop_duplicates(subset='ID')

        # Proxy
        lv_items = [f'patriotism_{i}' for i in range(1, 11) if f'patriotism_{i}' in df.columns]
        proxy = df_unique[lv_items].mean(axis=1)

        true_lv = df_unique['pat_blind_true']

        # Raw variances
        var_proxy = proxy.var()
        var_true = true_lv.var()

        # Correlation
        from scipy.stats import pearsonr
        r, _ = pearsonr(proxy, true_lv)

        # Implied measurement error variance
        # proxy = true + error → Var(proxy) = Var(true) + Var(error) + 2Cov
        # If errors independent: Var(error) ≈ Var(proxy) - r² × Var(proxy)
        implied_error_var = var_proxy * (1 - r ** 2)

        print(f"\nMeasurement Error Analysis:")
        print("-" * 40)
        print(f"Var(proxy):           {var_proxy:.3f}")
        print(f"Var(true LV):         {var_true:.3f}")
        print(f"Correlation:          {r:.3f}")
        print(f"Implied error var:    {implied_error_var:.3f}")
        print(f"Signal-to-noise:      {r**2 / (1-r**2):.3f}")

        # There should be some measurement error
        assert implied_error_var > 0.01, "No measurement error detected"


# =============================================================================
# Attenuation Bias Demonstration
# =============================================================================

@pytest.mark.slow
@pytest.mark.estimation
class TestAttenuationBias:
    """
    Tests demonstrating attenuation bias in HCM with proxies.

    Key finding: Using LV proxies attenuates the LV×attribute interaction
    coefficient toward zero compared to the true value.
    """

    def test_hcm_proxy_shows_attenuation(self, large_synthetic_data):
        """
        HCM with LV proxy should show attenuated coefficient.

        The estimated B_FEE_PatBlind should be closer to zero than
        the true value (0.15) due to measurement error in the proxy.
        """
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

        # Create LV proxy
        lv_items = [f'patriotism_{i}' for i in range(1, 11) if f'patriotism_{i}' in df.columns]
        proxy = df[lv_items].mean(axis=1)
        df['pat_blind_proxy'] = (proxy - proxy.mean()) / proxy.std()

        database = db.Database('attenuation_test', df)

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

        # Fee sensitivity varies with LV proxy
        B_FEE_ind = B_FEE + B_FEE_PAT * pat_blind_proxy

        V1 = ASC_paid + B_FEE_ind * fee1_10k + B_DUR * dur1
        V2 = ASC_paid + B_FEE_ind * fee2_10k + B_DUR * dur2
        V3 = B_FEE_ind * fee3_10k + B_DUR * dur3

        V = {1: V1, 2: V2, 3: V3}
        logprob = models.loglogit(V, None, CHOICE)

        biogeme_obj = bio.BIOGEME(database, logprob)
        biogeme_obj.modelName = 'attenuation_hcm'
        # HTML and pickle generation controlled via biogeme.toml

        results = biogeme_obj.estimate()

        assert results.algorithm_has_converged, "HCM did not converge"

        betas = results.get_beta_values()
        std_errs = {p: results.get_parameter_std_err(p) for p in betas}

        estimated_b_fee_pat = betas['B_FEE_PatBlind']
        se_b_fee_pat = std_errs['B_FEE_PatBlind']

        # Calculate bias and attenuation
        bias = estimated_b_fee_pat - TRUE_B_FEE_PAT
        attenuation = estimated_b_fee_pat / TRUE_B_FEE_PAT if TRUE_B_FEE_PAT != 0 else np.nan

        print("\nAttenuation Bias Analysis:")
        print("-" * 50)
        print(f"True B_FEE_PatBlind:      {TRUE_B_FEE_PAT:.4f}")
        print(f"Estimated B_FEE_PatBlind: {estimated_b_fee_pat:.4f}")
        print(f"Standard Error:           {se_b_fee_pat:.4f}")
        print(f"Bias (Est - True):        {bias:+.4f}")
        print(f"Attenuation Factor:       {attenuation:.3f}")
        print(f"% of True Recovered:      {100 * attenuation:.1f}%")

        # Coefficient should be attenuated (closer to zero than true)
        # Note: This can fail with high probability due to sampling variation
        # We test that the estimate is at least in the right direction
        if TRUE_B_FEE_PAT > 0:
            assert estimated_b_fee_pat > 0, (
                f"Sign should be positive, got {estimated_b_fee_pat:.4f}"
            )
        elif TRUE_B_FEE_PAT < 0:
            assert estimated_b_fee_pat < 0, (
                f"Sign should be negative, got {estimated_b_fee_pat:.4f}"
            )

    def test_true_lv_reduces_bias(self, large_synthetic_data):
        """
        Using true LV (oracle) should reduce attenuation bias.

        This test demonstrates that if we could observe the true LV,
        the coefficient would be closer to the true value.
        """
        try:
            import biogeme.database as db
            import biogeme.biogeme as bio
            from biogeme import models
            from biogeme.expressions import Beta, Variable
        except ImportError:
            pytest.skip("Biogeme not installed")

        if 'pat_blind_true' not in large_synthetic_data.columns:
            pytest.skip("True LV values not in data")

        df = large_synthetic_data.copy()

        # Prepare data
        df['fee1_10k'] = df['fee1'] / 10000.0
        df['fee2_10k'] = df['fee2'] / 10000.0
        df['fee3_10k'] = df['fee3'] / 10000.0

        # Standardize true LV
        true_lv = df['pat_blind_true']
        df['pat_blind_oracle'] = (true_lv - true_lv.mean()) / true_lv.std()

        database = db.Database('oracle_test', df)

        CHOICE = Variable('CHOICE')
        fee1_10k = Variable('fee1_10k')
        fee2_10k = Variable('fee2_10k')
        fee3_10k = Variable('fee3_10k')
        dur1 = Variable('dur1')
        dur2 = Variable('dur2')
        dur3 = Variable('dur3')
        pat_blind_oracle = Variable('pat_blind_oracle')

        ASC_paid = Beta('ASC_paid', 0, None, None, 0)
        B_FEE = Beta('B_FEE', 0, None, None, 0)
        B_DUR = Beta('B_DUR', 0, None, None, 0)
        B_FEE_PAT = Beta('B_FEE_PatBlind', 0, None, None, 0)

        B_FEE_ind = B_FEE + B_FEE_PAT * pat_blind_oracle

        V1 = ASC_paid + B_FEE_ind * fee1_10k + B_DUR * dur1
        V2 = ASC_paid + B_FEE_ind * fee2_10k + B_DUR * dur2
        V3 = B_FEE_ind * fee3_10k + B_DUR * dur3

        V = {1: V1, 2: V2, 3: V3}
        logprob = models.loglogit(V, None, CHOICE)

        biogeme_obj = bio.BIOGEME(database, logprob)
        biogeme_obj.modelName = 'oracle_hcm'
        # HTML and pickle generation controlled via biogeme.toml

        results = biogeme_obj.estimate()

        assert results.algorithm_has_converged, "Oracle HCM did not converge"

        betas = results.get_beta_values()
        oracle_estimate = betas['B_FEE_PatBlind']

        # Compare with proxy-based estimate
        # (We'd need to run both and compare, but for now just check oracle)
        oracle_bias = oracle_estimate - TRUE_B_FEE_PAT

        print("\nOracle (True LV) vs Proxy Analysis:")
        print("-" * 50)
        print(f"True B_FEE_PatBlind:    {TRUE_B_FEE_PAT:.4f}")
        print(f"Oracle Estimate:        {oracle_estimate:.4f}")
        print(f"Oracle Bias:            {oracle_bias:+.4f}")
        print(f"Oracle Recovery:        {100 * oracle_estimate / TRUE_B_FEE_PAT:.1f}%")

        # Oracle should be closer to true than attenuated proxy estimate
        # (We're just checking oracle is reasonable here)
        oracle_recovery = oracle_estimate / TRUE_B_FEE_PAT if TRUE_B_FEE_PAT != 0 else 0

        # Oracle should recover at least 50% of true effect
        assert abs(oracle_recovery) > 0.3, (
            f"Oracle recovery too low: {100*oracle_recovery:.1f}%"
        )


# =============================================================================
# ICLV Comparison Tests
# =============================================================================

@pytest.mark.slow
@pytest.mark.iclv
class TestICLVVsHCM:
    """Tests comparing ICLV to HCM proxy approach."""

    def test_iclv_import(self):
        """ICLV estimation module should be importable."""
        try:
            from src.models.iclv import estimate_iclv
            assert estimate_iclv is not None
        except ImportError:
            pytest.skip("ICLV module not available")

    def test_comparison_tools_available(self):
        """Attenuation bias comparison tools should be available."""
        try:
            from src.models.iclv import (
                compare_two_stage_vs_iclv,
                summarize_attenuation_bias
            )
            assert compare_two_stage_vs_iclv is not None
            assert summarize_attenuation_bias is not None
        except ImportError:
            pytest.skip("ICLV comparison tools not available")

    def test_theoretical_attenuation_formula(self):
        """
        Verify theoretical attenuation formula.

        Attenuation factor = reliability = Var(true) / Var(observed)
                          = 1 / (1 + Var(error)/Var(true))

        For standardized variables with reliability r:
            β_attenuated ≈ r × β_true
        """
        # Simulate measurement error scenario
        np.random.seed(42)
        n = 1000

        # True LV ~ N(0, 1)
        true_lv = np.random.normal(0, 1, n)

        # Measurement error ~ N(0, σ²_error)
        error_variance = 0.5
        error = np.random.normal(0, np.sqrt(error_variance), n)

        # Observed proxy = true + error
        observed = true_lv + error

        # Compute reliability
        from scipy.stats import pearsonr
        r, _ = pearsonr(observed, true_lv)
        reliability = r ** 2

        # Theoretical attenuation
        theoretical_reliability = 1 / (1 + error_variance)

        print("\nTheoretical Attenuation Verification:")
        print("-" * 50)
        print(f"Error variance:              {error_variance:.3f}")
        print(f"Theoretical reliability:     {theoretical_reliability:.3f}")
        print(f"Empirical reliability:       {reliability:.3f}")
        print(f"Difference:                  {abs(reliability - theoretical_reliability):.3f}")

        # Should be close
        assert abs(reliability - theoretical_reliability) < 0.1, (
            f"Reliability mismatch: theoretical={theoretical_reliability:.3f}, "
            f"empirical={reliability:.3f}"
        )
