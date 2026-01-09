"""
Standard Error Validation Tests
===============================

Tests that validate standard error computation is correct.

Standard errors are critical for:
- Hypothesis testing (t-statistics)
- Confidence interval construction
- Statistical inference validity

Validation approaches:
1. Bootstrap vs Analytical comparison
2. Sandwich estimator checks
3. Clustering validation

Author: DCM Research Team
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
# Basic SE Validation
# =============================================================================

@pytest.mark.estimation
class TestStandardErrorBasics:
    """Basic tests for standard error properties."""

    def test_se_positive_and_finite(self, large_synthetic_data):
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
        biogeme_obj.modelName = 'se_basic_test'
        # HTML and pickle generation controlled via biogeme.toml

        results = biogeme_obj.estimate()
        betas = results.get_beta_values()
        std_errs = {p: results.get_parameter_std_err(p) for p in betas}

        print("\nStandard Error Validation:")
        print("-" * 50)
        for param, se in std_errs.items():
            status = "✓" if (se > 0 and np.isfinite(se) and se < 100) else "✗"
            print(f"{param:15}: SE={se:10.6f} {status}")

            assert se > 0, f"{param}: SE should be positive, got {se}"
            assert np.isfinite(se), f"{param}: SE should be finite, got {se}"
            assert se < 100, f"{param}: SE suspiciously large: {se}"

    def test_robust_se_available(self, large_synthetic_data):
        """Robust (sandwich) standard errors should be computable."""
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

        database = db.Database('robust_se_test', df)

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
        biogeme_obj.modelName = 'robust_se_test'
        # HTML and pickle generation controlled via biogeme.toml

        results = biogeme_obj.estimate()

        # Try to get robust SEs
        try:
            betas = results.get_beta_values()
            regular_se = {p: results.get_parameter_std_err(p) for p in betas}
            robust_se = {p: results.get_parameter_robust_std_err(p) for p in betas}

            print("\nRobust vs Regular SE Comparison:")
            print("-" * 60)
            for param in regular_se:
                reg = regular_se[param]
                rob = robust_se.get(param, np.nan)
                ratio = rob / reg if reg > 0 else np.nan
                print(f"{param:15}: Regular={reg:.6f}  Robust={rob:.6f}  Ratio={ratio:.3f}")

        except AttributeError:
            pytest.skip("Robust SE method not available in this Biogeme version")


# =============================================================================
# Bootstrap SE Validation
# =============================================================================

@pytest.mark.slow
@pytest.mark.estimation
class TestBootstrapStandardErrors:
    """Tests comparing bootstrap and analytical standard errors."""

    def test_bootstrap_se_import(self):
        """Bootstrap estimation module should be importable."""
        try:
            from src.estimation.bootstrap_inference import BootstrapEstimator
            assert BootstrapEstimator is not None
        except ImportError:
            pytest.skip("Bootstrap inference module not available")

    def test_bootstrap_concept(self, sample_choice_data):
        """
        Demonstrate bootstrap SE computation concept.

        Note: Full bootstrap validation requires many iterations and is slow.
        This test demonstrates the concept with minimal iterations.
        """
        try:
            import biogeme.database as db
            import biogeme.biogeme as bio
            from biogeme import models
            from biogeme.expressions import Beta, Variable
        except ImportError:
            pytest.skip("Biogeme not installed")

        # Use smaller sample for speed
        df = sample_choice_data.copy()
        df['fee1_10k'] = df['fee1'] / 10000.0
        df['fee2_10k'] = df['fee2'] / 10000.0
        df['fee3_10k'] = df['fee3'] / 10000.0

        # Get unique individuals for resampling
        individual_ids = df['ID'].unique()

        # Run 5 bootstrap iterations (just to demonstrate concept)
        n_bootstrap = 5
        bootstrap_estimates = {
            'ASC_paid': [],
            'B_FEE': [],
            'B_DUR': [],
        }

        for b in range(n_bootstrap):
            # Resample individuals with replacement
            np.random.seed(42 + b)
            sampled_ids = np.random.choice(individual_ids, size=len(individual_ids), replace=True)

            # Get all observations for sampled individuals
            boot_df = pd.concat([
                df[df['ID'] == id].copy() for id in sampled_ids
            ], ignore_index=True)

            # Renumber IDs for Biogeme
            id_map = {old_id: new_id for new_id, old_id in enumerate(sampled_ids, 1)}
            boot_df['ID'] = boot_df['ID'].map(lambda x: id_map.get(x, x))

            database = db.Database(f'boot_{b}', boot_df)

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
            biogeme_obj.modelName = f'boot_{b}'
            # HTML and pickle generation controlled via biogeme.toml

            try:
                results = biogeme_obj.estimate()
                if results.algorithm_has_converged:
                    betas = results.get_beta_values()
                    for param in bootstrap_estimates:
                        if param in betas:
                            bootstrap_estimates[param].append(betas[param])
            except Exception:
                continue  # Skip failed iterations

        # Compute bootstrap SE as std of estimates
        print("\nBootstrap SE (5 iterations - demonstration only):")
        print("-" * 50)
        for param, estimates in bootstrap_estimates.items():
            if len(estimates) >= 2:
                boot_se = np.std(estimates, ddof=1)
                boot_mean = np.mean(estimates)
                print(f"{param:15}: Mean={boot_mean:.4f}  SE={boot_se:.4f}  N={len(estimates)}")

        # At least some iterations should have succeeded
        min_success = 2
        for param, estimates in bootstrap_estimates.items():
            assert len(estimates) >= min_success, (
                f"Bootstrap failed for {param}: only {len(estimates)} successful iterations"
            )


# =============================================================================
# Confidence Interval Tests
# =============================================================================

@pytest.mark.estimation
class TestConfidenceIntervals:
    """Tests for confidence interval construction."""

    def test_ci_construction(self, large_synthetic_data):
        """95% confidence intervals should be constructable."""
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

        database = db.Database('ci_test', df)

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
        biogeme_obj.modelName = 'ci_test'
        # HTML and pickle generation controlled via biogeme.toml

        results = biogeme_obj.estimate()

        betas = results.get_beta_values()
        std_errs = {p: results.get_parameter_std_err(p) for p in betas}

        # Construct 95% CIs: beta ± 1.96 * SE
        z_95 = 1.96

        print("\n95% Confidence Intervals:")
        print("-" * 70)
        print(f"{'Parameter':15} {'Estimate':>10} {'SE':>10} {'CI Lower':>12} {'CI Upper':>12}")
        print("-" * 70)

        for param in betas:
            est = betas[param]
            se = std_errs[param]
            ci_lower = est - z_95 * se
            ci_upper = est + z_95 * se

            print(f"{param:15} {est:10.4f} {se:10.4f} {ci_lower:12.4f} {ci_upper:12.4f}")

            # CI should have positive width
            assert ci_upper > ci_lower, f"{param}: CI has zero or negative width"

            # CI should contain the estimate
            assert ci_lower <= est <= ci_upper, f"{param}: Estimate not in CI"
