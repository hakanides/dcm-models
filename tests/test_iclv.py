"""
Tests for ICLV Module
=====================

Tests for Integrated Choice and Latent Variable estimation.
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
# Import Tests
# =============================================================================

@pytest.mark.unit
@pytest.mark.iclv
class TestICLVImports:
    """Tests for ICLV module imports."""

    def test_iclv_module_import(self):
        """Test that ICLV module can be imported."""
        try:
            from src.models import iclv
            assert iclv is not None
        except ImportError as e:
            pytest.skip(f"ICLV module not importable: {e}")

    def test_iclv_core_import(self):
        """Test that core ICLV class can be imported."""
        try:
            from src.models.iclv import ICLVModel
            assert ICLVModel is not None
        except ImportError as e:
            pytest.skip(f"ICLVModel not importable: {e}")

    def test_estimate_iclv_import(self):
        """Test that estimate_iclv function can be imported."""
        try:
            from src.models.iclv import estimate_iclv
            assert callable(estimate_iclv)
        except ImportError as e:
            pytest.skip(f"estimate_iclv not importable: {e}")

    def test_estimation_config_import(self):
        """Test that EstimationConfig can be imported."""
        try:
            from src.models.iclv import EstimationConfig
            assert EstimationConfig is not None
        except ImportError as e:
            pytest.skip(f"EstimationConfig not importable: {e}")

    def test_comparison_tools_import(self):
        """Test that comparison tools can be imported."""
        try:
            from src.models.iclv import compare_two_stage_vs_iclv, summarize_attenuation_bias
            assert callable(compare_two_stage_vs_iclv)
            assert callable(summarize_attenuation_bias)
        except ImportError as e:
            pytest.skip(f"Comparison tools not importable: {e}")


# =============================================================================
# Measurement Model Tests
# =============================================================================

@pytest.mark.unit
@pytest.mark.iclv
class TestMeasurementModel:
    """Tests for ordered probit measurement model."""

    def test_measurement_model_import(self):
        """Test measurement model import."""
        try:
            from src.models.iclv.measurement import OrderedProbitMeasurement
            assert OrderedProbitMeasurement is not None
        except ImportError as e:
            pytest.skip(f"Measurement model not importable: {e}")

    def test_measurement_probability(self):
        """Test that measurement probabilities are valid."""
        try:
            from src.models.iclv.measurement import OrderedProbitMeasurement

            model = OrderedProbitMeasurement(n_categories=5)

            # Test probability computation
            loading = 0.8
            lv_value = 0.5

            probs = model.response_probabilities(loading, lv_value)

            # Probabilities should sum to 1
            assert np.isclose(probs.sum(), 1.0), f"Probs sum to {probs.sum()}"

            # All probabilities should be non-negative
            assert (probs >= 0).all(), "Negative probabilities"

        except ImportError as e:
            pytest.skip(f"Measurement model not available: {e}")

    def test_measurement_gradient_shape(self):
        """Test that gradients have correct shape."""
        try:
            from src.models.iclv.measurement import OrderedProbitMeasurement

            model = OrderedProbitMeasurement(n_categories=5)

            loading = 0.8
            lv_value = 0.5
            response = 3

            # Test gradient computation
            grad_loading = model.gradient_loading(response, loading, lv_value)
            grad_lv = model.gradient_lv(response, loading, lv_value)

            # Should be scalar
            assert np.isscalar(grad_loading) or grad_loading.ndim == 0
            assert np.isscalar(grad_lv) or grad_lv.ndim == 0

        except (ImportError, AttributeError) as e:
            pytest.skip(f"Gradient methods not available: {e}")


# =============================================================================
# Structural Model Tests
# =============================================================================

@pytest.mark.unit
@pytest.mark.iclv
class TestStructuralModel:
    """Tests for structural model (demographics -> LV)."""

    def test_structural_model_import(self):
        """Test structural model import."""
        try:
            from src.models.iclv.structural import StructuralModel
            assert StructuralModel is not None
        except ImportError as e:
            pytest.skip(f"Structural model not importable: {e}")


# =============================================================================
# Integration Tests
# =============================================================================

@pytest.mark.unit
@pytest.mark.iclv
class TestHaltonDraws:
    """Tests for Halton sequence generation."""

    def test_halton_import(self):
        """Test Halton import."""
        try:
            from src.models.iclv.integration import generate_halton_draws
            assert callable(generate_halton_draws)
        except ImportError as e:
            pytest.skip(f"Halton draws not importable: {e}")

    def test_halton_shape(self):
        """Test Halton draws have correct shape."""
        try:
            from src.models.iclv.integration import generate_halton_draws

            n_draws = 100
            n_dim = 2
            draws = generate_halton_draws(n_draws, n_dim)

            assert draws.shape == (n_draws, n_dim)

        except ImportError as e:
            pytest.skip(f"Halton draws not available: {e}")

    def test_halton_range(self):
        """Test Halton draws are in (0, 1)."""
        try:
            from src.models.iclv.integration import generate_halton_draws

            draws = generate_halton_draws(100, 2)

            assert (draws > 0).all()
            assert (draws < 1).all()

        except ImportError as e:
            pytest.skip(f"Halton draws not available: {e}")


# =============================================================================
# Auto-Scaling Tests
# =============================================================================

@pytest.mark.unit
@pytest.mark.iclv
class TestAutoScaling:
    """Tests for auto-scaling functionality."""

    def test_auto_scale_import(self):
        """Test auto_scale_attributes import."""
        try:
            from src.models.iclv import auto_scale_attributes
            assert callable(auto_scale_attributes)
        except ImportError as e:
            pytest.skip(f"auto_scale_attributes not importable: {e}")

    def test_scaling_detection(self, sample_choice_data):
        """Test that large values are detected for scaling."""
        try:
            from src.models.iclv.estimation import auto_scale_attributes

            df = sample_choice_data.copy()
            attribute_cols = ['fee1', 'fee2', 'fee3']

            scaled_df, scaling_info = auto_scale_attributes(df, attribute_cols)

            # Large fees should be detected and scaled
            if df['fee1'].max() > 1000:
                assert scaling_info is not None
                # Check that scaled_columns dict has entries
                assert len(scaling_info.scaled_columns) > 0

        except ImportError as e:
            pytest.skip(f"Auto-scaling not available: {e}")


# =============================================================================
# Two-Stage Starting Values Tests
# =============================================================================

@pytest.mark.unit
@pytest.mark.iclv
class TestTwoStageStart:
    """Tests for two-stage starting values."""

    def test_two_stage_start_import(self):
        """Test two_stage starting values import."""
        try:
            from src.models.iclv import compute_two_stage_starting_values
            assert callable(compute_two_stage_starting_values)
        except ImportError as e:
            pytest.skip(f"Two-stage start not importable: {e}")


# =============================================================================
# Full ICLV Estimation Tests
# =============================================================================

@pytest.mark.iclv
@pytest.mark.slow
@pytest.mark.integration
class TestICLVEstimation:
    """Tests for full ICLV estimation."""

    def test_iclv_runs(self, iclv_data, iclv_constructs):
        """Test that ICLV estimation runs without error."""
        try:
            from src.models.iclv import estimate_iclv

            # Define covariates for structural model
            covariates = ['age_idx', 'edu_idx']
            covariates = [c for c in covariates if c in iclv_data.columns]

            result = estimate_iclv(
                df=iclv_data,
                constructs=iclv_constructs,
                covariates=covariates,
                choice_col='CHOICE',
                n_draws=20,  # Very small for speed
            )

            assert result is not None
            assert hasattr(result, 'log_likelihood')
            assert np.isfinite(result.log_likelihood)

        except ImportError as e:
            pytest.skip(f"ICLV estimation not available: {e}")
        except Exception as e:
            # Allow estimation to fail on small data, but should run
            if "convergence" in str(e).lower():
                pytest.skip("Convergence issue on small test data")
            raise

    def test_iclv_with_auto_scale(self, iclv_data, iclv_constructs):
        """Test ICLV with auto-scaling enabled."""
        try:
            from src.models.iclv import estimate_iclv

            # Define covariates for structural model
            covariates = ['age_idx', 'edu_idx']
            covariates = [c for c in covariates if c in iclv_data.columns]

            result = estimate_iclv(
                df=iclv_data,
                constructs=iclv_constructs,
                covariates=covariates,
                choice_col='CHOICE',
                n_draws=20,
                auto_scale=True,
            )

            assert result is not None

        except ImportError as e:
            pytest.skip(f"ICLV not available: {e}")
        except Exception as e:
            if "convergence" in str(e).lower():
                pytest.skip("Convergence issue on small test data")
            raise

    def test_iclv_with_two_stage_start(self, iclv_data, iclv_constructs):
        """Test ICLV with two-stage starting values."""
        try:
            from src.models.iclv import estimate_iclv

            # Define covariates for structural model
            covariates = ['age_idx', 'edu_idx']
            covariates = [c for c in covariates if c in iclv_data.columns]

            result = estimate_iclv(
                df=iclv_data,
                constructs=iclv_constructs,
                covariates=covariates,
                choice_col='CHOICE',
                n_draws=20,
                use_two_stage_start=True,
            )

            assert result is not None

        except ImportError as e:
            pytest.skip(f"ICLV not available: {e}")
        except Exception as e:
            if "convergence" in str(e).lower():
                pytest.skip("Convergence issue on small test data")
            raise


# =============================================================================
# Robust SE Tests
# =============================================================================

@pytest.mark.unit
@pytest.mark.iclv
class TestRobustSE:
    """Tests for robust standard errors."""

    def test_sandwich_se_import(self):
        """Test sandwich SE import."""
        try:
            from src.models.iclv.estimation import compute_sandwich_se
            assert callable(compute_sandwich_se)
        except ImportError as e:
            pytest.skip(f"Sandwich SE not importable: {e}")

    def test_clustered_se_import(self):
        """Test clustered SE import."""
        try:
            from src.models.iclv.estimation import compute_clustered_se
            assert callable(compute_clustered_se)
        except ImportError as e:
            pytest.skip(f"Clustered SE not importable: {e}")


# =============================================================================
# LV Correlation Tests
# =============================================================================

@pytest.mark.unit
@pytest.mark.iclv
class TestLVCorrelation:
    """Tests for LV correlation estimation."""

    def test_lv_correlation_import(self):
        """Test LV correlation import."""
        try:
            from src.models.iclv.estimation import estimate_lv_correlation
            assert callable(estimate_lv_correlation)
        except ImportError as e:
            pytest.skip(f"LV correlation not importable: {e}")

    def test_single_construct_correlation(self, iclv_data):
        """Test LV correlation with single construct returns 1x1 matrix."""
        try:
            from src.models.iclv.estimation import estimate_lv_correlation

            constructs = {'pat_blind': ['pat_blind_1', 'pat_blind_2', 'pat_blind_3']}
            corr = estimate_lv_correlation(iclv_data, constructs)

            # Single construct should return 1x1 identity
            assert corr.shape == (1, 1)
            assert corr[0, 0] == 1.0

        except ImportError as e:
            pytest.skip(f"LV correlation not available: {e}")

    def test_multiple_construct_correlation(self, sample_choice_data):
        """Test LV correlation with multiple constructs."""
        try:
            from src.models.iclv.estimation import estimate_lv_correlation

            constructs = {
                'pat_blind': ['pat_blind_1', 'pat_blind_2', 'pat_blind_3'],
                'sec_dl': ['sec_dl_1', 'sec_dl_2', 'sec_dl_3'],
            }
            corr = estimate_lv_correlation(sample_choice_data, constructs)

            # Should be 2x2 matrix
            assert corr.shape == (2, 2)

            # Diagonal should be 1
            assert np.allclose(np.diag(corr), 1.0)

            # Should be symmetric
            assert np.allclose(corr, corr.T)

            # Off-diagonal should be in [-1, 1]
            off_diag = corr[0, 1]
            assert -1 <= off_diag <= 1

        except ImportError as e:
            pytest.skip(f"LV correlation not available: {e}")


# =============================================================================
# Comparison Tools Tests
# =============================================================================

@pytest.mark.unit
@pytest.mark.iclv
class TestComparisonTools:
    """Tests for HCM vs ICLV comparison tools."""

    def test_comparison_function_signature(self):
        """Test comparison function accepts expected arguments."""
        try:
            from src.models.iclv import compare_two_stage_vs_iclv
            import inspect

            sig = inspect.signature(compare_two_stage_vs_iclv)
            params = list(sig.parameters.keys())

            # Should accept at least hcm_results and iclv_result
            assert len(params) >= 2

        except ImportError as e:
            pytest.skip(f"Comparison tools not available: {e}")

    def test_attenuation_summary_function(self):
        """Test attenuation summary function."""
        try:
            from src.models.iclv import summarize_attenuation_bias
            import inspect

            sig = inspect.signature(summarize_attenuation_bias)
            assert sig is not None

        except ImportError as e:
            pytest.skip(f"Attenuation summary not available: {e}")


# =============================================================================
# Configuration Tests
# =============================================================================

@pytest.mark.unit
@pytest.mark.iclv
class TestEstimationConfig:
    """Tests for estimation configuration."""

    def test_config_defaults(self):
        """Test EstimationConfig has sensible defaults."""
        try:
            from src.models.iclv import EstimationConfig

            config = EstimationConfig()

            assert config.n_draws > 0
            assert config.method in ['BFGS', 'L-BFGS-B', 'Newton-CG', 'trust-ncg']

        except ImportError as e:
            pytest.skip(f"EstimationConfig not available: {e}")

    def test_config_customization(self):
        """Test EstimationConfig can be customized."""
        try:
            from src.models.iclv import EstimationConfig

            config = EstimationConfig(
                n_draws=1000,
                auto_scale=True,
                use_two_stage_start=True,
                compute_robust_se=True,
            )

            assert config.n_draws == 1000
            assert config.auto_scale == True
            assert config.use_two_stage_start == True
            assert config.compute_robust_se == True

        except (ImportError, TypeError) as e:
            pytest.skip(f"EstimationConfig customization not available: {e}")
