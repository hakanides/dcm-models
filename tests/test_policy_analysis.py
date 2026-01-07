"""
Tests for Policy Analysis Module
================================
"""

import pytest
import numpy as np
import pandas as pd
from src.policy_analysis import (
    EstimationResult,
    PolicyScenario,
    PolicyAnalysisConfig,
    WTPCalculator,
    ElasticityCalculator,
    MarginalEffectCalculator,
    DemandForecaster,
    WelfareAnalyzer,
    compute_logit_probabilities,
    compute_utilities,
    compute_wtp_quick
)


@pytest.fixture
def sample_result():
    """Create sample estimation result for tests."""
    return EstimationResult(
        betas={'B_FEE': -0.5, 'B_DUR': -0.08, 'ASC_paid': 0.5},
        std_errs={'B_FEE': 0.05, 'B_DUR': 0.02, 'ASC_paid': 0.15}
    )


@pytest.fixture
def sample_scenario():
    """Create sample scenario with reasonable values."""
    # Use smaller fee values for more reasonable probabilities
    return PolicyScenario(
        name='Test',
        attributes={
            'fee': np.array([20000, 25000, 0]),  # 2, 2.5, 0 when scaled by 10000
            'dur': np.array([5, 4, 8])
        }
    )


@pytest.fixture
def config():
    """Create default configuration."""
    return PolicyAnalysisConfig()


class TestEstimationResult:
    """Tests for EstimationResult class."""

    def test_creation(self, sample_result):
        """Test basic creation."""
        assert sample_result.betas['B_FEE'] == -0.5
        assert sample_result.std_errs['B_FEE'] == 0.05

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            'betas': {'B_FEE': -0.3},
            'std_errs': {'B_FEE': 0.05}
        }
        result = EstimationResult.from_dict(data)
        assert result.betas['B_FEE'] == -0.3

    def test_get_t_stat(self, sample_result):
        """Test t-statistic calculation."""
        t_stat = sample_result.get_t_stat('B_FEE')
        expected = -0.5 / 0.05
        assert t_stat == expected

    def test_is_significant(self, sample_result):
        """Test significance check."""
        # B_FEE has t = -10, definitely significant
        assert sample_result.is_significant('B_FEE')


class TestPolicyScenario:
    """Tests for PolicyScenario class."""

    def test_creation(self, sample_scenario):
        """Test basic creation."""
        assert sample_scenario.name == 'Test'
        assert len(sample_scenario.attributes['fee']) == 3

    def test_get_attribute(self, sample_scenario):
        """Test attribute retrieval."""
        fee = sample_scenario.get_attribute('fee', 0)
        assert fee == 20000

    def test_with_modification(self, sample_scenario):
        """Test scenario modification."""
        modified = sample_scenario.with_modification('fee', 0, 15000)
        assert modified.get_attribute('fee', 0) == 15000
        # Original unchanged
        assert sample_scenario.get_attribute('fee', 0) == 20000


class TestComputeLogitProbabilities:
    """Tests for logit probability computation."""

    def test_probabilities_sum_to_one(self):
        """Probabilities should sum to 1."""
        utilities = np.array([1.0, 0.5, 0.0])
        probs = compute_logit_probabilities(utilities)
        assert np.isclose(np.sum(probs), 1.0)

    def test_higher_utility_higher_prob(self):
        """Higher utility means higher probability."""
        utilities = np.array([2.0, 1.0, 0.0])
        probs = compute_logit_probabilities(utilities)
        assert probs[0] > probs[1] > probs[2]

    def test_numerical_stability(self):
        """Should handle large utility differences."""
        utilities = np.array([100.0, 0.0, -100.0])
        probs = compute_logit_probabilities(utilities)
        assert np.isclose(np.sum(probs), 1.0)
        assert probs[0] > 0.99  # Almost certain


class TestWTPCalculator:
    """Tests for WTP calculations."""

    def test_wtp_computation(self, sample_result):
        """Test basic WTP computation."""
        calc = WTPCalculator(sample_result)
        wtp = calc.compute_wtp('B_DUR')

        # With for_improvement=True (default), WTP for "bad" attributes (β < 0)
        # is flipped to give WTP for REDUCTION (improvement).
        # B_DUR = -0.08 (negative = more duration is bad)
        # WTP for duration REDUCTION (saving days) = +1600 (positive)
        assert np.isclose(wtp.wtp_point, 1600)

    def test_wtp_raw_mrs(self, sample_result):
        """Test raw MRS formula (for_improvement=False)."""
        calc = WTPCalculator(sample_result)
        wtp = calc.compute_wtp('B_DUR', for_improvement=False)

        # Raw MRS: WTP = -B_DUR / B_FEE * scale
        # = -(-0.08) / (-0.5) * 10000 = -1600 (negative = wouldn't pay for MORE days)
        assert np.isclose(wtp.wtp_point, -1600)

    def test_wtp_has_confidence_interval(self, sample_result):
        """WTP should have CI."""
        calc = WTPCalculator(sample_result)
        wtp = calc.compute_wtp('B_DUR')
        assert wtp.ci_lower < wtp.wtp_point < wtp.ci_upper

    def test_wtp_quick(self):
        """Test quick WTP function."""
        betas = {'B_FEE': -0.5, 'B_DUR': -0.08}
        # Default for_improvement=True gives positive WTP for duration reduction
        wtp = compute_wtp_quick(betas)
        assert np.isclose(wtp, 1600)

        # Raw MRS with for_improvement=False
        wtp_raw = compute_wtp_quick(betas, for_improvement=False)
        assert np.isclose(wtp_raw, -1600)

    def test_wtp_krinsky_robb(self, sample_result):
        """Test Krinsky-Robb WTP."""
        calc = WTPCalculator(sample_result)
        wtp = calc.compute_wtp_krinsky_robb('B_DUR', n_draws=100)
        # With for_improvement=True (default), should be close to +1600
        assert abs(wtp.wtp_point - 1600) < 500  # Allow some simulation error


class TestElasticityCalculator:
    """Tests for elasticity calculations."""

    def test_own_price_elasticity(self, sample_result, sample_scenario):
        """Test own-price elasticity."""
        calc = ElasticityCalculator(sample_result)
        eta = calc.own_price_elasticity(sample_scenario, alternative=0)

        # η = β * x * (1 - P) where x is scaled fee
        # Negative for normal goods
        assert eta.elasticity < 0

    def test_cross_price_elasticity(self, sample_result, sample_scenario):
        """Test cross-price elasticity."""
        calc = ElasticityCalculator(sample_result)
        eta = calc.cross_price_elasticity(sample_scenario, alternative=0, with_respect_to=1)

        # Cross elasticity: -β * x * P, should be positive (substitutes)
        assert eta.elasticity > 0

    def test_elasticity_matrix_shape(self, sample_result, sample_scenario):
        """Test elasticity matrix dimensions."""
        calc = ElasticityCalculator(sample_result)
        matrix = calc.elasticity_matrix(sample_scenario)
        assert matrix.shape == (3, 3)


class TestMarginalEffectCalculator:
    """Tests for marginal effects."""

    def test_marginal_effect_at_point(self, sample_result, sample_scenario):
        """Test point marginal effect."""
        calc = MarginalEffectCalculator(sample_result)
        me = calc.marginal_effect_at_point(sample_scenario, alternative=0)

        # ME = β * P * (1-P) / scale
        # Should be negative for fee
        assert me.me < 0

    def test_cross_marginal_effect(self, sample_result, sample_scenario):
        """Test cross marginal effect."""
        calc = MarginalEffectCalculator(sample_result)
        me = calc.cross_marginal_effect(sample_scenario, alternative=0, with_respect_to=1)

        # Cross ME = -β * P_j * P_k / scale
        # Should be positive (fee increase in j helps k)
        assert me.me > 0

    def test_marginal_effect_matrix_shape(self, sample_result, sample_scenario):
        """Test ME matrix dimensions."""
        calc = MarginalEffectCalculator(sample_result)
        matrix = calc.marginal_effect_matrix(sample_scenario)
        assert matrix.shape == (3, 3)


class TestDemandForecaster:
    """Tests for demand forecasting."""

    def test_predict_shares_sum_to_one(self, sample_result, sample_scenario):
        """Predicted shares should sum to 1."""
        forecaster = DemandForecaster(sample_result)
        shares = forecaster.predict_market_shares(sample_scenario)
        assert np.isclose(np.sum(shares.shares), 1.0)

    def test_compare_scenarios(self, sample_result, sample_scenario):
        """Test scenario comparison."""
        forecaster = DemandForecaster(sample_result)

        # Create policy with lower fee on Alt 1
        policy = sample_scenario.with_modification('fee', 0, 15000)

        comparison = forecaster.compare_scenarios(sample_scenario, policy)

        # Lower fee should increase share
        assert comparison.share_changes[0] > 0

    def test_sensitivity_analysis(self, sample_result, sample_scenario):
        """Test sensitivity analysis."""
        forecaster = DemandForecaster(sample_result)
        sensitivity = forecaster.sensitivity_analysis(
            sample_scenario, 'fee', 0, [-10, 0, 10]
        )
        assert len(sensitivity) == 3


class TestWelfareAnalyzer:
    """Tests for welfare analysis."""

    def test_consumer_surplus(self, sample_result, sample_scenario):
        """Test CS computation."""
        analyzer = WelfareAnalyzer(sample_result)
        cs = analyzer.compute_consumer_surplus(sample_scenario)

        # CS should be finite
        assert np.isfinite(cs.cs)
        assert cs.logsum != 0

    def test_compensating_variation(self, sample_result, sample_scenario):
        """Test CV computation."""
        analyzer = WelfareAnalyzer(sample_result)

        # Policy with lower fee
        policy = sample_scenario.with_modification('fee', 0, 15000)

        cv = analyzer.compute_compensating_variation(sample_scenario, policy)

        # Lower fee should increase welfare (positive CV)
        assert cv.cv > 0

    def test_total_welfare_change(self, sample_result, sample_scenario):
        """Test total welfare computation."""
        analyzer = WelfareAnalyzer(sample_result)
        policy = sample_scenario.with_modification('fee', 0, 15000)

        total = analyzer.total_welfare_change(sample_scenario, policy, population=1000)

        assert 'total_welfare_change' in total
        assert total['population'] == 1000


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_analysis_workflow(self, sample_result, sample_scenario):
        """Test complete analysis workflow."""
        # WTP
        wtp_calc = WTPCalculator(sample_result)
        wtp = wtp_calc.compute_wtp('B_DUR')
        assert np.isfinite(wtp.wtp_point)

        # Elasticity
        elast_calc = ElasticityCalculator(sample_result)
        eta = elast_calc.own_price_elasticity(sample_scenario, 0)
        assert np.isfinite(eta.elasticity)

        # Marginal effects
        me_calc = MarginalEffectCalculator(sample_result)
        me = me_calc.marginal_effect_at_point(sample_scenario, 0)
        assert np.isfinite(me.me)

        # Market shares
        forecaster = DemandForecaster(sample_result)
        shares = forecaster.predict_market_shares(sample_scenario)
        assert np.allclose(np.sum(shares.shares), 1.0)

        # Welfare
        analyzer = WelfareAnalyzer(sample_result)
        cs = analyzer.compute_consumer_surplus(sample_scenario)
        assert np.isfinite(cs.cs)

    def test_dict_input(self, sample_scenario):
        """Test that calculators accept dict input."""
        result_dict = {
            'betas': {'B_FEE': -0.5, 'B_DUR': -0.08, 'ASC_paid': 0.5},
            'std_errs': {'B_FEE': 0.05, 'B_DUR': 0.02, 'ASC_paid': 0.15}
        }

        # All calculators should accept dict
        WTPCalculator(result_dict)
        ElasticityCalculator(result_dict)
        MarginalEffectCalculator(result_dict)
        DemandForecaster(result_dict)
        WelfareAnalyzer(result_dict)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
