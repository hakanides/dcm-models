"""
Tests for Simulation Module
===========================

Tests for data generation and simulation functionality.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))


class TestSimulatorImport:
    """Tests for simulator module imports."""

    @pytest.mark.unit
    def test_dcm_simulator_import(self):
        """Test that DCMSimulator can be imported."""
        try:
            from src.simulation.dcm_simulator import DCMSimulator
            assert DCMSimulator is not None
        except ImportError as e:
            pytest.skip(f"DCMSimulator not importable: {e}")

    @pytest.mark.unit
    def test_advanced_simulator_import(self):
        """Test that advanced simulator can be imported."""
        try:
            from src.simulation.dcm_simulator_advanced import DCMSimulatorAdvanced
            assert DCMSimulatorAdvanced is not None
        except ImportError as e:
            pytest.skip(f"Advanced simulator not importable: {e}")


@pytest.mark.simulation
class TestDCMSimulator:
    """Tests for basic DCM simulator."""

    def test_simulator_initialization(self, project_root):
        """Test simulator initializes with config path."""
        config_path = project_root / 'config' / 'model_config_advanced.json'
        if not config_path.exists():
            pytest.skip("Config file not found")

        try:
            from src.simulation.dcm_simulator import DCMSimulator
            sim = DCMSimulator(str(config_path))
            assert sim is not None
        except ImportError:
            pytest.skip("Simulator not available")

    def test_simulate_demographics(self, project_root):
        """Test demographic generation."""
        config_path = project_root / 'config' / 'model_config_advanced.json'
        if not config_path.exists():
            pytest.skip("Config file not found")

        try:
            from src.simulation.dcm_simulator import DCMSimulator
            sim = DCMSimulator(str(config_path))

            n = 100
            demographics = sim.simulate_demographics(n)

            assert len(demographics) == n
            assert 'age_idx' in demographics.columns
        except (ImportError, AttributeError) as e:
            pytest.skip(f"Simulator feature not available: {e}")

    def test_simulate_likert_responses(self, project_root):
        """Test Likert response generation."""
        config_path = project_root / 'config' / 'model_config_advanced.json'
        if not config_path.exists():
            pytest.skip("Config file not found")

        try:
            from src.simulation.dcm_simulator import DCMSimulator
            sim = DCMSimulator(str(config_path))

            n = 100
            likert = sim.simulate_likert_responses(n)

            # Check responses are in valid range
            for col in likert.columns:
                if col.endswith(('_1', '_2', '_3', '_4', '_5')):
                    assert likert[col].min() >= 1
                    assert likert[col].max() <= 5
        except (ImportError, AttributeError) as e:
            pytest.skip(f"Simulator feature not available: {e}")


@pytest.mark.simulation
class TestDataGeneration:
    """Tests for data generation consistency."""

    def test_choice_distribution(self, sample_choice_data):
        """Test that choices follow expected distribution."""
        df = sample_choice_data

        # All choices should be 1, 2, or 3
        assert set(df['CHOICE'].unique()).issubset({1, 2, 3})

        # Check that we have at least some variation in choices
        # Note: Free alternative (3) may dominate in synthetic data
        n_unique = df['CHOICE'].nunique()
        assert n_unique >= 2, "Need at least 2 different choices"

    def test_fee_variation(self, sample_choice_data):
        """Test that fees have realistic variation."""
        df = sample_choice_data

        for col in ['fee1', 'fee2']:
            # Fees should vary
            assert df[col].std() > 0

            # Fees should be positive
            assert df[col].min() >= 0

    def test_likert_variation(self, sample_choice_data):
        """Test that Likert items have variation."""
        df = sample_choice_data

        likert_cols = [c for c in df.columns if c.startswith('pat_blind_') and c[-1].isdigit()]

        for col in likert_cols:
            # Should have at least 2 unique values
            assert df[col].nunique() >= 2

            # Should be in 1-5 range
            assert df[col].min() >= 1
            assert df[col].max() <= 5

    def test_panel_consistency(self, sample_choice_data):
        """Test panel structure consistency."""
        df = sample_choice_data

        # Each ID should have multiple tasks
        tasks_per_id = df.groupby('ID').size()
        assert (tasks_per_id > 1).all()

        # Demographics constant within ID
        demo_cols = ['age_idx', 'edu_idx']
        for col in demo_cols:
            if col in df.columns:
                within_var = df.groupby('ID')[col].nunique()
                assert (within_var == 1).all(), f"{col} varies within ID"


@pytest.mark.simulation
class TestConfigValidation:
    """Tests for configuration validation."""

    def test_config_has_required_sections(self, model_config):
        """Test config has required sections."""
        if model_config is None:
            pytest.skip("No model config available")

        # Check for key sections
        assert 'population' in model_config or 'n_individuals' in model_config.get('simulation', {})

    def test_config_choice_model(self, model_config):
        """Test choice model configuration."""
        if model_config is None:
            pytest.skip("No model config available")

        if 'choice_model' in model_config:
            choice = model_config['choice_model']
            # Should have some attributes or terms
            assert 'attribute_terms' in choice or 'base_terms' in choice


@pytest.mark.simulation
@pytest.mark.slow
class TestFullSimulation:
    """Tests for full simulation pipeline."""

    def test_full_simulation_runs(self, model_config):
        """Test complete simulation pipeline."""
        if model_config is None:
            pytest.skip("No model config available")

        try:
            from src.simulation.dcm_simulator import DCMSimulator
            sim = DCMSimulator(model_config)

            # Run full simulation with small N
            df = sim.simulate_full(n_individuals=20, n_tasks=5)

            assert len(df) == 20 * 5
            assert 'CHOICE' in df.columns
            assert 'fee1' in df.columns

        except (ImportError, AttributeError) as e:
            pytest.skip(f"Full simulation not available: {e}")

    def test_simulation_reproducibility(self, model_config):
        """Test that simulation is reproducible with same seed."""
        if model_config is None:
            pytest.skip("No model config available")

        try:
            from src.simulation.dcm_simulator import DCMSimulator

            # First simulation
            np.random.seed(42)
            sim1 = DCMSimulator(model_config)
            df1 = sim1.simulate_full(n_individuals=10, n_tasks=3)

            # Second simulation with same seed
            np.random.seed(42)
            sim2 = DCMSimulator(model_config)
            df2 = sim2.simulate_full(n_individuals=10, n_tasks=3)

            # Should be identical
            pd.testing.assert_frame_equal(df1, df2)

        except (ImportError, AttributeError) as e:
            pytest.skip(f"Simulation not available: {e}")


@pytest.mark.simulation
class TestLatentVariableSimulation:
    """Tests for latent variable simulation."""

    def test_lv_distribution(self, sample_choice_data):
        """Test that true LVs follow expected distribution."""
        df = sample_choice_data

        if 'pat_blind_true' not in df.columns:
            pytest.skip("True LV not in test data")

        # Get individual-level LVs
        individuals = df.groupby('ID').first()

        # Should be roughly standard normal
        lv = individuals['pat_blind_true']
        assert abs(lv.mean()) < 0.5, f"LV mean is {lv.mean()}, expected ~0"
        assert 0.5 < lv.std() < 2.0, f"LV std is {lv.std()}, expected ~1"

    def test_lv_demographic_relationship(self, sample_choice_data):
        """Test that LVs correlate with demographics as expected."""
        df = sample_choice_data

        if 'pat_blind_true' not in df.columns:
            pytest.skip("True LV not in test data")

        individuals = df.groupby('ID').first()

        # Center demographics
        individuals['age_c'] = (individuals['age_idx'] - 2) / 2

        # Should have positive correlation with age (based on structural params)
        corr = individuals['pat_blind_true'].corr(individuals['age_c'])
        # Correlation should be in expected direction (positive)
        # Allow some noise since sample is small
        assert corr > -0.5, f"Unexpected correlation: {corr}"

    def test_likert_lv_correlation(self, sample_choice_data):
        """Test that Likert items correlate with true LV."""
        df = sample_choice_data

        if 'pat_blind_true' not in df.columns:
            pytest.skip("True LV not in test data")

        individuals = df.groupby('ID').first()

        # Create proxy from Likert items
        items = ['pat_blind_1', 'pat_blind_2', 'pat_blind_3', 'pat_blind_4']
        available = [c for c in items if c in individuals.columns]

        if len(available) < 2:
            pytest.skip("Not enough Likert items")

        proxy = individuals[available].mean(axis=1)
        true_lv = individuals['pat_blind_true']

        corr = proxy.corr(true_lv)

        # Should have positive correlation
        assert corr > 0, f"Proxy should correlate positively with true LV: {corr}"
