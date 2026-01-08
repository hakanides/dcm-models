"""
Pytest Configuration and Shared Fixtures
=========================================

Provides common test fixtures for DCM testing.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))


# =============================================================================
# Configuration Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def project_root():
    """Return project root path."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def model_config():
    """Load model configuration."""
    config_path = PROJECT_ROOT / 'config' / 'model_config_advanced.json'
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)

    # Fallback to simple config
    config_path = PROJECT_ROOT / 'config' / 'model_config.json'
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return None


@pytest.fixture(scope="session")
def model_config_path():
    """Return path to model configuration file."""
    config_path = PROJECT_ROOT / 'config' / 'model_config_advanced.json'
    if config_path.exists():
        return str(config_path)

    # Fallback to simple config
    config_path = PROJECT_ROOT / 'config' / 'model_config.json'
    if config_path.exists():
        return str(config_path)
    return None


@pytest.fixture(scope="session")
def items_config():
    """Load items configuration."""
    config_path = PROJECT_ROOT / 'config' / 'items_config_advanced.csv'
    if config_path.exists():
        return pd.read_csv(config_path)
    return None


# =============================================================================
# Data Fixtures - Synthetic Data with Known Parameters
# =============================================================================

@pytest.fixture
def true_parameters():
    """Known true parameters for validation."""
    return {
        'ASC_paid': 2.0,
        'B_FEE': -0.5,  # Per 10k TL
        'B_DUR': -0.025,
        'B_FEE_PatBlind': 0.15,
        'B_FEE_SecDL': -0.10,
        # Structural model parameters
        'gamma_pat_blind_age': 0.15,
        'gamma_pat_blind_income': -0.10,
        # Measurement model (loadings)
        'lambda_pat_blind_1': 1.0,  # Fixed for identification
        'lambda_pat_blind_2': 0.85,
        'lambda_pat_blind_3': 0.78,
    }


@pytest.fixture
def sample_choice_data():
    """Create minimal synthetic choice data for testing."""
    np.random.seed(42)
    n_individuals = 50
    n_tasks = 10

    rows = []
    for i in range(n_individuals):
        # Individual-level attributes
        age_idx = np.random.choice([0, 1, 2, 3, 4])
        edu_idx = np.random.choice([0, 1, 2, 3, 4, 5])
        income_indiv_idx = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7])
        income_house_idx = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7])
        marital_idx = np.random.choice([0, 1, 2])

        # Generate true latent variables
        age_c = (age_idx - 2) / 2
        inc_c = (income_indiv_idx - 3) / 2

        # True LV values (for validation)
        pat_blind_true = 0.15 * age_c - 0.10 * inc_c + np.random.normal(0, 1)
        sec_dl_true = -0.05 * age_c + 0.12 * inc_c + np.random.normal(0, 1)

        # Likert items (1-5 scale) from true LVs
        pat_blind_items = []
        for loading in [1.0, 0.85, 0.78, 0.72]:
            latent_value = loading * pat_blind_true
            # Ordered probit thresholds
            thresholds = [-1.5, -0.5, 0.5, 1.5]
            prob = np.random.random()
            if prob < 0.1:
                item = 1
            elif prob < 0.3:
                item = 2
            elif prob < 0.7:
                item = 3
            elif prob < 0.9:
                item = 4
            else:
                item = 5
            pat_blind_items.append(item)

        # Similar for other constructs (simplified)
        pat_const = np.random.randint(1, 6, 4)
        sec_dl_items = np.random.randint(1, 6, 4)
        sec_fp = np.random.randint(1, 6, 4)

        for t in range(n_tasks):
            # Choice task attributes
            fee1 = np.random.uniform(50000, 500000)
            fee2 = np.random.uniform(50000, 500000)
            fee3 = 0  # Free alternative
            dur1 = np.random.randint(1, 25)
            dur2 = np.random.randint(1, 25)
            dur3 = np.random.randint(1, 25)

            # Calculate utilities with true parameters
            fee1_10k = fee1 / 10000
            fee2_10k = fee2 / 10000
            fee3_10k = 0

            # Base coefficients
            ASC = 2.0
            B_FEE = -0.5
            B_DUR = -0.025
            B_FEE_PAT = 0.15

            # Individual fee sensitivity (modified by LV)
            B_FEE_i = B_FEE + B_FEE_PAT * pat_blind_true

            V1 = ASC + B_FEE_i * fee1_10k + B_DUR * dur1
            V2 = ASC + B_FEE_i * fee2_10k + B_DUR * dur2
            V3 = B_FEE_i * fee3_10k + B_DUR * dur3

            # Logit probabilities
            utils = [V1, V2, V3]
            max_u = max(utils)
            exp_utils = [np.exp(u - max_u) for u in utils]
            sum_exp = sum(exp_utils)
            probs = [e / sum_exp for e in exp_utils]

            choice = np.random.choice([1, 2, 3], p=probs)

            rows.append({
                'ID': i + 1,
                'TASK_ID': t + 1,
                'CHOICE': choice,
                'fee1': fee1,
                'fee2': fee2,
                'fee3': fee3,
                'dur1': dur1,
                'dur2': dur2,
                'dur3': dur3,
                'age_idx': age_idx,
                'edu_idx': edu_idx,
                'income_indiv_idx': income_indiv_idx,
                'income_house_idx': income_house_idx,
                'marital_idx': marital_idx,
                'pat_blind_1': pat_blind_items[0],
                'pat_blind_2': pat_blind_items[1],
                'pat_blind_3': pat_blind_items[2],
                'pat_blind_4': pat_blind_items[3],
                'pat_constructive_1': pat_const[0],
                'pat_constructive_2': pat_const[1],
                'pat_constructive_3': pat_const[2],
                'pat_constructive_4': pat_const[3],
                'sec_dl_1': sec_dl_items[0],
                'sec_dl_2': sec_dl_items[1],
                'sec_dl_3': sec_dl_items[2],
                'sec_dl_4': sec_dl_items[3],
                'sec_fp_1': sec_fp[0],
                'sec_fp_2': sec_fp[1],
                'sec_fp_3': sec_fp[2],
                'sec_fp_4': sec_fp[3],
                # True LV values for validation
                'pat_blind_true': pat_blind_true,
                'sec_dl_true': sec_dl_true,
            })

    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def large_synthetic_data():
    """
    Larger dataset (200 individuals × 20 tasks = 4000 obs) for parameter recovery.

    Methodological note: The smaller 500 observation fixture (sample_choice_data)
    may not guarantee convergence or accurate parameter recovery. Use this
    fixture for statistical validation and Monte Carlo studies.

    True parameters:
        ASC_paid = 2.0
        B_FEE = -0.5 (per 10k TL)
        B_DUR = -0.025
        B_FEE_PatBlind = 0.15
        gamma_pat_blind_age = 0.15
        gamma_pat_blind_income = -0.10
    """
    np.random.seed(42)
    n_individuals = 200
    n_tasks = 20

    # True parameters
    TRUE_ASC = 2.0
    TRUE_B_FEE = -0.5
    TRUE_B_DUR = -0.025
    TRUE_B_FEE_PAT = 0.15
    TRUE_GAMMA_AGE = 0.15
    TRUE_GAMMA_INC = -0.10

    rows = []
    for i in range(n_individuals):
        # Individual-level attributes
        age_idx = np.random.choice([0, 1, 2, 3, 4])
        edu_idx = np.random.choice([0, 1, 2, 3, 4, 5])
        income_indiv_idx = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7])
        income_house_idx = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7])
        marital_idx = np.random.choice([0, 1, 2])

        # Centered demographics
        age_c = (age_idx - 2) / 2
        inc_c = (income_indiv_idx - 3) / 2

        # True latent variables (structural model)
        pat_blind_true = TRUE_GAMMA_AGE * age_c + TRUE_GAMMA_INC * inc_c + np.random.normal(0, 1)
        sec_dl_true = -0.05 * age_c + 0.12 * inc_c + np.random.normal(0, 1)

        # Generate Likert items from true LVs (ordered probit)
        def generate_likert(lv_value, loading):
            """Generate ordinal response from latent value."""
            y_star = loading * lv_value + np.random.normal(0, 1)
            thresholds = [-1.0, -0.35, 0.35, 1.0]  # New balanced thresholds
            if y_star <= thresholds[0]:
                return 1
            elif y_star <= thresholds[1]:
                return 2
            elif y_star <= thresholds[2]:
                return 3
            elif y_star <= thresholds[3]:
                return 4
            else:
                return 5

        # Pat blind items (loadings: 1.0, 0.85, 0.78, 0.72)
        pat_blind_items = [
            generate_likert(pat_blind_true, 1.0),
            generate_likert(pat_blind_true, 0.85),
            generate_likert(pat_blind_true, 0.78),
            generate_likert(pat_blind_true, 0.72),
        ]

        # Pat constructive items
        pat_const_true = np.random.normal(0, 1)
        pat_const_items = [generate_likert(pat_const_true, l) for l in [1.0, 0.82, 0.75, 0.70]]

        # Sec DL items
        sec_dl_items = [generate_likert(sec_dl_true, l) for l in [1.0, 0.80, 0.73, 0.68]]

        # Sec FP items
        sec_fp_true = np.random.normal(0, 1)
        sec_fp_items = [generate_likert(sec_fp_true, l) for l in [1.0, 0.78, 0.72, 0.65]]

        # Individual-specific fee sensitivity
        B_FEE_i = TRUE_B_FEE + TRUE_B_FEE_PAT * pat_blind_true

        for t in range(n_tasks):
            # Choice task attributes
            fee1 = np.random.uniform(50000, 500000)
            fee2 = np.random.uniform(50000, 500000)
            fee3 = 0
            dur1 = np.random.randint(1, 25)
            dur2 = np.random.randint(1, 25)
            dur3 = np.random.randint(1, 25)

            # Calculate utilities with true parameters
            fee1_10k = fee1 / 10000
            fee2_10k = fee2 / 10000
            fee3_10k = 0

            V1 = TRUE_ASC + B_FEE_i * fee1_10k + TRUE_B_DUR * dur1
            V2 = TRUE_ASC + B_FEE_i * fee2_10k + TRUE_B_DUR * dur2
            V3 = B_FEE_i * fee3_10k + TRUE_B_DUR * dur3

            # Logit probabilities
            utils = [V1, V2, V3]
            max_u = max(utils)
            exp_utils = [np.exp(u - max_u) for u in utils]
            sum_exp = sum(exp_utils)
            probs = [e / sum_exp for e in exp_utils]

            choice = np.random.choice([1, 2, 3], p=probs)

            rows.append({
                'ID': i + 1,
                'TASK_ID': t + 1,
                'CHOICE': choice,
                'fee1': fee1,
                'fee2': fee2,
                'fee3': fee3,
                'dur1': dur1,
                'dur2': dur2,
                'dur3': dur3,
                'age_idx': age_idx,
                'edu_idx': edu_idx,
                'income_indiv_idx': income_indiv_idx,
                'income_house_idx': income_house_idx,
                'marital_idx': marital_idx,
                'pat_blind_1': pat_blind_items[0],
                'pat_blind_2': pat_blind_items[1],
                'pat_blind_3': pat_blind_items[2],
                'pat_blind_4': pat_blind_items[3],
                'pat_constructive_1': pat_const_items[0],
                'pat_constructive_2': pat_const_items[1],
                'pat_constructive_3': pat_const_items[2],
                'pat_constructive_4': pat_const_items[3],
                'sec_dl_1': sec_dl_items[0],
                'sec_dl_2': sec_dl_items[1],
                'sec_dl_3': sec_dl_items[2],
                'sec_dl_4': sec_dl_items[3],
                'sec_fp_1': sec_fp_items[0],
                'sec_fp_2': sec_fp_items[1],
                'sec_fp_3': sec_fp_items[2],
                'sec_fp_4': sec_fp_items[3],
                # True values for validation
                'pat_blind_true': pat_blind_true,
                'sec_dl_true': sec_dl_true,
                'B_FEE_individual': B_FEE_i,
            })

    return pd.DataFrame(rows)


@pytest.fixture
def small_choice_data():
    """Minimal data for quick unit tests."""
    np.random.seed(123)
    n_individuals = 10
    n_tasks = 5

    rows = []
    for i in range(n_individuals):
        age_idx = np.random.choice([0, 1, 2, 3, 4])
        for t in range(n_tasks):
            fee1 = np.random.uniform(10000, 100000)
            fee2 = np.random.uniform(10000, 100000)

            rows.append({
                'ID': i + 1,
                'TASK_ID': t + 1,
                'CHOICE': np.random.choice([1, 2, 3]),
                'fee1': fee1,
                'fee2': fee2,
                'fee3': 0,
                'dur1': np.random.randint(1, 12),
                'dur2': np.random.randint(1, 12),
                'dur3': np.random.randint(1, 12),
                'age_idx': age_idx,
                'edu_idx': np.random.choice([0, 1, 2, 3]),
                'income_indiv_idx': np.random.choice([0, 1, 2, 3]),
                'pat_blind_1': np.random.randint(1, 6),
                'pat_blind_2': np.random.randint(1, 6),
                'pat_blind_3': np.random.randint(1, 6),
                'sec_dl_1': np.random.randint(1, 6),
                'sec_dl_2': np.random.randint(1, 6),
                'sec_dl_3': np.random.randint(1, 6),
            })

    return pd.DataFrame(rows)


@pytest.fixture
def real_data_path():
    """Path to real test data if it exists."""
    path = PROJECT_ROOT / 'data' / 'simulated' / 'fresh_simulation.csv'
    if path.exists():
        return path

    path = PROJECT_ROOT / 'data' / 'simulated' / 'simulated_data.csv'
    if path.exists():
        return path
    return None


# =============================================================================
# Database Fixtures (Biogeme)
# =============================================================================

@pytest.fixture
def biogeme_database(sample_choice_data):
    """Prepare a Biogeme database from sample data."""
    try:
        import biogeme.database as db
    except ImportError:
        pytest.skip("Biogeme not installed")

    df = sample_choice_data.copy()

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
        'sec_dl': ['sec_dl_1', 'sec_dl_2', 'sec_dl_3', 'sec_dl_4'],
    }

    for lv_name, items in lv_items.items():
        available = [c for c in items if c in df.columns]
        if available:
            proxy = df[available].mean(axis=1)
            df[f'{lv_name}_proxy'] = (proxy - proxy.mean()) / proxy.std()

    # Drop non-numeric columns
    numeric_df = df.select_dtypes(include=[np.number])

    return db.Database('test_db', numeric_df)


# =============================================================================
# ICLV Fixtures
# =============================================================================

@pytest.fixture
def iclv_constructs():
    """Construct definitions for ICLV tests."""
    return {
        'pat_blind': ['pat_blind_1', 'pat_blind_2', 'pat_blind_3'],
    }


@pytest.fixture
def iclv_data(sample_choice_data):
    """Prepare data for ICLV testing."""
    df = sample_choice_data.copy()

    # Scale fees
    df['fee1_10k'] = df['fee1'] / 10000.0
    df['fee2_10k'] = df['fee2'] / 10000.0
    df['fee3_10k'] = df['fee3'] / 10000.0

    # Center demographics
    df['age_c'] = (df['age_idx'] - 2) / 2
    df['inc_c'] = (df['income_indiv_idx'] - 3) / 2

    return df


# =============================================================================
# Policy Analysis Fixtures
# =============================================================================

@pytest.fixture
def estimation_result():
    """Sample estimation result for policy analysis tests."""
    try:
        from src.policy_analysis import EstimationResult
        return EstimationResult(
            betas={'B_FEE': -0.5, 'B_DUR': -0.08, 'ASC_paid': 0.5},
            std_errs={'B_FEE': 0.05, 'B_DUR': 0.02, 'ASC_paid': 0.15}
        )
    except ImportError:
        pytest.skip("Policy analysis module not available")


@pytest.fixture
def policy_scenario():
    """Sample policy scenario for tests."""
    try:
        from src.policy_analysis import PolicyScenario
        return PolicyScenario(
            name='Test',
            attributes={
                'fee': np.array([20000, 25000, 0]),
                'dur': np.array([5, 4, 8])
            }
        )
    except ImportError:
        pytest.skip("Policy analysis module not available")


# =============================================================================
# Helper Functions
# =============================================================================

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
