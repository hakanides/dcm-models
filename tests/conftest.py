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
    config_path = PROJECT_ROOT / 'config' / 'model_config.json'
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
    config_path = PROJECT_ROOT / 'config' / 'model_config.json'
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
    config_path = PROJECT_ROOT / 'config' / 'items_config.csv'
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

        # Generate Likert items with new unified naming
        def generate_likert_item(lv_value, loading):
            y_star = loading * lv_value + np.random.normal(0, 1)
            thresholds = [-1.0, -0.35, 0.35, 1.0]
            if y_star <= thresholds[0]: return 1
            elif y_star <= thresholds[1]: return 2
            elif y_star <= thresholds[2]: return 3
            elif y_star <= thresholds[3]: return 4
            else: return 5

        # Generate patriotism items (1-20: blind 1-10, constructive 11-20)
        pat_const_true = np.random.normal(0, 1)
        patriotism_items = {}
        for item_num in range(1, 21):
            if item_num <= 10:
                lv = pat_blind_true
                loading = 0.85 - (item_num - 1) * 0.01
            else:
                lv = pat_const_true
                loading = 0.82 - (item_num - 11) * 0.01
            response = generate_likert_item(lv, loading)
            # Reverse items 5, 7, 12
            if item_num in [5, 7, 12]:
                response = 6 - response
            patriotism_items[f'patriotism_{item_num}'] = response

        # Generate secularism items (1-25: daily 1-15, faith 16-25)
        sec_fp_true = np.random.normal(0, 1)
        secularism_items = {}
        for item_num in range(1, 26):
            if item_num <= 15:
                lv = sec_dl_true
                loading = 0.80 - (item_num - 1) * 0.01
            else:
                lv = sec_fp_true
                loading = 0.78 - (item_num - 16) * 0.01
            response = generate_likert_item(lv, loading)
            # Reverse items 1, 13
            if item_num in [1, 13]:
                response = 6 - response
            secularism_items[f'secularism_{item_num}'] = response

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

            rec = {
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
                # True LV values for validation
                'pat_blind_true': pat_blind_true,
                'pat_constructive_true': pat_const_true,
                'sec_dl_true': sec_dl_true,
                'sec_fp_true': sec_fp_true,
            }
            rec.update(patriotism_items)
            rec.update(secularism_items)
            rows.append(rec)

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
            thresholds = [-1.0, -0.35, 0.35, 1.0]
            if y_star <= thresholds[0]: return 1
            elif y_star <= thresholds[1]: return 2
            elif y_star <= thresholds[2]: return 3
            elif y_star <= thresholds[3]: return 4
            else: return 5

        # Generate patriotism items (1-20: blind 1-10, constructive 11-20)
        pat_const_true = np.random.normal(0, 1)
        patriotism_items = {}
        for item_num in range(1, 21):
            if item_num <= 10:
                lv = pat_blind_true
                loading = 0.85 - (item_num - 1) * 0.01
            else:
                lv = pat_const_true
                loading = 0.82 - (item_num - 11) * 0.01
            response = generate_likert(lv, loading)
            # Reverse items 5, 7, 12
            if item_num in [5, 7, 12]:
                response = 6 - response
            patriotism_items[f'patriotism_{item_num}'] = response

        # Generate secularism items (1-25: daily 1-15, faith 16-25)
        sec_fp_true = np.random.normal(0, 1)
        secularism_items = {}
        for item_num in range(1, 26):
            if item_num <= 15:
                lv = sec_dl_true
                loading = 0.80 - (item_num - 1) * 0.01
            else:
                lv = sec_fp_true
                loading = 0.78 - (item_num - 16) * 0.01
            response = generate_likert(lv, loading)
            # Reverse items 1, 13
            if item_num in [1, 13]:
                response = 6 - response
            secularism_items[f'secularism_{item_num}'] = response

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

            rec = {
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
                # True values for validation
                'pat_blind_true': pat_blind_true,
                'pat_constructive_true': pat_const_true,
                'sec_dl_true': sec_dl_true,
                'sec_fp_true': sec_fp_true,
                'B_FEE_individual': B_FEE_i,
            }
            rec.update(patriotism_items)
            rec.update(secularism_items)
            rows.append(rec)

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

        # Generate minimal patriotism items (first 5)
        patriotism_items = {f'patriotism_{j}': np.random.randint(1, 6) for j in range(1, 6)}
        # Generate minimal secularism items (first 5)
        secularism_items = {f'secularism_{j}': np.random.randint(1, 6) for j in range(1, 6)}

        for t in range(n_tasks):
            fee1 = np.random.uniform(10000, 100000)
            fee2 = np.random.uniform(10000, 100000)

            rec = {
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
            }
            rec.update(patriotism_items)
            rec.update(secularism_items)
            rows.append(rec)

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

    # Create LV proxies using new unified naming
    lv_items = {
        'pat_blind': [f'patriotism_{i}' for i in range(1, 11)],      # patriotism_1-10
        'pat_const': [f'patriotism_{i}' for i in range(11, 21)],     # patriotism_11-20
        'sec_dl': [f'secularism_{i}' for i in range(1, 16)],         # secularism_1-15
        'sec_fp': [f'secularism_{i}' for i in range(16, 26)],        # secularism_16-25
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
    """Construct definitions for ICLV tests with new unified naming."""
    return {
        'pat_blind': [f'patriotism_{i}' for i in range(1, 11)],
        'pat_constructive': [f'patriotism_{i}' for i in range(11, 21)],
        'sec_dl': [f'secularism_{i}' for i in range(1, 16)],
        'sec_fp': [f'secularism_{i}' for i in range(16, 26)],
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

    # Create LV proxies using new unified naming
    lv_items = {
        'pat_blind': [f'patriotism_{i}' for i in range(1, 11)],      # patriotism_1-10
        'pat_const': [f'patriotism_{i}' for i in range(11, 21)],     # patriotism_11-20
        'sec_dl': [f'secularism_{i}' for i in range(1, 16)],         # secularism_1-15
        'sec_fp': [f'secularism_{i}' for i in range(16, 26)],        # secularism_16-25
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
