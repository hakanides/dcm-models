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

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))


@pytest.fixture
def sample_choice_data():
    """Create minimal synthetic choice data for testing."""
    np.random.seed(42)
    n_individuals = 20
    n_tasks = 10

    rows = []
    for i in range(n_individuals):
        # Individual-level attributes
        age_idx = np.random.choice([0, 1, 2, 3, 4])
        edu_idx = np.random.choice([0, 1, 2, 3, 4, 5])
        income_indiv_idx = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7])
        income_house_idx = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7])
        marital_idx = np.random.choice([0, 1, 2])

        # Likert items (1-5 scale)
        pat_blind = np.random.randint(1, 6, 4)
        pat_const = np.random.randint(1, 6, 4)
        sec_dl = np.random.randint(1, 6, 4)
        sec_fp = np.random.randint(1, 6, 4)

        for t in range(n_tasks):
            # Choice task attributes
            fee1 = np.random.uniform(10000, 500000)
            fee2 = np.random.uniform(10000, 500000)
            fee3 = 0  # Free alternative
            dur1 = np.random.randint(1, 25)
            dur2 = np.random.randint(1, 25)
            dur3 = np.random.randint(1, 25)

            # Simple choice model (lower fee preferred)
            utils = [-fee1/100000 - dur1*0.1, -fee2/100000 - dur2*0.1, -dur3*0.1]
            probs = np.exp(utils) / np.sum(np.exp(utils))
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
                'pat_blind_1': pat_blind[0],
                'pat_blind_2': pat_blind[1],
                'pat_blind_3': pat_blind[2],
                'pat_blind_4': pat_blind[3],
                'pat_constructive_1': pat_const[0],
                'pat_constructive_2': pat_const[1],
                'pat_constructive_3': pat_const[2],
                'pat_constructive_4': pat_const[3],
                'sec_dl_1': sec_dl[0],
                'sec_dl_2': sec_dl[1],
                'sec_dl_3': sec_dl[2],
                'sec_dl_4': sec_dl[3],
                'sec_fp_1': sec_fp[0],
                'sec_fp_2': sec_fp[1],
                'sec_fp_3': sec_fp[2],
                'sec_fp_4': sec_fp[3],
            })

    return pd.DataFrame(rows)


@pytest.fixture
def real_data_path():
    """Path to real test data if it exists."""
    path = PROJECT_ROOT / 'data' / 'test_small_sample.csv'
    if path.exists():
        return path
    return None


@pytest.fixture
def model_config():
    """Load model configuration."""
    import json
    config_path = PROJECT_ROOT / 'model_config.json'
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return None
