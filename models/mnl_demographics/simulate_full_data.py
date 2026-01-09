#!/usr/bin/env python3
"""
Standalone Data Generator for MNL Demographics Model
DGP: Demographic interactions on fee and duration sensitivity

True Parameters:
    ASC_paid = 5.0
    B_FEE = -0.08, B_FEE_AGE = 0.06, B_FEE_EDU = 0.08, B_FEE_INC = 0.12
    B_DUR = -0.08, B_DUR_EDU = -0.04, B_DUR_INC = -0.03

Model:
    B_FEE_i = B_FEE + B_FEE_AGE*age_c + B_FEE_EDU*edu_c + B_FEE_INC*inc_c
    B_DUR_i = B_DUR + B_DUR_EDU*edu_c + B_DUR_INC*inc_c
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path


def softmax(utilities: np.ndarray) -> np.ndarray:
    """Compute choice probabilities using softmax."""
    u = utilities - np.max(utilities)
    exp_u = np.exp(u)
    return exp_u / exp_u.sum()


def draw_categorical(rng: np.random.Generator, spec: dict) -> int:
    """Sample from a categorical distribution."""
    return rng.choice(spec['values'], p=spec['probs'])


def get_attribute_value(scenario: pd.Series, alt_idx: str, attribute: str, fee_scale: float) -> float:
    """Extract attribute value for an alternative from scenario."""
    col_name = f"{attribute}{alt_idx}"
    value = scenario[col_name]
    if attribute == 'fee':
        value = value / fee_scale
    return float(value)


def compute_utility(choice_cfg: dict, scenario: pd.Series, alt_idx: str, alt_info: dict,
                    demographics: dict, demo_specs: dict) -> float:
    """
    Compute utility for one alternative with demographic interactions.

    Demographics are centered and scaled according to config before applying interactions.
    """
    V = 0.0
    fee_scale = choice_cfg.get('fee_scale', 10000.0)
    alt_name = alt_info['name']

    # Add ASC if applicable
    for term in choice_cfg.get('base_terms', []):
        if alt_name in term.get('apply_to', []):
            V += term['coef']

    # Add attribute effects with demographic interactions
    for term in choice_cfg.get('attribute_terms', []):
        if alt_name not in term.get('apply_to', []):
            continue

        # Get attribute value
        attr_value = get_attribute_value(scenario, alt_idx, term['attribute'], fee_scale)

        # Compute coefficient with demographic interactions
        coef = term['base_coef']

        for inter in term.get('interactions', []):
            demo_name = inter['with']
            demo_raw = demographics[demo_name]

            # Center and scale the demographic
            demo_spec = demo_specs[demo_name]
            center = demo_spec.get('center', 0)
            scale = demo_spec.get('scale', 1)
            demo_centered = (demo_raw - center) / scale

            coef += inter['coef'] * demo_centered

        V += coef * attr_value

    return V


def simulate(config_path: Path, output_path: Path, verbose: bool = True) -> pd.DataFrame:
    """
    Main simulation function.

    Generates choice data with demographic interactions on taste parameters.
    """
    # Load configuration
    with open(config_path) as f:
        cfg = json.load(f)

    # Setup
    pop_cfg = cfg['population']
    N = pop_cfg['N']
    T = pop_cfg['T']
    seed = pop_cfg.get('seed', 42)

    rng = np.random.default_rng(seed)

    # Load scenarios
    scenarios_path = Path(config_path).parent / cfg['design']['path']
    scenarios = pd.read_csv(scenarios_path)
    n_scenarios = len(scenarios)

    if verbose:
        print(f"Loaded {n_scenarios} scenarios from {scenarios_path}")
        print(f"Simulating {N} individuals x {T} tasks = {N*T} observations")
        print(f"True parameters: {cfg['model_info']['true_values']}")

    # Get configuration
    alternatives = cfg['choice_model']['alternatives']
    alt_indices = list(alternatives.keys())
    demo_specs = cfg['demographics']

    records = []

    for i in range(N):
        # Draw demographics
        agent_demographics = {}
        for demo_name, demo_spec in demo_specs.items():
            agent_demographics[demo_name] = draw_categorical(rng, demo_spec)

        # Simulate T choice tasks
        for t in range(T):
            # Sample a scenario
            scenario_idx = rng.integers(0, n_scenarios)
            scenario = scenarios.iloc[scenario_idx]

            # Compute utilities for each alternative
            utilities = []
            for alt_idx in alt_indices:
                alt_info = alternatives[alt_idx]
                u = compute_utility(cfg['choice_model'], scenario, alt_idx, alt_info,
                                    agent_demographics, demo_specs)
                utilities.append(u)

            utilities = np.array(utilities)

            # Compute choice probabilities and draw choice
            probs = softmax(utilities)
            choice_idx = rng.choice(len(alt_indices), p=probs)
            choice = int(alt_indices[choice_idx])

            # Build record
            record = {
                'ID': i,
                'task': t,
                'CHOICE': choice,
                'scenario_id': scenario['scenario_id'],
                **agent_demographics,
                **{col: scenario[col] for col in scenarios.columns}
            }

            # Add scaled fee columns
            fee_scale = cfg['choice_model']['fee_scale']
            for alt_idx in alt_indices:
                record[f'fee{alt_idx}_10k'] = scenario[f'fee{alt_idx}'] / fee_scale

            # Add centered demographic columns for estimation
            for demo_name, demo_spec in demo_specs.items():
                center = demo_spec.get('center', 0)
                scale = demo_spec.get('scale', 1)
                record[f'{demo_name}_c'] = (agent_demographics[demo_name] - center) / scale

            records.append(record)

    # Create DataFrame
    df = pd.DataFrame(records)

    # Rename centered columns for estimation compatibility
    df['age_c'] = df['age_idx_c']
    df['edu_c'] = df['edu_idx_c']
    df['inc_c'] = df['income_indiv_idx_c']

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    if verbose:
        print(f"\nSaved {len(df)} observations to {output_path}")
        print(f"Choice distribution: {df['CHOICE'].value_counts().sort_index().to_dict()}")

    return df


def main():
    """Entry point for standalone execution."""
    model_dir = Path(__file__).parent
    config_path = model_dir / "config.json"
    output_path = model_dir / "data" / "simulated_data.csv"

    print("=" * 60)
    print("MNL Demographics - Data Simulation")
    print("=" * 60)

    df = simulate(config_path, output_path)

    print("\n" + "=" * 60)
    print("Simulation complete!")
    print("=" * 60)

    return df


if __name__ == "__main__":
    main()
