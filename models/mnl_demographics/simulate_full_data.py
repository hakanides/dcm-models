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

DESIGN NOTE - Asymmetric Interactions:
    Fee sensitivity: affected by age, education, and income
    Duration sensitivity: affected by education and income ONLY (no age effect)

    Rationale: Age affects fee sensitivity (older individuals are less price-sensitive,
    possibly due to accumulated wealth). However, duration/time sensitivity is driven
    by opportunity cost, which correlates with education and income rather than age.

Uses shared DGP library to eliminate code duplication.
"""

import json
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add shared utilities to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.sample_stats import generate_sample_stats
from shared.cleanup import cleanup_simulation_outputs

# Import shared DGP functions
from shared.dgp import (
    draw_categorical,
    compute_base_utility,
    simulate_choice,
    mnl_demographics_coefficient_function,
)


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
    fee_scale = cfg['choice_model'].get('fee_scale', 10000.0)

    records = []

    for i in range(N):
        # Draw demographics
        agent_demographics = {}
        for demo_name, demo_spec in demo_specs.items():
            agent_demographics[demo_name] = draw_categorical(rng, demo_spec)

        # Get individual coefficients with demographic interactions
        individual_coefs = mnl_demographics_coefficient_function(
            agent_demographics, {}, cfg, rng
        )

        # Simulate T choice tasks
        for t in range(T):
            # Sample a scenario
            scenario_idx = rng.integers(0, n_scenarios)
            scenario = scenarios.iloc[scenario_idx]

            # Compute utilities using shared DGP with individual coefficients
            utilities = []
            for alt_idx in alt_indices:
                alt_info = alternatives[alt_idx]
                u = compute_base_utility(
                    cfg['choice_model'], scenario, alt_idx, alt_info,
                    individual_coefficients=individual_coefs
                )
                utilities.append(u)

            utilities = np.array(utilities)

            # Simulate choice using RUM (V + Gumbel error)
            choice = simulate_choice(utilities, alt_indices, rng)

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

    # Clean previous simulation outputs
    cleanup_simulation_outputs(model_dir)

    # Run simulation
    df = simulate(config_path, output_path)

    # Generate sample statistics
    with open(config_path) as f:
        config = json.load(f)
    stats_dir = model_dir / "sample_stats"
    generate_sample_stats(df, config, stats_dir)

    print("\n" + "=" * 60)
    print("Simulation complete!")
    print("=" * 60)

    return df


if __name__ == "__main__":
    main()
