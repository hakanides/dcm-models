#!/usr/bin/env python3
"""
Standalone Data Generator for HCM Basic Model
DGP: Single latent variable (Blind Patriotism) interaction on fee sensitivity

True Parameters:
    ASC_paid = 5.0
    B_FEE = -0.08, B_FEE_PatBlind = -0.10
    B_DUR = -0.08

Latent Variable Structure:
    pat_blind = 0 + 0.20 * (age_idx - 2) + N(0,1)

Measurement Model:
    5 Likert items with loadings 0.85, 0.83, 0.81, 0.79, 0.78
    Ordered probit with thresholds: [-1.0, -0.35, 0.35, 1.0]

Note on Fixed Thresholds:
    This DGP uses FIXED thresholds for the ordered probit measurement model.
    This is intentional for validation studies where we want a known measurement
    structure to verify parameter recovery. For real-world applications with
    actual survey data, thresholds should be ESTIMATED from the data.
    See Train (2009) and Hensher et al. (2015) for discussion.

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
    hcm_coefficient_function,
    generate_all_latent_hcm,
)


def simulate(config_path: Path, output_path: Path, verbose: bool = True) -> pd.DataFrame:
    """Main simulation function for HCM."""
    with open(config_path) as f:
        cfg = json.load(f)

    pop_cfg = cfg['population']
    N = pop_cfg['N']
    T = pop_cfg['T']
    seed = pop_cfg.get('seed', 42)

    rng = np.random.default_rng(seed)

    scenarios_path = Path(config_path).parent / cfg['design']['path']
    scenarios = pd.read_csv(scenarios_path)
    n_scenarios = len(scenarios)

    if verbose:
        print(f"Loaded {n_scenarios} scenarios from {scenarios_path}")
        print(f"Simulating {N} individuals x {T} tasks = {N*T} observations")
        print(f"True parameters: {cfg['model_info']['true_values']}")

    alternatives = cfg['choice_model']['alternatives']
    alt_indices = list(alternatives.keys())
    latent_cfg = cfg.get('latent', {})
    demo_specs = cfg.get('demographics', {})
    fee_scale = cfg['choice_model'].get('fee_scale', 10000.0)

    records = []
    true_lv_values = []

    for i in range(N):
        # Draw demographics
        agent_demographics = {}
        for demo_name, demo_spec in demo_specs.items():
            agent_demographics[demo_name] = draw_categorical(rng, demo_spec)

        # Generate latent variables and Likert items using shared DGP
        latent_values, likert_responses = generate_all_latent_hcm(
            rng, agent_demographics, latent_cfg, demo_specs
        )

        # Store true LV for verification
        true_lv_values.append({'ID': i, **latent_values})

        # Get individual coefficients with latent variable effects
        individual_coefs = hcm_coefficient_function(
            agent_demographics, latent_values, cfg, rng
        )

        # Simulate T choice tasks
        for t in range(T):
            scenario_idx = rng.integers(0, n_scenarios)
            scenario = scenarios.iloc[scenario_idx]

            # Compute utilities using shared DGP
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
            choice = simulate_choice(utilities, rng, alt_indices)

            record = {
                'ID': i,
                'task': t,
                'CHOICE': choice,
                'scenario_id': scenario['scenario_id'],
                **agent_demographics,
                **likert_responses,
                **{col: scenario[col] for col in scenarios.columns}
            }

            # Add scaled fee columns
            for alt_idx in alt_indices:
                record[f'fee{alt_idx}_10k'] = scenario[f'fee{alt_idx}'] / fee_scale

            records.append(record)

    df = pd.DataFrame(records)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    # Save true LV values for verification
    lv_df = pd.DataFrame(true_lv_values)
    lv_path = output_path.parent / "true_latent_values.csv"
    lv_df.to_csv(lv_path, index=False)

    if verbose:
        print(f"\nSaved {len(df)} observations to {output_path}")
        print(f"Choice distribution: {df['CHOICE'].value_counts().sort_index().to_dict()}")
        print(f"\nTrue LV (pat_blind): mean={lv_df['pat_blind'].mean():.3f}, std={lv_df['pat_blind'].std():.3f}")

        # Report Likert item statistics
        items = list(latent_cfg.get('pat_blind', {}).get('measurement', {}).get('items', []))
        if items:
            print(f"\nLikert items ({items[0]}-{items[-1]}):")
            for item in items[:3]:
                print(f"  {item}: mean={df[item].mean():.2f}, std={df[item].std():.2f}")

    return df


def main():
    """Entry point for standalone execution."""
    model_dir = Path(__file__).parent
    config_path = model_dir / "config.json"
    output_path = model_dir / "data" / "simulated_data.csv"

    print("=" * 60)
    print("HCM Basic - Data Simulation")
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
