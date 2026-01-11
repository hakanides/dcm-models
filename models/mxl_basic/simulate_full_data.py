#!/usr/bin/env python3
"""
Standalone Data Generator for MXL Basic Model
DGP: Random coefficient on fee (unobserved taste heterogeneity)

True Parameters:
    ASC_paid = 5.0
    B_FEE ~ N(mu=-0.08, sigma=0.03)  <- Random coefficient
    B_DUR = -0.08

Each individual draws their own B_FEE from N(mu, sigma^2) which is
constant across all their choice tasks (panel structure).

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
    mxl_coefficient_function,
)


def simulate(config_path: Path, output_path: Path, verbose: bool = True) -> pd.DataFrame:
    """
    Main simulation function for MXL.

    Each individual draws their random coefficients once, then uses them
    for all T choice tasks (panel structure).
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
    fee_scale = cfg['choice_model'].get('fee_scale', 10000.0)

    records = []
    individual_betas = []  # Store for verification

    for i in range(N):
        # Draw demographics
        agent_demographics = {}
        for demo_name, demo_spec in cfg.get('demographics', {}).items():
            agent_demographics[demo_name] = draw_categorical(rng, demo_spec)

        # Draw random coefficients using shared DGP function
        individual_coefs = mxl_coefficient_function(
            agent_demographics, {}, cfg, rng
        )

        # Store for verification
        individual_betas.append({
            'ID': i,
            **individual_coefs
        })

        # Simulate T choice tasks for this individual
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
            choice = simulate_choice(utilities, rng, alt_indices)

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

            records.append(record)

    # Create DataFrame
    df = pd.DataFrame(records)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    # Save individual betas for verification
    betas_df = pd.DataFrame(individual_betas)
    betas_path = output_path.parent / "individual_coefficients.csv"
    betas_df.to_csv(betas_path, index=False)

    if verbose:
        print(f"\nSaved {len(df)} observations to {output_path}")
        print(f"Choice distribution: {df['CHOICE'].value_counts().sort_index().to_dict()}")

        # Report on individual coefficient distribution
        random_coefs = cfg.get('random_coefficients', {})
        if 'B_FEE' in betas_df.columns and 'B_FEE' in random_coefs:
            true_mu = random_coefs['B_FEE']['mean']
            true_sigma = random_coefs['B_FEE']['std']
            sim_mu = betas_df['B_FEE'].mean()
            sim_sigma = betas_df['B_FEE'].std()
            print(f"\nRandom coefficient B_FEE:")
            print(f"  True: N({true_mu}, {true_sigma}^2)")
            print(f"  Simulated: mean={sim_mu:.4f}, std={sim_sigma:.4f}")

    return df


def main():
    """Entry point for standalone execution."""
    model_dir = Path(__file__).parent
    config_path = model_dir / "config.json"
    output_path = model_dir / "data" / "simulated_data.csv"

    print("=" * 60)
    print("MXL Basic - Data Simulation")
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
