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


def softmax(utilities: np.ndarray) -> np.ndarray:
    """Compute choice probabilities using softmax."""
    u = utilities - np.max(utilities)
    exp_u = np.exp(u)
    return exp_u / exp_u.sum()


def draw_gumbel_errors(rng: np.random.Generator, n: int) -> np.ndarray:
    """
    Draw Gumbel(0,1) errors for random utility model.

    The Gumbel distribution: ε = -ln(-ln(U)) where U ~ Uniform(0,1).
    This makes the RUM foundation explicit: U_ij = V_ij + ε_ij
    """
    u = rng.uniform(1e-10, 1 - 1e-10, size=n)
    return -np.log(-np.log(u))


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
                    individual_coefficients: dict) -> float:
    """
    Compute utility for one alternative using individual-specific coefficients.

    For MXL, B_FEE is drawn per-individual from N(mu, sigma^2).
    """
    V = 0.0
    fee_scale = choice_cfg.get('fee_scale', 10000.0)
    alt_name = alt_info['name']

    # Add ASC if applicable
    for term in choice_cfg.get('base_terms', []):
        if alt_name in term.get('apply_to', []):
            V += term['coef']

    # Add attribute effects
    for term in choice_cfg.get('attribute_terms', []):
        if alt_name not in term.get('apply_to', []):
            continue

        # Get attribute value
        attr_value = get_attribute_value(scenario, alt_idx, term['attribute'], fee_scale)

        # Use individual-specific coefficient if random, otherwise base
        coef_name = term['name']
        if term.get('random', False) and coef_name in individual_coefficients:
            coef = individual_coefficients[coef_name]
        else:
            coef = term['base_coef']

        V += coef * attr_value

    return V


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

    # Get random coefficient specs
    random_coefs = cfg.get('random_coefficients', {})

    # Get configuration
    alternatives = cfg['choice_model']['alternatives']
    alt_indices = list(alternatives.keys())

    records = []
    individual_betas = []  # Store for verification

    for i in range(N):
        # Draw demographics
        agent_demographics = {}
        for demo_name, demo_spec in cfg.get('demographics', {}).items():
            agent_demographics[demo_name] = draw_categorical(rng, demo_spec)

        # Draw random coefficients for this individual
        individual_coefficients = {}
        for coef_name, coef_spec in random_coefs.items():
            if coef_spec['distribution'] == 'normal':
                mu = coef_spec['mean']
                sigma = coef_spec['std']
                individual_coefficients[coef_name] = rng.normal(mu, sigma)
            # Could add other distributions here (lognormal, etc.)

        # Store for verification
        individual_betas.append({
            'ID': i,
            **individual_coefficients
        })

        # Simulate T choice tasks for this individual
        for t in range(T):
            # Sample a scenario
            scenario_idx = rng.integers(0, n_scenarios)
            scenario = scenarios.iloc[scenario_idx]

            # Compute utilities for each alternative
            utilities = []
            for alt_idx in alt_indices:
                alt_info = alternatives[alt_idx]
                u = compute_utility(cfg['choice_model'], scenario, alt_idx, alt_info,
                                    individual_coefficients)
                utilities.append(u)

            utilities = np.array(utilities)

            # Add Gumbel errors to get total utilities (RUM: U = V + ε)
            gumbel_errors = draw_gumbel_errors(rng, len(utilities))
            total_utilities = utilities + gumbel_errors

            # Choose alternative with highest total utility
            choice_idx = np.argmax(total_utilities)
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
        if 'B_FEE' in betas_df.columns:
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
