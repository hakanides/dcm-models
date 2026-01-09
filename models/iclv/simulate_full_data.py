#!/usr/bin/env python3
"""
Standalone Data Generator for ICLV Model
DGP: Integrated Choice and Latent Variable (same structure as HCM, but for simultaneous estimation)

True Parameters:
    ASC_paid = 5.0
    B_FEE = -0.08, B_FEE_PatBlind = -0.10, B_FEE_SecDL = 0.05
    B_DUR = -0.08

Structural Model:
    pat_blind = gamma_age * (age_idx - 2) + N(0, 1)
    sec_dl = gamma_edu * (edu_idx - 3) + N(0, 1)

Measurement Model:
    3 Likert items per construct with specified loadings
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
from scipy import stats


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


def generate_latent_variable(rng: np.random.Generator, demographics: dict,
                              lv_config: dict) -> float:
    """Generate latent variable from structural model."""
    struct = lv_config['structural']

    # Compute systematic component
    lv = struct.get('intercept', 0.0)

    for demo_name, beta in struct.get('betas', {}).items():
        demo_val = demographics[demo_name]
        # Use center from the demographics config if not in structural
        center = 0
        if 'center' in struct:
            center = struct['center']
        lv += beta * (demo_val - center)

    # Add random component
    sigma = struct.get('sigma', 1.0)
    lv += rng.normal(0, sigma)

    return lv


def generate_likert_items(rng: np.random.Generator, lv_value: float,
                          measurement_config: dict) -> dict:
    """Generate Likert items from latent variable using ordered probit."""
    items = measurement_config['items']
    loadings = measurement_config['loadings']
    thresholds = measurement_config['thresholds']

    responses = {}
    for item_name, loading in zip(items, loadings):
        # Continuous latent response
        y_star = loading * lv_value + rng.normal(0, 1)

        # Convert to ordinal using thresholds
        response = 1
        for thresh in thresholds:
            if y_star > thresh:
                response += 1
        responses[item_name] = min(response, 5)

    return responses


def get_attribute_value(scenario: pd.Series, alt_idx: str, attribute: str, fee_scale: float) -> float:
    """Extract attribute value for an alternative from scenario."""
    col_name = f"{attribute}{alt_idx}"
    value = scenario[col_name]
    if attribute == 'fee':
        value = value / fee_scale
    return float(value)


def compute_utility(choice_cfg: dict, scenario: pd.Series, alt_idx: str, alt_info: dict,
                    latent_values: dict) -> float:
    """Compute utility with latent variable interactions."""
    V = 0.0
    fee_scale = choice_cfg.get('fee_scale', 10000.0)
    alt_name = alt_info['name']

    # Add ASC if applicable
    for term in choice_cfg.get('base_terms', []):
        if alt_name in term.get('apply_to', []):
            V += term['coef']

    # Add attribute effects with LV interactions
    for term in choice_cfg.get('attribute_terms', []):
        if alt_name not in term.get('apply_to', []):
            continue

        attr_value = get_attribute_value(scenario, alt_idx, term['attribute'], fee_scale)

        # Compute coefficient with interactions
        coef = term['base_coef']
        for inter in term.get('interactions', []):
            if inter.get('type') == 'latent':
                lv_name = inter['with']
                if lv_name in latent_values:
                    coef += inter['coef'] * latent_values[lv_name]

        V += coef * attr_value

    return V


def simulate(config_path: Path, output_path: Path, verbose: bool = True) -> pd.DataFrame:
    """Main simulation function for ICLV."""
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

    # Get demographic centering values
    demo_centers = {}
    for demo_name, demo_spec in cfg.get('demographics', {}).items():
        demo_centers[demo_name] = demo_spec.get('center', 0)

    records = []
    true_lv_values = []

    for i in range(N):
        # Draw demographics
        agent_demographics = {}
        for demo_name, demo_spec in cfg.get('demographics', {}).items():
            agent_demographics[demo_name] = draw_categorical(rng, demo_spec)

        # Generate latent variables with demographic centering
        latent_values = {}
        likert_responses = {}
        for lv_name, lv_config in latent_cfg.items():
            # Add centering info to structural config
            struct = lv_config['structural'].copy()
            for demo_name in struct.get('betas', {}).keys():
                if demo_name in demo_centers:
                    struct['center'] = demo_centers[demo_name]

            lv_config_with_center = lv_config.copy()
            lv_config_with_center['structural'] = struct

            # Generate with center from demographics config
            struct = lv_config['structural']
            lv = struct.get('intercept', 0.0)
            for demo_name, beta in struct.get('betas', {}).items():
                demo_val = agent_demographics[demo_name]
                center = demo_centers.get(demo_name, 0)
                lv += beta * (demo_val - center)
            sigma = struct.get('sigma', 1.0)
            lv += rng.normal(0, sigma)
            latent_values[lv_name] = lv

            # Generate Likert items
            items = generate_likert_items(rng, lv, lv_config['measurement'])
            likert_responses.update(items)

        # Store true LV for verification
        true_lv_values.append({'ID': i, **agent_demographics, **latent_values})

        # Simulate T choice tasks
        for t in range(T):
            scenario_idx = rng.integers(0, n_scenarios)
            scenario = scenarios.iloc[scenario_idx]

            utilities = []
            for alt_idx in alt_indices:
                alt_info = alternatives[alt_idx]
                u = compute_utility(cfg['choice_model'], scenario, alt_idx, alt_info, latent_values)
                utilities.append(u)

            utilities = np.array(utilities)

            # Add Gumbel errors to get total utilities (RUM: U = V + ε)
            gumbel_errors = draw_gumbel_errors(rng, len(utilities))
            total_utilities = utilities + gumbel_errors

            # Choose alternative with highest total utility
            choice_idx = np.argmax(total_utilities)
            choice = int(alt_indices[choice_idx])

            record = {
                'ID': i,
                'task': t,
                'CHOICE': choice,
                'scenario_id': scenario['scenario_id'],
                **agent_demographics,
                **likert_responses,
                **{col: scenario[col] for col in scenarios.columns}
            }

            fee_scale = cfg['choice_model']['fee_scale']
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

        print("\nTrue LV statistics:")
        for lv_name in latent_cfg.keys():
            if lv_name in lv_df.columns:
                print(f"  {lv_name}: mean={lv_df[lv_name].mean():.3f}, std={lv_df[lv_name].std():.3f}")

        # Report Likert summary
        print("\nLikert items per construct:")
        for lv_name, lv_config in latent_cfg.items():
            items = lv_config['measurement']['items']
            first_item = items[0]
            if first_item in df.columns:
                print(f"  {lv_name} ({len(items)} items): {items[0]} mean={df[first_item].mean():.2f}")

    return df


def main():
    """Entry point for standalone execution."""
    model_dir = Path(__file__).parent
    config_path = model_dir / "config.json"
    output_path = model_dir / "data" / "simulated_data.csv"

    print("=" * 60)
    print("ICLV - Data Simulation")
    print("=" * 60)
    print("Generating data with 2 latent variables for simultaneous estimation:")
    print("  - pat_blind (Blind Patriotism) ~ age")
    print("  - sec_dl (Daily Life Secularism) ~ education")
    print()

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
