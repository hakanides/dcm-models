"""
Scenario Generator for DCM Choice Experiments

Generates choice scenarios for military service duration/fee trade-offs.
Creates scenarios_prepared.csv with 3 alternatives:
    - Alternative 1 (Paid Option 1): variable duration and fee
    - Alternative 2 (Paid Option 2): variable duration and fee
    - Alternative 3 (Standard): 24 weeks, 0 fee

Design approach: Stratified-Factorial Design
    - Fee and duration are sampled INDEPENDENTLY within stratified cells
    - 4 quadrants: Q1 (short/high), Q2 (short/low), Q3 (long/high), Q4 (long/low)
    - Target: fee-duration correlation |r| < 0.25 for proper parameter identification

Key constraints applied:
    - Fee rounded to 1000 TL increments
    - Duration range: 1-23 weeks (paid options must be < standard)
    - Minimum fee: 10,000 TL for paid options
"""

import random
import csv
import math
import argparse
from pathlib import Path
from itertools import product


# Design space quadrants for stratified sampling
QUADRANTS = {
    'Q1': {'duration': 'short', 'fee': 'high'},  # Premium fast-track
    'Q2': {'duration': 'short', 'fee': 'low'},   # Budget fast-track
    'Q3': {'duration': 'long', 'fee': 'high'},   # Premium moderate reduction
    'Q4': {'duration': 'long', 'fee': 'low'},    # Budget moderate reduction
}

DEFAULT_DURATION_BOUNDS = {
    'short': (1, 11),
    'long': (12, 23),
}

DEFAULT_FEE_BOUNDS = {
    'low': (50000, 500000),
    'high': (500001, 2000000),
}


def generate_orthogonal_option(
    quadrant: str,
    standard_duration: int = 24,
    duration_bounds: dict = None,
    fee_bounds: dict = None,
) -> dict:
    """
    Generate a single paid option within specified quadrant.

    Fee and duration are sampled INDEPENDENTLY to break correlation.

    Args:
        quadrant: 'Q1', 'Q2', 'Q3', or 'Q4'
        standard_duration: Standard service duration
        duration_bounds: Dict with 'short' and 'long' tuples
        fee_bounds: Dict with 'low' and 'high' tuples

    Returns:
        Dict with 'duration' and 'fee' keys
    """
    if duration_bounds is None:
        duration_bounds = DEFAULT_DURATION_BOUNDS
    if fee_bounds is None:
        fee_bounds = DEFAULT_FEE_BOUNDS

    q_spec = QUADRANTS[quadrant]

    # Sample duration uniformly within quadrant's duration range
    dur_min, dur_max = duration_bounds[q_spec['duration']]
    duration = random.randint(dur_min, dur_max)

    # Sample fee uniformly within quadrant's fee range (INDEPENDENT of duration)
    fee_min, fee_max = fee_bounds[q_spec['fee']]
    fee = random.randint(fee_min, fee_max)

    # Round fee to nearest 1000 TL
    fee = round(fee / 1000) * 1000

    return {'duration': duration, 'fee': fee}


def generate_orthogonal_scenario(
    quadrant_pair: tuple,
    standard_duration: int = 24,
    duration_bounds: dict = None,
    fee_bounds: dict = None,
) -> dict:
    """
    Generate a scenario with two paid options from specified quadrants.

    Args:
        quadrant_pair: Tuple of (Q1-Q4, Q1-Q4) for paid1 and paid2
        standard_duration: Standard service duration
        duration_bounds: Dict with 'short' and 'long' tuples
        fee_bounds: Dict with 'low' and 'high' tuples

    Returns:
        Dictionary with scenario parameters
    """
    q1, q2 = quadrant_pair

    opt1 = generate_orthogonal_option(q1, standard_duration, duration_bounds, fee_bounds)
    opt2 = generate_orthogonal_option(q2, standard_duration, duration_bounds, fee_bounds)

    return {
        'standard_duration': standard_duration,
        'dur1': opt1['duration'],
        'dur2': opt2['duration'],
        'dur3': standard_duration,
        'fee1': opt1['fee'],
        'fee2': opt2['fee'],
        'fee3': 0,
        'exempt1': 1,
        'exempt2': 1,
        'exempt3': 0,
        'quadrant_pair': f"{q1}-{q2}",
    }


def validate_scenario(scenario: dict) -> list:
    """
    Validate a scenario for basic consistency.

    NOTE: Trade-off logic (shorter = more expensive) is NOT enforced
    to allow stratified-factorial design with independent fee/duration sampling.

    Args:
        scenario: Scenario dictionary

    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    standard_dur = scenario['dur3']

    # Check duration bounds
    if scenario['dur1'] >= standard_dur:
        errors.append(f"dur1 ({scenario['dur1']}) must be < standard ({standard_dur})")
    if scenario['dur2'] >= standard_dur:
        errors.append(f"dur2 ({scenario['dur2']}) must be < standard ({standard_dur})")
    if scenario['dur1'] < 1:
        errors.append(f"dur1 ({scenario['dur1']}) must be >= 1")
    if scenario['dur2'] < 1:
        errors.append(f"dur2 ({scenario['dur2']}) must be >= 1")

    # Check fee bounds
    if scenario['fee1'] < 10000:
        errors.append(f"fee1 ({scenario['fee1']}) must be >= 10000 TL")
    if scenario['fee2'] < 10000:
        errors.append(f"fee2 ({scenario['fee2']}) must be >= 10000 TL")

    # Check fee rounding
    if scenario['fee1'] % 1000 != 0:
        errors.append(f"fee1 not rounded to 1000: {scenario['fee1']}")
    if scenario['fee2'] % 1000 != 0:
        errors.append(f"fee2 not rounded to 1000: {scenario['fee2']}")

    # Check that paid options differ (avoid identical alternatives)
    if scenario['dur1'] == scenario['dur2'] and scenario['fee1'] == scenario['fee2']:
        errors.append("Paid options are identical (same duration and fee)")

    return errors


def check_dominance(scenario: dict) -> dict:
    """
    Check for dominance relationships between alternatives.

    Returns dict with 'dominated' flag and 'details' string.
    Dominance is allowed in stratified design but flagged for analysis.
    """
    dur1, fee1 = scenario['dur1'], scenario['fee1']
    dur2, fee2 = scenario['dur2'], scenario['fee2']
    dur3, fee3 = scenario['dur3'], scenario['fee3']

    dominated = []

    # Check if paid1 dominates paid2
    if dur1 <= dur2 and fee1 <= fee2 and (dur1 < dur2 or fee1 < fee2):
        dominated.append("paid1 dominates paid2")

    # Check if paid2 dominates paid1
    if dur2 <= dur1 and fee2 <= fee1 and (dur2 < dur1 or fee2 < fee1):
        dominated.append("paid2 dominates paid1")

    # Check if standard dominates (only possible if standard has lower/equal duration AND lower/equal fee)
    # Standard has fee3=0, so it always has lowest fee, but longest duration
    # So standard can never dominate paid options (longer duration is worse)

    # Check if a paid option dominates standard (shorter AND cheaper)
    # Since fee3=0, paid can't be cheaper, so no dominance possible

    return {
        'dominated': len(dominated) > 0,
        'details': '; '.join(dominated) if dominated else 'none',
    }


def generate_scenarios(
    num_scenarios: int = 1000,
    standard_duration: int = 24,
    duration_bounds: dict = None,
    fee_bounds: dict = None,
    seed: int = None,
    verbose: bool = True,
) -> list:
    """
    Generate scenarios using stratified-factorial design.

    Generates equal scenarios per quadrant-pair combination (16 combinations).
    This ensures fee and duration vary independently across the design space.

    Args:
        num_scenarios: Number of scenarios to generate
        standard_duration: Standard service duration in weeks
        duration_bounds: Dict with 'short' and 'long' tuples
        fee_bounds: Dict with 'low' and 'high' tuples
        seed: Random seed for reproducibility
        verbose: Print progress

    Returns:
        List of scenario dictionaries
    """
    if seed is not None:
        random.seed(seed)

    if duration_bounds is None:
        duration_bounds = DEFAULT_DURATION_BOUNDS
    if fee_bounds is None:
        fee_bounds = DEFAULT_FEE_BOUNDS

    # Generate all 16 quadrant pair combinations
    quadrant_names = ['Q1', 'Q2', 'Q3', 'Q4']
    all_pairs = list(product(quadrant_names, repeat=2))

    # Calculate scenarios per pair (distribute evenly)
    base_per_pair = num_scenarios // len(all_pairs)
    remainder = num_scenarios % len(all_pairs)

    if verbose:
        print(f"Generating {num_scenarios} scenarios using stratified-factorial design...")
        print(f"Standard duration: {standard_duration} weeks")
        print(f"Duration bounds: short={duration_bounds['short']}, long={duration_bounds['long']}")
        print(f"Fee bounds: low={fee_bounds['low']}, high={fee_bounds['high']}")
        print(f"Quadrant pairs: {len(all_pairs)} combinations, ~{base_per_pair} scenarios each")

    scenarios = []
    dominance_count = 0

    for pair_idx, pair in enumerate(all_pairs):
        # Distribute remainder evenly across first pairs
        n_for_pair = base_per_pair + (1 if pair_idx < remainder else 0)

        for _ in range(n_for_pair):
            # Generate scenario with retry for validation errors
            for attempt in range(100):
                scenario = generate_orthogonal_scenario(
                    quadrant_pair=pair,
                    standard_duration=standard_duration,
                    duration_bounds=duration_bounds,
                    fee_bounds=fee_bounds,
                )
                errors = validate_scenario(scenario)
                if not errors:
                    break
            else:
                # Fallback if can't generate valid scenario
                if verbose:
                    print(f"  Warning: Using fallback for pair {pair}")
                scenario = {
                    'dur1': 6 if pair[0] in ['Q1', 'Q2'] else 18,
                    'dur2': 6 if pair[1] in ['Q1', 'Q2'] else 18,
                    'dur3': standard_duration,
                    'fee1': 1000000 if pair[0] in ['Q1', 'Q3'] else 200000,
                    'fee2': 1000000 if pair[1] in ['Q1', 'Q3'] else 200000,
                    'fee3': 0,
                    'exempt1': 1,
                    'exempt2': 1,
                    'exempt3': 0,
                    'quadrant_pair': f"{pair[0]}-{pair[1]}",
                    'standard_duration': standard_duration,
                }

            scenario['scenario_id'] = len(scenarios) + 1

            # Check dominance (for analysis, not rejection)
            dom = check_dominance(scenario)
            scenario['dominance'] = dom['details']
            if dom['dominated']:
                dominance_count += 1

            scenarios.append(scenario)

        if verbose and (pair_idx + 1) % 4 == 0:
            print(f"  Generated scenarios for {pair_idx + 1}/{len(all_pairs)} quadrant pairs...")

    if verbose:
        print(f"\nGeneration complete:")
        print(f"  Total scenarios: {len(scenarios)}")
        print(f"  Scenarios with dominance: {dominance_count} ({100*dominance_count/len(scenarios):.1f}%)")

    return scenarios


def save_scenarios_csv(scenarios: list, filepath: str) -> None:
    """
    Save scenarios to CSV in the expected format.

    Args:
        scenarios: List of scenario dictionaries
        filepath: Output file path
    """
    fieldnames = [
        'scenario_id',
        'dur1', 'dur2', 'dur3',
        'fee1', 'fee2', 'fee3',
        'exempt1', 'exempt2', 'exempt3',
    ]

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for scenario in scenarios:
            row = {k: scenario[k] for k in fieldnames}
            writer.writerow(row)

    print(f"Saved {len(scenarios)} scenarios to {filepath}")


def save_scenarios_with_analysis(scenarios: list, filepath: str) -> None:
    """
    Save scenarios with additional analysis columns.

    Args:
        scenarios: List of scenario dictionaries
        filepath: Output file path
    """
    fieldnames = [
        'scenario_id',
        'dur1', 'dur2', 'dur3',
        'fee1', 'fee2', 'fee3',
        'exempt1', 'exempt2', 'exempt3',
        'tradeoff_type',
        'duration_ratio',
        'price_ratio',
        'ratio_difference',
    ]

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for scenario in scenarios:
            # Compute analysis fields
            if scenario['dur1'] < scenario['dur2']:
                tradeoff_type = 'Paid1_shorter_expensive'
            elif scenario['dur1'] > scenario['dur2']:
                tradeoff_type = 'Paid1_longer_cheaper'
            else:
                tradeoff_type = 'Equal_duration'

            dur_ratio = scenario['dur1'] / scenario['dur2'] if scenario['dur2'] > 0 else float('inf')
            price_ratio = scenario['fee1'] / scenario['fee2'] if scenario['fee2'] > 0 else float('inf')
            ratio_diff = abs(dur_ratio - price_ratio)

            row = {
                'scenario_id': scenario['scenario_id'],
                'dur1': scenario['dur1'],
                'dur2': scenario['dur2'],
                'dur3': scenario['dur3'],
                'fee1': scenario['fee1'],
                'fee2': scenario['fee2'],
                'fee3': scenario['fee3'],
                'exempt1': scenario['exempt1'],
                'exempt2': scenario['exempt2'],
                'exempt3': scenario['exempt3'],
                'tradeoff_type': tradeoff_type,
                'duration_ratio': round(dur_ratio, 3),
                'price_ratio': round(price_ratio, 3),
                'ratio_difference': round(ratio_diff, 3),
            }
            writer.writerow(row)

    print(f"Saved {len(scenarios)} scenarios with analysis to {filepath}")


def compute_correlation(x: list, y: list) -> float:
    """Compute Pearson correlation coefficient."""
    n = len(x)
    if n != len(y) or n < 2:
        return float('nan')

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / n
    std_x = (sum((xi - mean_x) ** 2 for xi in x) / n) ** 0.5
    std_y = (sum((yi - mean_y) ** 2 for yi in y) / n) ** 0.5

    if std_x == 0 or std_y == 0:
        return float('nan')

    return cov / (std_x * std_y)


def print_summary(scenarios: list) -> None:
    """Print summary statistics for generated scenarios."""
    print("\n" + "=" * 60)
    print("SCENARIO GENERATION SUMMARY")
    print("=" * 60)

    print(f"\nTotal scenarios: {len(scenarios)}")

    # Duration statistics
    dur1_values = [s['dur1'] for s in scenarios]
    dur2_values = [s['dur2'] for s in scenarios]
    print(f"\nDuration statistics:")
    print(f"  dur1 range: {min(dur1_values)} - {max(dur1_values)} weeks")
    print(f"  dur1 mean: {sum(dur1_values)/len(dur1_values):.1f} weeks")
    print(f"  dur2 range: {min(dur2_values)} - {max(dur2_values)} weeks")
    print(f"  dur2 mean: {sum(dur2_values)/len(dur2_values):.1f} weeks")

    # Fee statistics
    fee1_values = [s['fee1'] for s in scenarios]
    fee2_values = [s['fee2'] for s in scenarios]
    print(f"\nFee statistics:")
    print(f"  fee1 range: {min(fee1_values):,} - {max(fee1_values):,} TL")
    print(f"  fee1 mean: {sum(fee1_values)/len(fee1_values):,.0f} TL")
    print(f"  fee2 range: {min(fee2_values):,} - {max(fee2_values):,} TL")
    print(f"  fee2 mean: {sum(fee2_values)/len(fee2_values):,.0f} TL")

    # === CORRELATION DIAGNOSTICS (critical for MNL identification) ===
    print(f"\n{'='*60}")
    print("CORRELATION DIAGNOSTICS (target: |r| < 0.25)")
    print("="*60)

    # Correlation between duration and fee for each paid option
    corr_dur1_fee1 = compute_correlation(dur1_values, fee1_values)
    corr_dur2_fee2 = compute_correlation(dur2_values, fee2_values)

    # Pooled correlation (all duration-fee pairs)
    all_dur = dur1_values + dur2_values
    all_fee = fee1_values + fee2_values
    corr_pooled = compute_correlation(all_dur, all_fee)

    print(f"\n  Paid Option 1 (dur1 vs fee1): r = {corr_dur1_fee1:+.3f}", end="")
    print(f"  {'PASS' if abs(corr_dur1_fee1) < 0.25 else 'WARNING: HIGH CORRELATION'}")

    print(f"  Paid Option 2 (dur2 vs fee2): r = {corr_dur2_fee2:+.3f}", end="")
    print(f"  {'PASS' if abs(corr_dur2_fee2) < 0.25 else 'WARNING: HIGH CORRELATION'}")

    print(f"  Pooled (all dur vs fee):      r = {corr_pooled:+.3f}", end="")
    print(f"  {'PASS' if abs(corr_pooled) < 0.25 else 'WARNING: HIGH CORRELATION'}")

    if abs(corr_pooled) >= 0.25:
        print("\n  ** WARNING: High correlation may cause multicollinearity in MNL **")
        print("  ** Consider regenerating with different parameters **")

    # Quadrant distribution
    if 'quadrant_pair' in scenarios[0]:
        print(f"\nQuadrant pair distribution:")
        quadrant_counts = {}
        for s in scenarios:
            qp = s.get('quadrant_pair', 'unknown')
            quadrant_counts[qp] = quadrant_counts.get(qp, 0) + 1
        for qp in sorted(quadrant_counts.keys()):
            print(f"  {qp}: {quadrant_counts[qp]} ({100*quadrant_counts[qp]/len(scenarios):.1f}%)")

    # Dominance statistics
    if 'dominance' in scenarios[0]:
        dom_count = sum(1 for s in scenarios if s.get('dominance', 'none') != 'none')
        print(f"\nDominance analysis:")
        print(f"  Scenarios with dominance: {dom_count} ({100*dom_count/len(scenarios):.1f}%)")

    # Duration category distribution
    print(f"\nDuration category distribution:")
    short_dur1 = sum(1 for s in scenarios if s['dur1'] <= 11)
    long_dur1 = sum(1 for s in scenarios if s['dur1'] > 11)
    short_dur2 = sum(1 for s in scenarios if s['dur2'] <= 11)
    long_dur2 = sum(1 for s in scenarios if s['dur2'] > 11)
    print(f"  dur1: short={short_dur1} ({100*short_dur1/len(scenarios):.1f}%), long={long_dur1} ({100*long_dur1/len(scenarios):.1f}%)")
    print(f"  dur2: short={short_dur2} ({100*short_dur2/len(scenarios):.1f}%), long={long_dur2} ({100*long_dur2/len(scenarios):.1f}%)")

    # Fee category distribution
    print(f"\nFee category distribution:")
    low_fee1 = sum(1 for s in scenarios if s['fee1'] <= 500000)
    high_fee1 = sum(1 for s in scenarios if s['fee1'] > 500000)
    low_fee2 = sum(1 for s in scenarios if s['fee2'] <= 500000)
    high_fee2 = sum(1 for s in scenarios if s['fee2'] > 500000)
    print(f"  fee1: low={low_fee1} ({100*low_fee1/len(scenarios):.1f}%), high={high_fee1} ({100*high_fee1/len(scenarios):.1f}%)")
    print(f"  fee2: low={low_fee2} ({100*low_fee2/len(scenarios):.1f}%), high={high_fee2} ({100*high_fee2/len(scenarios):.1f}%)")

    # Sample scenarios
    print(f"\nSample scenarios (first 5):")
    print("-" * 90)
    print(f"{'ID':<4} {'dur1':<5} {'dur2':<5} {'dur3':<5} {'fee1':<12} {'fee2':<12} {'quadrant':<10}")
    print("-" * 90)
    for s in scenarios[:5]:
        qp = s.get('quadrant_pair', 'N/A')
        print(f"{s['scenario_id']:<4} {s['dur1']:<5} {s['dur2']:<5} {s['dur3']:<5} "
              f"{s['fee1']:<12,} {s['fee2']:<12,} {qp:<10}")
    print("-" * 90)


def main():
    parser = argparse.ArgumentParser(
        description='Generate DCM choice experiment scenarios using stratified-factorial design'
    )
    parser.add_argument(
        '--n', type=int, default=1000,
        help='Number of scenarios to generate (default: 1000)'
    )
    parser.add_argument(
        '--standard-duration', type=int, default=24,
        help='Standard service duration in weeks (default: 24)'
    )
    parser.add_argument(
        '--output', type=str, default='data/raw/scenarios_prepared.csv',
        help='Output file path (default: data/raw/scenarios_prepared.csv)'
    )
    parser.add_argument(
        '--output-analysis', type=str, default=None,
        help='Output file with analysis columns (optional)'
    )
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Suppress progress output'
    )

    args = parser.parse_args()

    # Generate scenarios using stratified-factorial design
    scenarios = generate_scenarios(
        num_scenarios=args.n,
        standard_duration=args.standard_duration,
        seed=args.seed,
        verbose=not args.quiet,
    )

    # Print summary
    if not args.quiet:
        print_summary(scenarios)

    # Resolve output path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        # Make relative to project root
        project_root = Path(__file__).parent.parent
        output_path = project_root / args.output

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save main output
    save_scenarios_csv(scenarios, str(output_path))

    # Save analysis output if requested
    if args.output_analysis:
        analysis_path = Path(args.output_analysis)
        if not analysis_path.is_absolute():
            project_root = Path(__file__).parent.parent
            analysis_path = project_root / args.output_analysis
        analysis_path.parent.mkdir(parents=True, exist_ok=True)
        save_scenarios_with_analysis(scenarios, str(analysis_path))

    print("\nDone!")


if __name__ == '__main__':
    main()
