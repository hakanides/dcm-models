"""
Prepare scenarios file for DCM simulation.

Transforms the raw scenarios CSV to the format expected by the simulator:
- Renames columns to standard format (dur1, fee1, etc.)
"""

import pandas as pd
import numpy as np
from pathlib import Path


def prepare_scenarios(
    input_path: str,
    output_path: str,
    seed: int = 42
) -> pd.DataFrame:
    """
    Transform raw scenarios to simulator format.

    Args:
        input_path: Path to raw scenarios CSV
        output_path: Path for output CSV
        seed: Random seed for reproducibility

    Returns:
        Prepared DataFrame
    """
    # Load raw scenarios
    raw = pd.read_csv(input_path)
    print(f"Loaded {len(raw)} scenarios from {input_path}")

    # Create output DataFrame
    df = pd.DataFrame()

    # Map columns
    df['scenario_id'] = raw['scenario_id']

    # Duration columns
    df['dur1'] = raw['paid1_duration']
    df['dur2'] = raw['paid2_duration']
    df['dur3'] = raw['standard_duration']

    # Fee columns (prices)
    df['fee1'] = raw['paid1_price']
    df['fee2'] = raw['paid2_price']
    df['fee3'] = 0  # Standard alternative has no fee

    # Save
    df.to_csv(output_path, index=False)
    print(f"Saved prepared scenarios to {output_path}")

    # Print summary
    print(f"\nSummary:")
    print(f"  Scenarios: {len(df)}")
    print(f"  Duration range (paid1): {df['dur1'].min()} - {df['dur1'].max()} weeks")
    print(f"  Duration range (paid2): {df['dur2'].min()} - {df['dur2'].max()} weeks")
    print(f"  Duration (standard): {df['dur3'].unique()[0]} weeks")
    print(f"  Fee range (paid1): {df['fee1'].min():,.0f} - {df['fee1'].max():,.0f} TL")
    print(f"  Fee range (paid2): {df['fee2'].min():,.0f} - {df['fee2'].max():,.0f} TL")

    return df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Prepare scenarios for DCM simulation')
    parser.add_argument('--input', required=True, help='Input CSV path')
    parser.add_argument('--output', required=True, help='Output CSV path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()
    prepare_scenarios(args.input, args.output, args.seed)
