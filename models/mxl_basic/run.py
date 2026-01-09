#!/usr/bin/env python3
"""
MXL Basic - Isolated Model Validation
=====================================

This script runs the complete validation pipeline for MXL Basic:
1. Generate simulated data with random fee coefficient
2. Estimate the model using simulated maximum likelihood
3. Compare estimates to true parameters

Expected result: Unbiased parameter recovery (<10% bias, 95% CI coverage)

Usage:
    python run.py
    python run.py --skip_simulation
    python run.py --draws 1000
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Run MXL Basic validation')
    parser.add_argument('--skip_simulation', action='store_true',
                        help='Skip simulation, use existing data')
    parser.add_argument('--draws', type=int, default=500,
                        help='Number of simulation draws (default: 500)')

    args = parser.parse_args()

    model_dir = Path(__file__).parent

    print("=" * 70)
    print("MXL BASIC - ISOLATED MODEL VALIDATION")
    print("=" * 70)
    print("\nThis validates that MXL Basic can recover true parameters")
    print("when the DGP includes random coefficient heterogeneity.\n")

    # Step 1: Generate data
    if not args.skip_simulation:
        print("-" * 70)
        print("STEP 1: Generating simulated data")
        print("-" * 70)
        result = subprocess.run(
            [sys.executable, str(model_dir / "simulate_full_data.py")],
            cwd=str(model_dir),
            capture_output=False
        )
        if result.returncode != 0:
            print("ERROR: Simulation failed!")
            sys.exit(1)
    else:
        print("-" * 70)
        print("STEP 1: Skipped (using existing data)")
        print("-" * 70)

    # Step 2: Estimate model
    print("\n" + "-" * 70)
    print("STEP 2: Estimating model")
    print("-" * 70)
    result = subprocess.run(
        [sys.executable, str(model_dir / "model.py")],
        cwd=str(model_dir),
        capture_output=False
    )
    if result.returncode != 0:
        print("ERROR: Estimation failed!")
        sys.exit(1)

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {model_dir / 'results'}")
    print(f"  - parameter_estimates.csv")
    print(f"  - model_comparison.csv")
    print(f"  - MXL_Basic.html")
    print(f"\nIndividual coefficients saved to: {model_dir / 'data' / 'individual_coefficients.csv'}")


if __name__ == "__main__":
    main()
