#!/usr/bin/env python3
"""
HCM Full - Isolated Model Validation
====================================

This script runs the complete validation pipeline for HCM Full:
1. Generate simulated data with 4 latent variable structures
2. Estimate LVs from Likert items (two-stage)
3. Estimate choice model with all 8 LV interactions
4. Compare estimates to true parameters

Expected: Unbiased on base params, ~15-25% attenuation on all LV effects

Usage:
    python run.py
    python run.py --skip_simulation
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Run HCM Full validation')
    parser.add_argument('--skip_simulation', action='store_true',
                        help='Skip simulation, use existing data')

    args = parser.parse_args()

    model_dir = Path(__file__).parent

    print("=" * 70)
    print("HCM FULL - ISOLATED MODEL VALIDATION")
    print("=" * 70)
    print("\nThis validates HCM Full parameter recovery when DGP includes")
    print("all 4 latent variables with interactions on fee and duration.\n")
    print("Latent Variables:")
    print("  - pat_blind: Blind Patriotism")
    print("  - pat_constructive: Constructive Patriotism")
    print("  - sec_dl: Daily Life Secularism")
    print("  - sec_fp: Faith & Prayer Secularism")
    print("\nNOTE: Two-stage estimation causes attenuation bias on ALL LV effects.\n")

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
    print("STEP 2: Estimating model (two-stage)")
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
    print(f"  - HCM_Full.html")
    print(f"\nTrue LV values saved to: {model_dir / 'data' / 'true_latent_values.csv'}")


if __name__ == "__main__":
    main()
