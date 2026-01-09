#!/usr/bin/env python3
"""
MNL Demographics - Isolated Model Validation
============================================

This script runs the complete validation pipeline for MNL Demographics:
1. Clean previous outputs
2. Generate simulated data with demographic interactions
3. Estimate the model
4. Generate policy analysis (WTP, elasticities, market shares)
5. Generate LaTeX tables
6. Compare estimates to true parameters

Expected result: Unbiased parameter recovery (<10% bias, 95% CI coverage)

Usage:
    python run.py
    python run.py --skip_simulation
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Add shared utilities to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.cleanup import cleanup_before_run


def main():
    parser = argparse.ArgumentParser(description='Run MNL Demographics validation')
    parser.add_argument('--skip_simulation', action='store_true',
                        help='Skip simulation, use existing data')
    parser.add_argument('--skip_cleanup', action='store_true',
                        help='Skip cleanup of previous outputs')

    args = parser.parse_args()

    model_dir = Path(__file__).parent

    print("=" * 70)
    print("MNL DEMOGRAPHICS - ISOLATED MODEL VALIDATION")
    print("=" * 70)
    print("\nThis validates that MNL Demographics can recover true parameters")
    print("when the DGP includes demographic interactions.\n")

    # Step 0: Cleanup previous outputs
    if not args.skip_cleanup:
        cleanup_before_run(model_dir)

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
    print(f"\nOutputs saved to:")
    print(f"  results/")
    print(f"    - parameter_estimates.csv")
    print(f"    - model_comparison.csv")
    print(f"    - MNL_Demographics.html")
    print(f"  output/latex/")
    print(f"    - parameter_table.tex")
    print(f"    - model_summary.tex")
    print(f"    - policy_summary.tex")
    print(f"  policy_analysis/")
    print(f"    - wtp_results.csv")
    print(f"    - elasticity_matrix.csv")
    print(f"    - market_shares.csv")
    print(f"    - wtp_by_age_idx.csv (segment analysis)")
    print(f"  sample_stats/")
    print(f"    - demographics_summary.csv")
    print(f"    - choice_distribution.csv")
    print(f"    - *.png (plots)")


if __name__ == "__main__":
    main()
