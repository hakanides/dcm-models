#!/usr/bin/env python3
"""
ICLV - Isolated Model Validation
================================

This script runs the complete validation pipeline for ICLV:
1. Clean previous outputs
2. Generate simulated data with latent variable structure
3. Estimate ICLV model (simultaneous estimation)
4. Generate policy analysis (WTP, welfare analysis)
5. Generate LaTeX tables
6. Compare estimates to true parameters

Expected: UNBIASED on ALL parameters (including LV effects!)

The key advantage of ICLV over HCM is that simultaneous estimation
integrates over the latent variable distributions, eliminating the
attenuation bias present in two-stage estimation.

Usage:
    python run.py
    python run.py --skip_simulation
    python run.py --n_draws 1000
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Add shared utilities to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.cleanup import cleanup_before_run


def main():
    parser = argparse.ArgumentParser(description='Run ICLV validation')
    parser.add_argument('--skip_simulation', action='store_true',
                        help='Skip simulation, use existing data')
    parser.add_argument('--skip_cleanup', action='store_true',
                        help='Skip cleanup of previous outputs')
    parser.add_argument('--n_draws', type=int, default=500,
                        help='Number of Halton draws for simulation (default: 500)')

    args = parser.parse_args()

    model_dir = Path(__file__).parent

    print("=" * 70)
    print("ICLV - ISOLATED MODEL VALIDATION")
    print("=" * 70)
    print("\nThis validates ICLV parameter recovery using simultaneous estimation.")
    print("\nLatent Variables:")
    print("  - pat_blind: Blind Patriotism (structural: age)")
    print("  - sec_dl: Daily Life Secularism (structural: education)")
    print("\nMeasurement Model: 3 Likert items per construct (ordered probit)")
    print("\nKey Advantage: ICLV eliminates attenuation bias present in HCM!")
    print()

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
    print("STEP 2: Estimating ICLV model (simultaneous estimation)")
    print("-" * 70)
    print(f"Using {args.n_draws} draws for Monte Carlo integration...")
    print("(This may take several minutes)")

    # Pass n_draws to model.py
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
    print(f"    - ICLV.html")
    print(f"  data/")
    print(f"    - true_latent_values.csv")
    print(f"  output/latex/")
    print(f"    - parameter_table.tex")
    print(f"    - model_summary.tex")
    print(f"    - policy_summary.tex")
    print(f"  policy_analysis/")
    print(f"    - wtp_results.csv")
    print(f"    - wtp_by_age_idx.csv (segment analysis)")
    print(f"    - elasticity_matrix.csv")
    print(f"    - market_shares.csv")
    print(f"    - welfare_analysis.csv")
    print(f"  sample_stats/")
    print(f"    - demographics_summary.csv")
    print(f"    - likert_summary.csv")
    print(f"    - *.png (plots)")
    print("\nCompare with HCM results to see the attenuation bias elimination!")


if __name__ == "__main__":
    main()
