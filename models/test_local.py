#!/usr/bin/env python3
"""
Lightweight Local Testing Script for HCM/ICLV Models
=====================================================

This script runs models with reduced parameters for local testing on machines
with limited RAM. For full estimation, use Colab or reduce draws/sample size.

Usage:
    python test_local.py hcm_basic --draws 100
    python test_local.py hcm_full --draws 50 --sample 500
    python test_local.py iclv --draws 100
"""

import argparse
import sys
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Lightweight model testing')
    parser.add_argument('model', choices=['hcm_basic', 'hcm_full', 'iclv', 'mnl_basic', 'mnl_demographics'],
                        help='Model to test')
    parser.add_argument('--draws', type=int, default=100,
                        help='Number of Monte Carlo draws (default: 100, full: 500-1000)')
    parser.add_argument('--sample', type=int, default=None,
                        help='Limit to first N observations (default: use all)')
    parser.add_argument('--syntax-only', action='store_true',
                        help='Only check syntax, do not run estimation')

    args = parser.parse_args()

    model_dir = Path(__file__).parent / args.model

    if not model_dir.exists():
        print(f"ERROR: Model directory not found: {model_dir}")
        sys.exit(1)

    print("=" * 70)
    print(f"LIGHTWEIGHT LOCAL TEST: {args.model.upper()}")
    print("=" * 70)
    print(f"\nSettings:")
    print(f"  Monte Carlo draws: {args.draws} (full estimation uses 500-1000)")
    print(f"  Sample limit: {args.sample or 'All observations'}")
    print(f"  Mode: {'Syntax check only' if args.syntax_only else 'Full estimation'}")

    # Add model directory to path
    sys.path.insert(0, str(model_dir))
    os.chdir(model_dir)

    print(f"\nLoading model from: {model_dir}")

    # Syntax check
    try:
        import model
        print("  Syntax check: PASSED")

        if args.syntax_only:
            print("\nSyntax check complete. Model is ready for estimation.")
            return

    except Exception as e:
        print(f"  Syntax check: FAILED")
        print(f"  Error: {e}")
        sys.exit(1)

    # Check if model has estimate function with n_draws parameter
    if hasattr(model, 'estimate'):
        print(f"\nRunning estimation with {args.draws} draws...")
        print("(This may take 2-10 minutes depending on model complexity)\n")

        try:
            # For models that support n_draws parameter
            import inspect
            sig = inspect.signature(model.estimate)

            kwargs = {'model_dir': model_dir, 'verbose': True}

            if 'n_draws' in sig.parameters:
                kwargs['n_draws'] = args.draws

            results = model.estimate(**kwargs)

            print("\n" + "=" * 70)
            print("TEST COMPLETED SUCCESSFULLY")
            print("=" * 70)

            if 'parameters' in results:
                print(f"\nEstimated {len(results['parameters'])} parameters")

        except MemoryError:
            print("\n" + "!" * 70)
            print("MEMORY ERROR: System ran out of RAM")
            print("!" * 70)
            print("\nSuggestions:")
            print(f"  1. Reduce draws: python test_local.py {args.model} --draws 50")
            print(f"  2. Use sample subset: python test_local.py {args.model} --draws 50 --sample 200")
            print("  3. Use Google Colab (see models/colab_runner.ipynb)")
            sys.exit(1)

        except Exception as e:
            print(f"\nEstimation failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        print("\nWARNING: Model does not have standard estimate() function")
        print("Running model.main() instead...")
        model.main()


if __name__ == "__main__":
    main()
