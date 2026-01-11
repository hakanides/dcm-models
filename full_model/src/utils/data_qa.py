"""
Data Quality Assurance for DCM Estimation
==========================================

Validates DCM data before model estimation to catch common issues:
1. Constant attributes (not identifiable)
2. Extreme choice share imbalance
3. Extreme fee/attribute values (numerical issues)
4. Missing values

Authors: Hakan Mülayim, Giray Girengir, Ataol Azeritürk
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from pathlib import Path


def validate_dcm_data(df: pd.DataFrame,
                      choice_col: str = 'CHOICE',
                      fee_cols: List[str] = None,
                      dur_cols: List[str] = None,
                      max_share_threshold: float = 0.8,
                      min_share_threshold: float = 0.05,
                      max_fee_threshold: float = 1e6,
                      fee_scale: float = 10000.0,
                      max_utility_contribution: float = 10.0,
                      fail_on_error: bool = False) -> Dict[str, Any]:
    """
    Validate DCM data and return issues found.

    Args:
        df: DataFrame with DCM data
        choice_col: Name of choice variable
        fee_cols: Fee column names (default: fee1, fee2, fee3)
        dur_cols: Duration column names (default: dur1, dur2, dur3)
        max_share_threshold: Warn if any alternative has share > this
        max_fee_threshold: Warn if max fee exceeds this
        fail_on_error: If True, raise exception on ERROR-level issues

    Returns:
        Dict with 'valid' (bool) and 'issues' (list of strings)
    """
    if fee_cols is None:
        fee_cols = ['fee1', 'fee2', 'fee3']
    if dur_cols is None:
        dur_cols = ['dur1', 'dur2', 'dur3']

    issues = []
    errors = []

    print("=" * 60)
    print("DATA QUALITY CHECK")
    print("=" * 60)

    # 1. Basic info
    n_obs = len(df)
    n_respondents = df['ID'].nunique() if 'ID' in df.columns else 'Unknown'
    print(f"\nObservations: {n_obs:,}")
    print(f"Respondents: {n_respondents}")

    # 2. Check choice share balance
    print("\n--- Choice Share Analysis ---")
    if choice_col in df.columns:
        shares = df[choice_col].value_counts(normalize=True).sort_index()
        print("Choice shares:")
        for alt, share in shares.items():
            status = "OK"
            if share > max_share_threshold:
                status = f"WARNING: >{max_share_threshold*100:.0f}% (dominant)"
                issues.append(f"WARNING: Alternative {alt} has {share:.1%} share (>{max_share_threshold*100:.0f}%) - quasi-separation risk")
            elif share < min_share_threshold:
                status = f"WARNING: <{min_share_threshold*100:.0f}% (rare)"
                issues.append(f"WARNING: Alternative {alt} has {share:.1%} share (<{min_share_threshold*100:.0f}%) - poor identification")
            print(f"  Alt {alt}: {share:.1%} {status}")

    # 3. Check attribute variation
    print("\n--- Attribute Variation Check ---")
    all_attr_cols = fee_cols + dur_cols

    for col in all_attr_cols:
        if col in df.columns:
            nunique = df[col].nunique()
            min_val = df[col].min()
            max_val = df[col].max()

            if nunique == 1:
                msg = f"ERROR: {col} is constant (value={min_val}) - NOT IDENTIFIABLE"
                errors.append(msg)
                print(f"  {col}: nunique=1 [{msg}]")
            else:
                print(f"  {col}: nunique={nunique}, range=[{min_val:.2f}, {max_val:.2f}]")

    # 4. Check fee scaling and utility contributions
    print("\n--- Fee Scaling & Utility Check ---")
    for col in fee_cols:
        if col in df.columns:
            max_fee = df[col].max()
            mean_fee = df[col].mean()

            # Estimate utility contribution with typical coefficient
            max_scaled = max_fee / fee_scale
            est_utility = abs(max_scaled)  # Assuming coefficient magnitude ~1

            status_parts = []
            if max_fee > max_fee_threshold:
                issues.append(f"WARNING: {col} max={max_fee:,.0f} - consider scaling")
                status_parts.append("HIGH FEE")

            if est_utility > max_utility_contribution:
                issues.append(f"WARNING: {col} scaled contribution ~{est_utility:.1f} may cause numerical issues")
                status_parts.append(f"UTILITY ~{est_utility:.1f}")

            status = " | ".join(status_parts) if status_parts else "OK"
            print(f"  {col}: max={max_fee:,.0f}, scaled={max_scaled:.2f} - {status}")

    # 5. Check for missing values
    print("\n--- Missing Value Check ---")
    missing = df[all_attr_cols + [choice_col]].isnull().sum()
    has_missing = missing.sum() > 0
    if has_missing:
        print("  Missing values found:")
        for col, count in missing[missing > 0].items():
            issues.append(f"WARNING: {col} has {count} missing values")
            print(f"    {col}: {count} missing")
    else:
        print("  No missing values - OK")

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    n_errors = len(errors)
    n_warnings = len([i for i in issues if "WARNING" in i])

    valid = n_errors == 0

    if valid:
        print("\n  Status: VALID - Data can be used for estimation")
    else:
        print(f"\n  Status: INVALID - {n_errors} error(s) found")
        print("\n  Errors:")
        for err in errors:
            print(f"    - {err}")

    if n_warnings > 0:
        print(f"\n  Warnings: {n_warnings}")
        for warn in [i for i in issues if "WARNING" in i]:
            print(f"    - {warn}")

    print("=" * 60)

    # Fail if requested
    if fail_on_error and not valid:
        raise ValueError(f"Data validation failed with {n_errors} error(s)")

    return {
        "valid": valid,
        "issues": errors + issues,
        "errors": errors,
        "warnings": [i for i in issues if "WARNING" in i],
        "n_observations": n_obs,
        "n_respondents": n_respondents
    }


def validate_and_report(data_path: str, output_path: str = None) -> Dict[str, Any]:
    """
    Load data, validate, and optionally save report.

    Args:
        data_path: Path to CSV data file
        output_path: Optional path to save validation report

    Returns:
        Validation results dict
    """
    df = pd.read_csv(data_path)
    results = validate_dcm_data(df)

    if output_path:
        report_lines = [
            "DCM Data Validation Report",
            "=" * 40,
            f"Data file: {data_path}",
            f"Observations: {results['n_observations']}",
            f"Respondents: {results['n_respondents']}",
            f"Valid: {results['valid']}",
            "",
            "Errors:" if results['errors'] else "No errors",
        ]
        for err in results['errors']:
            report_lines.append(f"  - {err}")

        report_lines.append("")
        report_lines.append("Warnings:" if results['warnings'] else "No warnings")
        for warn in results['warnings']:
            report_lines.append(f"  - {warn}")

        Path(output_path).write_text("\n".join(report_lines))
        print(f"\nReport saved to: {output_path}")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Validate DCM data before estimation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python data_qa.py --data data/test_small_sample.csv
  python data_qa.py --data data/test_small_sample.csv --fail-on-error
        """
    )
    parser.add_argument('--data', required=True, help='Path to data CSV')
    parser.add_argument('--output', help='Path to save validation report')
    parser.add_argument('--fail-on-error', action='store_true',
                        help='Exit with error if validation fails')
    parser.add_argument('--max-share', type=float, default=0.8,
                        help='Maximum acceptable choice share (default: 0.8)')

    args = parser.parse_args()

    df = pd.read_csv(args.data)
    results = validate_dcm_data(
        df,
        max_share_threshold=args.max_share,
        fail_on_error=args.fail_on_error
    )

    if args.output:
        validate_and_report(args.data, args.output)

    # Exit with error code if invalid
    if not results['valid']:
        exit(1)


if __name__ == '__main__':
    main()
