"""
Prepare Giray's survey data for the DCM pipeline.

This script transforms the raw survey data into the format expected by run_all_models.py.

Input: giray_data/synthetic_survey_data_v2_reversed.csv
Output: data/simulated/giray_prepared.csv

Transformations:
1. Rename columns to match pipeline conventions
2. Create CHOICE variable (1=paid1, 2=paid2, 3=standard)
3. Rename fee/duration columns
4. Scale fees by 10k
5. Rename demographics with _idx suffix
6. Center demographics
7. Rename Likert items (spaces to underscores)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def prepare_giray_data(input_path: str, output_path: str, verbose: bool = True):
    """Transform Giray's data to pipeline format."""

    df = pd.read_csv(input_path)

    if verbose:
        print(f"Loaded {len(df)} rows from {input_path}")
        print(f"Unique respondents: {df['ID'].nunique()}")

    # Create output dataframe
    out = pd.DataFrame()

    # =========================================================================
    # 1. ID and Task
    # =========================================================================
    out['ID_STR'] = df['ID']
    # Convert ID to numeric
    out['ID'] = df['ID'].str.extract(r'(\d+)').astype(int)
    out['task'] = df['choice_number']

    # =========================================================================
    # 2. Choice variable (1=paid1/A, 2=paid2/B, 3=standard/C)
    # =========================================================================
    # choice_A, choice_B, choice_C are binary indicators
    out['CHOICE'] = np.where(df['choice_A'] == 1, 1,
                    np.where(df['choice_B'] == 1, 2, 3))

    if verbose:
        print(f"\nChoice shares:")
        print(f"  1 (paid1): {(out['CHOICE']==1).mean()*100:.1f}%")
        print(f"  2 (paid2): {(out['CHOICE']==2).mean()*100:.1f}%")
        print(f"  3 (standard): {(out['CHOICE']==3).mean()*100:.1f}%")

    # =========================================================================
    # 3. Duration and Fee columns
    # =========================================================================
    # Map: paid1 -> alt1, paid2 -> alt2, standard -> alt3
    out['dur1'] = df['paid1_duration']
    out['dur2'] = df['paid2_duration']
    out['dur3'] = df['standard_duration']

    out['fee1'] = df['paid1_price']
    out['fee2'] = df['paid2_price']
    out['fee3'] = df['standard_price']

    # Scaled fees (divided by 10,000)
    FEE_SCALE = 10000.0
    out['fee1_10k'] = out['fee1'] / FEE_SCALE
    out['fee2_10k'] = out['fee2'] / FEE_SCALE
    out['fee3_10k'] = out['fee3'] / FEE_SCALE

    if verbose:
        print(f"\nFee ranges (raw):")
        print(f"  fee1: {out['fee1'].min():.0f} - {out['fee1'].max():.0f}")
        print(f"  fee2: {out['fee2'].min():.0f} - {out['fee2'].max():.0f}")
        print(f"  fee3: {out['fee3'].min():.0f} - {out['fee3'].max():.0f}")

    # =========================================================================
    # 4. Demographics with _idx suffix
    # =========================================================================
    demo_mapping = {
        'age': 'age_idx',
        'educ': 'edu_idx',
        'emp': 'work_idx',
        'hh_inc': 'income_house_idx',
        'inc': 'income_indiv_idx',
        'hh_size': 'hh_size',
        'hh_emp': 'hh_earners',
        'mrts': 'marital_idx',
    }

    for src, dst in demo_mapping.items():
        out[dst] = df[src]

    # =========================================================================
    # 5. Centered demographics (for interaction models)
    # =========================================================================
    # Use config-based centering values or compute from data
    centering = {
        'age_idx': {'center': 2.0, 'scale': 2.0},
        'edu_idx': {'center': 3.0, 'scale': 2.0},
        'income_indiv_idx': {'center': 3.0, 'scale': 2.0},
        'income_house_idx': {'center': 3.0, 'scale': 2.0},
        'marital_idx': {'center': 0.5, 'scale': 0.5},
    }

    for var, params in centering.items():
        centered_name = var.replace('_idx', '_c')
        out[centered_name] = (out[var] - params['center']) / params['scale']

    # Also create simple centered versions
    out['age_c'] = out['age_idx'] - out['age_idx'].mean()
    out['edu_c'] = out['edu_idx'] - out['edu_idx'].mean()
    out['inc_c'] = out['income_indiv_idx'] - out['income_indiv_idx'].mean()

    # =========================================================================
    # 6. Likert items - rename spaces to underscores
    # =========================================================================
    # Patriotism items
    for i in range(1, 21):
        src_col = f'patriotism {i}'
        dst_col = f'patriotism_{i}'
        if src_col in df.columns:
            out[dst_col] = df[src_col]

    # Secularism items
    for i in range(1, 26):
        src_col = f'secularism {i}'
        dst_col = f'secularism_{i}'
        if src_col in df.columns:
            out[dst_col] = df[src_col]

    # Reversed items (already computed in source)
    if 'patriotism 5_rev' in df.columns:
        out['patriotism_5_rev'] = df['patriotism 5_rev']
    if 'patriotism 7_rev' in df.columns:
        out['patriotism_7_rev'] = df['patriotism 7_rev']
    if 'secularism 13_rev' in df.columns:
        out['secularism_13_rev'] = df['secularism 13_rev']

    # =========================================================================
    # 7. Pre-computed scores (if needed for HCM proxy approach)
    # =========================================================================
    score_mapping = {
        'Score_Patriotism_Blind': 'pat_blind_proxy',
        'Score_Patriotism_Constructive': 'pat_constructive_proxy',
        'Score_Secularity_DailyLife': 'sec_dl_proxy',
        'Score_Secularity_Faith': 'sec_fp_proxy',
    }

    for src, dst in score_mapping.items():
        if src in df.columns:
            # Standardize the scores
            raw = df[src]
            out[dst] = (raw - raw.mean()) / raw.std()

    # =========================================================================
    # 8. Military service variables (potential controls)
    # =========================================================================
    mil_cols = ['mil_resp', 'mil_done', 'mil_curr']
    for col in mil_cols:
        if col in df.columns:
            out[col] = df[col]

    # =========================================================================
    # 9. Additional computed variables
    # =========================================================================
    # Bad task indicator (none in this data)
    out['bad_task'] = 0

    # Scenario ID (construct from task number)
    out['scenario_id'] = out['task']

    # =========================================================================
    # Save output
    # =========================================================================
    # Reorder columns to match expected format
    id_cols = ['ID_STR', 'ID', 'task', 'scenario_id']
    choice_cols = ['dur1', 'fee1', 'dur2', 'fee2', 'dur3', 'fee3', 'CHOICE', 'bad_task']
    fee_scaled = ['fee1_10k', 'fee2_10k', 'fee3_10k']
    demo_cols = ['age_idx', 'edu_idx', 'income_indiv_idx', 'income_house_idx',
                 'work_idx', 'marital_idx', 'hh_size', 'hh_earners']
    centered_cols = ['age_c', 'edu_c', 'inc_c']

    # Get Likert columns in order
    likert_cols = [c for c in out.columns if c.startswith('patriotism_') or c.startswith('secularism_')]
    likert_cols = sorted(likert_cols, key=lambda x: (x.split('_')[0], int(x.split('_')[1]) if x.split('_')[1].isdigit() else 999))

    # Get proxy columns
    proxy_cols = [c for c in out.columns if c.endswith('_proxy')]

    # Military columns
    mil_cols_out = [c for c in out.columns if c.startswith('mil_')]

    # Order columns
    ordered_cols = id_cols + choice_cols + fee_scaled + demo_cols + centered_cols + likert_cols + proxy_cols + mil_cols_out

    # Add any remaining columns
    remaining = [c for c in out.columns if c not in ordered_cols]
    ordered_cols += remaining

    out = out[ordered_cols]

    # Save
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    if verbose:
        print(f"\n=== Output Summary ===")
        print(f"Saved to: {output_path}")
        print(f"Shape: {out.shape}")
        print(f"Columns: {len(out.columns)}")
        print(f"\nColumn groups:")
        print(f"  ID/task: {len(id_cols)}")
        print(f"  Choice/attributes: {len(choice_cols)}")
        print(f"  Fee scaled: {len(fee_scaled)}")
        print(f"  Demographics: {len(demo_cols)}")
        print(f"  Centered: {len(centered_cols)}")
        print(f"  Likert items: {len(likert_cols)}")
        print(f"  LV proxies: {len(proxy_cols)}")

    return out


def main():
    parser = argparse.ArgumentParser(description="Prepare Giray's data for DCM pipeline")
    parser.add_argument('--input', default='giray_data/synthetic_survey_data_v2_reversed.csv',
                        help='Input CSV path')
    parser.add_argument('--output', default='data/simulated/giray_prepared.csv',
                        help='Output CSV path')
    parser.add_argument('--quiet', action='store_true', help='Suppress output')
    args = parser.parse_args()

    prepare_giray_data(args.input, args.output, verbose=not args.quiet)


if __name__ == '__main__':
    main()
