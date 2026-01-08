"""
HCM with Split Latent Variables (Robust Version)
==================================================

STATUS: CANONICAL - Recommended for production
This is the most mature HCM implementation with robust parameter bounds
and multiple model specifications.

Tests various specifications with 4 distinct latent constructs:
1. Blind Patriotism (pat_blind)
2. Constructive Patriotism (pat_const)
3. Daily Life Secularism (sec_dl)
4. Faith & Prayer Secularism (sec_fp)

Model Specifications:
A. Individual LV effects (which LV matters?)
B. Domain combinations (Patriotism vs Secularism)
C. Attribute-specific effects (Fee vs Duration)
D. Full specifications

Key Features (Robust Estimation):
- Parameter bounds to prevent extreme values
- Warm-start from baseline model
- Data validation before estimation
- Multiple optimizer retry strategy
- Identification diagnostics

METHODOLOGICAL LIMITATION - MEASUREMENT ERROR
=============================================
This implementation uses a TWO-STAGE APPROACH:
  Stage 1: Estimate latent variables from Likert items (CFA/weighted averages)
  Stage 2: Use estimated LVs as fixed regressors in choice model

This approach treats estimated latent variables as ERROR-FREE, which causes
ATTENUATION BIAS: LV effect estimates are biased toward zero.

The bias magnitude depends on:
- Reliability of the Likert scale (more items = less bias)
- True effect size (smaller effects more affected)
- Sample size (doesn't fix the bias, only precision)

For UNBIASED estimates, a full Integrated Choice and Latent Variable (ICLV)
model with SIMULTANEOUS estimation of measurement and choice models is needed.
This requires specialized software (e.g., Apollo in R, or custom likelihood).

INTERPRETATION GUIDANCE:
- If LV effects are significant here, they are likely significant (conservative)
- If LV effects are NOT significant, they may still exist (attenuated)
- Effect MAGNITUDES are underestimated; use for relative comparisons only

Author: DCM Research Team
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
import warnings
import time
import sys
# Selective warning suppression - allow important warnings through
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*overflow.*')
warnings.filterwarnings('ignore', message='.*divide by zero.*')

import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta, Variable

# Import robust estimation utilities
try:
    from src.estimation.robust_estimation import (
        DataValidator, check_identification, validate_results,
        print_estimation_summary
    )
    ROBUST_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    ROBUST_AVAILABLE = False
    # Silently fail - robust estimation is optional

# Import convergence management
try:
    from src.estimation.convergence_diagnostics import (
        ConvergenceChecker, ConvergenceDiagnostics,
        generate_convergence_table, generate_convergence_latex
    )
    from src.estimation.robust_estimation import (
        ConvergenceManager, StartingValueGenerator, SequentialEstimator
    )
    CONVERGENCE_MANAGEMENT_AVAILABLE = True
except ImportError:
    CONVERGENCE_MANAGEMENT_AVAILABLE = False

# Import publication-ready analysis modules
try:
    from src.estimation.measurement_validation import MeasurementValidator
    from src.estimation.model_comparison import ModelComparisonFramework
    from src.utils.latex_output import (
        generate_measurement_validation_latex,
        generate_factor_loadings_latex,
        generate_discriminant_validity_latex,
        generate_lr_test_matrix_latex
    )
    PUBLICATION_MODULES_AVAILABLE = True
except ImportError:
    PUBLICATION_MODULES_AVAILABLE = False

# Import ICLV for simultaneous estimation (eliminates attenuation bias)
try:
    from src.models.iclv import ICLVModel, estimate_iclv
    ICLV_AVAILABLE = True
except ImportError:
    ICLV_AVAILABLE = False


# =============================================================================
# PROGRESS TRACKING
# =============================================================================

class ProgressTracker:
    """
    Track estimation progress with ETA display.

    Provides real-time progress updates during HCM model estimation,
    including elapsed time and estimated time remaining.

    Example:
        >>> tracker = ProgressTracker(total_models=15)
        >>> for model in models:
        ...     start = time.time()
        ...     result = estimate(model)
        ...     tracker.update(result.name, time.time() - start)
        >>> tracker.finish()
    """

    def __init__(self, total_models: int, bar_width: int = 30):
        """
        Initialize progress tracker.

        Args:
            total_models: Total number of models to estimate
            bar_width: Width of progress bar in characters
        """
        self.total = total_models
        self.completed = 0
        self.start_time = time.time()
        self.model_times = []
        self.bar_width = bar_width
        self.current_model = ""

    def start_model(self, model_name: str):
        """Mark the start of a new model estimation."""
        self.current_model = model_name
        self._display_progress(estimating=True)

    def update(self, model_name: str, elapsed: float, result=None):
        """
        Update progress after completing a model.

        Args:
            model_name: Name of completed model
            elapsed: Time taken for this model (seconds)
            result: Optional ModelResult for summary display
        """
        self.completed += 1
        self.model_times.append(elapsed)
        self._display_completion(model_name, elapsed, result)

    def _format_time(self, seconds: float) -> str:
        """Format seconds as human-readable string."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"

    def _make_bar(self, pct: float) -> str:
        """Create progress bar string."""
        filled = int(self.bar_width * pct / 100)
        empty = self.bar_width - filled
        return f"[{'█' * filled}{'░' * empty}]"

    def _display_progress(self, estimating: bool = False):
        """Display current progress."""
        pct = self.completed / self.total * 100
        elapsed = time.time() - self.start_time

        if self.model_times:
            avg_time = np.mean(self.model_times)
            remaining = (self.total - self.completed) * avg_time
            eta_str = f"ETA: {self._format_time(remaining)}"
        else:
            eta_str = "ETA: calculating..."

        bar = self._make_bar(pct)
        status = "Estimating" if estimating else "Completed"

        line = f"\r{bar} {pct:5.1f}% | {status}: {self.current_model:<30} | Elapsed: {self._format_time(elapsed)} | {eta_str}"
        sys.stdout.write(line)
        sys.stdout.flush()

    def _display_completion(self, model_name: str, elapsed: float, result=None):
        """Display completion message for a model."""
        # Clear the progress line
        sys.stdout.write("\r" + " " * 120 + "\r")

        # Print completion message
        if result:
            print(f"  ✓ {model_name} ({elapsed:.1f}s) - LL: {result.ll:.2f}, AIC: {result.aic:.2f}")
        else:
            print(f"  ✓ {model_name} ({elapsed:.1f}s)")

        # Show updated progress
        self._display_progress()

    def finish(self):
        """Display final summary."""
        total_time = time.time() - self.start_time
        avg_time = np.mean(self.model_times) if self.model_times else 0

        # Clear progress line
        sys.stdout.write("\r" + " " * 120 + "\r")
        print()

        print("\n" + "=" * 70)
        print("ESTIMATION COMPLETE")
        print("=" * 70)
        print(f"Total time: {self._format_time(total_time)}")
        print(f"Average per model: {self._format_time(avg_time)}")
        print(f"Models estimated: {self.completed}/{self.total}")

        if self.model_times:
            fastest = min(self.model_times)
            slowest = max(self.model_times)
            print(f"Fastest model: {self._format_time(fastest)}")
            print(f"Slowest model: {self._format_time(slowest)}")


# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_data(data_path: str, validate: bool = True):
    """Load and prepare data with CFA latent estimates.

    Args:
        data_path: Path to CSV data file
        validate: If True, run data validation checks

    Returns:
        Tuple of (dataframe, database, null_ll)
    """
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} observations from {df['ID'].nunique()} respondents")

    df = df.copy()

    # Scale fees (divide by 10,000)
    df['fee1_10k'] = df['fee1'] / 10000.0
    df['fee2_10k'] = df['fee2'] / 10000.0
    df['fee3_10k'] = df['fee3'] / 10000.0

    # Get unique individuals for LV estimation
    individuals = df.groupby('ID').first().reset_index()

    # Define item groups
    constructs = {
        'pat_blind': [c for c in df.columns if c.startswith('pat_blind_') and c[-1].isdigit()],
        'pat_const': [c for c in df.columns if c.startswith('pat_constructive_') and c[-1].isdigit()],
        'sec_dl': [c for c in df.columns if c.startswith('sec_dl_') and c[-1].isdigit()],
        'sec_fp': [c for c in df.columns if c.startswith('sec_fp_') and c[-1].isdigit()],
    }

    print("\nLatent Variable Estimation (CFA):")
    for construct, items in constructs.items():
        if not items:
            continue
        X = individuals[items].values

        # Item-total correlation weights
        total = X.sum(axis=1)
        weights = []
        for i in range(X.shape[1]):
            corr = np.corrcoef(X[:, i], total)[0, 1]
            weights.append(max(0.1, corr))
        weights = np.array(weights) / sum(weights)

        # Weighted standardized score (ensures mean=0, std=1)
        score = (X * weights).sum(axis=1)
        score = (score - score.mean()) / score.std()
        individuals[f'LV_{construct}'] = score
        print(f"  {construct}: mean={score.mean():.3f}, std={score.std():.3f}")

    # Check LV correlations (important for multicollinearity)
    lv_cols = [f'LV_{c}' for c in constructs.keys()]
    print("\nLatent Variable Correlations:")
    corr_matrix = individuals[lv_cols].corr()
    for i, c1 in enumerate(lv_cols):
        for c2 in lv_cols[i+1:]:
            corr = corr_matrix.loc[c1, c2]
            warn = " ⚠️ HIGH" if abs(corr) > 0.7 else ""
            print(f"  {c1} x {c2}: r = {corr:.3f}{warn}")

    # Merge back
    df = df.merge(individuals[['ID'] + lv_cols], on='ID', how='left')

    # Validate against true values if available (SIMULATION ONLY)
    # This section only runs if *_true columns exist (simulated data)
    has_true_cols = any(f'LV_{c}_true' in df.columns or 'LV_pat_constructive_true' in df.columns
                        for c in constructs.keys())
    if has_true_cols:
        print("\nValidation (CFA vs True) - SIMULATION DATA:")
        for construct in constructs.keys():
            est_col = f'LV_{construct}'
            true_col = f'LV_{construct}_true' if construct != 'pat_const' else 'LV_pat_constructive_true'
            if est_col in df.columns and true_col in df.columns:
                corr = df.groupby('ID').first()[[est_col, true_col]].corr().iloc[0, 1]
                print(f"  {construct}: r = {corr:.3f}")

    # Data validation
    if validate and ROBUST_AVAILABLE:
        print("\n" + "=" * 50)
        print("DATA VALIDATION")
        print("=" * 50)
        validation = DataValidator.validate(
            df,
            choice_col='CHOICE',
            fee_cols=['fee1_10k', 'fee2_10k', 'fee3_10k'],
            lv_cols=lv_cols
        )
        DataValidator.print_validation(validation)

    # Drop string columns
    string_cols = df.select_dtypes(include=['object']).columns.tolist()
    df_num = df.drop(columns=string_cols)

    database = db.Database('hcm_split', df_num)
    n_obs = len(df_num)
    null_ll = n_obs * np.log(1/3)

    return df, database, null_ll


# =============================================================================
# MODEL SPECIFICATIONS
# =============================================================================

def get_base_vars():
    """Base choice variables."""
    return {
        'dur1': Variable('dur1'),
        'dur2': Variable('dur2'),
        'dur3': Variable('dur3'),
        'fee1': Variable('fee1_10k'),
        'fee2': Variable('fee2_10k'),
        'fee3': Variable('fee3_10k'),
        'CHOICE': Variable('CHOICE'),
    }


def get_lv_vars():
    """Latent variable references."""
    return {
        'pat_blind': Variable('LV_pat_blind'),
        'pat_const': Variable('LV_pat_const'),
        'sec_dl': Variable('LV_sec_dl'),
        'sec_fp': Variable('LV_sec_fp'),
    }


# -----------------------------------------------------------------------------
# A. INDIVIDUAL LV EFFECTS (which LV matters most?)
# -----------------------------------------------------------------------------

def model_baseline(db):
    """M0: Basic MNL (no LVs)

    Bounds added for numerical stability:
    - B_FEE: (-10, 0) - Fee should always be negative (disutility)
    - B_DUR: (-5, 0) - Duration should be negative
    """
    v = get_base_vars()

    # With bounds to prevent extreme values
    ASC = Beta('ASC_paid', 0, -10, 10, 0)
    B_FEE = Beta('B_FEE', -0.6, -10, 0, 0)  # Must be negative
    B_DUR = Beta('B_DUR', -0.05, -5, 0, 0)  # Must be negative

    V1 = ASC + B_FEE * v['fee1'] + B_DUR * v['dur1']
    V2 = ASC + B_FEE * v['fee2'] + B_DUR * v['dur2']
    V3 = B_FEE * v['fee3'] + B_DUR * v['dur3']

    return models.loglogit({1: V1, 2: V2, 3: V3}, {1:1, 2:1, 3:1}, v['CHOICE']), 'M0: Baseline MNL'


def model_pat_blind_only(db):
    """M1: Blind Patriotism effects on Fee"""
    v, lv = get_base_vars(), get_lv_vars()

    ASC = Beta('ASC_paid', 0, -10, 10, 0)
    B_FEE = Beta('B_FEE', -0.6, -10, 0, 0)
    B_DUR = Beta('B_DUR', -0.05, -5, 0, 0)

    B_FEE_PB = Beta('B_FEE_PatBlind', 0, -2, 2, 0)  # Bounded interaction

    B_FEE_i = B_FEE + B_FEE_PB * lv['pat_blind']

    V1 = ASC + B_FEE_i * v['fee1'] + B_DUR * v['dur1']
    V2 = ASC + B_FEE_i * v['fee2'] + B_DUR * v['dur2']
    V3 = B_FEE_i * v['fee3'] + B_DUR * v['dur3']

    return models.loglogit({1: V1, 2: V2, 3: V3}, {1:1, 2:1, 3:1}, v['CHOICE']), 'M1: Blind Patriotism'


def model_pat_const_only(db):
    """M2: Constructive Patriotism effects on Fee"""
    v, lv = get_base_vars(), get_lv_vars()

    ASC = Beta('ASC_paid', 0, -10, 10, 0)
    B_FEE = Beta('B_FEE', -0.6, -10, 0, 0)
    B_DUR = Beta('B_DUR', -0.05, -5, 0, 0)

    B_FEE_PC = Beta('B_FEE_PatConst', 0, -2, 2, 0)  # Bounded interaction

    B_FEE_i = B_FEE + B_FEE_PC * lv['pat_const']

    V1 = ASC + B_FEE_i * v['fee1'] + B_DUR * v['dur1']
    V2 = ASC + B_FEE_i * v['fee2'] + B_DUR * v['dur2']
    V3 = B_FEE_i * v['fee3'] + B_DUR * v['dur3']

    return models.loglogit({1: V1, 2: V2, 3: V3}, {1:1, 2:1, 3:1}, v['CHOICE']), 'M2: Constructive Patriotism'


def model_sec_dl_only(db):
    """M3: Daily Secularism effects on Fee"""
    v, lv = get_base_vars(), get_lv_vars()

    ASC = Beta('ASC_paid', 0, -10, 10, 0)
    B_FEE = Beta('B_FEE', -0.6, -10, 0, 0)
    B_DUR = Beta('B_DUR', -0.05, -5, 0, 0)

    B_FEE_SDL = Beta('B_FEE_SecDL', 0, -2, 2, 0)  # Bounded interaction

    B_FEE_i = B_FEE + B_FEE_SDL * lv['sec_dl']

    V1 = ASC + B_FEE_i * v['fee1'] + B_DUR * v['dur1']
    V2 = ASC + B_FEE_i * v['fee2'] + B_DUR * v['dur2']
    V3 = B_FEE_i * v['fee3'] + B_DUR * v['dur3']

    return models.loglogit({1: V1, 2: V2, 3: V3}, {1:1, 2:1, 3:1}, v['CHOICE']), 'M3: Daily Secularism'


def model_sec_fp_only(db):
    """M4: Faith Secularism effects on Fee"""
    v, lv = get_base_vars(), get_lv_vars()

    ASC = Beta('ASC_paid', 0, -10, 10, 0)
    B_FEE = Beta('B_FEE', -0.6, -10, 0, 0)
    B_DUR = Beta('B_DUR', -0.05, -5, 0, 0)

    B_FEE_SFP = Beta('B_FEE_SecFP', 0, -2, 2, 0)  # Bounded interaction

    B_FEE_i = B_FEE + B_FEE_SFP * lv['sec_fp']

    V1 = ASC + B_FEE_i * v['fee1'] + B_DUR * v['dur1']
    V2 = ASC + B_FEE_i * v['fee2'] + B_DUR * v['dur2']
    V3 = B_FEE_i * v['fee3'] + B_DUR * v['dur3']

    return models.loglogit({1: V1, 2: V2, 3: V3}, {1:1, 2:1, 3:1}, v['CHOICE']), 'M4: Faith Secularism'


# -----------------------------------------------------------------------------
# B. DOMAIN COMBINATIONS
# -----------------------------------------------------------------------------

def model_patriotism_both(db):
    """M5: Both Patriotism types (Blind + Constructive)"""
    v, lv = get_base_vars(), get_lv_vars()

    ASC = Beta('ASC_paid', 0, -10, 10, 0)
    B_FEE = Beta('B_FEE', -0.6, -10, 0, 0)
    B_DUR = Beta('B_DUR', -0.05, -5, 0, 0)

    B_FEE_PB = Beta('B_FEE_PatBlind', 0, -2, 2, 0)
    B_FEE_PC = Beta('B_FEE_PatConst', 0, -2, 2, 0)

    B_FEE_i = B_FEE + B_FEE_PB * lv['pat_blind'] + B_FEE_PC * lv['pat_const']

    V1 = ASC + B_FEE_i * v['fee1'] + B_DUR * v['dur1']
    V2 = ASC + B_FEE_i * v['fee2'] + B_DUR * v['dur2']
    V3 = B_FEE_i * v['fee3'] + B_DUR * v['dur3']

    return models.loglogit({1: V1, 2: V2, 3: V3}, {1:1, 2:1, 3:1}, v['CHOICE']), 'M5: Both Patriotism'


def model_secularism_both(db):
    """M6: Both Secularism types (Daily + Faith)"""
    v, lv = get_base_vars(), get_lv_vars()

    ASC = Beta('ASC_paid', 0, -10, 10, 0)
    B_FEE = Beta('B_FEE', -0.6, -10, 0, 0)
    B_DUR = Beta('B_DUR', -0.05, -5, 0, 0)

    B_FEE_SDL = Beta('B_FEE_SecDL', 0, -2, 2, 0)
    B_FEE_SFP = Beta('B_FEE_SecFP', 0, -2, 2, 0)

    B_FEE_i = B_FEE + B_FEE_SDL * lv['sec_dl'] + B_FEE_SFP * lv['sec_fp']

    V1 = ASC + B_FEE_i * v['fee1'] + B_DUR * v['dur1']
    V2 = ASC + B_FEE_i * v['fee2'] + B_DUR * v['dur2']
    V3 = B_FEE_i * v['fee3'] + B_DUR * v['dur3']

    return models.loglogit({1: V1, 2: V2, 3: V3}, {1:1, 2:1, 3:1}, v['CHOICE']), 'M6: Both Secularism'


def model_cross_domain(db):
    """M7: Blind Patriotism + Daily Secularism (cross-domain)"""
    v, lv = get_base_vars(), get_lv_vars()

    ASC = Beta('ASC_paid', 0, -10, 10, 0)
    B_FEE = Beta('B_FEE', -0.6, -10, 0, 0)
    B_DUR = Beta('B_DUR', -0.05, -5, 0, 0)

    B_FEE_PB = Beta('B_FEE_PatBlind', 0, -2, 2, 0)
    B_FEE_SDL = Beta('B_FEE_SecDL', 0, -2, 2, 0)

    B_FEE_i = B_FEE + B_FEE_PB * lv['pat_blind'] + B_FEE_SDL * lv['sec_dl']

    V1 = ASC + B_FEE_i * v['fee1'] + B_DUR * v['dur1']
    V2 = ASC + B_FEE_i * v['fee2'] + B_DUR * v['dur2']
    V3 = B_FEE_i * v['fee3'] + B_DUR * v['dur3']

    return models.loglogit({1: V1, 2: V2, 3: V3}, {1:1, 2:1, 3:1}, v['CHOICE']), 'M7: PatBlind + SecDL'


def model_all_four_fee(db):
    """M8: All 4 LVs on Fee"""
    v, lv = get_base_vars(), get_lv_vars()

    ASC = Beta('ASC_paid', 0, -10, 10, 0)
    B_FEE = Beta('B_FEE', -0.6, -10, 0, 0)
    B_DUR = Beta('B_DUR', -0.05, -5, 0, 0)

    B_FEE_PB = Beta('B_FEE_PatBlind', 0, -2, 2, 0)
    B_FEE_PC = Beta('B_FEE_PatConst', 0, -2, 2, 0)
    B_FEE_SDL = Beta('B_FEE_SecDL', 0, -2, 2, 0)
    B_FEE_SFP = Beta('B_FEE_SecFP', 0, -2, 2, 0)

    B_FEE_i = (B_FEE + B_FEE_PB * lv['pat_blind'] + B_FEE_PC * lv['pat_const'] +
               B_FEE_SDL * lv['sec_dl'] + B_FEE_SFP * lv['sec_fp'])

    V1 = ASC + B_FEE_i * v['fee1'] + B_DUR * v['dur1']
    V2 = ASC + B_FEE_i * v['fee2'] + B_DUR * v['dur2']
    V3 = B_FEE_i * v['fee3'] + B_DUR * v['dur3']

    return models.loglogit({1: V1, 2: V2, 3: V3}, {1:1, 2:1, 3:1}, v['CHOICE']), 'M8: All 4 LVs on Fee'


# -----------------------------------------------------------------------------
# C. ATTRIBUTE-SPECIFIC EFFECTS
# -----------------------------------------------------------------------------

def model_lvs_on_duration(db):
    """M9: LVs affect Duration sensitivity"""
    v, lv = get_base_vars(), get_lv_vars()

    ASC = Beta('ASC_paid', 0, -10, 10, 0)
    B_FEE = Beta('B_FEE', -0.6, -10, 0, 0)
    B_DUR = Beta('B_DUR', -0.05, -5, 0, 0)

    B_DUR_PB = Beta('B_DUR_PatBlind', 0, -1, 1, 0)
    B_DUR_SDL = Beta('B_DUR_SecDL', 0, -1, 1, 0)

    B_DUR_i = B_DUR + B_DUR_PB * lv['pat_blind'] + B_DUR_SDL * lv['sec_dl']

    V1 = ASC + B_FEE * v['fee1'] + B_DUR_i * v['dur1']
    V2 = ASC + B_FEE * v['fee2'] + B_DUR_i * v['dur2']
    V3 = B_FEE * v['fee3'] + B_DUR_i * v['dur3']

    return models.loglogit({1: V1, 2: V2, 3: V3}, {1:1, 2:1, 3:1}, v['CHOICE']), 'M9: LVs on Duration'


def model_lvs_on_asc(db):
    """M10: LVs affect ASC (preference for paid service)"""
    v, lv = get_base_vars(), get_lv_vars()

    ASC = Beta('ASC_paid', 0, -10, 10, 0)
    B_FEE = Beta('B_FEE', -0.6, -10, 0, 0)
    B_DUR = Beta('B_DUR', -0.05, -5, 0, 0)

    # LVs on ASC (preference for paid service)
    B_ASC_PB = Beta('B_ASC_PatBlind', 0, -5, 5, 0)
    B_ASC_SDL = Beta('B_ASC_SecDL', 0, -5, 5, 0)

    ASC_i = ASC + B_ASC_PB * lv['pat_blind'] + B_ASC_SDL * lv['sec_dl']

    V1 = ASC_i + B_FEE * v['fee1'] + B_DUR * v['dur1']
    V2 = ASC_i + B_FEE * v['fee2'] + B_DUR * v['dur2']
    V3 = B_FEE * v['fee3'] + B_DUR * v['dur3']

    return models.loglogit({1: V1, 2: V2, 3: V3}, {1:1, 2:1, 3:1}, v['CHOICE']), 'M10: LVs on ASC'


def model_fee_and_duration(db):
    """M11: LVs on both Fee and Duration"""
    v, lv = get_base_vars(), get_lv_vars()

    ASC = Beta('ASC_paid', 0, -10, 10, 0)
    B_FEE = Beta('B_FEE', -0.6, -10, 0, 0)
    B_DUR = Beta('B_DUR', -0.05, -5, 0, 0)

    # LVs on Fee
    B_FEE_PB = Beta('B_FEE_PatBlind', 0, -2, 2, 0)
    B_FEE_SDL = Beta('B_FEE_SecDL', 0, -2, 2, 0)

    # LVs on Duration
    B_DUR_PB = Beta('B_DUR_PatBlind', 0, -1, 1, 0)
    B_DUR_SDL = Beta('B_DUR_SecDL', 0, -1, 1, 0)

    B_FEE_i = B_FEE + B_FEE_PB * lv['pat_blind'] + B_FEE_SDL * lv['sec_dl']
    B_DUR_i = B_DUR + B_DUR_PB * lv['pat_blind'] + B_DUR_SDL * lv['sec_dl']

    V1 = ASC + B_FEE_i * v['fee1'] + B_DUR_i * v['dur1']
    V2 = ASC + B_FEE_i * v['fee2'] + B_DUR_i * v['dur2']
    V3 = B_FEE_i * v['fee3'] + B_DUR_i * v['dur3']

    return models.loglogit({1: V1, 2: V2, 3: V3}, {1:1, 2:1, 3:1}, v['CHOICE']), 'M11: LVs on Fee+Dur'


# -----------------------------------------------------------------------------
# D. COMPREHENSIVE MODELS
# -----------------------------------------------------------------------------

def model_full_patriotism(db):
    """M12: Both Patriotism types on Fee and Duration"""
    v, lv = get_base_vars(), get_lv_vars()

    ASC = Beta('ASC_paid', 0, -10, 10, 0)
    B_FEE = Beta('B_FEE', -0.6, -10, 0, 0)
    B_DUR = Beta('B_DUR', -0.05, -5, 0, 0)

    B_FEE_PB = Beta('B_FEE_PatBlind', 0, -2, 2, 0)
    B_FEE_PC = Beta('B_FEE_PatConst', 0, -2, 2, 0)
    B_DUR_PB = Beta('B_DUR_PatBlind', 0, -1, 1, 0)
    B_DUR_PC = Beta('B_DUR_PatConst', 0, -1, 1, 0)

    B_FEE_i = B_FEE + B_FEE_PB * lv['pat_blind'] + B_FEE_PC * lv['pat_const']
    B_DUR_i = B_DUR + B_DUR_PB * lv['pat_blind'] + B_DUR_PC * lv['pat_const']

    V1 = ASC + B_FEE_i * v['fee1'] + B_DUR_i * v['dur1']
    V2 = ASC + B_FEE_i * v['fee2'] + B_DUR_i * v['dur2']
    V3 = B_FEE_i * v['fee3'] + B_DUR_i * v['dur3']

    return models.loglogit({1: V1, 2: V2, 3: V3}, {1:1, 2:1, 3:1}, v['CHOICE']), 'M12: Full Patriotism'


def model_full_secularism(db):
    """M13: Both Secularism types on Fee and Duration"""
    v, lv = get_base_vars(), get_lv_vars()

    ASC = Beta('ASC_paid', 0, -10, 10, 0)
    B_FEE = Beta('B_FEE', -0.6, -10, 0, 0)
    B_DUR = Beta('B_DUR', -0.05, -5, 0, 0)

    B_FEE_SDL = Beta('B_FEE_SecDL', 0, -2, 2, 0)
    B_FEE_SFP = Beta('B_FEE_SecFP', 0, -2, 2, 0)
    B_DUR_SDL = Beta('B_DUR_SecDL', 0, -1, 1, 0)
    B_DUR_SFP = Beta('B_DUR_SecFP', 0, -1, 1, 0)

    B_FEE_i = B_FEE + B_FEE_SDL * lv['sec_dl'] + B_FEE_SFP * lv['sec_fp']
    B_DUR_i = B_DUR + B_DUR_SDL * lv['sec_dl'] + B_DUR_SFP * lv['sec_fp']

    V1 = ASC + B_FEE_i * v['fee1'] + B_DUR_i * v['dur1']
    V2 = ASC + B_FEE_i * v['fee2'] + B_DUR_i * v['dur2']
    V3 = B_FEE_i * v['fee3'] + B_DUR_i * v['dur3']

    return models.loglogit({1: V1, 2: V2, 3: V3}, {1:1, 2:1, 3:1}, v['CHOICE']), 'M13: Full Secularism'


def model_full_all(db):
    """M14: All 4 LVs on Fee and Duration"""
    v, lv = get_base_vars(), get_lv_vars()

    ASC = Beta('ASC_paid', 0, -10, 10, 0)
    B_FEE = Beta('B_FEE', -0.6, -10, 0, 0)
    B_DUR = Beta('B_DUR', -0.05, -5, 0, 0)

    # All 4 on Fee
    B_FEE_PB = Beta('B_FEE_PatBlind', 0, -2, 2, 0)
    B_FEE_PC = Beta('B_FEE_PatConst', 0, -2, 2, 0)
    B_FEE_SDL = Beta('B_FEE_SecDL', 0, -2, 2, 0)
    B_FEE_SFP = Beta('B_FEE_SecFP', 0, -2, 2, 0)

    # All 4 on Duration
    B_DUR_PB = Beta('B_DUR_PatBlind', 0, -1, 1, 0)
    B_DUR_PC = Beta('B_DUR_PatConst', 0, -1, 1, 0)
    B_DUR_SDL = Beta('B_DUR_SecDL', 0, -1, 1, 0)
    B_DUR_SFP = Beta('B_DUR_SecFP', 0, -1, 1, 0)

    B_FEE_i = (B_FEE + B_FEE_PB * lv['pat_blind'] + B_FEE_PC * lv['pat_const'] +
               B_FEE_SDL * lv['sec_dl'] + B_FEE_SFP * lv['sec_fp'])
    B_DUR_i = (B_DUR + B_DUR_PB * lv['pat_blind'] + B_DUR_PC * lv['pat_const'] +
               B_DUR_SDL * lv['sec_dl'] + B_DUR_SFP * lv['sec_fp'])

    V1 = ASC + B_FEE_i * v['fee1'] + B_DUR_i * v['dur1']
    V2 = ASC + B_FEE_i * v['fee2'] + B_DUR_i * v['dur2']
    V3 = B_FEE_i * v['fee3'] + B_DUR_i * v['dur3']

    return models.loglogit({1: V1, 2: V2, 3: V3}, {1:1, 2:1, 3:1}, v['CHOICE']), 'M14: Full Model (All LVs)'


# =============================================================================
# ESTIMATION
# =============================================================================

@dataclass
class ModelResult:
    name: str
    ll: float
    k: int
    aic: float
    bic: float
    rho2: float
    params: Dict[str, float]
    t_stats: Dict[str, float]
    converged: bool


def estimate(database, model_func, null_ll,
             warm_start: Dict[str, float] = None,
             use_robust: bool = True) -> Optional[ModelResult]:
    """
    Estimate single model with optional convergence management.

    Args:
        database: Biogeme database
        model_func: Model function returning (logprob, name)
        null_ll: Null log-likelihood for rho-squared
        warm_start: Optional starting values from previous model
        use_robust: If True, use robust convergence management

    Returns:
        ModelResult or None if estimation fails
    """
    try:
        logprob, name = model_func(database)

        # Use robust estimation with retry if available
        if use_robust and CONVERGENCE_MANAGEMENT_AVAILABLE:
            manager = ConvergenceManager(max_attempts=3, verbose=False)
            results, diagnostics = manager.estimate_until_convergence(
                database, model_func, model_name=name, warm_start=warm_start
            )
            if results is None:
                print(f"  WARNING: {name} failed all convergence attempts")
                return None
        else:
            # Fall back to basic estimation
            biogeme = bio.BIOGEME(database, logprob)
            if warm_start:
                try:
                    biogeme.set_variables_values(warm_start)
                except:
                    pass
            results = biogeme.estimate()
            diagnostics = None

        ll = results.final_loglikelihood
        k = results.number_of_free_parameters
        n = len(database.dataframe)

        aic = 2*k - 2*ll
        bic = k*np.log(n) - 2*ll
        rho2 = 1 - (ll / null_ll)

        betas = results.get_beta_values()
        t_stats = {}
        for p in betas:
            try:
                se = results.get_parameter_std_err(p)
                t_stats[p] = betas[p] / se if se > 0 else np.nan
            except:
                t_stats[p] = np.nan

        # Log convergence diagnostics if available
        if diagnostics and not diagnostics.publication_ready:
            print(f"  Convergence note for {name}:")
            if not diagnostics.converged:
                print(f"    - Did not fully converge")
            if diagnostics.gradient_norm > 1e-5:
                print(f"    - Gradient norm: {diagnostics.gradient_norm:.2e}")
            if diagnostics.hessian_condition_number > 1e6:
                print(f"    - High condition number: {diagnostics.hessian_condition_number:.2e}")

        return ModelResult(
            name=name, ll=ll, k=k, aic=aic, bic=bic, rho2=rho2,
            params=betas, t_stats=t_stats, converged=results.algorithm_has_converged
        )
    except Exception as e:
        print(f"  ERROR in {model_func.__name__}: {e}")
        return None


def print_result(r: ModelResult):
    """Print single model result."""
    print(f"\n{r.name}")
    print("-" * 50)
    print(f"  LL: {r.ll:.2f} | K: {r.k} | AIC: {r.aic:.2f} | rho2: {r.rho2:.4f}")

    for p, v in r.params.items():
        t = r.t_stats.get(p, np.nan)
        sig = '***' if abs(t) > 2.576 else '**' if abs(t) > 1.96 else '*' if abs(t) > 1.645 else ''
        print(f"  {p:20s}: {v:8.4f} (t={t:6.2f}) {sig}")


# =============================================================================
# MAIN
# =============================================================================

def run_split_lv_analysis(data_path: str, output_dir: str = None):
    """Run comprehensive HCM analysis with split latent variables."""
    print("=" * 70)
    print("HCM ANALYSIS WITH SPLIT LATENT VARIABLES")
    print("=" * 70)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

    # Load and prepare data
    df, database, null_ll = prepare_data(data_path)

    # Define all models
    all_models = [
        # A. Individual LV effects
        ("A. Individual LVs", [
            model_baseline,
            model_pat_blind_only,
            model_pat_const_only,
            model_sec_dl_only,
            model_sec_fp_only,
        ]),
        # B. Domain combinations
        ("B. Domain Combinations", [
            model_patriotism_both,
            model_secularism_both,
            model_cross_domain,
            model_all_four_fee,
        ]),
        # C. Attribute-specific
        ("C. Attribute-Specific", [
            model_lvs_on_duration,
            model_lvs_on_asc,
            model_fee_and_duration,
        ]),
        # D. Comprehensive
        ("D. Full Models", [
            model_full_patriotism,
            model_full_secularism,
            model_full_all,
        ]),
    ]

    results = []
    convergence_diagnostics = {}

    # Count total models
    total_models = sum(len(funcs) for _, funcs in all_models)

    # Initialize progress tracker
    tracker = ProgressTracker(total_models=total_models)

    print(f"\nEstimating {total_models} models...")
    print("-" * 70)

    # Estimate baseline first to use for warm-starting
    tracker.start_model("M0: Baseline MNL")
    model_start = time.time()
    baseline_result = estimate(database, model_baseline, null_ll, use_robust=True)
    model_elapsed = time.time() - model_start

    if baseline_result:
        tracker.update(baseline_result.name, model_elapsed, baseline_result)
        results.append(baseline_result)
        # Extract warm-start values from baseline
        baseline_params = baseline_result.params
    else:
        tracker.update("M0: Baseline MNL", model_elapsed)
        baseline_params = None

    # Estimate remaining models with warm-start
    for section_name, model_funcs in all_models:
        for model_func in model_funcs:
            # Skip baseline as already estimated
            if model_func == model_baseline:
                continue

            # Get model name for progress display
            model_name = model_func.__name__.replace('model_', '').replace('_', ' ').title()
            tracker.start_model(model_name)

            model_start = time.time()
            r = estimate(database, model_func, null_ll,
                        warm_start=baseline_params, use_robust=True)
            model_elapsed = time.time() - model_start

            if r:
                tracker.update(r.name, model_elapsed, r)
                results.append(r)
            else:
                tracker.update(model_name, model_elapsed)

    # Show final timing summary
    tracker.finish()

    # Summary comparison
    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)

    data = []
    for r in results:
        data.append({
            'Model': r.name,
            'LL': r.ll,
            'K': r.k,
            'AIC': r.aic,
            'BIC': r.bic,
            'rho2': r.rho2,
        })

    comp_df = pd.DataFrame(data).sort_values('AIC')
    comp_df['Rank'] = range(1, len(comp_df) + 1)

    print("\nRanked by AIC:")
    print(comp_df.to_string(index=False))

    # Best models
    print("\n" + "=" * 70)
    print("TOP 5 MODELS")
    print("=" * 70)

    for i, row in comp_df.head(5).iterrows():
        print(f"\n{row['Rank']}. {row['Model']}")
        print(f"   AIC: {row['AIC']:.2f}, BIC: {row['BIC']:.2f}, rho2: {row['rho2']:.4f}")

    # LR tests vs baseline
    print("\n" + "=" * 70)
    print("LIKELIHOOD RATIO TESTS (vs Baseline)")
    print("=" * 70)

    baseline = next((r for r in results if 'Baseline' in r.name), None)
    if baseline:
        for r in results:
            if r.name == baseline.name:
                continue
            if r.ll > baseline.ll:
                lr = 2 * (r.ll - baseline.ll)
                df_diff = r.k - baseline.k
                if df_diff > 0:
                    p_value = 1 - stats.chi2.cdf(lr, df_diff)
                    sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''
                    print(f"  {r.name}: LR={lr:.2f}, df={df_diff}, p={p_value:.4f} {sig}")

    # Save outputs
    if output_dir:
        comp_df.to_csv(output_dir / 'model_comparison.csv', index=False)

        # Parameters
        param_rows = []
        for r in results:
            for p, v in r.params.items():
                param_rows.append({
                    'Model': r.name,
                    'Parameter': p,
                    'Estimate': v,
                    't-stat': r.t_stats.get(p, np.nan)
                })
        pd.DataFrame(param_rows).to_csv(output_dir / 'parameters.csv', index=False)

        print(f"\nOutputs saved to: {output_dir}")

    # CONVERGENCE SUMMARY
    print("\n" + "=" * 70)
    print("CONVERGENCE SUMMARY")
    print("=" * 70)

    all_converged = all(r.converged for r in results)
    n_converged = sum(1 for r in results if r.converged)
    print(f"Models converged: {n_converged}/{len(results)}")

    if not all_converged:
        print("\nModels with convergence issues:")
        for r in results:
            if not r.converged:
                print(f"  - {r.name}")
    else:
        print("All models converged successfully!")

    # Generate convergence diagnostics table if available
    if CONVERGENCE_MANAGEMENT_AVAILABLE and output_dir:
        # Re-run diagnostics for summary
        checker = ConvergenceChecker()
        diagnostics_summary = []
        for r in results:
            diagnostics_summary.append({
                'Model': r.name,
                'Converged': r.converged,
                'LL': r.ll,
                'K': r.k,
                'AIC': r.aic,
            })
        pd.DataFrame(diagnostics_summary).to_csv(
            output_dir / 'convergence_summary.csv', index=False
        )

    # PUBLICATION-READY VALIDATION (NEW)
    # Run measurement validation
    if PUBLICATION_MODULES_AVAILABLE:
        validation_report = run_measurement_validation(df, output_dir)

        # Run systematic model comparison with LaTeX output
        comparison_results = run_systematic_model_comparison(results, output_dir)

    # ICLV guidance
    suggest_iclv_estimation()

    return results, comp_df


# Need to import stats for LR test
from scipy import stats


# =============================================================================
# PUBLICATION-READY VALIDATION
# =============================================================================

def run_measurement_validation(df: pd.DataFrame, output_dir: Path = None) -> Dict:
    """
    Run comprehensive measurement model validation.

    Computes psychometric properties for all latent constructs:
    - Cronbach's alpha (internal consistency)
    - Composite Reliability (CR)
    - Average Variance Extracted (AVE)
    - Fornell-Larcker discriminant validity test

    Args:
        df: DataFrame with indicator columns
        output_dir: Output directory for LaTeX tables

    Returns:
        Dict with validation results for each construct
    """
    if not PUBLICATION_MODULES_AVAILABLE:
        print("Warning: Publication modules not available. Install with:")
        print("  pip install scipy pandas numpy")
        return {}

    print("\n" + "=" * 70)
    print("MEASUREMENT MODEL VALIDATION")
    print("=" * 70)

    # Define item groups (same as in prepare_data)
    constructs = {
        'pat_blind': [c for c in df.columns if c.startswith('pat_blind_') and c[-1].isdigit()],
        'pat_const': [c for c in df.columns if c.startswith('pat_constructive_') and c[-1].isdigit()],
        'sec_dl': [c for c in df.columns if c.startswith('sec_dl_') and c[-1].isdigit()],
        'sec_fp': [c for c in df.columns if c.startswith('sec_fp_') and c[-1].isdigit()],
    }

    # Get unique individuals
    individuals = df.groupby('ID').first().reset_index()

    # Initialize validator
    validator = MeasurementValidator(
        df=individuals,
        constructs=constructs
    )

    # Run full validation and print report
    validation_result = validator.validate_all()
    validator.print_report()

    # Generate LaTeX tables if output directory specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        validator.to_latex(output_dir)
        print(f"\nLaTeX tables saved to: {output_dir}")

    # Return in dict format for compatibility
    report = {
        'constructs': {
            name: {
                'cronbachs_alpha': m.cronbachs_alpha,
                'composite_reliability': m.composite_reliability,
                'ave': m.average_variance_extracted,
                'n_items': m.n_items,
            }
            for name, m in validation_result.construct_metrics.items()
        },
        'overall_valid': validation_result.overall_valid,
        'discriminant_valid': validation_result.discriminant_validity_passed,
    }

    return report


def run_systematic_model_comparison(results: List[ModelResult],
                                    output_dir: Path = None) -> pd.DataFrame:
    """
    Run systematic model comparison with LR tests and information criteria.

    Args:
        results: List of ModelResult objects from estimation
        output_dir: Output directory for LaTeX tables

    Returns:
        DataFrame with comparison results
    """
    if not PUBLICATION_MODULES_AVAILABLE:
        print("Warning: Publication modules not available")
        return pd.DataFrame()

    print("\n" + "=" * 70)
    print("SYSTEMATIC MODEL COMPARISON")
    print("=" * 70)

    # Find baseline model
    baseline = next((r for r in results if 'Baseline' in r.name), results[0])

    # Build comparison data
    records = []
    for r in results:
        if r.name == baseline.name:
            lr_stat = 0
            df_diff = 0
            p_value = 1.0
        else:
            lr_stat = 2 * (r.ll - baseline.ll)
            df_diff = r.k - baseline.k
            if df_diff > 0 and lr_stat > 0:
                p_value = 1 - stats.chi2.cdf(lr_stat, df_diff)
            else:
                p_value = 1.0

        records.append({
            'model': r.name,
            'baseline': baseline.name,
            'lr_stat': max(0, lr_stat),
            'df': max(0, df_diff),
            'p_value': p_value,
            'significant': p_value < 0.05,
            'll': r.ll,
            'aic': r.aic,
            'bic': r.bic,
            'delta_aic': r.aic - baseline.aic,
            'delta_bic': r.bic - baseline.bic,
            'rho2': r.rho2,
        })

    comparison_df = pd.DataFrame(records)

    # Print LR test results
    print("\nLikelihood Ratio Tests vs Baseline:")
    print("-" * 70)
    print(f"{'Model':<30} {'LR Stat':>10} {'df':>5} {'p-value':>10} {'Sig':>5}")
    print("-" * 70)

    for _, row in comparison_df.iterrows():
        if row['model'] == baseline.name:
            continue
        sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
        print(f"{row['model']:<30} {row['lr_stat']:>10.2f} {row['df']:>5} {row['p_value']:>10.4f} {sig:>5}")

    # Print information criteria
    print("\nInformation Criteria Comparison:")
    print("-" * 70)
    print(f"{'Model':<30} {'AIC':>12} {'delta':>10} {'BIC':>12} {'delta':>10}")
    print("-" * 70)

    sorted_df = comparison_df.sort_values('aic')
    best_aic = sorted_df['aic'].min()
    best_bic = sorted_df['bic'].min()

    for _, row in sorted_df.iterrows():
        d_aic = row['aic'] - best_aic
        d_bic = row['bic'] - best_bic
        best_marker = " <--" if d_aic == 0 else ""
        print(f"{row['model']:<30} {row['aic']:>12.2f} {d_aic:>+10.2f} {row['bic']:>12.2f} {d_bic:>+10.2f}{best_marker}")

    # Generate LaTeX table
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        generate_lr_test_matrix_latex(comparison_df, output_dir)
        print(f"\nLaTeX tables saved to: {output_dir}/HCM/")

    return comparison_df


def suggest_iclv_estimation():
    """
    Print guidance on ICLV simultaneous estimation.

    Explains when and how to use ICLV to eliminate attenuation bias.
    """
    print("\n" + "=" * 70)
    print("RECOMMENDATION: ICLV SIMULTANEOUS ESTIMATION")
    print("=" * 70)

    if not ICLV_AVAILABLE:
        print("\nICLV module not available. Install dependencies to use.")
        return

    print("""
The two-stage approach used above has ATTENUATION BIAS:
  - LV effects are biased toward zero by 15-30%
  - Conservative: if effects are significant here, they're likely real
  - But effect MAGNITUDES are underestimated

For UNBIASED estimates, use ICLV simultaneous estimation:

    from src.models.iclv import estimate_iclv

    result = estimate_iclv(
        df=your_data,
        constructs={
            'pat_blind': ['pat_blind_1', 'pat_blind_2', ...],
            'pat_const': ['pat_constructive_1', ...],
            'sec_dl': ['sec_dl_1', ...],
            'sec_fp': ['sec_fp_1', ...],
        },
        covariates=['age', 'income', 'education'],
        choice_col='CHOICE',
        attribute_cols=['fee1_10k', 'fee2_10k', 'fee3_10k', 'dur1', 'dur2', 'dur3'],
        lv_effects={
            'pat_blind': 'B_FEE_PatBlind',
            'sec_dl': 'B_FEE_SecDL',
        },
        n_draws=500  # 500-1000 recommended
    )

    print(result.summary())

Benefits of ICLV:
  + Eliminates attenuation bias
  + Provides correct standard errors
  + Joint measurement and choice estimation
  + Proper handling of measurement error

Trade-offs:
  - Computationally intensive (Monte Carlo integration)
  - Requires more observations for convergence
  - More parameters to estimate
""")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', default='hcm_split_results')
    args = parser.parse_args()

    run_split_lv_analysis(args.data, args.output)


if __name__ == '__main__':
    main()
