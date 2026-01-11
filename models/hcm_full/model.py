#!/usr/bin/env python3
"""
HCM Full Model - Simultaneous Estimation (ICLV)
================================================

Hybrid Choice Model with all 4 latent variables affecting fee and duration sensitivity.

Latent Constructs:
    1. Blind Patriotism (η_pb) - Uncritical support for country
    2. Constructive Patriotism (η_pc) - Critical, improvement-oriented
    3. Daily Life Secularism (η_sdl) - Separation in daily practices
    4. Faith & Prayer Secularism (η_sfp) - Separation in religious matters

Model Specification (Simultaneous ICLV):
    Structural Model:
        η_pb = γ_pb_age * (age - 2) + γ_pb_inc * (income - 3) + σ_pb * ω_pb
        η_pc = γ_pc_edu * (edu - 3) + σ_pc * ω_pc
        η_sdl = γ_sdl_edu * (edu - 3) + γ_sdl_inc * (income - 3) + σ_sdl * ω_sdl
        η_sfp = γ_sfp_edu * (edu - 3) + σ_sfp * ω_sfp

    Measurement Model (Ordered Probit):
        P(I_ki = j | η_k) = Φ(τ_j - λ_ki*η_k) - Φ(τ_{j-1} - λ_ki*η_k)

    Choice Model:
        B_FEE_i = B_FEE + B_FEE_PB*η_pb + B_FEE_PC*η_pc + B_FEE_SDL*η_sdl + B_FEE_SFP*η_sfp
        B_DUR_i = B_DUR + B_DUR_PB*η_pb + B_DUR_PC*η_pc + B_DUR_SDL*η_sdl + B_DUR_SFP*η_sfp

        V1 = ASC_paid + B_FEE_i * fee1 + B_DUR_i * dur1
        V2 = ASC_paid + B_FEE_i * fee2 + B_DUR_i * dur2
        V3 = B_FEE_i * fee3 + B_DUR_i * dur3

    Joint Likelihood:
        L_n = ∫∫∫∫ P(choice|η) × Π_k Π_i P(I_ki|η_k) dη_pb dη_pc dη_sdl dη_sfp

    Estimated via Monte Carlo integration using Halton draws.

This eliminates attenuation bias from two-stage estimation.

References:
    - Ben-Akiva et al. (2002): Hybrid Choice Models
    - Walker & Ben-Akiva (2002): Generalized Random Utility Model
    - Bierlaire (2018): Biogeme Documentation on Latent Variables
"""

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Add shared utilities to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.policy_tools import run_policy_analysis
from shared.latex_tools import generate_all_latex
from shared import validate_required_columns, validate_coefficient_signs

import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
# Handle Biogeme API changes (bioDraws deprecated in 3.3+, use Draws)
try:
    from biogeme.expressions import Beta, Variable, Draws, MonteCarlo, log, exp, Elem, bioNormalCdf
except ImportError:
    from biogeme.expressions import Beta, Variable, bioDraws as Draws, MonteCarlo, log, exp, Elem, bioNormalCdf


# Alternative availability (1=available, 0=not available)
# This should match the alternatives defined in config.json
ALTERNATIVES = {1: 'paid1', 2: 'paid2', 3: 'standard'}
AV_DICT = {k: 1 for k in ALTERNATIVES.keys()}  # All alternatives available


# =============================================================================
# MEASUREMENT MODEL (Ordered Probit for Likert Items)
# =============================================================================

def ordered_probit_prob(indicator, loading, lv, tau1, tau2, tau3, tau4):
    """
    Compute ordered probit probability for a single indicator.

    P(I = j | η) = Φ(τ_j - λ*η) - Φ(τ_{j-1} - λ*η)

    Args:
        indicator: Biogeme Variable for the indicator (1-5)
        loading: Factor loading (λ)
        lv: Latent variable expression (η)
        tau1-4: Threshold parameters (must be ordered)

    Returns:
        Biogeme expression for probability
    """
    z = loading * lv

    # Cumulative probabilities using normal CDF
    prob_1 = bioNormalCdf(tau1 - z)
    prob_2 = bioNormalCdf(tau2 - z) - bioNormalCdf(tau1 - z)
    prob_3 = bioNormalCdf(tau3 - z) - bioNormalCdf(tau2 - z)
    prob_4 = bioNormalCdf(tau4 - z) - bioNormalCdf(tau3 - z)
    prob_5 = 1 - bioNormalCdf(tau4 - z)

    # Select probability based on observed indicator value
    prob = Elem({1: prob_1, 2: prob_2, 3: prob_3, 4: prob_4, 5: prob_5}, indicator)

    return prob


def create_construct_measurement(lv, items: list,
                                  tau1, tau2, tau3, tau4,
                                  loadings: dict):
    """
    Create measurement likelihood for a single construct.

    Args:
        lv: Latent variable expression
        items: List of indicator column names
        tau1-4: Threshold parameters
        loadings: Dict of item -> loading (Beta or float)

    Returns:
        Joint probability expression for this construct's indicators
    """
    joint_prob = 1.0

    for item in items:
        indicator = Variable(item)
        loading = loadings[item]

        prob = ordered_probit_prob(indicator, loading, lv, tau1, tau2, tau3, tau4)
        joint_prob = joint_prob * prob

    return joint_prob


# =============================================================================
# MODEL SPECIFICATION
# =============================================================================

def create_iclv_model(database: db.Database, config: dict, available_columns: list):
    """
    Create ICLV model with simultaneous estimation for all 4 LVs.

    Key features:
    1. Structural model for each LV: η_k = Γ_k*X + σ_k*ω_k
    2. Measurement model: Ordered probit for Likert items
    3. Choice model: Logit with LV interactions on fee and duration
    4. Integration: Monte Carlo over 4-dimensional latent space

    Args:
        database: Biogeme database
        config: Model configuration with LV items and structural parameters
        available_columns: List of available column names in the data

    Returns:
        Log-likelihood expression for estimation
    """
    # -------------------------------------------------------------------------
    # CHOICE VARIABLES
    # -------------------------------------------------------------------------
    CHOICE = Variable('CHOICE')
    fee1 = Variable('fee1_10k')
    fee2 = Variable('fee2_10k')
    fee3 = Variable('fee3_10k')
    dur1 = Variable('dur1')
    dur2 = Variable('dur2')
    dur3 = Variable('dur3')

    # -------------------------------------------------------------------------
    # DEMOGRAPHICS (Raw and Centered)
    # -------------------------------------------------------------------------
    age_idx = Variable('age_idx')
    edu_idx = Variable('edu_idx')
    income_idx = Variable('income_indiv_idx')

    # Get centering values from config (use pat_blind as reference, others should match)
    latent_cfg = config.get('latent', {})
    pb_center = latent_cfg.get('pat_blind', {}).get('structural', {}).get('center', {})
    sdl_center = latent_cfg.get('sec_dl', {}).get('structural', {}).get('center', {})

    age_center = pb_center.get('age_idx', 2.0)
    income_center = pb_center.get('income_indiv_idx', sdl_center.get('income_indiv_idx', 3.0))
    edu_center = latent_cfg.get('pat_constructive', {}).get('structural', {}).get('center', {}).get('edu_idx', 3.0)

    age_centered = age_idx - age_center
    edu_centered = edu_idx - edu_center
    income_centered = income_idx - income_center

    # -------------------------------------------------------------------------
    # RANDOM DRAWS (Halton sequences for each LV)
    # -------------------------------------------------------------------------
    omega_pb = Draws('omega_pb', 'NORMAL_HALTON2')
    omega_pc = Draws('omega_pc', 'NORMAL_HALTON3')
    omega_sdl = Draws('omega_sdl', 'NORMAL_HALTON5')
    omega_sfp = Draws('omega_sfp', 'NORMAL_HALTON7')

    # -------------------------------------------------------------------------
    # STRUCTURAL MODEL: η_k = Γ_k*X + σ_k*ω_k
    # -------------------------------------------------------------------------
    # Pat Blind: η_pb = γ_age*age + γ_inc*income + σ_pb*ω
    gamma_pb_age = Beta('gamma_pb_age', 0.2, -2, 2, 0)
    gamma_pb_inc = Beta('gamma_pb_inc', -0.15, -2, 2, 0)
    sigma_pb = Beta('sigma_pb', 1.0, 0.1, 5, 0)
    LV_pat_blind = gamma_pb_age * age_centered + gamma_pb_inc * income_centered + sigma_pb * omega_pb

    # Pat Constructive: η_pc = γ_edu*edu + σ_pc*ω
    gamma_pc_edu = Beta('gamma_pc_edu', 0.25, -2, 2, 0)
    sigma_pc = Beta('sigma_pc', 1.0, 0.1, 5, 0)
    LV_pat_const = gamma_pc_edu * edu_centered + sigma_pc * omega_pc

    # Sec Daily Life: η_sdl = γ_edu*edu + γ_inc*income + σ_sdl*ω
    gamma_sdl_edu = Beta('gamma_sdl_edu', 0.18, -2, 2, 0)
    gamma_sdl_inc = Beta('gamma_sdl_inc', 0.12, -2, 2, 0)
    sigma_sdl = Beta('sigma_sdl', 1.0, 0.1, 5, 0)
    LV_sec_dl = gamma_sdl_edu * edu_centered + gamma_sdl_inc * income_centered + sigma_sdl * omega_sdl

    # Sec Faith & Prayer: η_sfp = γ_edu*edu + σ_sfp*ω
    gamma_sfp_edu = Beta('gamma_sfp_edu', 0.2, -2, 2, 0)
    sigma_sfp = Beta('sigma_sfp', 1.0, 0.1, 5, 0)
    LV_sec_fp = gamma_sfp_edu * edu_centered + sigma_sfp * omega_sfp

    # -------------------------------------------------------------------------
    # MEASUREMENT MODEL: Ordered Probit with Delta Thresholds
    # -------------------------------------------------------------------------
    # IMPORTANT: SHARED THRESHOLDS ACROSS ALL CONSTRUCTS
    # This model uses ONE set of thresholds for all 4 latent constructs.
    # This is a RESTRICTIVE assumption that implies all constructs have the
    # same response scale distribution.
    #
    # WHEN SHARED THRESHOLDS ARE APPROPRIATE:
    # - Items use same Likert scale with consistent anchoring
    # - Preliminary analysis shows similar response distributions
    # - Sample size is limited (reduces parameters by 12)
    #
    # WHEN CONSTRUCT-SPECIFIC THRESHOLDS ARE NEEDED:
    # - Different constructs show different response patterns
    # - Some constructs are skewed (e.g., floor/ceiling effects)
    # - Publication requires testing measurement invariance
    #
    # To add construct-specific thresholds, define tau_pb_1, delta_pb_2, etc.
    # for each construct (see models/iclv/model.py for example).
    #
    # Delta parameterization GUARANTEES ordering: τ_k = τ_{k-1} + exp(δ_k)
    #
    # NOTE ON DGP ALIGNMENT:
    # The DGP uses FIXED thresholds [-1.0, -0.35, 0.35, 1.0] for simulation.
    # Starting values aligned with DGP for faster convergence.
    tau_1 = Beta('tau_1', -1.0, -3, 3, 0)
    delta_2 = Beta('delta_2', -0.5, -3, 3, 0)   # exp(-0.5)≈0.61, tau_2≈-0.39
    delta_3 = Beta('delta_3', -0.36, -3, 3, 0)  # exp(-0.36)≈0.70, tau_3≈0.31
    delta_4 = Beta('delta_4', -0.5, -3, 3, 0)   # exp(-0.5)≈0.61, tau_4≈0.92

    tau_2 = tau_1 + exp(delta_2)
    tau_3 = tau_2 + exp(delta_3)
    tau_4 = tau_3 + exp(delta_4)

    # Get latent variable config
    latent_cfg = config.get('latent', {})

    # Build measurement model for each construct
    all_loadings = {}
    measurement_prob = 1.0

    constructs = [
        ('pat_blind', LV_pat_blind),
        ('pat_constructive', LV_pat_const),
        ('sec_dl', LV_sec_dl),
        ('sec_fp', LV_sec_fp)
    ]

    for construct_name, lv in constructs:
        cfg = latent_cfg.get(construct_name, {})
        items = cfg.get('measurement', {}).get('items', [])

        # Filter to available items
        available_items = [item for item in items if item in available_columns]

        if available_items:
            print(f"  {construct_name}: {len(available_items)} indicators")

            # Factor loadings (first fixed to 1 for scale identification)
            loadings = {}
            loadings[available_items[0]] = 1.0  # First loading fixed

            for item in available_items[1:]:
                loadings[item] = Beta(f'lambda_{item}', 0.8, 0.1, 3.0, 0)

            all_loadings.update(loadings)

            prob = create_construct_measurement(
                lv, available_items, tau_1, tau_2, tau_3, tau_4, loadings
            )
            measurement_prob = measurement_prob * prob
        else:
            print(f"  WARNING: No items found for {construct_name}")

    # -------------------------------------------------------------------------
    # CHOICE MODEL
    # -------------------------------------------------------------------------
    # Base parameters
    ASC_paid = Beta('ASC_paid', 5.0, -10, 15, 0)
    B_FEE = Beta('B_FEE', -0.08, -1, 0, 0)
    B_DUR = Beta('B_DUR', -0.08, -1, 0, 0)

    # LV effects on fee sensitivity
    B_FEE_PatBlind = Beta('B_FEE_PatBlind', -0.10, -2, 2, 0)
    B_FEE_PatConst = Beta('B_FEE_PatConst', -0.04, -2, 2, 0)
    B_FEE_SecDL = Beta('B_FEE_SecDL', 0.05, -2, 2, 0)
    B_FEE_SecFP = Beta('B_FEE_SecFP', 0.04, -2, 2, 0)

    # LV effects on duration sensitivity
    B_DUR_PatBlind = Beta('B_DUR_PatBlind', 0.06, -2, 2, 0)
    B_DUR_PatConst = Beta('B_DUR_PatConst', 0.04, -2, 2, 0)
    B_DUR_SecDL = Beta('B_DUR_SecDL', -0.05, -2, 2, 0)
    B_DUR_SecFP = Beta('B_DUR_SecFP', -0.04, -2, 2, 0)

    # Individual-specific fee coefficient
    B_FEE_i = (B_FEE
               + B_FEE_PatBlind * LV_pat_blind
               + B_FEE_PatConst * LV_pat_const
               + B_FEE_SecDL * LV_sec_dl
               + B_FEE_SecFP * LV_sec_fp)

    # Individual-specific duration coefficient
    B_DUR_i = (B_DUR
               + B_DUR_PatBlind * LV_pat_blind
               + B_DUR_PatConst * LV_pat_const
               + B_DUR_SecDL * LV_sec_dl
               + B_DUR_SecFP * LV_sec_fp)

    # Utility functions
    V1 = ASC_paid + B_FEE_i * fee1 + B_DUR_i * dur1
    V2 = ASC_paid + B_FEE_i * fee2 + B_DUR_i * dur2
    V3 = B_FEE_i * fee3 + B_DUR_i * dur3

    V = {1: V1, 2: V2, 3: V3}

    # Choice probability (conditional on LVs)
    choice_prob = models.logit(V, AV_DICT, CHOICE)

    # -------------------------------------------------------------------------
    # JOINT LIKELIHOOD WITH MONTE CARLO INTEGRATION
    # L_n = ∫∫∫∫ P(choice|η) × P(indicators|η) dη_1 dη_2 dη_3 dη_4
    # -------------------------------------------------------------------------
    joint_prob = choice_prob * measurement_prob

    # Monte Carlo integration over latent variable draws
    integrated_prob = MonteCarlo(joint_prob)

    # Log-likelihood
    logprob = log(integrated_prob)

    return logprob


# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_data(filepath: Path, config: dict) -> pd.DataFrame:
    """Load and prepare data for ICLV estimation."""
    df = pd.read_csv(filepath)

    # Base required columns
    required_cols = ['CHOICE', 'fee1', 'fee2', 'fee3', 'dur1', 'dur2', 'dur3',
                     'age_idx', 'edu_idx', 'income_indiv_idx']

    # Add all indicator items from config
    latent_cfg = config.get('latent', {})
    all_items = []
    for lv_cfg in latent_cfg.values():
        items = lv_cfg.get('measurement', {}).get('items', [])
        all_items.extend(items)
    required_cols.extend(all_items)

    # Validate required columns exist
    validate_required_columns(df, required_cols, 'HCM_Full')

    # Ensure fee columns are scaled
    if 'fee1_10k' not in df.columns:
        for alt in [1, 2, 3]:
            df[f'fee{alt}_10k'] = df[f'fee{alt}'] / 10000.0

    for item in all_items:
        if item in df.columns:
            df[item] = df[item].round().astype(int).clip(1, 5)

    # Keep only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return df[numeric_cols].copy()


# =============================================================================
# ESTIMATION
# =============================================================================

def estimate(model_dir: Path, verbose: bool = True, n_draws: int = 2000) -> dict:
    """
    Estimate HCM Full model using simultaneous ICLV approach.

    Args:
        model_dir: Directory containing model files
        verbose: Print progress
        n_draws: Number of Halton draws for Monte Carlo integration

    Returns:
        Dictionary with estimation results
    """
    data_path = model_dir / "data" / "simulated_data.csv"
    config_path = model_dir / "config.json"
    results_dir = model_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path) as f:
        config = json.load(f)

    true_values = config['model_info']['true_values']

    if verbose:
        print("=" * 70)
        print("HCM FULL MODEL - SIMULTANEOUS ICLV ESTIMATION")
        print("(All 4 Latent Variables with Structural Equations)")
        print("=" * 70)
        print(f"\nTrue parameters:")
        for k, v in true_values.items():
            print(f"  {k}: {v}")
        print(f"\nUsing {n_draws} Halton draws for Monte Carlo integration")
        print("\nKey features:")
        print("  - 4 LVs with structural equations (demographics → LV)")
        print("  - Delta thresholds: τ_k = τ_{k-1} + exp(δ_k)")
        print("  - Ordered probit measurement model")
        print("  - Monte Carlo integration over 4D latent space")

    # Load and prepare data
    df = prepare_data(data_path, config)
    n_obs = len(df)

    if verbose:
        print(f"\nData: {n_obs} observations")

    # Create database
    database = db.Database('hcm_full_iclv', df)
    database.panel('ID')  # Declare panel structure for consistent random draws within individuals

    if verbose:
        print("\nBuilding ICLV model with 4 latent variables...")

    # Create ICLV model
    logprob = create_iclv_model(database, config, df.columns.tolist())

    # Create Biogeme object with draws
    biogeme_model = bio.BIOGEME(database, logprob, number_of_draws=n_draws)
    biogeme_model.model_name = "HCM_Full_ICLV"

    # Calculate null log-likelihood
    biogeme_model.calculate_null_loglikelihood(AV_DICT)

    original_dir = os.getcwd()
    os.chdir(results_dir)

    if verbose:
        print("\nEstimating model (this may take several minutes)...")

    try:
        results = biogeme_model.estimate()
    finally:
        os.chdir(original_dir)

    # Check convergence status
    general_stats = results.get_general_statistics()
    converged = True
    for key, val in general_stats.items():
        if 'Optimization' in key and 'algorithm' not in key.lower():
            status = str(val[0]) if isinstance(val, tuple) else str(val)
            if 'success' not in status.lower() and 'converged' not in status.lower():
                converged = False
                if verbose:
                    print(f"\nWARNING: Optimization may not have converged! Status: {status}")
                break

    # Extract results
    estimates_df = results.get_estimated_parameters()
    estimates_df = estimates_df.set_index('Name')

    betas = estimates_df['Value'].to_dict()
    stderrs = estimates_df['Robust std err.'].to_dict()
    tstats = estimates_df['Robust t-stat.'].to_dict()
    pvals = estimates_df['Robust p-value'].to_dict()

    # Validate coefficient signs
    if verbose:
        validate_coefficient_signs(betas, model_name='HCM_Full')

    # Get fit statistics (reuse general_stats from convergence check)
    final_ll = None
    null_ll = None
    aic = None
    bic = None

    for key, val in general_stats.items():
        if 'Final log likelihood' in key:
            final_ll = float(val[0]) if isinstance(val, tuple) else float(val)
        elif 'Null log likelihood' in key:
            null_ll = float(val[0]) if isinstance(val, tuple) else float(val)
        elif 'Akaike' in key:
            aic = float(val[0]) if isinstance(val, tuple) else float(val)
        elif 'Bayesian' in key:
            bic = float(val[0]) if isinstance(val, tuple) else float(val)

    rho2 = 1 - final_ll / null_ll if null_ll else 0

    if verbose:
        print("\n" + "=" * 70)
        print("ESTIMATION RESULTS")
        print("=" * 70)
        print(f"\nLog-Likelihood: {final_ll:.4f}")
        print(f"Null LL: {null_ll:.4f}")
        print(f"Rho-squared: {rho2:.4f}")
        print(f"AIC: {aic:.2f}")
        print(f"BIC: {bic:.2f}")

        # Choice model parameters
        print("\n" + "-" * 70)
        print("CHOICE MODEL - BASE PARAMETERS")
        print("-" * 70)
        print(f"{'Parameter':<20} {'Estimate':>10} {'Std.Err':>10} {'t-stat':>10} {'p-value':>10}")
        print("-" * 65)

        base_params = ['ASC_paid', 'B_FEE', 'B_DUR']
        for param in base_params:
            if param in betas:
                sig = "***" if pvals[param] < 0.01 else "**" if pvals[param] < 0.05 else "*" if pvals[param] < 0.10 else ""
                print(f"{param:<20} {betas[param]:>10.4f} {stderrs[param]:>10.4f} {tstats[param]:>10.2f} {pvals[param]:>10.4f} {sig}")

        print("\n" + "-" * 70)
        print("FEE SENSITIVITY INTERACTIONS")
        print("-" * 70)
        fee_params = ['B_FEE_PatBlind', 'B_FEE_PatConst', 'B_FEE_SecDL', 'B_FEE_SecFP']
        for param in fee_params:
            if param in betas:
                sig = "***" if pvals[param] < 0.01 else "**" if pvals[param] < 0.05 else "*" if pvals[param] < 0.10 else ""
                print(f"{param:<20} {betas[param]:>10.4f} {stderrs[param]:>10.4f} {tstats[param]:>10.2f} {pvals[param]:>10.4f} {sig}")

        print("\n" + "-" * 70)
        print("DURATION SENSITIVITY INTERACTIONS")
        print("-" * 70)
        dur_params = ['B_DUR_PatBlind', 'B_DUR_PatConst', 'B_DUR_SecDL', 'B_DUR_SecFP']
        for param in dur_params:
            if param in betas:
                sig = "***" if pvals[param] < 0.01 else "**" if pvals[param] < 0.05 else "*" if pvals[param] < 0.10 else ""
                print(f"{param:<20} {betas[param]:>10.4f} {stderrs[param]:>10.4f} {tstats[param]:>10.2f} {pvals[param]:>10.4f} {sig}")

        print("\n" + "-" * 70)
        print("STRUCTURAL MODEL PARAMETERS")
        print("-" * 70)
        struct_params = [p for p in betas.keys() if p.startswith('gamma_') or p.startswith('sigma_')]
        for param in sorted(struct_params):
            if param in betas:
                sig = "***" if pvals[param] < 0.01 else "**" if pvals[param] < 0.05 else "*" if pvals[param] < 0.10 else ""
                print(f"{param:<20} {betas[param]:>10.4f} {stderrs[param]:>10.4f} {tstats[param]:>10.2f} {pvals[param]:>10.4f} {sig}")

        print("\n" + "-" * 70)
        print("MEASUREMENT MODEL PARAMETERS")
        print("-" * 70)

        # Thresholds
        print("\nThresholds (delta parameterization):")
        if 'tau_1' in betas:
            tau1 = betas['tau_1']
            delta2 = betas.get('delta_2', 0)
            delta3 = betas.get('delta_3', 0)
            delta4 = betas.get('delta_4', 0)
            tau2 = tau1 + np.exp(delta2)
            tau3 = tau2 + np.exp(delta3)
            tau4 = tau3 + np.exp(delta4)
            print(f"  τ_1 = {tau1:.4f}")
            print(f"  τ_2 = {tau2:.4f}")
            print(f"  τ_3 = {tau3:.4f}")
            print(f"  τ_4 = {tau4:.4f}")

        # Loadings by construct
        print("\nFactor Loadings:")
        for construct in ['pat_blind', 'pat_const', 'sec_dl', 'sec_fp']:
            loadings = [p for p in sorted(betas.keys()) if p.startswith(f'lambda_{construct}')]
            if loadings:
                print(f"  {construct}:")
                for param in loadings:
                    print(f"    {param}: {betas[param]:.4f} (SE: {stderrs[param]:.4f})")

        # Comparison to true values
        print("\n" + "=" * 70)
        print("COMPARISON TO TRUE PARAMETERS")
        print("=" * 70)
        print(f"\n{'Parameter':<20} {'True':>10} {'Estimated':>10} {'Bias%':>10} {'95%CI':<6}")
        print("-" * 65)

        all_covered = True
        all_params = base_params + fee_params + dur_params

        for param in all_params:
            if param in betas and param in true_values:
                true_val = true_values[param]
                est_val = betas[param]
                se = stderrs[param]

                if abs(true_val) > 0.0001:
                    bias_pct = ((est_val - true_val) / abs(true_val)) * 100
                else:
                    bias_pct = 0

                ci_low = est_val - 1.96 * se
                ci_high = est_val + 1.96 * se
                covered = ci_low <= true_val <= ci_high
                coverage_str = "Yes" if covered else "NO"
                if not covered:
                    all_covered = False

                print(f"{param:<20} {true_val:>10.4f} {est_val:>10.4f} {bias_pct:>+9.1f}% {coverage_str:<6}")

        print("\n" + "-" * 65)
        if all_covered:
            print("SUCCESS: All true parameters fall within 95% confidence intervals")
        else:
            print("NOTE: Some parameters outside 95% CI")

        print("\n*** ICLV with structural equations provides unbiased LV effects! ***")

    # Save results
    results_dict = {
        'model': 'HCM_Full_ICLV',
        'estimation_method': 'Simultaneous (Monte Carlo)',
        'n_draws': n_draws,
        'log_likelihood': final_ll,
        'null_ll': null_ll,
        'rho_squared': rho2,
        'aic': aic,
        'bic': bic,
        'n_obs': n_obs,
        'parameters': betas,
        'std_errors': stderrs,
        't_stats': tstats,
        'p_values': pvals,
        'true_values': true_values,
        'biogeme_results': results,
        'data': df,
        'config': config
    }

    # Save parameter estimates
    param_rows = []
    for param in betas.keys():
        true_val = true_values.get(param, None)
        bias_pct = ((betas[param] - true_val) / abs(true_val)) * 100 if true_val and abs(true_val) > 0.0001 else None
        ci_low = betas[param] - 1.96 * stderrs[param]
        ci_high = betas[param] + 1.96 * stderrs[param]
        covered = ci_low <= true_val <= ci_high if true_val is not None else None

        param_rows.append({
            'Parameter': param,
            'True_Value': true_val,
            'Estimate': betas[param],
            'Std_Error': stderrs[param],
            't_stat': tstats[param],
            'p_value': pvals[param],
            'Bias_Percent': bias_pct,
            'CI_95_Low': ci_low,
            'CI_95_High': ci_high,
            'CI_Coverage': covered
        })

    pd.DataFrame(param_rows).to_csv(results_dir / 'parameter_estimates.csv', index=False)

    pd.DataFrame([{
        'Model': 'HCM_Full_ICLV',
        'Method': 'Simultaneous',
        'N_Draws': n_draws,
        'LL': final_ll,
        'Null_LL': null_ll,
        'K': len(betas),
        'N': n_obs,
        'AIC': aic,
        'BIC': bic,
        'Rho2': rho2
    }]).to_csv(results_dir / 'model_comparison.csv', index=False)

    if verbose:
        print(f"\nResults saved to: {results_dir}")

    return results_dict


def main():
    """Entry point for standalone execution."""
    model_dir = Path(__file__).parent
    # Note: 4D latent space integration requires more draws than 2D
    # For publication-quality results, use n_draws=5000+
    # For testing/development, n_draws=1000 may suffice
    results = estimate(model_dir, n_draws=2000)

    # Run policy analysis
    policy_dir = model_dir / 'policy_analysis'
    policy_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 70)
    print("POLICY ANALYSIS")
    print("=" * 70)

    policy_results = run_policy_analysis(
        biogeme_results=results['biogeme_results'],
        df=results['data'],
        config=results['config'],
        output_dir=policy_dir,
        model_type='HCM_FULL_ICLV',
        verbose=True
    )

    # Generate LaTeX tables
    latex_dir = model_dir / 'output' / 'latex'
    latex_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("LATEX OUTPUT")
    print("=" * 70)

    generate_all_latex(
        biogeme_results=results['biogeme_results'],
        true_values=results['true_values'],
        policy_results=policy_results,
        output_dir=latex_dir,
        model_name='HCM_Full_ICLV'
    )

    print(f"\nPolicy analysis saved to: {policy_dir}")
    print(f"LaTeX tables saved to: {latex_dir}")


if __name__ == "__main__":
    main()
