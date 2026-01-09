#!/usr/bin/env python3
"""
ICLV Model Estimation
=====================

Integrated Choice and Latent Variable model with simultaneous estimation.
This eliminates the attenuation bias present in two-stage HCM estimation.

Model Specification:
    Structural Model:
        pat_blind* = gamma_age * (age - 2) + sigma_pb * omega_pb
        sec_dl* = gamma_edu * (edu - 3) + sigma_sdl * omega_sdl

    Measurement Model (ordered probit):
        I_k* = lambda_k * LV* + epsilon_k
        I_k = ordinal response based on thresholds

    Choice Model:
        B_FEE_i = B_FEE + B_FEE_PatBlind * pat_blind* + B_FEE_SecDL * sec_dl*
        V1 = ASC_paid + B_FEE_i * fee1 + B_DUR * dur1
        V2 = ASC_paid + B_FEE_i * fee2 + B_DUR * dur2
        V3 = B_FEE_i * fee3 + B_DUR * dur3

Note: Simultaneous estimation integrates over the latent variable distributions,
eliminating attenuation bias. Uses Monte Carlo simulation.
"""

import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import (
    Beta, Variable, bioDraws, MonteCarlo, log, Elem, bioNormalCdf
)


def prepare_data(filepath: Path, config: dict) -> pd.DataFrame:
    """Load and prepare data for ICLV estimation."""
    df = pd.read_csv(filepath)

    # Ensure fee columns are scaled
    if 'fee1_10k' not in df.columns:
        for alt in [1, 2, 3]:
            df[f'fee{alt}_10k'] = df[f'fee{alt}'] / 10000.0

    # Center demographics as specified in config
    demographics_cfg = config.get('demographics', {})
    for demo_name, demo_spec in demographics_cfg.items():
        if demo_name in df.columns:
            center = demo_spec.get('center', 0)
            df[f'{demo_name}_centered'] = df[demo_name] - center

    # Ensure CHOICE is included and numeric
    required_cols = ['ID', 'CHOICE', 'fee1_10k', 'fee2_10k', 'fee3_10k',
                     'dur1', 'dur2', 'dur3',
                     'pat_blind_1', 'pat_blind_2', 'pat_blind_3',
                     'sec_dl_1', 'sec_dl_2', 'sec_dl_3',
                     'age_idx_centered', 'edu_idx_centered']

    # Keep required columns plus any other numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    all_cols = list(set(required_cols) & set(df.columns)) + [c for c in numeric_cols if c not in required_cols]

    return df[all_cols].copy()


def create_iclv_model(database: db.Database, config: dict):
    """
    Create ICLV model with simultaneous estimation.

    Uses Monte Carlo integration over the latent variable distributions.
    """
    # Variables
    CHOICE = Variable('CHOICE')
    fee1 = Variable('fee1_10k')
    fee2 = Variable('fee2_10k')
    fee3 = Variable('fee3_10k')
    dur1 = Variable('dur1')
    dur2 = Variable('dur2')
    dur3 = Variable('dur3')

    # Centered demographics
    age_centered = Variable('age_idx_centered')
    edu_centered = Variable('edu_idx_centered')

    # Likert items
    pat_blind_1 = Variable('pat_blind_1')
    pat_blind_2 = Variable('pat_blind_2')
    pat_blind_3 = Variable('pat_blind_3')
    sec_dl_1 = Variable('sec_dl_1')
    sec_dl_2 = Variable('sec_dl_2')
    sec_dl_3 = Variable('sec_dl_3')

    # ===== PARAMETERS =====
    # Choice model parameters
    ASC_paid = Beta('ASC_paid', 1.0, -10, 10, 0)
    B_FEE = Beta('B_FEE', -0.05, -10, 0, 0)
    B_DUR = Beta('B_DUR', -0.05, -5, 0, 0)
    B_FEE_PatBlind = Beta('B_FEE_PatBlind', 0, -2, 2, 0)
    B_FEE_SecDL = Beta('B_FEE_SecDL', 0, -2, 2, 0)

    # Structural model parameters
    gamma_age = Beta('gamma_age', 0.1, -2, 2, 0)
    gamma_edu = Beta('gamma_edu', 0.1, -2, 2, 0)
    sigma_pb = Beta('sigma_pb', 1.0, 0.1, 5, 0)  # LV standard deviation
    sigma_sdl = Beta('sigma_sdl', 1.0, 0.1, 5, 0)

    # Measurement model parameters - loadings (first item fixed to 1)
    lambda_pb_1 = Beta('lambda_pb_1', 1.0, None, None, 1)  # Fixed to 1
    lambda_pb_2 = Beta('lambda_pb_2', 0.8, 0.1, 2, 0)
    lambda_pb_3 = Beta('lambda_pb_3', 0.8, 0.1, 2, 0)
    lambda_sdl_1 = Beta('lambda_sdl_1', 1.0, None, None, 1)  # Fixed to 1
    lambda_sdl_2 = Beta('lambda_sdl_2', 0.8, 0.1, 2, 0)
    lambda_sdl_3 = Beta('lambda_sdl_3', 0.8, 0.1, 2, 0)

    # Measurement model - thresholds (ordered)
    # tau_1 < tau_2 < tau_3 < tau_4 for 5 categories
    tau_pb_1 = Beta('tau_pb_1', -1.0, -3, 3, 0)
    tau_pb_2 = Beta('tau_pb_2', -0.35, -3, 3, 0)
    tau_pb_3 = Beta('tau_pb_3', 0.35, -3, 3, 0)
    tau_pb_4 = Beta('tau_pb_4', 1.0, -3, 3, 0)

    tau_sdl_1 = Beta('tau_sdl_1', -1.0, -3, 3, 0)
    tau_sdl_2 = Beta('tau_sdl_2', -0.35, -3, 3, 0)
    tau_sdl_3 = Beta('tau_sdl_3', 0.35, -3, 3, 0)
    tau_sdl_4 = Beta('tau_sdl_4', 1.0, -3, 3, 0)

    # ===== RANDOM DRAWS =====
    # Standard normal draws for latent variables
    omega_pb = bioDraws('omega_pb', 'NORMAL')
    omega_sdl = bioDraws('omega_sdl', 'NORMAL')

    # ===== STRUCTURAL MODEL =====
    # Latent variable values (integrate over these)
    pat_blind_star = gamma_age * age_centered + sigma_pb * omega_pb
    sec_dl_star = gamma_edu * edu_centered + sigma_sdl * omega_sdl

    # ===== MEASUREMENT MODEL =====
    # Probability of observing each Likert response given LV
    def ordered_probit_prob(item_value, lv_star, loading, tau1, tau2, tau3, tau4):
        """Compute probability of ordinal response given continuous LV."""
        # y* = lambda * LV* + epsilon, epsilon ~ N(0,1)
        # P(Y=k) = Phi(tau_k - lambda*LV*) - Phi(tau_{k-1} - lambda*LV*)
        z = loading * lv_star
        prob_1 = bioNormalCdf(tau1 - z)
        prob_2 = bioNormalCdf(tau2 - z) - bioNormalCdf(tau1 - z)
        prob_3 = bioNormalCdf(tau3 - z) - bioNormalCdf(tau2 - z)
        prob_4 = bioNormalCdf(tau4 - z) - bioNormalCdf(tau3 - z)
        prob_5 = 1 - bioNormalCdf(tau4 - z)

        # Select probability based on observed value
        prob = Elem({1: prob_1, 2: prob_2, 3: prob_3, 4: prob_4, 5: prob_5}, item_value)
        return prob

    # Measurement probabilities for pat_blind items
    prob_pb_1 = ordered_probit_prob(pat_blind_1, pat_blind_star, lambda_pb_1,
                                     tau_pb_1, tau_pb_2, tau_pb_3, tau_pb_4)
    prob_pb_2 = ordered_probit_prob(pat_blind_2, pat_blind_star, lambda_pb_2,
                                     tau_pb_1, tau_pb_2, tau_pb_3, tau_pb_4)
    prob_pb_3 = ordered_probit_prob(pat_blind_3, pat_blind_star, lambda_pb_3,
                                     tau_pb_1, tau_pb_2, tau_pb_3, tau_pb_4)

    # Measurement probabilities for sec_dl items
    prob_sdl_1 = ordered_probit_prob(sec_dl_1, sec_dl_star, lambda_sdl_1,
                                      tau_sdl_1, tau_sdl_2, tau_sdl_3, tau_sdl_4)
    prob_sdl_2 = ordered_probit_prob(sec_dl_2, sec_dl_star, lambda_sdl_2,
                                      tau_sdl_1, tau_sdl_2, tau_sdl_3, tau_sdl_4)
    prob_sdl_3 = ordered_probit_prob(sec_dl_3, sec_dl_star, lambda_sdl_3,
                                      tau_sdl_1, tau_sdl_2, tau_sdl_3, tau_sdl_4)

    # Joint measurement likelihood
    prob_measurement = prob_pb_1 * prob_pb_2 * prob_pb_3 * prob_sdl_1 * prob_sdl_2 * prob_sdl_3

    # ===== CHOICE MODEL =====
    # Individual-specific fee coefficient
    B_FEE_i = B_FEE + B_FEE_PatBlind * pat_blind_star + B_FEE_SecDL * sec_dl_star

    # Utility functions
    V1 = ASC_paid + B_FEE_i * fee1 + B_DUR * dur1
    V2 = ASC_paid + B_FEE_i * fee2 + B_DUR * dur2
    V3 = B_FEE_i * fee3 + B_DUR * dur3

    V = {1: V1, 2: V2, 3: V3}
    av = {1: 1, 2: 1, 3: 1}

    # Choice probability
    prob_choice = models.logit(V, av, CHOICE)

    # ===== JOINT LIKELIHOOD =====
    # Joint probability of choice and measurements, integrated over LV distribution
    prob_joint = prob_choice * prob_measurement

    # Monte Carlo integration
    logprob = log(MonteCarlo(prob_joint))

    return logprob


def estimate(model_dir: Path, n_draws: int = 100, verbose: bool = True) -> dict:
    """Estimate ICLV model with simultaneous estimation."""
    data_path = model_dir / "data" / "simulated_data.csv"
    config_path = model_dir / "config.json"
    results_dir = model_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path) as f:
        config = json.load(f)

    true_values = config['model_info']['true_values']

    if verbose:
        print("=" * 70)
        print("ICLV MODEL ESTIMATION (SIMULTANEOUS)")
        print("=" * 70)
        print(f"\nModel: {config['model_info']['name']}")
        print(f"Draws: {n_draws}")
        print("\nTrue choice parameters:")
        for k, v in true_values.items():
            print(f"  {k}: {v}")
        print("\nNote: Simultaneous estimation eliminates attenuation bias!")

    # Load and prepare data
    df = prepare_data(data_path, config)
    n_obs = len(df)

    if verbose:
        print(f"\nData: {n_obs} observations")
        print(f"Individuals: {df['ID'].nunique()}")

    # Create database with draws
    # Note: Not using panel() to keep variable names simple
    # Each observation treated independently for this demonstration
    database = db.Database('iclv', df)

    # Create model
    logprob = create_iclv_model(database, config)

    biogeme_model = bio.BIOGEME(database, logprob, number_of_draws=n_draws)
    biogeme_model.model_name = "ICLV"

    original_dir = os.getcwd()
    os.chdir(results_dir)

    if verbose:
        print("\nEstimating ICLV model (this may take a few minutes)...")

    try:
        results = biogeme_model.estimate()
    finally:
        os.chdir(original_dir)

    # Extract results
    estimates_df = results.get_estimated_parameters()
    estimates_df = estimates_df.set_index('Name')

    betas = estimates_df['Value'].to_dict()
    stderrs = estimates_df['Robust std err.'].to_dict()
    tstats = estimates_df['Robust t-stat.'].to_dict()
    pvals = estimates_df['Robust p-value'].to_dict()

    # Get fit statistics
    general_stats = results.get_general_statistics()
    final_ll = None
    aic = None
    bic = None

    for key, val in general_stats.items():
        if 'Final log likelihood' in key:
            final_ll = float(val[0]) if isinstance(val, tuple) else float(val)
        elif 'Akaike' in key:
            aic = float(val[0]) if isinstance(val, tuple) else float(val)
        elif 'Bayesian' in key:
            bic = float(val[0]) if isinstance(val, tuple) else float(val)

    if verbose:
        print("\n" + "=" * 70)
        print("ESTIMATION RESULTS")
        print("=" * 70)
        print(f"\nLog-Likelihood: {final_ll:.4f}")
        print(f"AIC: {aic:.2f}" if aic else "")
        print(f"BIC: {bic:.2f}" if bic else "")

        print("\n--- Choice Model Parameters ---")
        print(f"{'Parameter':<20} {'Estimate':>10} {'Std.Err':>10} {'t-stat':>10} {'p-value':>10}")
        print("-" * 65)

        choice_params = ['ASC_paid', 'B_FEE', 'B_FEE_PatBlind', 'B_FEE_SecDL', 'B_DUR']
        for param in choice_params:
            if param in betas:
                sig = "***" if pvals[param] < 0.01 else "**" if pvals[param] < 0.05 else "*" if pvals[param] < 0.10 else ""
                print(f"{param:<20} {betas[param]:>10.4f} {stderrs[param]:>10.4f} {tstats[param]:>10.2f} {pvals[param]:>10.4f} {sig}")

        print("\n--- Structural Model Parameters ---")
        struct_params = ['gamma_age', 'gamma_edu', 'sigma_pb', 'sigma_sdl']
        for param in struct_params:
            if param in betas:
                sig = "***" if pvals[param] < 0.01 else "**" if pvals[param] < 0.05 else "*" if pvals[param] < 0.10 else ""
                print(f"{param:<20} {betas[param]:>10.4f} {stderrs[param]:>10.4f} {tstats[param]:>10.2f} {pvals[param]:>10.4f} {sig}")

        print("\n--- Measurement Model Parameters ---")
        meas_params = [p for p in betas.keys() if 'lambda' in p or 'tau' in p]
        for param in sorted(meas_params):
            if param in betas and pvals.get(param) is not None:
                sig = "***" if pvals[param] < 0.01 else "**" if pvals[param] < 0.05 else "*" if pvals[param] < 0.10 else ""
                print(f"{param:<20} {betas[param]:>10.4f} {stderrs[param]:>10.4f} {tstats[param]:>10.2f} {pvals[param]:>10.4f} {sig}")

        print("\n" + "=" * 70)
        print("COMPARISON TO TRUE PARAMETERS")
        print("=" * 70)
        print(f"\n{'Parameter':<20} {'True':>10} {'Estimated':>10} {'Bias%':>10} {'95%CI':<6}")
        print("-" * 65)

        all_covered = True
        for param in choice_params:
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
            print("WARNING: Some parameters outside 95% CI")

        print("\nNote: ICLV provides UNBIASED estimates of LV effects through")
        print("simultaneous estimation (unlike two-stage HCM).")

    # Save results
    results_dict = {
        'model': 'ICLV',
        'log_likelihood': final_ll,
        'aic': aic,
        'bic': bic,
        'n_obs': n_obs,
        'n_draws': n_draws,
        'parameters': betas,
        'std_errors': stderrs,
        't_stats': tstats,
        'p_values': pvals,
        'true_values': true_values
    }

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
            't_stat': tstats.get(param),
            'p_value': pvals.get(param),
            'Bias_Percent': bias_pct,
            'CI_95_Low': ci_low,
            'CI_95_High': ci_high,
            'CI_Coverage': covered
        })

    pd.DataFrame(param_rows).to_csv(results_dir / 'parameter_estimates.csv', index=False)

    pd.DataFrame([{
        'Model': 'ICLV',
        'LL': final_ll,
        'K': len(betas),
        'N': n_obs,
        'AIC': aic,
        'BIC': bic,
        'N_Draws': n_draws
    }]).to_csv(results_dir / 'model_comparison.csv', index=False)

    if verbose:
        print(f"\nResults saved to: {results_dir}")

    return results_dict


def main():
    model_dir = Path(__file__).parent
    estimate(model_dir)


if __name__ == "__main__":
    main()
