"""
Simulation Module for DCM Research
===================================

This module contains different data simulators for DCM model validation.

SIMULATOR COMPARISON (Issue #16)
================================

1. dcm_simulator.py (DCMSimulator class)
   - Basic simulator for simple MNL models
   - Supports attribute terms and base coefficients
   - Uses config-based variable definitions
   - Best for: Quick validation of basic MNL specifications

2. dcm_simulator_advanced.py (DCMSimulatorAdvanced class)
   - Full-featured simulator with all model components
   - Supports: demographic interactions, latent variables, MXL random coefficients
   - Generates Likert scale indicators from true latent values
   - Implements complex _get_term_value() with fallback chains
   - Best for: Comprehensive validation of HCM and MXL models

3. simulate_full_data.py (run_full_simulation function)
   - Standalone script that uses DCMSimulatorAdvanced
   - Generates complete dataset with all required variables
   - Saves to data/simulated/simulated_data.csv
   - Best for: Generating production simulation data

DIFFERENCES IN _get_term_value():
- Basic: Simple lookup with default to 0
- Advanced: Hierarchical fallback (exact match → column search → default)
            Handles reverse-coded Likert items automatically

RECOMMENDED USAGE:
- For basic MNL testing: dcm_simulator.py
- For full model validation: dcm_simulator_advanced.py (via simulate_full_data.py)
- For production data generation: python simulate_full_data.py

WARNING: Ensure you use the SAME simulator for both data generation and model
validation to avoid systematic bias from implementation differences.

ISSUE #38: RANDOM COEFFICIENTS IN BASIC SIMULATOR
=================================================
The basic dcm_simulator.py does NOT support random coefficients for MXL.
This means:
- You CANNOT validate MXL heterogeneity estimates with basic simulator data
- Data from basic simulator will show σ ≈ 0 (no true heterogeneity)
- This is expected behavior, not a bug

To validate MXL models:
1. Use dcm_simulator_advanced.py with random_coefficients config section
2. Or use model_config.json which specifies random coefficient distributions
3. The advanced simulator generates individual-specific βs with panel consistency

Example config for random coefficients (model_config.json):
    "random_coefficients": {
        "coefficients": {
            "b_fee_scaled": {
                "distribution": "normal",
                "mean": -0.5,
                "std": 0.15
            }
        }
    }
"""

# Note: Simulators are not imported here to avoid circular dependencies
# Import directly from the specific module:
#   from src.simulation.dcm_simulator import DCMSimulator
#   from src.simulation.dcm_simulator_advanced import DCMSimulatorAdvanced
