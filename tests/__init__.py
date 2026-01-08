"""
DCM Research Framework Test Suite
=================================

Comprehensive tests for discrete choice model estimation.

Test Organization:
- test_data_preparation.py: Data scaling, centering, quality checks
- test_model_convergence.py: MNL, HCM convergence tests
- test_latent_estimation.py: CFA-based latent variable estimation
- test_policy_analysis.py: WTP, elasticity, welfare analysis
- test_simulation.py: Data generation and simulation
- test_iclv.py: ICLV simultaneous estimation
- test_extended_models.py: Extended model specifications

Run tests with:
    pytest tests/ -v                    # All tests
    pytest tests/ -v -m "not slow"      # Skip slow tests
    pytest tests/ -v -m unit            # Only unit tests
    pytest tests/ -v -m iclv            # Only ICLV tests
    pytest tests/ -v -m simulation      # Only simulation tests
"""
