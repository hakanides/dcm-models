"""
Variable Configuration for DCM Models
=====================================

This module provides configurable variable definitions to reduce hardcoding
and make the codebase more reusable for different choice experiment designs.

CURRENT LIMITATION
==================
Most model files (run_all_models.py, mxl_models.py, etc.) have hardcoded
variable names like fee1, fee2, fee3, dur1, dur2, dur3. This limits
reusability for different choice designs (e.g., 4 alternatives, different
attribute names).

SOLUTION PATTERN
================
Use this module's configuration-based approach for new models:

    from src.utils.variable_config import VariableConfig, get_default_config

    config = get_default_config()
    variables = config.create_biogeme_variables()

    # Now use variables['fee'][1], variables['dur'][2], etc.

MIGRATION PATH
==============
To migrate existing code:
1. Import VariableConfig
2. Replace hardcoded Variable() calls with config.create_biogeme_variables()
3. Use the returned dict to access variables by name and alternative index

Authors: Hakan Mülayim, Giray Girengir, Ataol Azeritürk
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from biogeme.expressions import Variable


@dataclass
class VariableConfig:
    """Configuration for choice model variables.

    Attributes:
        n_alternatives: Number of choice alternatives
        fee_vars: List of fee variable names (one per alternative)
        dur_vars: List of duration variable names (one per alternative)
        fee_scale: Divisor for fee scaling (e.g., 10000.0)
        choice_var: Name of choice variable
        id_var: Name of panel ID variable
        extra_vars: Additional variables needed for estimation
    """
    n_alternatives: int = 3
    fee_vars: List[str] = field(default_factory=lambda: ['fee1', 'fee2', 'fee3'])
    dur_vars: List[str] = field(default_factory=lambda: ['dur1', 'dur2', 'dur3'])
    fee_scale: float = 10000.0
    choice_var: str = 'CHOICE'
    id_var: str = 'ID'
    extra_vars: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate configuration."""
        if len(self.fee_vars) != self.n_alternatives:
            raise ValueError(f"fee_vars length ({len(self.fee_vars)}) != n_alternatives ({self.n_alternatives})")
        if len(self.dur_vars) != self.n_alternatives:
            raise ValueError(f"dur_vars length ({len(self.dur_vars)}) != n_alternatives ({self.n_alternatives})")

    def get_scaled_fee_names(self) -> List[str]:
        """Get names for scaled fee variables."""
        suffix = f"_{int(self.fee_scale/1000)}k" if self.fee_scale >= 1000 else f"_scaled"
        return [f"{v}{suffix}" for v in self.fee_vars]

    def create_biogeme_variables(self) -> Dict[str, Dict[int, Variable]]:
        """Create Biogeme Variable objects for all configured variables.

        Returns:
            Dict with keys 'fee', 'dur', 'fee_scaled', 'CHOICE', and any extra vars.
            For fee/dur, values are dicts mapping alternative index (1-based) to Variable.

        Example:
            vars = config.create_biogeme_variables()
            fee1 = vars['fee_scaled'][1]  # Scaled fee for alternative 1
            dur2 = vars['dur'][2]         # Duration for alternative 2
            choice = vars['CHOICE']       # Choice variable
        """
        variables = {}

        # Fee variables (original and scaled)
        variables['fee'] = {
            i+1: Variable(name) for i, name in enumerate(self.fee_vars)
        }
        variables['fee_scaled'] = {
            i+1: Variable(name) for i, name in enumerate(self.get_scaled_fee_names())
        }

        # Duration variables
        variables['dur'] = {
            i+1: Variable(name) for i, name in enumerate(self.dur_vars)
        }

        # Choice variable
        variables[self.choice_var] = Variable(self.choice_var)

        # ID variable
        variables[self.id_var] = Variable(self.id_var)

        # Extra variables
        for var_name in self.extra_vars:
            variables[var_name] = Variable(var_name)

        return variables

    def get_availability(self) -> Dict[int, int]:
        """Get availability dict for all alternatives (all available)."""
        return {i+1: 1 for i in range(self.n_alternatives)}


def get_default_config() -> VariableConfig:
    """Get default configuration for the DCM project.

    This matches the current hardcoded setup:
    - 3 alternatives
    - fee1, fee2, fee3 (scaled by 10000)
    - dur1, dur2, dur3
    - CHOICE as choice variable
    - ID as panel identifier
    """
    return VariableConfig(
        n_alternatives=3,
        fee_vars=['fee1', 'fee2', 'fee3'],
        dur_vars=['dur1', 'dur2', 'dur3'],
        fee_scale=10000.0,
        choice_var='CHOICE',
        id_var='ID',
        extra_vars=['age_c', 'edu_c', 'inc_c', 'inc_hh_c', 'marital_c',
                    'pat_blind_proxy', 'pat_const_proxy', 'sec_dl_proxy', 'sec_fp_proxy']
    )


def get_four_alt_config() -> VariableConfig:
    """Example configuration for 4-alternative design.

    Shows how to configure for different experimental designs.
    """
    return VariableConfig(
        n_alternatives=4,
        fee_vars=['fee1', 'fee2', 'fee3', 'fee4'],
        dur_vars=['dur1', 'dur2', 'dur3', 'dur4'],
        fee_scale=10000.0,
        choice_var='CHOICE',
        id_var='ID'
    )


# Example usage
if __name__ == '__main__':
    # Demonstrate usage
    config = get_default_config()
    print(f"Configuration: {config.n_alternatives} alternatives")
    print(f"Fee vars: {config.fee_vars}")
    print(f"Scaled fee names: {config.get_scaled_fee_names()}")

    # Create Biogeme variables
    variables = config.create_biogeme_variables()
    print(f"\nBiogeme variables created:")
    print(f"  Fee scaled vars: {list(variables['fee_scaled'].keys())}")
    print(f"  Duration vars: {list(variables['dur'].keys())}")
    print(f"  Availability: {config.get_availability()}")
