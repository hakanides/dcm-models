"""
Agent-Based Discrete Choice Model (DCM) Simulator
=================================================

This module implements synthetic data generation for discrete choice experiments
using an agent-based approach. Each agent represents a simulated respondent with:

1. Demographics: Drawn from population distributions
2. Latent Variables: Attitudes (e.g., patriotism, secularism) as functions of demographics
3. Taste Parameters: Individual-specific coefficients for choice attributes
4. Choice Behavior: Utility-maximizing choices with random error

The simulation follows a known Data Generating Process (DGP), allowing for
validation of estimation methods by comparing estimated vs. true parameters.

Author: DCM Research Team
"""

import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path


# =============================================================================
# AGENT CLASS
# =============================================================================

@dataclass
class Agent:
    """
    Represents a single respondent/agent in the choice experiment.

    Each agent has:
    - Unique identifier
    - Demographics (age, education, income, etc.)
    - Latent variables (patriotism, secularism dimensions)
    - Taste parameters (individual-specific coefficients)
    - Likert responses (observed indicators of latent variables)
    """

    agent_id: int
    agent_id_str: str
    demographics: Dict[str, int]
    latent_variables: Dict[str, float]
    taste_parameters: Dict[str, float]
    likert_responses: Dict[str, int] = field(default_factory=dict)

    def get_covariate(self, name: str) -> float:
        """Get demographic or latent variable value by name."""
        if name in self.demographics:
            return float(self.demographics[name])
        elif name in self.latent_variables:
            return float(self.latent_variables[name])
        else:
            raise KeyError(f"Covariate '{name}' not found in demographics or latent variables")

    def to_dict(self) -> Dict[str, Any]:
        """Convert agent to dictionary for DataFrame export."""
        result = {
            'ID': self.agent_id,
            'ID_STR': self.agent_id_str,
        }
        result.update(self.demographics)
        result.update(self.likert_responses)
        return result

    def get_true_latents(self) -> Dict[str, float]:
        """Return true latent variable values (for validation)."""
        return {f'LV_{k}_true': v for k, v in self.latent_variables.items()}


# =============================================================================
# POPULATION CLASS
# =============================================================================

class Population:
    """
    Manages the creation and storage of agents (simulated respondents).

    Draws demographics from specified distributions and computes latent
    variables using structural equations.
    """

    def __init__(self, config: Dict, rng: np.random.Generator):
        """
        Initialize population with configuration.

        Args:
            config: Configuration dictionary with demographics, latent specs
            rng: NumPy random generator for reproducibility
        """
        self.config = config
        self.rng = rng
        self.agents: List[Agent] = []
        self.n_agents = config['population']['N']

    def create_agents(self) -> None:
        """Generate all agents in the population."""
        print(f"Creating {self.n_agents} agents...")

        for i in range(1, self.n_agents + 1):
            agent = self._create_single_agent(i)
            self.agents.append(agent)

        print(f"Created {len(self.agents)} agents successfully.")

    def _create_single_agent(self, agent_id: int) -> Agent:
        """Create a single agent with all attributes."""

        # 1. Draw demographics
        demographics = self._draw_demographics()

        # 2. Compute latent variables from structural model
        latent_vars = self._compute_latent_variables(demographics)

        # 3. Compute individual taste parameters
        taste_params = self._compute_taste_parameters(demographics, latent_vars)

        # 4. Generate Likert responses (measurement model)
        likert_responses = self._generate_likert_responses(latent_vars)

        return Agent(
            agent_id=agent_id,
            agent_id_str=f"SYN_{agent_id:06d}",
            demographics=demographics,
            latent_variables=latent_vars,
            taste_parameters=taste_params,
            likert_responses=likert_responses
        )

    def _draw_demographics(self) -> Dict[str, int]:
        """Draw demographic values from population distributions."""
        demographics = {}

        for var_name, spec in self.config['demographics'].items():
            if spec['type'] == 'categorical':
                values = np.array(spec['values'])
                probs = np.array(spec['probs'], dtype=float)
                probs = probs / probs.sum()  # Normalize
                demographics[var_name] = int(self.rng.choice(values, p=probs))
            else:
                raise ValueError(f"Unknown demographic type: {spec['type']}")

        return demographics

    def _compute_latent_variables(self, demographics: Dict[str, int]) -> Dict[str, float]:
        """
        Compute latent variables using structural equations.

        LV = intercept + sum(beta_i * demographic_i) + epsilon
        where epsilon ~ N(0, sigma^2)
        """
        latent_vars = {}
        structural = self.config['latent']['structural']

        for lv_name, spec in structural.items():
            # Start with intercept
            mu = float(spec.get('intercept', 0.0))

            # Add demographic effects
            for demo_var, beta in spec.get('betas', {}).items():
                if demo_var in demographics:
                    mu += float(beta) * float(demographics[demo_var])

            # Add random error
            sigma = float(spec.get('sigma', 1.0))
            epsilon = sigma * self.rng.standard_normal()

            latent_vars[lv_name] = mu + epsilon

        return latent_vars

    def _compute_taste_parameters(
        self,
        demographics: Dict[str, int],
        latent_vars: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Compute individual-specific taste parameters (betas).

        For each attribute term:
        beta_individual = base_coef + sum(interaction_coef * covariate)
        """
        taste_params = {}
        choice_cfg = self.config['choice_model']

        for term_spec in choice_cfg.get('attribute_terms', []):
            term_name = term_spec['name']

            # Start with base coefficient
            beta = float(term_spec.get('base_coef', 0.0))

            # Add interaction effects
            for interaction in term_spec.get('interactions', []):
                cov_name = interaction['with']
                coef = float(interaction.get('coef', 0.0))

                # Apply sign enforcement to interaction coefficient
                if interaction.get('enforce_sign') == 'positive':
                    coef = abs(coef)
                elif interaction.get('enforce_sign') == 'negative':
                    coef = -abs(coef)

                # Get covariate value (from demographics or latents)
                if cov_name in demographics:
                    z = float(demographics[cov_name])
                elif cov_name in latent_vars:
                    z = float(latent_vars[cov_name])
                else:
                    raise ValueError(f"Covariate '{cov_name}' not found")

                # Apply centering and scaling
                if 'center' in interaction and interaction['center'] is not None:
                    z -= float(interaction['center'])
                if 'scale' in interaction and interaction['scale'] not in (None, 0, 0.0):
                    z /= float(interaction['scale'])

                beta += coef * z

            # Apply sign enforcement to total beta
            enforce = term_spec.get('enforce_sign')
            if enforce == 'positive':
                beta = abs(beta)
            elif enforce == 'negative':
                beta = -abs(beta)

            taste_params[term_name] = beta

        return taste_params

    def _generate_likert_responses(self, latent_vars: Dict[str, float]) -> Dict[str, int]:
        """
        Generate Likert scale responses using ordinal probit measurement model.

        y* = loading * LV + epsilon, epsilon ~ N(0,1)
        y = k if tau_{k-1} < y* <= tau_k
        """
        if 'measurement' not in self.config:
            return {}

        meas_cfg = self.config['measurement']
        thresholds = tuple(meas_cfg.get('thresholds', [-1.5, -0.5, 0.5, 1.5]))

        # Load items configuration if available
        items_path = meas_cfg.get('items_path')
        if not items_path or not Path(items_path).exists():
            return {}

        items_df = pd.read_csv(items_path)
        responses = {}

        for _, item in items_df.iterrows():
            lv_name = self._factor_to_lv_name(item['factor'])
            lv_value = latent_vars.get(lv_name, 0.0)

            loading = float(item['loading'])
            y_star = loading * lv_value + self.rng.standard_normal()

            # Convert to ordinal scale
            response = self._ordinal_from_continuous(y_star, thresholds)

            # Apply reverse coding if needed
            if int(item.get('reverse', 0)) == 1:
                response = 6 - response

            responses[str(item['item_name'])] = response

        return responses

    @staticmethod
    def _ordinal_from_continuous(y_star: float, thresholds: Tuple[float, ...]) -> int:
        """Convert continuous latent response to ordinal scale."""
        t1, t2, t3, t4 = thresholds
        if y_star <= t1: return 1
        if y_star <= t2: return 2
        if y_star <= t3: return 3
        if y_star <= t4: return 4
        return 5

    @staticmethod
    def _factor_to_lv_name(factor: str) -> str:
        """Map factor name from items config to latent variable name."""
        fac = str(factor).strip().lower()
        mapping = {
            'blind': 'pat_blind',
            'constructive': 'pat_constructive',
            'daily': 'sec_dl',
            'dl': 'sec_dl',
            'dailylife': 'sec_dl',
            'faith': 'sec_fp',
            'fp': 'sec_fp',
            'faithandprayer': 'sec_fp',
            'faith_prayer': 'sec_fp',
        }
        if fac in mapping:
            return mapping[fac]
        raise ValueError(f"Unknown factor '{factor}' in items_config")

    def get_agent(self, agent_id: int) -> Agent:
        """Retrieve agent by ID (1-indexed)."""
        return self.agents[agent_id - 1]

    def __len__(self) -> int:
        return len(self.agents)

    def __iter__(self):
        return iter(self.agents)


# =============================================================================
# CHOICE MODEL CLASS
# =============================================================================

class ChoiceModel:
    """
    Implements the discrete choice model for utility calculation and choice simulation.

    Supports:
    - Alternative-specific constants (ASCs)
    - Attribute effects with individual-specific coefficients
    - Logit choice probabilities (softmax)
    """

    def __init__(self, config: Dict, rng: np.random.Generator):
        """
        Initialize choice model.

        Args:
            config: Configuration dictionary with choice model specs
            rng: NumPy random generator
        """
        self.config = config['choice_model']
        self.rng = rng
        self.fee_scale = float(self.config.get('fee_scale', 10000.0))
        self.alternatives = self.config['alts']  # {"1": "paid1", "2": "paid2", "3": "standard"}

    def compute_utility(
        self,
        agent: Agent,
        scenario: pd.Series,
        alternative: str
    ) -> float:
        """
        Compute deterministic utility for an alternative.

        U = ASC + sum(beta_k * x_k)

        Args:
            agent: Agent making the choice
            scenario: Row from design matrix with attribute values
            alternative: Alternative name ('paid1', 'paid2', 'standard')

        Returns:
            Deterministic utility value
        """
        U = 0.0

        # 1. Add base terms (ASCs)
        for bt in self.config.get('base_terms', []):
            if alternative in bt['apply_to']:
                coef = float(bt['coef'])
                x = self._get_term_value(bt['term'], scenario, alternative)
                U += coef * x

        # 2. Add attribute terms with individual-specific betas
        for at in self.config.get('attribute_terms', []):
            if alternative not in at['apply_to']:
                continue

            term_name = at['name']
            x = self._get_term_value(at['term'], scenario, alternative)
            beta = agent.taste_parameters.get(term_name, 0.0)
            U += beta * x

        return U

    def _get_term_value(self, term: str, scenario: pd.Series, alternative: str) -> float:
        """
        Get attribute value for a term and alternative.

        Args:
            term: Term name ('const', 'fee10k', 'dur')
            scenario: Scenario row with attribute values
            alternative: Alternative name

        Returns:
            Attribute value (x)
        """
        # Map alternative to column suffix
        alt_map = {'paid1': '1', 'paid2': '2', 'standard': '3'}
        suffix = alt_map.get(alternative, '3')

        if term == 'const':
            return 1.0

        # Fee terms with various scaling conventions
        if term.startswith('fee'):
            fee = float(scenario[f'fee{suffix}'])
            if term == 'fee10k':
                return fee / 10000.0
            elif term == 'fee100k':
                return fee / 100000.0
            else:
                return fee / self.fee_scale

        if term == 'dur':
            return float(scenario[f'dur{suffix}'])

        raise ValueError(f"Unknown term: {term}")

    def get_choice_probabilities(self, utilities: List[float]) -> np.ndarray:
        """
        Compute choice probabilities using softmax (logit model).

        P(j) = exp(U_j) / sum_k(exp(U_k))

        Args:
            utilities: List of utility values for each alternative

        Returns:
            Array of choice probabilities
        """
        u = np.array(utilities)
        u = u - np.max(u)  # Numerical stability
        exp_u = np.exp(u)
        return exp_u / exp_u.sum()

    def simulate_choice(self, agent: Agent, scenario: pd.Series) -> Tuple[int, np.ndarray]:
        """
        Simulate a choice for an agent facing a scenario.

        Args:
            agent: Agent making the choice
            scenario: Scenario with attribute values

        Returns:
            Tuple of (chosen alternative 1-3, probabilities array)
        """
        # Compute utilities for all alternatives
        utilities = []
        for alt_code in ['1', '2', '3']:
            alt_name = self.alternatives[alt_code]
            U = self.compute_utility(agent, scenario, alt_name)
            utilities.append(U)

        # Get choice probabilities
        probs = self.get_choice_probabilities(utilities)

        # Draw choice
        choice = int(self.rng.choice([1, 2, 3], p=probs))

        return choice, probs


# =============================================================================
# SIMULATOR CLASS
# =============================================================================

class DCMSimulator:
    """
    Main orchestrator for the agent-based discrete choice simulation.

    Coordinates:
    - Population creation
    - Scenario assignment
    - Choice simulation
    - Data export
    """

    def __init__(self, config_path: str):
        """
        Initialize simulator with configuration file.

        Args:
            config_path: Path to JSON configuration file
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)

        # Initialize random generator
        seed = int(self.config['population']['seed'])
        self.rng = np.random.default_rng(seed)

        # Initialize components
        self.population = Population(self.config, self.rng)
        self.choice_model = ChoiceModel(self.config, self.rng)

        # Load design matrix
        self.design = self._load_design()

        # Storage for results
        self.results: List[Dict] = []

        # Simulation parameters
        self.n_tasks = int(self.config['population']['T'])

    def _load_config(self, path: str) -> Dict:
        """Load and validate configuration file."""
        with open(path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # Validate required sections
        required = ['population', 'design', 'demographics', 'latent', 'choice_model']
        missing = [k for k in required if k not in config]
        if missing:
            raise ValueError(f"Config missing required sections: {missing}")

        return config

    def _load_design(self) -> pd.DataFrame:
        """Load experimental design (scenarios) from CSV."""
        design_path = self.config['design']['path']

        if not Path(design_path).exists():
            raise FileNotFoundError(f"Design file not found: {design_path}")

        df = pd.read_csv(design_path)

        # Required columns
        required_cols = [
            'scenario_id',
            'dur1', 'fee1', 'dur2', 'fee2', 'dur3', 'fee3'
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Design file missing columns: {missing}")

        # Get unique scenarios
        design = df[required_cols].drop_duplicates('scenario_id').reset_index(drop=True)

        print(f"Loaded {len(design)} unique scenarios from {design_path}")
        return design

    def run(self, keep_latent: bool = False) -> pd.DataFrame:
        """
        Run the full simulation.

        Args:
            keep_latent: If True, include true latent values in output

        Returns:
            DataFrame with simulated choice data
        """
        print("\n" + "="*60)
        print("AGENT-BASED DCM SIMULATION")
        print("="*60)

        # Create population
        self.population.create_agents()

        # Validate we have enough scenarios
        if len(self.design) < self.n_tasks:
            raise ValueError(
                f"Need at least {self.n_tasks} scenarios, but only {len(self.design)} available"
            )

        # Simulate choices for each agent
        print(f"\nSimulating {self.n_tasks} choice tasks per agent...")

        sample_replace = not self.config['design'].get('sample_without_replacement', True)

        for agent in self.population:
            self._simulate_agent_choices(agent, keep_latent, sample_replace)

        # Convert to DataFrame
        results_df = pd.DataFrame(self.results)

        # Print summary
        self._print_summary(results_df)

        return results_df

    def _simulate_agent_choices(
        self,
        agent: Agent,
        keep_latent: bool,
        replace: bool
    ) -> None:
        """Simulate all choice tasks for a single agent."""

        # Sample scenarios for this agent
        sampled_scenarios = self.design.sample(
            n=self.n_tasks,
            replace=replace,
            random_state=int(self.rng.integers(1, 1_000_000_000))
        )

        for task_num, (_, scenario) in enumerate(sampled_scenarios.iterrows(), start=1):
            # Simulate choice
            choice, probs = self.choice_model.simulate_choice(agent, scenario)

            # Build result record
            record = self._build_record(agent, scenario, task_num, choice, keep_latent)
            self.results.append(record)

    def _build_record(
        self,
        agent: Agent,
        scenario: pd.Series,
        task: int,
        choice: int,
        keep_latent: bool
    ) -> Dict:
        """Build a single output record."""
        record = {
            'ID_STR': agent.agent_id_str,
            'ID': agent.agent_id,
            'task': task,
            'scenario_id': int(scenario['scenario_id']),
            'dur1': float(scenario['dur1']),
            'fee1': float(scenario['fee1']),
            'dur2': float(scenario['dur2']),
            'fee2': float(scenario['fee2']),
            'dur3': float(scenario['dur3']),
            'fee3': float(scenario['fee3']),
            'CHOICE': choice,
        }

        # Add demographics
        record.update(agent.demographics)

        # Add Likert responses
        record.update(agent.likert_responses)

        # Optionally add true latent values
        if keep_latent:
            record.update(agent.get_true_latents())
            # Also add taste parameters for validation
            for k, v in agent.taste_parameters.items():
                record[f'beta_{k}_true'] = v

        return record

    def _print_summary(self, df: pd.DataFrame) -> None:
        """Print simulation summary statistics."""
        print("\n" + "="*60)
        print("SIMULATION COMPLETE")
        print("="*60)
        print(f"Total observations: {len(df):,}")
        print(f"Unique respondents: {df['ID'].nunique():,}")
        print(f"Tasks per respondent: {self.n_tasks}")
        print(f"\nChoice shares:")
        shares = df['CHOICE'].value_counts(normalize=True).sort_index()
        for choice, share in shares.items():
            alt_name = self.choice_model.alternatives[str(choice)]
            print(f"  Alternative {choice} ({alt_name}): {share:.1%}")

    def export(self, output_path: str, keep_latent: bool = False) -> None:
        """
        Run simulation and export to CSV.

        Args:
            output_path: Path for output CSV file
            keep_latent: If True, include true latent values
        """
        df = self.run(keep_latent=keep_latent)
        df.to_csv(output_path, index=False)
        print(f"\nExported to: {output_path}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Command-line interface for the simulator."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Agent-Based Discrete Choice Model Simulator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python dcm_simulator.py --config model_config.json --out simulated_data.csv
  python dcm_simulator.py --config model_config.json --out simulated_data.csv --keep_latent
        """
    )
    parser.add_argument(
        '--config',
        required=True,
        help='Path to JSON configuration file'
    )
    parser.add_argument(
        '--out',
        default='synthetic_dcm_data.csv',
        help='Output CSV file path (default: synthetic_dcm_data.csv)'
    )
    parser.add_argument(
        '--keep_latent',
        action='store_true',
        help='Include true latent variables and taste parameters in output'
    )

    args = parser.parse_args()

    # Run simulation
    simulator = DCMSimulator(args.config)
    simulator.export(args.out, keep_latent=args.keep_latent)


if __name__ == '__main__':
    main()
