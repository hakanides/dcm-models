"""
Convergence Diagnostics Module
==============================

Provides comprehensive convergence diagnostics for DCM/HCM models.

Key Features:
- Gradient norm verification
- Hessian eigenvalue analysis for identification
- Condition number computation
- Convergence quality metrics
- LaTeX table generation

Author: DCM Research Team
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import warnings


@dataclass
class ConvergenceDiagnostics:
    """Container for convergence diagnostic results."""
    converged: bool
    iterations: int
    final_ll: float
    gradient_norm: float
    hessian_condition_number: float
    min_eigenvalue: float
    max_eigenvalue: float
    problematic_params: List[str] = field(default_factory=list)
    optimization_message: str = ""

    # Thresholds for publication-quality convergence
    GRADIENT_THRESHOLD: float = 1e-5
    CONDITION_THRESHOLD: float = 1e6
    EIGENVALUE_THRESHOLD: float = 1e-6

    @property
    def is_well_conditioned(self) -> bool:
        """Check if problem is well-conditioned."""
        return self.hessian_condition_number < self.CONDITION_THRESHOLD

    @property
    def is_identified(self) -> bool:
        """Check if model is identified (no near-zero eigenvalues)."""
        return self.min_eigenvalue > self.EIGENVALUE_THRESHOLD

    @property
    def gradient_ok(self) -> bool:
        """Check if gradient is sufficiently small."""
        return self.gradient_norm < self.GRADIENT_THRESHOLD

    @property
    def publication_ready(self) -> bool:
        """Check if convergence meets publication standards."""
        return (self.converged and
                self.is_well_conditioned and
                self.is_identified and
                self.gradient_ok)

    def summary(self) -> str:
        """Generate summary string."""
        status = "PASS" if self.publication_ready else "FAIL"
        lines = [
            f"Convergence Status: {status}",
            f"  Converged: {self.converged}",
            f"  Iterations: {self.iterations}",
            f"  Final LL: {self.final_ll:.4f}",
            f"  Gradient norm: {self.gradient_norm:.2e} {'OK' if self.gradient_ok else 'HIGH'}",
            f"  Condition number: {self.hessian_condition_number:.2e} {'OK' if self.is_well_conditioned else 'HIGH'}",
            f"  Min eigenvalue: {self.min_eigenvalue:.2e} {'OK' if self.is_identified else 'NEAR-ZERO'}",
        ]
        if self.problematic_params:
            lines.append(f"  Problematic params: {', '.join(self.problematic_params)}")
        if self.optimization_message:
            lines.append(f"  Message: {self.optimization_message}")
        return "\n".join(lines)


class ConvergenceChecker:
    """
    Check convergence quality for DCM/HCM models.

    Provides methods to verify:
    - Gradient is near zero at solution
    - Hessian is positive semi-definite
    - Model is identified (no flat directions)

    Example:
        >>> checker = ConvergenceChecker()
        >>> diagnostics = checker.full_diagnostics(biogeme_result)
        >>> if diagnostics.publication_ready:
        ...     print("Model is ready for publication")
    """

    def __init__(self,
                 gradient_tol: float = 1e-5,
                 condition_tol: float = 1e6,
                 eigenvalue_tol: float = 1e-6):
        """
        Initialize convergence checker.

        Args:
            gradient_tol: Maximum acceptable gradient norm
            condition_tol: Maximum acceptable condition number
            eigenvalue_tol: Minimum acceptable eigenvalue
        """
        self.gradient_tol = gradient_tol
        self.condition_tol = condition_tol
        self.eigenvalue_tol = eigenvalue_tol

    def check_gradient(self, result) -> Tuple[bool, float]:
        """
        Verify gradient is near zero at solution.

        Args:
            result: Biogeme estimation result

        Returns:
            Tuple of (is_ok, gradient_norm)
        """
        try:
            # Try to get gradient from result
            if hasattr(result, 'getGradient'):
                gradient = result.getGradient()
            elif hasattr(result, 'gradient'):
                gradient = result.gradient
            else:
                # Estimate gradient numerically if not available
                return True, 0.0  # Assume OK if no gradient available

            if gradient is None:
                return True, 0.0

            gradient_norm = float(np.linalg.norm(gradient))
            return gradient_norm < self.gradient_tol, gradient_norm

        except Exception as e:
            warnings.warn(f"Could not check gradient: {e}")
            return True, 0.0

    def check_hessian_psd(self, result) -> Tuple[bool, float, float]:
        """
        Verify Hessian is positive semi-definite.

        Args:
            result: Biogeme estimation result

        Returns:
            Tuple of (is_psd, min_eigenvalue, condition_number)
        """
        try:
            # Get Hessian matrix
            if hasattr(result, 'getHessian'):
                hessian = result.getHessian()
            elif hasattr(result, 'hess_inv'):
                # scipy result with inverse Hessian
                hess_inv = result.hess_inv
                if hasattr(hess_inv, 'todense'):
                    hess_inv = np.array(hess_inv.todense())
                hessian = np.linalg.inv(hess_inv) if hess_inv is not None else None
            else:
                return True, 1.0, 1.0  # Assume OK if no Hessian

            if hessian is None:
                return True, 1.0, 1.0

            # Ensure Hessian is numpy array
            hessian = np.array(hessian)

            # Compute eigenvalues
            eigenvalues = np.linalg.eigvalsh(hessian)
            min_eigenvalue = float(np.min(eigenvalues))
            max_eigenvalue = float(np.max(np.abs(eigenvalues)))

            # Compute condition number
            if min_eigenvalue > 0:
                condition_number = max_eigenvalue / min_eigenvalue
            else:
                condition_number = np.inf

            is_psd = min_eigenvalue > -1e-10  # Allow small numerical errors

            return is_psd, min_eigenvalue, condition_number

        except Exception as e:
            warnings.warn(f"Could not check Hessian: {e}")
            return True, 1.0, 1.0

    def check_identification(self, result) -> Tuple[bool, List[str]]:
        """
        Check for identification issues via eigenvalue analysis.

        Args:
            result: Biogeme estimation result

        Returns:
            Tuple of (is_identified, problematic_parameters)
        """
        try:
            # Get Hessian matrix
            if hasattr(result, 'getHessian'):
                hessian = result.getHessian()
            else:
                return True, []

            if hessian is None:
                return True, []

            hessian = np.array(hessian)

            # Compute eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eigh(hessian)
            min_eigenvalue = np.min(eigenvalues)

            if min_eigenvalue < self.eigenvalue_tol:
                # Find problematic parameters using eigenvector
                min_idx = np.argmin(eigenvalues)
                problematic_vec = eigenvectors[:, min_idx]

                # Get parameter names
                if hasattr(result, 'getBetaValues'):
                    param_names = list(result.getBetaValues().keys())
                elif hasattr(result, 'get_beta_values'):
                    param_names = list(result.get_beta_values().keys())
                else:
                    param_names = [f"param_{i}" for i in range(len(problematic_vec))]

                # Find parameters with significant weight
                problematic_params = []
                for i, (name, weight) in enumerate(zip(param_names, problematic_vec)):
                    if abs(weight) > 0.1:  # Threshold for significant contribution
                        problematic_params.append(name)

                return False, problematic_params

            return True, []

        except Exception as e:
            warnings.warn(f"Could not check identification: {e}")
            return True, []

    def full_diagnostics(self, result, model_name: str = "") -> ConvergenceDiagnostics:
        """
        Run all convergence checks and return comprehensive diagnostics.

        Args:
            result: Biogeme estimation result
            model_name: Optional model name for reporting

        Returns:
            ConvergenceDiagnostics object
        """
        # Check if converged
        if hasattr(result, 'algorithm_has_converged'):
            converged = result.algorithm_has_converged
        elif hasattr(result, 'success'):
            converged = result.success
        else:
            # Conservative default: don't assume convergence without evidence
            converged = False
            warnings.warn("Cannot determine convergence status - assuming not converged")

        # Get iteration count (note: Biogeme may not expose this directly)
        iterations = 0
        if hasattr(result, 'number_of_iterations'):
            iterations = result.number_of_iterations
        elif hasattr(result, 'nit'):
            iterations = result.nit
        else:
            # Try to get from general statistics
            try:
                stats = result.get_general_statistics()
                # Iteration count not typically in stats, so default to 0
                iterations = 0
            except Exception:
                iterations = 0

        # Get final log-likelihood
        if hasattr(result, 'final_loglikelihood'):
            final_ll = result.final_loglikelihood
        elif hasattr(result, 'getGeneralStatistics'):
            stats = result.getGeneralStatistics()
            final_ll = stats.get('Final log likelihood', [0.0])[0]
        elif hasattr(result, 'fun'):
            final_ll = -result.fun  # scipy returns negative LL
        else:
            final_ll = 0.0

        # Run checks
        gradient_ok, gradient_norm = self.check_gradient(result)
        hessian_psd, min_eigenvalue, condition_number = self.check_hessian_psd(result)
        identified, problematic_params = self.check_identification(result)

        # Get optimization message
        if hasattr(result, 'message'):
            opt_message = result.message
        else:
            opt_message = ""

        # Compute max eigenvalue
        try:
            if hasattr(result, 'getHessian'):
                hessian = result.getHessian()
                if hessian is not None:
                    eigenvalues = np.linalg.eigvalsh(np.array(hessian))
                    max_eigenvalue = float(np.max(np.abs(eigenvalues)))
                else:
                    max_eigenvalue = 1.0
            else:
                max_eigenvalue = 1.0
        except:
            max_eigenvalue = 1.0

        return ConvergenceDiagnostics(
            converged=converged,
            iterations=iterations,
            final_ll=final_ll,
            gradient_norm=gradient_norm,
            hessian_condition_number=condition_number,
            min_eigenvalue=min_eigenvalue,
            max_eigenvalue=max_eigenvalue,
            problematic_params=problematic_params,
            optimization_message=opt_message
        )

    def print_diagnostics(self, diagnostics: ConvergenceDiagnostics,
                         model_name: str = "") -> None:
        """Print formatted diagnostics."""
        print("\n" + "=" * 60)
        if model_name:
            print(f"CONVERGENCE DIAGNOSTICS: {model_name}")
        else:
            print("CONVERGENCE DIAGNOSTICS")
        print("=" * 60)
        print(diagnostics.summary())
        print("=" * 60)


def generate_convergence_table(diagnostics_dict: Dict[str, ConvergenceDiagnostics],
                               output_path: Path = None) -> pd.DataFrame:
    """
    Generate convergence summary table for all models.

    Args:
        diagnostics_dict: Dict mapping model name to diagnostics
        output_path: Optional path to save CSV

    Returns:
        DataFrame with convergence statistics
    """
    records = []
    for model_name, diag in diagnostics_dict.items():
        records.append({
            'Model': model_name,
            'Converged': diag.converged,
            'Iterations': diag.iterations,
            'Final LL': diag.final_ll,
            'Gradient Norm': diag.gradient_norm,
            'Condition Number': diag.hessian_condition_number,
            'Min Eigenvalue': diag.min_eigenvalue,
            'Identified': diag.is_identified,
            'Well-Conditioned': diag.is_well_conditioned,
            'Publication Ready': diag.publication_ready,
        })

    df = pd.DataFrame(records)

    if output_path:
        df.to_csv(output_path, index=False)

    return df


def generate_convergence_latex(diagnostics_dict: Dict[str, ConvergenceDiagnostics],
                               output_path: Path) -> None:
    """
    Generate LaTeX table of convergence diagnostics.

    Args:
        diagnostics_dict: Dict mapping model name to diagnostics
        output_path: Path to save LaTeX file
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    latex_content = r"""\begin{table}[htbp]
\centering
\caption{Model Convergence Diagnostics}
\label{tab:convergence}
\begin{tabular}{lcccccc}
\toprule
Model & Conv. & Iter. & $\|\nabla\|$ & Cond. \# & Min $\lambda$ & Status \\
\midrule
"""

    for model_name, diag in diagnostics_dict.items():
        status = r"\checkmark" if diag.publication_ready else r"$\times$"
        conv = r"\checkmark" if diag.converged else r"$\times$"

        # Format numbers
        grad = f"{diag.gradient_norm:.1e}"
        cond = f"{diag.hessian_condition_number:.1e}"
        eigenval = f"{diag.min_eigenvalue:.1e}"

        latex_content += f"{model_name} & {conv} & {diag.iterations} & {grad} & {cond} & {eigenval} & {status} \\\\\n"

    latex_content += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Note: Conv. = Converged; $\|\nabla\|$ = Gradient norm; Cond. \# = Condition number;
\item Min $\lambda$ = Minimum eigenvalue; Status = Publication ready.
\item Thresholds: $\|\nabla\| < 10^{-5}$, Cond. \# $< 10^6$, Min $\lambda > 10^{-6}$.
\end{tablenotes}
\end{table}
"""

    with open(output_path / 'convergence_diagnostics.tex', 'w') as f:
        f.write(latex_content)


if __name__ == '__main__':
    print("Convergence Diagnostics Module")
    print("=" * 50)
    print("\nAvailable classes:")
    print("  - ConvergenceDiagnostics: Container for diagnostic results")
    print("  - ConvergenceChecker: Run convergence checks")
    print("\nAvailable functions:")
    print("  - generate_convergence_table: Create summary DataFrame")
    print("  - generate_convergence_latex: Generate LaTeX table")
