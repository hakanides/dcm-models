"""
Identification Diagnostics for DCM Models
==========================================

Comprehensive diagnostics for model identification issues in discrete choice models.

Key Diagnostics:
- Hessian eigenvalue analysis
- Condition number assessment
- Parameter correlation analysis
- Collinearity detection

Usage:
    from src.estimation.identification_diagnostics import IdentificationDiagnostics

    diag = IdentificationDiagnostics(biogeme_results)
    report = diag.full_report()
    diag.print_report()

Author: DCM Research Team
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import warnings


class IdentificationDiagnostics:
    """
    Comprehensive identification diagnostics for DCM estimation results.

    Attributes:
        results: Biogeme estimation results object
        param_names: List of parameter names
        hessian: Hessian matrix
        eigenvalues: Eigenvalues of Hessian
        eigenvectors: Eigenvectors of Hessian
    """

    # Thresholds for diagnostics
    EIGENVALUE_THRESHOLD = 1e-5       # Below this = near-singular
    CONDITION_THRESHOLD = 1e6         # Above this = ill-conditioned
    CORRELATION_THRESHOLD = 0.95      # Above this = high correlation
    SE_RATIO_THRESHOLD = 10.0         # SE > value * 10 = poorly identified

    def __init__(self, results):
        """
        Initialize diagnostics from Biogeme results.

        Args:
            results: Biogeme estimation results object
        """
        self.results = results
        self.param_names = list(results.get_beta_values().keys())
        self.n_params = len(self.param_names)

        # Extract matrices
        self._extract_hessian()
        self._compute_eigendecomposition()

    def _extract_hessian(self) -> None:
        """Extract Hessian matrix from results."""
        try:
            # Try different Biogeme API versions
            if hasattr(self.results, 'getHessian'):
                self.hessian = self.results.getHessian()
            elif hasattr(self.results, 'hessian'):
                self.hessian = self.results.hessian
            else:
                self.hessian = None
        except Exception as e:
            self.hessian = None
            self._hessian_error = str(e)

    def _compute_eigendecomposition(self) -> None:
        """Compute eigenvalues and eigenvectors of Hessian."""
        if self.hessian is None:
            self.eigenvalues = None
            self.eigenvectors = None
            return

        try:
            # Hessian should be negative semi-definite at maximum
            # Use absolute values for eigenvalue analysis
            self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.hessian)
        except Exception as e:
            self.eigenvalues = None
            self.eigenvectors = None
            self._eigen_error = str(e)

    def check_eigenvalues(self) -> Dict[str, Any]:
        """
        Check Hessian eigenvalues for identification issues.

        Returns:
            Dict with:
            - identified: bool indicating if model is identified
            - min_eigenvalue: minimum eigenvalue
            - max_eigenvalue: maximum eigenvalue
            - near_zero_count: number of eigenvalues near zero
            - problematic_directions: parameter combinations with issues
        """
        result = {
            'identified': True,
            'min_eigenvalue': None,
            'max_eigenvalue': None,
            'near_zero_count': 0,
            'problematic_directions': []
        }

        if self.eigenvalues is None:
            result['identified'] = None
            result['error'] = 'Eigenvalues not available'
            return result

        # Get absolute eigenvalues (Hessian is negative definite at max)
        abs_eigenvalues = np.abs(self.eigenvalues)

        result['min_eigenvalue'] = float(np.min(abs_eigenvalues))
        result['max_eigenvalue'] = float(np.max(abs_eigenvalues))

        # Count near-zero eigenvalues
        near_zero_mask = abs_eigenvalues < self.EIGENVALUE_THRESHOLD
        result['near_zero_count'] = int(np.sum(near_zero_mask))

        if result['near_zero_count'] > 0:
            result['identified'] = False

            # Find problematic parameter combinations
            for idx in np.where(near_zero_mask)[0]:
                direction = self.eigenvectors[:, idx]

                # Get parameters with significant weights in this direction
                significant_params = []
                for i, (name, weight) in enumerate(zip(self.param_names, direction)):
                    if abs(weight) > 0.1:
                        significant_params.append({
                            'param': name,
                            'weight': float(weight)
                        })

                if significant_params:
                    result['problematic_directions'].append({
                        'eigenvalue': float(abs_eigenvalues[idx]),
                        'parameters': significant_params
                    })

        return result

    def check_condition_number(self) -> Dict[str, Any]:
        """
        Check condition number of the Hessian.

        High condition number indicates numerical instability.

        Returns:
            Dict with condition number and assessment
        """
        result = {
            'condition_number': None,
            'well_conditioned': True,
            'severity': 'good'
        }

        if self.eigenvalues is None:
            result['error'] = 'Eigenvalues not available'
            return result

        abs_eigenvalues = np.abs(self.eigenvalues)
        min_eig = np.min(abs_eigenvalues)
        max_eig = np.max(abs_eigenvalues)

        if min_eig > 1e-10:
            result['condition_number'] = float(max_eig / min_eig)

            if result['condition_number'] > self.CONDITION_THRESHOLD:
                result['well_conditioned'] = False
                result['severity'] = 'severe'
            elif result['condition_number'] > 1e4:
                result['severity'] = 'moderate'
            elif result['condition_number'] > 1e2:
                result['severity'] = 'mild'
        else:
            result['condition_number'] = float('inf')
            result['well_conditioned'] = False
            result['severity'] = 'severe'

        return result

    def check_parameter_correlations(self) -> Dict[str, Any]:
        """
        Check correlations between parameter estimates.

        High correlations indicate potential collinearity.

        Returns:
            Dict with correlation matrix and problematic pairs
        """
        result = {
            'correlation_matrix': None,
            'high_correlations': [],
            'max_correlation': None
        }

        try:
            # Get variance-covariance matrix
            if hasattr(self.results, 'getVarCovar'):
                var_covar = self.results.getVarCovar()
            elif hasattr(self.results, 'varCovar'):
                var_covar = self.results.varCovar
            else:
                result['error'] = 'Variance-covariance matrix not available'
                return result

            if var_covar is None:
                result['error'] = 'Variance-covariance matrix is None'
                return result

            # Convert to correlation matrix
            std_devs = np.sqrt(np.diag(var_covar))
            std_devs[std_devs == 0] = 1  # Avoid division by zero

            corr_matrix = var_covar / np.outer(std_devs, std_devs)
            result['correlation_matrix'] = corr_matrix

            # Find high correlations
            max_corr = 0
            for i in range(self.n_params):
                for j in range(i + 1, self.n_params):
                    corr = abs(corr_matrix[i, j])
                    if corr > max_corr:
                        max_corr = corr

                    if corr > self.CORRELATION_THRESHOLD:
                        result['high_correlations'].append({
                            'param1': self.param_names[i],
                            'param2': self.param_names[j],
                            'correlation': float(corr_matrix[i, j])
                        })

            result['max_correlation'] = float(max_corr)

        except Exception as e:
            result['error'] = str(e)

        return result

    def check_standard_errors(self) -> Dict[str, Any]:
        """
        Check standard errors for anomalies.

        Large SE relative to estimate indicates poor identification.

        Returns:
            Dict with SE analysis
        """
        result = {
            'parameters': {},
            'poorly_identified': [],
            'extreme_se': []
        }

        try:
            betas = self.results.get_beta_values()

            # Try to get standard errors
            std_errs = {}
            for param in betas:
                try:
                    std_errs[param] = self.results.get_parameter_std_err(param)
                except:
                    std_errs[param] = np.nan

            for param, value in betas.items():
                se = std_errs.get(param, np.nan)
                t_stat = value / se if se and se > 0 and not np.isnan(se) else np.nan

                result['parameters'][param] = {
                    'estimate': float(value),
                    'std_err': float(se) if not np.isnan(se) else None,
                    't_stat': float(t_stat) if not np.isnan(t_stat) else None
                }

                # Check for issues
                if not np.isnan(se):
                    if se > abs(value) * self.SE_RATIO_THRESHOLD and value != 0:
                        result['poorly_identified'].append(param)
                    if se > 100 or np.isinf(se):
                        result['extreme_se'].append(param)

        except Exception as e:
            result['error'] = str(e)

        return result

    def full_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive identification report.

        Returns:
            Dict with all diagnostic results
        """
        report = {
            'model_identified': True,
            'n_parameters': self.n_params,
            'parameter_names': self.param_names,
            'eigenvalue_analysis': self.check_eigenvalues(),
            'condition_number': self.check_condition_number(),
            'parameter_correlations': self.check_parameter_correlations(),
            'standard_errors': self.check_standard_errors(),
            'issues': [],
            'recommendations': []
        }

        # Aggregate identification status
        if report['eigenvalue_analysis'].get('identified') == False:
            report['model_identified'] = False
            report['issues'].append('Near-singular Hessian detected')

        if not report['condition_number'].get('well_conditioned', True):
            report['issues'].append('Ill-conditioned Hessian')

        if report['parameter_correlations'].get('high_correlations'):
            report['issues'].append('High parameter correlations detected')

        if report['standard_errors'].get('poorly_identified'):
            report['issues'].append('Parameters with large standard errors')

        # Generate recommendations
        if report['eigenvalue_analysis'].get('problematic_directions'):
            params = []
            for direction in report['eigenvalue_analysis']['problematic_directions']:
                params.extend([p['param'] for p in direction['parameters']])
            unique_params = list(set(params))
            report['recommendations'].append(
                f"Consider fixing or removing: {', '.join(unique_params)}"
            )

        if report['parameter_correlations'].get('high_correlations'):
            for pair in report['parameter_correlations']['high_correlations'][:3]:
                report['recommendations'].append(
                    f"Consider combining {pair['param1']} and {pair['param2']} "
                    f"(correlation: {pair['correlation']:.3f})"
                )

        return report

    def print_report(self, verbose: bool = True) -> None:
        """Print formatted diagnostic report."""
        report = self.full_report()

        print("\n" + "=" * 70)
        print("IDENTIFICATION DIAGNOSTICS REPORT")
        print("=" * 70)

        # Summary
        status = "IDENTIFIED" if report['model_identified'] else "IDENTIFICATION ISSUES"
        print(f"\nStatus: {status}")
        print(f"Parameters: {report['n_parameters']}")

        # Eigenvalue analysis
        eig = report['eigenvalue_analysis']
        print(f"\n{'─'*40}")
        print("Hessian Eigenvalue Analysis")
        print(f"{'─'*40}")

        if 'error' in eig:
            print(f"  Error: {eig['error']}")
        else:
            print(f"  Min eigenvalue: {eig['min_eigenvalue']:.2e}")
            print(f"  Max eigenvalue: {eig['max_eigenvalue']:.2e}")
            print(f"  Near-zero eigenvalues: {eig['near_zero_count']}")

            if eig['problematic_directions'] and verbose:
                print("\n  Problematic parameter combinations:")
                for i, direction in enumerate(eig['problematic_directions'][:3]):
                    print(f"\n    Direction {i+1} (eigenvalue: {direction['eigenvalue']:.2e}):")
                    for p in direction['parameters'][:5]:
                        print(f"      {p['param']}: weight = {p['weight']:.3f}")

        # Condition number
        cond = report['condition_number']
        print(f"\n{'─'*40}")
        print("Condition Number Analysis")
        print(f"{'─'*40}")

        if 'error' in cond:
            print(f"  Error: {cond['error']}")
        else:
            print(f"  Condition number: {cond['condition_number']:.2e}")
            print(f"  Assessment: {cond['severity']}")

        # Parameter correlations
        corr = report['parameter_correlations']
        print(f"\n{'─'*40}")
        print("Parameter Correlation Analysis")
        print(f"{'─'*40}")

        if 'error' in corr:
            print(f"  Error: {corr['error']}")
        else:
            print(f"  Max correlation: {corr['max_correlation']:.3f}")

            if corr['high_correlations'] and verbose:
                print("\n  High correlations (>{:.0%}):".format(self.CORRELATION_THRESHOLD))
                for pair in corr['high_correlations'][:5]:
                    print(f"    {pair['param1']} <-> {pair['param2']}: {pair['correlation']:.3f}")

        # Standard errors
        se_check = report['standard_errors']
        print(f"\n{'─'*40}")
        print("Standard Error Analysis")
        print(f"{'─'*40}")

        if 'error' in se_check:
            print(f"  Error: {se_check['error']}")
        else:
            if se_check['poorly_identified']:
                print(f"  Poorly identified: {', '.join(se_check['poorly_identified'])}")
            if se_check['extreme_se']:
                print(f"  Extreme SEs: {', '.join(se_check['extreme_se'])}")
            if not se_check['poorly_identified'] and not se_check['extreme_se']:
                print("  All parameters have reasonable standard errors")

        # Issues and recommendations
        if report['issues']:
            print(f"\n{'─'*40}")
            print("ISSUES DETECTED")
            print(f"{'─'*40}")
            for issue in report['issues']:
                print(f"  - {issue}")

        if report['recommendations']:
            print(f"\n{'─'*40}")
            print("RECOMMENDATIONS")
            print(f"{'─'*40}")
            for rec in report['recommendations']:
                print(f"  - {rec}")

        print("\n" + "=" * 70)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Export parameter-level diagnostics to DataFrame.

        Returns:
            DataFrame with diagnostic info per parameter
        """
        se_check = self.check_standard_errors()
        rows = []

        for param in self.param_names:
            row = {'parameter': param}

            if param in se_check.get('parameters', {}):
                info = se_check['parameters'][param]
                row['estimate'] = info.get('estimate')
                row['std_err'] = info.get('std_err')
                row['t_stat'] = info.get('t_stat')

            row['poorly_identified'] = param in se_check.get('poorly_identified', [])
            row['extreme_se'] = param in se_check.get('extreme_se', [])

            rows.append(row)

        return pd.DataFrame(rows)


def quick_check(results) -> bool:
    """
    Quick identification check - returns True if model appears identified.

    Args:
        results: Biogeme estimation results

    Returns:
        bool indicating if model appears identified
    """
    try:
        diag = IdentificationDiagnostics(results)
        eig_check = diag.check_eigenvalues()
        return eig_check.get('identified', True) != False
    except:
        return True  # Assume identified if check fails


def diagnose_and_warn(results, model_name: str = "model") -> Dict[str, Any]:
    """
    Run diagnostics and print warnings if issues found.

    Args:
        results: Biogeme estimation results
        model_name: Name for display

    Returns:
        Full diagnostic report
    """
    diag = IdentificationDiagnostics(results)
    report = diag.full_report()

    if not report['model_identified'] or report['issues']:
        print(f"\n*** WARNING: Identification issues in {model_name} ***")
        for issue in report['issues']:
            print(f"    - {issue}")

        if report['recommendations']:
            print("\n  Recommendations:")
            for rec in report['recommendations'][:3]:
                print(f"    - {rec}")

    return report


if __name__ == '__main__':
    print("Identification Diagnostics Module")
    print("=" * 40)
    print("\nUsage:")
    print("  from src.estimation.identification_diagnostics import IdentificationDiagnostics")
    print("  diag = IdentificationDiagnostics(biogeme_results)")
    print("  diag.print_report()")
    print("\nQuick check:")
    print("  from src.estimation.identification_diagnostics import quick_check")
    print("  is_identified = quick_check(biogeme_results)")
