"""
Measurement Model Validation for HCM
=====================================

Comprehensive psychometric validation for latent variable constructs.
Required for publication-ready HCM models.

Metrics Implemented:
- Cronbach's Alpha: Internal consistency (threshold > 0.70)
- Factor Loadings: Item quality with standard errors (threshold > 0.50)
- Composite Reliability (CR): Construct reliability (threshold > 0.70)
- Average Variance Extracted (AVE): Convergent validity (threshold > 0.50)
- Fornell-Larcker Criterion: Discriminant validity (sqrt(AVE) > correlations)
- Item-Total Correlations: Item quality assessment

References:
- Hair, J.F. et al. (2019). Multivariate Data Analysis (8th ed.)
- Fornell, C. & Larcker, D.F. (1981). Evaluating SEM with unobserved variables

Author: DCM Research Team
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
from scipy import stats
import warnings

# Make sklearn optional
try:
    from sklearn.decomposition import FactorAnalysis
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    FactorAnalysis = None


@dataclass
class ConstructMetrics:
    """Validation metrics for a single construct."""
    name: str
    n_items: int
    cronbachs_alpha: float
    composite_reliability: float
    average_variance_extracted: float
    factor_loadings: Dict[str, float]
    factor_loadings_se: Dict[str, float]
    item_total_correlations: Dict[str, float]

    # Validation flags
    alpha_valid: bool = field(init=False)
    cr_valid: bool = field(init=False)
    ave_valid: bool = field(init=False)
    all_loadings_valid: bool = field(init=False)

    def __post_init__(self):
        self.alpha_valid = self.cronbachs_alpha >= 0.70
        self.cr_valid = self.composite_reliability >= 0.70
        self.ave_valid = self.average_variance_extracted >= 0.50
        self.all_loadings_valid = all(l >= 0.50 for l in self.factor_loadings.values())

    @property
    def is_valid(self) -> bool:
        """Check if construct meets all validity thresholds."""
        return self.alpha_valid and self.cr_valid and self.ave_valid and self.all_loadings_valid


@dataclass
class MeasurementValidationResult:
    """Complete measurement model validation results."""
    construct_metrics: Dict[str, ConstructMetrics]
    correlation_matrix: pd.DataFrame
    fornell_larcker_matrix: pd.DataFrame
    discriminant_validity_passed: bool
    overall_valid: bool
    warnings: List[str] = field(default_factory=list)


class MeasurementValidator:
    """
    Comprehensive measurement model validation for HCM.

    Validates latent variable constructs using psychometric standards
    required for publication in peer-reviewed journals.

    Example:
        >>> validator = MeasurementValidator(df, constructs={
        ...     'pat_blind': ['pat_blind_1', 'pat_blind_2', 'pat_blind_3', 'pat_blind_4'],
        ...     'pat_const': ['pat_constructive_1', 'pat_constructive_2', ...],
        ... })
        >>> results = validator.validate_all()
        >>> validator.print_report()
        >>> validator.to_latex(Path('output/latex'))
    """

    # Thresholds from Hair et al. (2019)
    ALPHA_THRESHOLD = 0.70
    CR_THRESHOLD = 0.70
    AVE_THRESHOLD = 0.50
    LOADING_THRESHOLD = 0.50

    def __init__(self,
                 df: pd.DataFrame,
                 constructs: Dict[str, List[str]],
                 items_config_path: Optional[str] = None):
        """
        Initialize measurement validator.

        Args:
            df: DataFrame with indicator columns (Likert items)
            constructs: Dict mapping construct name to list of item columns
                       e.g., {'pat_blind': ['pat_blind_1', 'pat_blind_2', ...]}
            items_config_path: Optional path to items configuration CSV
        """
        self.df = df
        self.constructs = constructs
        self.items_config = self._load_items_config(items_config_path)
        self._validate_data()

        # Results storage
        self._results: Optional[MeasurementValidationResult] = None

    def _load_items_config(self, path: Optional[str]) -> Optional[pd.DataFrame]:
        """Load items configuration if provided."""
        if path is None:
            return None
        try:
            return pd.read_csv(path)
        except Exception:
            return None

    def _validate_data(self):
        """Validate input data."""
        # Check all items exist in dataframe
        all_items = [item for items in self.constructs.values() for item in items]
        missing = [item for item in all_items if item not in self.df.columns]
        if missing:
            raise ValueError(f"Missing indicator columns: {missing}")

        # Check for sufficient variance
        for construct, items in self.constructs.items():
            for item in items:
                if self.df[item].std() < 0.01:
                    warnings.warn(f"Item '{item}' has near-zero variance")

    def cronbachs_alpha(self, construct: str) -> float:
        """
        Calculate Cronbach's alpha for internal consistency.

        Formula:
            α = (k / (k-1)) * (1 - Σvar(X_i) / var(X_total))

        where k = number of items

        Threshold: α > 0.70 (Hair et al., 2019)

        Args:
            construct: Name of construct in self.constructs

        Returns:
            Cronbach's alpha coefficient
        """
        items = self.constructs[construct]
        k = len(items)

        if k < 2:
            return np.nan

        X = self.df[items].values
        item_vars = X.var(axis=0, ddof=1)
        total_var = X.sum(axis=1).var(ddof=1)

        if total_var == 0:
            return np.nan

        alpha = (k / (k - 1)) * (1 - item_vars.sum() / total_var)
        return alpha

    def factor_loadings(self, construct: str,
                        n_bootstrap: int = 200) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Estimate standardized factor loadings with bootstrap standard errors.

        Uses maximum likelihood factor analysis with single factor extraction.

        Threshold: λ > 0.50 (Hair et al., 2019)

        Args:
            construct: Name of construct
            n_bootstrap: Number of bootstrap samples for SE estimation

        Returns:
            Tuple of (loadings_dict, se_dict)
        """
        items = self.constructs[construct]
        X = self.df[items].values
        n_samples = len(X)

        # Standardize for comparability
        X_std = (X - X.mean(axis=0)) / X.std(axis=0)

        # Fit single-factor model
        if HAS_SKLEARN:
            try:
                fa = FactorAnalysis(n_components=1, random_state=42)
                fa.fit(X_std)
                loadings = fa.components_[0]
            except Exception:
                # Fallback to correlation-based loadings
                total = X_std.sum(axis=1)
                loadings = np.array([np.corrcoef(X_std[:, i], total)[0, 1]
                                    for i in range(len(items))])
        else:
            # Use correlation-based loadings when sklearn not available
            total = X_std.sum(axis=1)
            loadings = np.array([np.corrcoef(X_std[:, i], total)[0, 1]
                                for i in range(len(items))])

        # Bootstrap standard errors
        bootstrap_loadings = []
        rng = np.random.default_rng(42)

        for _ in range(n_bootstrap):
            idx = rng.choice(n_samples, size=n_samples, replace=True)
            X_boot = X_std[idx]
            if HAS_SKLEARN:
                try:
                    fa_boot = FactorAnalysis(n_components=1, random_state=42)
                    fa_boot.fit(X_boot)
                    bootstrap_loadings.append(fa_boot.components_[0])
                except Exception:
                    continue
            else:
                # Correlation-based fallback for bootstrap
                total_boot = X_boot.sum(axis=1)
                boot_loadings = np.array([np.corrcoef(X_boot[:, i], total_boot)[0, 1]
                                         for i in range(len(items))])
                bootstrap_loadings.append(boot_loadings)

        if bootstrap_loadings:
            boot_array = np.array(bootstrap_loadings)
            se = boot_array.std(axis=0)
        else:
            se = np.zeros(len(items))

        loadings_dict = {item: abs(loadings[i]) for i, item in enumerate(items)}
        se_dict = {item: se[i] for i, item in enumerate(items)}

        return loadings_dict, se_dict

    def item_total_correlations(self, construct: str) -> Dict[str, float]:
        """
        Calculate corrected item-total correlations.

        Corrected: correlation of item with sum of OTHER items (not including itself).

        Threshold: r > 0.30 typically indicates acceptable item quality.

        Args:
            construct: Name of construct

        Returns:
            Dict mapping item name to corrected item-total correlation
        """
        items = self.constructs[construct]
        X = self.df[items].values

        correlations = {}
        for i, item in enumerate(items):
            # Sum of all other items
            other_items = np.delete(X, i, axis=1).sum(axis=1)
            r = np.corrcoef(X[:, i], other_items)[0, 1]
            correlations[item] = r

        return correlations

    def composite_reliability(self, construct: str) -> float:
        """
        Calculate Composite Reliability (CR).

        CR is preferred over Cronbach's alpha as it doesn't assume equal loadings.

        Formula:
            CR = (Σλ)² / [(Σλ)² + Σ(1 - λ²)]

        where λ = standardized factor loadings

        Threshold: CR > 0.70 (Hair et al., 2019)

        Args:
            construct: Name of construct

        Returns:
            Composite reliability coefficient
        """
        loadings, _ = self.factor_loadings(construct, n_bootstrap=0)
        lambdas = np.array(list(loadings.values()))

        sum_lambda = lambdas.sum()
        sum_error = (1 - lambdas**2).sum()

        cr = sum_lambda**2 / (sum_lambda**2 + sum_error)
        return cr

    def average_variance_extracted(self, construct: str) -> float:
        """
        Calculate Average Variance Extracted (AVE).

        AVE measures the amount of variance captured by the construct
        relative to measurement error.

        Formula:
            AVE = Σλ² / [Σλ² + Σ(1 - λ²)]

        Threshold: AVE > 0.50 (Fornell & Larcker, 1981)

        Args:
            construct: Name of construct

        Returns:
            Average variance extracted
        """
        loadings, _ = self.factor_loadings(construct, n_bootstrap=0)
        lambdas = np.array(list(loadings.values()))

        sum_lambda_sq = (lambdas**2).sum()
        sum_error = (1 - lambdas**2).sum()

        ave = sum_lambda_sq / (sum_lambda_sq + sum_error)
        return ave

    def construct_correlation_matrix(self) -> pd.DataFrame:
        """
        Calculate correlation matrix between construct scores.

        Uses weighted sum scores based on factor loadings.

        Returns:
            DataFrame with construct correlations
        """
        scores = {}
        for construct in self.constructs:
            items = self.constructs[construct]
            loadings, _ = self.factor_loadings(construct, n_bootstrap=0)

            # Weighted sum score
            weights = np.array([loadings[item] for item in items])
            weights = weights / weights.sum()

            X = self.df[items].values
            scores[construct] = (X * weights).sum(axis=1)

        scores_df = pd.DataFrame(scores)
        return scores_df.corr()

    def fornell_larcker_test(self) -> Tuple[pd.DataFrame, bool]:
        """
        Perform Fornell-Larcker criterion test for discriminant validity.

        Criterion: sqrt(AVE) for each construct should exceed its correlations
        with all other constructs.

        Returns:
            Tuple of (comparison matrix, passed: bool)
        """
        constructs = list(self.constructs.keys())
        n = len(constructs)

        # Calculate AVE for each construct
        aves = {c: self.average_variance_extracted(c) for c in constructs}
        sqrt_aves = {c: np.sqrt(ave) for c, ave in aves.items()}

        # Get correlation matrix
        corr_matrix = self.construct_correlation_matrix()

        # Build comparison matrix
        # Diagonal: sqrt(AVE), Off-diagonal: correlations
        fl_matrix = pd.DataFrame(
            np.zeros((n, n)),
            index=constructs,
            columns=constructs
        )

        for i, c1 in enumerate(constructs):
            for j, c2 in enumerate(constructs):
                if i == j:
                    fl_matrix.iloc[i, j] = sqrt_aves[c1]
                else:
                    fl_matrix.iloc[i, j] = corr_matrix.loc[c1, c2]

        # Check criterion: diagonal > all off-diagonal in same row/column
        passed = True
        for i, construct in enumerate(constructs):
            sqrt_ave = fl_matrix.iloc[i, i]
            for j in range(n):
                if i != j:
                    if abs(fl_matrix.iloc[i, j]) >= sqrt_ave:
                        passed = False
                        break

        return fl_matrix, passed

    def validate_construct(self, construct: str) -> ConstructMetrics:
        """
        Run all validation metrics for a single construct.

        Args:
            construct: Name of construct

        Returns:
            ConstructMetrics dataclass with all results
        """
        items = self.constructs[construct]
        loadings, loadings_se = self.factor_loadings(construct)

        return ConstructMetrics(
            name=construct,
            n_items=len(items),
            cronbachs_alpha=self.cronbachs_alpha(construct),
            composite_reliability=self.composite_reliability(construct),
            average_variance_extracted=self.average_variance_extracted(construct),
            factor_loadings=loadings,
            factor_loadings_se=loadings_se,
            item_total_correlations=self.item_total_correlations(construct)
        )

    def validate_all(self) -> MeasurementValidationResult:
        """
        Run complete measurement model validation.

        Returns:
            MeasurementValidationResult with all metrics
        """
        # Validate each construct
        construct_metrics = {}
        for construct in self.constructs:
            construct_metrics[construct] = self.validate_construct(construct)

        # Correlation matrix
        corr_matrix = self.construct_correlation_matrix()

        # Fornell-Larcker test
        fl_matrix, discriminant_valid = self.fornell_larcker_test()

        # Collect warnings
        warnings_list = []
        for name, metrics in construct_metrics.items():
            if not metrics.alpha_valid:
                warnings_list.append(
                    f"{name}: Cronbach's alpha ({metrics.cronbachs_alpha:.3f}) < 0.70"
                )
            if not metrics.cr_valid:
                warnings_list.append(
                    f"{name}: Composite reliability ({metrics.composite_reliability:.3f}) < 0.70"
                )
            if not metrics.ave_valid:
                warnings_list.append(
                    f"{name}: AVE ({metrics.average_variance_extracted:.3f}) < 0.50"
                )
            for item, loading in metrics.factor_loadings.items():
                if loading < self.LOADING_THRESHOLD:
                    warnings_list.append(
                        f"{name}/{item}: Loading ({loading:.3f}) < 0.50"
                    )

        if not discriminant_valid:
            warnings_list.append("Discriminant validity (Fornell-Larcker) NOT satisfied")

        # Overall validity
        overall_valid = (
            all(m.is_valid for m in construct_metrics.values())
            and discriminant_valid
        )

        self._results = MeasurementValidationResult(
            construct_metrics=construct_metrics,
            correlation_matrix=corr_matrix,
            fornell_larcker_matrix=fl_matrix,
            discriminant_validity_passed=discriminant_valid,
            overall_valid=overall_valid,
            warnings=warnings_list
        )

        return self._results

    def print_report(self):
        """Print formatted validation report."""
        if self._results is None:
            self._results = self.validate_all()

        print("\n" + "=" * 70)
        print("MEASUREMENT MODEL VALIDATION REPORT")
        print("=" * 70)

        # Overall status
        status = "PASSED" if self._results.overall_valid else "FAILED"
        print(f"\nOverall Status: {status}")

        # Construct-level metrics
        print("\n" + "-" * 70)
        print("CONSTRUCT RELIABILITY & VALIDITY")
        print("-" * 70)
        print(f"{'Construct':<15} {'Alpha':>8} {'CR':>8} {'AVE':>8} {'Status':>10}")
        print("-" * 70)

        for name, metrics in self._results.construct_metrics.items():
            status = "OK" if metrics.is_valid else "CHECK"
            print(f"{name:<15} {metrics.cronbachs_alpha:>8.3f} "
                  f"{metrics.composite_reliability:>8.3f} "
                  f"{metrics.average_variance_extracted:>8.3f} "
                  f"{status:>10}")

        print(f"\nThresholds: Alpha > 0.70, CR > 0.70, AVE > 0.50")

        # Factor loadings
        print("\n" + "-" * 70)
        print("FACTOR LOADINGS")
        print("-" * 70)

        for name, metrics in self._results.construct_metrics.items():
            print(f"\n{name}:")
            for item, loading in metrics.factor_loadings.items():
                se = metrics.factor_loadings_se.get(item, 0)
                itc = metrics.item_total_correlations.get(item, 0)
                flag = "" if loading >= 0.50 else " *LOW*"
                print(f"  {item:<25} λ={loading:.3f} (SE={se:.3f}) ITC={itc:.3f}{flag}")

        # Discriminant validity
        print("\n" + "-" * 70)
        print("DISCRIMINANT VALIDITY (Fornell-Larcker)")
        print("-" * 70)
        print("\nDiagonal = sqrt(AVE), Off-diagonal = correlations")
        print("Criterion: Diagonal > all off-diagonal in same row/column")
        print()
        print(self._results.fornell_larcker_matrix.to_string(float_format=lambda x: f'{x:.3f}'))

        status = "PASSED" if self._results.discriminant_validity_passed else "FAILED"
        print(f"\nDiscriminant Validity: {status}")

        # Warnings
        if self._results.warnings:
            print("\n" + "-" * 70)
            print("WARNINGS")
            print("-" * 70)
            for w in self._results.warnings:
                print(f"  - {w}")

        print("\n" + "=" * 70)

    def to_latex(self, output_dir: Path) -> Dict[str, Path]:
        """
        Generate publication-ready LaTeX tables.

        Args:
            output_dir: Directory for output files

        Returns:
            Dict mapping table name to file path
        """
        if self._results is None:
            self._results = self.validate_all()

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_files = {}

        # Table 1: Reliability Summary
        reliability_path = output_dir / "measurement_reliability.tex"
        self._latex_reliability_table(reliability_path)
        output_files['reliability'] = reliability_path

        # Table 2: Factor Loadings
        loadings_path = output_dir / "factor_loadings.tex"
        self._latex_loadings_table(loadings_path)
        output_files['loadings'] = loadings_path

        # Table 3: Discriminant Validity
        discriminant_path = output_dir / "discriminant_validity.tex"
        self._latex_discriminant_table(discriminant_path)
        output_files['discriminant'] = discriminant_path

        print(f"LaTeX tables saved to {output_dir}/")
        return output_files

    def _latex_reliability_table(self, path: Path):
        """Generate reliability summary table."""
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Measurement Model Reliability and Validity}",
            r"\label{tab:measurement_reliability}",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"Construct & Items & Cronbach's $\alpha$ & CR & AVE \\",
            r"\midrule",
        ]

        for name, metrics in self._results.construct_metrics.items():
            # Format construct name nicely
            display_name = name.replace('_', ' ').title()
            lines.append(
                f"{display_name} & {metrics.n_items} & "
                f"{metrics.cronbachs_alpha:.3f} & "
                f"{metrics.composite_reliability:.3f} & "
                f"{metrics.average_variance_extracted:.3f} \\\\"
            )

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{tablenotes}",
            r"\small",
            r"\item Note: CR = Composite Reliability; AVE = Average Variance Extracted.",
            r"\item Thresholds: $\alpha$ > 0.70, CR > 0.70, AVE > 0.50 (Hair et al., 2019).",
            r"\end{tablenotes}",
            r"\end{table}",
        ])

        with open(path, 'w') as f:
            f.write('\n'.join(lines))

    def _latex_loadings_table(self, path: Path):
        """Generate factor loadings table."""
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Standardized Factor Loadings}",
            r"\label{tab:factor_loadings}",
            r"\begin{tabular}{llccc}",
            r"\toprule",
            r"Construct & Item & Loading & SE & ITC \\",
            r"\midrule",
        ]

        for name, metrics in self._results.construct_metrics.items():
            display_name = name.replace('_', ' ').title()
            first = True
            for item, loading in metrics.factor_loadings.items():
                se = metrics.factor_loadings_se.get(item, 0)
                itc = metrics.item_total_correlations.get(item, 0)
                item_short = item.split('_')[-1] if '_' in item else item

                if first:
                    lines.append(
                        f"{display_name} & {item_short} & "
                        f"{loading:.3f} & {se:.3f} & {itc:.3f} \\\\"
                    )
                    first = False
                else:
                    lines.append(
                        f" & {item_short} & "
                        f"{loading:.3f} & {se:.3f} & {itc:.3f} \\\\"
                    )
            lines.append(r"\addlinespace")

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{tablenotes}",
            r"\small",
            r"\item Note: SE = Bootstrap standard error; ITC = Item-total correlation.",
            r"\item Threshold: Loading > 0.50 (Hair et al., 2019).",
            r"\end{tablenotes}",
            r"\end{table}",
        ])

        with open(path, 'w') as f:
            f.write('\n'.join(lines))

    def _latex_discriminant_table(self, path: Path):
        """Generate Fornell-Larcker discriminant validity table."""
        fl = self._results.fornell_larcker_matrix
        constructs = list(fl.columns)

        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Discriminant Validity: Fornell-Larcker Criterion}",
            r"\label{tab:discriminant_validity}",
            r"\begin{tabular}{l" + "c" * len(constructs) + "}",
            r"\toprule",
            " & " + " & ".join([c.replace('_', ' ').title() for c in constructs]) + r" \\",
            r"\midrule",
        ]

        for i, c1 in enumerate(constructs):
            display_name = c1.replace('_', ' ').title()
            values = []
            for j, c2 in enumerate(constructs):
                val = fl.iloc[i, j]
                if i == j:
                    # Diagonal (sqrt(AVE)) in bold
                    values.append(f"\\textbf{{{val:.3f}}}")
                elif j < i:
                    values.append(f"{val:.3f}")
                else:
                    values.append("")  # Upper triangle empty
            lines.append(f"{display_name} & " + " & ".join(values) + r" \\")

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{tablenotes}",
            r"\small",
            r"\item Note: Diagonal (bold) = $\sqrt{\text{AVE}}$; Off-diagonal = correlations.",
            r"\item Criterion: Diagonal values should exceed off-diagonal values in same row/column.",
            r"\end{tablenotes}",
            r"\end{table}",
        ])

        with open(path, 'w') as f:
            f.write('\n'.join(lines))


def auto_detect_constructs(df: pd.DataFrame,
                           patterns: Dict[str, str] = None) -> Dict[str, List[str]]:
    """
    Automatically detect constructs from column naming patterns.

    Args:
        df: DataFrame with indicator columns
        patterns: Dict mapping construct name to column prefix pattern
                 Default: standard HCM patterns

    Returns:
        Dict mapping construct names to item column lists
    """
    if patterns is None:
        patterns = {
            'pat_blind': 'pat_blind_',
            'pat_const': 'pat_constructive_',
            'sec_dl': 'sec_dl_',
            'sec_fp': 'sec_fp_'
        }

    constructs = {}
    for name, prefix in patterns.items():
        items = [c for c in df.columns if c.startswith(prefix) and c[-1].isdigit()]
        if items:
            constructs[name] = sorted(items)

    return constructs


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def validate_measurement_model(df: pd.DataFrame,
                               constructs: Dict[str, List[str]] = None,
                               output_dir: Path = None,
                               print_report: bool = True) -> MeasurementValidationResult:
    """
    Convenience function for measurement model validation.

    Args:
        df: DataFrame with indicator columns
        constructs: Dict mapping construct name to item columns (auto-detected if None)
        output_dir: Directory for LaTeX output (optional)
        print_report: Whether to print validation report

    Returns:
        MeasurementValidationResult
    """
    if constructs is None:
        constructs = auto_detect_constructs(df)

    if not constructs:
        raise ValueError("No constructs detected. Provide constructs dict or ensure "
                        "columns follow naming convention (e.g., pat_blind_1, sec_dl_2)")

    validator = MeasurementValidator(df, constructs)
    results = validator.validate_all()

    if print_report:
        validator.print_report()

    if output_dir is not None:
        validator.to_latex(output_dir)

    return results


if __name__ == '__main__':
    print("Measurement Validation Module")
    print("=" * 40)
    print("\nUsage:")
    print("  from src.estimation.measurement_validation import validate_measurement_model")
    print("  results = validate_measurement_model(df)")
    print("\nMetrics computed:")
    print("  - Cronbach's Alpha (internal consistency)")
    print("  - Factor Loadings with SE (item quality)")
    print("  - Composite Reliability (construct reliability)")
    print("  - Average Variance Extracted (convergent validity)")
    print("  - Fornell-Larcker (discriminant validity)")
