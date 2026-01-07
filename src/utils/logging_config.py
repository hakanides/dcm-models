"""
Structured Logging for DCM Estimation
======================================

Provides consistent logging across the DCM estimation framework.

Usage:
    from src.utils.logging_config import get_logger, EstimationLogger

    # Simple logging
    logger = get_logger(__name__)
    logger.info("Starting estimation")

    # Structured estimation logging
    est_log = EstimationLogger("MNL-Basic")
    est_log.start()
    est_log.converged(ll=-1234.5, k=5, aic=2479.0)

Author: DCM Research Team
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import json


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_style: str = "standard"
) -> None:
    """
    Configure logging for the DCM package.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for log output
        format_style: "standard", "detailed", or "json"
    """
    formats = {
        "standard": "%(asctime)s | %(levelname)-8s | %(message)s",
        "detailed": "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        "json": None  # Handled by JsonFormatter
    }

    # Create handlers
    handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    if format_style == "json":
        console_handler.setFormatter(JsonFormatter())
    else:
        console_handler.setFormatter(
            logging.Formatter(formats.get(format_style, formats["standard"]),
                            datefmt="%Y-%m-%d %H:%M:%S")
        )
    handlers.append(console_handler)

    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter(formats["detailed"], datefmt="%Y-%m-%d %H:%M:%S")
        )
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(level=level, handlers=handlers, force=True)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a module."""
    return logging.getLogger(name)


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if hasattr(record, "extra_data"):
            log_data["data"] = record.extra_data

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


# =============================================================================
# ESTIMATION LOGGER
# =============================================================================

class EstimationLogger:
    """
    Structured logger for DCM estimation progress.

    Provides consistent formatting for estimation output.

    Example:
        logger = EstimationLogger("MNL-Basic")
        logger.start()
        logger.iteration(1, ll=-2000.0)
        logger.converged(ll=-1800.0, k=5, aic=3610.0)
    """

    def __init__(self, model_name: str, verbose: bool = True):
        self.model_name = model_name
        self.verbose = verbose
        self.start_time: Optional[datetime] = None
        self._logger = get_logger(f"dcm.estimation.{model_name}")

    def _print(self, message: str) -> None:
        """Print if verbose mode is on."""
        if self.verbose:
            print(message)

    def start(self) -> None:
        """Log estimation start."""
        self.start_time = datetime.now()
        self._print(f"\n{'='*60}")
        self._print(f"Estimating: {self.model_name}")
        self._print(f"{'='*60}")
        self._logger.info(f"Started estimation: {self.model_name}")

    def iteration(self, n: int, ll: float, **kwargs) -> None:
        """Log iteration progress."""
        extra = " | ".join(f"{k}={v:.4f}" for k, v in kwargs.items())
        msg = f"  Iter {n:3d}: LL = {ll:.2f}"
        if extra:
            msg += f" | {extra}"
        self._print(msg)
        self._logger.debug(f"Iteration {n}: LL={ll:.2f} {extra}")

    def converged(self, ll: float, k: int, aic: float, bic: Optional[float] = None,
                  rho2: Optional[float] = None) -> None:
        """Log successful convergence."""
        elapsed = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0

        self._print(f"\n  CONVERGED in {elapsed:.1f}s")
        self._print(f"  LL: {ll:.2f} | K: {k} | AIC: {aic:.2f}", )
        if bic is not None:
            self._print(f"  BIC: {bic:.2f}")
        if rho2 is not None:
            self._print(f"  rho2: {rho2:.4f}")

        self._logger.info(
            f"Converged: {self.model_name} | LL={ll:.2f} | K={k} | "
            f"AIC={aic:.2f} | time={elapsed:.1f}s"
        )

    def failed(self, reason: str) -> None:
        """Log estimation failure."""
        elapsed = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        self._print(f"\n  FAILED after {elapsed:.1f}s: {reason}")
        self._logger.error(f"Failed: {self.model_name} | {reason}")

    def parameters(self, betas: Dict[str, float], std_errs: Optional[Dict[str, float]] = None) -> None:
        """Log estimated parameters."""
        self._print("\n  Parameters:")
        for name, value in sorted(betas.items()):
            se = std_errs.get(name, float('nan')) if std_errs else float('nan')
            t_stat = value / se if se and se != 0 else float('nan')
            sig = "***" if abs(t_stat) > 2.576 else "**" if abs(t_stat) > 1.96 else "*" if abs(t_stat) > 1.645 else ""
            self._print(f"    {name:20s}: {value:8.4f} (SE: {se:6.4f}) {sig}")

        self._logger.info(f"Parameters estimated: {list(betas.keys())}")

    def warm_start(self, from_model: str, n_params: int) -> None:
        """Log warm-start initialization."""
        self._print(f"  Warm-start from {from_model} ({n_params} params)")
        self._logger.info(f"Warm-start from {from_model}: {n_params} parameters")


# =============================================================================
# CROSS-VALIDATION LOGGER
# =============================================================================

class CVLogger:
    """Logger for cross-validation progress."""

    def __init__(self, n_folds: int, verbose: bool = True):
        self.n_folds = n_folds
        self.verbose = verbose
        self._logger = get_logger("dcm.cv")

    def _print(self, message: str) -> None:
        if self.verbose:
            print(message)

    def start(self, model_name: str, id_col: str) -> None:
        """Log CV start."""
        self._print(f"\n{'='*60}")
        self._print(f"CROSS-VALIDATION ({self.n_folds}-fold, split by {id_col})")
        self._print(f"Model: {model_name}")
        self._print(f"{'='*60}")
        self._logger.info(f"Starting {self.n_folds}-fold CV for {model_name}")

    def fold(self, fold_idx: int, n_train: int, n_test: int,
             n_train_ids: int, n_test_ids: int) -> None:
        """Log fold information."""
        self._print(f"\nFold {fold_idx + 1}/{self.n_folds}:")
        self._print(f"  Train: {n_train:,} obs from {n_train_ids} individuals")
        self._print(f"  Test:  {n_test:,} obs from {n_test_ids} individuals")
        self._logger.debug(f"Fold {fold_idx + 1}: train={n_train}, test={n_test}")

    def fold_result(self, train_ll: float, test_ll: float,
                    train_ll_per_obs: float, test_ll_per_obs: float) -> None:
        """Log fold results."""
        self._print(f"  Train LL: {train_ll:.2f} ({train_ll_per_obs:.4f}/obs)")
        self._print(f"  Test LL:  {test_ll:.2f} ({test_ll_per_obs:.4f}/obs)")

    def summary(self, train_mean: float, test_mean: float, overfit_ratio: float) -> None:
        """Log CV summary."""
        self._print(f"\n{'='*60}")
        self._print("CROSS-VALIDATION SUMMARY")
        self._print(f"{'='*60}")
        self._print(f"Mean Train LL/obs: {train_mean:.4f}")
        self._print(f"Mean Test LL/obs:  {test_mean:.4f}")
        self._print(f"Overfit ratio:     {overfit_ratio:.4f}")

        if overfit_ratio > 1.1:
            self._print("  WARNING: Potential overfitting detected")
        elif overfit_ratio < 0.95:
            self._print("  NOTE: Test performance similar to train (good generalization)")

        self._logger.info(
            f"CV complete: train={train_mean:.4f}, test={test_mean:.4f}, "
            f"overfit_ratio={overfit_ratio:.4f}"
        )


# =============================================================================
# MODEL COMPARISON LOGGER
# =============================================================================

class ComparisonLogger:
    """Logger for model comparison output."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._logger = get_logger("dcm.comparison")

    def _print(self, message: str) -> None:
        if self.verbose:
            print(message)

    def header(self, title: str = "MODEL COMPARISON") -> None:
        """Print comparison header."""
        self._print(f"\n{'#'*60}")
        self._print(f"# {title}")
        self._print(f"{'#'*60}")

    def model_result(self, name: str, ll: float, k: int, aic: float,
                     bic: Optional[float] = None, converged: bool = True) -> None:
        """Log single model result."""
        status = "OK" if converged else "WARN"
        self._print(f"\n{name}:")
        self._print(f"  LL: {ll:.2f} | K: {k} | AIC: {aic:.2f} | [{status}]")
        if bic is not None:
            self._print(f"  BIC: {bic:.2f}")

    def lr_test(self, restricted: str, full: str, lr_stat: float,
                df: int, p_value: float) -> None:
        """Log likelihood ratio test result."""
        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        self._print(f"\nLR Test: {restricted} vs {full}")
        self._print(f"  LR = {lr_stat:.2f}, df = {df}, p = {p_value:.4f} {sig}")
        self._logger.info(f"LR test: {restricted} vs {full}, LR={lr_stat:.2f}, p={p_value:.4f}")

    def best_model(self, name: str, criterion: str = "AIC") -> None:
        """Log best model selection."""
        self._print(f"\n{'='*60}")
        self._print(f"Best model by {criterion}: {name}")
        self._print(f"{'='*60}")
        self._logger.info(f"Best model ({criterion}): {name}")


# =============================================================================
# WARNING CONFIGURATION
# =============================================================================

def configure_warnings(debug_mode: bool = False) -> None:
    """
    Configure warning filters for DCM estimation.

    By default, suppresses expected warnings from Biogeme optimization.
    Set debug_mode=True to see all warnings for troubleshooting.

    Args:
        debug_mode: If True, show all warnings. If False, suppress expected ones.

    Suppressed warnings (when debug_mode=False):
        - FutureWarning: Biogeme API deprecation warnings (expected)
        - overflow: Numerical overflow during exp() in early iterations (expected)
        - divide by zero: Can occur during probability calculation (expected)
        - invalid value: NaN during optimization convergence (expected)
    """
    import warnings

    if debug_mode:
        # Show all warnings for debugging
        warnings.filterwarnings('default')
        logging.info("Debug mode: All warnings enabled")
    else:
        # Suppress expected optimization warnings
        warnings.filterwarnings('ignore', category=FutureWarning)
        warnings.filterwarnings('ignore', message='.*overflow.*')
        warnings.filterwarnings('ignore', message='.*divide by zero.*')
        warnings.filterwarnings('ignore', message='.*invalid value.*')


# Initialize default logging on import
setup_logging(level=logging.INFO)
