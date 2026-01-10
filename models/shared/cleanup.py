"""
Cleanup utilities for isolated DCM model folders.

Provides functions to clean output files before each run to ensure fresh results.
"""

import shutil
from pathlib import Path
from typing import List, Optional


def cleanup_before_run(model_dir: Path, verbose: bool = True) -> None:
    """
    Clean all output folders before running estimation.

    Cleans:
    - results/ folder (*.csv, *.html, *.yaml, *.iter)
    - biogeme.toml file
    - output/latex/ folder
    - policy_analysis/ folder
    - sample_stats/ folder

    Args:
        model_dir: Path to the model folder (e.g., models/mnl_basic)
        verbose: Whether to print cleanup messages
    """
    model_dir = Path(model_dir)

    if verbose:
        print("=" * 60)
        print("CLEANUP: Removing previous run outputs")
        print("=" * 60)

    # Clean results folder
    results_dir = model_dir / 'results'
    if results_dir.exists():
        patterns = ['*.csv', '*.html', '*.yaml', '*.iter']
        for pattern in patterns:
            for f in results_dir.glob(pattern):
                f.unlink()
                if verbose:
                    print(f"  Removed: results/{f.name}")

    # Also clean any backup files (e.g., Model~00.html)
    if results_dir.exists():
        for f in results_dir.glob('*~*.html'):
            f.unlink()
            if verbose:
                print(f"  Removed: results/{f.name}")
        for f in results_dir.glob('*~*.yaml'):
            f.unlink()
            if verbose:
                print(f"  Removed: results/{f.name}")

    # Clean biogeme.toml if exists
    toml_file = model_dir / 'biogeme.toml'
    if toml_file.exists():
        toml_file.unlink()
        if verbose:
            print(f"  Removed: biogeme.toml")

    # Also check results folder for biogeme.toml
    results_toml = model_dir / 'results' / 'biogeme.toml'
    if results_toml.exists():
        results_toml.unlink()
        if verbose:
            print(f"  Removed: results/biogeme.toml")

    # Clean output/latex folder
    latex_dir = model_dir / 'output' / 'latex'
    if latex_dir.exists():
        shutil.rmtree(latex_dir)
        if verbose:
            print(f"  Cleaned: output/latex/")
    latex_dir.mkdir(parents=True, exist_ok=True)

    # Clean policy_analysis folder
    policy_dir = model_dir / 'policy_analysis'
    if policy_dir.exists():
        shutil.rmtree(policy_dir)
        if verbose:
            print(f"  Cleaned: policy_analysis/")
    policy_dir.mkdir(exist_ok=True)

    # Clean sample_stats folder
    stats_dir = model_dir / 'sample_stats'
    if stats_dir.exists():
        shutil.rmtree(stats_dir)
        if verbose:
            print(f"  Cleaned: sample_stats/")
    stats_dir.mkdir(exist_ok=True)

    if verbose:
        print("Cleanup complete.\n")


def cleanup_simulation_outputs(model_dir: Path, verbose: bool = True) -> None:
    """
    Clean only simulation-related outputs (for running simulate_full_data.py alone).

    Cleans:
    - data/simulated_data.csv
    - sample_stats/ folder

    Args:
        model_dir: Path to the model folder
        verbose: Whether to print cleanup messages
    """
    model_dir = Path(model_dir)

    if verbose:
        print("=" * 60)
        print("CLEANUP: Removing previous simulation outputs")
        print("=" * 60)

    # Clean simulated data
    data_dir = model_dir / 'data'
    if data_dir.exists():
        for f in data_dir.glob('*.csv'):
            f.unlink()
            if verbose:
                print(f"  Removed: data/{f.name}")
    else:
        data_dir.mkdir(exist_ok=True)

    # Clean sample_stats folder
    stats_dir = model_dir / 'sample_stats'
    if stats_dir.exists():
        shutil.rmtree(stats_dir)
        if verbose:
            print(f"  Cleaned: sample_stats/")
    stats_dir.mkdir(exist_ok=True)

    if verbose:
        print("Cleanup complete.\n")


def ensure_directories(model_dir: Path) -> dict:
    """
    Ensure all required output directories exist.

    Args:
        model_dir: Path to the model folder

    Returns:
        Dictionary with paths to all output directories
    """
    model_dir = Path(model_dir)

    dirs = {
        'data': model_dir / 'data',
        'results': model_dir / 'results',
        'latex': model_dir / 'output' / 'latex',
        'policy': model_dir / 'policy_analysis',
        'stats': model_dir / 'sample_stats',
    }

    for name, path in dirs.items():
        path.mkdir(parents=True, exist_ok=True)

    return dirs
