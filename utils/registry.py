#!/usr/bin/env python3
"""
Experiment registry utilities for dynamic experiment discovery and loading.
"""

import os
import yaml
from typing import Dict, List, Optional, Any
from pathlib import Path


def get_project_root() -> Path:
    """Get the project root directory (where experiment_registry.yaml lives)."""
    current = Path(__file__).parent
    while current != current.parent:
        if (current / "experiment_registry.yaml").exists():
            return current
        current = current.parent
    raise FileNotFoundError("Could not find experiment_registry.yaml in parent directories")


def load_experiment_registry(registry_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load the experiment registry from YAML file.

    Args:
        registry_path: Optional path to registry file.
                       Defaults to project_root/experiment_registry.yaml

    Returns:
        Dict containing experiment definitions

    Example:
        >>> registry = load_experiment_registry()
        >>> print(registry["experiments"])
    """
    if registry_path is None:
        registry_path = get_project_root() / "experiment_registry.yaml"

    with open(registry_path, 'r') as f:
        return yaml.safe_load(f)


def get_experiment_config(
    experiment_name: str,
    registry_path: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Get configuration for a specific experiment by name.

    Args:
        experiment_name: Name of the experiment to find
        registry_path: Optional path to registry file

    Returns:
        Experiment config dict or None if not found
    """
    registry = load_experiment_registry(registry_path)

    for exp in registry.get("experiments", []):
        if exp.get("name") == experiment_name:
            return exp

    return None


def list_experiments(
    registry_path: Optional[str] = None,
    active_only: bool = False
) -> List[Dict[str, Any]]:
    """
    List all experiments in the registry.

    Args:
        registry_path: Optional path to registry file
        active_only: If True, only return active experiments

    Returns:
        List of experiment config dicts
    """
    registry = load_experiment_registry(registry_path)
    experiments = registry.get("experiments", [])

    if active_only:
        experiments = [e for e in experiments if e.get("active", True)]

    return experiments


def get_experiment_path(experiment_name: str) -> Optional[Path]:
    """
    Get the filesystem path for an experiment.

    Args:
        experiment_name: Name of the experiment

    Returns:
        Path to experiment directory or None if not found
    """
    exp_config = get_experiment_config(experiment_name)
    if exp_config is None:
        return None

    module_path = exp_config.get("module", experiment_name)
    project_root = get_project_root()

    # Try to find the experiment in the experiments folder
    exp_path = project_root / "experiments" / module_path
    if exp_path.exists():
        return exp_path

    # Check for direct path in module field
    if "/" in module_path:
        exp_path = project_root / "experiments" / module_path
        if exp_path.exists():
            return exp_path

    return None


def get_experiment_scripts(experiment_name: str) -> List[str]:
    """
    Get available scripts for an experiment.

    Args:
        experiment_name: Name of the experiment

    Returns:
        List of available script/task names
    """
    exp_config = get_experiment_config(experiment_name)
    if exp_config is None:
        return []

    return exp_config.get("scripts", [])


def print_registry_summary():
    """Print a summary of all registered experiments."""
    experiments = list_experiments()

    print("\n" + "=" * 60)
    print("📋 Experiment Registry Summary")
    print("=" * 60)

    for exp in experiments:
        status = "✅" if exp.get("active", True) else "⏸️"
        name = exp.get("name", "unnamed")
        desc = exp.get("description", "No description")
        module = exp.get("module", "N/A")
        scripts = exp.get("scripts", [])

        print(f"\n{status} {name}")
        print(f"   📁 Module: {module}")
        print(f"   📝 {desc}")
        if scripts:
            print(f"   🔧 Scripts: {', '.join(scripts)}")

    print("\n" + "=" * 60)
