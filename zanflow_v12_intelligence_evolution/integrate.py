#!/usr/bin/env python3
'''
Intelligence Evolution Integration Script
=======================================
Integrates Phase 5 enhancements into an existing ZANFLOW v12 installation.

Usage:
    python integrate.py /path/to/zanflow
'''

import os
import json
import yaml
import shutil
from datetime import datetime
import argparse


def backup_existing_files(base_dir: str) -> str:
    """Create backup of select configuration files in the given ZANFLOW directory.

    Returns the path to the created backup directory.
    """
    backup_dir = os.path.join(base_dir, f"backups/backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(backup_dir, exist_ok=True)

    # Files to backup
    files_to_backup = [
        "config/orchestrator_config.yaml",
        "config/agent_profiles.yaml",
        "core/risk_manager.py"
    ]

    for file_path in files_to_backup:
        full_path = os.path.join(base_dir, file_path)
        if os.path.exists(full_path):
            backup_path = os.path.join(backup_dir, file_path)
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)
            shutil.copy2(full_path, backup_path)
            print(f"Backed up: {file_path}")

    return backup_dir


PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

PHASE5_DIRS = [
    "core",
    "configs",
    "schemas",
]


def copy_phase5_files(target_dir: str) -> None:
    """Copy Phase 5 components into the target project."""
    for sub in PHASE5_DIRS:
        src = os.path.join(PACKAGE_DIR, sub)
        if not os.path.exists(src):
            continue
        dst = os.path.join(target_dir, sub)
        shutil.copytree(src, dst, dirs_exist_ok=True)
        print(f"Copied {sub} -> {dst}")


def update_orchestrator_config(target_dir: str) -> None:
    """Add intelligence component to orchestrator config if present."""
    cfg_path = os.path.join(target_dir, "orchestrator_config.yaml")
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(target_dir, "config", "orchestrator_config.yaml")
        if not os.path.exists(cfg_path):
            return

    with open(cfg_path, "r") as fh:
        data = yaml.safe_load(fh) or {}

    components = data.get("components", [])
    if "intelligence_evolution" not in components:
        components.append("intelligence_evolution")
        data["components"] = components
        with open(cfg_path, "w") as fh:
            yaml.dump(data, fh)
        print(f"Updated {cfg_path} with intelligence_evolution component")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Integrate Intelligence Evolution Phase 5 into an existing ZANFLOW installation.\n"
            "Example: python integrate.py /opt/zanflow"
        )
    )
    parser.add_argument(
        "target_dir",
        help="Path to the root of your ZANFLOW project",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    target = os.path.abspath(args.target_dir)

    print("=== ZANFLOW v12 Intelligence Evolution Integration ===")

    if not os.path.isdir(target):
        print(f"Error: {target} is not a directory")
        return

    print(f"Target directory: {target}\n")

    backup_existing_files(target)
    copy_phase5_files(target)
    update_orchestrator_config(target)

    print("\nIntegration complete.")


if __name__ == "__main__":
    main()
