#!/usr/bin/env python3
"""Integration script for phase 4 modules."""
import os
import shutil

PHASE4_DIR = "zanalytics_phase4_implementation"
TARGET_DIRS = {
    "monitoring": "monitoring",
    "caching": "caching",
    "optimization": "optimization",
    "deployment": "deployment",
}


def copy_tree(src: str, dst: str) -> None:
    if os.path.isdir(src):
        os.makedirs(dst, exist_ok=True)
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)


def integrate_phase4() -> None:
    print("Integrating Phase 4 components...")
    for src_name, dst_name in TARGET_DIRS.items():
        src_path = os.path.join(PHASE4_DIR, src_name)
        dst_path = os.path.join(dst_name)
        if os.path.exists(src_path):
            copy_tree(src_path, dst_path)
            print(f"Copied {src_name} -> {dst_path}")
        else:
            print(f"Missing {src_path}")
    print("Integration complete.")


if __name__ == "__main__":
    integrate_phase4()
