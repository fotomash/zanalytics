from typing import Dict
from pathlib import Path
import json


def load_strategy_profile(variant_name: str) -> Dict:
    """Load configuration for a strategy variant from strategy_profiles.json.

    Falls back to the 'default' profile when the requested variant
    is unavailable. Returns an empty dictionary if the profile file
    cannot be loaded.
    """
    config_path = Path("strategy_profiles.json")
    if not config_path.exists():
        print(f"[ERROR][ORCH_UTILS] strategy_profiles.json not found at {config_path.resolve()}.")
        return {}
    try:
        with open(config_path, "r") as f:
            all_profiles = json.load(f)
        if variant_name in all_profiles:
            print(f"[INFO][ORCH_UTILS] Loaded strategy profile for variant: {variant_name}")
            return all_profiles[variant_name]
        if "default" in all_profiles:
            print(f"[WARN][ORCH_UTILS] Strategy variant '{variant_name}' not found in profiles. Falling back to 'default'.")
            return all_profiles["default"]
        print(
            f"[WARN][ORCH_UTILS] Strategy variant '{variant_name}' not found and no 'default' profile present. Using empty config."
        )
        return {}
    except Exception as e:
        print(f"[ERROR][ORCH_UTILS] Failed to load or parse strategy_profiles.json: {e}")
        return {}
