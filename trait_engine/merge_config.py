# trait_engine/merge_config.py
# Dynamically merges runtime config layers: base, strategy, overrides

import json
import os

DEFAULT_PATHS = [
    "config/trait_config.json",
    "config/chart_config.json",
    "config/strategy_profiles.json",
    "core/macro_config.json",
]


def load_json(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


def merge_dicts(a, b):
    result = a.copy()
    for key, value in b.items():
        if isinstance(value, dict) and key in result:
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def merge_trait_config(overrides=None):
    config = {}
    for path in DEFAULT_PATHS:
        layer = load_json(path)
        config = merge_dicts(config, layer)

    if overrides:
        config = merge_dicts(config, overrides)

    return config


if __name__ == "__main__":
    full_config = merge_trait_config()
    print(json.dumps(full_config, indent=2))
