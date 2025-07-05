"""Deprecated: use :mod:`core.strategies.copilot` instead."""
from __future__ import annotations

import warnings
from typing import Dict

from .strategies import copilot


def handle_prompt(prompt: str) -> Dict:
    warnings.warn(
        "copilot_orchestrator.handle_prompt is deprecated",
        DeprecationWarning,
        stacklevel=2,
    )
    return copilot.handle_prompt(prompt)
