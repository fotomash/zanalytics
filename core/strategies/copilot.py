"""Minimal copilot prompt handler."""
from __future__ import annotations

from typing import Dict


def handle_prompt(prompt: str) -> Dict:
    """Echo the provided prompt.

    This stub replaces the heavy logic formerly in
    ``core.copilot_orchestrator`` which is deprecated.
    """
    return {"echo": prompt}
