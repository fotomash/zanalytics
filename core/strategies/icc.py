"""Simplified ICC orchestrator."""
from __future__ import annotations

from typing import Dict, Any


def run(symbol: str) -> Dict[str, Any]:
    """Return a minimal ICC workflow result."""
    return {
        "regime": {"playbook": "neutral_playbook"},
        "context": {"last_close": None},
        "catalyst": {},
        "confirmation": {"confirmed": False},
        "execution": {"status": "skipped"},
        "symbol": symbol,
    }
