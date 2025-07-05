"""Deprecated: use :mod:`core.strategies.advanced_smc` instead."""
from __future__ import annotations

import warnings
from typing import Dict
import pandas as pd

from .strategies import advanced_smc


def dispatch_pine_payload(entry_result: Dict) -> None:
    warnings.warn(
        "advanced_smc_orchestrator.dispatch_pine_payload is deprecated",
        DeprecationWarning,
        stacklevel=2,
    )
    payload = entry_result.get("pine_payload")
    if payload:
        print("[DEPRECATED] dispatching payload", payload)


def run_advanced_smc_strategy(
    all_tf_data: Dict[str, pd.DataFrame],
    strategy_variant: str,
    target_timestamp: pd.Timestamp,
    symbol: str = "XAUUSD",
) -> Dict:
    warnings.warn(
        "advanced_smc_orchestrator.run_advanced_smc_strategy is deprecated",
        DeprecationWarning,
        stacklevel=2,
    )
    return advanced_smc.run_advanced_smc_strategy(
        all_tf_data,
        strategy_variant,
        target_timestamp,
        symbol=symbol,
    )
