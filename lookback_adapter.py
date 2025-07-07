# lookback_adapter.py
# Utility to adapt lookback windows using recent volatility info.

import pandas as pd
from typing import Dict, Optional

try:
    from core.volatility_engine import get_volatility_profile
    VOL_ENGINE_AVAILABLE = True
except Exception:
    VOL_ENGINE_AVAILABLE = False

__all__ = ["adapt_lookback"]


def adapt_lookback(
    df: pd.DataFrame,
    base_lookback: int,
    min_lookback: int,
    max_lookback: int,
    vol_config: Optional[Dict] = None,
) -> int:
    """Return adjusted lookback window based on volatility regime.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV dataframe for volatility calculation.
    base_lookback : int
        Default lookback window.
    min_lookback : int
        Minimum allowed lookback.
    max_lookback : int
        Maximum allowed lookback.
    vol_config : dict, optional
        Configuration passed to ``get_volatility_profile``.
    """
    if df is None or df.empty or not VOL_ENGINE_AVAILABLE:
        return int(base_lookback)

    profile = get_volatility_profile(df, vol_config or {})
    if profile.get("error"):
        return int(base_lookback)

    regime = profile.get("volatility_regime", "Normal")
    lookback = base_lookback
    if regime == "Explosive":
        lookback = base_lookback - 2
    elif regime == "Quiet":
        lookback = base_lookback + 2

    lookback = max(min_lookback, min(max_lookback, lookback))
    return int(lookback)

