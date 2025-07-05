import pandas as pd
import numpy as np
from typing import Dict, Optional


def add_fractals(df: pd.DataFrame, config: Optional[Dict] = None) -> pd.DataFrame:
    """Identify Bill Williams fractal highs and lows."""
    if config is None:
        config = {}
    lookback = config.get("lookback", 2)
    window = lookback * 2 + 1

    df["fractal_high"] = df["High"].rolling(window, center=True).apply(
        lambda x: x[lookback] if x[lookback] == x.max() else np.nan,
        raw=True,
    )
    df["fractal_low"] = df["Low"].rolling(window, center=True).apply(
        lambda x: x[lookback] if x[lookback] == x.min() else np.nan,
        raw=True,
    )
    return df
