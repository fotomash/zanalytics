import pandas as pd
import numpy as np
from typing import Dict, Optional


def add_vwap(df: pd.DataFrame, tf: str = "", config: Optional[Dict] = None) -> pd.DataFrame:
    """Calculate Volume Weighted Average Price."""
    if config is None:
        config = {}
    if "Volume" not in df.columns:
        df["vwap"] = np.nan
        return df

    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    cumulative_tp_v = (typical_price * df["Volume"]).cumsum()
    cumulative_vol = df["Volume"].cumsum().replace(0, np.nan)
    df["vwap"] = cumulative_tp_v / cumulative_vol
    return df
