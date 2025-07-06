import pandas as pd
import numpy as np
from typing import Dict, Optional


def add_dss(df: pd.DataFrame, config: Optional[Dict] = None) -> pd.DataFrame:
    """Add Double Smoothed Stochastic (DSS) oscillator columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'High', 'Low' and 'Close' columns.
    config : dict, optional
        Optional configuration with keys 'lookback' and 'smooth'.
    """
    if config is None:
        config = {}
    lookback = config.get("lookback", 13)
    smooth = config.get("smooth", 8)

    highest_high = df["High"].rolling(lookback).max()
    lowest_low = df["Low"].rolling(lookback).min()
    denom = highest_high - lowest_low
    denom = denom.replace(0, np.nan)
    k = 100 * (df["Close"] - lowest_low) / denom
    k = k.ewm(span=smooth, adjust=False).mean()
    dss_k = k.ewm(span=smooth, adjust=False).mean()
    dss_d = dss_k.ewm(span=smooth, adjust=False).mean()

    df["dss_k"] = dss_k
    df["dss_d"] = dss_d
    return df
