import pandas as pd
import numpy as np
from typing import Dict, Optional


def calculate_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def add_rsi_divergence(df: pd.DataFrame, config: Optional[Dict] = None) -> pd.DataFrame:
    """Add RSI and basic divergence labels."""
    if config is None:
        config = {}
    rsi_period = config.get("rsi_period", 14)
    lookback = config.get("swing_lookback", 2)
    window = lookback * 2 + 1

    df = df.copy()
    df["rsi"] = calculate_rsi(df["Close"], rsi_period)

    df["swing_high"] = df["High"].rolling(window, center=True).apply(
        lambda x: x[lookback] if x[lookback] == x.max() else np.nan,
        raw=True,
    )
    df["swing_low"] = df["Low"].rolling(window, center=True).apply(
        lambda x: x[lookback] if x[lookback] == x.min() else np.nan,
        raw=True,
    )
    df["rsi_high"] = df["rsi"].rolling(window, center=True).apply(
        lambda x: x[lookback] if x[lookback] == x.max() else np.nan,
        raw=True,
    )
    df["rsi_low"] = df["rsi"].rolling(window, center=True).apply(
        lambda x: x[lookback] if x[lookback] == x.min() else np.nan,
        raw=True,
    )
    df["rsi_divergence"] = None

    highs = df.dropna(subset=["swing_high"]).index
    if len(highs) >= 2:
        prev, curr = highs[-2], highs[-1]
        price_prev = df.loc[prev, "swing_high"]
        price_curr = df.loc[curr, "swing_high"]
        rsi_prev = df.loc[prev, "rsi_high"]
        rsi_curr = df.loc[curr, "rsi_high"]
        if price_curr > price_prev and rsi_curr < rsi_prev:
            df.loc[curr, "rsi_divergence"] = "Bearish"
        elif price_curr < price_prev and rsi_curr > rsi_prev:
            df.loc[curr, "rsi_divergence"] = "HiddenBearish"
    lows = df.dropna(subset=["swing_low"]).index
    if len(lows) >= 2:
        prev, curr = lows[-2], lows[-1]
        price_prev = df.loc[prev, "swing_low"]
        price_curr = df.loc[curr, "swing_low"]
        rsi_prev = df.loc[prev, "rsi_low"]
        rsi_curr = df.loc[curr, "rsi_low"]
        if price_curr < price_prev and rsi_curr > rsi_prev:
            df.loc[curr, "rsi_divergence"] = "Bullish"
        elif price_curr > price_prev and rsi_curr < rsi_prev:
            df.loc[curr, "rsi_divergence"] = "HiddenBullish"
    return df
