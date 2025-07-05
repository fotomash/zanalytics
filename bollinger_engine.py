import pandas as pd
from typing import Dict, Optional


def add_bollinger_bands(df: pd.DataFrame, config: Optional[Dict] = None) -> pd.DataFrame:
    """Add Bollinger Bands columns to the DataFrame."""
    if config is None:
        config = {}
    window = config.get("window", 20)
    num_std = config.get("num_std", 2)

    mid = df["Close"].rolling(window).mean()
    std = df["Close"].rolling(window).std()

    df["bb_mid"] = mid
    df["bb_upper"] = mid + num_std * std
    df["bb_lower"] = mid - num_std * std
    return df
