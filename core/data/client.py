from datetime import datetime, timedelta, timezone
import pandas as pd
from .sources import fetch_finnhub_m1, fetch_local_m1
from .resampling import resample_all


TF_MINUTES = {
    'm1': 1,
    'm5': 5,
    'm15': 15,
    'm30': 30,
    'h1': 60,
    'h4': 240,
    'd1': 1440,
    'w1': 10080,
}


def get_market_data(symbol: str, timeframe: str, bars: int = 500) -> pd.DataFrame:
    """Fetch and resample market data for the given symbol and timeframe."""
    tf_key = timeframe.lower()
    minutes = TF_MINUTES.get(tf_key)
    if minutes is None:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(minutes=minutes * bars)

    df = fetch_finnhub_m1(symbol, start_dt, end_dt)
    if df is None:
        df = fetch_local_m1(symbol, start_dt, end_dt)
    if df is None or df.empty:
        return pd.DataFrame()

    df.set_index('timestamp', inplace=True)
    aggregated = resample_all(df)
    out = aggregated.get(tf_key, pd.DataFrame())
    out = out.tail(bars)
    out.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)
    out['Volume'].fillna(0, inplace=True)
    out.sort_index(inplace=True)
    return out

