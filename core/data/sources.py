import os
import requests
import pandas as pd
from datetime import datetime
from typing import Optional

FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY')


def fetch_finnhub_m1(symbol: str, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
    """Fetch M1 candles from Finnhub."""
    url = "https://finnhub.io/api/v1/crypto/candle"
    params = {
        "symbol": symbol,
        "resolution": 1,
        "from": int(start.timestamp()),
        "to": int(end.timestamp()),
        "token": FINNHUB_API_KEY,
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if data.get("s") != "ok":
            return None
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(data.get("t", []), unit="s", utc=True),
            "open": data.get("o", []),
            "high": data.get("h", []),
            "low": data.get("l", []),
            "close": data.get("c", []),
            "volume": data.get("v", []),
        })
        return df
    except Exception:
        return None


def fetch_local_m1(symbol: str, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
    """Alias for fetch_finnhub_m1 for backward compatibility."""
    return fetch_finnhub_m1(symbol, start, end)
