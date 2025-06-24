
# liquidity_detector.py

from typing import List, Dict, Optional
import pandas as pd
from core.lookback_adapter import adapt_lookback


def detect_swing_highs_lows(data: List[Dict], lookback: int = 3, config: Optional[Dict] = None) -> List[Dict]:
    """Detect swing highs and lows.

    When ``config['dynamic_lookback']`` is true the window size is
    adapted using :func:`core.lookback_adapter.adapt_lookback` based on
    recent volatility.
    """
    if config and config.get("dynamic_lookback"):
        df = pd.DataFrame(data)
        lookback = adapt_lookback(
            df,
            base_lookback=lookback,
            min_lookback=config.get("min_lookback", lookback),
            max_lookback=config.get("max_lookback", lookback),
            vol_config=config.get("volatility_config", {}),
        )

    swings: List[Dict] = []
    for i in range(lookback, len(data) - lookback):
        high = data[i]["high"]
        low = data[i]["low"]
        is_swing_high = all(
            high > data[i - j]["high"] and high > data[i + j]["high"]
            for j in range(1, lookback + 1)
        )
        is_swing_low = all(
            low < data[i - j]["low"] and low < data[i + j]["low"]
            for j in range(1, lookback + 1)
        )
        if is_swing_high:
            swings.append({"index": i, "type": "high", "price": high})
        elif is_swing_low:
            swings.append({"index": i, "type": "low", "price": low})
    return swings

def detect_session_highs_lows(data, session_start, session_end):
    session_data = [bar for bar in data if session_start <= bar['timestamp'].time() <= session_end]
    if not session_data:
        return {}
    high = max(session_data, key=lambda x: x['high'])['high']
    low = min(session_data, key=lambda x: x['low'])['low']
    return {'session_high': high, 'session_low': low}
