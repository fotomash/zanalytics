
# liquidity_detector.py

def detect_swing_highs_lows(data, lookback=3):
    swings = []
    for i in range(lookback, len(data) - lookback):
        high = data[i]['high']
        low = data[i]['low']
        is_swing_high = all(high > data[i - j]['high'] and high > data[i + j]['high'] for j in range(1, lookback + 1))
        is_swing_low = all(low < data[i - j]['low'] and low < data[i + j]['low'] for j in range(1, lookback + 1))
        if is_swing_high:
            swings.append({'index': i, 'type': 'high', 'price': high})
        elif is_swing_low:
            swings.append({'index': i, 'type': 'low', 'price': low})
    return swings

def detect_session_highs_lows(data, session_start, session_end):
    session_data = [bar for bar in data if session_start <= bar['timestamp'].time() <= session_end]
    if not session_data:
        return {}
    high = max(session_data, key=lambda x: x['high'])['high']
    low = min(session_data, key=lambda x: x['low'])['low']
    return {'session_high': high, 'session_low': low}
