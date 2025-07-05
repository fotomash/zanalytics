import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from typing import List, Tuple


def analyze_smc(df: pd.DataFrame) -> pd.DataFrame:
    """Comprehensive SMC analysis returning DataFrame with enrichment columns."""
    df = _identify_market_structure(df)
    df = _find_order_blocks(df)
    df = _detect_fair_value_gaps(df)
    df = _identify_liquidity_zones(df)
    df = _detect_break_of_structure(df)
    df = _find_premium_discount_zones(df)
    return df


def _identify_market_structure(df: pd.DataFrame) -> pd.DataFrame:
    """Identify market structure (HH, HL, LH, LL)."""
    try:
        highs = find_peaks(df['high'].values, distance=5)[0]
        lows = find_peaks(-df['low'].values, distance=5)[0]
        df['SMC_swing_high'] = False
        df['SMC_swing_low'] = False
        df['SMC_structure'] = 'neutral'
        if len(highs) > 0:
            df.iloc[highs, df.columns.get_loc('SMC_swing_high')] = True
        if len(lows) > 0:
            df.iloc[lows, df.columns.get_loc('SMC_swing_low')] = True
        all_points = []
        for h in highs:
            all_points.append((h, df['high'].iloc[h], 'high'))
        for l in lows:
            all_points.append((l, df['low'].iloc[l], 'low'))
        all_points.sort(key=lambda x: x[0])
        if len(all_points) >= 4:
            df['SMC_structure'] = _analyze_structure_pattern(all_points)
    except Exception:
        pass
    return df


def _analyze_structure_pattern(points: List[Tuple]) -> str:
    """Analyze HH, HL, LH, LL pattern."""
    if len(points) < 4:
        return 'insufficient_data'
    recent_points = points[-4:]
    pattern: List[str] = []
    for i in range(1, len(recent_points)):
        curr_point = recent_points[i]
        prev_point = recent_points[i-1]
        if curr_point[2] == 'high':
            if prev_point[2] == 'low':
                last_high = None
                for j in range(i-1, -1, -1):
                    if recent_points[j][2] == 'high':
                        last_high = recent_points[j]
                        break
                if last_high and curr_point[1] > last_high[1]:
                    pattern.append('HH')
                elif last_high:
                    pattern.append('LH')
        elif curr_point[2] == 'low':
            if prev_point[2] == 'high':
                last_low = None
                for j in range(i-1, -1, -1):
                    if recent_points[j][2] == 'low':
                        last_low = recent_points[j]
                        break
                if last_low and curr_point[1] > last_low[1]:
                    pattern.append('HL')
                elif last_low:
                    pattern.append('LL')
    if 'HH' in pattern and 'HL' in pattern:
        return 'bullish'
    if 'LH' in pattern and 'LL' in pattern:
        return 'bearish'
    return 'neutral'


def _find_order_blocks(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df['SMC_bullish_ob'] = False
        df['SMC_bearish_ob'] = False
        df['SMC_ob_strength'] = 0.0
        for i in range(20, len(df)-20):
            recent_low = df['low'].iloc[i-10:i].min()
            current_high = df['high'].iloc[i]
            future_high = df['high'].iloc[i:i+20].max()
            if current_high > recent_low * 1.01 and future_high > current_high * 1.005:
                df.iloc[i, df.columns.get_loc('SMC_bullish_ob')] = True
                strength = (current_high - recent_low) / recent_low
                df.iloc[i, df.columns.get_loc('SMC_ob_strength')] = strength
            recent_high = df['high'].iloc[i-10:i].max()
            current_low = df['low'].iloc[i]
            future_low = df['low'].iloc[i:i+20].min()
            if current_low < recent_high * 0.99 and future_low < current_low * 0.995:
                df.iloc[i, df.columns.get_loc('SMC_bearish_ob')] = True
                strength = (recent_high - current_low) / recent_high
                df.iloc[i, df.columns.get_loc('SMC_ob_strength')] = -strength
    except Exception:
        pass
    return df


def _detect_fair_value_gaps(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df['SMC_fvg_bullish'] = False
        df['SMC_fvg_bearish'] = False
        df['SMC_fvg_size'] = 0.0
        for i in range(2, len(df)):
            if df['low'].iloc[i-2] > df['high'].iloc[i]:
                gap_size = df['low'].iloc[i-2] - df['high'].iloc[i]
                df.iloc[i, df.columns.get_loc('SMC_fvg_bullish')] = True
                df.iloc[i, df.columns.get_loc('SMC_fvg_size')] = gap_size
            elif df['high'].iloc[i-2] < df['low'].iloc[i]:
                gap_size = df['low'].iloc[i] - df['high'].iloc[i-2]
                df.iloc[i, df.columns.get_loc('SMC_fvg_bearish')] = True
                df.iloc[i, df.columns.get_loc('SMC_fvg_size')] = -gap_size
    except Exception:
        pass
    return df


def _identify_liquidity_zones(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df['SMC_liquidity_grab'] = False
        df['SMC_liquidity_strength'] = 0.0
        rolling_high = df['high'].rolling(window=20).max()
        rolling_low = df['low'].rolling(window=20).min()
        for i in range(20, len(df)-5):
            if (df['low'].iloc[i] < rolling_low.iloc[i-1] and df['close'].iloc[i+5] > df['open'].iloc[i]):
                df.iloc[i, df.columns.get_loc('SMC_liquidity_grab')] = True
                strength = (df['close'].iloc[i+5] - df['low'].iloc[i]) / df['low'].iloc[i]
                df.iloc[i, df.columns.get_loc('SMC_liquidity_strength')] = strength
            elif (df['high'].iloc[i] > rolling_high.iloc[i-1] and df['close'].iloc[i+5] < df['open'].iloc[i]):
                df.iloc[i, df.columns.get_loc('SMC_liquidity_grab')] = True
                strength = (df['high'].iloc[i] - df['close'].iloc[i+5]) / df['high'].iloc[i]
                df.iloc[i, df.columns.get_loc('SMC_liquidity_strength')] = -strength
    except Exception:
        pass
    return df


def _detect_break_of_structure(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df['SMC_bos_bullish'] = False
        df['SMC_bos_bearish'] = False
        window = 50
        for i in range(window, len(df)-window):
            current_price = df['close'].iloc[i]
            resistance = df['high'].iloc[i-window:i].max()
            if current_price > resistance * 1.001:
                df.iloc[i, df.columns.get_loc('SMC_bos_bullish')] = True
            support = df['low'].iloc[i-window:i].min()
            if current_price < support * 0.999:
                df.iloc[i, df.columns.get_loc('SMC_bos_bearish')] = True
    except Exception:
        pass
    return df


def _find_premium_discount_zones(df: pd.DataFrame) -> pd.DataFrame:
    try:
        window = 100
        df['SMC_range_high'] = df['high'].rolling(window=window).max()
        df['SMC_range_low'] = df['low'].rolling(window=window).min()
        df['SMC_range_mid'] = (df['SMC_range_high'] + df['SMC_range_low']) / 2
        premium_threshold = df['SMC_range_low'] + 0.7 * (df['SMC_range_high'] - df['SMC_range_low'])
        df['SMC_premium_zone'] = df['close'] > premium_threshold
        discount_threshold = df['SMC_range_low'] + 0.3 * (df['SMC_range_high'] - df['SMC_range_low'])
        df['SMC_discount_zone'] = df['close'] < discount_threshold
        df['SMC_equilibrium'] = ~(df['SMC_premium_zone'] | df['SMC_discount_zone'])
    except Exception:
        pass
    return df

