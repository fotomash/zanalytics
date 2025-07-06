import pandas as pd
from typing import List


def analyze_micro_wyckoff(df: pd.DataFrame) -> pd.DataFrame:
    """Detect micro Wyckoff patterns and add enrichment columns."""
    df = _detect_accumulation_phases(df)
    df = _detect_distribution_phases(df)
    df = _analyze_volume_spread(df)
    df = _detect_spring_upthrust(df)
    return df


def _detect_accumulation_phases(df: pd.DataFrame) -> pd.DataFrame:
    try:
        window = 50
        df['wyckoff_accumulation'] = False
        df['wyckoff_acc_strength'] = 0.0
        for i in range(window, len(df)-window):
            price_range = df['high'].iloc[i-window:i+window].max() - df['low'].iloc[i-window:i+window].min()
            avg_price = df['close'].iloc[i-window:i+window].mean()
            if price_range < avg_price * 0.02:
                if 'volume' in df.columns and df['volume'].sum() > 0:
                    early_vol = df['volume'].iloc[i-window:i-window//2].mean()
                    late_vol = df['volume'].iloc[i:i+window//2].mean()
                    if late_vol > early_vol * 1.2:
                        df.iloc[i, df.columns.get_loc('wyckoff_accumulation')] = True
                        strength = late_vol / early_vol if early_vol > 0 else 1
                        df.iloc[i, df.columns.get_loc('wyckoff_acc_strength')] = strength
    except Exception:
        pass
    return df


def _detect_distribution_phases(df: pd.DataFrame) -> pd.DataFrame:
    try:
        window = 50
        df['wyckoff_distribution'] = False
        df['wyckoff_dist_strength'] = 0.0
        for i in range(window, len(df)-window):
            recent_high = df['high'].iloc[i-window:i].max()
            current_area = df['close'].iloc[i-10:i+10].mean()
            if current_area > recent_high * 0.95:
                price_range = df['high'].iloc[i-window:i+window].max() - df['low'].iloc[i-window:i+window].min()
                avg_price = df['close'].iloc[i-window:i+window].mean()
                if price_range < avg_price * 0.03:
                    if 'volume' in df.columns and df['volume'].sum() > 0:
                        current_vol = df['volume'].iloc[i-10:i+10].mean()
                        avg_vol = df['volume'].iloc[i-window:i].mean()
                        if current_vol > avg_vol * 1.5:
                            df.iloc[i, df.columns.get_loc('wyckoff_distribution')] = True
                            strength = current_vol / avg_vol if avg_vol > 0 else 1
                            df.iloc[i, df.columns.get_loc('wyckoff_dist_strength')] = strength
    except Exception:
        pass
    return df


def _analyze_volume_spread(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df['wyckoff_spread'] = df['high'] - df['low']
        df['wyckoff_spread_ma'] = df['wyckoff_spread'].rolling(window=20).mean()
        if 'volume' in df.columns and df['volume'].sum() > 0:
            df['wyckoff_volume_ma'] = df['volume'].rolling(window=20).mean()
            df['wyckoff_vs_ratio'] = df['volume'] / (df['wyckoff_spread'] + 0.0001)
            df['wyckoff_vs_ratio_ma'] = df['wyckoff_vs_ratio'].rolling(window=10).mean()
            df['wyckoff_effort'] = df['volume'] / df['wyckoff_volume_ma']
            df['wyckoff_result'] = df['wyckoff_spread'] / df['wyckoff_spread_ma']
            df['wyckoff_no_demand'] = (
                (df['wyckoff_spread'] > df['wyckoff_spread_ma']) &
                (df['volume'] < df['wyckoff_volume_ma']) &
                (df['close'] < df['open'])
            )
            df['wyckoff_no_supply'] = (
                (df['wyckoff_spread'] > df['wyckoff_spread_ma']) &
                (df['volume'] < df['wyckoff_volume_ma']) &
                (df['close'] > df['open'])
            )
    except Exception:
        pass
    return df


def _detect_spring_upthrust(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df['wyckoff_spring'] = False
        df['wyckoff_upthrust'] = False
        window = 30
        for i in range(window, len(df)-5):
            support = df['low'].iloc[i-window:i].min()
            if (
                df['low'].iloc[i] < support * 0.999 and
                df['close'].iloc[i] > df['open'].iloc[i] and
                df['close'].iloc[i+1:i+5].min() > df['low'].iloc[i]
            ):
                df.iloc[i, df.columns.get_loc('wyckoff_spring')] = True
            resistance = df['high'].iloc[i-window:i].max()
            if (
                df['high'].iloc[i] > resistance * 1.001 and
                df['close'].iloc[i] < df['open'].iloc[i] and
                df['close'].iloc[i+1:i+5].max() < df['high'].iloc[i]
            ):
                df.iloc[i, df.columns.get_loc('wyckoff_upthrust')] = True
    except Exception:
        pass
    return df

