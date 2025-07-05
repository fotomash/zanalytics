import pandas as pd
import numpy as np
import talib


def calculate_all_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate a wide range of technical indicators."""
    df = data.copy()
    required_cols = ['open', 'high', 'low', 'close']
    if not all(col in df.columns for col in required_cols):
        return df
    try:
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        open_price = df['open'].values
        volume = df['volume'].values if 'volume' in df.columns else np.ones(len(df))
        for period in [5, 8, 10, 13, 20, 21, 34, 50, 55, 89, 100, 144, 200, 233]:
            df[f'SMA_{period}'] = talib.SMA(close, timeperiod=period)
            df[f'EMA_{period}'] = talib.EMA(close, timeperiod=period)
        fib_periods = [8, 13, 21, 34, 55, 89, 144, 233]
        for period in fib_periods:
            df[f'FEMA_{period}'] = talib.EMA(close, timeperiod=period)
        macd, macdsignal, macdhist = talib.MACD(close)
        df['MACD'] = macd
        df['MACD_Signal'] = macdsignal
        df['MACD_Hist'] = macdhist
        macd_fast, macd_slow, macd_signal = talib.MACD(close, fastperiod=8, slowperiod=21, signalperiod=5)
        df['MACD_Fast'] = macd_fast
        df['MACD_Fast_Signal'] = macd_signal
        for period in [7, 14, 21, 25]:
            df[f'RSI_{period}'] = talib.RSI(close, timeperiod=period)
        slowk, slowd = talib.STOCH(high, low, close)
        df['STOCH_K'] = slowk
        df['STOCH_D'] = slowd
        fastk, fastd = talib.STOCHF(high, low, close)
        df['STOCHF_K'] = fastk
        df['STOCHF_D'] = fastd
        for period in [10, 20, 50]:
            for nbdev in [1.5, 2.0, 2.5]:
                upper, middle, lower = talib.BBANDS(close, timeperiod=period, nbdevup=nbdev, nbdevdn=nbdev)
                df[f'BB_Upper_{period}_{nbdev}'] = upper
                df[f'BB_Middle_{period}_{nbdev}'] = middle
                df[f'BB_Lower_{period}_{nbdev}'] = lower
                df[f'BB_Width_{period}_{nbdev}'] = (upper - lower) / middle
                df[f'BB_Position_{period}_{nbdev}'] = (close - lower) / (upper - lower)
        for period in [7, 14, 21, 50]:
            df[f'ATR_{period}'] = talib.ATR(high, low, close, timeperiod=period)
        df['ADX'] = talib.ADX(high, low, close, timeperiod=14)
        df['DI_Plus'] = talib.PLUS_DI(high, low, close, timeperiod=14)
        df['DI_Minus'] = talib.MINUS_DI(high, low, close, timeperiod=14)
        df['DX'] = talib.DX(high, low, close, timeperiod=14)
        df['CCI'] = talib.CCI(high, low, close, timeperiod=14)
        df['WILLR'] = talib.WILLR(high, low, close, timeperiod=14)
        df['MOM'] = talib.MOM(close, timeperiod=10)
        df['ROC'] = talib.ROC(close, timeperiod=10)
        df['CMO'] = talib.CMO(close, timeperiod=14)
        df['SAR'] = talib.SAR(high, low)
        df['TRIX'] = talib.TRIX(close, timeperiod=14)
        df['NATR'] = talib.NATR(high, low, close, timeperiod=14)
        df['TRANGE'] = talib.TRANGE(high, low, close)
        if 'volume' in df.columns and df['volume'].sum() > 0:
            df['OBV'] = talib.OBV(close, volume)
            df['AD'] = talib.AD(high, low, close, volume)
            df['ADOSC'] = talib.ADOSC(high, low, close, volume)
            df['VWAP'] = (volume * (high + low + close) / 3).cumsum() / volume.cumsum()
            df['Volume_SMA_20'] = talib.SMA(volume, timeperiod=20)
            df['Volume_Ratio'] = volume / df['Volume_SMA_20']
        patterns = [
            'DOJI', 'HAMMER', 'INVERTHAMMER', 'HANGINGMAN', 'SHOOTINGSTAR',
            'ENGULFING', 'HARAMI', 'HARAMICROSS', 'MORNINGSTAR', 'EVENINGSTAR',
            'DRAGONFLYDOJI', 'GRAVESTONEDOJI', 'MARUBOZU', 'SPINNINGTOP',
            'THREEWHITESOLDIERS', 'THREEBLACKCROWS', 'PIERCING', 'DARKCLOUDCOVER',
        ]
        for pattern in patterns:
            try:
                pattern_func = getattr(talib, f'CDL{pattern}')
                df[f'Pattern_{pattern}'] = pattern_func(open_price, high, low, close)
            except Exception:
                pass
        df['Pivot'] = (high + low + close) / 3
        df['R1'] = 2 * df['Pivot'] - low
        df['S1'] = 2 * df['Pivot'] - high
        df['R2'] = df['Pivot'] + (high - low)
        df['S2'] = df['Pivot'] - (high - low)
        df['R3'] = high + 2 * (df['Pivot'] - low)
        df['S3'] = low - 2 * (high - df['Pivot'])
        df['HL_Ratio'] = (high - low) / close
        df['Price_Change'] = close - open_price
        df['Price_Change_Pct'] = (close - open_price) / open_price * 100
        df['High_Low_Pct'] = (high - low) / low * 100
        df['Body_Size'] = abs(close - open_price)
        df['Upper_Shadow'] = high - np.maximum(close, open_price)
        df['Lower_Shadow'] = np.minimum(close, open_price) - low
        df['Trend_Strength'] = np.where(df['EMA_20'] > df['EMA_50'], 1,
                                        np.where(df['EMA_20'] < df['EMA_50'], -1, 0))
        short_emas = ['EMA_8', 'EMA_13', 'EMA_21']
        long_emas = ['EMA_34', 'EMA_55', 'EMA_89']
        short_trend = sum(
            [1 if df[ema].iloc[-1] > df[ema].iloc[-2] else -1 for ema in short_emas if ema in df.columns]
        ) / len(short_emas)
        long_trend = sum(
            [1 if df[ema].iloc[-1] > df[ema].iloc[-2] else -1 for ema in long_emas if ema in df.columns]
        ) / len(long_emas)
        df['Short_Term_Trend'] = short_trend
        df['Long_Term_Trend'] = long_trend
    except Exception:
        pass
    return df

