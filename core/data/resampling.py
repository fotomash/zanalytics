import pandas as pd
from typing import Dict


def resample_all(df_m1: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Aggregate 1-minute bars to standard higher timeframes."""
    rules = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }
    tf_map = {
        'm5': '5T', 'm15': '15T', 'm30': '30T',
        'h1': '1H', 'h4': '4H', 'd1': '1D', 'w1': '1W'
    }
    df_m1 = df_m1.sort_index()
    aggregated = {'m1': df_m1}
    for tf, freq in tf_map.items():
        df_tf = df_m1.resample(freq, label='right', closed='right').agg(rules)
        df_tf.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)
        df_tf['Volume'].fillna(0, inplace=True)
        aggregated[tf] = df_tf
    return aggregated


class Resampler:
    """Utility for resampling raw 1-minute bars to higher timeframes."""

    def __init__(self, tick_provider):
        self._tp = tick_provider

    def get_ohlc(self, timeframe: str) -> pd.DataFrame:
        df = self._tp.get_m1_bars()
        df = df.set_index('timestamp').sort_index()
        ohlc_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        return df.resample(timeframe).agg(ohlc_dict).dropna()


def parse_tick_csv(
    path: str,
    timestamp_col: str = 'Date',
    bid_col: str = 'Bid',
    ask_col: str = 'Ask',
    last_col: str = 'Last',
    vol_col: str = 'Volume',
    sep: str = '\t',
    tz: str | None = None
) -> pd.DataFrame:
    """Load a tick CSV/TSV and normalize columns."""
    df = pd.read_csv(
        path,
        sep=sep,
        parse_dates=[timestamp_col],
        dayfirst=False,
        infer_datetime_format=True
    )
    if last_col not in df.columns:
        df[last_col] = (df[bid_col] + df[ask_col]) / 2.0
    if vol_col not in df.columns:
        df[vol_col] = 1
    df.dropna(subset=[bid_col, ask_col, last_col], inplace=True)
    if tz is not None:
        df[timestamp_col] = df[timestamp_col].dt.tz_localize(tz)
    df.sort_values(by=timestamp_col, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df[[timestamp_col, bid_col, ask_col, last_col, vol_col]]


def ticks_to_bars(
    df_ticks: pd.DataFrame,
    timeframe: str = '1T',
    timestamp_col: str = 'Date',
    last_col: str = 'Last',
    vol_col: str = 'Volume',
    keep_empty_bars: bool = False
) -> pd.DataFrame:
    """Convert tick DataFrame to OHLCV bars."""
    df = df_ticks.set_index(timestamp_col).copy()
    ohlc = df[last_col].resample(timeframe, label='right', closed='right').ohlc()
    vol = df[vol_col].resample(timeframe, label='right', closed='right').sum().rename('Volume')
    bars = ohlc.join(vol)
    if keep_empty_bars:
        bars = bars.asfreq(timeframe)
        for col in ['open', 'high', 'low', 'close']:
            bars[col].ffill(inplace=True)
        bars['Volume'].fillna(0, inplace=True)
    else:
        bars.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
    bars.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    return bars
