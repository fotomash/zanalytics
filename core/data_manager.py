import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, timezone

try:
    from core.finnhub_data_fetcher import load_and_aggregate_m1
except Exception:
    load_and_aggregate_m1 = None

try:
    from core import m1_data_fetcher
except Exception:
    m1_data_fetcher = None

class DataManager:
    """Unified interface for OHLCV data fetching and resampling."""

    def __init__(self, m1_dir: str = "tick_data/m1", cache_enabled: bool = True):
        self.m1_dir = Path(m1_dir)
        self.m1_dir.mkdir(parents=True, exist_ok=True)
        self.cache_enabled = cache_enabled
        self._cache = {}

    # --------------------------------------------------------------
    def _resample_all(self, df_m1: pd.DataFrame) -> dict:
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

    # --------------------------------------------------------------
    def _fetch_via_finnhub(self, symbol: str, start: datetime, end: datetime):
        if not load_and_aggregate_m1:
            return None, "Finnhub fetcher unavailable"
        result = load_and_aggregate_m1(symbol, start, end)
        if result.get('status') != 'ok':
            return None, result.get('message', 'fetch failed')
        return result.get('data'), None

    def _fetch_via_local(self, symbol: str, start: datetime, end: datetime):
        if not m1_data_fetcher:
            return None, "Local fetcher unavailable"
        try:
            df = m1_data_fetcher._query(symbol, start, end)
            df.rename(columns={
                'timestamp': 'Timestamp',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }, inplace=True)
            df.set_index('Timestamp', inplace=True)
            return self._resample_all(df), None
        except Exception as e:
            return None, str(e)

    # --------------------------------------------------------------
    def get_data(self, symbol: str, timeframe: str, days_back: int = 5) -> pd.DataFrame:
        """Return OHLCV data for a symbol/timeframe."""
        key = (symbol, timeframe)
        if self.cache_enabled and key in self._cache:
            return self._cache[key]
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=days_back)
        data, err = self._fetch_via_finnhub(symbol, start_dt, end_dt)
        if data is None:
            data, err = self._fetch_via_local(symbol, start_dt, end_dt)
            if data is None:
                print(f"[DataManager] Failed fetching {symbol}: {err}")
                return pd.DataFrame()
        if timeframe not in data:
            if 'm1' in data:
                data = self._resample_all(data['m1'])
            else:
                print(f"[DataManager] Timeframe {timeframe} not available for {symbol}")
                return pd.DataFrame()
        df = data.get(timeframe, pd.DataFrame()).copy()
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame()
        df.index = pd.to_datetime(df.index)
        df.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)
        df['Volume'].fillna(0, inplace=True)
        df.sort_index(inplace=True)
        if self.cache_enabled:
            self._cache[key] = df
        return df

    def resample_csv_directory(self, m1_dir: str, output_dir: str, keep_empty: bool = False) -> None:
        """Resample all CSV files in a directory to higher timeframes."""
        m1_path = Path(m1_dir)
        out_path = Path(output_dir)
        timeframes = {
            'M5': '5T', 'M15': '15T', 'M30': '30T',
            'H1': '1H', 'H4': '4H', 'H12': '12H',
            'D': '1D', 'W': '1W'
        }
        for csv_file in m1_path.glob('*.csv'):
            df = pd.read_csv(csv_file, sep=',')
            if 'timestamp' not in df.columns:
                continue
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            symbol_name = Path(csv_file).stem.replace('_m1', '').replace('_M1', '')
            for tf, rule in timeframes.items():
                df_tf = df.resample(rule, label='right', closed='right').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                })
                if keep_empty:
                    df_tf = df_tf.asfreq(rule)
                    for col in ['open', 'high', 'low', 'close']:
                        df_tf[col].ffill(inplace=True)
                    df_tf['volume'].fillna(0, inplace=True)
                else:
                    df_tf.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
                save_path = out_path / f"{symbol_name}_{tf}.csv"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                df_tf.to_csv(save_path)
