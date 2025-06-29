# Stage 1: Data Processing Pipeline
# zanalytics_data_pipeline.py

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import yfinance as yf
from dataclasses import dataclass, asdict
import asyncio
import aiohttp
import ccxt.async_support as ccxt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Structured market data container"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str
    timeframe: str
    indicators: Dict[str, Any] = None
    metadata: Dict[str, Any] = None

class DataProcessor:
    """Core data processing pipeline for zanalytics"""

    def __init__(self, config_path: str = "config.json"):
        self.config = self._load_config(config_path)
        self.data_dir = Path(self.config.get("data_dir", "./data"))
        self.output_dir = Path(self.config.get("output_dir", "./output"))
        self.enriched_dir = Path(self.config.get("enriched_dir", "./enriched"))

        # Create directories
        for dir_path in [self.data_dir, self.output_dir, self.enriched_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        self.exchange = None
        self.symbols = self.config.get("symbols", ["BTC/USDT", "ETH/USDT"])
        self.timeframes = self.config.get("timeframes", ["1m", "5m", "15m", "1h", "4h", "1d"])

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using defaults.")
            return self._get_default_config()

    def _get_default_config(self) -> Dict:
        """Default configuration"""
        return {
            "data_dir": "./data",
            "output_dir": "./output",
            "enriched_dir": "./enriched",
            "symbols": ["BTC/USDT", "ETH/USDT"],
            "timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"],
            "exchange": "binance",
            "indicators": {
                "sma": [20, 50, 200],
                "ema": [9, 21, 55],
                "rsi": 14,
                "macd": [12, 26, 9],
                "bollinger": [20, 2],
                "volume_profile": True,
                "market_profile": True
            }
        }

    async def initialize_exchange(self):
        """Initialize cryptocurrency exchange connection"""
        exchange_id = self.config.get("exchange", "binance")
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class({
            'apiKey': self.config.get("api_key"),
            'secret': self.config.get("api_secret"),
            'enableRateLimit': True,
        })

    def process_csv_file(self, file_path: str) -> pd.DataFrame:
        """Process raw CSV file and standardize format"""
        logger.info(f"Processing CSV file: {file_path}")

        try:
            # Read CSV with various possible formats
            df = pd.read_csv(file_path, parse_dates=True)

            # Standardize column names
            column_mapping = {
                'time': 'timestamp', 'date': 'timestamp', 'datetime': 'timestamp',
                'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume',
                'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
            }

            df.rename(columns=column_mapping, inplace=True)

            # Ensure timestamp is datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)

            # Sort by index
            df.sort_index(inplace=True)

            # Remove duplicates
            df = df[~df.index.duplicated(keep='first')]

            logger.info(f"Processed {len(df)} rows from {file_path}")
            return df

        except Exception as e:
            logger.error(f"Error processing CSV file {file_path}: {e}")
            raise

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        logger.info("Calculating technical indicators")

        # Simple Moving Averages
        for period in self.config["indicators"]["sma"]:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()

        # Exponential Moving Averages
        for period in self.config["indicators"]["ema"]:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

        # RSI
        rsi_period = self.config["indicators"]["rsi"]
        df['rsi'] = self._calculate_rsi(df['close'], rsi_period)

        # MACD
        fast, slow, signal = self.config["indicators"]["macd"]
        df['macd'], df['macd_signal'], df['macd_hist'] = self._calculate_macd(df['close'], fast, slow, signal)

        # Bollinger Bands
        period, std_dev = self.config["indicators"]["bollinger"]
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self._calculate_bollinger_bands(df['close'], period, std_dev)

        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']

        # Price action
        df['hl2'] = (df['high'] + df['low']) / 2
        df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
        df['ohlc4'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4

        # ATR (Average True Range)
        df['atr'] = self._calculate_atr(df, period=14)

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist

    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(period).mean()
        return atr

    def enrich_with_market_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Enrich data with additional market information"""
        logger.info(f"Enriching data with market information for {symbol}")

        # Add market structure
        df['pivot_high'] = df['high'].rolling(window=5, center=True).max() == df['high']
        df['pivot_low'] = df['low'].rolling(window=5, center=True).min() == df['low']

        # Support and Resistance levels
        df['resistance'] = df[df['pivot_high']]['high'].rolling(window=20).max()
        df['support'] = df[df['pivot_low']]['low'].rolling(window=20).min()
        df['resistance'].fillna(method='ffill', inplace=True)
        df['support'].fillna(method='ffill', inplace=True)

        # Trend detection
        df['trend'] = np.where(df['ema_9'] > df['ema_21'], 1, 
                              np.where(df['ema_9'] < df['ema_21'], -1, 0))

        # Volume analysis
        df['volume_trend'] = df['volume'].rolling(window=20).mean()
        df['high_volume'] = df['volume'] > df['volume_trend'] * 1.5

        # Volatility
        df['volatility'] = df['close'].pct_change().rolling(window=20).std() * np.sqrt(252)

        # Market regime
        df['market_regime'] = self._detect_market_regime(df)

        return df

    def _detect_market_regime(self, df: pd.DataFrame) -> pd.Series:
        """Detect market regime (trending, ranging, volatile)"""
        # ADX for trend strength (simplified)
        df['adx'] = self._calculate_adx(df)

        # Volatility measure
        volatility = df['atr'] / df['close']

        # Define regimes
        conditions = [
            (df['adx'] > 25) & (df['trend'] == 1),  # Strong uptrend
            (df['adx'] > 25) & (df['trend'] == -1),  # Strong downtrend
            (df['adx'] < 20) & (volatility < volatility.rolling(50).mean()),  # Ranging
            (volatility > volatility.rolling(50).mean() * 1.5),  # High volatility
        ]

        choices = ['trending_up', 'trending_down', 'ranging', 'volatile']

        return pd.Series(np.select(conditions, choices, default='undefined'), index=df.index)

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index (simplified)"""
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()

        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        tr = self._calculate_atr(df, 1)  # True Range

        plus_di = 100 * (plus_dm.rolling(period).mean() / tr.rolling(period).mean())
        minus_di = 100 * (minus_dm.rolling(period).mean() / tr.rolling(period).mean())

        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()

        return adx

    def save_enriched_data(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Save enriched data in multiple formats"""
        base_name = f"{symbol.replace('/', '_')}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Save as CSV
        csv_path = self.enriched_dir / f"{base_name}.csv"
        df.to_csv(csv_path)
        logger.info(f"Saved CSV: {csv_path}")

        # Save as JSON with metadata
        json_data = {
            "metadata": {
                "symbol": symbol,
                "timeframe": timeframe,
                "start_date": str(df.index.min()),
                "end_date": str(df.index.max()),
                "total_records": len(df),
                "indicators": list(df.columns),
                "generated_at": datetime.now().isoformat()
            },
            "data": df.reset_index().to_dict(orient='records')
        }

        json_path = self.enriched_dir / f"{base_name}.json"
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        logger.info(f"Saved JSON: {json_path}")

        # Save summary statistics
        summary = {
            "symbol": symbol,
            "timeframe": timeframe,
            "period": f"{df.index.min()} to {df.index.max()}",
            "statistics": {
                "price": {
                    "mean": float(df['close'].mean()),
                    "std": float(df['close'].std()),
                    "min": float(df['close'].min()),
                    "max": float(df['close'].max()),
                    "current": float(df['close'].iloc[-1])
                },
                "volume": {
                    "mean": float(df['volume'].mean()),
                    "total": float(df['volume'].sum())
                },
                "indicators": {
                    "current_rsi": float(df['rsi'].iloc[-1]) if 'rsi' in df else None,
                    "current_trend": df['trend'].iloc[-1] if 'trend' in df else None,
                    "current_regime": df['market_regime'].iloc[-1] if 'market_regime' in df else None
                }
            }
        }

        summary_path = self.enriched_dir / f"{base_name}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved summary: {summary_path}")

        return csv_path, json_path, summary_path

    async def fetch_live_data(self, symbol: str, timeframe: str, limit: int = 1000):
        """Fetch live data from exchange"""
        if not self.exchange:
            await self.initialize_exchange()

        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            raise

    async def process_all_symbols(self):
        """Process all configured symbols"""
        for symbol in self.symbols:
            for timeframe in self.timeframes:
                try:
                    # Fetch live data
                    df = await self.fetch_live_data(symbol, timeframe)

                    # Calculate indicators
                    df = self.calculate_indicators(df)

                    # Enrich with market data
                    df = self.enrich_with_market_data(df, symbol)

                    # Save enriched data
                    self.save_enriched_data(df, symbol, timeframe)

                except Exception as e:
                    logger.error(f"Error processing {symbol} {timeframe}: {e}")
                    continue

        if self.exchange:
            await self.exchange.close()

# Main execution
async def main():
    """Main execution function"""
    processor = DataProcessor()
    await processor.process_all_symbols()

if __name__ == "__main__":
    asyncio.run(main())
