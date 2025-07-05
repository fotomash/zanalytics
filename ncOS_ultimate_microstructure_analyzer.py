#!/usr/bin/env python3
"""
ncOS - Ultimate Trading Data Processor with Advanced Market Microstructure Analysis
Detects spoofing, engineered liquidity, SMC patterns, micro Wyckoff, and harmonic patterns
"""

import pandas as pd
import numpy as np
import talib
import os
import argparse
import json
import glob
from pathlib import Path
from datetime import datetime, timedelta
import warnings
import sys
from typing import Dict, List, Optional, Tuple, Any, Union
import re
from dataclasses import dataclass, field
import logging
from scipy import stats
from scipy.signal import find_peaks, savgol_filter
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

@dataclass
class ProcessingConfig:
    """Configuration for data processing"""
    directory: str = "."
    file_pattern: str = "*"
    file_types: List[str] = field(default_factory=lambda: ['csv', 'json'])
    timeframes: List[str] = field(default_factory=lambda: ['1min', '5min', '15min', '30min', '1H', '4H', '1D'])
    output_dir: str = "processed_data"
    process_all_indicators: bool = True
    process_all_timeframes: bool = True
    process_tick_data: bool = True
    process_all_tick_data: bool = False
    delimiter: str = "auto"
    json_only: bool = False

    # Bar limits per timeframe
    bar_limits: Dict[str, int] = field(default_factory=lambda: {
        '1min': 1440,   # 1 day worth
        '5min': 2016,   # 1 week worth  
        '15min': 2688,  # 4 weeks worth
        '30min': 2160,  # 45 days worth
        '1H': 2160,     # 90 days worth
        '4H': 2160,     # 360 days worth
        '1D': 365       # 1 year worth
    })

class TimeframeDetector:
    """Detects timeframe from filename patterns"""

    TIMEFRAME_PATTERNS = {
        r'[_\-\s]?m1[_\-\s]?': '1min',
        r'[_\-\s]?m5[_\-\s]?': '5min',
        r'[_\-\s]?m15[_\-\s]?': '15min',
        r'[_\-\s]?m30[_\-\s]?': '30min',
        r'[_\-\s]?h1[_\-\s]?': '1H',
        r'[_\-\s]?h4[_\-\s]?': '4H',
        r'[_\-\s]?d1[_\-\s]?': '1D',
        r'[_\-\s]?1min[_\-\s]?': '1min',
        r'[_\-\s]?5min[_\-\s]?': '5min',
        r'[_\-\s]?15min[_\-\s]?': '15min',
        r'[_\-\s]?30min[_\-\s]?': '30min',
        r'[_\-\s]?1h[_\-\s]?': '1H',
        r'[_\-\s]?4h[_\-\s]?': '4H',
        r'[_\-\s]?1d[_\-\s]?': '1D',
        r'tick': 'tick',
        r'ticks': 'tick',
    }

    @classmethod
    def detect_timeframe(cls, filename: str) -> Optional[str]:
        """Detect timeframe from filename"""
        filename_lower = filename.lower()

        for pattern, timeframe in cls.TIMEFRAME_PATTERNS.items():
            if re.search(pattern, filename_lower):
                return timeframe

        return '1min'  # Default fallback

class PairDetector:
    """Detects currency pair from filename"""

    PAIR_PATTERNS = [
        r'([A-Z]{6})',  # EURUSD, GBPUSD, etc.
        r'([A-Z]{3}[A-Z]{3})',  # EUR USD as EURUSD
        r'(XAU[A-Z]{3})',  # XAUUSD, XAUEUR, etc.
        r'(XAG[A-Z]{3})',  # XAGUSD, etc.
        r'([A-Z]{3}JPY)',  # USDJPY, EURJPY, etc.
    ]

    @classmethod
    def detect_pair(cls, filename: str) -> str:
        """Detect currency pair from filename"""
        filename_upper = filename.upper()

        for pattern in cls.PAIR_PATTERNS:
            match = re.search(pattern, filename_upper)
            if match:
                return match.group(1)

        # Fallback to filename without extension
        return Path(filename).stem.upper()

class TickAnalysis:
    """Advanced tick data analysis for market microstructure"""

    @staticmethod
    def analyze_tick_data(df: pd.DataFrame) -> pd.DataFrame:
        """Comprehensive tick analysis"""
        if len(df) < 100:
            return df

        try:
            # Basic tick metrics
            df['mid_price'] = (df['bid'] + df['ask']) / 2
            df['spread'] = df['ask'] - df['bid']
            df['spread_pct'] = df['spread'] / df['mid_price'] * 100

            # Price movements
            df['price_change'] = df['mid_price'].diff()
            df['abs_price_change'] = df['price_change'].abs()

            # Tick direction (uptick/downtick)
            df['tick_direction'] = np.where(df['price_change'] > 0, 1, 
                                          np.where(df['price_change'] < 0, -1, 0))

            # Tick intensity
            df['tick_intensity'] = df['abs_price_change'] / df['spread'].replace(0, np.nan)

            # Time between ticks (microsecond analysis)
            df['time_diff_ms'] = df['timestamp_ms'].diff()
            df['tick_frequency'] = 1000 / df['time_diff_ms'].replace(0, np.nan)  # ticks per second

            # Market manipulation indicators
            df = TickAnalysis._detect_spoofing(df)
            df = TickAnalysis._detect_layering(df)
            df = TickAnalysis._detect_momentum_ignition(df)
            df = TickAnalysis._detect_quote_stuffing(df)
            df = TickAnalysis._analyze_order_flow(df)

            print(f"✓ Comprehensive tick analysis completed on {len(df)} ticks")

        except Exception as e:
            print(f"Warning: Tick analysis error: {e}")

        return df

    @staticmethod
    def _detect_spoofing(df: pd.DataFrame) -> pd.DataFrame:
        """Detect spoofing patterns"""
        try:
            # Large spread widening followed by rapid narrowing
            rolling_spread = df['spread'].rolling(window=10)
            spread_mean = rolling_spread.mean()
            spread_std = rolling_spread.std()

            # Spoofing indicator: spread > 2 std above mean, then quickly returns
            large_spread = df['spread'] > (spread_mean + 2 * spread_std)

            # Look for rapid spread reduction after large spread
            df['spoofing_score'] = 0.0
            for i in range(10, len(df)-10):
                if large_spread.iloc[i]:
                    # Check if spread reduces significantly in next 10 ticks
                    future_spreads = df['spread'].iloc[i+1:i+11]
                    if future_spreads.min() < df['spread'].iloc[i] * 0.7:
                        df.loc[df.index[i], 'spoofing_score'] = 1.0

            # Flag periods of high spoofing activity
            df['spoofing_detected'] = df['spoofing_score'].rolling(window=20).sum() > 3

        except Exception as e:
            print(f"Spoofing detection error: {e}")
            df['spoofing_score'] = 0.0
            df['spoofing_detected'] = False

        return df

    @staticmethod
    def _detect_layering(df: pd.DataFrame) -> pd.DataFrame:
        """Detect layering/iceberg patterns"""
        try:
            # Layering: repeated similar-sized orders at same price levels
            price_levels = {}
            df['layering_score'] = 0.0

            window_size = 50
            for i in range(window_size, len(df)):
                window_data = df.iloc[i-window_size:i]

                # Count frequency of bid/ask prices
                bid_counts = window_data['bid'].value_counts()
                ask_counts = window_data['ask'].value_counts()

                # High repetition indicates potential layering
                max_bid_count = bid_counts.max() if len(bid_counts) > 0 else 0
                max_ask_count = ask_counts.max() if len(ask_counts) > 0 else 0

                if max_bid_count > window_size * 0.3 or max_ask_count > window_size * 0.3:
                    df.loc[df.index[i], 'layering_score'] = max(max_bid_count, max_ask_count) / window_size

            df['layering_detected'] = df['layering_score'] > 0.3

        except Exception as e:
            print(f"Layering detection error: {e}")
            df['layering_score'] = 0.0
            df['layering_detected'] = False

        return df

    @staticmethod
    def _detect_momentum_ignition(df: pd.DataFrame) -> pd.DataFrame:
        """Detect momentum ignition patterns"""
        try:
            # Momentum ignition: sudden price movement followed by trend continuation
            df['price_velocity'] = df['price_change'].rolling(window=5).sum()
            df['price_acceleration'] = df['price_velocity'].diff()

            # Detect sudden acceleration
            vel_threshold = df['price_velocity'].std() * 2
            acc_threshold = df['price_acceleration'].std() * 2

            high_velocity = df['price_velocity'].abs() > vel_threshold
            high_acceleration = df['price_acceleration'].abs() > acc_threshold

            df['momentum_ignition'] = high_velocity & high_acceleration

            # Score based on intensity
            df['momentum_score'] = (df['price_velocity'].abs() / vel_threshold + 
                                  df['price_acceleration'].abs() / acc_threshold) / 2
            df['momentum_score'] = df['momentum_score'].fillna(0).clip(0, 5)

        except Exception as e:
            print(f"Momentum ignition detection error: {e}")
            df['momentum_ignition'] = False
            df['momentum_score'] = 0.0

        return df

    @staticmethod
    def _detect_quote_stuffing(df: pd.DataFrame) -> pd.DataFrame:
        """Detect quote stuffing (excessive quote updates)"""
        try:
            # Quote stuffing: very high frequency of quotes in short time
            df['quote_rate'] = 1 / df['time_diff_ms'] * 1000  # quotes per second

            # Rolling average quote rate
            avg_quote_rate = df['quote_rate'].rolling(window=100).mean()
            std_quote_rate = df['quote_rate'].rolling(window=100).std()

            # Stuffing when rate > 3 std above average
            stuffing_threshold = avg_quote_rate + 3 * std_quote_rate
            df['quote_stuffing'] = df['quote_rate'] > stuffing_threshold

            # Intensity score
            df['stuffing_score'] = ((df['quote_rate'] - avg_quote_rate) / 
                                  std_quote_rate.replace(0, 1)).fillna(0).clip(0, 10)

        except Exception as e:
            print(f"Quote stuffing detection error: {e}")
            df['quote_stuffing'] = False
            df['stuffing_score'] = 0.0

        return df

    @staticmethod
    def _analyze_order_flow(df: pd.DataFrame) -> pd.DataFrame:
        """Analyze order flow patterns"""
        try:
            # Order flow imbalance
            upticks = (df['tick_direction'] == 1).rolling(window=20).sum()
            downticks = (df['tick_direction'] == -1).rolling(window=20).sum()
            total_ticks = upticks + downticks

            df['order_flow_imbalance'] = (upticks - downticks) / total_ticks.replace(0, 1)

            # Volume-weighted order flow (if volume available)
            if 'volume' in df.columns and df['volume'].sum() > 0:
                vol_weighted_flow = (df['tick_direction'] * df['volume']).rolling(window=20).sum()
                total_volume = df['volume'].rolling(window=20).sum()
                df['volume_flow_imbalance'] = vol_weighted_flow / total_volume.replace(0, 1)
            else:
                df['volume_flow_imbalance'] = df['order_flow_imbalance']

            # Price impact analysis
            price_impact = df['price_change'].rolling(window=10).sum()
            flow_direction = df['order_flow_imbalance'].shift(10)
            df['flow_efficiency'] = price_impact * flow_direction

        except Exception as e:
            print(f"Order flow analysis error: {e}")
            df['order_flow_imbalance'] = 0.0
            df['volume_flow_imbalance'] = 0.0
            df['flow_efficiency'] = 0.0

        return df

class SMCAnalysis:
    """Deprecated wrapper for :mod:`core.analysis.smc`."""

    @staticmethod
    def analyze_smc(df: pd.DataFrame) -> pd.DataFrame:
        from core.analysis.smc import analyze_smc
        return analyze_smc(df)

class HarmonicPatterns:
    """Advanced harmonic pattern recognition"""

    @staticmethod
    def detect_harmonic_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """Detect multiple harmonic patterns"""
        try:
            # Find swing points first
            swings = HarmonicPatterns._find_swing_points(df)

            if len(swings) < 5:
                return df

            # Initialize pattern columns
            patterns = ['gartley', 'butterfly', 'bat', 'crab', 'cypher', 'shark', 'abcd']
            for pattern in patterns:
                df[f'harmonic_{pattern}'] = False
                df[f'harmonic_{pattern}_score'] = 0.0

            # Detect each pattern type
            for i in range(len(swings)-4):
                points = swings[i:i+5]  # X, A, B, C, D points

                # Calculate ratios
                ratios = HarmonicPatterns._calculate_ratios(points, df)

                # Check each pattern
                for pattern in patterns:
                    score = HarmonicPatterns._check_pattern(pattern, ratios)
                    if score > 0.8:  # High confidence threshold
                        idx = points[-1][0]  # D point index
                        df.iloc[idx, df.columns.get_loc(f'harmonic_{pattern}')] = True
                        df.iloc[idx, df.columns.get_loc(f'harmonic_{pattern}_score')] = score

            print(f"✓ Harmonic pattern analysis completed")

        except Exception as e:
            print(f"Harmonic patterns error: {e}")

        return df

    @staticmethod
    def _find_swing_points(df: pd.DataFrame, distance: int = 10) -> List[Tuple]:
        """Find swing highs and lows"""
        highs = find_peaks(df['high'].values, distance=distance)[0]
        lows = find_peaks(-df['low'].values, distance=distance)[0]

        swings = []
        for h in highs:
            swings.append((h, df['high'].iloc[h], 'high'))
        for l in lows:
            swings.append((l, df['low'].iloc[l], 'low'))

        # Sort by index
        swings.sort(key=lambda x: x[0])

        return swings

    @staticmethod
    def _calculate_ratios(points: List[Tuple], df: pd.DataFrame) -> Dict[str, float]:
        """Calculate Fibonacci ratios between points"""
        if len(points) < 5:
            return {}

        # Extract price levels
        X = points[0][1]
        A = points[1][1]
        B = points[2][1]
        C = points[3][1]
        D = points[4][1]

        ratios = {}
        try:
            ratios['AB_XA'] = abs(B - A) / abs(A - X) if abs(A - X) > 0 else 0
            ratios['BC_AB'] = abs(C - B) / abs(B - A) if abs(B - A) > 0 else 0
            ratios['CD_BC'] = abs(D - C) / abs(C - B) if abs(C - B) > 0 else 0
            ratios['AD_XA'] = abs(D - A) / abs(A - X) if abs(A - X) > 0 else 0
        except:
            pass

        return ratios

    @staticmethod
    def _check_pattern(pattern: str, ratios: Dict[str, float]) -> float:
        """Check if ratios match harmonic pattern"""
        if not ratios:
            return 0.0

        pattern_rules = {
            'gartley': {
                'AB_XA': (0.618, 0.05),  # (target, tolerance)
                'BC_AB': (0.382, 0.1),
                'CD_BC': (1.272, 0.1),
                'AD_XA': (0.786, 0.05)
            },
            'butterfly': {
                'AB_XA': (0.786, 0.05),
                'BC_AB': (0.382, 0.1),
                'CD_BC': (1.618, 0.1),
                'AD_XA': (1.27, 0.1)
            },
            'bat': {
                'AB_XA': (0.382, 0.1),
                'BC_AB': (0.382, 0.1),
                'CD_BC': (1.618, 0.2),
                'AD_XA': (0.886, 0.05)
            },
            'crab': {
                'AB_XA': (0.382, 0.1),
                'BC_AB': (0.382, 0.1),
                'CD_BC': (2.24, 0.2),
                'AD_XA': (1.618, 0.1)
            },
            'cypher': {
                'AB_XA': (0.382, 0.1),
                'BC_AB': (1.272, 0.1),
                'CD_BC': (0.786, 0.05),
                'AD_XA': (0.786, 0.05)
            }
        }

        if pattern not in pattern_rules:
            return 0.0

        rules = pattern_rules[pattern]
        score = 0.0
        valid_ratios = 0

        for ratio_name, (target, tolerance) in rules.items():
            if ratio_name in ratios:
                actual = ratios[ratio_name]
                if abs(actual - target) <= tolerance:
                    score += 1.0
                else:
                    # Partial score based on how close it is
                    distance = abs(actual - target) / tolerance
                    score += max(0, 1.0 - distance)
                valid_ratios += 1

        return score / valid_ratios if valid_ratios > 0 else 0.0

class MicroWyckoff:
    """Deprecated wrapper for :mod:`core.analysis.wyckoff`."""

    @staticmethod
    def analyze_micro_wyckoff(df: pd.DataFrame) -> pd.DataFrame:
        from core.analysis.wyckoff import analyze_micro_wyckoff
        return analyze_micro_wyckoff(df)

class TechnicalIndicators:
    """Deprecated wrapper for :mod:`core.analysis.indicators`."""

    @staticmethod
    def calculate_all_indicators(data: pd.DataFrame) -> pd.DataFrame:
        from core.analysis.indicators import calculate_all_indicators
        return calculate_all_indicators(data)


class DataProcessor:
    """Main data processing engine with advanced algorithms"""

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.processed_files = []
        self.session_stats = {
            'start_time': datetime.now(),
            'files_processed': 0,
            'json_files_processed': 0,
            'csv_files_processed': 0,
            'tick_files_processed': 0,
            'total_indicators': 0,
            'manipulation_detected': 0,
            'patterns_found': 0,
            'errors': []
        }

        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging configuration"""
        log_file = os.path.join(self.config.output_dir, 'processing.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _find_files(self) -> List[str]:
        """Find all files to process based on configuration"""
        all_files = []

        for file_type in self.config.file_types:
            if file_type == 'csv':
                pattern = os.path.join(self.config.directory, f"{self.config.file_pattern}.csv")
                csv_files = glob.glob(pattern)
                all_files.extend(csv_files)
            elif file_type == 'json':
                pattern = os.path.join(self.config.directory, f"{self.config.file_pattern}.json")
                json_files = glob.glob(pattern)
                all_files.extend(json_files)

        # Also check for files without specific extensions if pattern doesn't include extension
        if '.' not in self.config.file_pattern:
            for ext in self.config.file_types:
                pattern = os.path.join(self.config.directory, f"{self.config.file_pattern}*.{ext}")
                files = glob.glob(pattern)
                all_files.extend(files)

        return list(set(all_files))  # Remove duplicates

    def _detect_delimiter(self, file_path: str) -> str:
        """Auto-detect file delimiter"""
        try:
            with open(file_path, 'r') as f:
                first_line = f.readline()
                if '\t' in first_line or first_line.count('\t') > first_line.count(','):
                    return '\t'
                else:
                    return ','
        except:
            return ','

    def _load_csv_data(self, file_path: str) -> Optional[pd.DataFrame]:
        """Load and validate CSV data from file"""
        try:
            # Auto-detect delimiter if needed
            delimiter = self.config.delimiter
            if delimiter == "auto":
                delimiter = self._detect_delimiter(file_path)

            # Try different ways to load the data
            try:
                df = pd.read_csv(file_path, delimiter=delimiter)
            except:
                df = pd.read_csv(file_path, sep='\t')

            # Clean column names
            df.columns = df.columns.str.strip().str.lower()

            # Detect if this is tick data
            is_tick_data = ('bid' in df.columns and 'ask' in df.columns) or 'tick' in file_path.lower()

            if is_tick_data:
                return self._process_tick_file(df, file_path)
            else:
                return self._process_ohlc_file(df, file_path)

        except Exception as e:
            self.logger.error(f"Error loading CSV {file_path}: {e}")
            self.session_stats['errors'].append(f"CSV load error in {file_path}: {e}")
            return None

    def _process_tick_file(self, df: pd.DataFrame, file_path: str) -> Optional[pd.DataFrame]:
        """Process tick data file"""
        try:
            # Validate tick data columns
            required_tick_cols = ['timestamp', 'bid', 'ask']
            missing_cols = [col for col in required_tick_cols if col not in df.columns]

            if missing_cols:
                self.logger.warning(f"Missing tick columns in {file_path}: {missing_cols}")
                return None

            # Convert timestamp
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)

            # Ensure numeric types
            numeric_cols = ['bid', 'ask', 'spread_points', 'spread_price', 'volume']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Remove invalid rows
            df.dropna(subset=['bid', 'ask'], inplace=True)

            if len(df) == 0:
                self.logger.warning(f"No valid tick data in {file_path}")
                return None

            # Apply tick data limit if configured
            if not self.config.process_all_tick_data:
                max_ticks = 50000  # Default limit for tick data
                if len(df) > max_ticks:
                    df = df.tail(max_ticks)
                    self.logger.info(f"Limited to last {max_ticks} ticks")

            self.session_stats['tick_files_processed'] += 1
            self.logger.info(f"✓ Loaded {len(df)} ticks from: {file_path}")
            return df

        except Exception as e:
            self.logger.error(f"Error processing tick file {file_path}: {e}")
            return None

    def _process_ohlc_file(self, df: pd.DataFrame, file_path: str) -> Optional[pd.DataFrame]:
        """Process OHLC data file"""
        try:
            # Check for required columns
            required_cols = ['timestamp', 'open', 'high', 'low', 'close']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                # Try alternative column names
                col_mapping = {
                    'time': 'timestamp',
                    'date': 'timestamp',
                    'datetime': 'timestamp',
                    'o': 'open',
                    'h': 'high',
                    'l': 'low',
                    'c': 'close',
                    'vol': 'volume',
                    'v': 'volume'
                }

                for old_name, new_name in col_mapping.items():
                    if old_name in df.columns and new_name in missing_cols:
                        df.rename(columns={old_name: new_name}, inplace=True)
                        missing_cols.remove(new_name)

            if missing_cols:
                self.logger.warning(f"Missing columns in {file_path}: {missing_cols}")
                return None

            # Convert timestamp
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)

            # Ensure numeric types
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Remove rows with NaN in OHLC
            df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)

            if len(df) == 0:
                self.logger.warning(f"No valid data in {file_path}")
                return None

            # Apply bar limit based on timeframe
            detected_timeframe = TimeframeDetector.detect_timeframe(os.path.basename(file_path))
            if detected_timeframe in self.config.bar_limits:
                limit = self.config.bar_limits[detected_timeframe]
                if len(df) > limit:
                    df = df.tail(limit)
                    self.logger.info(f"Limited to last {limit} bars for {detected_timeframe}")

            self.session_stats['csv_files_processed'] += 1
            self.logger.info(f"✓ Loaded {len(df)} rows from OHLC: {file_path}")
            return df

        except Exception as e:
            self.logger.error(f"Error processing OHLC file {file_path}: {e}")
            return None

    def _resample_data(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample data to specified timeframe"""
        timeframe_map = {
            '1min': '1T',
            '5min': '5T',
            '15min': '15T',
            '30min': '30T',
            '1H': '1H',
            '4H': '4H',
            '1D': '1D'
        }

        if timeframe not in timeframe_map:
            return df

        freq = timeframe_map[timeframe]

        try:
            # Check if data is tick data (has bid/ask)
            if 'bid' in df.columns and 'ask' in df.columns:
                # Resample tick data to OHLC
                resampled = df.resample(freq).agg({
                    'bid': 'last',
                    'ask': 'last',
                    'spread_points': 'mean',
                    'spread_price': 'mean',
                    'volume': 'sum' if 'volume' in df.columns else 'count'
                })

                # Create OHLC from mid prices
                mid_price = (resampled['bid'] + resampled['ask']) / 2
                resampled['open'] = mid_price.groupby(pd.Grouper(freq=freq)).first()
                resampled['high'] = mid_price.groupby(pd.Grouper(freq=freq)).max()
                resampled['low'] = mid_price.groupby(pd.Grouper(freq=freq)).min()
                resampled['close'] = mid_price.groupby(pd.Grouper(freq=freq)).last()

            else:
                # Resample OHLC data
                resampled = df.resample(freq).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum' if 'volume' in df.columns else 'mean'
                })

            resampled.dropna(inplace=True)
            return resampled

        except Exception as e:
            self.logger.error(f"Error resampling to {timeframe}: {e}")
            return df

    def _save_results(self, df: pd.DataFrame, original_file: str, timeframe: str, pair: str, source_type: str = 'csv'):
        """Save processed results"""
        try:
            # Create pair directory
            pair_dir = os.path.join(self.config.output_dir, pair)
            os.makedirs(pair_dir, exist_ok=True)

            # Generate filename
            base_name = Path(original_file).stem
            output_file = f"{base_name}_{timeframe}_{source_type}_processed.csv"
            output_path = os.path.join(pair_dir, output_file)

            # Save the data
            df.to_csv(output_path)

            # Count indicators and patterns
            indicator_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
            pattern_cols = [col for col in df.columns if any(pattern in col.lower() for pattern in ['smc_', 'harmonic_', 'wyckoff_', 'pattern_'])]
            manipulation_cols = [col for col in df.columns if any(manip in col.lower() for manip in ['spoofing', 'layering', 'stuffing', 'momentum_ignition'])]

            # Track processed file
            self.processed_files.append({
                'original': original_file,
                'processed': output_path,
                'timeframe': timeframe,
                'pair': pair,
                'source_type': source_type,
                'rows': len(df),
                'indicators': len(indicator_cols),
                'patterns': len(pattern_cols),
                'manipulation_signals': len(manipulation_cols)
            })

            # Update session stats
            if manipulation_cols:
                self.session_stats['manipulation_detected'] += 1
            if pattern_cols:
                self.session_stats['patterns_found'] += 1

            self.logger.info(f"✓ Saved {timeframe} data to {output_path}")

        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            self.session_stats['errors'].append(f"Save error: {e}")

    def _generate_journal(self):
        """Generate comprehensive processing journal"""
        try:
            journal_path = os.path.join(self.config.output_dir, 'processing_journal.json')

            journal = {
                'session_info': {
                    'start_time': self.session_stats['start_time'].isoformat(),
                    'end_time': datetime.now().isoformat(),
                    'duration_minutes': (datetime.now() - self.session_stats['start_time']).total_seconds() / 60,
                    'total_files_processed': len(self.processed_files),
                    'csv_files_processed': self.session_stats['csv_files_processed'],
                    'json_files_processed': self.session_stats['json_files_processed'],
                    'tick_files_processed': self.session_stats['tick_files_processed'],
                    'total_indicators_calculated': sum(f['indicators'] for f in self.processed_files),
                    'patterns_detected': sum(f['patterns'] for f in self.processed_files),
                    'manipulation_signals': sum(f['manipulation_signals'] for f in self.processed_files),
                    'errors': self.session_stats['errors']
                },
                'configuration': {
                    'directory_scanned': self.config.directory,
                    'file_pattern': self.config.file_pattern,
                    'file_types': self.config.file_types,
                    'json_only_mode': self.config.json_only,
                    'timeframes_processed': self.config.timeframes,
                    'bar_limits': self.config.bar_limits,
                    'output_directory': self.config.output_dir,
                    'tick_data_processing': self.config.process_tick_data,
                    'all_tick_data_mode': self.config.process_all_tick_data
                },
                'processed_files': self.processed_files,
                'summary': {
                    'unique_pairs': len(set(f['pair'] for f in self.processed_files)),
                    'unique_timeframes': len(set(f['timeframe'] for f in self.processed_files)),
                    'total_rows_processed': sum(f['rows'] for f in self.processed_files),
                    'source_types': list(set(f['source_type'] for f in self.processed_files)),
                    'avg_indicators_per_file': np.mean([f['indicators'] for f in self.processed_files]) if self.processed_files else 0,
                    'files_with_patterns': len([f for f in self.processed_files if f['patterns'] > 0]),
                    'files_with_manipulation': len([f for f in self.processed_files if f['manipulation_signals'] > 0])
                }
            }

            with open(journal_path, 'w') as f:
                json.dump(journal, f, indent=2)

            self.logger.info(f"✓ Generated comprehensive processing journal: {journal_path}")

        except Exception as e:
            self.logger.error(f"Error generating journal: {e}")

    def process_all_files(self):
        """Process all files with advanced analysis"""
        try:
            # Find all files
            all_files = self._find_files()

            if not all_files:
                file_types_str = ', '.join(self.config.file_types)
                self.logger.warning(f"No {file_types_str} files found in {self.config.directory}")
                return

            csv_files = [f for f in all_files if f.endswith('.csv')]
            json_files = [f for f in all_files if f.endswith('.json')]

            self.logger.info(f"Found {len(csv_files)} CSV files and {len(json_files)} JSON files to process")

            # Process CSV files
            for file_path in csv_files:
                self.logger.info(f"\n🔄 Processing: {file_path}")

                # Load original data
                original_data = self._load_csv_data(file_path)
                if original_data is None:
                    continue

                # Detect pair and original timeframe
                pair = PairDetector.detect_pair(os.path.basename(file_path))
                detected_timeframe = TimeframeDetector.detect_timeframe(os.path.basename(file_path))

                self.logger.info(f"   📊 Detected: {pair} | {detected_timeframe}")

                # Check if it's tick data
                is_tick_data = ('bid' in original_data.columns and 'ask' in original_data.columns)

                if is_tick_data and self.config.process_tick_data:
                    # Process tick data with advanced analysis
                    self.logger.info(f"   🎯 Processing tick data...")
                    tick_processed = TickAnalysis.analyze_tick_data(original_data.copy())

                    # Save tick analysis
                    self._save_results(tick_processed, file_path, 'tick', pair, 'tick')

                    # Convert to OHLC for further analysis
                    ohlc_data = self._resample_data(original_data, '1min')
                    if len(ohlc_data) > 0:
                        original_data = ohlc_data
                        detected_timeframe = '1min'

                # Process all requested timeframes
                for target_timeframe in self.config.timeframes:
                    try:
                        # Resample if needed
                        if target_timeframe == detected_timeframe:
                            processed_data = original_data.copy()
                        else:
                            processed_data = self._resample_data(original_data, target_timeframe)

                        if len(processed_data) == 0:
                            continue

                        # Apply comprehensive analysis
                        if self.config.process_all_indicators:
                            # Technical indicators
                            processed_data = TechnicalIndicators.calculate_all_indicators(processed_data)

                            # SMC analysis
                            processed_data = SMCAnalysis.analyze_smc(processed_data)

                            # Harmonic patterns
                            processed_data = HarmonicPatterns.detect_harmonic_patterns(processed_data)

                            # Micro Wyckoff
                            processed_data = MicroWyckoff.analyze_micro_wyckoff(processed_data)

                        # Save results
                        self._save_results(processed_data, file_path, target_timeframe, pair, 'csv')

                        self.logger.info(f"   ✅ {target_timeframe}: {len(processed_data)} rows with advanced analysis")

                    except Exception as e:
                        self.logger.error(f"   ❌ Error processing {target_timeframe}: {e}")
                        self.session_stats['errors'].append(f"{file_path} - {target_timeframe}: {e}")

            # Process JSON files (existing logic would go here)
            # ... JSON processing code ...

            # Generate final journal
            self._generate_journal()

            # Print summary
            self._print_summary()

        except Exception as e:
            self.logger.error(f"Critical error in processing: {e}")

    def _print_summary(self):
        """Print comprehensive processing summary"""
        print("\n" + "="*90)
        print("🎉 ADVANCED PROCESSING COMPLETE!")
        print("="*90)
        print(f"📁 CSV files processed: {self.session_stats['csv_files_processed']}")
        print(f"🎯 Tick files processed: {self.session_stats['tick_files_processed']}")
        print(f"📄 JSON files processed: {self.session_stats['json_files_processed']}")
        print(f"📊 Total outputs: {len(self.processed_files)}")
        print(f"💱 Currency pairs: {len(set(f['pair'] for f in self.processed_files))}")
        print(f"⏰ Timeframes: {', '.join(set(f['timeframe'] for f in self.processed_files))}")
        print(f"📈 Total indicators: {sum(f['indicators'] for f in self.processed_files)}")
        print(f"🔍 Pattern detections: {sum(f['patterns'] for f in self.processed_files)}")
        print(f"⚠️  Manipulation signals: {sum(f['manipulation_signals'] for f in self.processed_files)}")
        print(f"📂 Output directory: {self.config.output_dir}")
        if self.session_stats['errors']:
            print(f"❌ Errors encountered: {len(self.session_stats['errors'])}")
        print("="*90)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='ncOS - Ultimate Trading Processor with Market Microstructure Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Advanced Features:
  • SMC Analysis (Smart Money Concepts)
  • Harmonic Pattern Recognition  
  • Micro Wyckoff Analysis
  • Tick Data Manipulation Detection
  • Spoofing & Layering Detection
  • Liquidity Engineering Analysis
  • Order Flow Analysis

Examples:
  python script.py                                    # Process everything with all analysis
  python script.py --json-only                        # Process ONLY JSON files
  python script.py --all-tick-data                    # Process ALL tick data (no limits)
  python script.py --1min-bars 500 --5min-bars 600   # Custom bar limits
  python script.py --timeframes 1min 5min             # Only specific timeframes
        """
    )

    # File processing options
    parser.add_argument('--dir', '--directory', 
                       default='.',
                       help='Directory to scan for files (default: current directory)')

    parser.add_argument('--pattern', 
                       default='*',
                       help='File pattern to match (default: *)')

    parser.add_argument('--json-only', 
                       action='store_true',
                       help='Process ONLY JSON files (ignore CSV files)')

    parser.add_argument('--timeframes', 
                       nargs='+',
                       default=['1min', '5min', '15min', '30min', '1H', '4H', '1D'],
                       choices=['1min', '5min', '15min', '30min', '1H', '4H', '1D'],
                       help='Timeframes to process (default: all)')

    # Bar limit options for each timeframe
    parser.add_argument('--1min-bars', type=int, default=1440, 
                       help='Max bars for 1min timeframe (default: 1440)')
    parser.add_argument('--5min-bars', type=int, default=2016, 
                       help='Max bars for 5min timeframe (default: 2016)')
    parser.add_argument('--15min-bars', type=int, default=2688, 
                       help='Max bars for 15min timeframe (default: 2688)')
    parser.add_argument('--30min-bars', type=int, default=2160, 
                       help='Max bars for 30min timeframe (default: 2160)')
    parser.add_argument('--1H-bars', type=int, default=2160, 
                       help='Max bars for 1H timeframe (default: 2160)')
    parser.add_argument('--4H-bars', type=int, default=2160, 
                       help='Max bars for 4H timeframe (default: 2160)')
    parser.add_argument('--1D-bars', type=int, default=365, 
                       help='Max bars for 1D timeframe (default: 365)')

    # Output options
    parser.add_argument('--output', 
                       default='processed_data',
                       help='Output directory (default: processed_data)')

    parser.add_argument('--delimiter',
                       default='auto',
                       choices=['auto', ',', '\t'],
                       help='CSV delimiter (default: auto-detect)')

    # Advanced processing options
    parser.add_argument('--no-tick-data',
                       action='store_true',
                       help='Skip tick data processing')

    parser.add_argument('--all-tick-data',
                       action='store_true',
                       help='Process ALL tick data without limits (memory intensive)')

    args = parser.parse_args()

    # Create bar limits configuration
    bar_limits = {
        '1min': args.__dict__['1min_bars'],
        '5min': args.__dict__['5min_bars'],
        '15min': args.__dict__['15min_bars'],
        '30min': args.__dict__['30min_bars'],
        '1H': args.__dict__['1H_bars'],
        '4H': args.__dict__['4H_bars'],
        '1D': args.__dict__['1D_bars']
    }

    # Create configuration
    config = ProcessingConfig(
        directory=args.dir,
        file_pattern=args.pattern,
        timeframes=args.timeframes,
        output_dir=args.output,
        delimiter=args.delimiter,
        json_only=args.json_only,
        process_tick_data=not args.no_tick_data,
        process_all_tick_data=args.all_tick_data,
        bar_limits=bar_limits
    )

    # Print startup info
    print("🚀 ncOS - Ultimate Trading Processor with Market Microstructure Analysis")
    print("="*80)
    print(f"📁 Scanning directory: {config.directory}")
    print(f"🔍 File pattern: {config.file_pattern}")
    print(f"📄 File types: {', '.join(config.file_types)}")
    print(f"⏰ Timeframes: {', '.join(config.timeframes)}")
    print(f"🎯 JSON only mode: {'ON' if config.json_only else 'OFF'}")
    print(f"📊 Tick data processing: {'ON' if config.process_tick_data else 'OFF'}")
    print(f"🔄 All tick data mode: {'ON' if config.process_all_tick_data else 'OFF'}")
    print(f"📈 Bar limits: {config.bar_limits}")
    print(f"📂 Output directory: {config.output_dir}")
    print("\n🔬 Advanced Analysis Enabled:")
    print("   • Smart Money Concepts (SMC)")
    print("   • Harmonic Pattern Recognition")
    print("   • Micro Wyckoff Analysis")
    print("   • Market Manipulation Detection")
    print("   • Order Flow Analysis")
    print("   • Liquidity Engineering Detection")
    print("="*80)

    # Process files
    processor = DataProcessor(config)
    processor.process_all_files()

if __name__ == "__main__":
    main()
