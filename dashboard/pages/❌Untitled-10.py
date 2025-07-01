
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import glob
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')
# Load secrets from TOML config
from pathlib import Path  # Move this line to the top BEFORE usage
import re

FINNHUB_API_KEY = st.secrets["finnhub_api_key"]
NEWSAPI_KEY = st.secrets["newsapi_key"]
TRADING_ECONOMICS_API_KEY = st.secrets["trading_economics_api_key"]
FRED_API_KEY = st.secrets["fred_api_key"]
DATA_DIRECTORY = st.secrets["data_directory"]
RAW_DATA_DIRECTORY = st.secrets["raw_data_directory"]
JSONDIR = st.secrets["JSONdir"]
BAR_DATA_DIRECTORY = st.secrets["bar_data_directory"]
DATA_PATH = st.secrets["data_path"]
# BAR_DATA_DIR: This directory is for bar data (OHLCV bars, etc).
# DATA_DIRECTORY: This directory is for primary market microstructure data (tick-level, etc).
BAR_DATA_DIR = Path(st.secrets["BAR_DATA_DIR"])
PARQUET_DATA_DIR = st.secrets["PARQUET_DATA_DIR"]
# Page config
st.set_page_config(
    page_title="ZanFlow Ultimate Trading Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.2rem;
    }
    .metric-card {
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)

# Microstructure Analysis Class
class MicrostructureAnalysis:
    """Advanced microstructure analysis"""

    @staticmethod
    def analyze_microstructure(df: pd.DataFrame) -> pd.DataFrame:
        """Main microstructure analysis"""
        try:
            df = MicrostructureAnalysis._calculate_spread_dynamics(df)
            df = MicrostructureAnalysis._analyze_order_flow(df)
            df = MicrostructureAnalysis._detect_stop_hunts(df)
            df = MicrostructureAnalysis._identify_liquidity_zones(df)
            df = MicrostructureAnalysis._detect_quote_stuffing(df)
            df = MicrostructureAnalysis._detect_spoofing(df)
            df = MicrostructureAnalysis._detect_layering(df)
            df = MicrostructureAnalysis._detect_momentum_ignition(df)
            df = MicrostructureAnalysis._calculate_manipulation_score(df)
            print(f"✓ Microstructure analysis completed")
        except Exception as e:
            print(f"Microstructure error: {e}")
        return df

    @staticmethod
    def _calculate_spread_dynamics(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate spread dynamics"""
        try:
            if 'bid' in df.columns and 'ask' in df.columns:
                df['micro_spread'] = df['ask'] - df['bid']
                df['micro_spread_pct'] = (df['micro_spread'] / df['mid']) * 100
                df['micro_spread_ma'] = df['micro_spread'].rolling(window=20).mean()
                df['micro_spread_std'] = df['micro_spread'].rolling(window=20).std()
                df['micro_spread_zscore'] = (df['micro_spread'] - df['micro_spread_ma']) / df['micro_spread_std']
        except Exception as e:
            print(f"Spread dynamics error: {e}")
        return df

    @staticmethod
    def _analyze_order_flow(df: pd.DataFrame) -> pd.DataFrame:
        """Analyze order flow imbalance"""
        try:
            # Estimate buy/sell pressure
            df['micro_price_change'] = df['mid'].diff()
            df['micro_buy_pressure'] = (df['micro_price_change'] > 0).astype(int)
            df['micro_sell_pressure'] = (df['micro_price_change'] < 0).astype(int)

            # Order flow imbalance
            window = 20
            df['micro_buy_volume'] = df['micro_buy_pressure'].rolling(window=window).sum()
            df['micro_sell_volume'] = df['micro_sell_pressure'].rolling(window=window).sum()
            df['micro_total_volume'] = df['micro_buy_volume'] + df['micro_sell_volume']

            df['micro_order_flow_imbalance'] = np.where(
                df['micro_total_volume'] > 0,
                (df['micro_buy_volume'] - df['micro_sell_volume']) / df['micro_total_volume'],
                0
            )

            # Flow efficiency
            df['micro_flow_efficiency'] = df['micro_order_flow_imbalance'].rolling(window=10).mean()

        except Exception as e:
            print(f"Order flow error: {e}")
        return df

    @staticmethod
    def _detect_stop_hunts(df: pd.DataFrame) -> pd.DataFrame:
        """Detect stop hunt patterns"""
        try:
            df['micro_stop_hunt'] = False
            window = 20

            for i in range(window, len(df)-5):
                # Look for sharp moves followed by reversals
                recent_high = df['high'].iloc[i-window:i].max()
                recent_low = df['low'].iloc[i-window:i].min()
                current_price = df['mid'].iloc[i]

                # Check for stop hunt above recent high
                if df['high'].iloc[i] > recent_high * 1.001:  # Break above
                    # Check for reversal
                    future_prices = df['mid'].iloc[i+1:i+6]
                    if len(future_prices) > 0 and future_prices.min() < recent_high:
                        df.iloc[i, df.columns.get_loc('micro_stop_hunt')] = True

                # Check for stop hunt below recent low
                elif df['low'].iloc[i] < recent_low * 0.999:  # Break below
                    # Check for reversal
                    future_prices = df['mid'].iloc[i+1:i+6]
                    if len(future_prices) > 0 and future_prices.max() > recent_low:
                        df.iloc[i, df.columns.get_loc('micro_stop_hunt')] = True

        except Exception as e:
            print(f"Stop hunt detection error: {e}")
        return df

    @staticmethod
    def _identify_liquidity_zones(df: pd.DataFrame) -> pd.DataFrame:
        """Identify liquidity concentration zones"""
        try:
            # Price levels with multiple touches
            df['micro_liquidity_zone'] = False
            price_levels = {}

            # Round prices to identify levels
            df['price_level'] = df['mid'].round(4)

            # Count touches at each level
            level_counts = df['price_level'].value_counts()
            significant_levels = level_counts[level_counts > 5].index

            # Mark liquidity zones
            df['micro_liquidity_zone'] = df['price_level'].isin(significant_levels)

        except Exception as e:
            print(f"Liquidity zones error: {e}")
        return df

    @staticmethod
    def _detect_quote_stuffing(df: pd.DataFrame) -> pd.DataFrame:
        """Detect quote stuffing (rapid quote updates)"""
        try:
            df['micro_quote_stuffing'] = False

            # Calculate quote update frequency
            df['timestamp_seconds'] = df['timestamp'].astype(np.int64) // 10**9
            df['quote_updates'] = 1

            # Count updates per second
            quote_freq = df.groupby('timestamp_seconds')['quote_updates'].count()

            # Flag high frequency periods (>50 updates/second)
            high_freq_times = quote_freq[quote_freq > 50].index
            df['micro_quote_stuffing'] = df['timestamp_seconds'].isin(high_freq_times)

        except Exception as e:
            print(f"Quote stuffing error: {e}")
        return df

    @staticmethod
    def _detect_spoofing(df: pd.DataFrame) -> pd.DataFrame:
        """Detect spoofing patterns"""
        try:
            df['micro_spoofing_detected'] = False

            if 'micro_spread' in df.columns:
                # Look for sudden spread widening followed by tightening
                df['spread_change'] = df['micro_spread'].diff()

                for i in range(10, len(df)-10):
                    # Sudden spread widening
                    if df['spread_change'].iloc[i] > df['micro_spread_std'].iloc[i] * 2:
                        # Followed by tightening without trade
                        if df['spread_change'].iloc[i+1:i+6].min() < -df['micro_spread_std'].iloc[i]:
                            df.iloc[i, df.columns.get_loc('micro_spoofing_detected')] = True

        except Exception as e:
            print(f"Spoofing detection error: {e}")
        return df

    @staticmethod
    def _detect_layering(df: pd.DataFrame) -> pd.DataFrame:
        """Detect layering manipulation"""
        try:
            df['micro_layering_detected'] = False

            # Look for one-sided pressure patterns
            if 'micro_order_flow_imbalance' in df.columns:
                # Extreme one-sided flow
                df['extreme_flow'] = df['micro_order_flow_imbalance'].abs() > 0.8

                # Followed by reversal
                for i in range(10, len(df)-10):
                    if df['extreme_flow'].iloc[i]:
                        flow_before = df['micro_order_flow_imbalance'].iloc[i]
                        flow_after = df['micro_order_flow_imbalance'].iloc[i+1:i+6].mean()

                        # Reversal detection
                        if flow_before * flow_after < -0.5:  # Sign change
                            df.iloc[i, df.columns.get_loc('micro_layering_detected')] = True

        except Exception as e:
            print(f"Layering detection error: {e}")
        return df

    @staticmethod
    def _detect_momentum_ignition(df: pd.DataFrame) -> pd.DataFrame:
        """Detect momentum ignition patterns"""
        try:
            df['micro_momentum_ignition'] = False

            # Rapid price movement detection
            df['price_velocity'] = df['mid'].diff().rolling(window=5).sum()
            df['price_acceleration'] = df['price_velocity'].diff()

            # High acceleration periods
            accel_threshold = df['price_acceleration'].std() * 2
            df['micro_momentum_ignition'] = df['price_acceleration'].abs() > accel_threshold

        except Exception as e:
            print(f"Momentum ignition error: {e}")
        return df

    @staticmethod
    def _calculate_manipulation_score(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate overall manipulation score"""
        try:
            manipulation_factors = []

            if 'micro_quote_stuffing' in df.columns:
                manipulation_factors.append(df['micro_quote_stuffing'].astype(int) * 0.2)
            if 'micro_spoofing_detected' in df.columns:
                manipulation_factors.append(df['micro_spoofing_detected'].astype(int) * 0.3)
            if 'micro_layering_detected' in df.columns:
                manipulation_factors.append(df['micro_layering_detected'].astype(int) * 0.3)
            if 'micro_momentum_ignition' in df.columns:
                manipulation_factors.append(df['micro_momentum_ignition'].astype(int) * 0.2)

            if manipulation_factors:
                df['micro_manipulation_score'] = sum(manipulation_factors)
                df['micro_manipulation_detected'] = df['micro_manipulation_score'] > 0.5
            else:
                df['micro_manipulation_score'] = 0
                df['micro_manipulation_detected'] = False

        except Exception as e:
            print(f"Manipulation score error: {e}")
        return df

# SMC Analysis Class
class SMCAnalysis:
    """Smart Money Concepts Analysis"""

    @staticmethod
    def analyze_smc(df: pd.DataFrame) -> pd.DataFrame:
        """Main SMC analysis"""
        try:
            df = SMCAnalysis._identify_market_structure(df)
            df = SMCAnalysis._find_order_blocks(df)
            df = SMCAnalysis._detect_fair_value_gaps(df)
            df = SMCAnalysis._identify_liquidity_zones(df)
            df = SMCAnalysis._detect_break_of_structure(df)
            df = SMCAnalysis._find_premium_discount_zones(df)
            print(f"✓ SMC analysis completed")
        except Exception as e:
            print(f"SMC error: {e}")
        return df

    @staticmethod
    def _identify_market_structure(df: pd.DataFrame) -> pd.DataFrame:
        """Identify market structure (HH, HL, LL, LH)"""
        try:
            # Find swing points
            window = 10
            df['SMC_swing_high'] = (df['high'] == df['high'].rolling(window=window*2+1, center=True).max())
            df['SMC_swing_low'] = (df['low'] == df['low'].rolling(window=window*2+1, center=True).min())

            # Determine structure
            df['SMC_structure'] = 'neutral'

            # Get swing highs and lows
            swing_highs = df[df['SMC_swing_high']]['high'].values
            swing_lows = df[df['SMC_swing_low']]['low'].values

            # Analyze structure
            if len(swing_highs) >= 2 and len(swing_lows) >= 2:
                # Check for higher highs and higher lows (bullish)
                if swing_highs[-1] > swing_highs[-2] and swing_lows[-1] > swing_lows[-2]:
                    df.loc[df.index[-1], 'SMC_structure'] = 'bullish'
                # Check for lower highs and lower lows (bearish)
                elif swing_highs[-1] < swing_highs[-2] and swing_lows[-1] < swing_lows[-2]:
                    df.loc[df.index[-1], 'SMC_structure'] = 'bearish'

        except Exception as e:
            print(f"Market structure error: {e}")
        return df

    @staticmethod
    def _find_order_blocks(df: pd.DataFrame) -> pd.DataFrame:
        """Identify order blocks"""
        try:
            df['SMC_bullish_ob'] = False
            df['SMC_bearish_ob'] = False
            df['SMC_ob_strength'] = 0.0

            # Look for significant moves followed by retracements
            for i in range(20, len(df)-20):
                # Bullish order block
                recent_low = df['low'].iloc[i-10:i].min()
                current_high = df['high'].iloc[i]
                future_high = df['high'].iloc[i:i+20].max()

                if current_high > recent_low * 1.01:  # 1% move
                    if future_high > current_high * 1.005:  # 0.5% continuation
                        df.iloc[i, df.columns.get_loc('SMC_bullish_ob')] = True
                        strength = (current_high - recent_low) / recent_low
                        df.iloc[i, df.columns.get_loc('SMC_ob_strength')] = strength

                # Bearish order block
                recent_high = df['high'].iloc[i-10:i].max()
                current_low = df['low'].iloc[i]
                future_low = df['low'].iloc[i:i+20].min()

                if current_low < recent_high * 0.99:  # 1% move
                    if future_low < current_low * 0.995:  # 0.5% continuation
                        df.iloc[i, df.columns.get_loc('SMC_bearish_ob')] = True
                        strength = (recent_high - current_low) / recent_high
                        df.iloc[i, df.columns.get_loc('SMC_ob_strength')] = -strength

        except Exception as e:
            print(f"Order blocks error: {e}")
        return df

    @staticmethod
    def _detect_fair_value_gaps(df: pd.DataFrame) -> pd.DataFrame:
        """Detect Fair Value Gaps (FVG)"""
        try:
            df['SMC_fvg_bullish'] = False
            df['SMC_fvg_bearish'] = False
            df['SMC_fvg_size'] = 0.0

            for i in range(2, len(df)):
                # Bullish FVG: gap between low[i-2] and high[i]
                if df['low'].iloc[i-2] > df['high'].iloc[i]:
                    gap_size = df['low'].iloc[i-2] - df['high'].iloc[i]
                    df.iloc[i, df.columns.get_loc('SMC_fvg_bullish')] = True
                    df.iloc[i, df.columns.get_loc('SMC_fvg_size')] = gap_size

                # Bearish FVG: gap between high[i-2] and low[i]
                if df['high'].iloc[i-2] < df['low'].iloc[i]:
                    gap_size = df['low'].iloc[i] - df['high'].iloc[i-2]
                    df.iloc[i, df.columns.get_loc('SMC_fvg_bearish')] = True
                    df.iloc[i, df.columns.get_loc('SMC_fvg_size')] = -gap_size

        except Exception as e:
            print(f"FVG detection error: {e}")
        return df

    @staticmethod
    def _identify_liquidity_zones(df: pd.DataFrame) -> pd.DataFrame:
        """Identify liquidity zones (equal highs/lows)"""
        try:
            df['SMC_liquidity_high'] = False
            df['SMC_liquidity_low'] = False

            # Equal highs (within 0.1% tolerance)
            for i in range(20, len(df)):
                recent_highs = df['high'].iloc[i-20:i]
                current_high = df['high'].iloc[i]

                # Count similar highs
                similar_highs = ((recent_highs >= current_high * 0.999) &
                               (recent_highs <= current_high * 1.001)).sum()

                if similar_highs >= 2:
                    df.iloc[i, df.columns.get_loc('SMC_liquidity_high')] = True

                # Equal lows
                recent_lows = df['low'].iloc[i-20:i]
                current_low = df['low'].iloc[i]
                similar_lows = ((recent_lows >= current_low * 0.999) &
                              (recent_lows <= current_low * 1.001)).sum()

                if similar_lows >= 2:
                    df.iloc[i, df.columns.get_loc('SMC_liquidity_low')] = True

        except Exception as e:
            print(f"Liquidity zones error: {e}")
        return df

    @staticmethod
    def _detect_break_of_structure(df: pd.DataFrame) -> pd.DataFrame:
        """Detect Break of Structure (BOS)"""
        try:
            df['SMC_bos_bullish'] = False
            df['SMC_bos_bearish'] = False

            # Find structure breaks
            if 'SMC_swing_high' in df.columns and 'SMC_swing_low' in df.columns:
                swing_highs = df[df['SMC_swing_high']].index
                swing_lows = df[df['SMC_swing_low']].index

                # Bullish BOS: price breaks above previous swing high
                for i in range(1, len(swing_highs)):
                    prev_high = df.loc[swing_highs[i-1], 'high']

                    # Check if price broke above
                    break_candles = df.loc[swing_highs[i-1]:swing_highs[i]]
                    breaks = break_candles[break_candles['close'] > prev_high]

                    if len(breaks) > 0:
                        first_break = breaks.index[0]
                        df.loc[first_break, 'SMC_bos_bullish'] = True

                # Bearish BOS: price breaks below previous swing low
                for i in range(1, len(swing_lows)):
                    prev_low = df.loc[swing_lows[i-1], 'low']

                    # Check if price broke below
                    break_candles = df.loc[swing_lows[i-1]:swing_lows[i]]
                    breaks = break_candles[break_candles['close'] < prev_low]

                    if len(breaks) > 0:
                        first_break = breaks.index[0]
                        df.loc[first_break, 'SMC_bos_bearish'] = True

        except Exception as e:
            print(f"BOS detection error: {e}")
        return df

    @staticmethod
    def _find_premium_discount_zones(df: pd.DataFrame) -> pd.DataFrame:
        """Find premium and discount zones"""
        try:
            # Calculate range
            lookback = 50
            df['SMC_range_high'] = df['high'].rolling(window=lookback).max()
            df['SMC_range_low'] = df['low'].rolling(window=lookback).min()
            df['SMC_range_mid'] = (df['SMC_range_high'] + df['SMC_range_low']) / 2

            # Premium/Discount zones
            df['SMC_premium_zone'] = df['close'] > df['SMC_range_mid']
            df['SMC_discount_zone'] = df['close'] < df['SMC_range_mid']

            # Zone strength (distance from midpoint)
            df['SMC_zone_strength'] = abs(df['close'] - df['SMC_range_mid']) / (df['SMC_range_high'] - df['SMC_range_low'])
            df['SMC_zone_strength'] = df['SMC_zone_strength'].fillna(0).clip(0, 1)

        except Exception as e:
            print(f"Premium/Discount zones error: {e}")
        return df

# Wyckoff Analysis Class
class WyckoffAnalysis:
    """Wyckoff Method Analysis"""

    @staticmethod
    def analyze_wyckoff(df: pd.DataFrame) -> pd.DataFrame:
        """Comprehensive Wyckoff analysis"""
        try:
            df = WyckoffAnalysis._detect_accumulation_phases(df)
            df = WyckoffAnalysis._detect_distribution_phases(df)
            df = WyckoffAnalysis._analyze_volume_spread(df)
            df = WyckoffAnalysis._detect_spring_upthrust(df)
            df = WyckoffAnalysis._identify_wyckoff_events(df)
            print(f"✓ Wyckoff analysis completed")
        except Exception as e:
            print(f"Wyckoff error: {e}")
        return df

    @staticmethod
    def _detect_accumulation_phases(df: pd.DataFrame) -> pd.DataFrame:
        """Detect accumulation phases"""
        try:
            window = 50
            df['wyckoff_accumulation'] = False
            df['wyckoff_acc_strength'] = 0.0

            for i in range(window, len(df)-window):
                # Check for sideways movement with increasing volume
                price_range = df['high'].iloc[i-window:i+window].max() - df['low'].iloc[i-window:i+window].min()
                avg_price = df['close'].iloc[i-window:i+window].mean()

                # Narrow range (< 2% of average price)
                if price_range < avg_price * 0.02:
                    # Check volume trend
                    if 'volume' in df.columns and df['volume'].sum() > 0:
                        early_vol = df['volume'].iloc[i-window:i-window//2].mean()
                        late_vol = df['volume'].iloc[i:i+window//2].mean()

                        # Increasing volume in narrow range
                        if late_vol > early_vol * 1.2:
                            df.iloc[i, df.columns.get_loc('wyckoff_accumulation')] = True
                            strength = late_vol / early_vol if early_vol > 0 else 1
                            df.iloc[i, df.columns.get_loc('wyckoff_acc_strength')] = strength

        except Exception as e:
            print(f"Accumulation detection error: {e}")
        return df

    @staticmethod
    def _detect_distribution_phases(df: pd.DataFrame) -> pd.DataFrame:
        """Detect distribution phases"""
        try:
            window = 50
            df['wyckoff_distribution'] = False
            df['wyckoff_dist_strength'] = 0.0

            for i in range(window, len(df)-window):
                # Check for topping pattern with decreasing momentum
                recent_high = df['high'].iloc[i-window:i].max()
                current_highs = df['high'].iloc[i:i+window]

                # Multiple tests of resistance
                resistance_tests = (current_highs > recent_high * 0.995).sum()

                if resistance_tests >= 3:
                    # Check for decreasing volume on rallies
                    if 'volume' in df.columns:
                        rally_volumes = df[df['close'] > df['open']]['volume'].iloc[i:i+window]
                        if len(rally_volumes) > 0:
                            vol_trend = rally_volumes.rolling(window=5).mean().diff().mean()
                            if vol_trend < 0:
                                df.iloc[i, df.columns.get_loc('wyckoff_distribution')] = True
                                df.iloc[i, df.columns.get_loc('wyckoff_dist_strength')] = abs(vol_trend)

        except Exception as e:
            print(f"Distribution detection error: {e}")
        return df

    @staticmethod
    def _analyze_volume_spread(df: pd.DataFrame) -> pd.DataFrame:
        """Analyze volume spread relationships"""
        try:
            if 'volume' in df.columns:
                # Volume spread analysis
                df['wyckoff_spread'] = df['high'] - df['low']
                df['wyckoff_volume_ma'] = df['volume'].rolling(window=20).mean()

                # High volume + narrow spread = absorption
                high_vol = df['volume'] > df['wyckoff_volume_ma'] * 1.5
                narrow_spread = df['wyckoff_spread'] < df['wyckoff_spread'].rolling(window=20).mean()
                df['wyckoff_absorption'] = high_vol & narrow_spread

                # Low volume + wide spread = lack of interest
                low_vol = df['volume'] < df['wyckoff_volume_ma'] * 0.5
                wide_spread = df['wyckoff_spread'] > df['wyckoff_spread'].rolling(window=20).mean()
                df['wyckoff_no_demand'] = low_vol & wide_spread

        except Exception as e:
            print(f"Volume spread error: {e}")
        return df

    @staticmethod
    def _detect_spring_upthrust(df: pd.DataFrame) -> pd.DataFrame:
        """Detect springs and upthrusts"""
        try:
            df['wyckoff_spring'] = False
            df['wyckoff_upthrust'] = False

            # Spring: false breakdown below support
            support_level = df['low'].rolling(window=50).min()
            spring_candidates = df[df['low'] < support_level * 0.995]

            for idx in spring_candidates.index:
                # Check if price quickly recovers
                if idx + 5 < len(df):
                    if df['close'].iloc[idx+5] > support_level.iloc[idx]:
                        df.loc[idx, 'wyckoff_spring'] = True

            # Upthrust: false breakout above resistance
            resistance_level = df['high'].rolling(window=50).max()
            upthrust_candidates = df[df['high'] > resistance_level * 1.005]

            for idx in upthrust_candidates.index:
                # Check if price quickly falls back
                if idx + 5 < len(df):
                    if df['close'].iloc[idx+5] < resistance_level.iloc[idx]:
                        df.loc[idx, 'wyckoff_upthrust'] = True

        except Exception as e:
            print(f"Spring/Upthrust error: {e}")
        return df

    @staticmethod
    def _identify_wyckoff_events(df: pd.DataFrame) -> pd.DataFrame:
        """Identify specific Wyckoff events"""
        try:
            # Combine all Wyckoff signals
            df['wyckoff_phase'] = 'neutral'

            # Mark phases
            if 'wyckoff_accumulation' in df.columns:
                df.loc[df['wyckoff_accumulation'], 'wyckoff_phase'] = 'accumulation'

            if 'wyckoff_distribution' in df.columns:
                df.loc[df['wyckoff_distribution'], 'wyckoff_phase'] = 'distribution'

            if 'wyckoff_spring' in df.columns:
                df.loc[df['wyckoff_spring'], 'wyckoff_phase'] = 'spring'

            if 'wyckoff_upthrust' in df.columns:
                df.loc[df['wyckoff_upthrust'], 'wyckoff_phase'] = 'upthrust'

        except Exception as e:
            print(f"Wyckoff events error: {e}")
        return df

# Rest of the code continues...'''


# Main data loader (not for bar data)
def load_data(data_folder=DATA_DIRECTORY):
    """Load data from the specified folder"""
    data_files = {}

    if os.path.exists(data_folder):
        for file in os.listdir(data_folder):
            if file.endswith('.csv'):
                try:
                    pair_name = file.replace('.csv', '')
                    file_path = os.path.join(data_folder, file)
                    df = pd.read_csv(file_path)
                    print(f"Loaded columns: {df.columns.tolist()} for {file_path}")
                    print(df.head())

                    # Ensure required columns
                    required_cols = ['timestamp', 'bid', 'ask', 'mid']
                    if all(col in df.columns for col in required_cols):
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df = df.sort_values('timestamp')

                        # Add OHLC if not present
                        if 'open' not in df.columns:
                            df['open'] = df['mid']
                        if 'high' not in df.columns:
                            df['high'] = df['mid']
                        if 'low' not in df.columns:
                            df['low'] = df['mid']
                        if 'close' not in df.columns:
                            df['close'] = df['mid']

                        data_files[pair_name] = df
                        print(f"✓ Loaded {pair_name}: {len(df)} rows")
                except Exception as e:
                    print(f"Error loading {file}: {e}")

    return data_files

# Bar data loader (for OHLCV bar data)
def load_bar_data(bar_folder=BAR_DATA_DIR):
    """
    Load bar data from the specified folder (BAR_DATA_DIR).
    Only loads files matching the pattern '*_M1_BARS.csv' (case-insensitive).
    """
    import re
    pattern = re.compile(r'.*_M1_BARS\.csv$', re.IGNORECASE)
    bar_files = {}

    if os.path.exists(bar_folder):
        for file in os.listdir(bar_folder):
            if pattern.match(file):
                try:
                    # Use file name as key
                    bar_name = file.replace('.csv', '')
                    file_path = os.path.join(bar_folder, file)
                    # Try tab-separated first, then fallback to comma-separated
                    try:
                        df = pd.read_csv(file_path, sep='\t')
                        # Clean column names: strip whitespace, make lower
                        df.columns = [col.strip().lower() for col in df.columns]
                    except Exception as e_tab:
                        print(f"Tab-separated failed for {file_path}, trying comma. {e_tab}")
                        try:
                            df = pd.read_csv(file_path)
                            df.columns = [col.strip().lower() for col in df.columns]
                        except Exception as e_comma:
                            print(f"Failed to load {file_path} as comma-separated. {e_comma}")
                            continue
                    print(f"Loaded columns: {df.columns.tolist()} for {file_path}")
                    print(df.head())
                    # Expecting OHLCV bar data as minimum
                    required_cols = ['timestamp', 'open', 'high', 'low', 'close']
                    if all(col in df.columns for col in required_cols):
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df = df.sort_values('timestamp')
                        bar_files[bar_name] = df
                        print(f"✓ Loaded bar data {bar_name}: {len(df)} rows")
                except Exception as e:
                    print(f"Error loading bar file {file}: {e}")

    return bar_files

def create_microstructure_plot(df: pd.DataFrame, pair_name: str) -> go.Figure:
    """Create comprehensive microstructure visualization"""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.4, 0.2, 0.2, 0.2],
        subplot_titles=(
            f"{pair_name} - Price & Microstructure",
            "Order Flow Imbalance",
            "Spread Analysis",
            "Manipulation Detection"
        )
    )

    # Main price chart with microstructure overlays
    fig.add_trace(
        go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price',
            showlegend=False
        ),
        row=1, col=1
    )

    # Add microstructure indicators
    if 'micro_stop_hunt' in df.columns:
        stop_hunts = df[df['micro_stop_hunt']]
        if len(stop_hunts) > 0:
            fig.add_trace(
                go.Scatter(
                    x=stop_hunts['timestamp'],
                    y=stop_hunts['high'],
                    mode='markers',
                    marker=dict(symbol='triangle-down', size=12, color='red'),
                    name='Stop Hunt',
                    text='Stop Hunt Detected',
                    hovertemplate='%{text}<br>Price: %{y:.4f}<br>Time: %{x}'
                ),
                row=1, col=1
            )

    # Order flow imbalance
    if 'micro_order_flow_imbalance' in df.columns:
        colors = ['red' if x < 0 else 'green' for x in df['micro_order_flow_imbalance']]
        fig.add_trace(
            go.Bar(
                x=df['timestamp'],
                y=df['micro_order_flow_imbalance'],
                name='Order Flow',
                marker_color=colors,
                showlegend=False
            ),
            row=2, col=1
        )

    # Spread analysis
    if 'micro_spread' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['micro_spread'],
                name='Spread',
                line=dict(color='orange', width=1),
                showlegend=False
            ),
            row=3, col=1
        )

        if 'micro_spread_ma' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['micro_spread_ma'],
                    name='Spread MA',
                    line=dict(color='blue', width=1),
                    showlegend=False
                ),
                row=3, col=1
            )

    # Manipulation score
    if 'micro_manipulation_score' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['micro_manipulation_score'],
                name='Manipulation Score',
                fill='tozeroy',
                line=dict(color='purple', width=1),
                showlegend=False
            ),
            row=4, col=1
        )

        # Add threshold line
        fig.add_hline(
            y=0.5, line_dash="dash", line_color="red",
            annotation_text="Manipulation Threshold",
            row=4, col=1
        )

    # Update layout
    fig.update_layout(
        title=f"{pair_name} - Microstructure Analysis",
        height=1000,
        template="plotly_dark",
        showlegend=True,
        hovermode='x unified'
    )

    fig.update_xaxes(title_text="Time", row=4, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Imbalance", row=2, col=1)
    fig.update_yaxes(title_text="Spread", row=3, col=1)
    fig.update_yaxes(title_text="Score", row=4, col=1)

    return fig

def create_smc_wyckoff_plot(df: pd.DataFrame, pair_name: str) -> go.Figure:
    """Create SMC and Wyckoff analysis plot"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=(
            f"{pair_name} - SMC & Wyckoff Analysis",
            "Volume Analysis",
            "Phase Detection"
        )
    )

    # Main price chart
    fig.add_trace(
        go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price',
            showlegend=False
        ),
        row=1, col=1
    )

    # SMC Order Blocks
    if 'SMC_bullish_ob' in df.columns:
        bullish_obs = df[df['SMC_bullish_ob']]
        if len(bullish_obs) > 0:
            fig.add_trace(
                go.Scatter(
                    x=bullish_obs['timestamp'],
                    y=bullish_obs['low'],
                    mode='markers',
                    marker=dict(symbol='square', size=10, color='green'),
                    name='Bullish OB',
                    text='Bullish Order Block',
                    hovertemplate='%{text}<br>Price: %{y:.4f}<br>Time: %{x}'
                ),
                row=1, col=1
            )

    if 'SMC_bearish_ob' in df.columns:
        bearish_obs = df[df['SMC_bearish_ob']]
        if len(bearish_obs) > 0:
            fig.add_trace(
                go.Scatter(
                    x=bearish_obs['timestamp'],
                    y=bearish_obs['high'],
                    mode='markers',
                    marker=dict(symbol='square', size=10, color='red'),
                    name='Bearish OB',
                    text='Bearish Order Block',
                    hovertemplate='%{text}<br>Price: %{y:.4f}<br>Time: %{x}'
                ),
                row=1, col=1
            )

    # Fair Value Gaps
    if 'SMC_fvg_bullish' in df.columns:
        bullish_fvgs = df[df['SMC_fvg_bullish']]
        for idx in bullish_fvgs.index:
            if idx >= 2:
                fig.add_shape(
                    type="rect",
                    x0=df.loc[idx-2, 'timestamp'],
                    x1=df.loc[idx, 'timestamp'],
                    y0=df.loc[idx, 'high'],
                    y1=df.loc[idx-2, 'low'],
                    fillcolor="green",
                    opacity=0.2,
                    row=1, col=1
                )

    # Wyckoff Springs
    if 'wyckoff_spring' in df.columns:
        springs = df[df['wyckoff_spring']]
        if len(springs) > 0:
            fig.add_trace(
                go.Scatter(
                    x=springs['timestamp'],
                    y=springs['low'],
                    mode='markers',
                    marker=dict(symbol='triangle-up', size=15, color='lime'),
                    name='Spring',
                    text='Wyckoff Spring',
                    hovertemplate='%{text}<br>Price: %{y:.4f}<br>Time: %{x}'
                ),
                row=1, col=1
            )

    # Volume with color coding
    if 'volume' in df.columns:
        colors = ['red' if c < o else 'green' for c, o in zip(df['close'], df['open'])]
        fig.add_trace(
            go.Bar(
                x=df['timestamp'],
                y=df['volume'],
                marker_color=colors,
                name='Volume',
                showlegend=False
            ),
            row=2, col=1
        )

    # Wyckoff phases
    if 'wyckoff_phase' in df.columns:
        phase_colors = {
            'accumulation': 'green',
            'distribution': 'red',
            'spring': 'lime',
            'upthrust': 'orange',
            'neutral': 'gray'
        }

        for phase, color in phase_colors.items():
            phase_data = df[df['wyckoff_phase'] == phase]
            if len(phase_data) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=phase_data['timestamp'],
                        y=[1] * len(phase_data),
                        mode='markers',
                        marker=dict(symbol='square', size=10, color=color),
                        name=phase.capitalize(),
                        showlegend=True
                    ),
                    row=3, col=1
                )

    # Update layout
    fig.update_layout(
        title=f"{pair_name} - SMC & Wyckoff Analysis",
        height=900,
        template="plotly_dark",
        showlegend=True,
        hovermode='x unified'
    )

    fig.update_xaxes(title_text="Time", row=3, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="Phase", row=3, col=1)

    return fig

def generate_microstructure_commentary(df: pd.DataFrame, pair_name: str) -> str:
    """Generate detailed microstructure commentary"""
    commentary = []

    # Basic stats
    total_ticks = len(df)
    time_range = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600

    commentary.append(f"### {pair_name} Microstructure Analysis")
    commentary.append(f"**Analysis Period**: {df['timestamp'].min().strftime('%Y-%m-%d %H:%M')} to {df['timestamp'].max().strftime('%Y-%m-%d %H:%M')}")
    commentary.append(f"**Total Ticks**: {total_ticks:,} over {time_range:.1f} hours")
    commentary.append("")

    # Spread Analysis
    if 'micro_spread' in df.columns:
        avg_spread = df['micro_spread'].mean()
        max_spread = df['micro_spread'].max()
        spread_volatility = df['micro_spread'].std()

        commentary.append("#### Spread Dynamics")
        commentary.append(f"- **Average Spread**: {avg_spread:.4f}")
        commentary.append(f"- **Maximum Spread**: {max_spread:.4f}")
        commentary.append(f"- **Spread Volatility**: {spread_volatility:.4f}")

        if 'micro_spread_zscore' in df.columns:
            extreme_spreads = (df['micro_spread_zscore'].abs() > 3).sum()
            commentary.append(f"- **Extreme Spread Events**: {extreme_spreads} ({extreme_spreads/total_ticks*100:.1f}% of ticks)")
        commentary.append("")

    # Order Flow Analysis
    if 'micro_order_flow_imbalance' in df.columns:
        avg_imbalance = df['micro_order_flow_imbalance'].mean()
        max_imbalance = df['micro_order_flow_imbalance'].abs().max()

        commentary.append("#### Order Flow Dynamics")
        if avg_imbalance > 0.1:
            commentary.append(f"- **Dominant Flow**: BULLISH (avg imbalance: {avg_imbalance:.3f})")
        elif avg_imbalance < -0.1:
            commentary.append(f"- **Dominant Flow**: BEARISH (avg imbalance: {avg_imbalance:.3f})")
        else:
            commentary.append(f"- **Flow Balanced** (avg imbalance: {avg_imbalance:.3f})")

        if 'micro_layering_detected' in df.columns:
            layering_events = df['micro_layering_detected'].sum()
            commentary.append(f"- **Layering Events**: {layering_events}")

        if 'micro_momentum_ignition' in df.columns:
            momentum_events = df['micro_momentum_ignition'].sum()
            commentary.append(f"- **Momentum Ignition Events**: {momentum_events}")
        commentary.append("")

    # Liquidity Analysis
    if 'micro_stop_hunt' in df.columns:
        stop_hunts = df['micro_stop_hunt'].sum()
        commentary.append("#### Liquidity & Stop Hunts")
        commentary.append(f"- **Stop Hunt Events**: {stop_hunts}")

        if 'micro_liquidity_zone' in df.columns:
            liquidity_zones = df['micro_liquidity_zone'].sum()
            commentary.append(f"- **Liquidity Zone Periods**: {liquidity_zones} ({liquidity_zones/total_ticks*100:.1f}% of time)")
        commentary.append("")

    # Compute average manipulation score before trading recommendations
    avg_manipulation = df['micro_manipulation_score'].mean() if 'micro_manipulation_score' in df.columns else 0

    # Trading Recommendations
    commentary.append("#### Trading Recommendations")

    if avg_manipulation > 0.3:
        commentary.append("⚠️ **High manipulation detected** - Use wider stops and reduce position size")

    if 'micro_spread_zscore' in df.columns:
        current_spread_zscore = df['micro_spread_zscore'].iloc[-1]
        if abs(current_spread_zscore) > 2:
            commentary.append("⚠️ **Abnormal spread conditions** - Wait for normalization before entering")

    if 'micro_order_flow_imbalance' in df.columns:
        recent_flow = df['micro_order_flow_imbalance'].iloc[-20:].mean()
        if recent_flow > 0.3:
            commentary.append("✅ **Strong bullish flow** - Consider long positions on pullbacks")
        elif recent_flow < -0.3:
            commentary.append("✅ **Strong bearish flow** - Consider short positions on rallies")

    return "\\n".join(commentary)

def generate_smc_wyckoff_commentary(df: pd.DataFrame, pair_name: str) -> str:
    """Generate SMC and Wyckoff commentary"""
    commentary = []

    commentary.append(f"### {pair_name} SMC & Wyckoff Analysis")
    commentary.append("")

    # SMC Analysis
    if 'SMC_structure' in df.columns:
        current_structure = df['SMC_structure'].iloc[-1]
        commentary.append("#### Smart Money Concepts")
        commentary.append(f"- **Market Structure**: {current_structure.upper()}")

        if 'SMC_bullish_ob' in df.columns:
            bullish_obs = df['SMC_bullish_ob'].sum()
            bearish_obs = df['SMC_bearish_ob'].sum() if 'SMC_bearish_ob' in df.columns else 0
            commentary.append(f"- **Order Blocks**: {bullish_obs} bullish, {bearish_obs} bearish")

        if 'SMC_fvg_bullish' in df.columns:
            bullish_fvgs = df['SMC_fvg_bullish'].sum()
            bearish_fvgs = df['SMC_fvg_bearish'].sum() if 'SMC_fvg_bearish' in df.columns else 0
            commentary.append(f"- **Fair Value Gaps**: {bullish_fvgs} bullish, {bearish_fvgs} bearish")

            if bullish_fvgs > bearish_fvgs:
                commentary.append("  → Bullish bias in institutional order flow")
            elif bearish_fvgs > bullish_fvgs:
                commentary.append("  → Bearish bias in institutional order flow")

        if 'SMC_bos_bullish' in df.columns:
            recent_bos = df[['SMC_bos_bullish', 'SMC_bos_bearish']].iloc[-50:].sum()
            if recent_bos['SMC_bos_bullish'] > 0:
                commentary.append("- **Recent Break of Structure**: BULLISH")
            elif recent_bos['SMC_bos_bearish'] > 0:
                commentary.append("- **Recent Break of Structure**: BEARISH")
        commentary.append("")

    # Wyckoff Analysis
    if 'wyckoff_phase' in df.columns:
        phase_counts = df['wyckoff_phase'].value_counts()
        dominant_phase = phase_counts.index[0] if len(phase_counts) > 0 else 'neutral'

        commentary.append("#### Wyckoff Analysis")
        commentary.append(f"- **Dominant Phase**: {dominant_phase.upper()}")

        if 'wyckoff_accumulation' in df.columns:
            acc_periods = df['wyckoff_accumulation'].sum()
            if acc_periods > 0:
                commentary.append(f"- **Accumulation Periods**: {acc_periods}")

        if 'wyckoff_distribution' in df.columns:
            dist_periods = df['wyckoff_distribution'].sum()
            if dist_periods > 0:
                commentary.append(f"- **Distribution Periods**: {dist_periods}")

        if 'wyckoff_spring' in df.columns:
            springs = df['wyckoff_spring'].sum()
            if springs > 0:
                commentary.append(f"- **Spring Events**: {springs} (potential bullish reversals)")

        if 'wyckoff_upthrust' in df.columns:
            upthrusts = df['wyckoff_upthrust'].sum()
            if upthrusts > 0:
                commentary.append(f"- **Upthrust Events**: {upthrusts} (potential bearish reversals)")
        commentary.append("")

    # Combined Strategy
    commentary.append("#### Strategic Outlook")

    if 'SMC_structure' in df.columns and 'wyckoff_phase' in df.columns:
        if dominant_phase == 'accumulation' and current_structure == 'bullish':
            commentary.append("🟢 **STRONG BULLISH SETUP** - Accumulation + Bullish structure")
            commentary.append("→ Look for long entries at order blocks or after springs")
        elif dominant_phase == 'distribution' and current_structure == 'bearish':
            commentary.append("🔴 **STRONG BEARISH SETUP** - Distribution + Bearish structure")
            commentary.append("→ Look for short entries at order blocks or after upthrusts")
        else:
            commentary.append("🟡 **MIXED SIGNALS** - Wait for clearer alignment")

    return "\\n".join(commentary)

# Main Dashboard Pages
def home_page():
    """Landing page with overview"""
    st.title("🚀 ZanFlow Ultimate Trading Dashboard")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### 📊 Microstructure Analysis
        - Order flow imbalance tracking
        - Spread dynamics monitoring
        - Manipulation detection
        - Liquidity zone identification
        """)

    with col2:
        st.markdown("""
        ### 💡 Smart Money Concepts
        - Market structure analysis
        - Order block detection
        - Fair value gap tracking
        - Break of structure alerts
        """)

    with col3:
        st.markdown("""
        ### 📈 Wyckoff Method
        - Accumulation/Distribution phases
        - Spring and upthrust detection
        - Volume spread analysis
        - Phase identification
        """)

    st.markdown("---")
    st.info(f"📁 Place your BAR CSV files in the `{BAR_DATA_DIR}` folder. Each file should contain: timestamp, open, high, low, close columns.")

def microstructure_page(data_files):
    """Microstructure analysis page"""
    st.title("🔬 Microstructure Analysis")

    if not data_files:
        st.warning(f"No data files found. Please add CSV files to the `{BAR_DATA_DIR}` folder. Each file should contain: timestamp, open, high, low, close columns.")
        return

    # Select trading pair
    selected_pair = st.selectbox("Select Trading Pair", list(data_files.keys()))

    if selected_pair:
        df = data_files[selected_pair].copy()

        # Run microstructure analysis
        with st.spinner("Analyzing microstructure..."):
            df = MicrostructureAnalysis.analyze_microstructure(df)

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if 'micro_spread' in df.columns:
                avg_spread = df['micro_spread'].mean()
                st.metric("Avg Spread", f"{avg_spread:.4f}")

        with col2:
            if 'micro_order_flow_imbalance' in df.columns:
                flow_imbalance = df['micro_order_flow_imbalance'].iloc[-20:].mean()
                st.metric("Flow Imbalance", f"{flow_imbalance:.3f}",
                         delta="Bullish" if flow_imbalance > 0 else "Bearish")

        with col3:
            if 'micro_manipulation_score' in df.columns:
                manip_score = df['micro_manipulation_score'].mean()
                st.metric("Manipulation Score", f"{manip_score:.3f}",
                         delta="High" if manip_score > 0.5 else "Normal")

        with col4:
            if 'micro_stop_hunt' in df.columns:
                stop_hunts = df['micro_stop_hunt'].sum()
                st.metric("Stop Hunts", stop_hunts)

        # Plot
        st.plotly_chart(create_microstructure_plot(df, selected_pair), use_container_width=True)

        # Commentary
        with st.expander("📝 Detailed Commentary", expanded=True):
            st.markdown(generate_microstructure_commentary(df, selected_pair))

        # Recent Events
        st.subheader("🚨 Recent Market Events")
        recent_events = []
        if 'micro_stop_hunt' in df.columns:
            recent_hunts = df[df['micro_stop_hunt']].tail(5)
            for idx, row in recent_hunts.iterrows():
                recent_events.append(f"🎯 Stop Hunt at {row['timestamp']} - Price: {row['close']:.4f}")

        if 'micro_manipulation_detected' in df.columns:
            recent_manip = df[df['micro_manipulation_detected']].tail(5)
            for idx, row in recent_manip.iterrows():
                recent_events.append(f"⚠️ Manipulation at {row['timestamp']} - Score: {row['micro_manipulation_score']:.3f}")

        if recent_events:
            for event in recent_events[-10:]:
                st.write(event)
        else:
            st.info("No recent events detected")

def smc_wyckoff_page(data_files):
    """SMC and Wyckoff analysis page"""
    st.title("🎯 Smart Money Concepts & Wyckoff Analysis")

    if not data_files:
        st.warning(f"No data files found. Please add CSV files to the `{BAR_DATA_DIR}` folder. Each file should contain: timestamp, open, high, low, close columns.")
        return

    # Select trading pair
    selected_pair = st.selectbox("Select Trading Pair", list(data_files.keys()))

    if selected_pair:
        df = data_files[selected_pair].copy()

        # Run analysis
        with st.spinner("Analyzing SMC and Wyckoff patterns..."):
            df = SMCAnalysis.analyze_smc(df)
            df = WyckoffAnalysis.analyze_wyckoff(df)

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if 'SMC_structure' in df.columns:
                structure = df['SMC_structure'].iloc[-1]
                st.metric("Market Structure", structure.upper())

        with col2:
            if 'wyckoff_phase' in df.columns:
                phase_counts = df['wyckoff_phase'].value_counts()
                dominant = phase_counts.index[0] if len(phase_counts) > 0 else 'neutral'
                st.metric("Wyckoff Phase", dominant.upper())

        with col3:
            if 'SMC_fvg_bullish' in df.columns:
                fvg_bull = df['SMC_fvg_bullish'].sum()
                fvg_bear = df['SMC_fvg_bearish'].sum() if 'SMC_fvg_bearish' in df.columns else 0
                st.metric("Fair Value Gaps", f"↑{fvg_bull} ↓{fvg_bear}")

        with col4:
            if 'wyckoff_spring' in df.columns:
                springs = df['wyckoff_spring'].sum()
                upthrusts = df['wyckoff_upthrust'].sum() if 'wyckoff_upthrust' in df.columns else 0
                st.metric("Spring/Upthrust", f"S:{springs} U:{upthrusts}")

        # Plot
        st.plotly_chart(create_smc_wyckoff_plot(df, selected_pair), use_container_width=True)

        # Commentary
        with st.expander("📝 Strategic Analysis", expanded=True):
            st.markdown(generate_smc_wyckoff_commentary(df, selected_pair))

        # Key Levels
        st.subheader("🎯 Key Trading Levels")
        key_levels = []

        # Recent order blocks
        if 'SMC_bullish_ob' in df.columns:
            recent_bull_ob = df[df['SMC_bullish_ob']].tail(3)
            for idx, row in recent_bull_ob.iterrows():
                key_levels.append(("Bullish OB", row['low'], "green"))

        if 'SMC_bearish_ob' in df.columns:
            recent_bear_ob = df[df['SMC_bearish_ob']].tail(3)
            for idx, row in recent_bear_ob.iterrows():
                key_levels.append(("Bearish OB", row['high'], "red"))

        if key_levels:
            for level_type, price, color in key_levels:
                st.markdown(f"<span style='color:{color}'>● {level_type}: {price:.4f}</span>",
                           unsafe_allow_html=True)
        else:
            st.info("No key levels identified")

def settings_page():
    """Settings and configuration page"""
    st.title("⚙️ Settings & Configuration")

    st.subheader("Data Configuration")
    data_path = st.text_input("Bar Data Folder Path", value=BAR_DATA_DIR)

    st.subheader("Analysis Parameters")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Microstructure Settings")
        spread_window = st.slider("Spread MA Window", 10, 100, 20)
        flow_window = st.slider("Order Flow Window", 10, 50, 20)
        manip_threshold = st.slider("Manipulation Threshold", 0.1, 1.0, 0.5)

    with col2:
        st.markdown("#### SMC/Wyckoff Settings")
        structure_lookback = st.slider("Structure Lookback", 20, 100, 50)
        ob_threshold = st.slider("Order Block Threshold %", 0.5, 2.0, 1.0)
        wyckoff_window = st.slider("Wyckoff Window", 30, 100, 50)

    if st.button("Save Settings"):
        settings = {
            "data_path": data_path,
            "microstructure": {
                "spread_window": spread_window,
                "flow_window": flow_window,
                "manipulation_threshold": manip_threshold
            },
            "smc_wyckoff": {
                "structure_lookback": structure_lookback,
                "ob_threshold": ob_threshold,
                "wyckoff_window": wyckoff_window
            }
        }
        st.success("Settings saved successfully!")
        st.json(settings)

# Main dashboard entrypoint
def main():
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")

    # --- Parquet Mode: User selects symbol and timeframe, app loads that Parquet file ---
    parquet_base_dir = st.sidebar.text_input("Parquet Base Folder", value=str(PARQUET_DATA_DIR))
    # List all symbol subfolders
    try:
        symbols = [f.name for f in os.scandir(parquet_base_dir) if f.is_dir()]
    except Exception as e:
        st.sidebar.error(f"Error scanning base folder: {e}")
        symbols = []
    if symbols:
        selected_symbol = st.sidebar.selectbox("Symbol", symbols)
        symbol_dir = os.path.join(parquet_base_dir, selected_symbol)
        parquet_files = [f for f in os.listdir(symbol_dir) if f.endswith('.parquet')]
        # Extract timeframes from filenames like BTCUSD_1min.parquet, BTCUSD_5min.parquet, etc.
        timeframes = []
        tf_map = {}
        for f in parquet_files:
            m = re.match(rf"{re.escape(selected_symbol)}_(.+)\.parquet", f)
            if m:
                tf = m.group(1)
                timeframes.append(tf)
                tf_map[tf] = f
        if timeframes:
            selected_tf = st.sidebar.selectbox("Timeframe", timeframes)
            parquet_file = tf_map[selected_tf]
            full_path = os.path.join(symbol_dir, parquet_file)
            try:
                df = pd.read_parquet(full_path)
                # Clean and normalize columns
                df.columns = [col.strip().lower() for col in df.columns]
                # If no time column, check if index holds the time information
                if "timestamp" not in df.columns:
                    df = df.reset_index()
                    df.columns = [col.strip().lower() for col in df.columns]
                if "timestamp" not in df.columns:
                    for col in df.columns:
                        if col.startswith("time") or col.startswith("date"):
                            df.rename(columns={col: "timestamp"}, inplace=True)
                            break
                # --- Robust timestamp column selection ---
                if "timestamp" not in df.columns:
                    # Let the user pick a timestamp column manually if one is present
                    candidate_cols = [col for col in df.columns if df[col].dtype == "O" or np.issubdtype(df[col].dtype, np.datetime64)]
                    if candidate_cols:
                        manual_timestamp_col = st.sidebar.selectbox("Select the time column", candidate_cols, key="manual_timestamp_col")
                        if manual_timestamp_col:
                            df.rename(columns={manual_timestamp_col: "timestamp"}, inplace=True)
                    # After user selection, check again:
                    if "timestamp" not in df.columns:
                        st.sidebar.error("No 'timestamp' column found in the selected data. Please select the correct column above. Available columns: " + str(list(df.columns)))
                        df = None
                if df is not None:
                    st.sidebar.success(f"Loaded {selected_symbol} {selected_tf} ({df.shape[0]} rows)")
            except Exception as e:
                st.sidebar.error(f"Failed to load: {full_path}\n{e}")
                df = None
        else:
            st.sidebar.warning("No Parquet files found for selected symbol.")
            df = None
    else:
        st.sidebar.warning("No symbol subfolders found.")
        df = None

    # Sidebar: Dropdown for available subfolders in BAR_DATA_DIR (legacy CSV mode)
    bar_data_dir_str = str(BAR_DATA_DIR)
    subfolders = [f.path for f in os.scandir(bar_data_dir_str) if f.is_dir()]
    if not subfolders:
        subfolders = [bar_data_dir_str]
    subfolder_labels = [os.path.basename(f) if os.path.basename(f) else f for f in subfolders]
    selected_subfolder = st.sidebar.selectbox(
        "Select Bar Data Folder", subfolders, format_func=lambda x: os.path.basename(x) if os.path.basename(x) else x
    )

    pages = {
        "🏠 Home": home_page,
        "🔬 Microstructure": lambda: microstructure_page(data_files),
        "🎯 SMC & Wyckoff": lambda: smc_wyckoff_page(data_files),
        "⚙️ Settings": settings_page
    }

    selected_page = st.sidebar.radio("Go to", list(pages.keys()))

    # Overwrite data_files with Parquet loader logic if a dataframe is loaded
    data_files = {}
    if df is not None:
        data_files[f"{selected_symbol}_{selected_tf}"] = df
    else:
        # Load data from selected folder (legacy CSV loader)
        data_files = load_bar_data(selected_subfolder)

    # Display selected page
    pages[selected_page]()

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("ZanFlow Ultimate Trading Dashboard v2.0")
    st.sidebar.caption("Last update: Every minute")

if __name__ == "__main__":
    main()
