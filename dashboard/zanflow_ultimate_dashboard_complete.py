import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime, timedelta
import warnings
from scipy.signal import find_peaks
from scipy.stats import zscore
import json
from typing import Dict, List, Tuple, Optional
warnings.filterwarnings('ignore')

# Create complete dashboard file
dashboard_code = '''import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime, timedelta
import warnings
from scipy.signal import find_peaks
from scipy.stats import zscore
import json
from typing import Dict, List, Tuple, Optional
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="ZanFlow Ultimate Trading Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: rgba(25, 25, 25, 0.8);
        border-radius: 8px;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
    }
    .signal-positive { color: #00ff88; }
    .signal-negative { color: #ff4444; }
    .signal-neutral { color: #ffd700; }
</style>
""", unsafe_allow_html=True)

class MicrostructureAnalysis:
    """Advanced microstructure analysis including order flow, spread dynamics, and manipulation detection"""
    
    @staticmethod
    def analyze_microstructure(df: pd.DataFrame) -> pd.DataFrame:
        """Comprehensive microstructure analysis"""
        try:
            # Core microstructure metrics
            df = MicrostructureAnalysis._calculate_spread_metrics(df)
            df = MicrostructureAnalysis._analyze_tick_dynamics(df)
            df = MicrostructureAnalysis._detect_quote_stuffing(df)
            df = MicrostructureAnalysis._analyze_order_flow(df)
            df = MicrostructureAnalysis._detect_manipulation_patterns(df)
            df = MicrostructureAnalysis._calculate_volume_profile(df)
            df = MicrostructureAnalysis._detect_liquidity_zones(df)
            
            print(f"✓ Microstructure analysis completed with {len([c for c in df.columns if 'micro_' in c])} indicators")
            
        except Exception as e:
            print(f"Warning: Microstructure analysis error: {e}")
            
        return df
    
    @staticmethod
    def _calculate_spread_metrics(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate bid-ask spread metrics"""
        try:
            if 'bid' in df.columns and 'ask' in df.columns:
                df['micro_spread'] = df['ask'] - df['bid']
                df['micro_spread_pct'] = (df['micro_spread'] / df['mid']) * 100
                df['micro_spread_ma'] = df['micro_spread'].rolling(window=20).mean()
                df['micro_spread_std'] = df['micro_spread'].rolling(window=20).std()
                df['micro_spread_zscore'] = (df['micro_spread'] - df['micro_spread_ma']) / df['micro_spread_std'].replace(0, 1)
                
                # Spread volatility
                df['micro_spread_volatility'] = df['micro_spread'].rolling(window=20).std()
                
                # Effective spread (using trades)
                if 'price' in df.columns:
                    df['micro_effective_spread'] = 2 * abs(df['price'] - df['mid'])
                    df['micro_realized_spread'] = df['micro_effective_spread'] - df['micro_spread']
        except Exception as e:
            print(f"Spread metrics error: {e}")
            
        return df
    
    @staticmethod
    def _analyze_tick_dynamics(df: pd.DataFrame) -> pd.DataFrame:
        """Analyze tick-by-tick dynamics"""
        try:
            # Price changes
            df['micro_price_change'] = df['mid'].diff()
            df['micro_price_change_pct'] = df['mid'].pct_change() * 100
            
            # Tick direction
            df['micro_tick_direction'] = np.sign(df['micro_price_change'])
            df['micro_upticks'] = (df['micro_tick_direction'] == 1).rolling(window=20).sum()
            df['micro_downticks'] = (df['micro_tick_direction'] == -1).rolling(window=20).sum()
            df['micro_tick_imbalance'] = df['micro_upticks'] - df['micro_downticks']
            
            # Time between ticks
            if 'timestamp' in df.columns:
                df['micro_time_diff'] = df['timestamp'].diff().dt.total_seconds()
                df['micro_tick_rate'] = 1 / df['micro_time_diff'].replace(0, np.inf)
                
            # Volatility per tick
            df['micro_tick_volatility'] = df['micro_price_change'].rolling(window=20).std()
            
        except Exception as e:
            print(f"Tick dynamics error: {e}")
            
        return df
    
    @staticmethod
    def _detect_quote_stuffing(df: pd.DataFrame) -> pd.DataFrame:
        """Detect quote stuffing (excessive quote updates)"""
        try:
            # Quote update frequency
            if 'micro_tick_rate' in df.columns:
                df['micro_quote_rate_ma'] = df['micro_tick_rate'].rolling(window=100).mean()
                df['micro_quote_rate_std'] = df['micro_tick_rate'].rolling(window=100).std()
                
                # Quote stuffing when rate > 3 std above average
                stuffing_threshold = df['micro_quote_rate_ma'] + 3 * df['micro_quote_rate_std']
                df['micro_quote_stuffing'] = df['micro_tick_rate'] > stuffing_threshold
                
                # Intensity score
                df['micro_stuffing_score'] = ((df['micro_tick_rate'] - df['micro_quote_rate_ma']) / 
                                             df['micro_quote_rate_std'].replace(0, 1)).fillna(0).clip(0, 10)
                
        except Exception as e:
            print(f"Quote stuffing detection error: {e}")
            
        return df
    
    @staticmethod
    def _analyze_order_flow(df: pd.DataFrame) -> pd.DataFrame:
        """Analyze order flow patterns"""
        try:
            # Order flow imbalance
            if 'micro_tick_direction' in df.columns:
                window = 20
                upticks = (df['micro_tick_direction'] == 1).rolling(window=window).sum()
                downticks = (df['micro_tick_direction'] == -1).rolling(window=window).sum()
                total_ticks = upticks + downticks
                
                df['micro_order_flow_imbalance'] = (upticks - downticks) / total_ticks.replace(0, 1)
                
                # Volume-weighted order flow
                if 'volume' in df.columns and df['volume'].sum() > 0:
                    vol_weighted_flow = (df['micro_tick_direction'] * df['volume']).rolling(window=window).sum()
                    total_volume = df['volume'].rolling(window=window).sum()
                    df['micro_volume_flow_imbalance'] = vol_weighted_flow / total_volume.replace(0, 1)
                else:
                    df['micro_volume_flow_imbalance'] = df['micro_order_flow_imbalance']
                    
                # Price impact
                price_impact = df['micro_price_change'].rolling(window=10).sum()
                flow_direction = df['micro_order_flow_imbalance'].shift(10)
                df['micro_flow_efficiency'] = price_impact * flow_direction
                
        except Exception as e:
            print(f"Order flow analysis error: {e}")
            
        return df
    
    @staticmethod
    def _detect_manipulation_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """Detect various market manipulation patterns"""
        try:
            # Spoofing detection
            df = MicrostructureAnalysis._detect_spoofing(df)
            
            # Layering detection
            df = MicrostructureAnalysis._detect_layering(df)
            
            # Momentum ignition
            df = MicrostructureAnalysis._detect_momentum_ignition(df)
            
            # Combined manipulation score
            manipulation_indicators = ['micro_spoofing_score', 'micro_layering_score', 'micro_momentum_score']
            existing_indicators = [col for col in manipulation_indicators if col in df.columns]
            
            if existing_indicators:
                df['micro_manipulation_score'] = df[existing_indicators].mean(axis=1)
                df['micro_manipulation_detected'] = df['micro_manipulation_score'] > 0.5
                
        except Exception as e:
            print(f"Manipulation detection error: {e}")
            
        return df
    
    @staticmethod
    def _detect_spoofing(df: pd.DataFrame) -> pd.DataFrame:
        """Detect spoofing patterns"""
        try:
            df['micro_spoofing_score'] = 0.0
            
            if 'micro_spread' in df.columns:
                # Large spread that quickly disappears
                window = 20
                for i in range(window, len(df) - window):
                    current_spread = df['micro_spread'].iloc[i]
                    avg_spread = df['micro_spread'].iloc[i-window:i].mean()
                    
                    # Check if spread is unusually large
                    if current_spread > avg_spread * 2:
                        # Check if it quickly returns to normal
                        future_spreads = df['micro_spread'].iloc[i:i+5]
                        if future_spreads.min() < avg_spread * 1.2:
                            df.loc[df.index[i], 'micro_spoofing_score'] = 1.0
                            
            df['micro_spoofing_detected'] = df['micro_spoofing_score'].rolling(window=20).sum() > 3
            
        except Exception as e:
            print(f"Spoofing detection error: {e}")
            
        return df
    
    @staticmethod
    def _detect_layering(df: pd.DataFrame) -> pd.DataFrame:
        """Detect layering patterns"""
        try:
            df['micro_layering_score'] = 0.0
            
            if 'bid' in df.columns and 'ask' in df.columns:
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
                        df.loc[df.index[i], 'micro_layering_score'] = max(max_bid_count, max_ask_count) / window_size
                        
            df['micro_layering_detected'] = df['micro_layering_score'] > 0.3
            
        except Exception as e:
            print(f"Layering detection error: {e}")
            
        return df
    
    @staticmethod
    def _detect_momentum_ignition(df: pd.DataFrame) -> pd.DataFrame:
        """Detect momentum ignition patterns"""
        try:
            if 'micro_price_change' in df.columns:
                # Price velocity and acceleration
                df['micro_price_velocity'] = df['micro_price_change'].rolling(window=5).sum()
                df['micro_price_acceleration'] = df['micro_price_velocity'].diff()
                
                # Detect sudden acceleration
                vel_threshold = df['micro_price_velocity'].std() * 2
                acc_threshold = df['micro_price_acceleration'].std() * 2
                
                high_velocity = df['micro_price_velocity'].abs() > vel_threshold
                high_acceleration = df['micro_price_acceleration'].abs() > acc_threshold
                
                df['micro_momentum_ignition'] = high_velocity & high_acceleration
                
                # Score based on intensity
                df['micro_momentum_score'] = (df['micro_price_velocity'].abs() / vel_threshold + 
                                             df['micro_price_acceleration'].abs() / acc_threshold) / 2
                df['micro_momentum_score'] = df['micro_momentum_score'].fillna(0).clip(0, 5)
                
        except Exception as e:
            print(f"Momentum ignition detection error: {e}")
            
        return df
    
    @staticmethod
    def _calculate_volume_profile(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume profile"""
        try:
            if 'volume' in df.columns and df['volume'].sum() > 0:
                # Price levels
                price_bins = pd.qcut(df['mid'], q=20, duplicates='drop')
                
                # Volume at each price level
                volume_profile = df.groupby(price_bins)['volume'].sum()
                
                # Point of Control (POC) - price level with highest volume
                poc_level = volume_profile.idxmax()
                df['micro_poc_distance'] = abs(df['mid'] - poc_level.mid)
                
                # Value area (70% of volume)
                cumsum = volume_profile.sort_values(ascending=False).cumsum()
                value_area_levels = cumsum[cumsum <= cumsum.iloc[-1] * 0.7].index
                
                # Mark if price is in value area
                df['micro_in_value_area'] = df['mid'].apply(
                    lambda x: any(level.left <= x <= level.right for level in value_area_levels)
                )
                
        except Exception as e:
            print(f"Volume profile error: {e}")
            
        return df
    
    @staticmethod
    def _detect_liquidity_zones(df: pd.DataFrame) -> pd.DataFrame:
        """Detect liquidity zones"""
        try:
            # High volume nodes
            if 'volume' in df.columns:
                df['micro_volume_ma'] = df['volume'].rolling(window=20).mean()
                df['micro_high_volume_node'] = df['volume'] > df['micro_volume_ma'] * 1.5
                
            # Price clusters (sideways movement)
            df['micro_price_range'] = df['high'].rolling(window=20).max() - df['low'].rolling(window=20).min()
            df['micro_avg_range'] = df['micro_price_range'].rolling(window=50).mean()
            df['micro_liquidity_zone'] = df['micro_price_range'] < df['micro_avg_range'] * 0.5
            
            # Stop hunt zones (spike and reverse)
            if 'high' in df.columns and 'low' in df.columns:
                # Detect spikes
                high_spikes = (df['high'] > df['high'].rolling(window=20).max().shift(1))
                low_spikes = (df['low'] < df['low'].rolling(window=20).min().shift(1))
                
                # Check for reversal after spike
                df['micro_stop_hunt_high'] = high_spikes & (df['close'].shift(-5) < df['high'])
                df['micro_stop_hunt_low'] = low_spikes & (df['close'].shift(-5) > df['low'])
                df['micro_stop_hunt'] = df['micro_stop_hunt_high'] | df['micro_stop_hunt_low']
                
        except Exception as e:
            print(f"Liquidity zones error: {e}")
            
        return df

class SMCAnalysis:
    """Smart Money Concepts Analysis"""
    
    @staticmethod
    def analyze_smc(df: pd.DataFrame) -> pd.DataFrame:
        """Comprehensive SMC analysis"""
        try:
            # Market structure
            df = SMCAnalysis._identify_market_structure(df)
            df = SMCAnalysis._find_order_blocks(df)
            df = SMCAnalysis._detect_fair_value_gaps(df)
            df = SMCAnalysis._identify_liquidity_zones(df)
            df = SMCAnalysis._detect_break_of_structure(df)
            df = SMCAnalysis._find_premium_discount_zones(df)
            
            print(f"✓ SMC analysis completed with {len([c for c in df.columns if 'smc_' in c.lower()])} indicators")
            
        except Exception as e:
            print(f"Warning: SMC analysis error: {e}")
            
        return df
    
    @staticmethod
    def _identify_market_structure(df: pd.DataFrame) -> pd.DataFrame:
        """Identify market structure (HH, HL, LH, LL)"""
        try:
            # Find swing highs and lows
            highs = find_peaks(df['high'].values, distance=5)[0]
            lows = find_peaks(-df['low'].values, distance=5)[0]
            
            # Initialize structure arrays
            df['SMC_swing_high'] = False
            df['SMC_swing_low'] = False
            df['SMC_structure'] = 'neutral'
            
            # Mark swing points
            if len(highs) > 0:
                df.iloc[highs, df.columns.get_loc('SMC_swing_high')] = True
            if len(lows) > 0:
                df.iloc[lows, df.columns.get_loc('SMC_swing_low')] = True
                
            # Determine structure trend
            all_points = []
            for h in highs:
                all_points.append((h, df['high'].iloc[h], 'high'))
            for l in lows:
                all_points.append((l, df['low'].iloc[l], 'low'))
                
            all_points.sort(key=lambda x: x[0])
            
            # Analyze structure pattern
            if len(all_points) >= 4:
                structure_trend = SMCAnalysis._analyze_structure_pattern(all_points)
                df['SMC_structure'] = structure_trend
                
        except Exception as e:
            print(f"Market structure error: {e}")
            
        return df
    
    @staticmethod
    def _analyze_structure_pattern(points: List[Tuple]) -> str:
        """Analyze HH, HL, LH, LL pattern"""
        if len(points) < 4:
            return 'insufficient_data'
            
        recent_points = points[-4:]
        pattern = []
        
        for i in range(1, len(recent_points)):
            curr_point = recent_points[i]
            prev_point = recent_points[i-1]
            
            if curr_point[2] == 'high':  # Current is high
                if prev_point[2] == 'low':  # Previous was low
                    # Compare with last high
                    last_high = None
                    for j in range(i-1, -1, -1):
                        if recent_points[j][2] == 'high':
                            last_high = recent_points[j]
                            break
                            
                    if last_high and curr_point[1] > last_high[1]:
                        pattern.append('HH')
                    elif last_high:
                        pattern.append('LH')
                        
            elif curr_point[2] == 'low':  # Current is low
                if prev_point[2] == 'high':  # Previous was high
                    # Compare with last low
                    last_low = None
                    for j in range(i-1, -1, -1):
                        if recent_points[j][2] == 'low':
                            last_low = recent_points[j]
                            break
                            
                    if last_low and curr_point[1] > last_low[1]:
                        pattern.append('HL')
                    elif last_low:
                        pattern.append('LL')
                        
        # Determine overall trend
        if 'HH' in pattern and 'HL' in pattern:
            return 'bullish'
        elif 'LH' in pattern and 'LL' in pattern:
            return 'bearish'
        else:
            return 'neutral'
    
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

# Helper Functions
def load_data(data_folder="./data"):
    """Load data from the specified folder"""
    data_files = {}
    
    if os.path.exists(data_folder):
        for file in os.listdir(data_folder):
            if file.endswith('.csv'):
                try:
                    pair_name = file.replace('.csv', '')
                    file_path = os.path.join(data_folder, file)
                    df = pd.read_csv(file_path)
                    
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
                    hovertemplate='%{text}
Code
(Output token limit reached)


continue

Claude Opus 4 Thinking
Thoughts
The user wants me to continue with the full dashboard file. I need to continue from where I left off in the code generation.

python
Copy Code
}<br>Time: %{x}'
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
            commentary.append(f"- **Dominant Flow**: BALANCED (avg imbalance: {avg_imbalance:.3f})")
        
        if 'micro_flow_efficiency' in df.columns:
            efficiency = df['micro_flow_efficiency'].mean()
            commentary.append(f"- **Flow Efficiency**: {efficiency:.3f} (higher = more directional)")
        commentary.append("")
    
    # Manipulation Detection
    if 'micro_manipulation_score' in df.columns:
        avg_manipulation = df['micro_manipulation_score'].mean()
        high_manipulation = (df['micro_manipulation_score'] > 0.5).sum()
        
        commentary.append("#### Market Manipulation Indicators")
        commentary.append(f"- **Average Manipulation Score**: {avg_manipulation:.3f}")
        commentary.append(f"- **High Manipulation Periods**: {high_manipulation} ({high_manipulation/total_ticks*100:.1f}% of ticks)")
        
        if 'micro_spoofing_detected' in df.columns:
            spoofing_events = df['micro_spoofing_detected'].sum()
            commentary.append(f"- **Spoofing Events**: {spoofing_events}")
            
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
    st.info("📁 Place your CSV files in the `./data` folder. Each file should contain: timestamp, bid, ask, mid columns.")

def microstructure_page(data_files):
    """Microstructure analysis page"""
    st.title("🔬 Microstructure Analysis")
    
    if not data_files:
        st.warning("No data files found. Please add CSV files to the ./data folder.")
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
        st.warning("No data files found. Please add CSV files to the ./data folder.")
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
    data_path = st.text_input("Data Folder Path", value="./data")
    
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

# Main App
def main():
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    
    pages = {
        "🏠 Home": home_page,
        "🔬 Microstructure": lambda: microstructure_page(data_files),
        "🎯 SMC & Wyckoff": lambda: smc_wyckoff_page(data_files),
        "⚙️ Settings": settings_page
    }
    
    selected_page = st.sidebar.radio("Go to", list(pages.keys()))
    
    # Load data
    data_files = load_data()
    
    # Display selected page
    pages[selected_page]()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("ZanFlow Ultimate Trading Dashboard v2.0")
    st.sidebar.caption("Last update: Every minute")

if __name__ == "__main__":
    main()
'''

# Save the complete dashboard file
with open('zanflow_ultimate_dashboard_complete.py', 'w', encoding='utf-8') as f:
    f.write(dashboard_code)

print("✅ Complete dashboard file created: zanflow_ultimate_dashboard_complete.py")
print("\n📊 Dashboard Features:")
print("- Full Microstructure Analysis with order flow, spread dynamics, and manipulation detection")
print("- Smart Money Concepts (SMC) with order blocks, FVGs, and market structure")
print("- Wyckoff Analysis with accumulation/distribution phases and spring/upthrust detection")
print("- Multi-page layout with Home, Microstructure, SMC & Wyckoff, and Settings pages")
print("- Real-time commentary generation")
print("- Interactive Plotly charts")
print("\n🚀 To run: streamlit run zanflow_ultimate_dashboard_complete.py")