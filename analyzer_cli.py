

from typing import Dict, List, Optional, Union, Tuple, Any, Set
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
from scipy import stats, signal
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class AnalyzerConfig:
    """Configuration for the market microstructure analyzer."""
    
    # Input/Output paths
    input_dir: Optional[Path] = None
    tick_data_paths: List[Path] = field(default_factory=list)
    bar_data_paths: List[Path] = field(default_factory=list)
    output_dir: Path = Path("./output")
    
    # Processing parameters
    symbols: List[str] = field(default_factory=lambda: ["XAUUSD"])
    max_tick_rows: Optional[int] = None
    max_bar_rows_per_timeframe: Optional[int] = None
    timeframes: List[str] = field(default_factory=lambda: ["M1", "M5", "M15", "H1"])
    remove_duplicates: bool = True
    
    # Analysis features
    enable_smc: bool = True
    enable_wyckoff: bool = True
    enable_inducement: bool = True
    enable_order_flow: bool = True
    enable_ml_features: bool = True
    enable_advanced_tick_analysis: bool = True
    
    # Technical indicators
    ema_periods: List[int] = field(default_factory=lambda: [9, 21, 50, 200])
    atr_period: int = 14
    rsi_period: int = 14
    
    # Market structure parameters
    structure_lookback: int = 50
    swing_threshold: float = 0.0001
    volume_profile_bins: int = 50
    
    # Advanced analysis parameters
    smc_lookback: int = 50
    wyckoff_window: int = 100
    ml_feature_window: int = 20
    tick_cluster_eps: float = 0.5
    tick_cluster_min_samples: int = 5
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AnalyzerConfig':
        """Create config from dictionary."""
        # Handle Path conversions
        path_fields = ['input_dir', 'output_dir']
        for field in path_fields:
            if field in config_dict and config_dict[field] is not None:
                config_dict[field] = Path(config_dict[field])
        
        if 'tick_data_paths' in config_dict:
            config_dict['tick_data_paths'] = [Path(p) for p in config_dict['tick_data_paths']]
        if 'bar_data_paths' in config_dict:
            config_dict['bar_data_paths'] = [Path(p) for p in config_dict['bar_data_paths']]
        
        return cls(
            ask_volume = pd.Series(0, index=df.index))
        
        for i in range(len(df)):
            if df.iloc[i].get('delta', 0) > 0:
                ask_volume.iloc[i] = df.iloc[i]['volume']
            else:
                bid_volume.iloc[i] = df.iloc[i]['volume']
        
        # Calculate rolling ratio
        bid_sum = bid_volume.rolling(20).sum()
        ask_sum = ask_volume.rolling(20).sum()
        
        ratio = ask_sum / (bid_sum + 1)  # Avoid division by zero
        
        return ratio
    
    def _calculate_advanced_imbalance(self, df: pd.DataFrame) -> pd.Series:
        """Calculate advanced order flow imbalance."""
        imbalance = pd.Series(0, index=df.index)
        
        # Calculate rolling buy/sell volumes
        buy_volume = df[df['delta'] > 0]['volume'].rolling(20).sum()
        sell_volume = df[df['delta'] < 0]['volume'].abs().rolling(20).sum()
        
        # Imbalance calculation
        total_volume = buy_volume + sell_volume
        imbalance = (buy_volume - sell_volume) / (total_volume + 1)
        
        return imbalance.fillna(0)
    
    def _detect_large_orders_advanced(self, df: pd.DataFrame) -> pd.Series:
        """Detect large orders with advanced filtering."""
        large_order = pd.Series(False, index=df.index)
        
        # Dynamic threshold based on recent volume
        volume_mean = df['volume'].rolling(100).mean()
        volume_std = df['volume'].rolling(100).std()
        
        # Large order criteria
        threshold = volume_mean + self.large_order_threshold * volume_std
        
        # Additional filters
        for i in range(100, len(df)):
            if df.iloc[i]['volume'] > threshold.iloc[i]:
                # Check if it's genuine (not split across multiple ticks)
                surrounding_volume = df.iloc[max(0, i-5):i+5]['volume'].sum()
                if df.iloc[i]['volume'] > surrounding_volume * 0.5:
                    large_order.iloc[i] = True
        
        return large_order
    
    def _detect_iceberg_orders(self, df: pd.DataFrame) -> pd.Series:
        """Detect iceberg orders (hidden large orders)."""
        iceberg = pd.Series(False, index=df.index)
        
        # Look for consistent volume at same price level
        for i in range(self.iceberg_detection_window, len(df)):
            window = df.iloc[i-self.iceberg_detection_window:i]
            
            # Group by price level
            price_volumes = window.groupby('mid_price')['volume'].agg(['count', 'sum', 'mean'])
            
            # Iceberg characteristics:
            # - Multiple trades at same price
            # - Consistent volume per trade
            # - Total volume is large
            for price, stats in price_volumes.iterrows():
                if (stats['count'] >= 5 and  # Multiple trades
                    stats['sum'] > df['volume'].rolling(100).sum().iloc[i] * 0.1 and  # Large total
                    stats['std'] < stats['mean'] * 0.3):  # Consistent size
                    
                    # Mark ticks at this price as iceberg
                    iceberg.loc[window[window['mid_price'] == price].index] = True
        
        return iceberg
    
    def _detect_sweep_orders(self, df: pd.DataFrame) -> pd.Series:
        """Detect sweep orders (aggressive liquidity taking)."""
        sweep = pd.Series(False, index=df.index)
        
        # Sweep characteristics: rapid price movement with high volume
        for i in range(5, len(df)):
            recent_ticks = df.iloc[i-5:i]
            
            price_move = abs(recent_ticks['mid_price'].iloc[-1] - recent_ticks['mid_price'].iloc[0])
            avg_spread = recent_ticks['spread'].mean()
            
            # Check if moved through multiple price levels quickly
            if (price_move > avg_spread * 3 and  # Moved multiple levels
                recent_ticks['volume'].sum() > df['volume'].rolling(50).sum().iloc[i] * 0.2 and  # High volume
                recent_ticks['trade_intensity'].mean() > 2):  # Fast execution
                
                sweep.iloc[i-5:i] = True
        
        return sweep
    
    def _detect_absorption_advanced(self, df: pd.DataFrame) -> pd.Series:
        """Detect absorption (large orders absorbing market orders)."""
        absorption = pd.Series('none', index=df.index)
        
        for i in range(20, len(df)):
            window = df.iloc[i-20:i]
            
            # Check for price stability despite volume
            price_range = window['mid_price'].max() - window['mid_price'].min()
            avg_range = df.iloc[i-100:i]['mid_price'].rolling(20).apply(lambda x: x.max() - x.min()).mean()
            
            # High volume but small price movement = absorption
            if (window['volume'].sum() > df['volume'].rolling(20).sum().mean() * self.absorption_threshold and
                price_range < avg_range * 0.3):
                
                # Determine direction
                if window['delta'].sum() > 0:
                    absorption.iloc[i] = 'selling_absorption'  # Sellers absorbed
                else:
                    absorption.iloc[i] = 'buying_absorption'  # Buyers absorbed
        
        return absorption
    
    def _detect_exhaustion(self, df: pd.DataFrame) -> pd.Series:
        """Detect exhaustion patterns in order flow."""
        exhaustion = pd.Series('none', index=df.index)
        
        for i in range(50, len(df)):
            # Check for decreasing momentum
            recent_delta = df.iloc[i-20:i]['delta'].sum()
            previous_delta = df.iloc[i-40:i-20]['delta'].sum()
            
            # Volume decreasing
            recent_volume = df.iloc[i-20:i]['volume'].sum()
            previous_volume = df.iloc[i-40:i-20]['volume'].sum()
            
            # Price movement slowing
            recent_move = abs(df.iloc[i]['mid_price'] - df.iloc[i-20]['mid_price'])
            previous_move = abs(df.iloc[i-20]['mid_price'] - df.iloc[i-40]['mid_price'])
            
            # Exhaustion criteria
            if (abs(recent_delta) < abs(previous_delta) * 0.5 and
                recent_volume < previous_volume * 0.7 and
                recent_move < previous_move * 0.5):
                
                if previous_delta > 0:
                    exhaustion.iloc[i] = 'buying_exhaustion'
                else:
                    exhaustion.iloc[i] = 'selling_exhaustion'
        
        return exhaustion
    
    def _detect_hidden_liquidity(self, df: pd.DataFrame) -> pd.Series:
        """Detect hidden liquidity (dark pool activity indicators)."""
        hidden = pd.Series(False, index=df.index)
        
        # Look for price improvements and unusual executions
        for i in range(1, len(df)):
            # Price improvement (execution inside spread)
            if (df.iloc[i].get('last', df.iloc[i]['mid_price']) > df.iloc[i]['bid'] + 0.01 and
                df.iloc[i].get('last', df.iloc[i]['mid_price']) < df.iloc[i]['ask'] - 0.01):
                hidden.iloc[i] = True
            
            # Large volume with minimal market impact
            if (df.iloc[i]['volume'] > df['volume'].rolling(100).mean().iloc[i] * 3 and
                abs(df.iloc[i]['mid_price'] - df.iloc[i-1]['mid_price']) < df['spread'].iloc[i] * 0.1):
                hidden.iloc[i] = True
        
        return hidden
    
    def _detect_spoofing_patterns(self, df: pd.DataFrame) -> pd.Series:
        """Detect potential spoofing patterns."""
        spoofing = pd.Series(False, index=df.index)
        
        # Look for rapid cancellations and price movements
        # Note: This is approximate without full order book data
        for i in range(10, len(df)-10):
            # Check for sudden price movement after period of stability
            pre_window = df.iloc[i-10:i]
            post_window = df.iloc[i:i+10]
            
            pre_volatility = pre_window['mid_price'].std()
            price_jump = abs(post_window['mid_price'].iloc[0] - pre_window['mid_price'].iloc[-1])
            
            # Spoofing indicator: sudden move after quiet period
            if (pre_volatility < df['mid_price'].rolling(50).std().iloc[i] * 0.3 and
                price_jump > df['spread'].iloc[i] * 2 and
                post_window['volume'].iloc[0] < df['volume'].rolling(50).mean().iloc[i] * 0.5):
                spoofing.iloc[i] = True
        
        return spoofing
    
    def _analyze_footprint_advanced(self, df: pd.DataFrame) -> None:
        """Advanced footprint chart analysis."""
        # Create price levels for footprint
        df['price_level'] = (df['mid_price'] * 100).round() / 100  # Round to nearest cent
        
        # Aggregate volume at each price level
        df['level_buy_volume'] = 0
        df['level_sell_volume'] = 0
        
        for level in df['price_level'].unique():
            level_data = df[df['price_level'] == level]
            buy_vol = level_data[level_data['delta'] > 0]['volume'].sum()
            sell_vol = level_data[level_data['delta'] < 0]['volume'].abs().sum()
            
            df.loc[df['price_level'] == level, 'level_buy_volume'] = buy_vol
            df.loc[df['price_level'] == level, 'level_sell_volume'] = sell_vol
        
        # Calculate imbalances at each level
        df['level_imbalance'] = (df['level_buy_volume'] - df['level_sell_volume']) / (df['level_buy_volume'] + df['level_sell_volume'] + 1)
        
        # Identify high volume nodes (HVN) and low volume nodes (LVN)
        level_volumes = df.groupby('price_level')['volume'].sum()
        hvn_threshold = level_volumes.quantile(0.7)
        lvn_threshold = level_volumes.quantile(0.3)
        
        df['volume_node_type'] = 'normal'
        df.loc[df['price_level'].isin(level_volumes[level_volumes > hvn_threshold].index), 'volume_node_type'] = 'HVN'
        df.loc[df['price_level'].isin(level_volumes[level_volumes < lvn_threshold].index), 'volume_node_type'] = 'LVN'
    
    def _analyze_tick_clusters(self, df: pd.DataFrame) -> None:
        """Analyze tick clustering patterns using DBSCAN."""
        if len(df) < 100:
            df['tick_cluster'] = -1
            return
        
        # Features for clustering
        features = []
        for i in range(len(df)):
            features.append([
                df.iloc[i]['mid_price'],
                df.iloc[i]['volume'],
                df.iloc[i].get('trade_intensity', 1),
                df.iloc[i].get('delta', 0)
            ])
        
        # Normalize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Apply DBSCAN
        clustering = DBSCAN(eps=self.tick_cluster_eps, min_samples=self.tick_cluster_min_samples)
        df['tick_cluster'] = clustering.fit_predict(features_scaled)
        
        # Analyze cluster characteristics
        for cluster_id in df['tick_cluster'].unique():
            if cluster_id != -1:  # -1 is noise
                cluster_data = df[df['tick_cluster'] == cluster_id]
                
                # Determine cluster type
                if cluster_data['delta'].sum() > 0 and cluster_data['volume'].sum() > df['volume'].sum() * 0.05:
                    df.loc[df['tick_cluster'] == cluster_id, 'cluster_type'] = 'accumulation'
                elif cluster_data['delta'].sum() < 0 and cluster_data['volume'].sum() > df['volume'].sum() * 0.05:
                    df.loc[df['tick_cluster'] == cluster_id, 'cluster_type'] = 'distribution'
                else:
                    df.loc[df['tick_cluster'] == cluster_id, 'cluster_type'] = 'neutral'


class AdvancedTickAnalyzer:
    """Sophisticated tick-level analysis for ultra-high frequency patterns."""
    
    def __init__(self):
        self.microstructure_window = 100
        self.tick_aggregation_levels = [10, 50, 100, 500]
    
    def analyze_tick_microstructure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform advanced tick-level microstructure analysis."""
        df = df.copy()
        
        # Tick-level metrics
        df['tick_direction'] = self._calculate_tick_direction(df)
        df['tick_run_length'] = self._calculate_tick_runs(df)
        df['micro_volatility'] = self._calculate_micro_volatility(df)
        df['quote_intensity'] = self._calculate_quote_intensity(df)
        
        # Advanced spread analysis
        df['effective_spread'] = self._calculate_effective_spread(df)
        df['realized_spread'] = self._calculate_realized_spread(df)
        df['spread_volatility'] = df['spread'].rolling(50).std()
        
        # Tick aggregation analysis
        self._analyze_tick_aggregations(df)
        
        # Micro patterns
        df['micro_reversal'] = self._detect_micro_reversals(df)
        df['tick_momentum'] = self._calculate_tick_momentum(df)
        df['micro_trend'] = self._detect_micro_trends(df)
        
        # Quote dynamics
        df['quote_slope'] = self._calculate_quote_slope(df)
        df['bid_ask_bounce'] = self._detect_bid_ask_bounce(df)
        
        # Information content
        df['tick_information'] = self._calculate_tick_information(df)
        df['price_discovery'] = self._calculate_price_discovery(df)
        
        # HFT detection
        df['hft_activity'] = self._detect_hft_patterns(df)
        
        return df
    
    def _calculate_tick_direction(self, df: pd.DataFrame) -> pd.Series:
        """Calculate tick direction (uptick, downtick, zerotick)."""
        direction = pd.Series('zero', index=df.index)
        
        price_change = df['mid_price'].diff()
        direction[price_change > 0] = 'up'
        direction[price_change < 0] = 'down'
        
        return direction
    
    def _calculate_tick_runs(self, df: pd.DataFrame) -> pd.Series:
        """Calculate consecutive tick runs in same direction."""
        runs = pd.Series(1, index=df.index)
        current_run = 1
        
        for i in range(1, len(df)):
            if df.iloc[i]['tick_direction'] == df.iloc[i-1]['tick_direction'] and df.iloc[i]['tick_direction'] != 'zero':
                current_run += 1
            else:
                current_run = 1
            runs.iloc[i] = current_run
        
        return runs
    
    def _calculate_micro_volatility(self, df: pd.DataFrame) -> pd.Series:
        """Calculate micro-level volatility."""
        # Use high-frequency volatility estimator
        micro_vol = pd.Series(0, index=df.index)
        
        for i in range(10, len(df)):
            window = df.iloc[i-10:i]
            
            # Realized volatility at tick level
            returns = window['mid_price'].pct_change().dropna()
            if len(returns) > 0:
                micro_vol.iloc[i] = returns.std() * np.sqrt(252 * 6.5 * 60 * 60 / 10)  # Annualized
        
        return micro_vol
    
    def _calculate_quote_intensity(self, df: pd.DataFrame) -> pd.Series:
        """Calculate quote update intensity."""
        intensity = pd.Series(1, index=df.index)
        
        # Count quote changes
        quote_changes = ((df['bid'].diff() != 0) | (df['ask'].diff() != 0)).astype(int)
        
        # Rolling intensity
        intensity = quote_changes.rolling(20).sum() / 20
        
        return intensity
    
    def _calculate_effective_spread(self, df: pd.DataFrame) -> pd.Series:
        """Calculate effective spread (execution vs mid-price)."""
        if 'last' not in df.columns:
            return df['spread']
        
        effective = 2 * abs(df['last'] - df['mid_price'])
        return effective
    
    def _calculate_realized_spread(self, df: pd.DataFrame) -> pd.Series:
        """Calculate realized spread (execution vs future mid-price)."""
        realized = pd.Series(0, index=df.index)
        
        forward_window = 50  # Look ahead 50 ticks
        
        for i in range(len(df) - forward_window):
            if 'last' in df.columns:
                current_trade = df.iloc[i]['last']
                future_mid = df.iloc[i + forward_window]['mid_price']
                
                # Trade direction
                if df.iloc[i]['delta'] > 0:  # Buy
                    realized.iloc[i] = 2 * (current_trade - future_mid)
                else:  # Sell
                    realized.iloc[i] = 2 * (future_mid - current_trade)
        
        return realized
    
    def _analyze_tick_aggregations(self, df: pd.DataFrame) -> None:
        """Analyze patterns at different tick aggregation levels."""
        for agg_level in self.tick_aggregation_levels:
            # Volume-weighted metrics
            vwap_col = f'vwap_{agg_level}'
            df[vwap_col] = df['mid_price'].rolling(agg_level).apply(
                lambda x: np.average(x, weights=df.loc[x.index, 'volume']) if len(x) > 0 else x.mean()
            )
            
            # Aggregated volume
            df[f'volume_sum_{agg_level}'] = df['volume'].rolling(agg_level).sum()
            
            # Net order flow
            df[f'net_flow_{agg_level}'] = df['delta'].rolling(agg_level).sum()
            
            # Tick volatility
            df[f'tick_vol_{agg_level}'] = df['mid_price'].rolling(agg_level).std()
    
    def _detect_micro_reversals(self, df: pd.DataFrame) -> pd.Series:
        """Detect micro-level price reversals."""
        reversals = pd.Series(False, index=df.index)
        
        for i in range(5, len(df)-5):
            # V-shaped reversal
            if (df.iloc[i-3:i]['mid_price'].is_monotonic_decreasing and
                df.iloc[i:i+3]['mid_price'].is_monotonic_increasing):
                reversals.iloc[i] = True
            
            # Inverted V reversal
            elif (df.iloc[i-3:i]['mid_price'].is_monotonic_increasing and
                  df.iloc[i:i+3]['mid_price'].is_monotonic_decreasing):
                reversals.iloc[i] = True
        
        return reversals
    
    def _calculate_tick_momentum(self, df: pd.DataFrame) -> pd.Series:
        """Calculate tick-level momentum."""
        # Use tick runs and volume
        momentum = df['tick_run_length'] * df['volume'] * df['tick_direction'].map({'up': 1, 'down': -1, 'zero': 0})
        
        # Normalize
        momentum = momentum / momentum.rolling(100).std()
        
        return momentum.fillna(0)
    
    def _detect_micro_trends(self, df: pd.DataFrame) -> pd.Series:
        """Detect micro trends using linear regression."""
        trend = pd.Series('neutral', index=df.index)
        
        window = 20
        
        for i in range(window, len(df)):
            y = df.iloc[i-window:i]['mid_price'].values
            x = np.arange(len(y))
            
            # Linear regression
            slope = np.polyfit(x, y, 1)[0]
            
            # Classify trend
            if slope > df.iloc[i]['spread'] / window:
                trend.iloc[i] = 'up'
            elif slope < -df.iloc[i]['spread'] / window:
                trend.iloc[i] = 'down'
        
        return trend
    
    def _calculate_quote_slope(self, df: pd.DataFrame) -> pd.Series:
        """Calculate the slope of quote midpoint changes."""
        slopes = pd.Series(0, index=df.index)
        
        window = 10
        
        for i in range(window, len(df)):
            mid_prices = df.iloc[i-window:i]['mid_price'].values
            
            if len(set(mid_prices)) > 1:  # Avoid constant prices
                x = np.arange(len(mid_prices))
                slope = np.polyfit(x, mid_prices, 1)[0]
                slopes.iloc[i] = slope
        
        return slopes
    
    def _detect_bid_ask_bounce(self, df: pd.DataFrame) -> pd.Series:
        """Detect bid-ask bounce patterns."""
        bounce = pd.Series(False, index=df.index)
        
        if 'last' not in df.columns:
            return bounce
        
        for i in range(2, len(df)):
            # Check if price bounces between bid and ask
            if (abs(df.iloc[i-2]['last'] - df.iloc[i-2]['bid']) < 0.01 and
                abs(df.iloc[i-1]['last'] - df.iloc[i-1]['ask']) < 0.01 and
                abs(df.iloc[i]['last'] - df.iloc[i]['bid']) < 0.01):
                bounce.iloc[i] = True
        
        return bounce
    
    def _calculate_tick_information(self, df: pd.DataFrame) -> pd.Series:
        """Calculate information content of each tick."""
        info = pd.Series(0, index=df.index)
        
        # Price impact as proxy for information
        for i in range(1, len(df)):
            price_change = abs(df.iloc[i]['mid_price'] - df.iloc[i-1]['mid_price'])
            volume = df.iloc[i]['volume']
            
            # Kyle's lambda approximation
            if volume > 0:
                info.iloc[i] = price_change / np.sqrt(volume)
        
        # Normalize
        info = info / info.rolling(100).mean()
        
        return info.fillna(1)
    
    def _calculate_price_discovery(self, df: pd.DataFrame) -> pd.Series:
        """Calculate price discovery metrics."""
        discovery = pd.Series(0, index=df.index)
        
        # Use information share approach
        window = 50
        
        for i in range(window, len(df)):
            # Calculate price innovations
            price_changes = df.iloc[i-window:i]['mid_price'].diff().dropna()
            
            if len(price_changes) > 0:
                # Variance of price changes
                innovation_var = price_changes.var()
                
                # Contribution to price discovery
                discovery.iloc[i] = innovation_var * df.iloc[i]['volume']
        
        # Normalize
        discovery = discovery / discovery.rolling(200).mean()
        
        return discovery.fillna(1)
    
    def _detect_hft_patterns(self, df: pd.DataFrame) -> pd.Series:
        """Detect potential HFT activity patterns."""
        hft = pd.Series(False, index=df.index)
        
        # HFT indicators:
        # 1. Very short time between trades
        # 2. Small trade sizes
        # 3. Quote stuffing patterns
        # 4. Rapid cancellations (approximated)
        
        for i in range(10, len(df)):
            window = df.iloc[i-10:i]
            
            # Check for HFT characteristics
            avg_time_between = window.index.to_series().diff().mean().total_seconds() if hasattr(window.index, 'to_series') else 1
            
            small_trades = (window['volume'] < df['volume'].quantile(0.2)).sum() > 7
            high_quote_intensity = window['quote_intensity'].mean() > 2
            rapid_price_changes = window['micro_volatility'].mean() > df['micro_volatility'].quantile(0.8)
            
            if (avg_time_between < 0.1 and  # Sub-second trading
                small_trades and
                (high_quote_intensity or rapid_price_changes)):
                hft.iloc[i] = True
        
        return hft


class MarketStructureAnalyzer:
    """Analyze market structure patterns."""
    
    def __init__(self, lookback: int = 50):
        self.lookback = lookback
    
    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect market structure patterns."""
        df = df.copy()
        
        # Basic structure
        df['pivot_high'] = self._find_pivot_highs(df)
        df['pivot_low'] = self._find_pivot_lows(df)
        df['swing_high'] = self._find_swing_highs(df)
        df['swing_low'] = self._find_swing_lows(df)
        
        # Trend structure
        df['market_structure'] = self._determine_market_structure(df)
        df['trend_strength'] = self._calculate_trend_strength(df)
        
        # Support/Resistance
        self._identify_support_resistance(df)
        
        return df
    
    def _find_pivot_highs(self, df: pd.DataFrame, left_bars: int = 5, right_bars: int = 5) -> pd.Series:
        """Find pivot highs."""
        pivot_highs = pd.Series(False, index=df.index)
        
        for i in range(left_bars, len(df) - right_bars):
            is_pivot = True
            
            # Check left side
            for j in range(i - left_bars, i):
                if df.iloc[j]['high'] >= df.iloc[i]['high']:
                    is_pivot = False
                    break
            
            # Check right side
            if is_pivot:
                for j in range(i + 1, i + right_bars + 1):
                    if df.iloc[j]['high'] >= df.iloc[i]['high']:
                        is_pivot = False
                        break
            
            pivot_highs.iloc[i] = is_pivot
        
        return pivot_highs
    
    def _find_pivot_lows(self, df: pd.DataFrame, left_bars: int = 5, right_bars: int = 5) -> pd.Series:
        """Find pivot lows."""
        pivot_lows = pd.Series(False, index=df.index)
        
        for i in range(left_bars, len(df) - right_bars):
            is_pivot = True
            
            # Check left side
            for j in range(i - left_bars, i):
                if df.iloc[j]['low'] <= df.iloc[i]['low']:
                    is_pivot = False
                    break
            
            # Check right side
            if is_pivot:
                for j in range(i + 1, i + right_bars + 1):
                    if df.iloc[j]['low'] <= df.iloc[i]['low']:
                        is_pivot = False
                        break
            
            pivot_lows.iloc[i] = is_pivot
        
        return pivot_lows
    
    def _find_swing_highs(self, df: pd.DataFrame) -> pd.Series:
        """Find significant swing highs."""
        swing_highs = pd.Series(False, index=df.index)
        pivot_highs = df[df['pivot_high']]
        
        if len(pivot_highs) < 2:
            return swing_highs
        
        # Swing high must be higher than surrounding pivot highs
        for i in range(1, len(pivot_highs) - 1):
            current_idx = pivot_highs.index[i]
            current_high = pivot_highs.iloc[i]['high']
            prev_high = pivot_highs.iloc[i-1]['high']
            next_high = pivot_highs.iloc[i+1]['high']
            
            if current_high > prev_high and current_high > next_high:
                swing_highs.loc[current_idx] = True
        
        return swing_highs
    
    def _find_swing_lows(self, df: pd.DataFrame) -> pd.Series:
        """Find significant swing lows."""
        swing_lows = pd.Series(False, index=df.index)
        pivot_lows = df[df['pivot_low']]
        
        if len(pivot_lows) < 2:
            return swing_lows
        
        # Swing low must be lower than surrounding pivot lows
        for i in range(1, len(pivot_lows) - 1):
            current_idx = pivot_lows.index[i]
            current_low = pivot_lows.iloc[i]['low']
            prev_low = pivot_lows.iloc[i-1]['low']
            next_low = pivot_lows.iloc[i+1]['low']
            
            if current_low < prev_low and current_low < next_low:
                swing_lows.loc[current_idx] = True
        
        return swing_lows
    
    def _determine_market_structure(self, df: pd.DataFrame) -> pd.Series:
        """Determine overall market structure."""
        structure = pd.Series('ranging', index=df.index)
        
        # Track higher highs/lows and lower highs/lows
        swing_highs = df[df['swing_high']]['high'].values
        swing_lows = df[df['swing_low']]['low'].values
        
        for i in range(self.lookback, len(df)):
            recent_highs = []
            recent_lows = []
            
            # Get recent swing points
            for idx, high in enumerate(df[df['swing_high']].index):
                if i - self.lookback <= high <= i:
                    recent_highs.append(df.loc[high, 'high'])
            
            for idx, low in enumerate(df[df['swing_low']].index):
                if i - self.lookback <= low <= i:
                    recent_lows.append(df.loc[low, 'low'])
            
            # Determine structure
            if len(recent_highs) >= 2 and len(recent_lows) >= 2:
                if (recent_highs[-1] > recent_highs[-2] and 
                    recent_lows[-1] > recent_lows[-2]):
                    structure.iloc[i] = 'bullish'
                elif (recent_highs[-1] < recent_highs[-2] and 
                      recent_lows[-1] < recent_lows[-2]):
                    structure.iloc[i] = 'bearish'
        
        return structure
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> pd.Series:
        """Calculate trend strength using ADX-like approach."""
        # Simplified trend strength calculation
        high_low_range = df['high'] - df['low']
        close_change = df['close'].diff().abs()
        
        # Directional movement
        up_move = df['high'].diff()
        down_move = -df['low'].diff()
        
        # Smooth the movements
        period = 14
        smooth_up = up_move.rolling(period).mean()
        smooth_down = down_move.rolling(period).mean()
        
        # Calculate strength
        strength = (smooth_up - smooth_down).abs() / (smooth_up + smooth_down + 0.001)
        
        return strength.fillna(0)
    
    def _identify_support_resistance(self, df: pd.DataFrame) -> None:
        """Identify support and resistance levels."""
        # Find levels where price reversed multiple times
        df['is_support'] = False
        df['is_resistance'] = False
        
        # Price levels rounded to significant figures
        price_levels = df['close'].round(2)
        level_counts = price_levels.value_counts()
        
        # Significant levels (touched at least 3 times)
        significant_levels = level_counts[level_counts >= 3].index
        
        for level in significant_levels:
            # Check if level acted as support or resistance
            level_indices = df[abs(df['low'] - level) < level * 0.001].index
            
            for idx in level_indices:
                if idx > 10 and idx < len(df) - 10:
                    # Check price action around level
                    before = df.loc[idx-10:idx-1, 'close'].mean()
                    after = df.loc[idx+1:idx+10, 'close'].mean()
                    
                    if before > level and after > level and df.loc[idx, 'low'] <= level:
                        df.loc[idx, 'is_support'] = True
                    elif before < level and after < level and df.loc[idx, 'high'] >= level:
                        df.loc[idx, 'is_resistance'] = True


class TechnicalIndicatorCalculator:
    """Calculate technical indicators."""
    
    @staticmethod
    def calculate_indicators(df: pd.DataFrame, config: AnalyzerConfig) -> pd.DataFrame:
        """Calculate all technical indicators."""
        df = df.copy()
        
        # Price-based calculations
        df['hl_range'] = df['high'] - df['low']
        df['mid_price'] = (df['high'] + df['low']) / 2
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        
        # Candle patterns
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
        df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']
        
        # Moving averages
        for period in config.ema_periods:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
        
        # ATR
        df[f'atr_{config.atr_period}'] = TechnicalIndicatorCalculator._calculate_atr(df, config.atr_period)
        
        # RSI
        df[f'rsi_{config.rsi_period}'] = TechnicalIndicatorCalculator._calculate_rsi(df['close'], config.rsi_period)
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        
        # VWAP
        df['vwap'] = (df['typical_price'] * df['volume']).cumsum() / df['volume'].cumsum()
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Momentum
        df['momentum_10'] = df['close'] - df['close'].shift(10)
        df['roc_10'] = df['close'].pct_change(10) * 100
        
        return df
    
    @staticmethod
    def _calculate_atr(df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range."""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(period).mean()
        
        return atr
    
# ... rest of code remains same

@staticmethod
def _calculate_rsi(prices: pd.Series, period: int) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = prices.diff()
    # Positive gains
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    # Negative losses (absolute value)
    loss = delta.where(delta < 0, 0.0).abs().rolling(period).mean()
    
    # Prevent division by zero
    rs = gain / (loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.fillna(0)
# ... rest of code remains same

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Unified Market Microstructure Analyzer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze with config file
  python analyzer_cli.py --config config.toml
  
  # Analyze specific directory with limits
  python analyzer_cli.py --input-dir ./data --max-ticks 10000 --max-bars 5000
  
  # Analyze specific files
  python analyzer_cli.py --tick-files tick1.csv tick2.csv --bar-files bars.csv
  
  # Enable specific features
  python analyzer_cli.py --enable-smc --enable-wyckoff --enable-order-flow
        """
    )
    
    # Input/Output options
    io_group = parser.add_argument_group('Input/Output Options')
    io_group.add_argument('--config', type=Path, help='Path to configuration TOML file')
    io_group.add_argument('--input-dir', type=Path, help='Input directory containing data files')
    io_group.add_argument('--output-dir', type=Path, default=Path('./output'), 
                         help='Output directory for results (default: ./output)')
    io_group.add_argument('--tick-files', nargs='+', type=Path, 
                         help='Specific tick data files to process')
    io_group.add_argument('--bar-files', nargs='+', type=Path, 
                         help='Specific bar data files to process')
    
    # Processing limits
    limit_group = parser.add_argument_group('Processing Limits')
    limit_group.add_argument('--max-ticks', type=int, 
                           help='Maximum number of tick records to process per file')
    limit_group.add_argument('--max-bars', type=int, 
                           help='Maximum number of bar records to process per timeframe')
    limit_group.add_argument('--symbols', nargs='+', default=['XAUUSD'], 
                           help='Symbols to process (default: XAUUSD)')
    
    # Analysis features
    feature_group = parser.add_argument_group('Analysis Features')
    feature_group.add_argument('--enable-smc', action='store_true', 
                             help='Enable Smart Money Concepts analysis')
    feature_group.add_argument('--enable-wyckoff', action='store_true', 
                             help='Enable Wyckoff method analysis')
    feature_group.add_argument('--enable-order-flow', action='store_true', 
                             help='Enable order flow analysis')
    feature_group.add_argument('--enable-ml', action='store_true', 
                             help='Enable machine learning features')
    feature_group.add_argument('--enable-advanced-tick', action='store_true', 
                             help='Enable advanced tick analysis')
    feature_group.add_argument('--enable-all', action='store_true', 
                             help='Enable all analysis features')
    
    # Technical parameters
    tech_group = parser.add_argument_group('Technical Parameters')
    tech_group.add_argument('--ema-periods', nargs='+', type=int, default=[9, 21, 50, 200],
                          help='EMA periods (default: 9 21 50 200)')
    tech_group.add_argument('--atr-period', type=int, default=14,
                          help='ATR period (default: 14)')
    tech_group.add_argument('--rsi-period', type=int, default=14,
                          help='RSI period (default: 14)')
    
    # Advanced parameters
    adv_group = parser.add_argument_group('Advanced Parameters')
    adv_group.add_argument('--smc-lookback', type=int, default=50,
                         help='SMC analysis lookback period (default: 50)')
    adv_group.add_argument('--structure-lookback', type=int, default=50,
                         help='Market structure lookback period (default: 50)')
    adv_group.add_argument('--wyckoff-window', type=int, default=100,
                         help='Wyckoff analysis window (default: 100)')
    
    # Other options
    parser.add_argument('--remove-duplicates', action='store_true',
                       help='Remove duplicate records')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be processed without actually processing')
    
    return parser.parse_args()


def build_config(args) -> AnalyzerConfig:
    """Build configuration from arguments and config file."""
    config_dict = {}
    
    # Load from config file if provided
    if args.config and args.config.exists():
        config_dict = load_config(args.config)
    
    # Override with command-line arguments
    if args.input_dir:
        config_dict['input_dir'] = args.input_dir
    
    if args.output_dir:
        config_dict['output_dir'] = args.output_dir
    
    if args.tick_files:
        config_dict['tick_data_paths'] = args.tick_files
    elif args.input_dir:
        # Auto-discover tick files
        config_dict['tick_data_paths'] = find_data_files(args.input_dir, 'tick')
    
    if args.bar_files:
        config_dict['bar_data_paths'] = args.bar_files
    elif args.input_dir:
        # Auto-discover bar files
        config_dict['bar_data_paths'] = find_data_files(args.input_dir, 'bar')
    
    # Processing limits
    if args.max_ticks:
        config_dict['max_tick_rows'] = args.max_ticks
    
    if args.max_bars:
        config_dict['max_bar_rows_per_timeframe'] = args.max_bars
    
    if args.symbols:
        config_dict['symbols'] = args.symbols
    
    # Features
    if args.enable_all:
        config_dict['enable_smc'] = True
        config_dict['enable_wyckoff'] = True
        config_dict['enable_order_flow'] = True
        config_dict['enable_ml_features'] = True
        config_dict['enable_advanced_tick_analysis'] = True
    else:
        if args.enable_smc:
            config_dict['enable_smc'] = True
        if args.enable_wyckoff:
            config_dict['enable_wyckoff'] = True
        if args.enable_order_flow:
            config_dict['enable_order_flow'] = True
        if args.enable_ml:
            config_dict['enable_ml_features'] = True
        if args.enable_advanced_tick:
            config_dict['enable_advanced_tick_analysis'] = True
    
    # Technical parameters
    if args.ema_periods:
        config_dict['ema_periods'] = args.ema_periods
    if args.atr_period:
        config_dict['atr_period'] = args.atr_period
    if args.rsi_period:
        config_dict['rsi_period'] = args.rsi_period
    
    # Advanced parameters
    if args.smc_lookback:
        config_dict['smc_lookback'] = args.smc_lookback
    if args.structure_lookback:
        config_dict['structure_lookback'] = args.structure_lookback
    if args.wyckoff_window:
        config_dict['wyckoff_window'] = args.wyckoff_window
    
    # Other options
    if args.remove_duplicates:
        config_dict['remove_duplicates'] = True
    
    return AnalyzerConfig.from_dict(config_dict)


def process_files(analyzer: UnifiedAnalyzer, config: AnalyzerConfig, logger: logging.Logger):
    """Process all configured files."""
    results = {
        'tick_data': {},
        'bar_data': {},
        'metadata': {
            'processing_time': datetime.now().isoformat(),
            'config': config.__dict__
        }
    }
    
    # Process tick data
    if config.tick_data_paths:
        logger.info(f"Processing {len(config.tick_data_paths)} tick data files")
        
        for tick_file in config.tick_data_paths:
            if not tick_file.exists():
                logger.warning(f"Tick file not found: {tick_file}")
                continue
            
            logger.info(f"Processing tick file: {tick_file}")
            try:
                tick_df = load_tick_data(tick_file)
                processed_tick_df = analyzer.process_tick_data(tick_df)
                
                # Save results
                output_file = config.output_dir / f"{tick_file.stem}_analyzed.parquet"
                save_to_parquet(processed_tick_df, output_file)
                logger.info(f"Saved tick analysis to: {output_file}")
                
                results['tick_data'][str(tick_file)] = {
                    'records_processed': len(processed_tick_df),
                    'output_file': str(output_file)
                }
                
            except Exception as e:
                logger.error(f"Error processing tick file {tick_file}: {e}")
                results['tick_data'][str(tick_file)] = {'error': str(e)}
    
    # Process bar data
    if config.bar_data_paths:
        logger.info(f"Processing {len(config.bar_data_paths)} bar data files")
        
        for bar_file in config.bar_data_paths:
            if not bar_file.exists():
                logger.warning(f"Bar file not found: {bar_file}")
                continue
            
            logger.info(f"Processing bar file: {bar_file}")
            try:
                bar_df = load_bar_data(bar_file)
                
                # Detect timeframe from filename or data
                timeframe = detect_timeframe(bar_file, bar_df)
                
                processed_bar_df = analyzer.process_bar_data(bar_df, timeframe)
                
                # Save results
                output_file = config.output_dir / f"{bar_file.stem}_analyzed.parquet"
                save_to_parquet(processed_bar_df, output_file)
                logger.info(f"Saved bar analysis to: {output_file}")
                
                results['bar_data'][str(bar_file)] = {
                    'records_processed': len(processed_bar_df),
                    'timeframe': timeframe,
                    'output_file': str(output_file)
                }
                
            except Exception as e:
                logger.error(f"Error processing bar file {bar_file}: {e}")
                results['bar_data'][str(bar_file)] = {'error': str(e)}
    
    return results


def detect_timeframe(file_path: Path, df: pd.DataFrame) -> str:
    """Detect timeframe from filename or data."""
    filename = file_path.stem.upper()
    
    # Check filename for timeframe indicators
    timeframe_map = {
        'M1': 'M1', '1M': 'M1', '1MIN': 'M1',
        'M5': 'M5', '5M': 'M5', '5MIN': 'M5',
        'M15': 'M15', '15M': 'M15', '15MIN': 'M15',
        'M30': 'M30', '30M': 'M30', '30MIN': 'M30',
        'H1': 'H1', '1H': 'H1', '60M': 'H1',
        'H4': 'H4', '4H': 'H4',
        'D1': 'D1', '1D': 'D1', 'DAILY': 'D1'
    }
    
    for key, value in timeframe_map.items():
        if key in filename:
            return value
    
    # Try to detect from timestamp intervals
    if 'timestamp' in df.columns and len(df) > 1:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        intervals = df['timestamp'].diff().dropna()
        median_interval = intervals.median()
        
        if median_interval <= pd.Timedelta(minutes=1.5):
            return 'M1'
        elif median_interval <= pd.Timedelta(minutes=7):
            return 'M5'
        elif median_interval <= pd.Timedelta(minutes=20):
            return 'M15'
        elif median_interval <= pd.Timedelta(minutes=40):
            return 'M30'
        elif median_interval <= pd.Timedelta(hours=1.5):
            return 'H1'
        elif median_interval <= pd.Timedelta(hours=5):
            return 'H4'
        else:
            return 'D1'
    
    # Default
    return 'M1'


def print_summary(results: Dict[str, Any], logger: logging.Logger):
    """Print processing summary."""
    logger.info("\n" + "="*60)
    logger.info("PROCESSING SUMMARY")
    logger.info("="*60)
    
    # Tick data summary
    if results['tick_data']:
        logger.info("\nTick Data Processing:")
        for file, info in results['tick_data'].items():
            if 'error' in info:
                logger.info(f"  {file}: ERROR - {info['error']}")
            else:
                logger.info(f"  {file}: {info['records_processed']} records -> {info['output_file']}")
    
    # Bar data summary
    if results['bar_data']:
        logger.info("\nBar Data Processing:")
        for file, info in results['bar_data'].items():
            if 'error' in info:
                logger.info(f"  {file}: ERROR - {info['error']}")
            else:
                logger.info(f"  {file}: {info['records_processed']} {info['timeframe']} bars -> {info['output_file']}")
    
    logger.info("\n" + "="*60)


def main():
    """Main entry point."""
    args = parse_arguments()
    logger = setup_logging(args.verbose)
    
    try:
        # Build configuration
        config = build_config(args)
        
        # Create output directory
        config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Log configuration
        logger.info("Configuration:")
        logger.info(f"  Input directory: {config.input_dir}")
        logger.info(f"  Output directory: {config.output_dir}")
        logger.info(f"  Tick files: {len(config.tick_data_paths)}")
        logger.info(f"  Bar files: {len(config.bar_data_paths)}")
        logger.info(f"  Max tick rows: {config.max_tick_rows}")
        logger.info(f"  Max bar rows: {config.max_bar_rows_per_timeframe}")
        logger.info(f"  Features enabled: SMC={config.enable_smc}, Wyckoff={config.enable_wyckoff}, "
                   f"OrderFlow={config.enable_order_flow}, ML={config.enable_ml_features}, "
                   f"AdvancedTick={config.enable_advanced_tick_analysis}")
        
        if args.dry_run:
            logger.info("\nDRY RUN - No processing will be performed")
            return
        
        # Create analyzer
        analyzer = UnifiedAnalyzer(config)
        
        # Process files
        results = process_files(analyzer, config, logger)
        
        # Save metadata
        metadata_file = config.output_dir / 'processing_metadata.json'
        import json
        with open(metadata_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Print summary
        print_summary(results, logger)
        
        logger.info(f"\nProcessing complete! Results saved to: {config.output_dir}")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
'''

# Save the updated CLI
with open('analyzer_cli.py', 'w') as f:
    f.write(updated_cli + updated_cli_continued)

print("Created analyzer_cli.py (updated with advanced options)")

# Create an enhanced config.toml with all options
enhanced_config = '''# Unified Market Microstructure Analyzer Configuration

[paths]
# Input directory containing data files (optional if specifying individual files)
input_dir = "./data"

# Output directory for results
output_dir = "./output"

# Specific tick data files (optional)
# tick_data_paths = ["tick1.csv", "tick2.csv"]

# Specific bar data files (optional)
# bar_data_paths = ["bars_M1.csv", "bars_M5.csv"]

[processing]
# Symbols to process
symbols = ["XAUUSD"]

# Maximum rows to process (null for unlimited)
max_tick_rows = 100000  # Process last 100k ticks
max_bar_rows_per_timeframe = 10000  # Process last 10k bars per timeframe

# Timeframes to analyze for bar data
timeframes = ["M1", "M5", "M15", "H1"]

# Remove duplicate records
remove_duplicates = true

[features]
# Enable/disable analysis features
enable_smc = true                    # Smart Money Concepts
enable_wyckoff = true                # Wyckoff Method
enable_inducement = true             # Inducement patterns
enable_order_flow = true             # Order flow analysis
enable_ml_features = true            # Machine learning features
enable_advanced_tick_analysis = true # Advanced tick microstructure

[technical]
# Technical indicator parameters
ema_periods = [9, 21, 50, 200]
atr_period = 14
rsi_period = 14

[analysis]
# Market structure parameters
structure_lookback = 50
swing_threshold = 0.0001
volume_profile_bins = 50

# SMC parameters
smc_lookback = 50

# Wyckoff parameters
wyckoff_window = 100

# ML feature parameters
ml_feature_window = 20

# Tick clustering parameters
tick_cluster_eps = 0.5
tick_cluster_min_samples = 5

# Order flow parameters
[analysis.order_flow]
imbalance_threshold = 0.7
large_order_threshold = 2.0
iceberg_detection_window = 20
absorption_threshold = 1.5

# Advanced tick analysis parameters
[analysis.tick]
microstructure_window = 100
tick_aggregation_levels = [10, 50, 100, 500]

# Visualization settings (for future use)
[visualization]
generate_plots = false
plot_format = "png"
plot_dpi = 150
'''

with open('config.toml', 'w') as f:
    f.write(enhanced_config)

print("Created config.toml (enhanced configuration)")

# Create a comprehensive test file
test_analyzer = '''"""
Test suite for the Unified Market Microstructure Analyzer.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

from unified_microstructure_analyzer import (
    AnalyzerConfig, UnifiedAnalyzer, AdvancedSMCAnalyzer,
    AdvancedWyckoffAnalyzer, AdvancedOrderFlowAnalyzer,
    AdvancedTickAnalyzer, MarketStructureAnalyzer,
    TechnicalIndicatorCalculator
)


@pytest.fixture
def sample_tick_data():
    """Generate sample tick data for testing."""
    n_ticks = 1000
    base_price = 3300.0
    
    # Generate realistic tick data
    timestamps = pd.date_range(start='2025-06-26 20:00:00', periods=n_ticks, freq='100ms')
    
    # Random walk for price
    price_changes = np.random.normal(0, 0.1, n_ticks)
    mid_prices = base_price + np.cumsum(price_changes)
    
    # Bid-ask spread
    spreads = np.random.uniform(0.1, 0.3, n_ticks)
    
    data = {
        'timestamp': timestamps,
        'timestamp_ms': [int(ts.timestamp() * 1000) for ts in timestamps],
        'bid': mid_prices - spreads/2,
        'ask': mid_prices + spreads/2,
        'spread': spreads,
        'volume': np.random.exponential(10, n_ticks),
        'flags': 6,
        'last': mid_prices + np.random.uniform(-spreads/2, spreads/2, n_ticks)
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_bar_data():
    """Generate sample bar data for testing."""
    n_bars = 500
    base_price = 3300.0
    
    # Generate OHLC data
    timestamps = pd.date_range(start='2025-06-26', periods=n_bars, freq='1min')
    
    opens = base_price + np.cumsum(np.random.normal(0, 0.5, n_bars))
    
    # Generate high/low/close relative to open
    data = []
    for i, (ts, open_price) in enumerate(zip(timestamps, opens)):
        high = open_price + abs(np.random.normal(0, 0.3))
        low = open_price - abs(np.random.normal(0, 0.3))
        close = np.random.uniform(low, high)
        volume = np.random.exponential(50)
        
        data.append({
            'timestamp': ts,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume,
            'spread': np.random.uniform(10, 30),
            'real_volume': 0
        })
    
    return pd.DataFrame(data)


@pytest.fixture
def analyzer_config():
    """Create test analyzer configuration."""
    return AnalyzerConfig(
        max_tick_rows=1000,
        max_bar_rows_per_timeframe=500,
        enable_smc=True,
        enable_wyckoff=True,
        enable_order_flow=True,
        enable_advanced_tick_analysis=True
    )


class TestAnalyzerConfig:
    """Test configuration handling."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = AnalyzerConfig()
        assert config.symbols == ["XAUUSD"]
        assert config.enable_smc == True
        assert config.atr_period == 14
    
    def test_config_from_dict(self):
        """Test configuration from dictionary."""
        config_dict = {
            'symbols': ['EURUSD', 'GBPUSD'],
            'max_tick_rows': 5000,
            'enable_smc': False,
            'output_dir': './test_output'
        }
        
        config = AnalyzerConfig.from_dict(config_dict)
        assert config.symbols == ['EURUSD', 'GBPUSD']
        assert config.max_tick_rows == 5000
        assert config.enable_smc == False
        assert config.output_dir == Path('./test_output')


class TestSMCAnalyzer:
    """Test SMC analyzer functionality."""
    
    def test_smc_pattern_detection(self, sample_bar_data):
        """Test basic SMC pattern detection."""
        analyzer = AdvancedSMCAnalyzer()
        
        # Add required columns
        sample_bar_data['swing_high'] = False
        sample_bar_data['swing_low'] = False
        sample_bar_data['hl_range'] = sample_bar_data['high'] - sample_bar_data['low']
        
        # Mark some swing points
        sample_bar_data.loc[10, 'swing_high'] = True
        sample_bar_data.loc[20, 'swing_low'] = True
        
        result = analyzer.analyze(sample_bar_data)
        
        # Check that SMC columns were added
        assert 'bos_bullish' in result.columns
        assert 'fvg_bullish' in result.columns
        assert 'order_block' in result.columns
    
    def test_fair_value_gap_detection(self):
        """Test FVG detection logic."""
        analyzer = AdvancedSMCAnalyzer()
        
        # Create data with clear FVG
        data = pd.DataFrame({
            'high': [100, 101, 105, 106],
            'low': [99, 100, 104, 105],
            'close': [100, 101, 105, 106],
            'open': [99.5, 100.5, 104.5, 105.5],
            'volume': [100, 100, 200, 100],
            'atr_14': [1.0, 1.0, 1.0, 1.0]
        })
        
        # Initialize required columns
        for col in ['fvg_bullish', 'fvg_bearish']:
            data[col] = False
        
        analyzer._detect_fair_value_gaps_advanced(data)
        
        # Should detect bullish FVG between bars 1 and 3
        assert data['fvg_bullish'].any()


class TestWyckoffAnalyzer:
    """Test Wyckoff analyzer functionality."""
    
    def test_wyckoff_phase_detection(self, sample_bar_data):
        """Test Wyckoff phase detection."""
        analyzer = AdvancedWyckoffAnalyzer()
        
        result = analyzer.analyze(sample_bar_data)
        
        # Check that Wyckoff columns were added
        assert 'wyckoff_phase' in result.columns
        assert 'wyckoff_event' in result.columns
        assert 'price_volume_relationship' in result.columns
    
    def test_volume_analysis(self, sample_bar_data):
        """Test price-volume relationship analysis."""
        analyzer = AdvancedWyckoffAnalyzer()
        
        # Ensure volume column exists
        if 'volume' not in sample_bar_data.columns:
            sample_bar_data['volume'] = 100
        
        analyzer._analyze_price_volume_relationship(sample_bar_data)
        
        assert 'effort' in sample_bar_data.columns
        assert 'result' in sample_bar_data.columns
        assert 'price_volume_relationship' in sample_bar_data.columns


class TestOrderFlowAnalyzer:
    """Test order flow analyzer functionality."""
    
    def test_delta_calculation(self, sample_tick_data):
        """Test volume delta calculation."""
        analyzer = AdvancedOrderFlowAnalyzer()
        
        # Add mid_price column
        sample_tick_data['mid_price'] = (sample_tick_data['bid'] + sample_tick_data['ask']) / 2
        
        delta = analyzer._calculate_advanced_delta(sample_tick_data)
        
        assert len(delta) == len(sample_tick_data)
        assert delta.dtype in [np.float64, np.int64]
    
    def test_order_flow_imbalance(self, sample_tick_data):
        """Test order flow imbalance calculation."""
        analyzer = AdvancedOrderFlowAnalyzer()
        
        # Prepare data
        sample_tick_data['mid_price'] = (sample_tick_data['bid'] + sample_tick_data['ask']) / 2
        sample_tick_data['delta'] = analyzer._calculate_advanced_delta(sample_tick_data)
        
        imbalance = analyzer._calculate_advanced_imbalance(sample_tick_data)
        
        assert len(imbalance) == len(sample_tick_data)
        assert imbalance.min() >= -1
        assert imbalance.max() <= 1


class TestTickAnalyzer:
    """Test tick analyzer functionality."""
    
    def test_tick_microstructure_analysis(self, sample_tick_data):
        """Test tick microstructure analysis."""
        analyzer = AdvancedTickAnalyzer()
        
        # Add required columns
        sample_tick_data['mid_price'] = (sample_tick_data['bid'] + sample_tick_data['ask']) / 2
        sample_tick_data['spread'] = sample_tick_data['ask'] - sample_tick_data['bid']
        
        result = analyzer.analyze_tick_microstructure(sample_tick_data)
        
        # Check that tick analysis columns were added
        assert 'tick_direction' in result.columns
        assert 'micro_volatility' in result.columns
        assert 'tick_momentum' in result.columns
    
    def test_tick_clustering(self, sample_tick_data):
        """Test tick clustering functionality."""
        analyzer = AdvancedTickAnalyzer()
        
        # Prepare data
        sample_tick_data['mid_price'] = (sample_tick_data['bid'] + sample_tick_data['ask']) / 2
        sample_tick_data['delta'] = 0
        sample_tick_data['trade_intensity'] = 1
        
        analyzer._analyze_tick_clusters(sample_tick_data)
        
        assert 'tick_cluster' in sample_tick_data.columns


class TestMarketStructure:
    """Test market structure analyzer."""
    
    def test_pivot_detection(self, sample_bar_data):
        """Test pivot high/low detection."""
        analyzer = MarketStructureAnalyzer()
        
        result = analyzer.analyze(sample_bar_data)
        
        assert 'pivot_high' in result.columns
        assert 'pivot_low' in result.columns
        assert 'swing_high' in result.columns
        assert 'swing_low' in result.columns
    
    def test_trend_detection(self, sample_bar_data):
        """Test market structure trend detection."""
        analyzer = MarketStructureAnalyzer()
        
        result = analyzer.analyze(sample_bar_data)
        
        assert 'market_structure' in result.columns
        assert 'trend_strength' in result.columns
        
        # Check valid market structure values
        valid_structures = ['bullish', 'bearish', 'ranging']
        assert all(v in valid_structures for v in result['market_structure'].unique())


class TestTechnicalIndicators:
    """Test technical indicator calculations."""
    
    def test_indicator_calculation(self, sample_bar_data, analyzer_config):
        """Test technical indicator calculations."""
        calculator = TechnicalIndicatorCalculator()
        
        result = calculator.calculate_indicators(sample_bar_data, analyzer_config)
        
        # Check that indicators were calculated
        assert 'ema_9' in result.columns
        assert 'atr_14' in result.columns
        assert 'rsi_14' in result.columns
        assert 'bb_upper' in result.columns
        assert 'vwap' in result.columns
    
    def test_rsi_calculation(self):
        """Test RSI calculation accuracy."""
        # Create simple test data
        prices = pd.Series([100, 102, 101, 103, 102, 104, 103, 105, 104, 106])
        
        rsi = TechnicalIndicatorCalculator._calculate_rsi(prices, period=5)
        
        assert len(rsi) == len(prices)
        assert rsi.min() >= 0
        assert rsi.max() <= 100


class TestUnifiedAnalyzer:
    """Test the main unified analyzer."""
    
    def test_tick_processing(self, sample_tick_data, analyzer_config):
        """Test tick data processing."""
        analyzer = UnifiedAnalyzer(analyzer_config)
        
        result = analyzer.process_tick_data(sample_tick_data)
        
        assert len(result) <= analyzer_config.max_tick_rows
        assert 'mid_price' in result.columns
        assert 'spread' in result.columns
    
    def test_bar_processing(self, sample_bar_data, analyzer_config):
        """Test bar data processing."""
        analyzer = UnifiedAnalyzer(analyzer_config)
        
        result = analyzer.process_bar_data(sample_bar_data, 'M1')
        
        assert len(result) <= analyzer_config.max_bar_rows_per_timeframe
        assert 'timeframe' in result.columns
        assert result['timeframe'].iloc[0] == 'M1'
    
    def test_duplicate_removal(self, sample_tick_data):
        """Test duplicate removal functionality."""
        # Create config with duplicate removal enabled
        config = AnalyzerConfig(remove_duplicates=True)
        analyzer = UnifiedAnalyzer(config)
        
        # Add duplicate rows
        duplicate_data = pd.concat([sample_tick_data, sample_tick_data.iloc[:10]])
        
        result = analyzer._preprocess_tick_data(duplicate_data)
        
        assert len(result) == len(sample_tick_data)


class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    def test_full_pipeline(self, sample_tick_data, sample_bar_data, analyzer_config):
        """Test complete analysis pipeline."""
        analyzer = UnifiedAnalyzer(analyzer_config)
        
        # Process tick data
        tick_result = analyzer.process_tick_data(sample_tick_data)
        assert isinstance(tick_result, pd.DataFrame)
        assert len(tick_result) > 0
        
        # Process bar data
        bar_result = analyzer.process_bar_data(sample_bar_data, 'M1')
        assert isinstance(bar_result, pd.DataFrame)
        assert len(bar_result) > 0
        
        # Check that all enabled features produced output
        if analyzer_config.enable_smc:
            assert any(col.startswith('bos_') or col.startswith('fvg_') 
                      for col in bar_result.columns)
        
        if analyzer_config.enable_wyckoff:
            assert 'wyckoff_phase' in bar_result.columns
        
        if analyzer_config.enable_order_flow:
            assert 'delta' in tick_result.columns
        
        if analyzer_config.enable_advanced_tick_analysis:
            assert 'tick_direction' in tick_result.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])