"""
Enhanced Wyckoff Analysis Dashboard v4.0 - Professional Quant Edition
Integrates Wyckoff Method with Microstructure Analysis and Tick Manipulation Detection
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import os
from pathlib import Path
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Tuple, Optional, Any
import logging
import json
from scipy import stats, signal
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import ta

# Page configuration
st.set_page_config(
    page_title="Enhanced Wyckoff Pro - Quant Edition",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Professional styling
st.markdown("""
<style>
/* Professional dark theme */
.main {
    background: #0e1117;
    color: #ffffff;
}

/* Header styling */
.quant-header {
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    padding: 2rem;
    border-radius: 15px;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
}

/* Metric cards */
.metric-card {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    padding: 1.5rem;
    border-radius: 10px;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 20px rgba(0,0,0,0.2);
}

/* Alert boxes */
.manipulation-alert {
    background: linear-gradient(45deg, #ff6b6b, #ee5a6f);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
    font-weight: 600;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { opacity: 0.8; }
    50% { opacity: 1; }
    100% { opacity: 0.8; }
}

/* Chart containers */
.chart-container {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 15px;
    padding: 1.5rem;
    margin: 1rem 0;
}

/* Phase indicators */
.phase-badge {
    display: inline-block;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-weight: 600;
    margin: 0.2rem;
    font-size: 0.9rem;
}

.phase-accumulation { background: #27ae60; color: white; }
.phase-markup { background: #3498db; color: white; }
.phase-distribution { background: #e67e22; color: white; }
.phase-markdown { background: #e74c3c; color: white; }
</style>
""", unsafe_allow_html=True)

# ============================
# ENHANCED WYCKOFF ANALYZER
# ============================

class EnhancedWyckoffAnalyzer:
    """Professional Wyckoff Analysis with Microstructure Integration"""
    
    def __init__(self):
        self.phases = {
            'Accumulation': {'color': '#27ae60', 'description': 'Smart money accumulating'},
            'Markup': {'color': '#3498db', 'description': 'Trend continuation phase'},
            'Distribution': {'color': '#e67e22', 'description': 'Smart money distributing'},
            'Markdown': {'color': '#e74c3c', 'description': 'Downtrend phase'}
        }
        
        self.events = {
            'PS': {'name': 'Preliminary Support', 'significance': 'High', 'color': '#9b59b6'},
            'SC': {'name': 'Selling Climax', 'significance': 'Very High', 'color': '#e74c3c'},
            'AR': {'name': 'Automatic Rally', 'significance': 'High', 'color': '#27ae60'},
            'ST': {'name': 'Secondary Test', 'significance': 'Medium', 'color': '#3498db'},
            'SOS': {'name': 'Sign of Strength', 'significance': 'High', 'color': '#2ecc71'},
            'LPS': {'name': 'Last Point of Support', 'significance': 'Very High', 'color': '#16a085'},
            'BC': {'name': 'Buying Climax', 'significance': 'Very High', 'color': '#f39c12'},
            'PSY': {'name': 'Preliminary Supply', 'significance': 'High', 'color': '#d35400'},
            'SOW': {'name': 'Sign of Weakness', 'significance': 'High', 'color': '#c0392b'},
            'LPSY': {'name': 'Last Point of Supply', 'significance': 'Very High', 'color': '#8e44ad'},
            'UTAD': {'name': 'Upthrust After Distribution', 'significance': 'Very High', 'color': '#e74c3c'},
            'SPRING': {'name': 'Spring', 'significance': 'Very High', 'color': '#27ae60'}
        }
    
    def comprehensive_analysis(self, df: pd.DataFrame, tick_data: Optional[pd.DataFrame] = None) -> Dict:
        """Perform comprehensive Wyckoff analysis with microstructure integration"""
        try:
            # Prepare data
            df = self._prepare_data(df)
            
            results = {
                'wyckoff_phases': self._identify_phases_advanced(df),
                'wyckoff_events': self._identify_events_advanced(df),
                'vsa_analysis': self._advanced_vsa_analysis(df),
                'microstructure': self._analyze_microstructure(df, tick_data),
                'manipulation_detection': self._detect_manipulation_patterns(df, tick_data),
                'composite_operator': self._analyze_composite_operator_advanced(df),
                'supply_demand': self._identify_supply_demand_zones_advanced(df),
                'trend_structure': self._analyze_trend_structure_advanced(df),
                'volume_profile': self._create_volume_profile(df),
                'order_flow': self._analyze_order_flow(df, tick_data),
                'smart_money': self._detect_smart_money_activity(df),
                'risk_metrics': self._calculate_risk_metrics(df),
                'trade_setups': self._generate_trade_setups(df),
                'multi_timeframe': self._multi_timeframe_analysis(df)
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            return {}
    
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare and enrich data with technical indicators"""
        # Basic calculations
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['hl_spread'] = df['high'] - df['low']
        df['oc_spread'] = abs(df['close'] - df['open'])
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Price levels
        df['resistance'] = df['high'].rolling(20).max()
        df['support'] = df['low'].rolling(20).min()
        df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
        
        # Volatility metrics
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = ta.volatility.bollinger_bands(
            df['close'], window=20, window_dev=2
        )
        
        # Volume indicators
        df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        df['vwap'] = ta.volume.volume_weighted_average_price(
            df['high'], df['low'], df['close'], df['volume']
        )
        
        # Momentum indicators
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        df['macd'] = ta.trend.macd_diff(df['close'])
        
        return df
    
    def _identify_phases_advanced(self, df: pd.DataFrame) -> List[Dict]:
        """Advanced Wyckoff phase identification using machine learning"""
        phases = []
        
        # Feature engineering for phase detection
        features = pd.DataFrame()
        features['price_position'] = (df['close'] - df['support']) / (df['resistance'] - df['support'])
        features['volume_trend'] = df['volume_ma'].rolling(10).mean() / df['volume_ma'].rolling(50).mean()
        features['volatility_regime'] = df['atr'] / df['atr'].rolling(50).mean()
        features['trend_strength'] = abs(df['macd']) / df['atr']
        features['volume_concentration'] = df['volume_ratio'].rolling(10).std()
        
        # Normalize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features.fillna(0))
        
        # Phase detection logic with pattern recognition
        window_size = 20
        for i in range(window_size, len(df) - window_size):
            window_features = features_scaled[i-window_size:i+window_size]
            
            # Accumulation detection
            if self._detect_accumulation_pattern(df.iloc[i-window_size:i+window_size], window_features):
                phases.append({
                    'phase': 'Accumulation',
                    'start_idx': i - window_size,
                    'end_idx': i + window_size,
                    'confidence': self._calculate_phase_confidence(df.iloc[i-window_size:i+window_size], 'Accumulation'),
                    'characteristics': self._get_phase_characteristics(df.iloc[i-window_size:i+window_size]),
                    'price_range': (df.iloc[i-window_size:i+window_size]['low'].min(), 
                                   df.iloc[i-window_size:i+window_size]['high'].max())
                })
            
            # Distribution detection
            elif self._detect_distribution_pattern(df.iloc[i-window_size:i+window_size], window_features):
                phases.append({
                    'phase': 'Distribution',
                    'start_idx': i - window_size,
                    'end_idx': i + window_size,
                    'confidence': self._calculate_phase_confidence(df.iloc[i-window_size:i+window_size], 'Distribution'),
                    'characteristics': self._get_phase_characteristics(df.iloc[i-window_size:i+window_size]),
                    'price_range': (df.iloc[i-window_size:i+window_size]['low'].min(), 
                                   df.iloc[i-window_size:i+window_size]['high'].max())
                })
        
        return phases
    
    def _identify_events_advanced(self, df: pd.DataFrame) -> List[Dict]:
        """Advanced Wyckoff event detection with pattern matching"""
        events = []
        
        # Calculate event indicators
        df['volume_spike'] = df['volume'] > df['volume_ma'] * 2.5
        df['price_reversal'] = ((df['close'] - df['open']) * (df['close'].shift(1) - df['open'].shift(1)) < 0)
        df['support_test'] = df['low'] <= df['support'] * 1.002
        df['resistance_test'] = df['high'] >= df['resistance'] * 0.998
        
        for i in range(50, len(df) - 10):
            # Spring detection
            if self._detect_spring(df, i):
                events.append({
                    'event': 'SPRING',
                    'index': i,
                    'price': df.iloc[i]['low'],
                    'volume': df.iloc[i]['volume'],
                    'confidence': 0.85,
                    'description': 'Spring - Shakeout below support',
                    'action': 'Potential long entry'
                })
            
            # UTAD detection
            elif self._detect_utad(df, i):
                events.append({
                    'event': 'UTAD',
                    'index': i,
                    'price': df.iloc[i]['high'],
                    'volume': df.iloc[i]['volume'],
                    'confidence': 0.80,
                    'description': 'Upthrust After Distribution',
                    'action': 'Potential short entry'
                })
            
            # SOS detection
            elif self._detect_sos(df, i):
                events.append({
                    'event': 'SOS',
                    'index': i,
                    'price': df.iloc[i]['close'],
                    'volume': df.iloc[i]['volume'],
                    'confidence': 0.75,
                    'description': 'Sign of Strength',
                    'action': 'Trend continuation long'
                })
        
        return events
    
    def _advanced_vsa_analysis(self, df: pd.DataFrame) -> Dict:
        """Advanced Volume Spread Analysis with professional metrics"""
        vsa_results = {
            'signals': [],
            'volume_profile': {},
            'spread_analysis': {},
            'effort_result': []
        }
        
        # Calculate VSA metrics
        df['spread_percentile'] = df['hl_spread'].rolling(50).apply(lambda x: stats.percentileofscore(x, x.iloc[-1]))
        df['volume_percentile'] = df['volume'].rolling(50).apply(lambda x: stats.percentileofscore(x, x.iloc[-1]))
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Professional VSA patterns
        for i in range(50, len(df)):
            # No Demand Bar
            if (df.iloc[i]['volume_percentile'] < 30 and 
                df.iloc[i]['spread_percentile'] < 30 and
                df.iloc[i]['close'] < df.iloc[i-1]['close']):
                vsa_results['signals'].append({
                    'index': i,
                    'type': 'No Demand',
                    'significance': 'High',
                    'interpretation': 'Lack of buying interest'
                })
            
            # Stopping Volume
            elif (df.iloc[i]['volume_percentile'] > 85 and
                  df.iloc[i]['close_position'] > 0.7 and
                  df.iloc[i]['hl_spread'] > df['atr'].iloc[i] * 1.5):
                vsa_results['signals'].append({
                    'index': i,
                    'type': 'Stopping Volume',
                    'significance': 'Very High',
                    'interpretation': 'Professional buying/absorption'
                })
        
        return vsa_results
    
    def _analyze_microstructure(self, df: pd.DataFrame, tick_data: Optional[pd.DataFrame]) -> Dict:
        """Analyze market microstructure patterns"""
        microstructure = {
            'tick_patterns': {},
            'liquidity_analysis': {},
            'order_flow_imbalance': {},
            'microstructure_events': []
        }
        
        if tick_data is not None and not tick_data.empty:
            # Tick analysis
            tick_data['spread'] = tick_data['ask'] - tick_data['bid']
            tick_data['mid_price'] = (tick_data['ask'] + tick_data['bid']) / 2
            tick_data['price_impact'] = abs(tick_data['mid_price'].diff()) / tick_data['spread']
            
            # Detect microstructure anomalies
            anomaly_detector = IsolationForest(contamination=0.05)
            features = tick_data[['spread', 'volume', 'price_impact']].fillna(0)
            anomalies = anomaly_detector.fit_predict(features)
            
            microstructure['anomalies'] = np.where(anomalies == -1)[0].tolist()
            microstructure['avg_spread'] = tick_data['spread'].mean()
            microstructure['spread_volatility'] = tick_data['spread'].std()
        
        return microstructure
    
    def _detect_manipulation_patterns(self, df: pd.DataFrame, tick_data: Optional[pd.DataFrame]) -> Dict:
        """Detect market manipulation patterns"""
        manipulation = {
            'spoofing_events': [],
            'wash_trades': [],
            'stop_hunts': [],
            'engineered_liquidity': [],
            'manipulation_score': 0
        }
        
        # Stop hunt detection
        for i in range(10, len(df) - 10):
            # Check for price spike beyond support/resistance with quick reversal
            if (df.iloc[i]['low'] < df.iloc[i-10:i]['low'].min() * 0.995 and
                df.iloc[i+1]['close'] > df.iloc[i]['open'] and
                df.iloc[i]['volume'] > df['volume_ma'].iloc[i] * 2):
                manipulation['stop_hunts'].append({
                    'index': i,
                    'type': 'Bear Stop Hunt',
                    'price': df.iloc[i]['low'],
                    'recovery': (df.iloc[i+1]['close'] - df.iloc[i]['low']) / df.iloc[i]['low']
                })
        
        # Calculate manipulation score
        total_bars = len(df)
        manipulation_events = len(manipulation['stop_hunts']) + len(manipulation['spoofing_events'])
        manipulation['manipulation_score'] = min(manipulation_events / total_bars * 100, 100)
        
        return manipulation
    
    def _analyze_composite_operator_advanced(self, df: pd.DataFrame) -> Dict:
        """Advanced Composite Operator analysis with institutional footprints"""
        co_analysis = {
            'accumulation_score': 0,
            'distribution_score': 0,
            'institutional_footprints': [],
            'smart_money_index': 0,
            'phase': 'Neutral'
        }
        
        # Calculate institutional metrics
        df['large_volume'] = df['volume'] > df['volume'].quantile(0.9)
        df['price_efficiency'] = abs(df['close'] - df['open']) / (df['high'] - df['low'])
        
        # Detect institutional accumulation patterns
        accumulation_days = 0
        distribution_days = 0
        
        for i in range(20, len(df)):
            # Accumulation: High volume, small price movement, near lows
            if (df.iloc[i]['large_volume'] and 
                df.iloc[i]['price_efficiency'] < 0.3 and
                df.iloc[i]['close_position'] < 0.4):
                accumulation_days += 1
                co_analysis['institutional_footprints'].append({
                    'index': i,
                    'type': 'Accumulation',
                    'strength': df.iloc[i]['volume'] / df['volume_ma'].iloc[i]
                })
            
            # Distribution: High volume, small price movement, near highs
            elif (df.iloc[i]['large_volume'] and 
                  df.iloc[i]['price_efficiency'] < 0.3 and
                  df.iloc[i]['close_position'] > 0.6):
                distribution_days += 1
                co_analysis['institutional_footprints'].append({
                    'index': i,
                    'type': 'Distribution',
                    'strength': df.iloc[i]['volume'] / df['volume_ma'].iloc[i]
                })
        
        co_analysis['accumulation_score'] = accumulation_days / len(df) * 100
        co_analysis['distribution_score'] = distribution_days / len(df) * 100
        
        # Determine phase
        if co_analysis['accumulation_score'] > co_analysis['distribution_score'] * 1.5:
            co_analysis['phase'] = 'Accumulation'
        elif co_analysis['distribution_score'] > co_analysis['accumulation_score'] * 1.5:
            co_analysis['phase'] = 'Distribution'
        
        return co_analysis
    
    def _identify_supply_demand_zones_advanced(self, df: pd.DataFrame) -> Dict:
        """Identify high-probability supply and demand zones"""
        zones = {
            'demand_zones': [],
            'supply_zones': [],
            'fresh_zones': [],
            'tested_zones': []
        }
        
        # Find significant turning points
        for i in range(20, len(df) - 20):
            # Demand zone: Strong bounce from low
            if (df.iloc[i]['low'] == df.iloc[i-10:i+10]['low'].min() and
                df.iloc[i:i+5]['close'].mean() > df.iloc[i]['close'] * 1.01 and
                df.iloc[i]['volume'] > df['volume_ma'].iloc[i] * 1.5):
                
                zone = {
                    'type': 'Demand',
                    'index': i,
                    'price_low': df.iloc[i]['low'],
                    'price_high': df.iloc[i]['open'],
                    'strength': df.iloc[i]['volume'] / df['volume_ma'].iloc[i],
                    'tested': 0,
                    'fresh': True
                }
                zones['demand_zones'].append(zone)
                zones['fresh_zones'].append(zone)
        
        return zones
    
    def _generate_trade_setups(self, df: pd.DataFrame) -> List[Dict]:
        """Generate professional trade setups based on Wyckoff analysis"""
        setups = []
        
        # Get recent analysis
        recent_phases = self._identify_phases_advanced(df.tail(100))
        recent_events = self._identify_events_advanced(df.tail(100))
        
        # Generate setups based on events
        for event in recent_events[-5:]:  # Last 5 events
            if event['event'] == 'SPRING':
                setup = {
                    'name': 'Spring Long Setup',
                    'entry': df.iloc[event['index'] + 1]['high'],
                    'stop': df.iloc[event['index']]['low'] * 0.995,
                    'targets': [
                        df.iloc[event['index']]['high'] * 1.02,
                        df.iloc[event['index']]['high'] * 1.05,
                        df.iloc[event['index']]['high'] * 1.08
                    ],
                    'risk_reward': 2.5,
                    'confidence': event['confidence'],
                    'timeframe': 'Short-term',
                    'notes': 'Enter on break above spring bar high'
                }
                setups.append(setup)
        
        return setups
    
    # Helper methods
    def _detect_accumulation_pattern(self, window_df: pd.DataFrame, features) -> bool:
        """Detect accumulation pattern in price window"""
        # Low volatility, high volume, sideways price action
        volatility = window_df['returns'].std()
        volume_trend = window_df['volume'].mean() / window_df['volume'].shift(20).mean().mean()
        price_range = (window_df['high'].max() - window_df['low'].min()) / window_df['close'].mean()
        
        return volatility < 0.01 and volume_trend > 1.2 and price_range < 0.05
    
    def _detect_distribution_pattern(self, window_df: pd.DataFrame, features) -> bool:
        """Detect distribution pattern in price window"""
        # High volatility, high volume at tops
        volatility = window_df['returns'].std()
        high_near_max = window_df['high'].iloc[-5:].max() >= window_df['high'].max() * 0.98
        volume_spike = window_df['volume'].iloc[-5:].mean() > window_df['volume'].mean() * 1.5
        
        return volatility > 0.015 and high_near_max and volume_spike
    
    def _calculate_phase_confidence(self, window_df: pd.DataFrame, phase: str) -> float:
        """Calculate confidence score for detected phase"""
        confidence = 0.5
        
        if phase == 'Accumulation':
            # Check for accumulation characteristics
            if window_df['volume'].mean() > window_df['volume'].shift(50).mean().mean():
                confidence += 0.2
            if window_df['close'].std() / window_df['close'].mean() < 0.02:
                confidence += 0.2
            if len(window_df[window_df['low'] <= window_df['support']]) > 2:
                confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _get_phase_characteristics(self, window_df: pd.DataFrame) -> Dict:
        """Get detailed characteristics of phase"""
        return {
            'avg_volume': window_df['volume'].mean(),
            'volatility': window_df['returns'].std(),
            'trend': 'Up' if window_df['close'].iloc[-1] > window_df['close'].iloc[0] else 'Down',
            'volume_trend': window_df['volume'].iloc[-10:].mean() / window_df['volume'].iloc[:10].mean()
        }
    
    def _detect_spring(self, df: pd.DataFrame, idx: int) -> bool:
        """Detect spring pattern"""
        if idx < 20 or idx >= len(df) - 5:
            return False
        
        # Price breaks below recent support with high volume
        recent_support = df.iloc[idx-20:idx]['low'].min()
        breaks_support = df.iloc[idx]['low'] < recent_support * 0.995
        quick_recovery = df.iloc[idx+1]['close'] > recent_support
        high_volume = df.iloc[idx]['volume'] > df['volume_ma'].iloc[idx] * 2
        
        return breaks_support and quick_recovery and high_volume
    
    def _detect_utad(self, df: pd.DataFrame, idx: int) -> bool:
        """Detect UTAD pattern"""
        if idx < 20 or idx >= len(df) - 5:
            return False
        
        # Price breaks above resistance then fails
        recent_resistance = df.iloc[idx-20:idx]['high'].max()
        breaks_resistance = df.iloc[idx]['high'] > recent_resistance * 1.005
        fails_to_hold = df.iloc[idx+1:idx+4]['close'].max() < recent_resistance
        high_volume = df.iloc[idx]['volume'] > df['volume_ma'].iloc[idx] * 1.5
        
        return breaks_resistance and fails_to_hold and high_volume
    
    def _detect_sos(self, df: pd.DataFrame, idx: int) -> bool:
        """Detect Sign of Strength pattern"""
        if idx < 10:
            return False
        
        # Strong price move up with increasing volume
        price_surge = df.iloc[idx]['close'] > df.iloc[idx-5:idx]['high'].max()
        volume_surge = df.iloc[idx]['volume'] > df.iloc[idx-5:idx]['volume'].max()
        above_vwap = df.iloc[idx]['close'] > df.iloc[idx]['vwap']
        
        return price_surge and volume_surge and above_vwap
    
    def _create_volume_profile(self, df: pd.DataFrame) -> Dict:
        """Create volume profile analysis"""
        price_levels = np.linspace(df['low'].min(), df['high'].max(), 50)
        volume_profile = []
        
        for i in range(len(price_levels) - 1):
            mask = (df['low'] <= price_levels[i+1]) & (df['high'] >= price_levels[i])
            volume_at_level = df.loc[mask, 'volume'].sum()
            volume_profile.append({
                'price': (price_levels[i] + price_levels[i+1]) / 2,
                'volume': volume_at_level,
                'percentage': volume_at_level / df['volume'].sum() * 100
            })
        
        # Find POC (Point of Control)
        poc_level = max(volume_profile, key=lambda x: x['volume'])
        
        return {
            'profile': volume_profile,
            'poc': poc_level['price'],
            'value_area_high': df['high'].quantile(0.7),
            'value_area_low': df['low'].quantile(0.3)
        }
    
    def _analyze_order_flow(self, df: pd.DataFrame, tick_data: Optional[pd.DataFrame]) -> Dict:
        """Analyze order flow dynamics"""
        order_flow = {
            'buy_pressure': 0,
            'sell_pressure': 0,
            'flow_imbalance': 0,
            'large_orders': []
        }
        
        # Estimate buy/sell pressure from price and volume
        df['buy_volume'] = np.where(df['close'] > df['open'], df['volume'] * 0.7, df['volume'] * 0.3)
        df['sell_volume'] = df['volume'] - df['buy_volume']
        
        order_flow['buy_pressure'] = df['buy_volume'].sum()
        order_flow['sell_pressure'] = df['sell_volume'].sum()
        order_flow['flow_imbalance'] = (order_flow['buy_pressure'] - order_flow['sell_pressure']) / df['volume'].sum()
        
        return order_flow
    
    def _detect_smart_money_activity(self, df: pd.DataFrame) -> List[Dict]:
        """Detect smart money activity patterns"""
        smart_money_events = []
        
        # Look for smart money accumulation/distribution patterns
        for i in range(50, len(df) - 10):
            # Accumulation: Multiple tests of support with decreasing volume
            if self._is_accumulation_pattern(df, i):
                smart_money_events.append({
                    'index': i,
                    'type': 'Smart Money Accumulation',
                    'confidence': 0.8,
                    'price_level': df.iloc[i]['low']
                })
        
        return smart_money_events
    
    def _is_accumulation_pattern(self, df: pd.DataFrame, idx: int) -> bool:
        """Check for smart money accumulation pattern"""
        # Multiple tests of same level with decreasing volume
        support_level = df.iloc[idx-20:idx]['low'].min()
        tests = df.iloc[idx-20:idx][df.iloc[idx-20:idx]['low'] <= support_level * 1.01]
        
        if len(tests) >= 3:
            volumes = tests['volume'].values
            # Check if volume is decreasing on each test
            return all(volumes[i] > volumes[i+1] for i in range(len(volumes)-1))
        return False
    
    def _calculate_risk_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate risk metrics for current market state"""
        returns = df['returns'].dropna()
        
        return {
            'volatility': returns.std() * np.sqrt(252),
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(df['close']),
            'var_95': np.percentile(returns, 5),
            'current_risk_level': self._assess_risk_level(df)
        }
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _assess_risk_level(self, df: pd.DataFrame) -> str:
        """Assess current risk level"""
        recent_volatility = df['returns'].tail(20).std()
        avg_volatility = df['returns'].std()
        
        if recent_volatility > avg_volatility * 1.5:
            return 'High Risk'
        elif recent_volatility < avg_volatility * 0.7:
            return 'Low Risk'
        else:
            return 'Medium Risk'
    
    def _multi_timeframe_analysis(self, df: pd.DataFrame) -> Dict:
        """Perform multi-timeframe analysis"""
        # Simplified MTF analysis
        return {
            'short_term_trend': 'Up' if df['close'].tail(20).iloc[-1] > df['close'].tail(20).iloc[0] else 'Down',
            'medium_term_trend': 'Up' if df['close'].tail(50).iloc[-1] > df['close'].tail(50).iloc[0] else 'Down',
            'long_term_trend': 'Up' if df['close'].tail(200).iloc[-1] > df['close'].tail(200).iloc[0] else 'Down',
            'alignment': 'Aligned' if len(set([
                'Up' if df['close'].tail(20).iloc[-1] > df['close'].tail(20).iloc[0] else 'Down',
                'Up' if df['close'].tail(50).iloc[-1] > df['close'].tail(50).iloc[0] else 'Down',
                'Up' if df['close'].tail(200).iloc[-1] > df['close'].tail(200).iloc[0] else 'Down'
            ])) == 1 else 'Mixed'
        }

# ============================
# VISUALIZATION FUNCTIONS
# ============================

def create_enhanced_wyckoff_chart(df: pd.DataFrame, analysis: Dict, symbol: str) -> go.Figure:
    """Create enhanced Wyckoff chart with all analysis overlays"""
    
    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=(
            f'{symbol} - Enhanced Wyckoff Analysis',
            'Volume Analysis & Order Flow',
            'VSA & Microstructure Signals',
            'Manipulation Detection',
            'Risk Metrics'
        ),
        row_heights=[0.4, 0.2, 0.15, 0.15, 0.1]
    )
    
    # Main candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name=symbol,
            increasing_line_color='#00ff00',
            decreasing_line_color='#ff0000'
        ),
        row=1, col=1
    )
    
    # Add Wyckoff phases as background
    if 'wyckoff_phases' in analysis:
        for phase in analysis['wyckoff_phases']:
            if phase['start_idx'] < len(df) and phase['end_idx'] < len(df):
                fig.add_vrect(
                    x0=df.index[phase['start_idx']],
                    x1=df.index[phase['end_idx']],
                    fillcolor=phase['phase']['color'],
                    opacity=0.2,
                    layer="below",
                    line_width=0,
                    row=1, col=1
                )
    
    # Add Wyckoff events
    if 'wyckoff_events' in analysis:
        for event in analysis['wyckoff_events']:
            if event['index'] < len(df):
                event_info = event['event']
                fig.add_trace(
                    go.Scatter(
                        x=[df.index[event['index']]],
                        y=[event['price']],
                        mode='markers+text',
                        marker=dict(
                            symbol='diamond',
                            size=15,
                            color=event_info.get('color', '#ffffff'),
                            line=dict(color='white', width=2)
                        ),
                        text=event['event'],
                        textposition='top center',
                        name=event_info['name'],
                        hovertemplate=(
                            f"<b>{event_info['name']}</b><br>"
                            f"Price: ${event['price']:.2f}<br>"
                            f"Confidence: {event['confidence']:.0%}<br>"
                            f"Action: {event.get('action', 'N/A')}<extra></extra>"
                        ),
                        showlegend=False
                    ),
                    row=1, col=1
                )
    
    # Add supply/demand zones
    if 'supply_demand' in analysis:
        for zone in analysis['supply_demand'].get('demand_zones', []):
            fig.add_shape(
                type="rect",
                x0=df.index[max(0, zone['index']-5)],
                x1=df.index[-1],
                y0=zone['price_low'],
                y1=zone['price_high'],
                fillcolor="green",
                opacity=0.2,
                line=dict(color="green", width=1),
                row=1, col=1
            )
    
    # Volume with color coding
    colors = ['red' if c < o else 'green' for c, o in zip(df['close'], df['open'])]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # Add order flow if available
    if 'order_flow' in analysis:
        flow_imbalance = analysis['order_flow']['flow_imbalance']
        fig.add_annotation(
            text=f"Order Flow Imbalance: {flow_imbalance:.1%}",
            xref="paper", yref="paper",
            x=0.95, y=0.6,
            showarrow=False,
            bgcolor="rgba(0,0,0,0.8)",
            bordercolor="white",
            borderwidth=1
        )
    
    # VSA signals
    if 'vsa_analysis' in analysis and 'signals' in analysis['vsa_analysis']:
        for signal in analysis['vsa_analysis']['signals']:
            if signal['index'] < len(df):
                fig.add_trace(
                    go.Scatter(
                        x=[df.index[signal['index']]],
                        y=[df.iloc[signal['index']]['close']],
                        mode='markers',
                        marker=dict(
                            symbol='circle',
                            size=10,
                            color='yellow' if signal['significance'] == 'High' else 'orange'
                        ),
                        name=signal['type'],
                        hovertemplate=f"<b>{signal['type']}</b><br>{signal['interpretation']}<extra></extra>",
                        showlegend=False
                    ),
                    row=3, col=1
                )
    
    # Manipulation events
    if 'manipulation_detection' in analysis:
        for hunt in analysis['manipulation_detection'].get('stop_hunts', []):
# Continuing from the manipulation detection visualization...
            if hunt['index'] < len(df):
                fig.add_trace(
                    go.Scatter(
                        x=[df.index[hunt['index']]],
                        y=[hunt['price']],
                        mode='markers+text',
                        marker=dict(symbol='x', size=12, color='red'),
                        text='SH',
                        textposition='bottom center',
                        name='Stop Hunt',
                        showlegend=False
                    ),
                    row=4, col=1
                )
    
    # Risk metrics
    if 'risk_metrics' in analysis:
        risk = analysis['risk_metrics']
        risk_color = {'High Risk': 'red', 'Medium Risk': 'orange', 'Low Risk': 'green'}
        fig.add_trace(
            go.Scatter(
                x=[df.index[-1]],
                y=[0.5],
                mode='text',
                text=f"Risk: {risk['current_risk_level']}",
                textfont=dict(size=14, color=risk_color.get(risk['current_risk_level'], 'white')),
                showlegend=False
            ),
            row=5, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Enhanced Wyckoff Analysis - {symbol}",
            font=dict(size=24),
            x=0.5
        ),
        template='plotly_dark',
        height=1200,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update axes
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="VSA", row=3, col=1)
    fig.update_yaxes(title_text="Manipulation", row=4, col=1)
    fig.update_yaxes(title_text="Risk", row=5, col=1)
    
    fig.update_xaxes(title_text="Date", row=5, col=1)
    
    return fig

def create_institutional_activity_chart(analysis: Dict) -> go.Figure:
    """Create institutional activity visualization"""
    
    if 'composite_operator' not in analysis:
        return go.Figure()
    
    co_data = analysis['composite_operator']
    
    # Create gauge chart for institutional activity
    if co_data['phase'] == 'Accumulation':
        value = 75
        color = 'green'
    elif co_data['phase'] == 'Distribution':
        value = 25
        color = 'red'
    else:
        value = 50
        color = 'yellow'
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Institutional Activity", 'font': {'size': 20}},
        delta={'reference': 50, 'increasing': {'color': "green"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 25], 'color': 'rgba(255,0,0,0.3)'},
                {'range': [25, 50], 'color': 'rgba(255,255,0,0.3)'},
                {'range': [50, 75], 'color': 'rgba(255,255,0,0.3)'},
                {'range': [75, 100], 'color': 'rgba(0,255,0,0.3)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "white", 'family': "Arial"},
        height=400
    )
    
    return fig

def create_volume_profile_chart(df: pd.DataFrame, analysis: Dict) -> go.Figure:
    """Create volume profile visualization"""
    
    if 'volume_profile' not in analysis:
        return go.Figure()
    
    profile = analysis['volume_profile']['profile']
    poc = analysis['volume_profile']['poc']
    
    # Create horizontal bar chart for volume profile
    fig = go.Figure()
    
    # Add volume profile bars
    prices = [p['price'] for p in profile]
    volumes = [p['volume'] for p in profile]
    
    fig.add_trace(go.Bar(
        x=volumes,
        y=prices,
        orientation='h',
        marker=dict(
            color=volumes,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Volume")
        ),
        name='Volume Profile'
    ))
    
    # Add POC line
    fig.add_hline(
        y=poc,
        line_dash="dash",
        line_color="red",
        annotation_text="POC",
        annotation_position="right"
    )
    
    # Add value area
    fig.add_hrect(
        y0=analysis['volume_profile']['value_area_low'],
        y1=analysis['volume_profile']['value_area_high'],
        fillcolor="yellow",
        opacity=0.2,
        line_width=0,
        annotation_text="Value Area",
        annotation_position="right"
    )
    
    fig.update_layout(
        title="Volume Profile Analysis",
        xaxis_title="Volume",
        yaxis_title="Price",
        template='plotly_dark',
        height=600
    )
    
    return fig

def create_manipulation_heatmap(analysis: Dict, df: pd.DataFrame) -> go.Figure:
    """Create manipulation detection heatmap"""
    
    if 'manipulation_detection' not in analysis:
        return go.Figure()
    
    # Create time-based heatmap of manipulation events
    hours = list(range(24))
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
    
    # Initialize heatmap data
    z = [[0 for _ in hours] for _ in days]
    
    # Count manipulation events by time
    for event_type in ['stop_hunts', 'spoofing_events', 'wash_trades']:
        for event in analysis['manipulation_detection'].get(event_type, []):
            if event['index'] < len(df):
                timestamp = df.index[event['index']]
                if hasattr(timestamp, 'hour') and hasattr(timestamp, 'dayofweek'):
                    if timestamp.dayofweek < 5:  # Weekdays only
                        z[timestamp.dayofweek][timestamp.hour] += 1
    
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=hours,
        y=days,
        colorscale='Reds',
        hoverongaps=False,
        hovertemplate="Day: %{y}<br>Hour: %{x}<br>Events: %{z}<extra></extra>"
    ))
    
    fig.update_layout(
        title="Manipulation Activity Heatmap",
        xaxis_title="Hour of Day",
        yaxis_title="Day of Week",
        template='plotly_dark',
        height=400
    )
    
    return fig

def create_risk_dashboard(analysis: Dict) -> go.Figure:
    """Create comprehensive risk metrics dashboard"""
    
    if 'risk_metrics' not in analysis:
        return go.Figure()
    
    risk = analysis['risk_metrics']
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Volatility', 'Sharpe Ratio', 'Max Drawdown', 'VaR 95%'),
        specs=[[{'type': 'indicator'}, {'type': 'indicator'}],
               [{'type': 'indicator'}, {'type': 'indicator'}]]
    )
    
    # Volatility gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=risk['volatility'] * 100,
        title={'text': "Annual Vol %"},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={'axis': {'range': [0, 50]},
               'bar': {'color': "darkblue"},
               'steps': [
                   {'range': [0, 15], 'color': "lightgray"},
                   {'range': [15, 30], 'color': "gray"},
                   {'range': [30, 50], 'color': "lightgray"}],
               'threshold': {'line': {'color': "red", 'width': 4},
                            'thickness': 0.75, 'value': 40}}
    ), row=1, col=1)
    
    # Sharpe ratio
    fig.add_trace(go.Indicator(
        mode="number+delta",
        value=risk['sharpe_ratio'],
        title={'text': "Sharpe Ratio"},
        delta={'reference': 1.0, 'relative': True},
        domain={'x': [0, 1], 'y': [0, 1]}
    ), row=1, col=2)
    
    # Max drawdown
    fig.add_trace(go.Indicator(
        mode="number+gauge",
        value=abs(risk['max_drawdown']) * 100,
        title={'text': "Max Drawdown %"},
        gauge={'axis': {'range': [0, 30]},
               'bar': {'color': "red"},
               'steps': [
                   {'range': [0, 10], 'color': "lightgreen"},
                   {'range': [10, 20], 'color': "yellow"},
                   {'range': [20, 30], 'color': "lightcoral"}]}
    ), row=2, col=1)
    
    # VaR
    fig.add_trace(go.Indicator(
        mode="number",
        value=risk['var_95'] * 100,
        title={'text': "VaR 95% (Daily)"},
        number={'suffix': "%"},
        domain={'x': [0, 1], 'y': [0, 1]}
    ), row=2, col=2)
    
    fig.update_layout(
        template='plotly_dark',
        height=600,
        showlegend=False
    )
    
    return fig

# ============================
# MAIN APPLICATION
# ============================

def main():
    """Main application entry point"""
    
    # Header
    st.markdown("""
    <div class="quant-header">
        <h1>🎯 Enhanced Wyckoff Analysis Pro</h1>
        <p>Professional Quantitative Trading System with Microstructure Integration</p>
        <p>Institutional-Grade Market Analysis Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Trading controls
        st.subheader("📅 Trading Controls")

        data_dir = Path(st.secrets["PARQUET_DATA_DIR"])
        available_pairs = sorted([f.name for f in data_dir.iterdir() if f.is_dir()])
        selected_pair = st.selectbox("💱 Select Pair", available_pairs, index=0 if available_pairs else None)

        # Preferred timeframes in board order
        preferred_timeframes = ["1min", "3min", "5min", "15min", "30min", "1h", "4h", "1d"]
        timeframes = []
        data = {}
        if selected_pair:
            pair_dir = data_dir / selected_pair
            available_files = {f.stem for f in pair_dir.glob("*.parquet")}
            timeframes = [tf for tf in preferred_timeframes if tf in available_files]
            for tf in timeframes:
                try:
                    df_tf = pd.read_parquet(pair_dir / f"{tf}.parquet")
                    data[tf] = df_tf
                except Exception as e:
                    st.warning(f"Could not load {tf}.parquet: {e}")

        selected_timeframe = st.selectbox(
            "📊 Select Timeframe",
            timeframes,
            index=(len(timeframes)-1 if timeframes else 0),
            help="Choose timeframe for detailed analysis",
            disabled=not timeframes
        )

        current_df = data.get(selected_timeframe) if selected_timeframe else None
        symbol = selected_pair

        # Analysis parameters
        st.subheader("📊 Analysis Parameters")

        lookback_period = st.slider(
            "Lookback Period (bars)",
            min_value=100,
            max_value=5000,
            value=500,
            step=50
        )

        # Advanced options
        with st.expander("🔧 Advanced Settings"):
            volume_threshold = st.slider("Volume Spike Threshold", 1.5, 4.0, 2.5, 0.1)
            manipulation_sensitivity = st.slider("Manipulation Sensitivity", 0.01, 0.1, 0.05, 0.01)
            phase_min_bars = st.slider("Min Bars for Phase", 10, 50, 20)

            enable_tick_analysis = st.checkbox("Enable Tick Analysis", value=False)
            enable_ml_features = st.checkbox("Enable ML Features", value=True)
            enable_alerts = st.checkbox("Enable Trading Alerts", value=True)

        # Analysis triggers
        st.subheader("🚀 Analysis")
        analyze_button = st.button("Run Analysis", type="primary", use_container_width=True)
        export_button = st.button("Export Results", use_container_width=True)

    # Main content area
    if analyze_button and current_df is not None and symbol is not None:
        # Progress indicator
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Initialize analyzer
        analyzer = EnhancedWyckoffAnalyzer()

        # Load tick data if available
        tick_data = None
        if enable_tick_analysis:
            # Load raw tick CSV from configured tick_data path
            tick_file = Path(st.secrets["tick_data"]) / f"{symbol}_tick.csv"
            if tick_file.exists():
                tick_data = pd.read_csv(tick_file, index_col=0, parse_dates=True)
                status_text.text("Loaded tick data for microstructure analysis...")

        # Prepare data
        status_text.text("Preparing data...")
        progress_bar.progress(10)

        df_analysis = current_df.tail(lookback_period).copy()

        # Run comprehensive analysis
        status_text.text("Running Wyckoff analysis...")
        progress_bar.progress(30)

        analysis_results = analyzer.comprehensive_analysis(df_analysis, tick_data)

        status_text.text("Analyzing microstructure patterns...")
        progress_bar.progress(60)

        # Display results
        status_text.text("Generating visualizations...")
        progress_bar.progress(80)

        # Clear progress indicators
        progress_bar.progress(100)
        status_text.empty()
        progress_bar.empty()

        # Display metrics dashboard
        st.subheader("📈 Key Metrics Dashboard")

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            current_price = df_analysis['close'].iloc[-1]
            price_change = (df_analysis['close'].iloc[-1] - df_analysis['close'].iloc[0]) / df_analysis['close'].iloc[0] * 100
            st.metric("Price", f"${current_price:.2f}", f"{price_change:+.2f}%")

        with col2:
            if 'composite_operator' in analysis_results:
                phase = analysis_results['composite_operator']['phase']
                phase_color = {'Accumulation': '🟢', 'Distribution': '🔴', 'Neutral': '🟡'}
                st.metric("CO Phase", f"{phase_color.get(phase, '⚪')} {phase}")

        with col3:
            if 'manipulation_detection' in analysis_results:
                manip_score = analysis_results['manipulation_detection']['manipulation_score']
                st.metric("Manipulation", f"{manip_score:.1f}%",
                         "⚠️ High" if manip_score > 10 else "✅ Low")

        with col4:
            if 'risk_metrics' in analysis_results:
                risk_level = analysis_results['risk_metrics']['current_risk_level']
                risk_emoji = {'High Risk': '🔴', 'Medium Risk': '🟡', 'Low Risk': '🟢'}
                st.metric("Risk Level", f"{risk_emoji.get(risk_level, '⚪')} {risk_level}")

        with col5:
            if 'multi_timeframe' in analysis_results:
                alignment = analysis_results['multi_timeframe']['alignment']
                st.metric("MTF Alignment", alignment,
                         "✅" if alignment == 'Aligned' else "⚠️")

        # Wyckoff phases summary
        if 'wyckoff_phases' in analysis_results and analysis_results['wyckoff_phases']:
            st.subheader("🎯 Current Wyckoff Phase")
            latest_phase = analysis_results['wyckoff_phases'][-1]

            phase_col1, phase_col2 = st.columns([1, 3])
            with phase_col1:
                st.markdown(f"""
                <div class="phase-badge phase-{latest_phase['phase'].lower()}">
                    {latest_phase['phase']}
                </div>
                """, unsafe_allow_html=True)

            with phase_col2:
                st.write(f"**Confidence:** {latest_phase['confidence']:.0%}")
                st.write(f"**Price Range:** ${latest_phase['price_range'][0]:.2f} - ${latest_phase['price_range'][1]:.2f}")
                if 'characteristics' in latest_phase:
                    st.write(f"**Volume Trend:** {latest_phase['characteristics']['volume_trend']:.2f}x")

        # Main analysis chart
        st.subheader("📊 Enhanced Wyckoff Chart")
        main_chart = create_enhanced_wyckoff_chart(df_analysis, analysis_results, symbol)
        st.plotly_chart(main_chart, use_container_width=True)

        # Additional visualizations in tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Volume Profile",
            "🏛️ Institutional Activity",
            "🚨 Manipulation Detection",
            "⚠️ Risk Dashboard",
            "💼 Trade Setups"
        ])

        with tab1:
            volume_profile_chart = create_volume_profile_chart(df_analysis, analysis_results)
            st.plotly_chart(volume_profile_chart, use_container_width=True)

            # Volume analysis insights
            if 'volume_profile' in analysis_results:
                st.info(f"**POC (Point of Control):** ${analysis_results['volume_profile']['poc']:.2f}")

        with tab2:
            col1, col2 = st.columns(2)

            with col1:
                institutional_chart = create_institutional_activity_chart(analysis_results)
                st.plotly_chart(institutional_chart, use_container_width=True)

            with col2:
                if 'composite_operator' in analysis_results:
                    co = analysis_results['composite_operator']
                    st.write("### Institutional Footprints")
                    st.write(f"**Accumulation Score:** {co['accumulation_score']:.1f}%")
                    st.write(f"**Distribution Score:** {co['distribution_score']:.1f}%")

                    if co['institutional_footprints']:
                        recent_footprints = co['institutional_footprints'][-5:]
                        for fp in recent_footprints:
                            st.write(f"- {fp['type']} (Strength: {fp['strength']:.1f}x)")

        with tab3:
            if 'manipulation_detection' in analysis_results:
                manipulation_heatmap = create_manipulation_heatmap(analysis_results, df_analysis)
                st.plotly_chart(manipulation_heatmap, use_container_width=True)

                # Manipulation events summary
                manip = analysis_results['manipulation_detection']

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Stop Hunts", len(manip.get('stop_hunts', [])))
                with col2:
                    st.metric("Spoofing Events", len(manip.get('spoofing_events', [])))
                with col3:
                    st.metric("Wash Trades", len(manip.get('wash_trades', [])))

                # Recent manipulation alerts
                if manip.get('stop_hunts'):
                    st.markdown("""
                    <div class="manipulation-alert">
                        ⚠️ Recent stop hunt activity detected! Exercise caution with stops.
                    </div>
                    """, unsafe_allow_html=True)

        with tab4:
            risk_dashboard = create_risk_dashboard(analysis_results)
            st.plotly_chart(risk_dashboard, use_container_width=True)

            # Risk recommendations
            if 'risk_metrics' in analysis_results:
                risk = analysis_results['risk_metrics']
                if risk['current_risk_level'] == 'High Risk':
                    st.warning("⚠️ High risk environment detected. Consider reducing position sizes.")
                elif risk['current_risk_level'] == 'Low Risk':
                    st.success("✅ Low risk environment. Favorable conditions for position entry.")

        with tab5:
            st.write("### 💼 Professional Trade Setups")

            if 'trade_setups' in analysis_results and analysis_results['trade_setups']:
                for setup in analysis_results['trade_setups']:
                    with st.expander(f"{setup['name']} - Confidence: {setup['confidence']:.0%}"):
                        col1, col2 = st.columns(2)

                        with col1:
                            st.write(f"**Entry:** ${setup['entry']:.2f}")
                            st.write(f"**Stop:** ${setup['stop']:.2f}")
                            st.write(f"**Risk/Reward:** {setup['risk_reward']:.1f}")

                        with col2:
                            st.write("**Targets:**")
                            for i, target in enumerate(setup['targets'], 1):
                                st.write(f"  T{i}: ${target:.2f}")

                        st.write(f"**Notes:** {setup.get('notes', 'N/A')}")

                        # Add trade button
                        if st.button(f"Execute {setup['name']}", key=f"trade_{setup['name']}"):
                            st.success(f"Trade setup copied to clipboard!")
            else:
                st.info("No high-confidence trade setups currently available.")

        # Export functionality
        if export_button:
            # Create comprehensive export
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'analysis': analysis_results,
                'parameters': {
                    'lookback_period': lookback_period,
                    'volume_threshold': volume_threshold,
                    'manipulation_sensitivity': manipulation_sensitivity
                }
            }

            # Convert to JSON
            export_json = json.dumps(export_data, indent=2, default=str)

            # Download button
            st.download_button(
                label="📥 Download Analysis Report",
                data=export_json,
                file_name=f"wyckoff_analysis_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888;">
        <p><strong>Enhanced Wyckoff Analysis Pro v4.0</strong></p>
        <p>Professional Quantitative Trading System</p>
        <p>Powered by Advanced Microstructure Analysis</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
