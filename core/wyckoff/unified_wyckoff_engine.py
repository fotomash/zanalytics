# zanalytics/core/wyckoff/unified_wyckoff_engine.py
#!/usr/bin/env python3
"""
Unified Wyckoff Analysis Engine
Consolidates all Wyckoff logic into a single, comprehensive analyzer
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass
from enum import Enum

try:
    from .state_machine import WyckoffStateMachine
except ModuleNotFoundError:
    # Stub state machine if the real implementation is missing
    class WyckoffStateMachine:
        def __init__(self, *args, **kwargs):
            pass
        def process_event(self, event):
            pass
        @property
        def current_phase(self):
            from enum import Enum
            return WyckoffPhase.TRANSITION
        @property
        def phase_confidence(self):
            return 0.0
        @property
        def phase_duration(self):
            return 0
        @property
        def phase_history(self):
            return []
from .event_detector import WyckoffEventDetector
try:
    from .vsa_signals_mentfx import VSAAnalyzer
except (ModuleNotFoundError, ImportError):
    # Stub VSAAnalyzer if the real module is missing or missing attribute
    class VSAAnalyzer:
        def __init__(self, *args, **kwargs):
            pass
        def analyze(self, df):
            # Return empty analysis
            return {}
try:
    from ..base_analyzer import BaseAnalyzer
except ModuleNotFoundError:
    # Stub BaseAnalyzer if core/base_analyzer.py is not present
    class BaseAnalyzer:
        def __init__(self, config=None):
            # Dummy initializer
            self.config = config or {}

logger = logging.getLogger(__name__)

class WyckoffPhase(Enum):
    ACCUMULATION = "accumulation"
    MARKUP = "markup" 
    DISTRIBUTION = "distribution"
    MARKDOWN = "markdown"
    TRANSITION = "transition"

@dataclass
class WyckoffEvent:
    event_type: str
    timestamp: datetime
    price: float
    volume: float
    significance: str
    description: str
    confidence: float

@dataclass
class WyckoffAnalysis:
    symbol: str
    timeframe: str
    current_phase: WyckoffPhase
    phase_confidence: float
    phase_duration: int
    events: List[WyckoffEvent]
    vsa_signals: Dict
    supply_demand_zones: Dict
    composite_operator_score: float
    trend_structure: Dict
    trade_setups: List[Dict]
    timestamp: datetime

class UnifiedWyckoffEngine(BaseAnalyzer):
    """
    Unified Wyckoff Analysis Engine
    Consolidates all Wyckoff methodologies into one comprehensive analyzer
    """
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.config = config or {}
        
        # Initialize components
        self.state_machine = WyckoffStateMachine()
        self.event_detector = WyckoffEventDetector(
            volume_threshold=self.config.get('volume_threshold', 1.5),
            price_threshold=self.config.get('price_threshold', 0.02)
        )
        self.vsa_analyzer = VSAAnalyzer()
        
        # Analysis parameters
        self.lookback_periods = self.config.get('lookback_periods', 100)
        self.phase_sensitivity = self.config.get('phase_sensitivity', 1.0)
        self.min_phase_duration = self.config.get('min_phase_duration', 10)
        
        logger.info("Unified Wyckoff Engine initialized")
    
    def analyze(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> WyckoffAnalysis:
        """
        Comprehensive Wyckoff analysis
        
        Args:
            df: OHLCV DataFrame with columns: timestamp, open, high, low, close, volume
            symbol: Trading symbol
            
        Returns:
            WyckoffAnalysis object with complete analysis
        """
        try:
            # Validate and prepare data
            df = self._prepare_data(df)
            
            if len(df) < self.min_phase_duration:
                logger.warning(f"Insufficient data for analysis: {len(df)} bars")
                return self._empty_analysis(symbol)
            
            # Core analysis components
            events = self._detect_events(df)
            phase_analysis = self._analyze_phases(df, events)
            vsa_analysis = self._analyze_volume_spread(df)
            structure_analysis = self._analyze_market_structure(df)
            zones = self._identify_supply_demand_zones(df)
            co_score = self._calculate_composite_operator_score(df, events)
            setups = self._generate_trade_setups(df, phase_analysis, events)
            
            # Compile results
            analysis = WyckoffAnalysis(
                symbol=symbol,
                timeframe=self._detect_timeframe(df),
                current_phase=phase_analysis['current_phase'],
                phase_confidence=phase_analysis['confidence'],
                phase_duration=phase_analysis['duration'],
                events=events,
                vsa_signals=vsa_analysis,
                supply_demand_zones=zones,
                composite_operator_score=co_score,
                trend_structure=structure_analysis,
                trade_setups=setups,
                timestamp=datetime.now()
            )
            
            logger.info(f"Wyckoff analysis completed for {symbol}: {analysis.current_phase.value}")
            return analysis
            
        except Exception as e:
            logger.error(f"Wyckoff analysis failed for {symbol}: {e}")
            return self._empty_analysis(symbol)
    
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize DataFrame format"""
        # Ensure required columns exist
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        # Handle different timestamp column names
        if 'datetime' in df.columns and 'timestamp' not in df.columns:
            df = df.rename(columns={'datetime': 'timestamp'})
        
        # Validate columns
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Ensure proper data types
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].astype(float)
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Add derived columns
        df['price_range'] = df['high'] - df['low']
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        df['close_position'] = (df['close'] - df['low']) / df['price_range']
        
        # Volume metrics
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        df['volume_spike'] = df['volume_ratio'] > self.config.get('volume_threshold', 1.5)
        
        # Price metrics
        df['price_change'] = df['close'].pct_change()
        df['price_volatility'] = df['close'].rolling(20).std()
        
        return df
    
    def _detect_events(self, df: pd.DataFrame) -> List[WyckoffEvent]:
        """Detect Wyckoff events using the event detector"""
        return self.event_detector.detect_events(df)
    
    def _analyze_phases(self, df: pd.DataFrame, events: List[WyckoffEvent]) -> Dict:
        """Analyze Wyckoff phases using state machine"""
        
        # Process events through state machine
        for event in events:
            self.state_machine.process_event(event)
        
        # Determine current phase
        current_phase = self.state_machine.current_phase
        confidence = self.state_machine.phase_confidence
        duration = self.state_machine.phase_duration
        
        # Additional phase validation using price/volume patterns
        phase_validation = self._validate_phase_with_patterns(df, current_phase)
        
        return {
            'current_phase': current_phase,
            'confidence': min(confidence, phase_validation['confidence']),
            'duration': duration,
            'validation': phase_validation,
            'phase_history': self.state_machine.phase_history
        }
    
    def _validate_phase_with_patterns(self, df: pd.DataFrame, phase: WyckoffPhase) -> Dict:
        """Validate phase using price/volume patterns"""
        recent_data = df.tail(20)
        
        if phase == WyckoffPhase.ACCUMULATION:
            # Low volatility, high volume, sideways price action
            volatility_score = 1.0 - (recent_data['price_volatility'].mean() / df['price_volatility'].quantile(0.8))
            volume_score = recent_data['volume_ratio'].mean() / 2.0
            sideways_score = 1.0 - abs(recent_data['price_change'].mean()) * 100
            
            confidence = np.mean([volatility_score, volume_score, sideways_score])
            
        elif phase == WyckoffPhase.MARKUP:
            # Rising prices with volume support
            price_trend = (recent_data['close'].iloc[-1] / recent_data['close'].iloc[0]) - 1
            volume_support = recent_data['volume_ratio'].mean()
            
            confidence = min(price_trend * 10, volume_support / 2.0)
            
        elif phase == WyckoffPhase.DISTRIBUTION:
            # High volatility, high volume, topping action
            volatility_score = recent_data['price_volatility'].mean() / df['price_volatility'].quantile(0.8)
            volume_score = recent_data['volume_ratio'].mean() / 2.0
            topping_score = (recent_data['high'].max() == df['high'].tail(50).max())
            
            confidence = np.mean([volatility_score, volume_score, float(topping_score)])
            
        elif phase == WyckoffPhase.MARKDOWN:
            # Declining prices with volume
            price_trend = 1.0 - ((recent_data['close'].iloc[-1] / recent_data['close'].iloc[0]) - 1)
            volume_support = recent_data['volume_ratio'].mean()
            
            confidence = min(price_trend * 10, volume_support / 2.0)
            
        else:
            confidence = 0.5
        
        return {
            'confidence': max(0.0, min(1.0, confidence)),
            'supporting_evidence': self._get_supporting_evidence(recent_data, phase)
        }
    
    def _analyze_volume_spread(self, df: pd.DataFrame) -> Dict:
        """Comprehensive Volume Spread Analysis"""
        return self.vsa_analyzer.analyze(df)
    
    def _analyze_market_structure(self, df: pd.DataFrame) -> Dict:
        """Analyze market structure and trend"""
        
        # Find swing highs and lows
        swing_highs = self._find_swing_points(df, 'high', window=5)
        swing_lows = self._find_swing_points(df, 'low', window=5)
        
        # Determine trend
        trend = self._determine_trend(swing_highs, swing_lows)
        
        # Find structure breaks
        structure_breaks = self._find_structure_breaks(swing_highs, swing_lows)
        
        return {
            'trend': trend,
            'swing_highs': swing_highs[-10:],  # Last 10
            'swing_lows': swing_lows[-10:],
            'structure_breaks': structure_breaks,
            'trend_strength': self._calculate_trend_strength(df)
        }
    
    def _identify_supply_demand_zones(self, df: pd.DataFrame) -> Dict:
        """Identify key supply and demand zones"""
        
        supply_zones = []
        demand_zones = []
        
        # Find significant highs and lows with volume confirmation
        for i in range(10, len(df) - 10):
            window = df.iloc[i-10:i+10]
            
            # Supply zone at significant high
            if (df.iloc[i]['high'] == window['high'].max() and
                df.iloc[i]['volume'] > df.iloc[i]['volume_ma'] * 1.5):
                
                supply_zones.append({
                    'price': df.iloc[i]['high'],
                    'timestamp': df.iloc[i]['timestamp'],
                    'strength': df.iloc[i]['volume_ratio'],
                    'type': 'supply'
                })
            
            # Demand zone at significant low
            if (df.iloc[i]['low'] == window['low'].min() and
                df.iloc[i]['volume'] > df.iloc[i]['volume_ma'] * 1.5):
                
                demand_zones.append({
                    'price': df.iloc[i]['low'],
                    'timestamp': df.iloc[i]['timestamp'],
                    'strength': df.iloc[i]['volume_ratio'],
                    'type': 'demand'
                })
        
        return {
            'supply_zones': supply_zones[-5:],  # Last 5
            'demand_zones': demand_zones[-5:],
            'current_bias': self._determine_current_bias(df)
        }
    
    def _calculate_composite_operator_score(self, df: pd.DataFrame, events: List[WyckoffEvent]) -> float:
        """Calculate Composite Operator activity score"""
        
        # Analyze institutional footprints
        accumulation_signals = 0
        distribution_signals = 0
        
        for i in range(50, len(df)):
            # High volume, low price movement = potential accumulation
            if (df.iloc[i]['volume_ratio'] > 1.5 and
                abs(df.iloc[i]['price_change']) < 0.01):
                accumulation_signals += 1
            
            # High volume at extremes = potential distribution
            if (df.iloc[i]['volume_ratio'] > 1.5 and
                (df.iloc[i]['high'] == df['high'].iloc[i-10:i+1].max() or
                 df.iloc[i]['low'] == df['low'].iloc[i-10:i+1].min())):
                distribution_signals += 1
        
        # Score from -1 (distributing) to +1 (accumulating)
        if accumulation_signals + distribution_signals == 0:
            return 0.0
        
        score = (accumulation_signals - distribution_signals) / (accumulation_signals + distribution_signals)
        return score
    
    def _generate_trade_setups(self, df: pd.DataFrame, phase_analysis: Dict, events: List[WyckoffEvent]) -> List[Dict]:
        """Generate trade setups based on Wyckoff analysis"""
        
        setups = []
        current_phase = phase_analysis['current_phase']
        current_price = df['close'].iloc[-1]
        
        if current_phase == WyckoffPhase.ACCUMULATION:
            # Look for spring or test setups
            recent_lows = df['low'].tail(20)
            if len(recent_lows) > 0:
                support_level = recent_lows.min()
                
                setups.append({
                    'name': 'Accumulation Spring Setup',
                    'type': 'long',
                    'entry': support_level * 1.002,  # Slightly above support
                    'stop': support_level * 0.995,   # Below support
                    'targets': [
                        support_level * 1.01,
                        support_level * 1.02,
                        support_level * 1.035
                    ],
                    'risk_reward': 2.5,
                    'confidence': phase_analysis['confidence'],
                    'phase': current_phase.value,
                    'description': 'Long setup on successful test of support in accumulation'
                })
        
        elif current_phase == WyckoffPhase.MARKUP:
            # Look for continuation setups
            recent_highs = df['high'].tail(20)
            if len(recent_highs) > 0:
                resistance_level = recent_highs.max()
                
                setups.append({
                    'name': 'Markup Continuation',
                    'type': 'long',
                    'entry': resistance_level * 1.001,
                    'stop': current_price * 0.98,
                    'targets': [
                        resistance_level * 1.015,
                        resistance_level * 1.03,
                        resistance_level * 1.05
                    ],
                    'risk_reward': 1.8,
                    'confidence': phase_analysis['confidence'],
                    'phase': current_phase.value,
                    'description': 'Breakout continuation in markup phase'
                })
        
        elif current_phase == WyckoffPhase.DISTRIBUTION:
            # Look for distribution setups
            recent_highs = df['high'].tail(20)
            if len(recent_highs) > 0:
                resistance_level = recent_highs.max()
                
                setups.append({
                    'name': 'Distribution Short Setup',
                    'type': 'short',
                    'entry': resistance_level * 0.998,
                    'stop': resistance_level * 1.005,
                    'targets': [
                        resistance_level * 0.985,
                        resistance_level * 0.97,
                        resistance_level * 0.95
                    ],
                    'risk_reward': 2.2,
                    'confidence': phase_analysis['confidence'],
                    'phase': current_phase.value,
                    'description': 'Short setup on failed breakout in distribution'
                })
        
        return setups
    
    # Helper methods
    def _find_swing_points(self, df: pd.DataFrame, column: str, window: int = 5) -> List[Dict]:
        """Find swing highs or lows"""
        swings = []
        
        for i in range(window, len(df) - window):
            if column == 'high':
                if df[column].iloc[i] == df[column].iloc[i-window:i+window].max():
                    swings.append({
                        'index': i,
                        'timestamp': df.iloc[i]['timestamp'],
                        'price': df.iloc[i][column],
                        'type': 'swing_high'
                    })
            else:  # low
                if df[column].iloc[i] == df[column].iloc[i-window:i+window].min():
                    swings.append({
                        'index': i,
                        'timestamp': df.iloc[i]['timestamp'],
                        'price': df.iloc[i][column],
                        'type': 'swing_low'
                    })
        
        return swings
    
    def _determine_trend(self, swing_highs: List, swing_lows: List) -> str:
        """Determine overall trend from swing points"""
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return 'insufficient_data'
        
        recent_highs = swing_highs[-2:]
        recent_lows = swing_lows[-2:]
        
        higher_highs = recent_highs[1]['price'] > recent_highs[0]['price']
        higher_lows = recent_lows[1]['price'] > recent_lows[0]['price']
        
        if higher_highs and higher_lows:
            return 'uptrend'
        elif not higher_highs and not higher_lows:
            return 'downtrend'
        else:
            return 'sideways'
    
    def _find_structure_breaks(self, swing_highs: List, swing_lows: List) -> List[Dict]:
        """Find market structure breaks"""
        breaks = []
        # Implementation for structure break detection
        return breaks
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """Calculate trend strength (0-1)"""
        if len(df) < 20:
            return 0.5
        
        price_momentum = (df['close'].iloc[-1] / df['close'].iloc[-20]) - 1
        volume_confirmation = df['volume_ratio'].tail(10).mean()
        
        strength = min(abs(price_momentum) * 5, volume_confirmation / 2)
        return max(0.0, min(1.0, strength))
    
    def _determine_current_bias(self, df: pd.DataFrame) -> str:
        """Determine current market bias"""
        recent_data = df.tail(20)
        
        volume_trend = recent_data['volume_ratio'].mean()
        price_trend = (recent_data['close'].iloc[-1] / recent_data['close'].iloc[0]) - 1
        
        if price_trend > 0.02 and volume_trend > 1.1:
            return 'bullish_with_volume'
        elif price_trend < -0.02 and volume_trend > 1.1:
            return 'bearish_with_volume'
        elif volume_trend < 0.9:
            return 'low_volume_caution'
        else:
            return 'neutral'
    
    def _detect_timeframe(self, df: pd.DataFrame) -> str:
        """Auto-detect timeframe from data"""
        if len(df) < 2:
            return 'unknown'
        
        time_diff = df['timestamp'].iloc[1] - df['timestamp'].iloc[0]
        minutes = time_diff.total_seconds() / 60
        
        if minutes <= 1:
            return '1m'
        elif minutes <= 5:
            return '5m'
        elif minutes <= 15:
            return '15m'
        elif minutes <= 60:
            return '1h'
        elif minutes <= 240:
            return '4h'
        else:
            return '1d'
    
    def _get_supporting_evidence(self, df: pd.DataFrame, phase: WyckoffPhase) -> List[str]:
        """Get supporting evidence for phase classification"""
        evidence = []
        
        if phase == WyckoffPhase.ACCUMULATION:
            if df['volume_ratio'].mean() > 1.2:
                evidence.append("Above average volume")
            if df['price_volatility'].mean() < df['price_volatility'].quantile(0.3):
                evidence.append("Low volatility")
        
        # Add more evidence logic for other phases
        
        return evidence
    
    def _empty_analysis(self, symbol: str) -> WyckoffAnalysis:
        """Return empty analysis for error cases"""
        return WyckoffAnalysis(
            symbol=symbol,
            timeframe='unknown',
            current_phase=WyckoffPhase.TRANSITION,
            phase_confidence=0.0,
            phase_duration=0,
            events=[],
            vsa_signals={},
            supply_demand_zones={'supply_zones': [], 'demand_zones': [], 'current_bias': 'neutral'},
            composite_operator_score=0.0,
            trend_structure={'trend': 'unknown'},
            trade_setups=[],
            timestamp=datetime.now()
        )

# Export the unified engine
__all__ = ['UnifiedWyckoffEngine', 'WyckoffAnalysis', 'WyckoffEvent', 'WyckoffPhase']