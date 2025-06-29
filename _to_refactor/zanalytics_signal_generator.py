# Stage 3: Trading Signal Generator
# zanalytics_signal_generator.py

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from collections import defaultdict
import statistics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('zanalytics_signals.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SignalType(Enum):
    """Types of trading signals"""
    BUY = "buy"
    SELL = "sell"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"
    NEUTRAL = "neutral"
    EXIT_LONG = "exit_long"
    EXIT_SHORT = "exit_short"

class SignalPriority(Enum):
    """Signal priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class TradingSignal:
    """Comprehensive trading signal with all necessary information"""
    timestamp: datetime
    symbol: str
    timeframe: str
    signal_type: SignalType
    priority: SignalPriority
    entry_price: float
    stop_loss: float
    take_profit_targets: List[float]
    confidence: float
    risk_reward_ratio: float
    position_size_suggestion: float = 0.0
    reasoning: List[str] = field(default_factory=list)
    supporting_indicators: Dict[str, Any] = field(default_factory=dict)
    conflicting_indicators: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    expiry: Optional[datetime] = None

@dataclass
class MarketContext:
    """Current market context for signal generation"""
    trend: str  # bullish, bearish, sideways
    volatility: float
    volume_profile: str  # increasing, decreasing, stable
    market_regime: str  # trending, ranging, volatile
    key_levels: Dict[str, List[float]]
    recent_signals: List[TradingSignal]

class SignalGenerator:
    """Advanced trading signal generator with multi-timeframe analysis"""

    def __init__(self, config_path: str = "signal_config.json"):
        self.config = self._load_config(config_path)
        self.signals_dir = Path(self.config.get("signals_dir", "./signals"))
        self.signals_dir.mkdir(parents=True, exist_ok=True)

        # Signal history for pattern recognition
        self.signal_history: Dict[str, List[TradingSignal]] = defaultdict(list)

        # Risk management parameters
        self.risk_per_trade = self.config.get("risk_per_trade", 0.02)
        self.max_open_positions = self.config.get("max_open_positions", 3)
        self.max_correlation = self.config.get("max_correlation", 0.7)

    def _load_config(self, config_path: str) -> Dict:
        """Load signal configuration"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using defaults.")
            return self._get_default_config()

    def _get_default_config(self) -> Dict:
        """Default signal configuration"""
        return {
            "signals_dir": "./signals",
            "risk_per_trade": 0.02,
            "max_open_positions": 3,
            "max_correlation": 0.7,
            "signal_filters": {
                "min_confidence": 0.6,
                "min_risk_reward": 1.5,
                "max_drawdown": 0.05,
                "required_confirmations": 2
            },
            "timeframe_weights": {
                "1m": 0.1,
                "5m": 0.15,
                "15m": 0.2,
                "1h": 0.25,
                "4h": 0.2,
                "1d": 0.1
            },
            "position_sizing": {
                "method": "kelly_criterion",
                "max_position_size": 0.1,
                "scale_with_confidence": True
            },
            "stop_loss": {
                "atr_multiplier": 2.0,
                "min_distance": 0.002,
                "max_distance": 0.05,
                "trailing_enabled": True,
                "trailing_distance": 0.015
            },
            "take_profit": {
                "targets": [1.5, 2.5, 4.0],  # Risk-reward ratios
                "partial_exits": [0.4, 0.3, 0.3]  # Position percentages
            }
        }

    async def generate_signals(self, integrated_analysis: Dict[str, Any], 
                             market_data: pd.DataFrame) -> List[TradingSignal]:
        """Generate trading signals from integrated analysis"""
        symbol = integrated_analysis["symbol"]
        timeframe = integrated_analysis["timeframe"]

        logger.info(f"Generating signals for {symbol} {timeframe}")

        # Extract market context
        context = self._extract_market_context(integrated_analysis, market_data)

        # Generate base signals from consensus
        base_signals = self._generate_base_signals(integrated_analysis, market_data, context)

        # Apply filters and validations
        filtered_signals = self._filter_signals(base_signals, context)

        # Add risk management parameters
        final_signals = self._add_risk_management(filtered_signals, market_data, context)

        # Store in history
        self.signal_history[symbol].extend(final_signals)

        # Save signals
        self._save_signals(final_signals, symbol, timeframe)

        return final_signals

    def _extract_market_context(self, analysis: Dict[str, Any], 
                               data: pd.DataFrame) -> MarketContext:
        """Extract current market context from analysis"""
        consensus = analysis["consensus"]

        # Determine trend
        trend = "sideways"
        if consensus["overall_sentiment"] == "bullish":
            trend = "bullish"
        elif consensus["overall_sentiment"] == "bearish":
            trend = "bearish"

        # Calculate volatility
        volatility = data['close'].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252)

        # Volume profile
        volume_ma = data['volume'].rolling(20).mean()
        volume_trend = "stable"
        if data['volume'].iloc[-5:].mean() > volume_ma.iloc[-1] * 1.2:
            volume_trend = "increasing"
        elif data['volume'].iloc[-5:].mean() < volume_ma.iloc[-1] * 0.8:
            volume_trend = "decreasing"

        # Market regime (from consensus)
        market_regime = "undefined"
        for analyzer_result in analysis["individual_analyses"].values():
            if "market_regime" in analyzer_result.get("results", {}):
                market_regime = analyzer_result["results"]["market_regime"]
                break

        # Key levels
        key_levels = consensus.get("key_levels", {})

        # Recent signals
        symbol = analysis["symbol"]
        recent_signals = self.signal_history[symbol][-10:] if symbol in self.signal_history else []

        return MarketContext(
            trend=trend,
            volatility=volatility,
            volume_profile=volume_trend,
            market_regime=market_regime,
            key_levels=key_levels,
            recent_signals=recent_signals
        )

    def _generate_base_signals(self, analysis: Dict[str, Any], 
                              data: pd.DataFrame, 
                              context: MarketContext) -> List[TradingSignal]:
        """Generate base signals from analysis"""
        signals = []
        consensus = analysis["consensus"]
        current_price = data['close'].iloc[-1]

        # Process consensus signals
        for signal_data in consensus["signals"]:
            signal_type = self._determine_signal_type(signal_data, context)
            priority = self._determine_priority(signal_data, context)

            # Skip neutral signals
            if signal_type == SignalType.NEUTRAL:
                continue

            # Create base signal
            signal = TradingSignal(
                timestamp=datetime.now(),
                symbol=analysis["symbol"],
                timeframe=analysis["timeframe"],
                signal_type=signal_type,
                priority=priority,
                entry_price=current_price,
                stop_loss=0.0,  # Will be calculated later
                take_profit_targets=[],  # Will be calculated later
                confidence=signal_data.get("confidence", 0.5),
                risk_reward_ratio=0.0,  # Will be calculated later
                reasoning=[signal_data.get("reason", "No reason provided")],
                supporting_indicators=self._extract_supporting_indicators(analysis, signal_data),
                metadata={
                    "source": signal_data.get("source", "unknown"),
                    "analysis_timestamp": analysis["timestamp"]
                }
            )

            signals.append(signal)

        # Generate signals from market structure
        structure_signals = self._generate_structure_signals(analysis, data, context)
        signals.extend(structure_signals)

        # Generate signals from patterns
        pattern_signals = self._generate_pattern_signals(analysis, data, context)
        signals.extend(pattern_signals)

        return signals

    def _determine_signal_type(self, signal_data: Dict[str, Any], 
                              context: MarketContext) -> SignalType:
        """Determine signal type based on signal data and context"""
        action = signal_data.get("action", "").lower()
        strength = signal_data.get("strength", "normal").lower()

        if action == "buy":
            if strength == "strong" or signal_data.get("confidence", 0) > 0.8:
                return SignalType.STRONG_BUY
            return SignalType.BUY
        elif action == "sell":
            if strength == "strong" or signal_data.get("confidence", 0) > 0.8:
                return SignalType.STRONG_SELL
            return SignalType.SELL
        elif action == "exit_long":
            return SignalType.EXIT_LONG
        elif action == "exit_short":
            return SignalType.EXIT_SHORT
        else:
            return SignalType.NEUTRAL

    def _determine_priority(self, signal_data: Dict[str, Any], 
                           context: MarketContext) -> SignalPriority:
        """Determine signal priority"""
        confidence = signal_data.get("confidence", 0.5)

        if confidence > 0.85 or signal_data.get("priority") == "critical":
            return SignalPriority.CRITICAL
        elif confidence > 0.7:
            return SignalPriority.HIGH
        elif confidence > 0.5:
            return SignalPriority.MEDIUM
        else:
            return SignalPriority.LOW

    def _extract_supporting_indicators(self, analysis: Dict[str, Any], 
                                     signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract supporting indicators for the signal"""
        supporting = {}

        # Get indicators from individual analyses
        for analyzer_name, analyzer_result in analysis["individual_analyses"].items():
            if not analyzer_result.get("errors"):
                results = analyzer_result.get("results", {})

                # Extract relevant indicators
                if "indicators" in results:
                    supporting[analyzer_name] = results["indicators"]

                # Extract patterns
                if "patterns" in results:
                    supporting[f"{analyzer_name}_patterns"] = results["patterns"]

        return supporting

    def _generate_structure_signals(self, analysis: Dict[str, Any], 
                                   data: pd.DataFrame, 
                                   context: MarketContext) -> List[TradingSignal]:
        """Generate signals from market structure analysis"""
        signals = []
        current_price = data['close'].iloc[-1]

        # Check for breakouts
        if "resistance" in context.key_levels and context.key_levels["resistance"]:
            nearest_resistance = min(context.key_levels["resistance"], 
                                   key=lambda x: abs(x - current_price))

            if current_price > nearest_resistance * 0.998:  # Near resistance
                signal = TradingSignal(
                    timestamp=datetime.now(),
                    symbol=analysis["symbol"],
                    timeframe=analysis["timeframe"],
                    signal_type=SignalType.SELL if context.trend != "bullish" else SignalType.NEUTRAL,
                    priority=SignalPriority.MEDIUM,
                    entry_price=current_price,
                    stop_loss=0.0,
                    take_profit_targets=[],
                    confidence=0.65,
                    risk_reward_ratio=0.0,
                    reasoning=["Price approaching resistance level", 
                              f"Resistance at {nearest_resistance}"],
                    metadata={"pattern": "resistance_test"}
                )
                signals.append(signal)

        # Check for support bounces
        if "support" in context.key_levels and context.key_levels["support"]:
            nearest_support = min(context.key_levels["support"], 
                                key=lambda x: abs(x - current_price))

            if current_price < nearest_support * 1.002:  # Near support
                signal = TradingSignal(
                    timestamp=datetime.now(),
                    symbol=analysis["symbol"],
                    timeframe=analysis["timeframe"],
                    signal_type=SignalType.BUY if context.trend != "bearish" else SignalType.NEUTRAL,
                    priority=SignalPriority.MEDIUM,
                    entry_price=current_price,
                    stop_loss=0.0,
                    take_profit_targets=[],
                    confidence=0.65,
                    risk_reward_ratio=0.0,
                    reasoning=["Price approaching support level", 
                              f"Support at {nearest_support}"],
                    metadata={"pattern": "support_test"}
                )
                signals.append(signal)

        return signals

    def _generate_pattern_signals(self, analysis: Dict[str, Any], 
                                 data: pd.DataFrame, 
                                 context: MarketContext) -> List[TradingSignal]:
        """Generate signals from pattern recognition"""
        signals = []

        # Extract patterns from individual analyzers
        all_patterns = []
        for analyzer_result in analysis["individual_analyses"].values():
            if "patterns" in analyzer_result.get("results", {}):
                patterns = analyzer_result["results"]["patterns"]
                if isinstance(patterns, list):
                    all_patterns.extend(patterns)

        # Process patterns
        for pattern in all_patterns:
            if isinstance(pattern, dict) and pattern.get("tradeable", False):
                signal_type = self._pattern_to_signal_type(pattern)
                if signal_type != SignalType.NEUTRAL:
                    signal = TradingSignal(
                        timestamp=datetime.now(),
                        symbol=analysis["symbol"],
                        timeframe=analysis["timeframe"],
                        signal_type=signal_type,
                        priority=SignalPriority.HIGH if pattern.get("reliability", 0) > 0.7 else SignalPriority.MEDIUM,
                        entry_price=pattern.get("entry_price", data['close'].iloc[-1]),
                        stop_loss=0.0,
                        take_profit_targets=[],
                        confidence=pattern.get("confidence", 0.6),
                        risk_reward_ratio=0.0,
                        reasoning=[f"Pattern detected: {pattern.get('name', 'Unknown')}",
                                  pattern.get("description", "")],
                        metadata={"pattern": pattern}
                    )
                    signals.append(signal)

        return signals

    def _pattern_to_signal_type(self, pattern: Dict[str, Any]) -> SignalType:
        """Convert pattern to signal type"""
        pattern_type = pattern.get("type", "").lower()
        pattern_name = pattern.get("name", "").lower()

        bullish_patterns = ["double_bottom", "inverse_head_shoulders", "bullish_flag", 
                           "ascending_triangle", "bullish_engulfing", "morning_star"]
        bearish_patterns = ["double_top", "head_shoulders", "bearish_flag", 
                           "descending_triangle", "bearish_engulfing", "evening_star"]

        if any(p in pattern_name for p in bullish_patterns) or pattern_type == "bullish":
            return SignalType.BUY
        elif any(p in pattern_name for p in bearish_patterns) or pattern_type == "bearish":
            return SignalType.SELL
        else:
            return SignalType.NEUTRAL

    def _filter_signals(self, signals: List[TradingSignal], 
                       context: MarketContext) -> List[TradingSignal]:
        """Filter signals based on quality criteria"""
        filtered = []
        filters = self.config["signal_filters"]

        for signal in signals:
            # Confidence filter
            if signal.confidence < filters["min_confidence"]:
                continue

            # Trend alignment filter
            if not self._is_trend_aligned(signal, context):
                continue

            # Avoid contradicting recent signals
            if self._contradicts_recent_signals(signal, context.recent_signals):
                continue

            # Volume confirmation
            if context.volume_profile == "decreasing" and signal.priority != SignalPriority.CRITICAL:
                signal.confidence *= 0.8  # Reduce confidence

            filtered.append(signal)

        # Remove duplicates
        filtered = self._remove_duplicate_signals(filtered)

        return filtered

    def _is_trend_aligned(self, signal: TradingSignal, context: MarketContext) -> bool:
        """Check if signal aligns with current trend"""
        if context.trend == "bullish" and signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
            return signal.priority == SignalPriority.CRITICAL
        elif context.trend == "bearish" and signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
            return signal.priority == SignalPriority.CRITICAL
        return True

    def _contradicts_recent_signals(self, signal: TradingSignal, 
                                   recent_signals: List[TradingSignal]) -> bool:
        """Check if signal contradicts recent signals"""
        if not recent_signals:
            return False

        # Check last 3 signals
        for recent in recent_signals[-3:]:
            time_diff = (signal.timestamp - recent.timestamp).total_seconds() / 60

            # If recent signal in last 30 minutes
            if time_diff < 30:
                if (signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY] and 
                    recent.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]):
                    return True
                elif (signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL] and 
                      recent.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]):
                    return True

        return False

    def _remove_duplicate_signals(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """Remove duplicate signals, keeping highest confidence"""
        unique_signals = {}

        for signal in signals:
            key = (signal.symbol, signal.signal_type, round(signal.entry_price, 4))

            if key not in unique_signals or signal.confidence > unique_signals[key].confidence:
                unique_signals[key] = signal

        return list(unique_signals.values())

    def _add_risk_management(self, signals: List[TradingSignal], 
                           data: pd.DataFrame, 
                           context: MarketContext) -> List[TradingSignal]:
        """Add stop loss, take profit, and position sizing"""
        enhanced_signals = []

        for signal in signals:
            # Calculate ATR for dynamic stops
            atr = data['close'].rolling(14).apply(
                lambda x: np.mean(np.abs(np.diff(x)))
            ).iloc[-1]

            # Set stop loss
            stop_loss_distance = self._calculate_stop_loss_distance(signal, atr, data)
            if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                signal.stop_loss = signal.entry_price * (1 - stop_loss_distance)
            else:
                signal.stop_loss = signal.entry_price * (1 + stop_loss_distance)

            # Set take profit targets
            signal.take_profit_targets = self._calculate_take_profit_targets(
                signal, stop_loss_distance
            )

            # Calculate risk-reward ratio
            if signal.take_profit_targets:
                avg_target_distance = np.mean([
                    abs(target - signal.entry_price) / signal.entry_price 
                    for target in signal.take_profit_targets
                ])
                signal.risk_reward_ratio = avg_target_distance / stop_loss_distance

            # Skip if risk-reward is too low
            if signal.risk_reward_ratio < self.config["signal_filters"]["min_risk_reward"]:
                continue

            # Calculate position size
            signal.position_size_suggestion = self._calculate_position_size(
                signal, context, data
            )

            # Set expiry
            signal.expiry = signal.timestamp + timedelta(
                minutes=self._get_signal_expiry_minutes(signal.timeframe)
            )

            enhanced_signals.append(signal)

        return enhanced_signals

    def _calculate_stop_loss_distance(self, signal: TradingSignal, 
                                    atr: float, 
                                    data: pd.DataFrame) -> float:
        """Calculate appropriate stop loss distance"""
        config = self.config["stop_loss"]

        # Base distance on ATR
        base_distance = (atr / signal.entry_price) * config["atr_multiplier"]

        # Apply min/max constraints
        distance = max(config["min_distance"], min(base_distance, config["max_distance"]))

        # Adjust based on signal priority
        if signal.priority == SignalPriority.CRITICAL:
            distance *= 1.2  # Wider stop for high conviction
        elif signal.priority == SignalPriority.LOW:
            distance *= 0.8  # Tighter stop for low conviction

        return distance

    def _calculate_take_profit_targets(self, signal: TradingSignal, 
                                     stop_distance: float) -> List[float]:
        """Calculate take profit targets based on risk-reward ratios"""
        targets = []
        tp_config = self.config["take_profit"]

        for rr_ratio in tp_config["targets"]:
            profit_distance = stop_distance * rr_ratio

            if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                target = signal.entry_price * (1 + profit_distance)
            else:
                target = signal.entry_price * (1 - profit_distance)

            targets.append(target)

        return targets

    def _calculate_position_size(self, signal: TradingSignal, 
                               context: MarketContext, 
                               data: pd.DataFrame) -> float:
        """Calculate position size using Kelly Criterion or fixed risk"""
        method = self.config["position_sizing"]["method"]

        if method == "kelly_criterion":
            # Simplified Kelly Criterion
            win_rate = self._estimate_win_rate(signal, context)
            avg_win_loss_ratio = signal.risk_reward_ratio

            kelly_percentage = (win_rate * avg_win_loss_ratio - (1 - win_rate)) / avg_win_loss_ratio
            kelly_percentage = max(0, min(kelly_percentage, 0.25))  # Cap at 25%

            # Scale with confidence if enabled
            if self.config["position_sizing"]["scale_with_confidence"]:
                kelly_percentage *= signal.confidence

            position_size = kelly_percentage
        else:
            # Fixed risk per trade
            position_size = self.risk_per_trade

        # Apply maximum position size limit
        max_size = self.config["position_sizing"]["max_position_size"]
        position_size = min(position_size, max_size)

        # Reduce size in high volatility
        if context.volatility > 0.3:  # 30% annualized volatility
            position_size *= 0.7

        return round(position_size, 4)

    def _estimate_win_rate(self, signal: TradingSignal, context: MarketContext) -> float:
        """Estimate win rate based on historical performance and current conditions"""
        # Base win rate on confidence
        base_win_rate = 0.4 + (signal.confidence * 0.3)

        # Adjust for trend alignment
        if context.trend == "bullish" and signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
            base_win_rate += 0.1
        elif context.trend == "bearish" and signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
            base_win_rate += 0.1

        # Adjust for market regime
        if context.market_regime == "trending":
            base_win_rate += 0.05
        elif context.market_regime == "volatile":
            base_win_rate -= 0.1

        return max(0.3, min(base_win_rate, 0.8))

    def _get_signal_expiry_minutes(self, timeframe: str) -> int:
        """Get signal expiry time based on timeframe"""
        expiry_map = {
            "1m": 15,
            "5m": 30,
            "15m": 60,
            "1h": 240,
            "4h": 720,
            "1d": 1440
        }
        return expiry_map.get(timeframe, 60)

    def _save_signals(self, signals: List[TradingSignal], symbol: str, timeframe: str):
        """Save signals to file"""
        if not signals:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{symbol.replace('/', '_')}_{timeframe}_{timestamp}_signals.json"
        filepath = self.signals_dir / filename

        # Convert signals to dict format
        signals_data = []
        for signal in signals:
            signal_dict = {
                "timestamp": signal.timestamp.isoformat(),
                "symbol": signal.symbol,
                "timeframe": signal.timeframe,
                "signal_type": signal.signal_type.value,
                "priority": signal.priority.value,
                "entry_price": signal.entry_price,
                "stop_loss": signal.stop_loss,
                "take_profit_targets": signal.take_profit_targets,
                "confidence": signal.confidence,
                "risk_reward_ratio": signal.risk_reward_ratio,
                "position_size_suggestion": signal.position_size_suggestion,
                "reasoning": signal.reasoning,
                "supporting_indicators": signal.supporting_indicators,
                "conflicting_indicators": signal.conflicting_indicators,
                "metadata": signal.metadata,
                "expiry": signal.expiry.isoformat() if signal.expiry else None
            }
            signals_data.append(signal_dict)

        # Save with metadata
        output_data = {
            "generated_at": datetime.now().isoformat(),
            "symbol": symbol,
            "timeframe": timeframe,
            "total_signals": len(signals),
            "signals": signals_data,
            "market_context": {
                "current_price": signals[0].entry_price if signals else 0,
                "signal_summary": {
                    "buy_signals": sum(1 for s in signals if s.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]),
                    "sell_signals": sum(1 for s in signals if s.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]),
                    "exit_signals": sum(1 for s in signals if s.signal_type in [SignalType.EXIT_LONG, SignalType.EXIT_SHORT])
                }
            }
        }

        with open(filepath, 'w') as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"Saved {len(signals)} signals to {filepath}")

    async def monitor_signals(self, symbols: List[str], interval_seconds: int = 60):
        """Monitor multiple symbols and generate signals periodically"""
        logger.info(f"Starting signal monitoring for {symbols}")

        while True:
            try:
                for symbol in symbols:
                    # This would integrate with your data pipeline and integration engine
                    # For now, it's a placeholder
                    logger.info(f"Checking {symbol} for new signals...")

                await asyncio.sleep(interval_seconds)

            except KeyboardInterrupt:
                logger.info("Signal monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in signal monitoring: {e}")
                await asyncio.sleep(interval_seconds)

# Main execution
async def main():
    """Example usage of signal generator"""
    generator = SignalGenerator()

    # Example: Generate signals from integrated analysis
    # integrated_analysis = load_integrated_analysis()  # Load from integration engine
    # market_data = pd.read_csv("market_data.csv", index_col=0, parse_dates=True)
    # signals = await generator.generate_signals(integrated_analysis, market_data)

    logger.info("Signal generator ready for use")

if __name__ == "__main__":
    asyncio.run(main())
