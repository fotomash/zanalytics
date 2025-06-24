#!/usr/bin/env python3
"""
Ultimate Strategy Output Merger
Consolidates multi-timeframe JSONs + tick data into XANA-ready format
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
import logging
from typing import Dict, List, Optional, Any
import glob
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StrategyOutputMerger:
    """Merges strategy outputs into consolidated XANA format"""

    def __init__(self, data_dir: str = "./data", tick_window_size: int = 100):
        self.data_dir = Path(data_dir)
        self.tick_window_size = tick_window_size

    def merge_strategy_outputs(self, symbol: str = "XAUUSD") -> Dict[str, Any]:
        """Main merge function - creates consolidated output"""

        merged = {
            "symbol": symbol,
            "timestamp": datetime.utcnow().isoformat(),
            "summaries": {},
            "microstructure": {},
            "entry_signals": {},
            "meta": {
                "merger_version": "1.0",
                "data_source": str(self.data_dir),
                "tick_window_size": self.tick_window_size
            }
        }

        try:
            # 1. Load multi-timeframe summaries
            merged["summaries"] = self._load_timeframe_summaries(symbol)

            # 2. Load tick microstructure
            merged["microstructure"] = self._load_tick_microstructure(symbol)

            # 3. Generate entry signals
            merged["entry_signals"] = self._generate_entry_signals(merged)

            # 4. Add metadata
            merged["meta"]["timeframes_found"] = list(merged["summaries"].keys())
            merged["meta"]["total_indicators"] = self._count_indicators(merged["summaries"])

            logger.info(f"✅ Merged {len(merged['summaries'])} timeframes for {symbol}")
            return merged

        except Exception as e:
            logger.error(f"❌ Merger failed: {e}")
            merged["error"] = str(e)
            return merged

    def _load_timeframe_summaries(self, symbol: str) -> Dict[str, Any]:
        """Load all SUMMARY_*.json files"""
        summaries = {}

        # Pattern: SYMBOL_*_SUMMARY_*.json (from your ncOS analyzer)
        pattern = f"{symbol}_*_SUMMARY_*.json"
        files = list(self.data_dir.glob(pattern))

        if not files:
            # Alternative pattern: just SUMMARY_*.json
            files = list(self.data_dir.glob("SUMMARY_*.json"))

        logger.info(f"Found {len(files)} summary files")

        for file_path in files:
            try:
                # Extract timeframe from filename
                # e.g., XAUUSD_M1_bars_SUMMARY_1T.json -> 1T
                timeframe = self._extract_timeframe_from_filename(file_path.name)

                with open(file_path, 'r') as f:
                    data = json.load(f)
                    summaries[timeframe] = data

                logger.info(f"  ✅ Loaded {timeframe}: {file_path.name}")

            except Exception as e:
                logger.warning(f"  ❌ Failed to load {file_path.name}: {e}")

        return summaries

    def _extract_timeframe_from_filename(self, filename: str) -> str:
        """Extract timeframe from filename"""
        # Patterns: SUMMARY_1T.json, SUMMARY_5T.json, etc.
        if "SUMMARY_" in filename:
            parts = filename.split("_")
            for part in parts:
                if part.replace(".json", "") in ["1T", "5T", "15T", "30T", "1H", "4H", "1D", "1W", "1M"]:
                    return part.replace(".json", "")

        # Fallback
        if "1T" in filename: return "1T"
        if "5T" in filename: return "5T"
        if "15T" in filename: return "15T"
        if "30T" in filename: return "30T"
        if "1H" in filename: return "1H"
        if "4H" in filename: return "4H"
        if "1D" in filename: return "1D"

        return "unknown"

    def _load_tick_microstructure(self, symbol: str) -> Dict[str, Any]:
        """Load tick data and create microstructure snapshot"""
        microstructure = {
            "tick_window": [],
            "tick_analysis": {},
            "market_state": {}
        }

        # Look for tick files
        tick_patterns = [
            f"{symbol}_TICK*.csv",
            f"{symbol}_tick*.csv", 
            "*TICK*.csv",
            "*tick*.csv"
        ]

        tick_file = None
        for pattern in tick_patterns:
            files = list(self.data_dir.glob(pattern))
            if files:
                tick_file = files[0]  # Take the first match
                break

        if not tick_file:
            logger.warning("No tick file found")
            return microstructure

        try:
            # Load tick data
            df = pd.read_csv(tick_file)

            # Take last N ticks
            if len(df) > self.tick_window_size:
                df = df.tail(self.tick_window_size)

            # Create tick window
            tick_window = []
            for _, row in df.iterrows():
                tick = {
                    "ts": str(row.get('timestamp', '')),
                    "bid": float(row.get('bid', 0)),
                    "ask": float(row.get('ask', 0)),
                    "last": float(row.get('last', 0)),
                    "volume": float(row.get('volume', 0)),
                    "spread": float(row.get('spread_price', row.get('spread', 0)))
                }

                # Calculate mid price
                if tick["bid"] > 0 and tick["ask"] > 0:
                    tick["mid"] = (tick["bid"] + tick["ask"]) / 2
                else:
                    tick["mid"] = tick["last"]

                # Add momentum score (simple)
                tick["momentum"] = 0.5  # Placeholder
                tick["spring"] = False  # Placeholder

                tick_window.append(tick)

            microstructure["tick_window"] = tick_window

            # Add tick analysis
            if len(df) > 0:
                microstructure["tick_analysis"] = {
                    "total_ticks": len(df),
                    "avg_spread": float(df.get('spread_price', df.get('spread', [0])).mean()),
                    "last_price": float(df.iloc[-1].get('last', 0)),
                    "price_range": float(df.get('last', [0]).max() - df.get('last', [0]).min()),
                    "volume_total": float(df.get('volume', [0]).sum())
                }

            logger.info(f"✅ Loaded {len(tick_window)} ticks from {tick_file.name}")

        except Exception as e:
            logger.warning(f"❌ Failed to load tick data: {e}")

        return microstructure

    def _generate_entry_signals(self, merged_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate entry signals from merged data"""
        signals = {
            "v5": {"direction": "NEUTRAL", "confidence": 0.5, "poi_tap": False},
            "v10": {"judas": False, "killzone": "unknown", "bias": "neutral"},
            "ncos": {"manipulation_detected": False, "liquidity_sweep": False},
            "confluence": {"total_score": 0, "timeframes_aligned": 0}
        }

        try:
            summaries = merged_data.get("summaries", {})

            # Analyze confluence across timeframes
            bullish_signals = 0
            bearish_signals = 0
            total_timeframes = len(summaries)

            for timeframe, data in summaries.items():
                if isinstance(data, dict):
                    # Check for bullish signals
                    if self._is_bullish_signal(data):
                        bullish_signals += 1
                    elif self._is_bearish_signal(data):
                        bearish_signals += 1

            # Calculate confluence
            if total_timeframes > 0:
                bullish_ratio = bullish_signals / total_timeframes
                bearish_ratio = bearish_signals / total_timeframes

                if bullish_ratio > 0.6:
                    signals["v5"]["direction"] = "BUY"
                    signals["v5"]["confidence"] = min(0.9, bullish_ratio)
                    signals["v10"]["bias"] = "long"
                elif bearish_ratio > 0.6:
                    signals["v5"]["direction"] = "SELL"
                    signals["v5"]["confidence"] = min(0.9, bearish_ratio)
                    signals["v10"]["bias"] = "short"

                signals["confluence"]["timeframes_aligned"] = max(bullish_signals, bearish_signals)
                signals["confluence"]["total_score"] = max(bullish_ratio, bearish_ratio)

            # Add microstructure signals
            microstructure = merged_data.get("microstructure", {})
            tick_analysis = microstructure.get("tick_analysis", {})

            if tick_analysis.get("avg_spread", 0) > 0.5:  # Wide spread
                signals["ncos"]["manipulation_detected"] = True

            # Session detection (simplified)
            current_hour = datetime.utcnow().hour
            if 7 <= current_hour <= 10:
                signals["v10"]["killzone"] = "London"
            elif 13 <= current_hour <= 16:
                signals["v10"]["killzone"] = "New York"
            else:
                signals["v10"]["killzone"] = "Asian"

        except Exception as e:
            logger.warning(f"Signal generation failed: {e}")

        return signals

    def _is_bullish_signal(self, data: Dict) -> bool:
        """Check if data contains bullish signals"""
        # This is a simplified check - you can enhance based on your indicators
        bullish_indicators = 0

        # Check RSI
        if isinstance(data, dict):
            for key, value in data.items():
                if "rsi" in key.lower() and isinstance(value, (int, float)):
                    if 30 <= value <= 50:  # RSI in oversold recovery
                        bullish_indicators += 1

                # Check MACD
                if "macd" in key.lower() and "signal" not in key.lower():
                    if isinstance(value, (int, float)) and value > 0:
                        bullish_indicators += 1

                # Check trend
                if "trend" in key.lower() and isinstance(value, str):
                    if "up" in value.lower() or "bull" in value.lower():
                        bullish_indicators += 1

        return bullish_indicators >= 2

    def _is_bearish_signal(self, data: Dict) -> bool:
        """Check if data contains bearish signals"""
        bearish_indicators = 0

        if isinstance(data, dict):
            for key, value in data.items():
                if "rsi" in key.lower() and isinstance(value, (int, float)):
                    if 50 <= value <= 70:  # RSI in overbought territory
                        bearish_indicators += 1

                if "macd" in key.lower() and "signal" not in key.lower():
                    if isinstance(value, (int, float)) and value < 0:
                        bearish_indicators += 1

                if "trend" in key.lower() and isinstance(value, str):
                    if "down" in value.lower() or "bear" in value.lower():
                        bearish_indicators += 1

        return bearish_indicators >= 2

    def _count_indicators(self, summaries: Dict[str, Any]) -> int:
        """Count total indicators across all timeframes"""
        total = 0
        for tf_data in summaries.values():
            if isinstance(tf_data, dict):
                total += len(tf_data)
        return total

    def save_merged_output(self, merged_data: Dict[str, Any], output_file: str = None) -> str:
        """Save merged output to JSON file"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.data_dir / f"merged_snapshot_{timestamp}.json"

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(merged_data, f, indent=2, default=str)

        logger.info(f"💾 Saved merged output: {output_path}")
        return str(output_path)

def main():
    """CLI interface for the merger"""
    parser = argparse.ArgumentParser(description='Ultimate Strategy Output Merger')

    parser.add_argument('--data-dir', '-d', type=str, default='./data',
                       help='Directory containing SUMMARY_*.json and tick files')
    parser.add_argument('--symbol', '-s', type=str, default='XAUUSD',
                       help='Symbol to process (default: XAUUSD)')
    parser.add_argument('--tick-window', '-t', type=int, default=100,
                       help='Number of recent ticks to include (default: 100)')
    parser.add_argument('--output', '-o', type=str,
                       help='Output file path (default: auto-generated)')
    parser.add_argument('--watch', '-w', action='store_true',
                       help='Watch mode - continuously merge every 30 seconds')

    args = parser.parse_args()

    # Create merger
    merger = StrategyOutputMerger(
        data_dir=args.data_dir,
        tick_window_size=args.tick_window
    )

    if args.watch:
        # Watch mode
        import time
        logger.info(f"🔄 Watch mode: merging every 30 seconds...")

        while True:
            try:
                merged = merger.merge_strategy_outputs(args.symbol)
                output_file = merger.save_merged_output(merged, args.output)
                logger.info(f"✅ Updated: {output_file}")
                time.sleep(30)
            except KeyboardInterrupt:
                logger.info("👋 Watch mode stopped")
                break
            except Exception as e:
                logger.error(f"❌ Watch error: {e}")
                time.sleep(30)
    else:
        # Single run
        merged = merger.merge_strategy_outputs(args.symbol)
        output_file = merger.save_merged_output(merged, args.output)

        # Print summary
        print(f"\n🎉 MERGER COMPLETE")
        print(f"📊 Timeframes: {len(merged.get('summaries', {}))}")
        print(f"🎯 Ticks: {len(merged.get('microstructure', {}).get('tick_window', []))}")
        print(f"📈 Signals: {merged.get('entry_signals', {}).get('confluence', {}).get('total_score', 0):.1%} confidence")
        print(f"💾 Output: {output_file}")

if __name__ == "__main__":
    main()
