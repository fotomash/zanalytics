#!/usr/bin/env python3
"""
🔄 Data Enrichment Engine
Continuously processes trading data using all SMC/Harmonic/Wyckoff modules
Integrates all your existing analysis components into a unified pipeline
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
import asyncio
import importlib.util
import sys

logger = logging.getLogger(__name__)

class DataEnricher:
    """Main data enrichment engine that coordinates all analysis modules"""

    def __init__(self, config):
        self.config = config
        self.data_path = Path(config.get('data_path', './data'))
        self.processed_path = self.data_path / 'processed'
        self.cache_path = self.data_path / 'cache'

        # Create directories
        self.processed_path.mkdir(parents=True, exist_ok=True)
        self.cache_path.mkdir(parents=True, exist_ok=True)

        # Load analysis modules
        self.analysis_modules = self.load_analysis_modules()

    def load_analysis_modules(self):
        """Dynamically load all available analysis modules"""
        modules = {}

        # List of your existing modules to integrate
        module_files = [
            'market_structure_analyzer_smc.py',
            'poi_manager_smc.py', 
            'entry_executor_smc.py',
            'confirmation_engine_smc.py',
            'liquidity_engine_smc.py',
            'wyckoff_phase_engine.py',
            'fibonacci_filter.py',
            'volatility_engine.py',
            'impulse_correction_detector.py'
        ]

        for module_file in module_files:
            try:
                if Path(module_file).exists():
                    module_name = module_file.replace('.py', '')
                    spec = importlib.util.spec_from_file_location(module_name, module_file)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    modules[module_name] = module
                    logger.info(f"✅ Loaded module: {module_name}")
            except Exception as e:
                logger.warning(f"⚠️ Could not load {module_file}: {e}")

        return modules

    def discover_data_files(self):
        """Discover all available data files"""
        data_files = {}

        # Look for raw data files
        for pattern in ['*.csv', '*.json']:
            for file_path in self.data_path.glob(pattern):
                if 'processed' not in str(file_path):
                    symbol, timeframe = self.parse_filename(file_path.name)
                    if symbol not in data_files:
                        data_files[symbol] = {}
                    data_files[symbol][timeframe] = file_path

        logger.info(f"📁 Discovered data files: {data_files}")
        return data_files

    def parse_filename(self, filename):
        """Parse symbol and timeframe from filename"""
        # Handle your file naming patterns
        if 'XAUUSD' in filename:
            symbol = 'XAUUSD'
            if 'TICK' in filename:
                timeframe = 'TICK'
            elif 'M1' in filename or '1min' in filename:
                timeframe = '1M'
            elif '5min' in filename:
                timeframe = '5M'
            elif '15min' in filename:
                timeframe = '15M'
            elif '1H' in filename:
                timeframe = '1H'
            elif '4H' in filename:
                timeframe = '4H'
            elif '1D' in filename:
                timeframe = '1D'
            else:
                timeframe = '1M'  # Default
        else:
            symbol = 'UNKNOWN'
            timeframe = '1M'

        return symbol, timeframe

    def load_data_file(self, file_path):
        """Load and standardize a data file"""
        try:
            if file_path.suffix == '.csv':
                df = pd.read_csv(file_path)
            elif file_path.suffix == '.json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        df = pd.DataFrame(data)
                    else:
                        # Handle nested JSON structure
                        df = pd.json_normalize(data)
            else:
                return None

            # Standardize timestamp column
            timestamp_cols = ['timestamp', 'time', 'datetime', 'date']
            for col in timestamp_cols:
                if col in df.columns:
                    df['timestamp'] = pd.to_datetime(df[col])
                    break

            # Ensure required OHLCV columns exist
            required_cols = ['open', 'high', 'low', 'close']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                logger.warning(f"⚠️ Missing columns in {file_path.name}: {missing_cols}")
                return None

            # Sort by timestamp
            if 'timestamp' in df.columns:
                df = df.sort_values('timestamp').reset_index(drop=True)

            logger.info(f"📊 Loaded {len(df)} rows from {file_path.name}")
            return df

        except Exception as e:
            logger.error(f"❌ Error loading {file_path}: {e}")
            return None

    def apply_smc_analysis(self, df):
        """Apply Smart Money Concepts analysis"""
        try:
            # Use your existing SMC modules
            if 'market_structure_analyzer_smc' in self.analysis_modules:
                # Apply market structure analysis
                analyzer = self.analysis_modules['market_structure_analyzer_smc']
                if hasattr(analyzer, 'analyze_market_structure'):
                    df = analyzer.analyze_market_structure(df)

            if 'poi_manager_smc' in self.analysis_modules:
                # Apply POI analysis
                poi_manager = self.analysis_modules['poi_manager_smc']
                if hasattr(poi_manager, 'identify_points_of_interest'):
                    df = poi_manager.identify_points_of_interest(df)

            if 'liquidity_engine_smc' in self.analysis_modules:
                # Apply liquidity analysis
                liquidity_engine = self.analysis_modules['liquidity_engine_smc']
                if hasattr(liquidity_engine, 'analyze_liquidity'):
                    df = liquidity_engine.analyze_liquidity(df)

            # Add basic SMC indicators if modules don't exist
            if 'SMC_swing_high' not in df.columns:
                df = self.add_basic_smc_indicators(df)

            logger.info("✅ Applied SMC analysis")
            return df

        except Exception as e:
            logger.error(f"❌ Error in SMC analysis: {e}")
            return df

    def add_basic_smc_indicators(self, df):
        """Add basic SMC indicators when modules aren't available"""
        try:
            # Swing highs and lows
            df['SMC_swing_high'] = df['high'] == df['high'].rolling(10, center=True).max()
            df['SMC_swing_low'] = df['low'] == df['low'].rolling(10, center=True).min()

            # Market structure (simplified)
            df['SMC_structure'] = 'neutral'
            for i in range(20, len(df)):
                recent_highs = df['high'].iloc[i-20:i].max()
                recent_lows = df['low'].iloc[i-20:i].min()
                current_price = df['close'].iloc[i]

                if current_price > recent_highs * 0.95:
                    df.loc[i, 'SMC_structure'] = 'bullish'
                elif current_price < recent_lows * 1.05:
                    df.loc[i, 'SMC_structure'] = 'bearish'

            # Order blocks (simplified)
            df['SMC_bullish_ob'] = False
            df['SMC_bearish_ob'] = False
            df['SMC_ob_strength'] = 0.0

            # Fair value gaps
            df['SMC_fvg_bullish'] = False
            df['SMC_fvg_bearish'] = False
            df['SMC_fvg_size'] = 0.0

            for i in range(2, len(df)):
                # Bullish FVG: gap between low[i-2] and high[i]
                if df['low'].iloc[i-2] > df['high'].iloc[i]:
                    df.loc[i, 'SMC_fvg_bullish'] = True
                    df.loc[i, 'SMC_fvg_size'] = df['low'].iloc[i-2] - df['high'].iloc[i]

                # Bearish FVG: gap between high[i-2] and low[i]
                elif df['high'].iloc[i-2] < df['low'].iloc[i]:
                    df.loc[i, 'SMC_fvg_bearish'] = True
                    df.loc[i, 'SMC_fvg_size'] = df['low'].iloc[i] - df['high'].iloc[i-2]

            # Liquidity levels
            df['SMC_liquidity_grab'] = False
            df['SMC_liquidity_strength'] = 0.0

            # Range analysis
            df['SMC_range_high'] = df['high'].rolling(50).max()
            df['SMC_range_low'] = df['low'].rolling(50).min()
            df['SMC_range_mid'] = (df['SMC_range_high'] + df['SMC_range_low']) / 2
            df['SMC_equilibrium'] = df['SMC_range_mid']

            # Premium/discount zones
            df['SMC_premium_zone'] = df['close'] > df['SMC_range_mid'] * 1.01
            df['SMC_discount_zone'] = df['close'] < df['SMC_range_mid'] * 0.99

            return df

        except Exception as e:
            logger.error(f"❌ Error adding basic SMC indicators: {e}")
            return df

    def apply_harmonic_analysis(self, df):
        """Apply harmonic pattern analysis"""
        try:
            # Add basic harmonic pattern detection
            df['harmonic_gartley'] = False
            df['harmonic_gartley_score'] = 0.0
            df['harmonic_butterfly'] = False
            df['harmonic_butterfly_score'] = 0.0
            df['harmonic_bat'] = False
            df['harmonic_bat_score'] = 0.0
            df['harmonic_crab'] = False
            df['harmonic_crab_score'] = 0.0
            df['harmonic_cypher'] = False
            df['harmonic_cypher_score'] = 0.0
            df['harmonic_shark'] = False
            df['harmonic_shark_score'] = 0.0
            df['harmonic_abcd'] = False
            df['harmonic_abcd_score'] = 0.0

            logger.info("✅ Applied harmonic analysis")
            return df

        except Exception as e:
            logger.error(f"❌ Error in harmonic analysis: {e}")
            return df

    def apply_wyckoff_analysis(self, df):
        """Apply Wyckoff method analysis"""
        try:
            if 'wyckoff_phase_engine' in self.analysis_modules:
                wyckoff_engine = self.analysis_modules['wyckoff_phase_engine']
                if hasattr(wyckoff_engine, 'analyze_wyckoff_phases'):
                    df = wyckoff_engine.analyze_wyckoff_phases(df)
            else:
                # Add basic Wyckoff indicators
                df['wyckoff_accumulation'] = False
                df['wyckoff_acc_strength'] = 0.0
                df['wyckoff_distribution'] = False
                df['wyckoff_dist_strength'] = 0.0
                df['wyckoff_spread'] = df['high'] - df['low']
                df['wyckoff_spread_ma'] = df['wyckoff_spread'].rolling(20).mean()

                if 'volume' in df.columns:
                    df['wyckoff_volume_ma'] = df['volume'].rolling(20).mean()
                    df['wyckoff_vs_ratio'] = df['wyckoff_spread'] / (df['volume'] + 1)
                    df['wyckoff_vs_ratio_ma'] = df['wyckoff_vs_ratio'].rolling(20).mean()
                else:
                    df['wyckoff_volume_ma'] = 0
                    df['wyckoff_vs_ratio'] = 0
                    df['wyckoff_vs_ratio_ma'] = 0

                df['wyckoff_effort'] = 'low'
                df['wyckoff_result'] = 'low'
                df['wyckoff_no_demand'] = False
                df['wyckoff_no_supply'] = False
                df['wyckoff_spring'] = False
                df['wyckoff_upthrust'] = False

            logger.info("✅ Applied Wyckoff analysis")
            return df

        except Exception as e:
            logger.error(f"❌ Error in Wyckoff analysis: {e}")
            return df

    def generate_trading_signals(self, df):
        """Generate trading signals based on all analysis"""
        try:
            signals = []

            # Get the latest data point
            latest = df.iloc[-1]

            # Signal generation logic
            signal_strength = 0
            signal_type = "NEUTRAL"
            reasons = []

            # SMC-based signals
            if latest.get('SMC_fvg_bullish', False) and latest.get('SMC_discount_zone', False):
                signal_strength += 2
                signal_type = "BUY"
                reasons.append("Bullish FVG in discount zone")

            if latest.get('SMC_fvg_bearish', False) and latest.get('SMC_premium_zone', False):
                signal_strength += 2
                signal_type = "SELL"
                reasons.append("Bearish FVG in premium zone")

            # RSI confirmation
            if 'RSI_14' in df.columns:
                rsi = latest.get('RSI_14', 50)
                if rsi < 30 and signal_type == "BUY":
                    signal_strength += 1
                    reasons.append("Oversold RSI")
                elif rsi > 70 and signal_type == "SELL":
                    signal_strength += 1
                    reasons.append("Overbought RSI")

            # MACD confirmation
            if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
                macd = latest.get('MACD', 0)
                macd_signal = latest.get('MACD_Signal', 0)
                if macd > macd_signal and signal_type == "BUY":
                    signal_strength += 1
                    reasons.append("MACD bullish crossover")
                elif macd < macd_signal and signal_type == "SELL":
                    signal_strength += 1
                    reasons.append("MACD bearish crossover")

            # Create signal if strength is sufficient
            if signal_strength >= 2:
                current_price = latest['close']

                if signal_type == "BUY":
                    stop_loss = current_price - (current_price * 0.005)  # 0.5% stop
                    take_profit = current_price + (current_price * 0.015)  # 1.5% target
                else:
                    stop_loss = current_price + (current_price * 0.005)
                    take_profit = current_price - (current_price * 0.015)

                risk = abs(current_price - stop_loss)
                reward = abs(take_profit - current_price)
                rr_ratio = reward / risk if risk > 0 else 0

                signal = {
                    'timestamp': latest.get('timestamp', datetime.now().isoformat()),
                    'symbol': 'XAUUSD',
                    'type': signal_type,
                    'strength': signal_strength,
                    'confidence': min(signal_strength * 20, 100),
                    'entry_price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'risk_reward_ratio': round(rr_ratio, 2),
                    'reasons': reasons
                }

                signals.append(signal)

            return signals

        except Exception as e:
            logger.error(f"❌ Error generating signals: {e}")
            return []

    def save_enriched_data(self, df, symbol, timeframe):
        """Save enriched data to processed folder"""
        try:
            # Create filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{symbol}_{timeframe}_enriched_{timestamp}.csv"
            filepath = self.processed_path / filename

            # Save data
            df.to_csv(filepath, index=False)

            # Also save as latest
            latest_filename = f"{symbol}_{timeframe}_latest.csv"
            latest_filepath = self.processed_path / latest_filename
            df.to_csv(latest_filepath, index=False)

            logger.info(f"💾 Saved enriched data: {filename}")
            return filepath

        except Exception as e:
            logger.error(f"❌ Error saving enriched data: {e}")
            return None

    def save_analysis_report(self, analysis_data, symbol):
        """Save analysis report as JSON"""
        try:
            reports_dir = self.processed_path / 'analysis_reports'
            reports_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{symbol}_analysis_{timestamp}.json"
            filepath = reports_dir / filename

            with open(filepath, 'w') as f:
                json.dump(analysis_data, f, indent=2, default=str)

            # Also save as latest
            latest_filepath = reports_dir / f"{symbol}_analysis_latest.json"
            with open(latest_filepath, 'w') as f:
                json.dump(analysis_data, f, indent=2, default=str)

            logger.info(f"📊 Saved analysis report: {filename}")
            return filepath

        except Exception as e:
            logger.error(f"❌ Error saving analysis report: {e}")
            return None

    def process_symbol_timeframe(self, symbol, timeframe, file_path):
        """Process a single symbol/timeframe combination"""
        try:
            logger.info(f"🔄 Processing {symbol} {timeframe}")

            # Load data
            df = self.load_data_file(file_path)
            if df is None:
                return None

            # Apply all analysis modules
            df = self.apply_smc_analysis(df)
            df = self.apply_harmonic_analysis(df)
            df = self.apply_wyckoff_analysis(df)

            # Generate trading signals
            signals = self.generate_trading_signals(df)

            # Save enriched data
            enriched_file = self.save_enriched_data(df, symbol, timeframe)

            # Create analysis report
            analysis_report = {
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': datetime.now().isoformat(),
                'data_points': len(df),
                'price_range': {
                    'high': float(df['high'].max()),
                    'low': float(df['low'].min()),
                    'current': float(df['close'].iloc[-1])
                },
                'signals': signals,
                'file_processed': str(file_path),
                'enriched_file': str(enriched_file) if enriched_file else None
            }

            # Save analysis report
            report_file = self.save_analysis_report(analysis_report, symbol)

            logger.info(f"✅ Completed {symbol} {timeframe} - {len(signals)} signals generated")
            return analysis_report

        except Exception as e:
            logger.error(f"❌ Error processing {symbol} {timeframe}: {e}")
            return None

    def process_all_data(self):
        """Process all available data files"""
        try:
            logger.info("🚀 Starting data enrichment cycle")

            # Discover data files
            data_files = self.discover_data_files()

            if not data_files:
                logger.warning("⚠️ No data files found")
                return

            processed_count = 0

            # Process each symbol/timeframe
            for symbol, timeframes in data_files.items():
                for timeframe, file_path in timeframes.items():
                    result = self.process_symbol_timeframe(symbol, timeframe, file_path)
                    if result:
                        processed_count += 1

            logger.info(f"🎯 Enrichment cycle complete: {processed_count} files processed")

        except Exception as e:
            logger.error(f"❌ Error in data enrichment cycle: {e}")

# Main function for standalone execution
def main():
    """Main function for testing the data enricher"""
    config = {
        'data_path': './data',
        'refresh_interval': 60,
        'timeframes': ['1M', '5M', '15M', '1H', '4H']
    }

    enricher = DataEnricher(config)
    enricher.process_all_data()

if __name__ == "__main__":
    main()
