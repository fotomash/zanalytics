#!/usr/bin/env python3
"""
ZANFLOW Bridge - Connects MT5 SMC/Wyckoff data with Zanalytics engines
Fixed version with proper import paths
"""

import redis
import json
import pandas as pd
from datetime import datetime
import time
import sys
import os
import numpy as np

# Add the correct paths to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
zanalytics_path = os.path.join(script_dir, 'zanalytics-main')

# Try multiple possible paths
possible_paths = [
    zanalytics_path,
    os.path.join(script_dir, '..'),  # Parent directory
    script_dir,  # Current directory
    '/Users/tom/Documents/zanalytics',  # Your actual path
    '/Users/tom/Documents/zanalytics/zanalytics-main'
]

for path in possible_paths:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)
        print(f"✅ Added to path: {path}")

# Now try to import the engines
engines_loaded = {}

try:
    from core.analysis.smc_enrichment_engine import tag_smc_zones
    engines_loaded['smc'] = True
    print("✅ Loaded SMC enrichment engine")
except ImportError as e:
    print(f"⚠️ Could not load SMC engine: {e}")
    engines_loaded['smc'] = False

try:
    from core.wyckoff_phase_engine import tag_wyckoff_phases
    engines_loaded['wyckoff'] = True
    print("✅ Loaded Wyckoff phase engine")
except ImportError as e:
    print(f"⚠️ Could not load Wyckoff engine: {e}")
    engines_loaded['wyckoff'] = False

try:
    from core.orchestrators.main_orchestrator import MainOrchestrator
    engines_loaded['orchestrator'] = True
    print("✅ Loaded Main orchestrator")
except ImportError as e:
    print(f"⚠️ Could not load Main orchestrator: {e}")
    engines_loaded['orchestrator'] = False

try:
    from core.strategies.advanced_smc import run_advanced_smc_strategy
    engines_loaded['advanced_smc'] = True
    print("✅ Loaded Advanced SMC strategy")
except ImportError as e:
    print(f"⚠️ Could not load Advanced SMC: {e}")
    engines_loaded['advanced_smc'] = False

print(f"\n📊 Engines loaded: {sum(engines_loaded.values())}/{len(engines_loaded)}")

class ZanflowBridge:
    def __init__(self, redis_host='localhost', redis_port=6379):
        self.r = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.orchestrator = None
        self.engines_loaded = engines_loaded

        if engines_loaded['orchestrator']:
            try:
                self.orchestrator = MainOrchestrator()
                print("✅ Main orchestrator initialized")
            except Exception as e:
                print(f"⚠️ Orchestrator init failed: {e}")
        else:
            print("⚠️ Running without orchestrator (not loaded)")

    def get_mt5_data(self, symbol):
        """Get latest MT5 data with SMC/Wyckoff analysis"""
        try:
            data = self.r.get(f"mt5:{symbol}:latest")
            if data:
                return json.loads(data)
        except:
            pass
        return None

    def get_tick_history(self, symbol, limit=100):
        """Get historical ticks for analysis"""
        try:
            ticks = self.r.lrange(f"mt5:{symbol}:ticks", 0, limit-1)
            return [json.loads(t) for t in ticks]
        except:
            return []

    def process_with_zanalytics(self, symbol):
        """Process MT5 data through available zanalytics engines"""
        # Get tick history
        ticks = self.get_tick_history(symbol, 100)
        if not ticks:
            return None

        # Convert to DataFrame
        df_data = []
        for tick in ticks:
            row = {
                'Time': datetime.fromtimestamp(tick['timestamp']),
                'Open': tick['bid'],
                'High': tick['bid'] + tick['spread'] * 0.00001,
                'Low': tick['bid'] - tick['spread'] * 0.00001,
                'Close': tick['ask'],
                'Volume': tick.get('volume', 100),
                'tick_volume': tick.get('volume', 100)
            }

            # Add MT5 calculated values
            if 'smc' in tick:
                row['mt5_market_structure'] = tick['smc']['market_structure']
                row['mt5_has_order_block'] = tick['smc']['has_order_block']
                row['mt5_has_fvg'] = tick['smc']['has_fvg']

            if 'wyckoff' in tick:
                row['mt5_wyckoff_phase'] = tick['wyckoff']['phase']
                row['mt5_wyckoff_event'] = tick['wyckoff']['event']

            df_data.append(row)

        if not df_data:
            return None

        df = pd.DataFrame(df_data)
        df.set_index('Time', inplace=True)

        result = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'mt5_analysis': ticks[-1] if ticks else {},
            'zanalytics_enhanced': {}
        }

        # Apply available engines
        try:
            # SMC Engine
            if self.engines_loaded['smc']:
                try:
                    df = tag_smc_zones(df, tf='M1')
                    result['zanalytics_enhanced']['smc_zones'] = True
                    print(f"✅ Applied SMC analysis to {symbol}")
                except Exception as e:
                    print(f"⚠️ SMC analysis failed: {e}")

            # Wyckoff Engine
            if self.engines_loaded['wyckoff']:
                try:
                    df = tag_wyckoff_phases(df, 'M1')
                    result['zanalytics_enhanced']['wyckoff_phases'] = True
                    print(f"✅ Applied Wyckoff analysis to {symbol}")
                except Exception as e:
                    print(f"⚠️ Wyckoff analysis failed: {e}")

            # Advanced Strategy
            if self.engines_loaded['advanced_smc'] and self.orchestrator:
                try:
                    all_tf_data = {'M1': df}
                    strategy_result = run_advanced_smc_strategy(
                        all_tf_data,
                        strategy_variant='aggressive',
                        target_timestamp=df.index[-1],
                        symbol=symbol
                    )
                    result['zanalytics_enhanced']['strategy'] = strategy_result
                    print(f"✅ Applied advanced strategy to {symbol}")
                except Exception as e:
                    print(f"⚠️ Strategy analysis failed: {e}")

            # If no engines loaded, use simplified analysis
            if not any(self.engines_loaded.values()):
                print("ℹ️ Using simplified analysis (no engines loaded)")
                result['zanalytics_enhanced'] = self._simplified_analysis(ticks[-1])

        except Exception as e:
            print(f"Error in processing: {e}")
            result['zanalytics_enhanced'] = self._simplified_analysis(ticks[-1])

        return result

    def _simplified_analysis(self, tick_data):
        """Fallback simplified analysis when engines not available"""
        if not tick_data:
            return {}

        score = 0
        reasons = []

        # Analyze MT5 SMC data
        if 'smc' in tick_data:
            smc = tick_data['smc']
            if smc.get('market_structure') == 'Bullish':
                score += 1
                reasons.append("Bullish structure")
            elif smc.get('market_structure') == 'Bearish':
                score -= 1
                reasons.append("Bearish structure")

            if smc.get('has_order_block'):
                if smc.get('order_block_type') == 'Bullish':
                    score += 2
                    reasons.append("Bullish OB")
                else:
                    score -= 2
                    reasons.append("Bearish OB")

        # Analyze Wyckoff
        if 'wyckoff' in tick_data:
            wyckoff = tick_data['wyckoff']
            if wyckoff.get('phase') == 'Accumulation':
                score += 1
                reasons.append("Accumulation")
            elif wyckoff.get('phase') == 'Distribution':
                score -= 1
                reasons.append("Distribution")

        # Determine signal
        if score >= 2:
            signal = 'buy'
        elif score <= -2:
            signal = 'sell'
        else:
            signal = 'hold'

        return {
            'signal': signal,
            'score': score,
            'confidence': min(abs(score) / 4.0, 1.0),
            'reasons': reasons,
            'simplified': True
        }

    def monitor_symbols(self, symbols=None):
        """Monitor symbols and process through zanalytics"""
        print(f"\n🔍 Monitoring symbols...")
        print(f"Engines available: {list(k for k,v in self.engines_loaded.items() if v)}")

        while True:
            try:
                # Get all available symbols
                keys = self.r.keys("mt5:*:latest")
                current_symbols = set()

                for key in keys:
                    parts = key.split(":")
                    if len(parts) >= 2:
                        current_symbols.add(parts[1])

                if current_symbols:
                    print(f"\n📊 Processing {len(current_symbols)} symbols: {', '.join(current_symbols)}")

                    for symbol in current_symbols:
                        result = self.process_with_zanalytics(symbol)

                        if result:
                            enhanced = result.get('zanalytics_enhanced', {})
                            mt5_data = result.get('mt5_analysis', {})

                            # Check for signals
                            signal = None
                            if 'strategy' in enhanced:
                                signal = enhanced['strategy'].get('action', 'hold')
                            elif 'signal' in enhanced:
                                signal = enhanced['signal']

                            if signal and signal != 'hold':
                                print(f"\n{'='*60}")
                                print(f"🎯 {symbol} - Signal: {signal.upper()}")

                                if 'smc' in mt5_data:
                                    print(f"📊 MT5 Structure: {mt5_data['smc']['market_structure']}")
                                if 'wyckoff' in mt5_data:
                                    print(f"🎲 MT5 Wyckoff: {mt5_data['wyckoff']['phase']}")

                                if enhanced.get('simplified'):
                                    print(f"💡 Analysis: Simplified (score: {enhanced.get('score', 0)})")
                                else:
                                    print(f"🧠 Analysis: Full Zanalytics")

                            # Store enhanced analysis
                            self.r.set(
                                f"zanalytics:{symbol}:enhanced",
                                json.dumps(result, default=str),
                                ex=3600
                            )
                else:
                    print("⏳ Waiting for MT5 data...")

            except Exception as e:
                print(f"❌ Error in monitoring loop: {e}")

            time.sleep(2)

def main():
    """Main entry point"""
    print("🚀 Starting ZANFLOW Bridge (Fixed Version)...")
    print("This connects MT5 SMC/Wyckoff data with your Zanalytics engines")

    # Show current working directory
    print(f"\n📁 Current directory: {os.getcwd()}")
    print(f"📂 Script directory: {os.path.dirname(os.path.abspath(__file__))}")

    bridge = ZanflowBridge()

    try:
        bridge.r.ping()
        print("\n✅ Redis connection successful")
    except:
        print("\n❌ Cannot connect to Redis. Make sure Redis is running!")
        return

    bridge.monitor_symbols()

if __name__ == "__main__":
    main()
