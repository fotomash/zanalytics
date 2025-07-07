#!/usr/bin/env python3
"""
ZANFLOW Bridge - Connects MT5 SMC/Wyckoff data with Zanalytics engines
"""

import redis
import json
import pandas as pd
from datetime import datetime
import time
import sys
import os

# Add zanalytics to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'zanalytics_extracted/zanalytics-main'))

# Import your engines
try:
    from core.analysis.smc_enrichment_engine import tag_smc_zones
    from core.wyckoff_phase_engine import tag_wyckoff_phases
    from core.orchestrators.main_orchestrator import MainOrchestrator
    from core.strategies.advanced_smc import run_advanced_smc_strategy
    print("✅ Successfully imported zanalytics engines")
except ImportError as e:
    print(f"⚠️ Could not import zanalytics engines: {e}")
    print("Using simplified processing instead")

class ZanflowBridge:
    def __init__(self, redis_host='localhost', redis_port=6379):
        self.r = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.orchestrator = None

        try:
            self.orchestrator = MainOrchestrator()
            print("✅ Main orchestrator initialized")
        except:
            print("⚠️ Running without orchestrator")

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
        """Process MT5 data through zanalytics engines"""
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
                'Volume': tick['volume']
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

        df = pd.DataFrame(df_data)
        df.set_index('Time', inplace=True)

        # Apply your advanced analysis
        try:
            # Tag with your SMC engine
            df = tag_smc_zones(df, tf='M1')

            # Tag with your Wyckoff engine
            df = tag_wyckoff_phases(df, 'M1')

            # Run advanced strategy
            if self.orchestrator:
                all_tf_data = {'M1': df}  # You can add more timeframes
                result = run_advanced_smc_strategy(
                    all_tf_data,
                    strategy_variant='aggressive',
                    target_timestamp=df.index[-1],
                    symbol=symbol
                )

                return {
                    'symbol': symbol,
                    'timestamp': datetime.now(),
                    'mt5_analysis': ticks[-1],
                    'zanalytics_enhanced': {
                        'smc_zones': df['bos'].iloc[-1] if 'bos' in df else None,
                        'wyckoff_phase': df['wyckoff_phase'].iloc[-1] if 'wyckoff_phase' in df else None,
                        'strategy_signal': result.get('action', 'hold'),
                        'confidence': result.get('confidence', 0)
                    }
                }
        except Exception as e:
            print(f"Error in advanced processing: {e}")

        return {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'mt5_analysis': ticks[-1],
            'zanalytics_enhanced': None
        }

    def monitor_symbols(self, symbols=None):
        """Monitor symbols and process through zanalytics"""
        if not symbols:
            # Get all available symbols
            keys = self.r.keys("mt5:*:latest")
            symbols = []
            for key in keys:
                parts = key.split(":")
                if len(parts) >= 2:
                    symbols.append(parts[1])

        print(f"Monitoring {len(symbols)} symbols...")

        while True:
            for symbol in symbols:
                result = self.process_with_zanalytics(symbol)

                if result and result['zanalytics_enhanced']:
                    enhanced = result['zanalytics_enhanced']
                    print(f"{'='*60}")
                    print(f"Symbol: {symbol}")
                    print(f"MT5 Structure: {result['mt5_analysis']['smc']['market_structure']}")
                    print(f"MT5 Wyckoff: {result['mt5_analysis']['wyckoff']['phase']}")

                    if enhanced:
                        print(f"Zanalytics Signal: {enhanced['strategy_signal']}")
                        print(f"Confidence: {enhanced['confidence']:.2%}")

                    # Store enhanced analysis back to Redis
                    self.r.set(
                        f"zanalytics:{symbol}:enhanced",
                        json.dumps(result, default=str),
                        ex=3600  # 1 hour expiry
                    )

            time.sleep(1)  # Process every second

def main():
    """Main entry point"""
    print("🚀 Starting ZANFLOW Bridge...")
    print("This connects MT5 SMC/Wyckoff data with your Zanalytics engines")

    bridge = ZanflowBridge()

    # You can specify symbols or let it auto-detect
    # bridge.monitor_symbols(['EURUSD', 'GBPUSD', 'XAUUSD'])
    bridge.monitor_symbols()  # Monitor all available

if __name__ == "__main__":
    main()
