#!/usr/bin/env python3
"""
ZANFLOW Bridge (Simplified) - Connects MT5 SMC/Wyckoff data without complex dependencies
"""

import redis
import json
import pandas as pd
from datetime import datetime
import time
import numpy as np

class SimplifiedZanflowBridge:
    def __init__(self, redis_host='localhost', redis_port=6379):
        self.r = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        print("✅ Connected to Redis")

    def get_mt5_data(self, symbol):
        """Get latest MT5 data with SMC/Wyckoff analysis"""
        try:
            data = self.r.get(f"mt5:{symbol}:latest")
            if data:
                return json.loads(data)
        except Exception as e:
            print(f"Error getting data for {symbol}: {e}")
        return None

    def get_tick_history(self, symbol, limit=100):
        """Get historical ticks for analysis"""
        try:
            ticks = self.r.lrange(f"mt5:{symbol}:ticks", 0, limit-1)
            return [json.loads(t) for t in ticks]
        except:
            return []

    def calculate_smc_signals(self, data):
        """Calculate trading signals based on SMC data"""
        if not data or 'smc' not in data:
            return None

        smc = data['smc']
        wyckoff = data.get('wyckoff', {})

        # Signal scoring system
        score = 0
        reasons = []

        # SMC Analysis
        if smc.get('market_structure') == 'Bullish':
            score += 1
            reasons.append("Bullish structure")
        elif smc.get('market_structure') == 'Bearish':
            score -= 1
            reasons.append("Bearish structure")

        # Order Blocks
        if smc.get('has_order_block'):
            ob_type = smc.get('order_block_type', '')
            if ob_type == 'Bullish':
                score += 2
                reasons.append("Bullish OB")
            elif ob_type == 'Bearish':
                score -= 2
                reasons.append("Bearish OB")

        # Fair Value Gaps
        if smc.get('has_fvg'):
            fvg_type = smc.get('fvg_type', '')
            if fvg_type == 'Bullish':
                score += 1
                reasons.append("Bullish FVG")
            elif fvg_type == 'Bearish':
                score -= 1
                reasons.append("Bearish FVG")

        # Liquidity Sweeps
        if smc.get('has_liquidity_sweep'):
            sweep_type = smc.get('sweep_type', '')
            if sweep_type == 'BSL':  # Buy Side Liquidity
                score -= 1  # Potential reversal down
                reasons.append("BSL swept")
            elif sweep_type == 'SSL':  # Sell Side Liquidity
                score += 1  # Potential reversal up
                reasons.append("SSL swept")

        # Wyckoff Analysis
        if wyckoff:
            phase = wyckoff.get('phase', '')
            event = wyckoff.get('event', '')

            if phase == 'Accumulation':
                score += 1
                reasons.append("Accumulation phase")
            elif phase == 'Distribution':
                score -= 1
                reasons.append("Distribution phase")

            # Strong events
            if event in ['Spring', 'Test']:
                score += 2
                reasons.append(f"Wyckoff {event}")
            elif event in ['UTAD', 'SOW']:
                score -= 2
                reasons.append(f"Wyckoff {event}")

        # Determine signal
        if score >= 3:
            signal = 'strong_buy'
        elif score >= 2:
            signal = 'buy'
        elif score <= -3:
            signal = 'strong_sell'
        elif score <= -2:
            signal = 'sell'
        else:
            signal = 'hold'

        confidence = min(abs(score) / 5.0, 1.0)  # Max confidence at score 5

        return {
            'signal': signal,
            'score': score,
            'confidence': confidence,
            'reasons': reasons
        }

    def process_symbol(self, symbol):
        """Process a symbol and generate analysis"""
        data = self.get_mt5_data(symbol)
        if not data:
            return None

        # Calculate signals
        signals = self.calculate_smc_signals(data)

        if signals:
            result = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'price': {
                    'bid': data.get('bid', 0),
                    'ask': data.get('ask', 0),
                    'spread': data.get('spread', 0)
                },
                'smc': data.get('smc', {}),
                'wyckoff': data.get('wyckoff', {}),
                'analysis': signals
            }

            # Store enhanced analysis
            self.r.set(
                f"zanalytics:{symbol}:enhanced",
                json.dumps(result),
                ex=3600  # 1 hour expiry
            )

            return result

        return None

    def monitor_symbols(self, symbols=None):
        """Monitor symbols and process analysis"""
        print("🔍 Scanning for available symbols...")

        while True:
            # Get all available symbols
            keys = self.r.keys("mt5:*:latest")
            current_symbols = set()

            for key in keys:
                parts = key.split(":")
                if len(parts) >= 2:
                    current_symbols.add(parts[1])

            if current_symbols:
                print(f"
📊 Found {len(current_symbols)} active symbols: {', '.join(current_symbols)}")

                for symbol in current_symbols:
                    result = self.process_symbol(symbol)

                    if result and result['analysis']['signal'] != 'hold':
                        analysis = result['analysis']
                        print(f"
{'='*60}")
                        print(f"🎯 {symbol} - Signal: {analysis['signal'].upper()}")
                        print(f"📊 Score: {analysis['score']} | Confidence: {analysis['confidence']:.1%}")
                        print(f"📝 Reasons: {', '.join(analysis['reasons'])}")

                        # Show current market state
                        smc = result['smc']
                        wyckoff = result['wyckoff']
                        print(f"🏗️ Structure: {smc.get('market_structure', 'Unknown')}")
                        print(f"📈 Wyckoff: {wyckoff.get('phase', 'Unknown')} - {wyckoff.get('event', 'None')}")
            else:
                print("⏳ Waiting for MT5 data... Make sure EA is running and sending data.")

            time.sleep(2)  # Check every 2 seconds

def main():
    """Main entry point"""
    print("🚀 Starting Simplified ZANFLOW Bridge...")
    print("This analyzes MT5 SMC/Wyckoff data and generates trading signals")
    print("
💡 Make sure:")
    print("1. Redis is running")
    print("2. API server is running (api_server_smc_wyckoff.py)")
    print("3. MT5 EA is attached to charts and sending data")

    bridge = SimplifiedZanflowBridge()

    try:
        # Test Redis connection
        bridge.r.ping()
        print("
✅ Redis connection successful")
    except:
        print("
❌ Cannot connect to Redis. Make sure Redis is running!")
        return

    # Start monitoring
    bridge.monitor_symbols()

if __name__ == "__main__":
    main()
