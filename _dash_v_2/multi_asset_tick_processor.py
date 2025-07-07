#!/usr/bin/env python3
"""
ZANFLOW Multi-Asset Tick Analysis Integration
Processes tick data from MT5 EA and runs enrichment analysis
"""

import json
import redis
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import sys
import time
from flask import Flask, request, jsonify
import threading

# Add path to your tick analyzer
sys.path.append('.')

# Import your tick analysis modules
try:
    from zanflow_microstructure_analyzer import MicrostructureAnalyzer
except ImportError:
    print("Warning: zanflow_microstructure_analyzer not found")
    MicrostructureAnalyzer = None

try:
    from convert_final_enhanced_smc_ULTIMATE import TechnicalIndicatorEngine, ProcessingConfig
except ImportError:
    print("Warning: Enhanced SMC processor not found")
    TechnicalIndicatorEngine = None

app = Flask(__name__)
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

class MultiAssetTickProcessor:
    def __init__(self):
        self.symbols = {}
        self.tick_data_path = Path("MT5_TickData")
        self.tick_data_path.mkdir(exist_ok=True)
        self.analysis_results = {}

        # Initialize indicator engine if available
        if TechnicalIndicatorEngine:
            self.indicator_engine = TechnicalIndicatorEngine(use_tick_volume=True)
        else:
            self.indicator_engine = None

    def process_multi_asset_data(self, data):
        """Process incoming multi-asset tick data"""
        timestamp = data.get('timestamp', datetime.now().isoformat())

        for symbol_data in data.get('symbols', []):
            symbol = symbol_data['symbol']

            # Store in Redis for real-time access
            redis_key = f"mt5:{symbol}:latest"
            r.set(redis_key, json.dumps(symbol_data), ex=60)

            # Add to tick history
            tick_history_key = f"mt5:{symbol}:ticks"
            r.lpush(tick_history_key, json.dumps({
                'timestamp': time.time(),
                'bid': symbol_data['bid'],
                'ask': symbol_data['ask'],
                'spread': symbol_data['spread'],
                'volume': symbol_data.get('volume', 0)
            }))
            r.ltrim(tick_history_key, 0, 10000)  # Keep last 10000 ticks

            # Process enriched bars if available
            if 'enriched_bars' in symbol_data:
                self.process_enriched_bars(symbol, symbol_data['enriched_bars'])

            # Check for manipulation alerts
            if 'tick_analysis' in symbol_data:
                self.check_manipulation_alerts(symbol, symbol_data['tick_analysis'])

            # Update symbol tracking
            if symbol not in self.symbols:
                self.symbols[symbol] = {
                    'first_seen': timestamp,
                    'tick_count': 0,
                    'last_analysis': None
                }

            self.symbols[symbol]['last_update'] = timestamp
            self.symbols[symbol]['tick_count'] += 1

            # Trigger deep analysis every 1000 ticks
            if self.symbols[symbol]['tick_count'] % 1000 == 0:
                threading.Thread(
                    target=self.run_deep_analysis,
                    args=(symbol,)
                ).start()

    def process_enriched_bars(self, symbol, enriched_data):
        """Process 250 enriched bars data"""
        try:
            # Create DataFrame from enriched data
            bars = enriched_data.get('bars', 0)

            if bars > 0:
                # Store enrichment results
                enrichment_key = f"mt5:{symbol}:enriched"
                r.set(enrichment_key, json.dumps({
                    'timestamp': datetime.now().isoformat(),
                    'bars': bars,
                    'avg_close': enriched_data.get('avg_close'),
                    'volatility': enriched_data.get('volatility'),
                    'price_change': enriched_data.get('price_change'),
                    'change_percent': enriched_data.get('change_percent')
                }), ex=300)

                print(f"✅ {symbol}: Processed {bars} enriched bars")

        except Exception as e:
            print(f"❌ Error processing enriched bars for {symbol}: {e}")

    def check_manipulation_alerts(self, symbol, tick_analysis):
        """Check for manipulation alerts"""
        manipulation_score = tick_analysis.get('manipulation_score', 0)

        if manipulation_score > 50:
            alert = {
                'symbol': symbol,
                'type': 'HIGH_MANIPULATION',
                'score': manipulation_score,
                'details': tick_analysis,
                'timestamp': datetime.now().isoformat()
            }

            # Store alert
            alert_key = f"alerts:{symbol}:manipulation"
            r.lpush(alert_key, json.dumps(alert))
            r.ltrim(alert_key, 0, 100)

            print(f"🚨 {symbol}: High manipulation detected! Score: {manipulation_score}")

    def run_deep_analysis(self, symbol):
        """Run deep microstructure analysis using your modules"""
        try:
            print(f"🔍 Running deep analysis for {symbol}...")

            # Get tick file path
            tick_file = self.tick_data_path / f"{symbol}_ticks.csv"

            if not tick_file.exists():
                print(f"No tick file found for {symbol}")
                return

            # Run microstructure analysis
            if MicrostructureAnalyzer:
                analyzer = MicrostructureAnalyzer(
                    str(tick_file),
                    limit_ticks=5000,
                    export_json=True,
                    no_dashboard=True,
                    output_dir=str(self.tick_data_path)
                )

                if analyzer.run_full_analysis():
                    print(f"✅ {symbol}: Microstructure analysis complete")

                    # Store results
                    results_key = f"analysis:{symbol}:microstructure"
                    r.set(results_key, json.dumps({
                        'timestamp': datetime.now().isoformat(),
                        'results': analyzer.analysis_results
                    }), ex=3600)

            # Run indicator enrichment
            if self.indicator_engine:
                df = pd.read_csv(tick_file)
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)

                # Rename columns for processor
                df.rename(columns={
                    'bid': 'close',
                    'ask': 'high',
                    'bid': 'low',
                    'bid': 'open'
                }, inplace=True)

                # Calculate indicators
                enriched_df = self.indicator_engine.calculate_all_indicators(df)

                # Save enriched data
                enriched_file = self.tick_data_path / f"{symbol}_enriched.csv"
                enriched_df.to_csv(enriched_file)

                print(f"✅ {symbol}: Indicator enrichment complete")

            self.symbols[symbol]['last_analysis'] = datetime.now().isoformat()

        except Exception as e:
            print(f"❌ Deep analysis failed for {symbol}: {e}")

    def get_analysis_summary(self):
        """Get summary of all analyses"""
        summary = {
            'symbols_tracked': len(self.symbols),
            'symbols': {}
        }

        for symbol, data in self.symbols.items():
            # Get latest data from Redis
            latest = r.get(f"mt5:{symbol}:latest")
            latest_data = json.loads(latest) if latest else {}

            # Get analysis results
            analysis = r.get(f"analysis:{symbol}:microstructure")
            analysis_data = json.loads(analysis) if analysis else {}

            summary['symbols'][symbol] = {
                'tick_count': data['tick_count'],
                'last_update': data.get('last_update'),
                'last_analysis': data.get('last_analysis'),
                'current_bid': latest_data.get('bid'),
                'current_ask': latest_data.get('ask'),
                'manipulation_score': latest_data.get('tick_analysis', {}).get('manipulation_score', 0),
                'has_analysis': bool(analysis_data)
            }

        return summary

# Global processor instance
processor = MultiAssetTickProcessor()

@app.route('/webhook', methods=['POST'])
def webhook():
    """Receive multi-asset tick data from MT5"""
    try:
        data = request.json

        if data.get('type') == 'multi_asset':
            processor.process_multi_asset_data(data)
            return jsonify({'status': 'success', 'symbols_processed': len(data.get('symbols', []))})
        else:
            # Handle single asset for compatibility
            return jsonify({'status': 'success', 'type': 'single_asset'})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/analysis/<symbol>', methods=['GET'])
def get_analysis(symbol):
    """Get analysis results for a symbol"""
    try:
        # Get latest tick data
        latest = r.get(f"mt5:{symbol}:latest")
        latest_data = json.loads(latest) if latest else None

        # Get microstructure analysis
        analysis = r.get(f"analysis:{symbol}:microstructure")
        analysis_data = json.loads(analysis) if analysis else None

        # Get enriched data
        enriched = r.get(f"mt5:{symbol}:enriched")
        enriched_data = json.loads(enriched) if enriched else None

        # Get recent alerts
        alerts = r.lrange(f"alerts:{symbol}:manipulation", 0, 10)
        alert_data = [json.loads(alert) for alert in alerts]

        return jsonify({
            'symbol': symbol,
            'latest_tick': latest_data,
            'microstructure_analysis': analysis_data,
            'enriched_bars': enriched_data,
            'recent_alerts': alert_data,
            'tick_count': processor.symbols.get(symbol, {}).get('tick_count', 0)
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/summary', methods=['GET'])
def get_summary():
    """Get summary of all tracked symbols"""
    try:
        summary = processor.get_analysis_summary()
        return jsonify(summary)

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/trigger_analysis/<symbol>', methods=['POST'])
def trigger_analysis(symbol):
    """Manually trigger deep analysis for a symbol"""
    try:
        if symbol in processor.symbols:
            threading.Thread(
                target=processor.run_deep_analysis,
                args=(symbol,)
            ).start()
            return jsonify({
                'status': 'success',
                'message': f'Analysis triggered for {symbol}'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': f'Symbol {symbol} not found'
            }), 404

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

def run_periodic_analysis():
    """Run periodic analysis on all symbols"""
    while True:
        time.sleep(300)  # Every 5 minutes

        for symbol in processor.symbols:
            # Check if needs analysis
            last_analysis = processor.symbols[symbol].get('last_analysis')
            if last_analysis:
                last_time = datetime.fromisoformat(last_analysis)
                if datetime.now() - last_time < timedelta(minutes=30):
                    continue

            # Run analysis
            threading.Thread(
                target=processor.run_deep_analysis,
                args=(symbol,)
            ).start()

            time.sleep(10)  # Stagger analyses

if __name__ == '__main__':
    print("🚀 ZANFLOW Multi-Asset Tick Analysis Server")
    print("=" * 50)
    print("Endpoints:")
    print("  POST /webhook - Receive MT5 data")
    print("  GET  /analysis/<symbol> - Get analysis")
    print("  GET  /summary - Get all symbols summary")
    print("  POST /trigger_analysis/<symbol> - Trigger analysis")
    print("=" * 50)

    # Start periodic analysis thread
    analysis_thread = threading.Thread(target=run_periodic_analysis, daemon=True)
    analysis_thread.start()

    # Run server
    app.run(host='0.0.0.0', port=5000, debug=False)
