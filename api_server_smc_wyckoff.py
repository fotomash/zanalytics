from flask import Flask, request, jsonify
import redis
import json
from datetime import datetime
import threading
import time

app = Flask(__name__)

# Redis connection
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Store for analysis results
analysis_cache = {}

@app.route('/webhook', methods=['POST'])
def webhook():
    """Receive data from MT5 EA with SMC/Wyckoff analysis"""
    try:
        data = request.json

        # Add timestamp if not present
        if 'timestamp' not in data:
            data['timestamp'] = time.time()

        symbol = data.get('symbol', 'UNKNOWN')

        # Store in Redis with different keys for different data types
        # 1. Latest complete data
        r.set(f"mt5:{symbol}:latest", json.dumps(data))

        # 2. Tick history
        r.lpush(f"mt5:{symbol}:ticks", json.dumps(data))
        r.ltrim(f"mt5:{symbol}:ticks", 0, 999)

        # 3. SMC specific data
        if 'smc' in data:
            r.set(f"mt5:{symbol}:smc", json.dumps(data['smc']))

        # 4. Wyckoff specific data
        if 'wyckoff' in data:
            r.set(f"mt5:{symbol}:wyckoff", json.dumps(data['wyckoff']))

        # Set expiry
        r.expire(f"mt5:{symbol}:latest", 86400)
        r.expire(f"mt5:{symbol}:ticks", 86400)
        r.expire(f"mt5:{symbol}:smc", 86400)
        r.expire(f"mt5:{symbol}:wyckoff", 86400)

        # Log analysis
        if 'smc' in data:
            smc = data['smc']
            print(f"📊 {symbol} - Structure: {smc['market_structure']}", end="")
            if smc['has_order_block']:
                print(f" | OB: {smc['order_block_type']}", end="")
            if smc['has_fvg']:
                print(f" | FVG: {smc['fvg_type']}", end="")
            if smc['has_liquidity_sweep']:
                print(f" | Sweep: {smc['sweep_type']}", end="")
            print()

        if 'wyckoff' in data:
            wyckoff = data['wyckoff']
            print(f"🎯 {symbol} - Phase: {wyckoff['phase']} | Event: {wyckoff['event']}")

        return jsonify({"status": "success", "symbol": symbol}), 200

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/analysis/<symbol>', methods=['GET'])
def get_analysis(symbol):
    """Get complete analysis for a symbol"""
    try:
        result = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat()
        }

        # Get latest data
        latest = r.get(f"mt5:{symbol}:latest")
        if latest:
            result['latest'] = json.loads(latest)

        # Get SMC analysis
        smc = r.get(f"mt5:{symbol}:smc")
        if smc:
            result['smc'] = json.loads(smc)

        # Get Wyckoff analysis
        wyckoff = r.get(f"mt5:{symbol}:wyckoff")
        if wyckoff:
            result['wyckoff'] = json.loads(wyckoff)

        # Get enhanced analysis from bridge
        enhanced = r.get(f"zanalytics:{symbol}:enhanced")
        if enhanced:
            result['zanalytics'] = json.loads(enhanced)

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/signals', methods=['GET'])
def get_signals():
    """Get trading signals from all symbols"""
    try:
        signals = []

        # Get all symbols with enhanced analysis
        keys = r.keys("zanalytics:*:enhanced")

        for key in keys:
            data = r.get(key)
            if data:
                analysis = json.loads(data)
                if analysis.get('zanalytics_enhanced'):
                    enhanced = analysis['zanalytics_enhanced']
                    if enhanced.get('strategy_signal') != 'hold':
                        signals.append({
                            'symbol': analysis['symbol'],
                            'signal': enhanced['strategy_signal'],
                            'confidence': enhanced['confidence'],
                            'timestamp': analysis['timestamp']
                        })

        # Sort by confidence
        signals.sort(key=lambda x: x['confidence'], reverse=True)

        return jsonify({'signals': signals}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check with system status"""
    try:
        # Count active symbols
        mt5_symbols = len(r.keys("mt5:*:latest"))
        enhanced_symbols = len(r.keys("zanalytics:*:enhanced"))

        return jsonify({
            "status": "healthy",
            "redis": "connected" if r.ping() else "disconnected",
            "mt5_symbols": mt5_symbols,
            "enhanced_symbols": enhanced_symbols,
            "timestamp": datetime.now().isoformat()
        }), 200
    except:
        return jsonify({"status": "error"}), 500

if __name__ == '__main__':
    print("🚀 Starting Enhanced ZANFLOW API Server")
    print("📊 Endpoints:")
    print("   POST /webhook - Receive MT5 SMC/Wyckoff data")
    print("   GET  /analysis/<symbol> - Get complete analysis")
    print("   GET  /signals - Get trading signals")
    print("   GET  /health - System health check")
    print("
💡 Make sure to run zanflow_bridge.py for enhanced analysis!")

    app.run(host='0.0.0.0', port=5000, debug=True)
