
import os
import time
import json
import threading
import pandas as pd
from flask import Flask, request, jsonify
import redis

# Import analysis modules with proper error handling
try:
    import zanflow_microstructure_analyzer as zma
    zma_available = True
    print("✅ zanflow_microstructure_analyzer loaded successfully")
except ImportError:
    zma_available = False
    print("❌ Warning: zanflow_microstructure_analyzer not found")

try:
    from convert_final_enhanced_smc_ULTIMATE import analyze_smc
    smc_available = True
    print("✅ Enhanced SMC processor loaded successfully")
except ImportError:
    smc_available = False
    print("❌ Warning: Enhanced SMC processor not found")

# Configure Redis
try:
    r = redis.Redis(host='localhost', port=6379, db=0)
    r.ping()  # Test connection
    redis_available = True
    print("✅ Redis connection successful")
except Exception as e:
    redis_available = False
    print(f"❌ Redis connection failed: {e}")
    print("Using fallback memory storage")

# Fallback storage if Redis is unavailable
memory_storage = {
    'symbols': {},
    'last_update': {}
}

app = Flask("multi_asset_tick_processor")

# Utility functions
def store_tick_data(symbol, tick_data):
    """Store tick data in Redis or memory fallback"""
    if redis_available:
        # Store the latest 5000 ticks for each symbol
        r.lpush(f"ticks:{symbol}", json.dumps(tick_data))
        r.ltrim(f"ticks:{symbol}", 0, 4999)
        r.hset("last_update", symbol, int(time.time()))
    else:
        # Memory fallback
        if symbol not in memory_storage['symbols']:
            memory_storage['symbols'][symbol] = []
        memory_storage['symbols'][symbol].append(tick_data)
        memory_storage['symbols'][symbol] = memory_storage['symbols'][symbol][-5000:]
        memory_storage['last_update'][symbol] = int(time.time())

def get_tick_data(symbol, limit=100):
    """Retrieve tick data from Redis or memory fallback"""
    if redis_available:
        ticks = r.lrange(f"ticks:{symbol}", 0, limit - 1)
        return [json.loads(tick) for tick in ticks]
    else:
        if symbol in memory_storage['symbols']:
            return memory_storage['symbols'][symbol][-limit:]
        return []

def store_analysis_result(symbol, analysis_type, result):
    """Store analysis result in Redis or memory fallback"""
    if redis_available:
        r.hset(f"analysis:{symbol}", analysis_type, json.dumps(result))
        r.hset("last_analysis", symbol, int(time.time()))
    else:
        if 'analysis' not in memory_storage:
            memory_storage['analysis'] = {}
        if symbol not in memory_storage['analysis']:
            memory_storage['analysis'][symbol] = {}
        memory_storage['analysis'][symbol][analysis_type] = result
        if 'last_analysis' not in memory_storage:
            memory_storage['last_analysis'] = {}
        memory_storage['last_analysis'][symbol] = int(time.time())

def get_analysis_result(symbol, analysis_type=None):
    """Retrieve analysis result from Redis or memory fallback"""
    if redis_available:
        if analysis_type:
            result = r.hget(f"analysis:{symbol}", analysis_type)
            return json.loads(result) if result else None
        else:
            result = r.hgetall(f"analysis:{symbol}")
            return {k.decode(): json.loads(v) for k, v in result.items()} if result else {}
    else:
        if 'analysis' in memory_storage and symbol in memory_storage['analysis']:
            if analysis_type:
                return memory_storage['analysis'][symbol].get(analysis_type)
            else:
                return memory_storage['analysis'][symbol]
        return {} if analysis_type is None else None

def run_tick_analysis(symbol, ticks):
    """Run analysis on tick data using available analyzers"""
    results = {
        'symbol': symbol,
        'timestamp': time.time(),
        'tick_count': len(ticks)
    }

    # Convert ticks to dataframe for analysis
    df = pd.DataFrame(ticks)

    # Run ZanFlow Microstructure Analysis if available
    if zma_available and len(df) > 10:
        try:
            # Assuming zma.analyze_ticks function exists
            zma_results = zma.analyze_ticks(df)
            results['microstructure'] = zma_results
            print(f"✅ Microstructure analysis complete for {symbol}")
        except Exception as e:
            print(f"❌ Error in microstructure analysis: {e}")
            results['microstructure'] = {'error': str(e)}

    # Run SMC Analysis if available
    if smc_available and len(df) > 10:
        try:
            # Assuming analyze_smc function works with our tick data
            smc_results = analyze_smc(df)
            results['smc'] = smc_results
            print(f"✅ SMC analysis complete for {symbol}")
        except Exception as e:
            print(f"❌ Error in SMC analysis: {e}")
            results['smc'] = {'error': str(e)}

    # Store the analysis results
    store_analysis_result(symbol, 'latest', results)

    return results

# Flask routes
@app.route('/webhook', methods=['POST'])
def webhook():
    """Endpoint for receiving tick data from MT5"""
    data = request.get_json()

    if not data:
        return jsonify({'status': 'error', 'message': 'No data provided'}), 400

    required_fields = ['symbol', 'time_msc', 'bid', 'ask']
    if not all(field in data for field in required_fields):
        return jsonify({'status': 'error', 'message': 'Missing required fields'}), 400

    symbol = data['symbol']
    store_tick_data(symbol, data)

    # If we have enough ticks, trigger analysis in a background thread
    ticks = get_tick_data(symbol, 250)  # Use 250 ticks for analysis
    if len(ticks) >= 250:
        threading.Thread(target=run_tick_analysis, args=(symbol, ticks)).start()

    return jsonify({'status': 'success', 'message': f'Tick data received for {symbol}'}), 200

@app.route('/analysis/<symbol>', methods=['GET'])
def get_analysis(symbol):
    """Endpoint for retrieving analysis results"""
    analysis = get_analysis_result(symbol)
    if analysis:
        return jsonify({'status': 'success', 'analysis': analysis}), 200
    else:
        return jsonify({'status': 'error', 'message': 'No analysis data available'}), 404

@app.route('/summary', methods=['GET'])
def get_summary():
    """Endpoint for retrieving a summary of all symbols"""
    if redis_available:
        symbols = set()
        for key in r.keys("ticks:*"):
            symbols.add(key.decode().split(':')[1])

        summary = {}
        for symbol in symbols:
            last_update = r.hget("last_update", symbol)
            last_update = int(last_update) if last_update else 0

            last_analysis = r.hget("last_analysis", symbol)
            last_analysis = int(last_analysis) if last_analysis else 0

            tick_count = r.llen(f"ticks:{symbol}")

            analysis = get_analysis_result(symbol, 'latest')

            summary[symbol] = {
                'last_update': last_update,
                'last_analysis': last_analysis,
                'tick_count': tick_count,
                'has_analysis': analysis is not None
            }
    else:
        summary = {}
        for symbol in memory_storage['symbols']:
            summary[symbol] = {
                'last_update': memory_storage['last_update'].get(symbol, 0),
                'last_analysis': memory_storage.get('last_analysis', {}).get(symbol, 0),
                'tick_count': len(memory_storage['symbols'][symbol]),
                'has_analysis': symbol in memory_storage.get('analysis', {})
            }

    return jsonify({'status': 'success', 'summary': summary}), 200

@app.route('/trigger_analysis/<symbol>', methods=['POST'])
def trigger_analysis(symbol):
    """Endpoint for manually triggering analysis"""
    ticks = get_tick_data(symbol, 250)
    if len(ticks) < 10:
        return jsonify({'status': 'error', 'message': 'Not enough tick data for analysis'}), 400

    results = run_tick_analysis(symbol, ticks)
    return jsonify({'status': 'success', 'analysis': results}), 200

if __name__ == '__main__':
    print("""🚀 ZANFLOW Multi-Asset Tick Analysis Server
====
Endpoints:
  POST /webhook - Receive MT5 data
  GET  /analysis/<symbol> - Get analysis
  GET  /summary - Get all symbols summary
  POST /trigger_analysis/<symbol> - Trigger analysis
====
""")
    app.run(host='0.0.0.0', port=5000)
