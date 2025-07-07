#!/usr/bin/env python3
"""
ZANFLOW Enhanced API Server
With data enrichment and Redis management
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import redis
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from collections import deque
import threading
import time

app = Flask(__name__)
CORS(app)

# Redis connection
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Data enrichment storage
symbol_data = {}

class SymbolEnrichment:
    """Enriches symbol data with calculated metrics"""

    def __init__(self, symbol):
        self.symbol = symbol
        self.price_history = deque(maxlen=1000)  # Keep last 1000 ticks
        self.volume_history = deque(maxlen=1000)
        self.timestamps = deque(maxlen=1000)

    def add_tick(self, bid, ask, volume, timestamp):
        """Add new tick data"""
        mid_price = (bid + ask) / 2
        self.price_history.append(mid_price)
        self.volume_history.append(volume)
        self.timestamps.append(timestamp)

    def calculate_metrics(self):
        """Calculate enriched metrics"""
        if len(self.price_history) < 2:
            return {}

        prices = list(self.price_history)
        volumes = list(self.volume_history)

        # Price metrics
        metrics = {
            'volatility_1min': self.calculate_volatility(60),
            'volatility_5min': self.calculate_volatility(300),
            'rsi': self.calculate_rsi(14),
            'momentum': self.calculate_momentum(10),
            'vwap': self.calculate_vwap(),
            'volume_profile': self.calculate_volume_profile(),
            'trend': self.calculate_trend()
        }

        return metrics

    def calculate_volatility(self, seconds):
        """Calculate volatility over time period"""
        now = time.time()
        recent_prices = []

        for i in range(len(self.timestamps)-1, -1, -1):
            if now - self.timestamps[i] <= seconds:
                recent_prices.append(self.price_history[i])
            else:
                break

        if len(recent_prices) > 1:
            return float(np.std(recent_prices))
        return 0

    def calculate_rsi(self, period=14):
        """Calculate RSI"""
        if len(self.price_history) < period + 1:
            return 50  # Neutral

        prices = list(self.price_history)[-period-1:]
        deltas = np.diff(prices)
        gains = deltas[deltas > 0].sum() / period
        losses = -deltas[deltas < 0].sum() / period

        if losses == 0:
            return 100

        rs = gains / losses
        rsi = 100 - (100 / (1 + rs))
        return float(rsi)

    def calculate_momentum(self, period=10):
        """Calculate price momentum"""
        if len(self.price_history) < period:
            return 0

        current = self.price_history[-1]
        past = self.price_history[-period]
        return float((current - past) / past * 100)

    def calculate_vwap(self):
        """Calculate Volume Weighted Average Price"""
        if len(self.price_history) < 1:
            return 0

        # Last 100 ticks
        prices = list(self.price_history)[-100:]
        volumes = list(self.volume_history)[-100:]

        if sum(volumes) == 0:
            return float(np.mean(prices))

        return float(np.sum(np.array(prices) * np.array(volumes)) / np.sum(volumes))

    def calculate_volume_profile(self):
        """Calculate volume at different price levels"""
        if len(self.price_history) < 10:
            return {}

        prices = list(self.price_history)[-100:]
        volumes = list(self.volume_history)[-100:]

        # Create price bins
        min_price = min(prices)
        max_price = max(prices)
        bins = np.linspace(min_price, max_price, 10)

        profile = {}
        for i in range(len(bins)-1):
            level = f"{bins[i]:.5f}-{bins[i+1]:.5f}"
            vol = sum(v for p, v in zip(prices, volumes) if bins[i] <= p < bins[i+1])
            profile[level] = float(vol)

        return profile

    def calculate_trend(self):
        """Determine current trend"""
        if len(self.price_history) < 20:
            return "NEUTRAL"

        # Simple MA crossover
        ma_fast = np.mean(list(self.price_history)[-10:])
        ma_slow = np.mean(list(self.price_history)[-20:])

        if ma_fast > ma_slow * 1.0001:  # 0.01% threshold
            return "UPTREND"
        elif ma_fast < ma_slow * 0.9999:
            return "DOWNTREND"
        else:
            return "NEUTRAL"

@app.route('/webhook', methods=['POST'])
def webhook():
    """Enhanced webhook with data enrichment"""
    try:
        data = request.get_json()

        # Handle multi-symbol data
        if 'symbols' in data:
            for symbol_data_item in data['symbols']:
                process_symbol_data(symbol_data_item)
        else:
            # Single symbol data
            process_symbol_data(data)

        # Store account data
        if 'account' in data:
            r.hset('account:info', mapping=data['account'])

        return jsonify({"status": "ok"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def process_symbol_data(data):
    """Process and enrich symbol data"""
    symbol = data.get('symbol')
    if not symbol:
        return

    # Initialize enrichment if needed
    if symbol not in symbol_data:
        symbol_data[symbol] = SymbolEnrichment(symbol)

    # Add tick data
    enrichment = symbol_data[symbol]
    enrichment.add_tick(
        float(data.get('bid', 0)),
        float(data.get('ask', 0)),
        float(data.get('volume', {}).get('tick', 0)),
        time.time()
    )

    # Calculate enriched metrics
    metrics = enrichment.calculate_metrics()
    data['enriched'] = metrics

    # Store in Redis with TTL
    r.setex(f"mt5:{symbol}:latest", 300, json.dumps(data))

    # Store in history
    r.lpush(f"mt5:{symbol}:history", json.dumps(data))
    r.ltrim(f"mt5:{symbol}:history", 0, 999)  # Keep last 1000

    # Store enriched data separately
    r.setex(f"mt5:{symbol}:enriched", 300, json.dumps(metrics))

    # Publish to Redis pub/sub for real-time updates
    r.publish(f"mt5:{symbol}", json.dumps(data))

@app.route('/symbols', methods=['GET'])
def get_symbols():
    """Get all available symbols"""
    keys = r.keys("mt5:*:latest")
    symbols = [key.split(':')[1] for key in keys]
    return jsonify(symbols)

@app.route('/data/<symbol>', methods=['GET'])
def get_symbol_data(symbol):
    """Get enriched data for a symbol"""
    latest = r.get(f"mt5:{symbol}:latest")
    enriched = r.get(f"mt5:{symbol}:enriched")

    if latest:
        data = json.loads(latest)
        if enriched:
            data['enriched'] = json.loads(enriched)
        return jsonify(data)

    return jsonify({"error": "Symbol not found"}), 404

@app.route('/redis/info', methods=['GET'])
def redis_info():
    """Get Redis storage information"""
    info = {
        'keys': len(r.keys()),
        'memory': r.info('memory')['used_memory_human'],
        'symbols': {},
        'ttl_info': {}
    }

    # Get info for each symbol
    for key in r.keys("mt5:*:latest"):
        symbol = key.split(':')[1]
        info['symbols'][symbol] = {
            'history_length': r.llen(f"mt5:{symbol}:history"),
            'ttl': r.ttl(key)
        }

    return jsonify(info)

@app.route('/redis/cleanup', methods=['POST'])
def redis_cleanup():
    """Clean up old Redis data"""
    deleted = 0

    # Delete old history entries
    for key in r.keys("mt5:*:history"):
        # Keep only last 1000 entries
        r.ltrim(key, 0, 999)
        deleted += 1

    return jsonify({"cleaned": deleted})

if __name__ == '__main__':
    print("""
    🚀 ZANFLOW Enhanced API Server
    ==============================
    Endpoints:
    - POST /webhook - Receive MT5 data
    - GET /symbols - List all symbols
    - GET /data/<symbol> - Get enriched symbol data
    - GET /redis/info - Redis storage info
    - POST /redis/cleanup - Clean old data

    Redis Data Structure:
    - mt5:{symbol}:latest - Latest tick data (5min TTL)
    - mt5:{symbol}:history - Historical data (last 1000)
    - mt5:{symbol}:enriched - Calculated metrics
    - account:info - Account information
    """)

    app.run(host='0.0.0.0', port=8080, debug=True)
