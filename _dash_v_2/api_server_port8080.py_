#!/usr/bin/env python3
"""
ZANFLOW API Server - Port 8080 Version
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import redis
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Redis connection
try:
    r = redis.Redis(host='localhost', port=6379, decode_responses=True)
    r.ping()
    logger.info("✅ Connected to Redis")
except:
    logger.error("❌ Redis not connected - make sure Redis is running!")
    r = None

@app.route('/webhook', methods=['POST'])
def webhook():
    """Receive data from MT5"""
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No data received"}), 400

        # Add timestamp
        data['server_timestamp'] = datetime.now().isoformat()

        # Store in Redis if available
        if r:
            key = f"mt5:{data.get('symbol', 'unknown')}:latest"
            r.set(key, json.dumps(data))
            r.expire(key, 300)  # Expire after 5 minutes

            # Also store in a list for history
            history_key = f"mt5:{data.get('symbol', 'unknown')}:history"
            r.lpush(history_key, json.dumps(data))
            r.ltrim(history_key, 0, 99)  # Keep last 100 entries

        logger.info(f"✅ Received data for {data.get('symbol', 'unknown')}")

        return jsonify({"status": "ok", "message": "Data received"}), 200

    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "redis": "connected" if r and r.ping() else "disconnected",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/', methods=['GET'])
def home():
    """Home page"""
    return """
    <h1>ZANFLOW API Server - Port 8080</h1>
    <p>Status: Running</p>
    <p>Endpoints:</p>
    <ul>
        <li>POST /webhook - Receive MT5 data</li>
        <li>GET /health - Health check</li>
    </ul>
    """

if __name__ == '__main__':
    # CHANGE PORT HERE! 
    PORT = 8080  # Changed from 5000 to 8080

    print(f"""
    🚀 ZANFLOW API Server Starting on Port {PORT}
    =====================================
    URL: http://127.0.0.1:{PORT}
    Webhook: http://127.0.0.1:{PORT}/webhook

    Add this to MT5: 127.0.0.1:{PORT}
    """)

    app.run(host='0.0.0.0', port=PORT, debug=True)
