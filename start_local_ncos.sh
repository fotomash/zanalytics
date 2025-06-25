#!/bin/bash
# start_local_ncos.sh - Quick start script for local ncOS engine

echo "\U1F680 Starting ncOS Local System (No API Keys)"
echo "=========================================="

# Install minimal requirements
pip install flask flask-cors pandas numpy python-dotenv >/dev/null

# Start the local engine
python ncos_local_engine.py &
ENGINE_PID=$!

# Wait for server to start
sleep 2

# Test the connection
echo "Testing local server..."
curl -s http://localhost:8000/status || {
  echo "Local engine did not respond";
  kill $ENGINE_PID;
  exit 1;
}

echo ""
echo "\xE2\x9C\x85 ncOS Local Engine is running!"
echo "\xF0\x9F\x93\xA1 Local: http://localhost:8000"
echo "\xF0\x9F\x94\xA7 No API keys required - everything runs locally"
