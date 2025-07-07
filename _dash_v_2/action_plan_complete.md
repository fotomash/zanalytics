# 🎯 ZANFLOW Action Plan - Step by Step

## Step 1: Prepare Your Environment
```bash
# Install Redis (if not installed)
brew install redis  # macOS
# or
sudo apt-get install redis-server  # Ubuntu

# Install Python dependencies
pip install redis flask flask-cors streamlit pandas numpy plotly pyyaml scikit-learn requests
```

## Step 2: Navigate to the Correct Directory
```bash
cd ~/Documents/GitHub/zanalytics/zanalytics-main/_dash_v_2
```

## Step 3: Copy the New Startup Script
Copy `zanflow_start.py` to the `_dash_v_2` directory

## Step 4: Start the System
```bash
python zanflow_start.py
```

## Step 5: Setup MetaTrader
1. Copy `HttpsJsonSender_Simple.mq5` to MT5 Experts folder
2. Update URL to `http://localhost:5000/webhook`
3. Compile and attach to chart
4. Enable AutoTrading

## Step 6: Verify Everything Works
- Dashboard: http://localhost:8501
- API health: http://localhost:5000/status
- Redis: `redis-cli ping` (should return PONG)

## Troubleshooting Commands
```bash
# Check if Redis is running
redis-cli ping

# Check if ports are in use
lsof -i :5000  # API port
lsof -i :8501  # Dashboard port
lsof -i :6379  # Redis port

# Kill processes if needed
pkill -f "streamlit"
pkill -f "api_server.py"
```

## What Each Component Does
- **Redis**: Real-time message broker for tick data
- **API Server**: Receives data from MT5, processes it, sends to Redis
- **Dashboard**: Reads from Redis, displays analysis and charts

## Common Issues & Fixes
1. **Import Error**: Run from _dash_v_2 directory
2. **Port in use**: Kill existing processes or change ports
3. **MT5 Error 4014**: Check API is running and URL is correct
4. **Redis Connection Error**: Start Redis manually first
