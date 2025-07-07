# 📚 ZANFLOW Complete Integration Guide

## 1. 🌐 Exposing Data via Ngrok

### Quick Setup:
```bash
# Install ngrok
pip install pyngrok

# Run the ngrok setup script
python ngrok_setup.py
```

### Manual Setup:
```bash
# Download ngrok from https://ngrok.com
# Run ngrok
ngrok http 8080

# You'll get a URL like: https://abc123.ngrok.io
# Update your MT5 EA with this URL
```

### Production Setup:
- Use a VPS with static IP
- Set up proper domain name
- Use HTTPS with SSL certificate
- Consider using CloudFlare for DDoS protection

## 2. 📊 Multiple Pairs Support

Use the `MultiSymbolSender.mq5` EA which supports:
- All symbols from Market Watch
- Specific symbol lists
- Efficient batch sending

### Configuration:
```mql5
input bool InpAllSymbols = true;  // Send all Market Watch symbols
input string InpSymbols = "EURUSD,GBPUSD,USDJPY";  // Or specific symbols
```

## 3. 📈 Volume Calculation

The MultiSymbolSender EA provides:
- **Tick Volume**: Direct from MT5
- **Real Volume**: Where available (stocks, futures)
- **Volume per Minute**: Calculated from tick data
- **5-Minute Volume**: Rolling calculation

### Volume Types:
```json
"volume": {
    "tick": 1234,           // Tick volume
    "per_minute": 45.67,    // Calculated per minute
    "last_5min": 234.56     // 5-minute average
}
```

## 4. 🔧 Full Project Integration

### Directory Structure:
```
zanalytics-main/
├── _dash_v_2/
│   ├── enhanced_api_server.py    # Main API with enrichment
│   ├── pages/
│   │   ├── home.py               # Main dashboard
│   │   ├── mt5_monitor.py        # MT5 monitoring
│   │   ├── symbol_analysis.py    # Per-symbol analysis
│   │   └── strategy_signals.py   # Strategy signals
│   ├── utils/
│   │   ├── redis_manager.py      # Redis utilities
│   │   ├── data_enrichment.py    # Data processing
│   │   └── alerts.py             # Alert system
│   └── config/
│       └── settings.py           # Configuration
└── mt5_experts/
    ├── MultiSymbolSender.mq5     # Multi-symbol EA
    └── strategies/               # Strategy EAs
```

### Integration Steps:

1. **Replace your current API server**:
   ```bash
   cp enhanced_api_server.py api_server_port8080.py
   ```

2. **Update your dashboards** to use enriched data:
   ```python
   # In your dashboard pages
   import requests

   # Get all symbols
   symbols = requests.get('http://localhost:8080/symbols').json()

   # Get enriched data for a symbol
   data = requests.get('http://localhost:8080/data/EURUSD').json()

   # Access enriched metrics
   volatility = data['enriched']['volatility_1min']
   rsi = data['enriched']['rsi']
   trend = data['enriched']['trend']
   ```

## 5. 🎯 Data Enrichment & Redis Usage

### What Redis Stores:

1. **Latest Data** (5-minute TTL):
   - Key: `mt5:EURUSD:latest`
   - Contains: Current tick, spread, volume

2. **Historical Data** (Last 1000 ticks):
   - Key: `mt5:EURUSD:history`
   - Contains: List of recent ticks

3. **Enriched Metrics**:
   - Key: `mt5:EURUSD:enriched`
   - Contains: Calculated indicators

4. **Account Info**:
   - Key: `account:info`
   - Contains: Balance, equity, margin

### Enriched Metrics Available:
- **Volatility**: 1-min and 5-min
- **RSI**: 14-period
- **Momentum**: 10-period
- **VWAP**: Volume-weighted average
- **Volume Profile**: Distribution at price levels
- **Trend**: Current market direction

### Using Redis Properly:

```python
import redis
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Get latest data
latest = json.loads(r.get('mt5:EURUSD:latest'))

# Get history
history = [json.loads(h) for h in r.lrange('mt5:EURUSD:history', 0, 99)]

# Subscribe to real-time updates
pubsub = r.pubsub()
pubsub.subscribe('mt5:EURUSD')
for message in pubsub.listen():
    if message['type'] == 'message':
        data = json.loads(message['data'])
        # Process real-time data

# Monitor Redis memory
info = r.info('memory')
print(f"Memory used: {info['used_memory_human']}")
```

### Redis Best Practices:
1. **Set TTL** on all keys to prevent memory bloat
2. **Use pub/sub** for real-time updates
3. **Trim lists** to reasonable lengths
4. **Monitor memory** usage regularly
5. **Use Redis persistence** for important data

## 🚀 Complete Setup Commands:

```bash
# 1. Start Redis
redis-server

# 2. Start enhanced API
python enhanced_api_server.py

# 3. Setup ngrok
python ngrok_setup.py

# 4. Start dashboard
streamlit run pages/home.py

# 5. Install MT5 EA
# Copy MultiSymbolSender.mq5 to MT5
# Update URL with ngrok address
# Add all symbols to Market Watch
```

## 📊 Monitoring & Maintenance:

- Check Redis info: `http://localhost:8080/redis/info`
- Clean old data: `POST http://localhost:8080/redis/cleanup`
- Monitor symbols: `http://localhost:8080/symbols`
- View enriched data: `http://localhost:8080/data/EURUSD`
