# 🔗 ZANFLOW MT5 Integration Guide

## Current Setup Status ✅
- MT5 EA is sending data to your API (port 8080)
- API is receiving data and storing in Redis
- Dashboard can visualize the data

## 📊 To Run the Dashboard:

```bash
cd zanalytics-main/_dash_v_2
streamlit run dashboard_mt5.py
```

Dashboard will be available at: http://localhost:8501

## 🌐 Using Ngrok for Remote Access:

### 1. Install ngrok:
```bash
# Download from https://ngrok.com/download
# Or use pip:
pip install pyngrok
```

### 2. Expose your API:
```bash
ngrok http 8080
```

This will give you a public URL like: `https://abc123.ngrok.io`

### 3. Update MT5 EA:
- Change the URL in EA to your ngrok URL
- Add the ngrok domain to MT5 allowed URLs

## 🔧 Integration with Other Strategies:

### 1. Create Strategy-Specific EAs:
```mql5
// Example: RSI Strategy EA
input string StrategyName = "RSI_Strategy";
input int RSI_Period = 14;

void SendStrategyData()
{
    double rsi = iRSI(_Symbol, _Period, RSI_Period, PRICE_CLOSE, 0);

    string json = "{";
    json += "\"strategy\": \"" + StrategyName + "\",";
    json += "\"symbol\": \"" + _Symbol + "\",";
    json += "\"rsi\": " + DoubleToString(rsi, 2) + ",";
    json += "\"signal\": \"" + GetSignal(rsi) + "\"";
    json += "}";

    // Send to your API
    SendHttpRequest(json);
}
```

### 2. API Endpoints for Different Strategies:
```python
# In your API server
@app.route('/strategy/<strategy_name>', methods=['POST'])
def strategy_webhook(strategy_name):
    data = request.get_json()
    # Store with strategy-specific key
    r.set(f"strategy:{strategy_name}:latest", json.dumps(data))
    return jsonify({"status": "ok"})
```

### 3. Multi-Strategy Dashboard:
- Create tabs for different strategies
- Show strategy-specific metrics
- Compare performance across strategies

## 📁 File Organization:

```
zanalytics-main/
├── _dash_v_2/
│   ├── api_server_port8080.py      # Your API server
│   ├── dashboard_mt5.py            # MT5 dashboard
│   ├── pages/
│   │   ├── home.py                 # Main dashboard
│   │   ├── strategy_rsi.py         # RSI strategy page
│   │   └── strategy_ma.py          # MA strategy page
│   └── strategies/
│       ├── rsi_analyzer.py         # RSI analysis
│       └── ma_analyzer.py          # MA analysis
└── mt5_experts/
    ├── HttpsJsonSender_Port8080.mq5
    ├── RSI_Strategy.mq5
    └── MA_Strategy.mq5
```

## 🚀 Next Steps:

1. **Test the dashboard** with live MT5 data
2. **Create strategy-specific EAs** for your trading strategies
3. **Build analysis modules** for each strategy
4. **Set up alerts** based on strategy signals
5. **Add backtesting** capabilities

## 💡 Pro Tips:

- Use Redis pub/sub for real-time updates
- Store historical data in PostgreSQL for analysis
- Use WebSockets for live dashboard updates
- Implement strategy performance tracking
