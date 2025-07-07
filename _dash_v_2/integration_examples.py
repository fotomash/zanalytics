# ZANFLOW Multi-Asset Integration Examples

## 1. Programmatic Access to Tick Data

```python
import redis
import json
import pandas as pd

r = redis.Redis(decode_responses=True)

# Get all monitored symbols
symbols = []
for key in r.keys("mt5:*:latest"):
    symbol = key.split(":")[1]
    symbols.append(symbol)

print(f"Monitoring {len(symbols)} symbols")

# Get tick history for analysis
symbol = "EURUSD"
ticks = r.lrange(f"mt5:{symbol}:ticks", 0, 1000)
tick_data = [json.loads(t) for t in ticks]

# Convert to DataFrame
df = pd.DataFrame(tick_data)
print(f"\n{symbol} Statistics:")
print(f"Average Spread: {df['spread'].mean():.2f}")
print(f"Max Spread: {df['spread'].max():.2f}")
print(f"Price Range: {df['bid'].max() - df['bid'].min():.5f}")
```

## 2. Running Your Tick Analysis Module

```python
from zanflow_microstructure_analyzer import MicrostructureAnalyzer

# Analyze saved tick file
analyzer = MicrostructureAnalyzer(
    "MT5_TickData/XAUUSD_ticks.csv",
    limit_ticks=5000,
    export_json=True,
    output_dir="analysis_results"
)

# Run full analysis
analyzer.run_full_analysis()

# Access results
print(f"Manipulation Events: {analyzer.analysis_results}")
```

## 3. Triggering Analysis via API

```python
import requests

# Trigger deep analysis for specific symbol
response = requests.post("http://localhost:5000/trigger_analysis/GBPUSD")
print(response.json())

# Get analysis results
response = requests.get("http://localhost:5000/analysis/GBPUSD")
data = response.json()

# Check manipulation score
if data['latest_tick']['tick_analysis']['manipulation_score'] > 50:
    print("⚠️ High manipulation detected!")
    print(f"Recent alerts: {data['recent_alerts']}")
```

## 4. Custom Alert System

```python
import redis
import json
import time
from datetime import datetime

r = redis.Redis(decode_responses=True)

def monitor_manipulation(threshold=50):
    """Monitor all symbols for high manipulation"""
    while True:
        for key in r.keys("mt5:*:latest"):
            symbol = key.split(":")[1]
            data = json.loads(r.get(key))

            tick_analysis = data.get('tick_analysis', {})
            score = tick_analysis.get('manipulation_score', 0)

            if score > threshold:
                alert = {
                    'symbol': symbol,
                    'score': score,
                    'bid': data['bid'],
                    'ask': data['ask'],
                    'spread': data['spread'],
                    'timestamp': datetime.now().isoformat()
                }

                # Send to your notification system
                print(f"🚨 ALERT: {symbol} manipulation score: {score}")

                # Log to file
                with open('manipulation_alerts.json', 'a') as f:
                    f.write(json.dumps(alert) + '\n')

        time.sleep(10)  # Check every 10 seconds

# Run monitor
monitor_manipulation(threshold=60)
```

## 5. Batch Processing Tick Files

```python
from pathlib import Path
from convert_final_enhanced_smc_ULTIMATE import DataProcessor, ProcessingConfig

# Configure processor
config = ProcessingConfig(
    output_dir="enriched_ticks",
    tick_limit=5000,  # Last 5000 ticks
    bar_limit=250,    # 250 bars as requested
    calculate_signals=True,
    include_market_structure=True
)

# Process all tick files
processor = DataProcessor(config)
tick_dir = Path("MT5_TickData")

for tick_file in tick_dir.glob("*_ticks.csv"):
    print(f"Processing {tick_file.name}...")
    processor.process_file(tick_file)

print("✅ All tick files enriched with 200+ indicators")
```

## 6. Real-time Strategy Integration

```python
class ManipulationStrategy:
    def __init__(self):
        self.r = redis.Redis(decode_responses=True)
        self.positions = {}

    def check_entry_conditions(self, symbol):
        # Get latest data
        data = json.loads(self.r.get(f"mt5:{symbol}:latest"))

        # Get analysis
        analysis = json.loads(self.r.get(f"analysis:{symbol}:microstructure") or '{}')

        # Entry logic
        manipulation_score = data['tick_analysis']['manipulation_score']
        spread_spikes = data['tick_analysis']['spread_spikes']

        # High manipulation + spread spikes = potential reversal
        if manipulation_score > 70 and spread_spikes > 5:
            # Check enriched data for confirmation
            enriched = data.get('enriched_bars', {})
            if enriched.get('volatility', 0) > enriched.get('avg_close', 1) * 0.001:
                return 'SHORT'  # Manipulation often precedes reversal

        return None

    def run(self):
        while True:
            for symbol in ['EURUSD', 'GBPUSD', 'XAUUSD']:
                signal = self.check_entry_conditions(symbol)
                if signal:
                    print(f"📊 {symbol}: {signal} signal detected!")

            time.sleep(1)
```

## 7. Connect to Your Existing Dashboards

Add this to your Streamlit pages:

```python
# In your pages/tick_manipulation.py
import streamlit as st
import requests
import plotly.graph_objects as go

st.title("Tick Manipulation Insights")

# Get data from multi-asset processor
resp = requests.get("http://localhost:5000/summary")
if resp.status_code == 200:
    data = resp.json()

    # Show high manipulation symbols
    high_manip = []
    for symbol, info in data['symbols'].items():
        if info['manipulation_score'] > 50:
            high_manip.append({
                'Symbol': symbol,
                'Score': info['manipulation_score'],
                'Ticks': info['tick_count']
            })

    if high_manip:
        st.error("⚠️ High Manipulation Detected!")
        st.dataframe(high_manip)
```

These examples show how to integrate the multi-asset tick analysis with your existing ZANFLOW infrastructure! 🚀
