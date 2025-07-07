# ZANFLOW Multi-Asset Tick Analysis Setup

## Overview
This system monitors multiple assets from MT5 Market Watch, collects tick data, and performs sophisticated microstructure analysis using your existing modules.

## Components

### 1. MT5 EA: ZANFLOW_MultiAsset_TickAnalyzer_EA.mq5
- Monitors all symbols in Market Watch (or filtered list)
- Collects tick data with circular buffer
- Calculates 250-bar enrichment data
- Performs real-time tick manipulation detection
- Saves tick data to CSV files for Python processing

### 2. Python Server: multi_asset_tick_processor.py
- Receives multi-asset data via webhook
- Stores in Redis for real-time access
- Triggers your microstructure analysis
- Runs indicator enrichment on 250 bars
- Detects manipulation patterns

## Installation

### MT5 Setup
1. Copy `ZANFLOW_MultiAsset_TickAnalyzer_EA.mq5` to MT5 Experts folder
2. Compile in MetaEditor
3. Attach to ANY chart (it monitors all symbols)
4. Configure settings:
   - `SymbolFilter`: Leave empty for all, or "EURUSD,GBPUSD,XAUUSD"
   - `MaxTicksPerSymbol`: 10000 (adjust based on memory)
   - `EnrichmentBars`: 250 (bars to analyze)
   - `SaveTicksToFile`: true (for deep analysis)

### Python Setup
```bash
# Install dependencies
pip install flask pandas numpy redis

# Start Redis
redis-server

# Run the processor
python multi_asset_tick_processor.py
```

## Features

### Multi-Asset Monitoring
- Automatically tracks all Market Watch symbols
- Or use filtered list: "EURUSD,GBPUSD,XAUUSD,BTCUSD"
- Efficient memory management with circular buffers

### Tick Analysis (Real-time)
- Spread analysis and spike detection
- Price reversal detection
- Manipulation score (0-100)
- Microstructure metrics

### 250-Bar Enrichment
- Automatic calculation every tick
- Basic statistics: avg, volatility, range
- Price change and percentage
- Volume analysis

### Deep Analysis Integration
- Triggers your `zanflow_microstructure_analyzer.py`
- Runs full SMC/Wyckoff analysis
- Saves results to Redis
- Generates alerts for high manipulation

## API Endpoints

- `POST /webhook` - Receives MT5 data
- `GET /analysis/<symbol>` - Get analysis for symbol
- `GET /summary` - Overview of all symbols
- `POST /trigger_analysis/<symbol>` - Manual analysis trigger

## Example Usage

### Check Symbol Status
```python
import requests

# Get summary
resp = requests.get('http://localhost:5000/summary')
print(resp.json())

# Get EURUSD analysis
resp = requests.get('http://localhost:5000/analysis/EURUSD')
data = resp.json()
print(f"Manipulation Score: {data['latest_tick']['tick_analysis']['manipulation_score']}")
```

### Trigger Deep Analysis
```python
# Manually trigger analysis for XAUUSD
resp = requests.post('http://localhost:5000/trigger_analysis/XAUUSD')
print(resp.json())
```

## Integration with Your Modules

The system automatically uses your existing modules:

1. **zanflow_microstructure_analyzer.py**
   - Full tick manipulation detection
   - Wyckoff phase analysis
   - SMC pattern recognition
   - Inducement analysis

2. **convert_final_enhanced_smc_ULTIMATE.py**
   - 200+ technical indicators
   - Market structure analysis
   - Risk metrics calculation
   - Signal generation

## Monitoring Dashboard

Create a simple monitoring dashboard:
```python
import streamlit as st
import requests
import pandas as pd

st.title("ZANFLOW Multi-Asset Monitor")

# Get summary
summary = requests.get('http://localhost:5000/summary').json()

# Display symbols
for symbol, data in summary['symbols'].items():
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(symbol, f"{data['current_bid']:.5f}")
    with col2:
        st.metric("Ticks", data['tick_count'])
    with col3:
        score = data['manipulation_score']
        color = "🔴" if score > 50 else "🟡" if score > 30 else "🟢"
        st.metric("Manipulation", f"{color} {score:.1f}")
```

## Performance Tips

1. **Symbol Filtering**: Don't monitor all 100+ symbols
   ```
   SymbolFilter = "EURUSD,GBPUSD,XAUUSD,USDJPY,AUDUSD"
   ```

2. **Tick Buffer Size**: Adjust based on your needs
   ```
   MaxTicksPerSymbol = 5000  // Lower for more symbols
   ```

3. **Analysis Frequency**: Deep analysis runs every 1000 ticks
   - Adjust in Python code if needed

4. **File Storage**: Tick CSVs are saved every 60 seconds
   - Location: `MT5/MQL5/Files/ZANFLOW_TickData/`

## Troubleshooting

- **High CPU**: Reduce number of monitored symbols
- **Memory Usage**: Lower MaxTicksPerSymbol
- **Missing Analysis**: Check if modules are in Python path
- **No Data**: Verify MT5 has market data for symbols

This system gives you professional-grade multi-asset tick analysis with your sophisticated algorithms! 🚀
