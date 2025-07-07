# Integration Instructions for 6_ 🚀 Ultimate Analysis.py

## Step 1: Add Required Imports
Add these imports at the top of your file:
```python
import requests
import time
```

## Step 2: Add the LiveDataConnector Class
Add this class after your imports:

```python
class LiveDataConnector:
    def __init__(self, api_url="http://mm20.local:8080"):
        self.api_url = api_url
        self._test_connection()

    def _test_connection(self):
        try:
            response = requests.get(f"{self.api_url}/symbols", timeout=2)
            if response.status_code == 200:
                st.sidebar.success("✅ Connected to mm20.local")
            else:
                st.sidebar.warning("⚠️ API connected but returned error")
        except:
            st.sidebar.error("❌ Cannot connect to mm20.local:8080")

    def get_symbols(self):
        try:
            response = requests.get(f"{self.api_url}/symbols")
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return []

    def get_live_data(self, symbol):
        try:
            response = requests.get(f"{self.api_url}/data/{symbol}")
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return None
```

## Step 3: Initialize the Connector
In your `UltimateZANFLOWDashboard.__init__` method, add:
```python
self.live_connector = LiveDataConnector("http://mm20.local:8080")
```

## Step 4: Add the Enhanced Chart Method
Copy the `create_live_price_chart` method from ultimate_analysis_enhanced.py and add it to your class.

## Step 5: Update display_ultimate_analysis
Replace:
```python
self.create_ultimate_price_chart(df_display, pair, timeframe)
```

With:
```python
self.create_live_price_chart(df_display, pair, timeframe, self.live_connector)
```

## Step 6: Add Auto-Refresh to Sidebar
In your sidebar creation, add:
```python
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔴 Live Data")
auto_refresh = st.sidebar.checkbox("Auto-refresh (5s)", value=False)
refresh_now = st.sidebar.button("🔄 Refresh Now")

if auto_refresh:
    time.sleep(5)
    st.rerun()
elif refresh_now:
    st.rerun()
```

## Key Features Added:
1. **Live Price Markers**: Yellow dotted line showing current price
2. **Bid/Ask Triangles**: Green up triangle for bid, red down for ask
3. **Spread Annotation**: Shows current spread
4. **Live Volume Bar**: Yellow bar for current volume
5. **Live RSI Marker**: Shows current RSI value
6. **Live Metrics Row**: Below chart showing bid, ask, spread, trend, vol/min
7. **Connection Status**: Shows connection to mm20.local in sidebar

## Styling Maintained:
- Black background
- Same color scheme (limegreen/crimson candles)
- Same MA colors
- Same subplot layout
- Same font and sizing
