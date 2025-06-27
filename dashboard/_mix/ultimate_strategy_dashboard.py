#!/usr/bin/env python3
"""
Ultimate Strategy Dashboard
Streamlit dashboard for XANA-ready trading data visualization
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
import time
import numpy as np
from typing import Dict, Any

# Page config
st.set_page_config(
    page_title="Ultimate Strategy Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff6b6b;
    }
    .signal-bullish {
        color: #00c851;
        font-weight: bold;
    }
    .signal-bearish {
        color: #ff4444;
        font-weight: bold;
    }
    .signal-neutral {
        color: #ffbb33;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class DashboardAPI:
    """API client for the dashboard"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')

    def test_connection(self) -> bool:
        """Test API connection"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False

    def get_consolidated_data(self, symbol: str = "XAUUSD") -> Dict[str, Any]:
        """Get consolidated strategy data"""
        try:
            response = requests.get(f"{self.base_url}/summary/consolidated", 
                                  params={"symbol": symbol}, timeout=10)
            return response.json() if response.status_code == 200 else {}
        except Exception as e:
            st.error(f"API Error: {e}")
            return {}

    def get_tick_window(self, symbol: str = "XAUUSD", limit: int = 100) -> Dict[str, Any]:
        """Get tick microstructure data"""
        try:
            response = requests.get(f"{self.base_url}/microstructure/tick-window",
                                  params={"symbol": symbol, "limit": limit}, timeout=10)
            return response.json() if response.status_code == 200 else {}
        except:
            return {}

    def trigger_merge(self, symbol: str = "XAUUSD") -> Dict[str, Any]:
        """Trigger a fresh merge"""
        try:
            response = requests.post(f"{self.base_url}/merge/trigger",
                                   params={"symbol": symbol}, timeout=15)
            return response.json() if response.status_code == 200 else {}
        except Exception as e:
            st.error(f"Merge Error: {e}")
            return {}

def main():
    """Main dashboard function"""

    # Title and header
    st.title("🚀 Ultimate Strategy Dashboard")
    st.markdown("**XANA-Ready Trading Data Visualization**")

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")

    # API settings
    api_url = st.sidebar.text_input("API URL", value="http://localhost:8000")
    symbol = st.sidebar.selectbox("Symbol", ["XAUUSD", "EURUSD", "GBPUSD", "BTCUSD"])

    # Auto-refresh settings
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=False)
    refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 10, 300, 30)

    # Initialize API client
    api = DashboardAPI(api_url)

    # Connection status
    if api.test_connection():
        st.sidebar.success("🟢 API Connected")
    else:
        st.sidebar.error("🔴 API Disconnected")
        st.error(f"Cannot connect to API at {api_url}")
        st.info("Make sure your API server is running:\n```bash\npython ultimate_strategy_api.py```")
        return

    # Manual refresh button
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🔄 Refresh"):
            st.rerun()
    with col2:
        if st.button("⚡ Trigger Merge"):
            with st.spinner("Triggering merge..."):
                result = api.trigger_merge(symbol)
                if result:
                    st.success("Merge triggered!")
                    time.sleep(2)
                    st.rerun()

    # Main content
    display_dashboard(api, symbol)

    # Auto-refresh logic
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

def display_dashboard(api: DashboardAPI, symbol: str):
    """Display the main dashboard content"""

    # Get consolidated data
    with st.spinner("Loading strategy data..."):
        data = api.get_consolidated_data(symbol)

    if not data:
        st.error("No data available")
        return

    # Display timestamp
    timestamp = data.get("timestamp", "")
    if timestamp:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        st.markdown(f"**Last Updated:** {dt.strftime('%Y-%m-%d %H:%M:%S UTC')}")

    # Main metrics row
    display_main_metrics(data)

    # Three column layout
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        display_timeframe_analysis(data)
        display_tick_microstructure(data)

    with col2:
        display_entry_signals(data)
        display_confluence_chart(data)

    with col3:
        display_market_state(data)
        display_risk_metrics(data)

def display_main_metrics(data: Dict[str, Any]):
    """Display main metrics row"""

    # Get key metrics
    summaries = data.get("summaries", {})
    entry_signals = data.get("entry_signals", {})
    microstructure = data.get("microstructure", {})

    # Calculate metrics
    total_timeframes = len(summaries)
    confluence_score = entry_signals.get("confluence", {}).get("total_score", 0)
    signal_direction = entry_signals.get("v5", {}).get("direction", "NEUTRAL")
    tick_count = len(microstructure.get("tick_window", []))

    # Display metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Timeframes", total_timeframes, delta=None)

    with col2:
        st.metric("Confluence", f"{confluence_score:.1%}", 
                 delta=f"{confluence_score-0.5:.1%}" if confluence_score != 0.5 else None)

    with col3:
        color = "normal"
        if signal_direction == "BUY":
            color = "inverse"
        elif signal_direction == "SELL":
            color = "off"
        st.metric("Signal", signal_direction, delta=None)

    with col4:
        st.metric("Live Ticks", tick_count, delta=None)

    with col5:
        confidence = entry_signals.get("v5", {}).get("confidence", 0.5)
        st.metric("Confidence", f"{confidence:.1%}", delta=None)

def display_timeframe_analysis(data: Dict[str, Any]):
    """Display timeframe analysis"""

    st.subheader("📊 Multi-Timeframe Analysis")

    summaries = data.get("summaries", {})

    if not summaries:
        st.warning("No timeframe data available")
        return

    # Create timeframe comparison table
    tf_data = []

    for tf, tf_data_dict in summaries.items():
        if isinstance(tf_data_dict, dict):
            # Extract key indicators (simplified)
            rsi = "N/A"
            trend = "N/A"
            signal = "NEUTRAL"

            # Try to find RSI
            for key, value in tf_data_dict.items():
                if "rsi" in key.lower() and isinstance(value, (int, float)):
                    rsi = f"{value:.1f}"
                    break

            # Try to find trend
            for key, value in tf_data_dict.items():
                if "trend" in key.lower() and isinstance(value, str):
                    trend = value
                    break

            # Determine signal
            if isinstance(tf_data_dict, dict):
                bullish_count = sum(1 for k, v in tf_data_dict.items() 
                                  if "bull" in str(v).lower() or "up" in str(v).lower())
                bearish_count = sum(1 for k, v in tf_data_dict.items() 
                                  if "bear" in str(v).lower() or "down" in str(v).lower())

                if bullish_count > bearish_count:
                    signal = "🟢 BULLISH"
                elif bearish_count > bullish_count:
                    signal = "🔴 BEARISH"

            tf_data.append({
                "Timeframe": tf,
                "RSI": rsi,
                "Trend": trend,
                "Signal": signal,
                "Indicators": len(tf_data_dict)
            })

    if tf_data:
        df = pd.DataFrame(tf_data)
        st.dataframe(df, use_container_width=True)

def display_tick_microstructure(data: Dict[str, Any]):
    """Display tick microstructure analysis"""

    st.subheader("⚡ Tick Microstructure")

    microstructure = data.get("microstructure", {})
    tick_window = microstructure.get("tick_window", [])

    if not tick_window:
        st.warning("No tick data available")
        return

    # Convert to DataFrame
    df_ticks = pd.DataFrame(tick_window)

    if len(df_ticks) == 0:
        st.warning("Empty tick window")
        return

    # Display tick metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        if 'mid' in df_ticks.columns:
            latest_price = df_ticks['mid'].iloc[-1] if len(df_ticks) > 0 else 0
            st.metric("Latest Price", f"{latest_price:.5f}")

    with col2:
        if 'spread' in df_ticks.columns:
            avg_spread = df_ticks['spread'].mean()
            st.metric("Avg Spread", f"{avg_spread:.5f}")

    with col3:
        if 'volume' in df_ticks.columns:
            total_volume = df_ticks['volume'].sum()
            st.metric("Total Volume", f"{total_volume:.0f}")

    # Price chart
    if 'mid' in df_ticks.columns and len(df_ticks) > 1:
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            y=df_ticks['mid'],
            mode='lines',
            name='Mid Price',
            line=dict(color='blue')
        ))

        fig.update_layout(
            title="Tick Price Movement",
            height=300,
            showlegend=False,
            margin=dict(l=0, r=0, t=30, b=0)
        )

        st.plotly_chart(fig, use_container_width=True)

def display_entry_signals(data: Dict[str, Any]):
    """Display entry signals"""

    st.subheader("🎯 Entry Signals")

    entry_signals = data.get("entry_signals", {})

    if not entry_signals:
        st.warning("No entry signals available")
        return

    # V5 Signals
    v5 = entry_signals.get("v5", {})
    if v5:
        direction = v5.get("direction", "NEUTRAL")
        confidence = v5.get("confidence", 0.5)
        poi_tap = v5.get("poi_tap", False)

        # Signal display with color
        if direction == "BUY":
            st.markdown(f'<p class="signal-bullish">V5 Signal: {direction} ({confidence:.1%})</p>', 
                       unsafe_allow_html=True)
        elif direction == "SELL":
            st.markdown(f'<p class="signal-bearish">V5 Signal: {direction} ({confidence:.1%})</p>', 
                       unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="signal-neutral">V5 Signal: {direction} ({confidence:.1%})</p>', 
                       unsafe_allow_html=True)

        if poi_tap:
            st.success("✅ POI Tap Detected")

    # V10 Signals
    v10 = entry_signals.get("v10", {})
    if v10:
        bias = v10.get("bias", "neutral")
        killzone = v10.get("killzone", "unknown")
        judas = v10.get("judas", False)

        st.markdown(f"**V10 Bias:** {bias.upper()}")
        st.markdown(f"**Kill Zone:** {killzone}")

        if judas:
            st.warning("⚠️ Judas Swing Detected")

    # NCOS Signals
    ncos = entry_signals.get("ncos", {})
    if ncos:
        manipulation = ncos.get("manipulation_detected", False)
        liquidity_sweep = ncos.get("liquidity_sweep", False)

        if manipulation:
            st.error("🚨 Manipulation Detected")

        if liquidity_sweep:
            st.info("💧 Liquidity Sweep")

def display_confluence_chart(data: Dict[str, Any]):
    """Display confluence visualization"""

    st.subheader("🔄 Signal Confluence")

    entry_signals = data.get("entry_signals", {})
    confluence = entry_signals.get("confluence", {})

    if not confluence:
        st.warning("No confluence data available")
        return

    # Confluence metrics
    total_score = confluence.get("total_score", 0)
    timeframes_aligned = confluence.get("timeframes_aligned", 0)

    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=total_score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confluence Score (%)"},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 25], 'color': "lightgray"},
                {'range': [25, 50], 'color': "gray"},
                {'range': [50, 75], 'color': "lightgreen"},
                {'range': [75, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))

    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # Additional metrics
    st.markdown(f"**Timeframes Aligned:** {timeframes_aligned}")

def display_market_state(data: Dict[str, Any]):
    """Display current market state"""

    st.subheader("🌊 Market State")

    # Extract market state info
    microstructure = data.get("microstructure", {})
    tick_analysis = microstructure.get("tick_analysis", {})

    # Display market conditions
    if tick_analysis:
        st.markdown("**Liquidity Conditions:**")
        avg_spread = tick_analysis.get("avg_spread", 0)

        if avg_spread > 0:
            if avg_spread < 0.3:
                st.success("🟢 High Liquidity")
            elif avg_spread < 0.6:
                st.warning("🟡 Medium Liquidity")
            else:
                st.error("🔴 Low Liquidity")

        # Volume analysis
        volume_total = tick_analysis.get("volume_total", 0)
        if volume_total > 0:
            st.markdown(f"**Volume:** {volume_total:.0f}")

    # Session info
    entry_signals = data.get("entry_signals", {})
    v10 = entry_signals.get("v10", {})
    killzone = v10.get("killzone", "unknown")

    st.markdown(f"**Active Session:** {killzone}")

    # Time info
    now = datetime.utcnow()
    st.markdown(f"**UTC Time:** {now.strftime('%H:%M:%S')}")

def display_risk_metrics(data: Dict[str, Any]):
    """Display risk metrics"""

    st.subheader("⚖️ Risk Metrics")

    # Extract risk info from summaries
    summaries = data.get("summaries", {})

    # Calculate volatility proxy
    volatility_scores = []
    for tf_data in summaries.values():
        if isinstance(tf_data, dict):
            for key, value in tf_data.items():
                if "atr" in key.lower() and isinstance(value, (int, float)):
                    volatility_scores.append(value)

    if volatility_scores:
        avg_volatility = np.mean(volatility_scores)
        st.markdown(f"**Volatility Score:** {avg_volatility:.4f}")

    # Market structure risk
    microstructure = data.get("microstructure", {})
    if microstructure.get("tick_analysis", {}):
        price_range = microstructure["tick_analysis"].get("price_range", 0)
        if price_range > 0:
            st.markdown(f"**Price Range:** {price_range:.5f}")

    # Signal confidence as risk proxy
    entry_signals = data.get("entry_signals", {})
    confidence = entry_signals.get("v5", {}).get("confidence", 0.5)

    risk_level = "Low" if confidence > 0.8 else "Medium" if confidence > 0.6 else "High"

    if risk_level == "Low":
        st.success(f"🟢 Risk: {risk_level}")
    elif risk_level == "Medium":
        st.warning(f"🟡 Risk: {risk_level}")
    else:
        st.error(f"🔴 Risk: {risk_level}")

if __name__ == "__main__":
    main()
