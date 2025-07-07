
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
import time
import json
from datetime import datetime

# Configuration
PROCESSOR_URL = "http://localhost:5000"
UPDATE_INTERVAL = 5  # seconds

# Page config
st.set_page_config(
    page_title="ZANFLOW Multi-Asset Tick Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dashboard title
st.title("📊 ZANFLOW Multi-Asset Tick Analyzer")

# Sidebar for controls
st.sidebar.title("Controls")
auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
refresh_rate = st.sidebar.slider("Refresh rate (seconds)", min_value=1, max_value=60, value=5)

# Helper functions
def get_summary():
    """Get summary of all symbols from the processor"""
    try:
        response = requests.get(f"{PROCESSOR_URL}/summary", timeout=2)
        if response.status_code == 200:
            return response.json().get('summary', {})
        return {}
    except Exception as e:
        st.error(f"Error connecting to processor: {e}")
        return {}

def get_analysis(symbol):
    """Get detailed analysis for a specific symbol"""
    try:
        response = requests.get(f"{PROCESSOR_URL}/analysis/{symbol}", timeout=2)
        if response.status_code == 200:
            return response.json().get('analysis', {})
        return {}
    except Exception as e:
        st.error(f"Error fetching analysis: {e}")
        return {}

def trigger_analysis(symbol):
    """Trigger analysis for a specific symbol"""
    try:
        response = requests.post(f"{PROCESSOR_URL}/trigger_analysis/{symbol}", timeout=5)
        return response.status_code == 200
    except Exception as e:
        st.error(f"Error triggering analysis: {e}")
        return False

# Main dashboard components
def display_summary(summary):
    """Display summary of all symbols"""
    if not summary:
        st.warning("No data available. Make sure the processor is running and receiving tick data.")
        return

    # Create a dataframe for the summary
    df = pd.DataFrame.from_dict(summary, orient='index')

    # Add human-readable timestamps
    df['last_update_time'] = df['last_update'].apply(
        lambda x: datetime.fromtimestamp(x).strftime('%H:%M:%S') if x > 0 else 'Never'
    )
    df['last_analysis_time'] = df['last_analysis'].apply(
        lambda x: datetime.fromtimestamp(x).strftime('%H:%M:%S') if x > 0 else 'Never'
    )

    # Sort by tick count
    df = df.sort_values('tick_count', ascending=False)

    # Create columns for symbols and their stats
    cols = st.columns(3)
    with cols[0]:
        st.subheader("Symbol Overview")
        st.dataframe(
            df[['tick_count', 'last_update_time', 'last_analysis_time', 'has_analysis']], 
            use_container_width=True
        )

    # Create heatmap of tick activity
    with cols[1]:
        st.subheader("Tick Activity Heatmap")
        if len(df) > 0:
            # Normalize tick counts for the heatmap
            max_ticks = df['tick_count'].max()
            normalized_ticks = df['tick_count'] / max_ticks if max_ticks > 0 else df['tick_count']

            fig = px.imshow(
                np.array([normalized_ticks.values]),
                x=df.index,
                color_continuous_scale='Viridis',
                labels=dict(x="Symbol", y="", color="Relative Activity")
            )
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No tick data available for heatmap.")

    # Create a "last updated" indicator
    with cols[2]:
        st.subheader("System Status")
        now = datetime.now()
        latest_update = df['last_update'].max() if len(df) > 0 else 0

        if latest_update > 0:
            time_diff = now.timestamp() - latest_update
            if time_diff < 10:
                st.success(f"✅ System active - Last update: {time_diff:.1f}s ago")
            elif time_diff < 60:
                st.info(f"ℹ️ System active - Last update: {time_diff:.1f}s ago")
            else:
                st.warning(f"⚠️ System inactive - Last update: {int(time_diff/60)}m {int(time_diff%60)}s ago")
        else:
            st.error("❌ No recent updates detected")

        # Add processor connection status
        try:
            requests.get(f"{PROCESSOR_URL}/summary", timeout=1)
            st.success("✅ Connected to processor")
        except:
            st.error("❌ Cannot connect to processor")

def display_symbol_analysis(symbol, analysis):
    """Display detailed analysis for a specific symbol"""
    st.subheader(f"Analysis for {symbol}")

    if not analysis:
        st.info(f"No analysis data available for {symbol}. Try triggering an analysis.")
        if st.button(f"Trigger Analysis for {symbol}"):
            with st.spinner("Running analysis..."):
                if trigger_analysis(symbol):
                    st.success("Analysis triggered successfully!")
                    time.sleep(2)  # Give the server a moment to process
                    # Refresh the analysis data
                    analysis = get_analysis(symbol)
                else:
                    st.error("Failed to trigger analysis.")
        return

    # Get the latest analysis
    latest = analysis.get('latest', {})

    # Display tick data statistics
    st.subheader("Tick Data Statistics")
    cols = st.columns(4)

    tick_count = latest.get('tick_count', 0)
    timestamp = latest.get('timestamp', 0)

    cols[0].metric("Tick Count", tick_count)
    cols[1].metric("Last Update", datetime.fromtimestamp(timestamp).strftime('%H:%M:%S') if timestamp > 0 else 'Never')

    # Display microstructure analysis if available
    if 'microstructure' in latest:
        st.subheader("Microstructure Analysis")
        micro = latest['microstructure']

        if isinstance(micro, dict) and not micro.get('error'):
            # Create visualization based on available data
            # This will depend on the actual structure of your microstructure analysis output
            if 'manipulation_score' in micro:
                cols[2].metric("Manipulation Score", f"{micro['manipulation_score']:.2f}")

            if 'volume_imbalance' in micro:
                cols[3].metric("Volume Imbalance", f"{micro['volume_imbalance']:.2f}")

            # Create any charts from the microstructure data
            # For example, if there's a time series of scores:
            if 'time_series' in micro and isinstance(micro['time_series'], list) and len(micro['time_series']) > 0:
                st.subheader("Manipulation Score Timeline")
                df_series = pd.DataFrame(micro['time_series'])
                fig = px.line(df_series, x='timestamp', y='score', title="Manipulation Score Over Time")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Microstructure analysis data not in expected format or contains errors.")
            if 'error' in micro:
                st.error(f"Error in microstructure analysis: {micro['error']}")

    # Display SMC analysis if available
    if 'smc' in latest:
        st.subheader("SMC Analysis")
        smc = latest['smc']

        if isinstance(smc, dict) and not smc.get('error'):
            # Display SMC metrics (adjust based on your actual SMC output)
            if 'key_levels' in smc and isinstance(smc['key_levels'], list):
                st.write("Key Price Levels:")
                for level in smc['key_levels']:
                    st.write(f"- {level['type']}: {level['price']}")

            # Add more SMC visualizations as needed
        else:
            st.warning("SMC analysis data not in expected format or contains errors.")
            if 'error' in smc:
                st.error(f"Error in SMC analysis: {smc['error']}")

# Main app logic
def main():
    # Add tabs for different views
    tab1, tab2 = st.tabs(["Overview", "Symbol Analysis"])

    with tab1:
        summary = get_summary()
        display_summary(summary)

    with tab2:
        # Get list of symbols from summary
        summary = get_summary()
        symbols = list(summary.keys())

        if not symbols:
            st.warning("No symbols available. Make sure the processor is receiving tick data.")
        else:
            selected_symbol = st.selectbox("Select Symbol", symbols)
            analysis = get_analysis(selected_symbol)
            display_symbol_analysis(selected_symbol, analysis)

    # Auto-refresh logic
    if auto_refresh:
        time.sleep(refresh_rate)
        st.experimental_rerun()

if __name__ == "__main__":
    main()
