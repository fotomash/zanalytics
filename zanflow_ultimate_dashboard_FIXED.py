#!/usr/bin/env python3
"""
ZANFLOW v12 Ultimate Trading Analysis Platform - FIXED
Handles NaN values and loads your actual data files
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    """Load and process your actual data files"""

    def __init__(self):
        self.data_cache = {}

    def find_data_files(self):
        """Find all your data files"""
        data_files = {
            'comprehensive': [],
            'processed': [],
            'tick': [],
            'json': []
        }

        for file in Path('.').glob('*.csv'):
            filename = file.name
            if 'COMPREHENSIVE' in filename:
                data_files['comprehensive'].append(filename)
            elif 'processed' in filename:
                data_files['processed'].append(filename)
            elif 'TICK' in filename:
                data_files['tick'].append(filename)

        for file in Path('.').glob('*.json'):
            data_files['json'].append(file.name)

        return data_files

    def load_csv_safe(self, filepath):
        """Safely load CSV with error handling"""
        try:
            df = pd.read_csv(filepath)
            if df.shape[1] == 1:  # Try tab-separated
                df = pd.read_csv(filepath, sep='\t')

            # Ensure timestamp column
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')

            return df
        except Exception as e:
            st.error(f"Error loading {filepath}: {e}")
            return None

    def get_data_summary(self, df):
        """Get safe data summary without NaN issues"""
        if df is None or df.empty:
            return {
                'total_bars': 0,
                'indicators': 0,
                'date_range': 'No data',
                'timeframe': 'Unknown'
            }

        # Safe calculations
        total_bars = len(df)
        indicators = len([col for col in df.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']])

        if 'timestamp' in df.columns and not df['timestamp'].isna().all():
            start_date = df['timestamp'].min()
            end_date = df['timestamp'].max()
            date_range = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        else:
            date_range = 'No timestamp data'

        return {
            'total_bars': total_bars,
            'indicators': indicators,
            'date_range': date_range,
            'timeframe': 'M1'
        }

def safe_metric_display(label, value, default="N/A"):
    """Safely display metrics without NaN errors"""
    try:
        if pd.isna(value) or value is None:
            st.metric(label, default)
        elif isinstance(value, (int, float)):
            if pd.isna(value):
                st.metric(label, default)
            else:
                st.metric(label, f"{int(value)}" if value == int(value) else f"{value:.2f}")
        else:
            st.metric(label, str(value))
    except (ValueError, TypeError):
        st.metric(label, default)

def create_price_chart(df):
    """Create a price chart"""
    if df is None or df.empty:
        st.warning("No data available for chart")
        return

    # Check required columns
    required_cols = ['open', 'high', 'low', 'close']
    if not all(col in df.columns for col in required_cols):
        st.warning("Missing OHLC data for chart")
        return

    # Use last 100 bars for better performance
    plot_data = df.tail(100).copy()

    fig = go.Figure()

    # Add candlestick
    fig.add_trace(go.Candlestick(
        x=plot_data.index if 'timestamp' not in plot_data.columns else plot_data['timestamp'],
        open=plot_data['open'],
        high=plot_data['high'],
        low=plot_data['low'],
        close=plot_data['close'],
        name='XAUUSD'
    ))

    fig.update_layout(
        title='XAUUSD Price Chart (Last 100 Bars)',
        xaxis_title='Time',
        yaxis_title='Price',
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

def main():
    """Main dashboard function"""
    st.set_page_config(
        page_title="ZANFLOW v12 - Fixed",
        page_icon="🚀",
        layout="wide"
    )

    st.title("🚀 ZANFLOW v12 Ultimate Trading Analysis Platform - FIXED")
    st.markdown("**Comprehensive Market Microstructure • Smart Money Concepts • Wyckoff Analysis • Top-Down Analysis**")

    # Initialize data loader
    loader = DataLoader()

    # Load data files
    with st.spinner("Loading data files..."):
        data_files = loader.find_data_files()

    # Market Overview
    st.header("🌍 Market Overview & Analysis Summary")

    col1, col2, col3, col4 = st.columns(4)

    # Safe metrics display
    total_files = sum(len(files) for files in data_files.values())
    with col1:
        safe_metric_display("Currency Pairs", 1)
    with col2:
        safe_metric_display("Total Datasets", total_files)
    with col3:
        safe_metric_display("CSV Files", len(data_files['comprehensive']) + len(data_files['processed']))
    with col4:
        safe_metric_display("JSON Files", len(data_files['json']))

    # Data Files Section
    st.header("📁 Available Data Files")

    tab1, tab2, tab3, tab4 = st.tabs(["Comprehensive", "Processed", "Tick Data", "JSON"])

    with tab1:
        st.subheader("Comprehensive Data Files")
        if data_files['comprehensive']:
            for file in data_files['comprehensive']:
                st.write(f"📊 {file}")
        else:
            st.info("No comprehensive files found")

    with tab2:
        st.subheader("Processed Data Files")
        if data_files['processed']:
            for file in data_files['processed']:
                st.write(f"📈 {file}")
        else:
            st.info("No processed files found")

    with tab3:
        st.subheader("Tick Data Files")
        if data_files['tick']:
            for file in data_files['tick']:
                st.write(f"⚡ {file}")
        else:
            st.info("No tick files found")

    with tab4:
        st.subheader("JSON Analysis Files")
        if data_files['json']:
            for file in data_files['json']:
                st.write(f"📋 {file}")
        else:
            st.info("No JSON files found")

    # Data Analysis Section
    st.header("📊 Data Analysis")

    # Select data file for analysis
    all_csv_files = data_files['comprehensive'] + data_files['processed']
    if all_csv_files:
        selected_file = st.selectbox("Select data file for analysis:", all_csv_files)

        if st.button("Analyze Selected File"):
            with st.spinner(f"Analyzing {selected_file}..."):
                df = loader.load_csv_safe(selected_file)

                if df is not None:
                    summary = loader.get_data_summary(df)

                    # Display summary
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        safe_metric_display("Total Bars", summary['total_bars'])
                    with col2:
                        safe_metric_display("Indicators", summary['indicators'])
                    with col3:
                        safe_metric_display("Timeframe", summary['timeframe'])
                    with col4:
                        st.metric("Date Range", summary['date_range'])

                    # Show current price if available
                    if 'close' in df.columns and not df['close'].empty:
                        current_price = df['close'].iloc[-1]
                        st.success(f"💰 Current XAUUSD Price: ${current_price:,.2f}")

                    # Create chart
                    create_price_chart(df)

                    # Show data sample
                    st.subheader("📋 Data Sample")
                    display_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    available_cols = [col for col in display_cols if col in df.columns]
                    st.dataframe(df[available_cols].tail(10))

                else:
                    st.error("Failed to load data file")
    else:
        st.warning("No CSV data files found in current directory")

    # Footer
    st.markdown("---")
    st.markdown("✅ **ZANFLOW v12 - Fixed Version** | No more NaN errors!")

if __name__ == "__main__":
    main()
