#!/usr/bin/env python3
"""
ZANFLOW Market Overview Dashboard

A focused dashboard for at-a-glance market intelligence, featuring a multi-timeframe
performance heatmap and correlation analysis.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
import glob
from pathlib import Path
from datetime import datetime, timedelta
import warnings
import re
from typing import Dict, List, Optional, Tuple, Any, Union

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')


class MarketOverviewDashboard:
    """
    Encapsulates all functionality for the Market Overview Dashboard.
    """

    def __init__(self, data_directory="./data"):
        """Initialize the dashboard."""
        self.data_dir = Path(data_directory)
        # --- Configuration ---
        self.supported_pairs = ["XAUUSD", "BTCUSD", "EURUSD", "GBPUSD", "USDJPY", "ETHUSD", "USDCAD", "AUDUSD",
                                "NZDUSD"]
        self.timeframes = ["1min", "5min", "15min", "30min", "1H", "4H", "1D", "1W"]

        # --- Color Scheme ---
        self.colors = {
            'background': '#1e1e2e',
            'text': '#f8f8f2',
            'grid': '#2d3748'
        }

        # --- Initialize Streamlit Session State ---
        if 'chart_theme' not in st.session_state:
            st.session_state.chart_theme = 'plotly_dark'

    def run(self):
        """Main execution function to run the dashboard."""
        st.set_page_config(
            page_title="Market Overview",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        st.markdown("""
        <style>
            .stMetric {
                border-radius: 10px;
                padding: 15px;
                background-color: #2a2a39;
            }
            /* Hide the Streamlit main menu and footer */
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
        </style>
        """, unsafe_allow_html=True)

        with st.spinner("🛰️ Scanning all data sources..."):
            data_sources = self.scan_all_data_sources()

        self.create_sidebar(data_sources)
        self.display_market_overview(data_sources)

    def create_sidebar(self, data_sources):
        """Create a simple sidebar for data source summary."""
        st.sidebar.title("Data Status")
        summary = []
        for pair in self.supported_pairs:
            count = len(data_sources.get(pair, {}))
            if count > 0:
                summary.append({'Pair': pair, 'Timeframes Found': count})

        if summary:
            st.sidebar.dataframe(pd.DataFrame(summary).set_index('Pair'), use_container_width=True)
        else:
            st.sidebar.warning("No data found.")

    def display_market_overview(self, data_sources):
        """Display a high-level overview of the market."""
        st.markdown("## 📈 Market Overview & Intelligence")

        market_data = {}
        with st.spinner("Loading market data for all timeframes..."):
            for pair, files in data_sources.items():
                if pair not in market_data:
                    market_data[pair] = {}
                for tf, file_path in files.items():
                    df = self.load_comprehensive_data(file_path, max_records=35)  # Load enough for 30-day correlation
                    if df is not None and not df.empty:
                        market_data[pair][tf] = df

        if not market_data:
            st.warning("No market data available for overview. Please check the `./data` directory.")
            return

        self.create_market_heatmap(market_data)

        col1, col2 = st.columns(2)
        with col1:
            self.create_top_movers(market_data)
        with col2:
            self.create_correlation_matrix(market_data)

    def create_market_heatmap(self, market_data):
        """Creates and displays a performance heatmap across all timeframes."""
        st.markdown("### Multi-Timeframe Momentum Heatmap")

        heatmap_data = {}
        # Ensure all pairs have a row, even if no data is found
        for pair in self.supported_pairs:
            heatmap_data[pair] = {tf: np.nan for tf in self.timeframes}

        for pair, tfs_data in market_data.items():
            for tf, df in tfs_data.items():
                # Defensive check for 'close' column and sufficient data
                if 'close' in df.columns and len(df) >= 2:
                    last_close = df['close'].iloc[-1]
                    prev_close = df['close'].iloc[-2]
                    if prev_close > 0:
                        change = ((last_close / prev_close) - 1) * 100
                        heatmap_data[pair][tf] = change

        # Convert to DataFrame and reorder columns logically
        heatmap_df = pd.DataFrame(heatmap_data).T
        heatmap_df = heatmap_df[self.timeframes]  # Ensure column order

        # Drop rows (pairs) that have no data at all
        heatmap_df.dropna(how='all', inplace=True)

        if not heatmap_df.empty:
            fig = px.imshow(heatmap_df,
                            text_auto=".2f%",
                            aspect="auto",
                            color_continuous_scale='RdYlGn',
                            labels=dict(x="Timeframe", y="Pair", color="Momentum %"),
                            title="Momentum of Most Recent Candle on Each Timeframe")
            fig.update_layout(template=st.session_state.chart_theme, height=500)
            fig.update_xaxes(side="top")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data to generate heatmap.")

    def create_top_movers(self, market_data):
        """Creates and displays top movers based on daily data."""
        st.markdown("### Top Movers (Daily)")
        movers = []
        for pair, tfs_data in market_data.items():
            # Defensive check for '1D' data and 'close' column
            if '1D' in tfs_data and 'close' in tfs_data['1D'].columns and len(tfs_data['1D']) >= 2:
                df_daily = tfs_data['1D']
                perf = ((df_daily['close'].iloc[-1] / df_daily['close'].iloc[-2]) - 1) * 100
                vol = df_daily['close'].pct_change().std() * 100
                movers.append({'Pair': pair, 'Performance %': perf, 'Volatility %': vol})

        if movers:
            movers_df = pd.DataFrame(movers).sort_values('Performance %', ascending=False).set_index('Pair')
            st.dataframe(movers_df, use_container_width=True)
        else:
            st.info("No daily data available to calculate top movers.")

    def create_correlation_matrix(self, market_data):
        """Creates and displays a correlation matrix based on daily data."""
        st.markdown("### Price Correlation Matrix (Daily)")
        daily_returns = {}
        for pair, tfs_data in market_data.items():
            # Defensive check for '1D' data and 'close' column
            if '1D' in tfs_data and 'close' in tfs_data['1D'].columns:
                daily_returns[pair] = tfs_data['1D']['close'].pct_change()

        returns_df = pd.DataFrame(daily_returns).tail(30)  # 30-day correlation

        if len(returns_df.columns) >= 2:
            corr_matrix = returns_df.corr()
            fig = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r')
            fig.update_layout(template=st.session_state.chart_theme, height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data for correlation matrix (requires at least 2 pairs with daily data).")

    def scan_all_data_sources(self):
        """Scans the data directory for all supported file types."""
        data_sources = {}
        all_files = glob.glob(os.path.join(self.data_dir, "**", "*.*"), recursive=True)

        for f_path in all_files:
            for pair in self.supported_pairs:
                if pair in f_path and f_path.endswith(('.csv', '.parquet')):
                    for tf in self.timeframes:
                        if tf in f_path:
                            if pair not in data_sources:
                                data_sources[pair] = {}
                            data_sources[pair][tf] = f_path
                            break
        return data_sources

    def load_comprehensive_data(self, file_path, max_records=None):
        """Loads and preprocesses data from a given file path."""
        try:
            if file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
            else:
                df = pd.read_csv(file_path, sep=None, engine='python')

            df.columns = [col.lower().strip() for col in df.columns]

            for col in ['timestamp', 'datetime', 'date']:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
                    df.set_index(col, inplace=True)
                    break

            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            rename_map = {'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}
            df.rename(columns=rename_map, inplace=True)

            if max_records:
                df = df.tail(max_records)

            return df
        except Exception:
            # Silently fail if a file can't be loaded, as it might be malformed.
            return None


if __name__ == "__main__":
    dashboard = MarketOverviewDashboard()
    dashboard.run()
