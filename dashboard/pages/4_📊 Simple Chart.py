#!/usr/bin/env python3
"""
Zanalytics - Ultimate Trading Dashboard
Comprehensive microstructure, SMC, Wyckoff, and top-down analysis.
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
import os
import glob
from typing import Dict, List, Optional, Tuple, Any
import re

warnings.filterwarnings('ignore')


class ZanalyticsDashboard:
    def __init__(self):
        """Initialize ultimate dashboard with a configurable data directory."""
        try:
            data_directory = st.secrets["data_directory"]
        except (FileNotFoundError, KeyError):
            data_directory = "./data"

        self.data_dir = Path(data_directory)
        self.pairs_data = {}
        self.analysis_reports = {}
        self.smc_analysis = {}
        self.wyckoff_analysis = {}
        self.microstructure_data = {}
        self.latest_txt_reports = {}
        self.latest_json_insights = {}

        # Initialize session state keys
        if 'selected_pair' not in st.session_state:
            st.session_state.selected_pair = None
        if 'selected_timeframe' not in st.session_state:
            st.session_state.selected_timeframe = None
        if 'chart_theme' not in st.session_state:
            st.session_state.chart_theme = 'plotly_dark'
        if 'lookback_bars' not in st.session_state:
            st.session_state.lookback_bars = 500

    def load_all_data(self):
        """Load all processed data silently."""
        if not self.data_dir.exists():
            return

        pair_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]

        for pair_dir in pair_dirs:
            pair_name = re.split(r'[_\-]', pair_dir.name)[0].upper()
            if pair_name not in self.pairs_data:
                self.pairs_data[pair_name] = {}

            csv_files = list(pair_dir.glob("*_COMPREHENSIVE_*.csv"))
            for csv_file in csv_files:
                parts = csv_file.stem.split('_COMPREHENSIVE_')
                if len(parts) == 2:
                    timeframe = parts[1]
                    try:
                        df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
                        self.pairs_data[pair_name][timeframe] = df
                    except Exception:
                        continue

    def create_main_dashboard(self):
        """Create the ultimate dashboard interface"""
        st.set_page_config(
            page_title="Zanalytics - Ultimate Dashboard",
            page_icon="�",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        st.markdown("""
        <style>
        .metric-card {
            background: #f0f2f6;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #1e3c72;
        }
        </style>
        """, unsafe_allow_html=True)

        if not self.data_dir.exists():
            st.error(f"Data directory not found at: `{self.data_dir}`")
            st.info(
                "Please create this directory or configure the correct path in your `.streamlit/secrets.toml` file.")
            st.code('data_directory = "/path/to/your/data"')
            return

        with st.spinner("Initializing analysis engine..."):
            self.load_all_data()

        if not self.pairs_data:
            st.error("❌ No processed data found. Please check your data directory.")
            return

        self.create_sidebar_controls()

        if st.session_state.get('selected_pair') and st.session_state.get('selected_timeframe'):
            self.display_ultimate_analysis()
        else:
            self.display_market_overview()

    def create_sidebar_controls(self):
        """Create comprehensive sidebar controls"""
        st.sidebar.title("🎛️ Analysis Control Center")

        available_pairs = list(self.pairs_data.keys())
        if available_pairs:
            selected_pair = st.sidebar.selectbox(
                "📈 Select Currency Pair",
                available_pairs,
                key="selected_pair"
            )

            if selected_pair in self.pairs_data:
                available_timeframes = list(self.pairs_data[selected_pair].keys())
                if available_timeframes:
                    selected_timeframe = st.sidebar.selectbox(
                        "⏱️ Select Timeframe",
                        available_timeframes,
                        key="selected_timeframe"
                    )

                    if selected_timeframe in self.pairs_data[selected_pair]:
                        df = self.pairs_data[selected_pair][selected_timeframe]
                        latest_price = df['close'].iloc[-1]
                        price_change = ((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100

                        st.sidebar.markdown("---")
                        st.sidebar.markdown("### 📊 Market Status")
                        col1, col2 = st.sidebar.columns(2)
                        with col1:
                            st.metric("Price", f"{latest_price:.4f}")
                        with col2:
                            st.metric("Change", f"{price_change:+.2f}%")

        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📈 Chart Settings")
        st.session_state['lookback_bars'] = st.sidebar.slider("Lookback Period", 100, 2000, 500)
        st.session_state['chart_theme'] = st.sidebar.selectbox("Chart Theme",
                                                               ["plotly_dark", "plotly_white", "ggplot2"])

    def display_market_overview(self):
        """Display comprehensive market overview"""
        st.markdown("## 🌍 Market Overview & Analysis Summary")
        st.info("Please select a currency pair and timeframe from the sidebar to begin analysis.")

    def display_ultimate_analysis(self):
        """Display comprehensive analysis dashboard"""
        pair = st.session_state['selected_pair']
        timeframe = st.session_state['selected_timeframe']

        if pair not in self.pairs_data or timeframe not in self.pairs_data[pair]:
            st.error("Selected data not available")
            return

        df = self.pairs_data[pair][timeframe]
        lookback = st.session_state.get('lookback_bars', 500)
        df_display = df.tail(lookback).copy()

        st.markdown(f"# 🚀 {pair} {timeframe} - Ultimate Analysis")
        self.create_ultimate_price_chart(df_display, pair, timeframe)

    def create_ultimate_price_chart(self, df, pair, timeframe):
        """Create ultimate price chart with all overlays"""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3]
        )

        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'], high=df['high'],
            low=df['low'], close=df['close'],
            name="Price",
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4444'
        ), row=1, col=1)

        if 'volume' in df.columns:
            colors = ['#00ff88' if c >= o else '#ff4444' for c, o in zip(df['close'], df['open'])]
            fig.add_trace(go.Bar(
                x=df.index, y=df['volume'],
                name='Volume', marker_color=colors, opacity=0.7
            ), row=2, col=1)

        fig.update_layout(
            title=f"{pair} {timeframe} Ultimate Analysis",
            template=st.session_state.get('chart_theme', 'plotly_dark'),
            height=800,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    dashboard = ZanalyticsDashboard()
    dashboard.create_main_dashboard()
