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

    def __init__(self, data_directory=None):
        import streamlit as st
        self.data_dir = Path(data_directory or st.secrets.get("JSONdir", "./data"))
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

    def display_institutional_analysis(self):
        st.markdown("## 🏛️ Institutional Grade Analysis")

        json_files = list(self.data_dir.rglob("*.json"))
        available_assets = sorted(set(Path(f).stem.split("_")[0] for f in json_files))
        if not available_assets:
            st.info("No institutional JSON files found.")
            return

        selected_asset = st.selectbox("Select Asset", available_assets)
        selected_sections = st.multiselect(
            "Select Sections to Display",
            ["Market Overview", "Smart Money Concepts", "Wyckoff Analysis", "Advanced Stats", "Risk Analysis",
             "Price Chart & Heatmap"],
            default=["Market Overview", "Smart Money Concepts", "Price Chart & Heatmap"]
        )

        matching_file = next((f for f in json_files if selected_asset in f.name), None)
        if not matching_file:
            st.warning("No matching JSON data found for the selected asset.")
            return

        with open(matching_file, "r") as f:
            data = json.load(f)

        key = next(iter(data))
        meta = data[key]
        stats = meta.get("basic_stats", {})
        indicators = meta.get("indicators", {})
        smc = meta.get("smc_analysis", {})

        price = stats.get("last_price")
        pct = stats.get("price_change_pct")
        vol = stats.get("volatility")
        high = stats.get("high_price")
        low = stats.get("low_price")
        rsi = indicators.get("RSI_14", "N/A")
        atr = indicators.get("ATR_14", "N/A")
        trend = "🔴 BEAR" if isinstance(pct, (int, float)) and pct < 0 else "🟢 BULL"

        st.markdown(f"""
        ### 🚀 {selected_asset} Institutional Snapshot  
        **Current Price:** ${price:,.2f} if price is not None else "N/A"
        **Change %:** {pct:+.2f}%  
