#!/usr/bin/env python3
"""
Zanalytics Dashboard

A focused dashboard providing at-a-glance market intelligence and an overview of available data.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import glob
from pathlib import Path 
from datetime import datetime
import warnings
import re
from typing import Dict, Optional
import base64
import yfinance as yf
from fredapi import Fred

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')


# --- Utility Function for Background Image ---
def get_image_as_base64(path):
    """Reads an image file and returns its base64 encoded string."""
    try:
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except FileNotFoundError:
        st.warning(f"Background image not found at '{path}'. Please ensure it's in the same directory as the script.")
        return None


# --- Economic Data Manager ---
class EconomicDataManager:
    """ Manages fetching live economic data using yfinance. """

    def get_dxy_data(self) -> Optional[pd.DataFrame]:
        """ Fetches weekly OHLC data for DXY for the last year. """
        try:
            ticker = yf.Ticker("DX-Y.NYB")
            hist = ticker.history(period="1y", interval="1d")
            if not hist.empty:
                # Resample to weekly
                hist_weekly = hist.resample('W').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last'
                }).dropna()
                return hist_weekly.tail(26)  # Last 6 months of weekly data
            return None
        except Exception:
            return None


class ZanalyticsDashboard:
    def __init__(self):
        """
        Initializes the dashboard, loading configuration from Streamlit secrets.
        """
        try:
            # Load data directory from secrets.toml
            data_directory = st.secrets["data_directory"]
        except (FileNotFoundError, KeyError):
            # Fallback to a default directory if secrets.toml or the key is missing
            data_directory = "./data"

        self.data_dir = Path(data_directory)
        self.supported_pairs = ["XAUUSD", "BTCUSD", "EURUSD", "GBPUSD", "USDJPY", "ETHUSD", "USDCAD", "AUDUSD",
                                "NZDUSD", "DXYCAS"]
        self.timeframes = ["1min", "5min", "15min", "30min", "1H", "4H", "1D", "1W", "5T"]

        self.economic_manager = EconomicDataManager()
        self.fred = Fred(api_key="6a980b8c2421503564570ecf4d765173")

        if 'chart_theme' not in st.session_state:
            st.session_state.chart_theme = 'plotly_dark'

    def run(self):
        st.set_page_config(page_title="Zanalytics Dashboard", page_icon="🚀", layout="wide",
                           initial_sidebar_state="expanded")

        img_base64 = get_image_as_base64("image_af247b.jpg")
        if img_base64:
            background_style = f"""
            <style>
            [data-testid="stAppViewContainer"] > .main {{
                background-image: linear-gradient(rgba(0,0,0,0.8), rgba(0,0,0,0.8)), url(data:image/jpeg;base64,{img_base64});
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            #MainMenu {{visibility: hidden;}}
            footer {{visibility: hidden;}}
            </style>
            """
            st.markdown(background_style, unsafe_allow_html=True)

        if not self.data_dir.exists():
            st.error(f"Data directory not found at: `{self.data_dir}`")
            st.info(
                "Please create this directory or configure the correct path in your `.streamlit/secrets.toml` file.")
            st.code('data_directory = "/path/to/your/data"')
            return

        with st.spinner("🛰️ Scanning all data sources..."):
            data_sources = self.scan_all_data_sources()
        st.success(f"📂  Loaded **{len(data_sources)}** asset folders • **{sum(len(v) for v in data_sources.values())}** timeframe files detected")

        self.display_home_page(data_sources)

    def display_home_page(self, data_sources):
        st.markdown(
            """
            <h2 style='text-align:center; margin-bottom:0.2rem'>
                🚀 Zanalytics Dashboard  <small style='font-size:65%'>(v1.4)</small>
            </h2>
            <p style='text-align:center; font-size:0.9rem; margin-top:0'>
                A concise, trader‑oriented command center
            </p>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style='text-align:center; font-size:1.06rem; margin-bottom:1.2rem; margin-top:0.7rem; color:#e7eaf0;'>
            <b>Core Tools</b><br>
            • Market Overview – cross‑asset direction &amp; risk<br>
            • Pair Insights – multi‑TF technical stacks<br>
            • Wyckoff / SMC – phase &amp; smart‑money levels<br>
            • Microstructure – tick &amp; volume analytics
            </div>
            """,
            unsafe_allow_html=True,
        )

        self.create_dxy_chart()
        st.markdown("<hr style='margin-top:1.5rem'>", unsafe_allow_html=True)

        latest_yields, previous_yields = self.get_10y_yields()
        st.markdown("""
<style>
.yields-table {
    background: #1a222d;
    color: #e7eaf0;
    font-size: 1.05rem;
    border-radius: 8px;
    border: 1px solid #253047;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    margin-bottom: 1.5rem;
    margin-top: 0.5rem;
    width: 100%;
    max-width: 370px;
}
.yields-table th, .yields-table td {
    text-align: center !important;
    padding: 0.25rem 0.6rem;
}
</style>
""", unsafe_allow_html=True)

        st.markdown("<h5 style='text-align:center;'>🌍 10‑Year Government Bond Yields</h5>", unsafe_allow_html=True)
        cols = st.columns(len(latest_yields))
        for i, (country, val) in enumerate(latest_yields.items()):
            prev_val = previous_yields.get(country)
            delta = None
            if prev_val is not None and val != "N/A":
                delta = round(val - prev_val, 3)
            cols[i].metric(country, f"{val}%" if val != 'N/A' else val, delta)

        # ─── Available datasets (bottom, plain) ────────────────────────────────
        st.markdown("---")
        self.display_available_data(data_sources)

    def get_10y_yields(self):
        """Fetches the latest and previous available 10Y government bond yields from FRED."""
        tickers = {
            "US": "DGS10",
            "Germany": "IRLTLT01DEM156N",
            "Japan": "IRLTLT01JPM156N",
            "UK": "IRLTLT01GBM156N",
        }
        latest_yields, previous_yields = {}, {}
        for country, code in tickers.items():
            try:
                series = self.fred.get_series(code).dropna()
                latest = float(series.iloc[-1])
                prev = float(series.iloc[-2]) if len(series) > 1 else None
                latest_yields[country] = round(latest, 3)
                previous_yields[country] = round(prev, 3) if prev else None
            except Exception:
                latest_yields[country] = "N/A"
                previous_yields[country] = None
        return latest_yields, previous_yields

    def display_available_data(self, data_sources):
        """Lists the pairs and timeframes found in the data directory."""
        st.markdown("##### Available Datasets")
        if not data_sources:
            st.warning("No data found in the configured directory.")
            return

        for pair, tfs in sorted(data_sources.items()):
            if tfs:
                # Sort timeframes logically
                tf_list = ", ".join(
                    sorted(tfs.keys(), key=lambda t: (self.timeframes.index(t) if t in self.timeframes else 99, t)))
                st.markdown(f"{pair}: {tf_list}")

    def create_dxy_chart(self):
        st.markdown("#### 💵 U.S. Dollar Index (DXY) - Weekly")
        dxy_data = self.economic_manager.get_dxy_data()
        if dxy_data is not None and not dxy_data.empty:
            fig = go.Figure(data=[go.Candlestick(
                x=dxy_data.index,
                open=dxy_data['Open'],
                high=dxy_data['High'],
                low=dxy_data['Low'],
                close=dxy_data['Close'],
                increasing_line_color='#26de81',
                decreasing_line_color='#fc5c65'
            )])
            fig.update_layout(
                title="DXY Performance (Last 6 Months)",
                template=st.session_state.get('chart_theme', 'plotly_dark'),
                height=500,
                showlegend=False,
                yaxis_title="DXY Value",
                xaxis_title="Date",
                xaxis_rangeslider_visible=False
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Could not load DXY chart data.")

    def scan_all_data_sources(self):
        """Scans for data files in the configured directory and its subdirectories."""
        data_sources = {}
        # Search for files in the root data directory and any subdirectories
        all_files = glob.glob(os.path.join(self.data_dir, "**", "*.*"), recursive=True)

        for f_path in all_files:
            # Use the parent directory name as the pair if it's a supported pair
            parent_dir_name = Path(f_path).parent.name

            found_pair = None
            if parent_dir_name.upper() in self.supported_pairs:
                found_pair = parent_dir_name.upper()
            else:
                # Fallback to searching the filename
                for pair in self.supported_pairs:
                    if pair in Path(f_path).name:
                        found_pair = pair
                        break

            if found_pair and f_path.endswith(('.csv', '.parquet')):
                for tf in self.timeframes:
                    if tf in Path(f_path).name:
                        if found_pair not in data_sources:
                            data_sources[found_pair] = {}
                        data_sources[found_pair][tf] = f_path
                        break
        return data_sources


if __name__ == "__main__":
    dashboard = ZanalyticsDashboard()
    dashboard.run()
