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
        """ Fetches 15-min OHLC data for DXY for the last 60 days. """
        try:
            ticker = yf.Ticker("DX-Y.NYB")
            hist = ticker.history(period="60d", interval="15m")
            if not hist.empty:
                return hist.tail(100)  # Last 100 M15 bars
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
        st.markdown("""
        <style>
        section[data-testid="stSidebar"] {
            background-color: rgba(0,0,0,0.8) !important;
            box-shadow: none !important;
        }
        </style>
        """, unsafe_allow_html=True)

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
        # Patch: Add semi-transparent panel background after image is set
        st.markdown("""
        <style>
        .main .block-container {
            background-color: rgba(0,0,0,0.05) !important;
        }
        </style>
        """, unsafe_allow_html=True)

        if not self.data_dir.exists():
            st.error(f"Data directory not found at: `{self.data_dir}`")
            st.info(
                "Please create this directory or configure the correct path in your `.streamlit/secrets.toml` file.")
            st.code('data_directory = "/path/to/your/data"')
            return

        with st.spinner("🛰️ Scanning all data sources..."):
            data_sources = self.scan_all_data_sources()
        # Moved st.success to display_home_page
        self.display_home_page(data_sources)

    def display_home_page(self, data_sources):
        # --- Quant-Desk Welcome Block (Updated Design) ---
        st.markdown(
            """
            <div style='
                margin: 0 auto 1.1rem auto;
                max-width: 900px;
                text-align: center;
                padding: 0.2em 0 0.1em 0;
                background: linear-gradient(to right, rgba(103,116,255,0.25), rgba(176,66,255,0.25));
                border-radius: 12px;
                border: 2px solid rgba(251,213,1,0.4);
                box-shadow: 0 2px 12px rgba(103,116,255,0.10);
            '>
                <span style='
                    font-size: 1.65rem;
                    font-weight: 700;
                    color: #fff;
                    letter-spacing: -0.02em;
                    display: block;
                    margin-bottom: 0.15em;
                '>
                    Zanalytics Quant Market Desk
                </span>
                <span style='
                    font-size: 1.09rem;
                    color: #eee;
                    font-weight: 600;
                    display: block;
                    margin-bottom: 0.2em;
                '>
                    AI-Powered Global Market Intelligence
                </span>
                <span style='
                    font-size: 1.01rem;
                    color: #dbeafe;
                    display: block;
                    margin-bottom: 0.25em;
                '>
                    Microstructure • Liquidity Flows • Macro Trends
                </span>
                <span style='
                    font-size: 0.97rem;
                    color: #bfc8da;
                    line-height: 1.42;
                    display: block;
                '>
                    Built for active risk, scenario analysis & edge discovery.
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

        # --- Microstructure 3D Surface Demo (XAUUSD) ---
        import plotly.graph_objects as go

        xau_ticks_path = Path(st.secrets["raw_data_directory"]) / "XAUUSD_ticks.csv"
        if xau_ticks_path.exists():
            try:
                # Ingest standard comma‑separated CSV (no custom delimiter)
                df_ticks = pd.read_csv(xau_ticks_path, nrows=1000, encoding_errors='ignore')
                if 'timestamp' in df_ticks.columns:
                    df_ticks['timestamp'] = pd.to_datetime(df_ticks['timestamp'], errors='coerce')
                if 'bid' in df_ticks.columns and 'ask' in df_ticks.columns:
                    df_ticks['price_mid'] = (df_ticks['bid'] + df_ticks['ask']) / 2
                else:
                    df_ticks['price_mid'] = df_ticks['last'] if 'last' in df_ticks.columns else df_ticks.iloc[:, 1]
                df_ticks['inferred_volume'] = df_ticks['tickvol'] if 'tickvol' in df_ticks.columns else 1

                # Bin timestamps and price for 3D surface
                time_bins = pd.cut(df_ticks.index, bins=50, labels=False)
                price_bins = pd.cut(df_ticks['price_mid'], bins=50, labels=False)
                surface_data = pd.pivot_table(
                    df_ticks, values='inferred_volume',
                    index=price_bins, columns=time_bins,
                    aggfunc='sum', fill_value=0
                )

                if not surface_data.empty:
                    fig = go.Figure(
                        data=[go.Surface(
                            z=surface_data.values,
                            colorscale='Viridis',
                            name='Volume Surface'
                        )]
                    )
                    fig.update_layout(
                        title="XAUUSD Microstructure 3D Volume Map (Recent Tick Data)",
                        autosize=True,
                        height=400,
                        margin=dict(l=40, r=40, t=60, b=40),
                        template=st.session_state.get('chart_theme', 'plotly_dark'),
                        scene=dict(
                            xaxis_title="Time Bin",
                            yaxis_title="Price Bin",
                            zaxis_title="Inferred Volume"
                        ),
                        paper_bgcolor="rgba(0,0,0,0.05)",
                        plot_bgcolor="rgba(0,0,0,0.05)",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Not enough tick data for 3D surface.")
            except Exception as e:
                st.warning(f"Error loading or plotting XAU tick file: {e}")
        else:
            st.info("XAUUSD tick data not found for 3D surface demo.")

        self.create_dxy_chart()
        st.markdown(
            "<div style='text-align:center; font-size:0.97rem; color:#bbb; margin-bottom:1.2rem;'>"
            "Bar chart above: U.S. Dollar Index (DXY) OHLC – weekly bars, auto-updated."
            "</div>",
            unsafe_allow_html=True,
        )

        # --- DXY 3D Surface Chart ---
        dxy_data = self.economic_manager.get_dxy_data()
        if dxy_data is not None and not dxy_data.empty:
            time_bins = np.arange(len(dxy_data))
            z_vals = np.abs(dxy_data['Close'].diff().fillna(0).values)
            fig3d = go.Figure(data=[go.Scatter3d(
                x=time_bins,
                y=dxy_data['Close'],
                z=z_vals,
                mode='lines+markers',
                marker=dict(size=4, color=z_vals, colorscale='Viridis', opacity=0.8),
                line=dict(color='royalblue', width=2)
            )])
            fig3d.update_layout(
                title="DXY - 3D Price and Volatility Chart",
                scene=dict(
                    xaxis_title="Time",
                    yaxis_title="DXY Price",
                    zaxis_title="|Price Δ|"
                ),
                margin=dict(l=20, r=20, t=40, b=20),
                template=st.session_state.get('chart_theme', 'plotly_dark'),
                height=440,
                paper_bgcolor="rgba(0,0,0,0.05)",
                plot_bgcolor="rgba(0,0,0,0.05)",
            )
            # Only show the Plotly chart title for the DXY 3D chart (remove st.markdown title)
            st.plotly_chart(fig3d, use_container_width=True)

        st.markdown("<hr style='margin-top:1.5rem'>", unsafe_allow_html=True)

        latest_yields, previous_yields = self.get_10y_yields()
        st.markdown("""
<style>
.yields-table {
    background: rgba(26,34,45,0.85);
    color: #e7eaf0;
    font-size: 1.05rem;
    border-radius: 8px;
    border: 1px solid rgba(37,48,71,0.4);
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
        st.markdown("""
<div style='
    background-color: rgba(0, 0, 0, 0.25);
    padding: 1.1rem;
    margin: 0.8rem 0 1.4rem 0;
    border-radius: 12px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.12);
'>
""", unsafe_allow_html=True)
        cols = st.columns(len(latest_yields))
        for i, (country, val) in enumerate(latest_yields.items()):
            prev_val = previous_yields.get(country)
            delta = None
            if prev_val is not None and val != "N/A":
                delta = round(val - prev_val, 3)
            cols[i].metric(country, f"{val}%" if val != 'N/A' else val, delta)
        st.markdown("</div>", unsafe_allow_html=True)

        # Move the "About Zanalytics Trading Frameworks" expander here, after yields table, before datasets and footer
        with st.expander("ℹ️ About Zanalytics Trading Frameworks"):
            st.markdown("""
            **Wyckoff Methodology:**  
            - Analyzes price and volume to identify the four classic market phases: Accumulation, Markup, Distribution, and Markdown.
            - Tracks composite operator (CO) behavior, supply/demand dynamics, and timing of breakouts using patterns like springs and upthrusts.

            **Smart Money Concepts (SMC):**  
            - Focuses on institutional order flow, mapping liquidity pools, inducements, and engineered stop hunts.
            - Highlights “order blocks” where banks and funds accumulate/distribute positions.

            **Microstructure & Volume Analytics:**  
            - Provides tick-level delta, spread, and footprint charts.
            - Reveals hidden buying/selling pressure and identifies value areas and volume imbalances.

            This dashboard is designed for professionals who demand a statistical, repeatable approach to discretionary or systematic trading.
            """)

        # Insert Zanalytics expander (details) immediately after About Zanalytics Trading Frameworks
        with st.expander("ℹ️ What is Zanalytics? (Click to expand details)"):
            st.markdown("""
            <div style='font-size:1.02rem; color:#e7eaf0;'>
            <b>Institutional-Grade Analytics for Traders & Portfolio Managers</b>
            <br><br>
            This dashboard integrates advanced trading frameworks including
            <b>Wyckoff Methodology</b>, <b>Smart Money Concepts (SMC)</b>, and <b>volume microstructure analysis</b>.
            <br><br>
            Developed for serious traders, it enables deep market phase identification, liquidity zone mapping, and order flow insights,
            supporting decision-making at both tactical and strategic levels.
            <br><br>
            <b>Core Features:</b><br>
            • <b>Wyckoff Analysis:</b> Detect accumulation, distribution, springs, upthrusts, and phase transitions.<br>
            • <b>Smart Money Concepts:</b> Map institutional liquidity pools, order blocks, inducements, and market structure shifts.<br>
            • <b>Microstructure Tools:</b> Tick-level volume, delta, spread, and execution flow visualization.<br>
            • <b>Technical Confluence:</b> Multi-timeframe screening and cross-asset overlays.
            </div>
            """, unsafe_allow_html=True)

        # Horizontal rule before available datasets
        st.markdown("---")
        # ─── Available datasets (bottom, plain) ────────────────────────────────
        self.display_available_data(data_sources)

        st.markdown(
            "<div style='text-align:center; color:#8899a6; font-size:0.97rem; margin-top:2.5rem;'>"
            "© 2025 Zanalytics. Powered by institutional-grade market microstructure analytics.<br>"
            "<span style='font-size:0.92rem;'>Data and visualizations for professional and educational use only.</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.success(f"📂  Loaded **{len(data_sources)}** asset folders • **{sum(len(v) for v in data_sources.values())}** timeframe files detected")

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
                xaxis_rangeslider_visible=False,
                paper_bgcolor="rgba(0,0,0,0.05)",
                plot_bgcolor="rgba(0,0,0,0.05)",
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
