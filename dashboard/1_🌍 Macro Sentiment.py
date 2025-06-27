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
import requests
import os
import glob
from pathlib import Path
from datetime import datetime, timedelta
import warnings
import re
from typing import Dict, List, Optional, Tuple, Any, Union
from bs4 import BeautifulSoup  # Added for web scraping
import base64  # Added for image encoding

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')


# --- Utility Function for Background Image ---
def get_image_as_base64(path):
    """Reads an image file and returns its base64 encoded string."""
    try:
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except FileNotFoundError:
        # This warning will appear on the dashboard if the image is not found.
        st.warning(f"Background image not found at '{path}'. Please ensure it's in the same directory as the script.")
        return None


# --- Economic Data Manager ---
class EconomicDataManager:
    """ Manages fetching live economic data. """

    def __init__(self, api_key: Optional[str]):
        self.api_key = api_key
        self.base_url = "https://api.twelvedata.com"

    def get_svix_quote(self) -> Dict[str, Any]:
        """ Fetches the latest quote for SVIX (Short VIX). """
        if not self.api_key:
            return {'name': 'SVIX', 'error': 'API key not set'}
        try:
            symbol = "SVIX"
            url = f"{self.base_url}/quote?symbol={symbol}&apikey={self.api_key}"
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            if data and 'close' in data and data['close'] is not None:
                return {'name': 'SVIX', 'current': float(data['close']), 'change': float(data.get('change', 0))}
            return {'name': 'SVIX', 'error': f'Symbol {symbol} not found'}
        except requests.exceptions.RequestException as e:
            return {'name': 'SVIX', 'error': "Network error"}
        except Exception as e:
            return {'name': 'SVIX', 'error': str(e)}

    def get_economic_events(self) -> pd.DataFrame:
        """
        Fetches upcoming high-impact economic events by scraping Trading Economics.
        """
        try:
            url = "https://tradingeconomics.com/calendar"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')
            table = soup.find('table', {'id': 'calendar'})

            events = []
            if table:
                rows = table.find_all('tr', {'data-importance': '3'})
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) > 3:
                        time = cols[0].text.strip()
                        country = cols[1].text.strip()
                        event_text = cols[2].text.strip()
                        event_text = re.sub(r'\s{2,}', ' ', event_text)
                        events.append({'Time': time, 'Country': country, 'Event': event_text})

            return pd.DataFrame(events) if events else pd.DataFrame()
        except Exception:
            return pd.DataFrame()


class MarketOverviewDashboard:
    """
    Encapsulates all functionality for the Market Overview Dashboard.
    """

    def __init__(self, data_directory="./data"):
        self.data_dir = Path(data_directory)
        self.supported_pairs = ["XAUUSD", "BTCUSD", "EURUSD", "GBPUSD", "USDJPY", "ETHUSD", "USDCAD", "AUDUSD",
                                "NZDUSD"]
        self.timeframes = ["1min", "5min", "15min", "30min", "1H", "4H", "1D", "1W"]

        try:
            api_key = st.secrets.get("twelve_data_api_key")
        except (FileNotFoundError, KeyError):
            api_key = None
        self.economic_manager = EconomicDataManager(api_key)

        # --- Initialize Session State for Navigation ---
        if 'chart_theme' not in st.session_state:
            st.session_state.chart_theme = 'plotly_dark'
        if 'current_page' not in st.session_state:
            st.session_state.current_page = "Home"

    def run(self):
        st.set_page_config(page_title="ZANFLOW Dashboard", page_icon="🚀", layout="wide",
                           initial_sidebar_state="expanded")

        # Note: Ensure 'image_af247b.jpg' is in the same directory as this script.
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
            .stMetric {{ border-radius: 10px; padding: 15px; background-color: #2a2a39; }}
            #MainMenu {{visibility: hidden;}}
            footer {{visibility: hidden;}}
            </style>
            """
            st.markdown(background_style, unsafe_allow_html=True)

        with st.spinner("🛰️ Scanning all data sources..."):
            data_sources = self.scan_all_data_sources()

        self.create_sidebar_navigation()

        # Page routing
        if st.session_state.current_page == "Home":
            self.display_home_page(data_sources)
        elif st.session_state.current_page == "Market Overview":
            self.display_market_overview(data_sources)

    def create_sidebar_navigation(self):
        """Create the main sidebar navigation."""
        st.sidebar.title("🧭 Navigation")
        pages = ["Home", "Market Overview"]

        for page in pages:
            if st.sidebar.button(page, key=f"nav_{page}"):
                st.session_state.current_page = page

    def display_home_page(self, data_sources):
        """Displays the main home/landing page of the dashboard."""
        st.markdown("## 🏠 Welcome to ZANFLOW Ultimate Trading Dashboard")

        market_data = self._load_market_data(data_sources)

        # Dashboard Summary Section
        st.markdown("### Ultimate Trading Dashboard")
        st.caption("""
        This comprehensive trading analytics platform provides advanced analysis tools for traders, including: 
        Global Market Analysis, Pair Analysis, Smart Money Concepts, Wyckoff Analysis, Microstructure, and Risk Analytics.
        """)

        col1, col2, col3, col4 = st.columns(4)
        total_datasets = sum(len(tfs) for tfs in market_data.values())
        total_bars = sum(len(df) for tfs in market_data.values() for df in tfs.values())

        col1.metric("Available Pairs", len(market_data))
        col2.metric("Total Datasets", total_datasets)
        col3.metric("Total Bars", f"{total_bars:,}")
        col4.metric("Last Update", datetime.now().strftime('%H:%M:%S'))

        st.markdown("---")

        # Top Movers Section
        self.display_top_movers(market_data)

    def display_top_movers(self, market_data):
        """Calculates and displays top gainers and losers."""
        st.markdown("### 🚀 Top Movers")
        movers_data = []
        for pair, timeframes in market_data.items():
            if timeframes:
                tf = sorted(timeframes.keys(), key=lambda t: pd.to_timedelta(
                    t.replace('min', 'T').replace('H', 'H').replace('D', 'D').replace('W', 'W')))[0]
                df = timeframes[tf]

                # Defensive check for 'close' column and sufficient data
                if 'close' in df.columns and len(df) > 10:
                    try:
                        recent_data = df.tail(10)
                        performance = ((recent_data['close'].iloc[-1] / recent_data['close'].iloc[0]) - 1) * 100
                        rsi = recent_data['rsi_14'].iloc[-1] if 'rsi_14' in recent_data.columns and not recent_data[
                            'rsi_14'].empty else np.nan
                        movers_data.append({'Pair': pair, 'Timeframe': tf, 'Performance': performance, 'RSI': rsi})
                    except KeyError as e:
                        st.error(f"Missing column for Top Movers calculation in {pair}/{tf}: {e}")
                        continue  # Skip this dataframe

        if movers_data:
            movers_df = pd.DataFrame(movers_data).sort_values('Performance', ascending=False)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 📈 Top Gainers")
                for _, row in movers_df.head(5).iterrows():
                    rsi_val = f"RSI: {row['RSI']:.1f}" if pd.notna(row['RSI']) else ""
                    st.markdown(f"**{row['Pair']} ({row['Timeframe']})**: {row['Performance']:+.2f}% {rsi_val}")

            with col2:
                st.markdown("#### 📉 Top Losers")
                # Correctly get the N smallest values
                for _, row in movers_df.nsmallest(5, 'Performance').iterrows():
                    rsi_val = f"RSI: {row['RSI']:.1f}" if pd.notna(row['RSI']) else ""
                    st.markdown(f"**{row['Pair']} ({row['Timeframe']})**: {row['Performance']:+.2f}% {rsi_val}")
        else:
            st.info("Not enough data to calculate Top Movers.")

    def display_market_overview(self, data_sources):
        """Display a high-level overview of the market."""
        st.markdown("## 📈 Market Overview & Intelligence")

        market_data = self._load_market_data(data_sources)
        if not market_data:
            st.warning("No market data available for overview. Please check the `./data` directory.")
            return

        self.create_market_heatmap(market_data)

        col1, col2 = st.columns([0.6, 0.4])
        with col1:
            self.display_macro_sentiment()
            self.display_key_economic_events()
        with col2:
            self.create_correlation_matrix(market_data)

    def _load_market_data(self, data_sources):
        """Helper to load all market data."""
        market_data = {}
        for pair, files in data_sources.items():
            if pair not in market_data: market_data[pair] = {}
            for tf, file_path in files.items():
                df = self.load_comprehensive_data(file_path, max_records=50)
                if df is not None and not df.empty:
                    market_data[pair][tf] = df
        return market_data

    def display_macro_sentiment(self):
        st.markdown("### 🧭 Macro Sentiment Overview")
        svix_data = self.economic_manager.get_svix_quote()
        col1, col2, col3 = st.columns(3)
        with col1:
            if svix_data.get('error'):
                st.metric("SVIX (Short VIX)", "N/A", delta=svix_data['error'], delta_color="off")
            else:
                st.metric("SVIX (Short VIX)", f"{svix_data.get('current', 0):.2f}",
                          f"{svix_data.get('change', 0):+.2f}", delta_color="normal")
        with col2:
            st.metric("DXY (U.S. Dollar Index)", "Flat/Weak", "No flight-to-safety")
        with col3:
            st.metric("Treasury Yields", "Rising", "Growth expectations")

    def display_key_economic_events(self):
        st.markdown("### 🗓️ Key Economic Events (High Impact)")
        events_df = self.economic_manager.get_economic_events()
        if not events_df.empty:
            st.dataframe(events_df, use_container_width=True, hide_index=True, height=200)
        else:
            st.info("No high-impact economic events found or scraper failed.")

    def create_market_heatmap(self, market_data):
        st.markdown("### Multi-Timeframe Momentum Heatmap")
        heatmap_data = {pair: {tf: np.nan for tf in self.timeframes} for pair in self.supported_pairs}
        for pair, tfs_data in market_data.items():
            for tf, df in tfs_data.items():
                if 'close' in df.columns and len(df) >= 2:
                    last_close, prev_close = df['close'].iloc[-1], df['close'].iloc[-2]
                    if prev_close > 0:
                        heatmap_data[pair][tf] = ((last_close / prev_close) - 1) * 100
        heatmap_df = pd.DataFrame(heatmap_data).T[self.timeframes].dropna(how='all')
        if not heatmap_df.empty:
            fig = px.imshow(heatmap_df, text_auto=".2f%", aspect="auto", color_continuous_scale='RdYlGn',
                            labels=dict(x="Timeframe", y="Pair", color="Momentum %"),
                            title="Momentum of Most Recent Candle on Each Timeframe")
            fig.update_layout(template=st.session_state.get('chart_theme', 'plotly_dark'), height=500,
                              coloraxis_showscale=False)
            fig.update_xaxes(side="top")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data to generate heatmap.")

    def create_correlation_matrix(self, market_data):
        st.markdown("### Price Correlation Matrix (Daily)")
        daily_returns = {pair: data['1D']['close'].pct_change() for pair, data in market_data.items() if
                         '1D' in data and 'close' in data['1D'].columns}
        returns_df = pd.DataFrame(daily_returns).tail(30)
        if len(returns_df.columns) >= 2:
            corr_matrix = returns_df.corr()
            fig = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r')
            fig.update_layout(template=st.session_state.get('chart_theme', 'plotly_dark'), height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data. Requires daily data for at least two pairs.")

    def scan_all_data_sources(self):
        data_sources = {}
        all_files = glob.glob(os.path.join(self.data_dir, "**", "*.*"), recursive=True)
        for f_path in all_files:
            for pair in self.supported_pairs:
                if pair in f_path and f_path.endswith(('.csv', '.parquet')):
                    for tf in self.timeframes:
                        if tf in f_path:
                            if pair not in data_sources: data_sources[pair] = {}
                            data_sources[pair][tf] = f_path
                            break
        return data_sources

    def load_comprehensive_data(self, file_path, max_records=None):
        try:
            df = pd.read_parquet(file_path) if file_path.endswith('.parquet') else pd.read_csv(file_path, sep=None,
                                                                                               engine='python')
            df.columns = [col.lower().strip() for col in df.columns]
            for col in ['timestamp', 'datetime', 'date']:
                if col in df.columns:
                    df.set_index(pd.to_datetime(df[col]), inplace=True)
                    df.drop(columns=[col], inplace=True)
                    break
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}, inplace=True)

            if 'rsi_14' not in df.columns and 'close' in df.columns and len(df) >= 15:
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['rsi_14'] = 100 - (100 / (1 + rs))

            if max_records: df = df.tail(max_records)
            return df
        except Exception:
            return None


if __name__ == "__main__":
    dashboard = MarketOverviewDashboard()
    dashboard.run()
