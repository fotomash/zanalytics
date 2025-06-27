#!/usr/bin/env python3
"""
Economic Data Integration for ZANFLOW v12 Ultimate Trading Dashboard
Integrates macro sentiment analysis with intermarket data
"""

import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# --- [Placeholder Base Class] ---
# This class provides the necessary structure for the main dashboard to inherit from.
class UltimateZANFLOWDashboard:
    """
    A base class for the ZANFLOW Trading Dashboard.
    This is a placeholder to ensure the extended class works.
    """
    def __init__(self, data_directory: str):
        st.set_page_config(layout="wide", page_title="ZANFLOW v12 Dashboard")
        self.data_directory = data_directory
        self.pairs_data = {}  # Initialize attribute needed by child class
        self.initialize_session_state()

    def initialize_session_state(self):
        """Initialize Streamlit session state keys to prevent errors."""
        if 'chart_theme' not in st.session_state:
            st.session_state['chart_theme'] = 'plotly_dark'
        if 'selected_pair' not in st.session_state:
            st.session_state['selected_pair'] = 'EUR/USD'
        if 'show_macro_sentiment' not in st.session_state:
            st.session_state['show_macro_sentiment'] = True
        if 'show_intermarket' not in st.session_state:
            st.session_state['show_intermarket'] = True
        if 'api_logs' not in st.session_state:
            st.session_state['api_logs'] = []


    def create_sidebar_controls(self):
        """Create basic sidebar controls for the dashboard."""
        st.sidebar.markdown("## 📊 ZANFLOW Controls")
        st.session_state['selected_pair'] = st.sidebar.selectbox(
            "Select Currency Pair",
            ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'USD/CAD'],
            key='base_selected_pair'
        )

    def display_market_overview(self):
        """Display the basic market overview section."""
        st.markdown("---")
        st.markdown("### 📈 Market Overview")
        st.write(f"Displaying base analysis for **{st.session_state['selected_pair']}**.")

    def display_ultimate_analysis(self):
        """Placeholder for the ultimate analysis display."""
        st.markdown("---")
        st.markdown(f"### 🎯 Ultimate Analysis for {st.session_state['selected_pair']}")
        st.write("This section would contain detailed technical and fundamental analysis.")

# --- [Economic Data Manager Class] ---
class EconomicDataManager:
    """
    Manages economic data fetching and analysis, using the Twelve Data API.
    """
    def __init__(self, twelve_data_api_key: str):
        self.twelve_data_key = twelve_data_api_key
        self.base_url_twelve = "https://api.twelvedata.com"
        self.cache = {}
        st.session_state.api_logs = [] # Reset logs on initialization

    def _log_status(self, symbol: str, success: bool, message: str):
        """Logs the status of an API call to the session state."""
        log_entry = f"{'✅' if success else '❌'} {symbol}: {message}"
        st.session_state.api_logs.append(log_entry)

    def _get_generic_quote(self, symbols: List[str], name: str) -> Dict[str, Any]:
        """
        A more robust function to fetch quote data.
        It tries a list of symbols until one succeeds.
        """
        if isinstance(symbols, str):
            symbols = [symbols] # Allow passing a single string for convenience

        for symbol in symbols:
            try:
                url = f"{self.base_url_twelve}/quote?symbol={symbol}&apikey={self.twelve_data_key}"
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                if data.get('code') == 404 or 'close' not in data or data.get('close') is None:
                    raise ValueError(data.get('message', f'Symbol not found or invalid response for {symbol}'))

                self._log_status(symbol, True, "Success")
                return {
                    'name': name, 'symbol': symbol,
                    'current': float(data['close']),
                    'change': float(data.get('change', 0)),
                    'change_pct': float(data.get('percent_change', 0))
                }
            except Exception as e:
                self._log_status(symbol, False, str(e))
                continue # Try the next symbol in the list
        
        # If the loop completes without returning, all symbols failed.
        return {'name': name, 'symbol': symbols[0], 'current': 0, 'change': 0, 'change_pct': 0}

    def fetch_macro_snapshot(self) -> Dict[str, Any]:
        """Fetch comprehensive macro sentiment snapshot"""
        st.session_state.api_logs = [] # Clear previous logs before fetching
        try:
            snapshot = {
                'timestamp': datetime.now(),
                # [REVISED] Fetch only SVIX
                'svix': self._get_generic_quote("SVIX", "Short VIX (SVIX)"),
            }
            self.cache['macro_snapshot'] = snapshot
            return snapshot
        except Exception as e:
            st.error(f"Error fetching macro data: {e}")
            return self.get_cached_or_default()

    def get_cached_or_default(self) -> Dict[str, Any]:
        return {'timestamp': datetime.now(), 'svix': {'name': 'Short VIX (SVIX)', 'symbol': 'SVIX', 'current': 0, 'change': 0, 'change_pct': 0}}

# --- [Main Dashboard Class] ---
class UltimateZANFLOWDashboardWithEconomics(UltimateZANFLOWDashboard):
    """Extended ZANFLOW Dashboard with Economic Data Integration"""

    def __init__(self, data_directory="."):
        super().__init__(data_directory)
        try:
            self.economic_manager = EconomicDataManager(
                twelve_data_api_key=st.secrets["twelve_data_api_key"]
            )
        except (FileNotFoundError, KeyError):
            st.error("API key 'twelve_data_api_key' not found in .streamlit/secrets.toml.")
            st.stop()
        except Exception as e:
            st.error(f"An unexpected error occurred during initialization: {e}")
            st.stop()
        self.macro_data = None

    def display_macro_sentiment_overview(self):
        """Display macro sentiment overview"""
        st.markdown("## 🌍 Macro Sentiment")
        if self.macro_data is None:
            with st.spinner("Loading market data..."):
                self.macro_data = self.economic_manager.fetch_macro_snapshot()

        if self.macro_data:
            # --- [REVISED] SVIX Indicator Only ---
            svix_data = self.macro_data.get('svix', {})
            st.metric("Short VIX (SVIX)", f"{svix_data.get('current', 0):.1f}", f"{svix_data.get('change', 0):+.2f}", delta_color="normal")
            
            # --- Data Status Expander ---
            with st.expander("Data Status & Logs"):
                for log in st.session_state.get('api_logs', []):
                    st.write(log)


    def create_main_dashboard(self):
        """The main method to build and display the Streamlit dashboard."""
        st.title("ZANFLOW v12 Ultimate Trading Dashboard")
        self.create_sidebar_controls()
        self.display_market_overview()
        if st.session_state.get('show_macro_sentiment', True):
            self.display_macro_sentiment_overview()
        self.display_ultimate_analysis()

# --- [Application Entry Point] ---
def main():
    """Main application entry point with economic integration"""
    dashboard = UltimateZANFLOWDashboardWithEconomics()
    dashboard.create_main_dashboard()

if __name__ == "__main__":
    main()
