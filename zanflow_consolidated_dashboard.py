#!/usr/bin/env python3
"""
ZANFLOW ULTIMATE MEGA DASHBOARD v16
The Complete All-in-One Trading Analysis Platform.
This is the user's baseline zanalytics_dasboard_v10.py, with the data
scanning logic corrected to handle the `./data/{pair}` directory structure.
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
import re
from typing import Dict, List, Optional, Tuple, Any
try:
    from scipy import stats
except ImportError:
    stats = None

warnings.filterwarnings('ignore')

class ZANFLOWUltimateMegaDashboard:
    def __init__(self, data_directory="."):
        """Initialize the MEGA dashboard with all features"""
        self.data_dir = Path(data_directory)
        
        # Data storage
        self.symbols_data = {}
        self.available_symbols = []
        self.available_timeframes = {}
        self.tick_data = {}
        self.bar_data = {}
        self.txt_reports = {}
        self.json_reports = {}
        self.microstructure_data = {}
        
        # Color scheme for consistent visualization
        self.colors = {
            'bullish': '#00ff88',
            'bearish': '#ff4444',
            'neutral': '#aaaaaa',
            'supply_zone': '#ff6b6b',
            'demand_zone': '#4ecdc4',
            'fair_value_gap': '#45b7d1',
            'liquidity': '#ffcc00',
            'manipulation': '#ff00ff',
            'wyckoff_accumulation': '#96ceb4',
            'wyckoff_distribution': '#ff9a8b',
            'bos': '#00ffff',
            'choch': '#ff00ff'
        }

    # --- FIXED Data Scanning Logic ---
    def scan_available_symbols(self):
        """
        Robust symbol scanner that correctly works with the `./data/{pair}` directory structure.
        This replaces the original faulty logic from the v10 baseline.
        """
        data_path = self.data_dir / "data"
        if not data_path.exists():
            st.error(f"Directory not found: {data_path}. Please create it and add your symbol folders (e.g., ./data/XAUUSD/).")
            return False

        # Reset data structures
        self.symbols_data = {}
        self.tick_data = {}
        self.bar_data = {}
        self.microstructure_data = {}
        self.txt_reports = {}

        # Iterate through each subdirectory in the data folder. The subdirectory name is the symbol.
        for symbol_dir in data_path.iterdir():
            if symbol_dir.is_dir():
                symbol = symbol_dir.name.upper()
                
                if symbol not in self.symbols_data:
                    self.symbols_data[symbol] = {}
                    self.tick_data[symbol] = {}
                    self.bar_data[symbol] = {}
                    self.txt_reports[symbol] = []
                    self.microstructure_data[symbol] = []

                # Find all data files in the symbol directory
                for file_path in symbol_dir.glob('*.csv'):
                    try:
                        filename = file_path.name
                        
                        # Logic from original script to determine timeframe and type
                        if 'TICK' in filename:
                            if 'tick_tick' in filename: timeframe = 'tick'
                            elif '1min' in filename: timeframe = '1min_tick'
                            elif '5min' in filename: timeframe = '5min_tick'
                            elif '15min' in filename: timeframe = '15min_tick'
                            elif '30min' in filename: timeframe = '30min_tick'
                            else: parts = filename.split('_'); timeframe = parts[-2] if len(parts) > 2 else 'unknown_tick'
                            self.tick_data[symbol][timeframe] = file_path
                        elif 'bars' in filename or 'M1' in filename:
                            if '1min' in filename: timeframe = '1min_bar'
                            elif '5min' in filename: timeframe = '5min_bar'
                            elif '15min' in filename: timeframe = '15min_bar'
                            elif '30min' in filename: timeframe = '30min_bar'
                            elif '1H' in filename: timeframe = '1H_bar'
                            elif '4H' in filename: timeframe = '4H_bar'
                            elif '1D' in filename: timeframe = '1D_bar'
                            else: parts = filename.split('_'); timeframe = parts[-2] if len(parts) > 2 else 'unknown_bar'
                            self.bar_data[symbol][timeframe] = file_path
                        else:
                            parts = filename.split('_'); timeframe = parts[-2] if len(parts) > 2 else 'unknown'
                        
                        self.symbols_data[symbol][timeframe] = file_path
                    except Exception:
                        continue
                
                # Load reports from the symbol directory
                for report_path in symbol_dir.glob('*.txt'):
                    try:
                        with open(report_path, 'r', encoding='utf-8') as f:
                            self.txt_reports[symbol].append({'filename': report_path.name, 'content': f.read(), 'timestamp': report_path.stat().st_mtime})
                    except Exception: continue
                
                for report_path in symbol_dir.glob('*.json'):
                     try:
                        with open(report_path, 'r', encoding='utf-8') as f:
                            self.microstructure_data[symbol].append({'filename': report_path.name, 'data': json.load(f), 'timestamp': report_path.stat().st_mtime})
                     except Exception: continue

        self.available_symbols = sorted(list(self.symbols_data.keys()))
        self.available_timeframes = {s: sorted(list(t.keys())) for s, t in self.symbols_data.items()}
        
        return len(self.available_symbols) > 0

    def load_symbol_data(self, symbol, timeframe):
        """Load data for a specific symbol and timeframe using the correct path."""
        try:
            file_path = self.symbols_data.get(symbol, {}).get(timeframe)
            if file_path and file_path.exists():
                df = pd.read_csv(file_path)
                
                # Handle timestamp column
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                elif df.columns[0] == 'Unnamed: 0':
                    df.index = pd.to_datetime(df[df.columns[0]])
                    df = df.drop(df.columns[0], axis=1)
                
                return df
            return None
        except Exception as e:
            st.error(f"Error loading data for {symbol} {timeframe}: {e}")
            return None

    def load_latest_txt_report(self, symbol):
        """Load the latest TXT report for a symbol"""
        if symbol in self.txt_reports and self.txt_reports[symbol]:
            latest_report = sorted(self.txt_reports[symbol], key=lambda x: x['timestamp'], reverse=True)[0]
            return latest_report['content']
        return None

    def calculate_market_metrics(self, df):
        """Calculate comprehensive market metrics"""
        try:
            if df is None or len(df) == 0:
                return {'spread': 'N/A', 'atr': 'N/A', 'rsi': 'N/A', 'manipulation': 'N/A', 'trend': 'N/A', 'volatility': 'N/A'}
            
            metrics = {}
            
            if 'spread' in df.columns: metrics['spread'] = f"{df['spread'].iloc[-1]:.2f}"
            elif 'spread_price' in df.columns: metrics['spread'] = f"{df['spread_price'].iloc[-1]:.5f}"
            elif 'bid' in df.columns and 'ask' in df.columns: metrics['spread'] = f"{df['ask'].iloc[-1] - df['bid'].iloc[-1]:.5f}"
            else: metrics['spread'] = f"{(df['high'] - df['low']).tail(20).mean():.4f}"
            
            atr_val = np.nan
            for col in ['ATR_14', 'atr_14', 'ATR', 'atr']:
                if col in df.columns and not pd.isna(df[col].iloc[-1]):
                    atr_val = df[col].iloc[-1]; break
            if pd.isna(atr_val) and len(df) > 14:
                high_low = df['high'] - df['low']
                high_close = abs(df['high'] - df['close'].shift(1))
                low_close = abs(df['low'] - df['close'].shift(1))
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                atr_val = true_range.rolling(14).mean().iloc[-1]
            metrics['atr'] = f"{atr_val:.4f}" if not pd.isna(atr_val) else "N/A"
            
            rsi_val = np.nan
            for col in ['RSI_14', 'rsi_14', 'RSI', 'rsi']:
                if col in df.columns and not pd.isna(df[col].iloc[-1]):
                    rsi_val = df[col].iloc[-1]; break
            if pd.isna(rsi_val) and len(df) > 14:
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                rsi_val = 100 - (100 / (1 + rs)).iloc[-1]
            if pd.notna(rsi_val):
                rsi_color = "🟢" if rsi_val < 30 else "🔴" if rsi_val > 70 else "🟡"
                metrics['rsi'] = f"{rsi_color} {rsi_val:.1f}"
            else: metrics['rsi'] = "N/A"
            
            manipulation_score = sum(df[col].tail(50).sum() for col in ['spoofing_detected', 'layering_detected', 'momentum_ignition'] if col in df.columns)
            if manipulation_score > 10: metrics['manipulation'] = "🔴 HIGH"
            elif manipulation_score > 5: metrics['manipulation'] = "🟡 MEDIUM"
            elif manipulation_score > 0: metrics['manipulation'] = "🟢 LOW"
            else: metrics['manipulation'] = "🟢 CLEAN"
            
            if len(df) >= 21:
                short_ma = df['close'].rolling(8).mean().iloc[-1]
                long_ma = df['close'].rolling(21).mean().iloc[-1]
                metrics['trend'] = "🟢 BULL" if short_ma > long_ma else "🔴 BEAR"
            else: metrics['trend'] = "🟡 N/A"
            
            if len(df) > 20:
                vol = df['close'].pct_change().tail(20).std()
                vol_threshold = df['close'].pct_change().std()
                metrics['volatility'] = "🔥 HIGH" if vol > vol_threshold * 1.5 else "❄️ LOW"
            else: metrics['volatility'] = "🟡 N/A"
            
            return metrics
        except Exception:
            return {'spread': 'ERR', 'atr': 'ERR', 'rsi': 'ERR', 'manipulation': 'ERR', 'trend': 'ERR', 'volatility': 'ERR'}

    def create_mega_dashboard(self):
        st.set_page_config(page_title="ZANFLOW MEGA Dashboard v16", page_icon="🚀", layout="wide", initial_sidebar_state="expanded")
        
        st.markdown(
            """
            <style>
            .main-header { background: linear-gradient(135deg, #1F2937 0%, #111827 100%); padding: 2rem; border-radius: 15px; color: white; text-align: center; margin-bottom: 2rem; box-shadow: 0 8px 32px rgba(0,0,0,0.2); }
            </style>
            """, unsafe_allow_html=True
        )
        
        st.markdown("<div class='main-header'><h1>🚀 ZANFLOW ULTIMATE MEGA DASHBOARD v16</h1><p><strong>The Complete Trading Intelligence Platform</strong></p></div>", unsafe_allow_html=True)
        
        if "initialized" not in st.session_state:
            with st.spinner("🔍 Initializing MEGA analysis engine..."):
                if self.scan_available_symbols():
                    st.session_state.initialized = True
                else:
                    st.error("❌ No trading data found. Please ensure your data is in `./data/{SYMBOL}/` folders.")
                    st.stop()
        
        self.create_mega_sidebar()
        
        if st.session_state.get('selected_symbol') and st.session_state.get('selected_timeframe'):
            self.display_mega_analysis()
        else:
            self.display_mega_overview()

    def create_mega_sidebar(self):
        st.sidebar.markdown("# 🎛️ MEGA Control Center")
        
        if self.available_symbols:
            selected_symbol = st.sidebar.selectbox("📈 Select Trading Symbol", sorted(self.available_symbols), key="selected_symbol")
            
            if selected_symbol:
                available_tfs = self.available_timeframes.get(selected_symbol, [])
                if available_tfs:
                    selected_timeframe = st.sidebar.selectbox("⏱️ Select Timeframe", sorted(available_tfs), key="selected_timeframe")
                else:
                    st.sidebar.warning("No timeframes found for this symbol.")
                    st.session_state.selected_timeframe = None
                    return
                
                df = self.load_symbol_data(selected_symbol, selected_timeframe)
                if df is not None:
                    metrics = self.calculate_market_metrics(df)
                    st.sidebar.markdown("---")
                    st.sidebar.markdown("### 🌍 Real-Time Market Status")
                    col1, col2 = st.sidebar.columns(2)
                    with col1:
                        st.metric("📏 Spread", metrics['spread'])
                        st.metric("📊 ATR", metrics['atr'])
                    with col2:
                        st.metric("📈 RSI", metrics['rsi'])
                        st.metric("📈 Trend", metrics['trend'])

        st.sidebar.markdown("---")
        st.session_state['lookback_bars'] = st.sidebar.slider("Lookback Bars", 50, 5000, 500)
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🔬 Analysis Modules")
        st.session_state['show_overview'] = st.sidebar.checkbox("🌍 Market Overview", True)
        st.session_state['show_price_analysis'] = st.sidebar.checkbox("📈 Advanced Price Analysis", True)
        st.session_state['show_microstructure'] = st.sidebar.checkbox("🔍 Microstructure", True)
        st.session_state['show_smc'] = st.sidebar.checkbox("🧠 Smart Money Concepts", True)
        st.session_state['show_wyckoff'] = st.sidebar.checkbox("📈 Wyckoff Analysis", True)
        st.session_state['show_reports'] = st.sidebar.checkbox("📄 Analysis Reports", True)

    def display_mega_overview(self):
        st.markdown("## 🌍 MEGA Market Overview & Intelligence Hub")
        if not self.available_symbols:
            st.warning("No symbols found. Please check your data files.")
            return

        cols = st.columns(3)
        cols[0].metric("📊 Total Symbols", len(self.available_symbols))
        cols[1].metric("⏱️ Total Datasets", sum(len(t) for t in self.available_timeframes.values()))
        cols[2].metric("📄 Analysis Reports", sum(len(self.txt_reports.get(s, [])) + len(self.microstructure_data.get(s,[])) for s in self.available_symbols))
        
        st.markdown("### 📊 Symbol Intelligence Dashboard")
        symbol_data = []
        for symbol in sorted(self.available_symbols):
            timeframes = self.available_timeframes.get(symbol, [])
            symbol_data.append({
                'Symbol': symbol,
                'Timeframes': len(timeframes),
                'Available Timeframes': ', '.join(sorted(timeframes)),
                'Has Reports': '✅' if self.txt_reports.get(symbol) or self.microstructure_data.get(symbol) else '❌'
            })
        if symbol_data:
            st.dataframe(pd.DataFrame(symbol_data), use_container_width=True)

    def display_mega_analysis(self):
        symbol = st.session_state.get('selected_symbol')
        timeframe = st.session_state.get('selected_timeframe')
        
        if not symbol or not timeframe: return

        df = self.load_symbol_data(symbol, timeframe)
        if df is None: return
        
        df_display = df.tail(st.session_state.lookback_bars).copy()
        
        st.markdown(f"# 🚀 {symbol} - {timeframe} MEGA Analysis Suite")
        
        if st.session_state.get('show_overview', True):
            self.display_mega_market_status(df_display, symbol)
        
        if st.session_state.get('show_price_analysis', True):
            self.create_mega_price_chart(df_display, symbol, timeframe)
        
        tabs_to_show = []
        if st.session_state.get('show_microstructure', True): tabs_to_show.append("🔍 Microstructure")
        if st.session_state.get('show_smc', True): tabs_to_show.append("🧠 SMC")
        if st.session_state.get('show_wyckoff', True): tabs_to_show.append("📈 Wyckoff")
        if st.session_state.get('show_reports', True): tabs_to_show.append("📄 Reports")
        
        if tabs_to_show:
            tabs = st.tabs(tabs_to_show)
            tab_map = {name: tab for name, tab in zip(tabs_to_show, tabs)}
            
            if "🔍 Microstructure" in tab_map:
                with tab_map["🔍 Microstructure"]: self.create_microstructure_analysis(df_display, symbol, timeframe)
            if "🧠 SMC" in tab_map:
                with tab_map["🧠 SMC"]: self.create_smc_analysis(df_display)
            if "📈 Wyckoff" in tab_map:
                with tab_map["📈 Wyckoff"]: self.create_wyckoff_analysis(df_display)
            if "📄 Reports" in tab_map:
                with tab_map["📄 Reports"]: self.display_txt_report_analysis(symbol)

    def display_mega_market_status(self, df, symbol):
        st.markdown('<div class="market-overview">', unsafe_allow_html=True)
        st.markdown("## 🌍 Real-Time Market Intelligence")
        metrics = self.calculate_market_metrics(df)
        cols = st.columns(7)
        price_col = 'close' if 'close' in df.columns else 'mid_price'
        current_price = df[price_col].iloc[-1]
        cols[0].metric("💰 Price", f"{current_price:.4f}")
        cols[1].metric("📏 Spread", metrics['spread'])
        cols[2].metric("📊 ATR", metrics['atr'])
        cols[3].metric("📈 RSI", metrics['rsi'])
        cols[4].metric("📈 Trend", metrics['trend'])
        cols[5].metric("🔥 Volatility", metrics['volatility'])
        cols[6].metric("🛡️ Manipulation", metrics['manipulation'])
        st.markdown('</div>', unsafe_allow_html=True)

    def create_mega_price_chart(self, df, symbol, timeframe):
        st.markdown("## 📈 Advanced Price Action & Technical Analysis")
        is_tick_data = 'mid_price' in df.columns or ('bid' in df.columns and 'ask' in df.columns)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.8, 0.2])
        
        if is_tick_data:
            if 'mid_price' in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df['mid_price'], mode='lines', name='Mid Price'), row=1, col=1)
        else:
            fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="Price"), row=1, col=1)
            
        if 'volume' in df.columns:
            fig.add_trace(go.Bar(x=df.index, y=df['volume'], name='Volume', marker_color='grey', opacity=0.5), row=2, col=1)

        fig.update_layout(title_text=f"{symbol} {timeframe} - Price Action", template="plotly_dark", height=700, xaxis_rangeslider_visible=False, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    def create_microstructure_analysis(self, df, symbol, timeframe):
        st.subheader("Microstructure & Order Flow Analysis")
        st.info("Microstructure analysis module.")
        if 'spread' in df.columns:
            st.metric("Average Spread", f"{df['spread'].mean():.4f}")

    def create_smc_analysis(self, df_display):
        st.subheader("Smart Money Concepts Analysis")
        st.info("SMC analysis module.")
        if 'SMC_fvg_bullish' in df_display.columns:
            st.metric("Bullish FVGs", df_display['SMC_fvg_bullish'].sum())

    def create_wyckoff_analysis(self, df_display):
        st.subheader("Wyckoff Method Analysis")
        st.info("Wyckoff analysis module.")
        if 'wyckoff_accumulation' in df_display.columns:
            st.metric("Accumulation Signals", df_display['wyckoff_accumulation'].sum())

    def create_pattern_analysis(self, df_display):
        st.subheader("Pattern Recognition Analysis")
        st.info("Pattern analysis module.")

    def create_volume_analysis(self, df_display):
        st.subheader("Advanced Volume Analysis")
        st.info("Volume analysis module.")

    def create_advanced_analytics(self, df_display):
        st.subheader("Advanced Analytics (ML & Stats)")
        st.info("Advanced analytics module.")

    def display_txt_report_analysis(self, symbol):
        """Display analysis from TXT reports"""
        st.subheader("📄 Latest Analysis Report")
        txt_content = self.load_latest_txt_report(symbol)
        if txt_content:
            st.text_area("Report", txt_content, height=400)
        else:
            st.info(f"No analysis report found for {symbol}")

if __name__ == "__main__":
    dashboard = ZANFLOWUltimateMegaDashboard()
    dashboard.create_mega_dashboard()
