#!/usr/bin/env python3
"""
ZANFLOW v13 ULTIMATE MEGA DASHBOARD
Comprehensive Tick-Level Microstructure + SMC + Wyckoff + Top-Down Analysis
Feeds from ./data/{symbol} directory structure with microstructure analysis
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

class UltimateMicrostructureZANFLOW:
    def __init__(self, data_directory="./data"):
        """Initialize ultimate dashboard with local data directory"""
        self.data_dir = Path(data_directory)
        self.symbols_data = {}
        self.microstructure_data = {}
        self.tick_data = {}
        self.bar_data = {}
        self.analysis_reports = {}

        # Timeframe mappings
        self.timeframe_mapping = {
            'tick': 'tick_tick_processed.csv',
            '1min': '1min_csv_processed.csv', 
            '5min': '5min_csv_processed.csv',
            '15min': '15min_csv_processed.csv',
            '30min': '30min_csv_processed.csv',
            '1H': '1H_csv_processed.csv',
            '4H': '4H_csv_processed.csv',
            '1D': '1D_csv_processed.csv'
        }

    def load_symbol_data(self, symbol="XAUUSD"):
        """Load all data for a specific symbol from directory structure"""
        symbol_dir = self.data_dir / symbol

        if not symbol_dir.exists():
            st.warning(f"Symbol directory {symbol_dir} not found. Using current directory files.")
            symbol_dir = Path(".")

        self.symbols_data[symbol] = {}

        # Load tick data
        tick_files = [
            f"{symbol}_TICK_tick_tick_processed.csv",
            f"{symbol}_TICK_1min_csv_processed.csv",
            f"{symbol}_TICK_5min_csv_processed.csv", 
            f"{symbol}_TICK_15min_csv_processed.csv",
            f"{symbol}_TICK_30min_csv_processed.csv"
        ]

        self.tick_data[symbol] = {}
        for tick_file in tick_files:
            file_path = symbol_dir / tick_file
            if not file_path.exists():
                file_path = Path(tick_file)  # Try current directory

            if file_path.exists():
                try:
                    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                    timeframe = tick_file.split('_')[2] + ("_" + tick_file.split('_')[3] if len(tick_file.split('_')) > 3 else "")
                    self.tick_data[symbol][timeframe] = df
                    st.success(f"✅ Loaded {len(df)} tick records for {timeframe}")
                except Exception as e:
                    st.warning(f"Failed to load {tick_file}: {e}")

        # Load bar data
        bar_files = [
            f"{symbol}_M1_bars_1min_csv_processed.csv",
            f"{symbol}_M1_bars_5min_csv_processed.csv",
            f"{symbol}_M1_bars_15min_csv_processed.csv", 
            f"{symbol}_M1_bars_30min_csv_processed.csv",
            f"{symbol}_M1_bars_1H_csv_processed.csv",
            f"{symbol}_M1_bars_4H_csv_processed.csv",
            f"{symbol}_M1_bars_1D_csv_processed.csv"
        ]

        self.bar_data[symbol] = {}
        for bar_file in bar_files:
            file_path = symbol_dir / bar_file
            if not file_path.exists():
                file_path = Path(bar_file)  # Try current directory

            if file_path.exists():
                try:
                    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                    timeframe = bar_file.split('_')[3]
                    self.bar_data[symbol][timeframe] = df
                    st.success(f"✅ Loaded {len(df)} bars for {timeframe}")
                except Exception as e:
                    st.warning(f"Failed to load {bar_file}: {e}")

        # Load microstructure analysis files
        self.load_microstructure_analysis(symbol, symbol_dir)

        return len(self.tick_data.get(symbol, {})) + len(self.bar_data.get(symbol, {}))

    def load_microstructure_analysis(self, symbol, symbol_dir):
        """Load microstructure analysis JSON and TXT files"""
        microstructure_dir = symbol_dir / "microstructure"
        if not microstructure_dir.exists():
            microstructure_dir = Path(".")  # Try current directory

        self.microstructure_data[symbol] = {
            'json_reports': {},
            'txt_reports': {},
            'analysis_summary': {}
        }

        # Load JSON files
        json_files = list(microstructure_dir.glob("*Microstructure_Analysis*.json"))
        if not json_files:
            json_files = list(Path(".").glob("*Microstructure_Analysis*.json"))

        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    timestamp = json_file.stem.split('_')[-1]
                    self.microstructure_data[symbol]['json_reports'][timestamp] = data
                    st.success(f"✅ Loaded microstructure JSON: {json_file.name}")
            except Exception as e:
                st.warning(f"Failed to load {json_file}: {e}")

        # Load TXT files  
        txt_files = list(microstructure_dir.glob("*Microstructure_Analysis*.txt"))
        if not txt_files:
            txt_files = list(Path(".").glob("*Microstructure_Analysis*.txt"))

        for txt_file in txt_files:
            try:
                with open(txt_file, 'r') as f:
                    content = f.read()
                    timestamp = txt_file.stem.split('_')[-1] 
                    self.microstructure_data[symbol]['txt_reports'][timestamp] = content
                    st.success(f"✅ Loaded microstructure TXT: {txt_file.name}")
            except Exception as e:
                st.warning(f"Failed to load {txt_file}: {e}")

    def create_main_dashboard(self):
        """Create the ultimate microstructure dashboard"""
        st.set_page_config(
            page_title="ZANFLOW v13 Ultimate Microstructure", 
            page_icon="🚀", 
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Enhanced CSS styling
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        .metric-card {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 5px solid #667eea;
            margin: 0.5rem 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .microstructure-section {
            background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
            border: 2px solid #ff9a8b;
        }
        .tick-analysis {
            background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
            padding: 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
        }
        .manipulation-alert {
            background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
            padding: 1rem;
            border-radius: 8px;
            border: 2px solid #ff6b6b;
            margin: 1rem 0;
        }
        .smc-signal {
            background: linear-gradient(135deg, #a8e6cf 0%, #dcedc1 100%);
            padding: 0.8rem;
            border-radius: 8px;
            margin: 0.5rem 0;
            border-left: 4px solid #52b788;
        }
        .wyckoff-phase {
            padding: 0.8rem;
            border-radius: 8px;
            margin: 0.5rem 0;
            font-weight: bold;
            text-align: center;
        }
        .accumulation { background: linear-gradient(135deg, #e8f5e8 0%, #a8e6cf 100%); color: #2d5016; }
        .distribution { background: linear-gradient(135deg, #ffe8e8 0%, #ffcccb 100%); color: #8b0000; }
        .markup { background: linear-gradient(135deg, #e8f8ff 0%, #b3d9ff 100%); color: #003d82; }
        .markdown { background: linear-gradient(135deg, #fff3e0 0%, #ffe0b3 100%); color: #cc5500; }
        </style>
        """, unsafe_allow_html=True)

        # Main header
        st.markdown("""
        <div class="main-header">
        <h1>🚀 ZANFLOW v13 Ultimate Microstructure Dashboard</h1>
        <p>Advanced Tick-Level Analysis • Market Manipulation Detection • Smart Money Concepts • Wyckoff Methodology</p>
        <p><strong>Real-time Microstructure Intelligence & Multi-Timeframe Analysis</strong></p>
        </div>
        """, unsafe_allow_html=True)

        # Load data
        symbol = st.sidebar.selectbox("📊 Select Symbol", ["XAUUSD"], key="symbol_select")

        with st.spinner("🔄 Loading comprehensive market data..."):
            data_loaded = self.load_symbol_data(symbol)

        if data_loaded == 0:
            st.error("❌ No data found. Please ensure data files are in ./data/{symbol}/ directory")
            return

        st.success(f"✅ Successfully loaded {data_loaded} datasets for {symbol}")

        # Sidebar controls
        self.create_enhanced_sidebar_controls(symbol)

        # Main dashboard content
        if st.session_state.get('show_overview', True):
            self.display_market_overview(symbol)

        if st.session_state.get('show_microstructure', True):
            self.display_microstructure_dashboard(symbol)

        if st.session_state.get('show_multi_timeframe', True):
            self.display_multi_timeframe_analysis(symbol)

        if st.session_state.get('show_smc_analysis', True):
            self.display_advanced_smc_analysis(symbol)

        if st.session_state.get('show_wyckoff_analysis', True):
            self.display_wyckoff_analysis(symbol)

    def create_enhanced_sidebar_controls(self, symbol):
        """Create enhanced sidebar with all controls"""
        st.sidebar.title("🎛️ Ultimate Analysis Control Center")

        # Data overview
        st.sidebar.markdown("### 📊 Data Overview")
        tick_datasets = len(self.tick_data.get(symbol, {}))
        bar_datasets = len(self.bar_data.get(symbol, {}))
        microstructure_reports = len(self.microstructure_data.get(symbol, {}).get('json_reports', {}))

        st.sidebar.info(f"""
        **📈 Tick Datasets**: {tick_datasets}  
        **📊 Bar Datasets**: {bar_datasets}  
        **🔬 Microstructure Reports**: {microstructure_reports}
        """)

        # Analysis sections
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🔬 Analysis Sections")

        st.session_state['show_overview'] = st.sidebar.checkbox("🌍 Market Overview", True)
        st.session_state['show_microstructure'] = st.sidebar.checkbox("🔍 Microstructure Analysis", True)
        st.session_state['show_multi_timeframe'] = st.sidebar.checkbox("⏱️ Multi-Timeframe Analysis", True)
        st.session_state['show_smc_analysis'] = st.sidebar.checkbox("🧠 Smart Money Concepts", True)
        st.session_state['show_wyckoff_analysis'] = st.sidebar.checkbox("📈 Wyckoff Analysis", True)

        # Advanced options
        st.sidebar.markdown("---")
        st.sidebar.markdown("### ⚙️ Advanced Options")

        st.session_state['show_manipulation_detection'] = st.sidebar.checkbox("🚨 Manipulation Detection", True)
        st.session_state['show_order_flow'] = st.sidebar.checkbox("💹 Order Flow Analysis", True)
        st.session_state['show_liquidity_analysis'] = st.sidebar.checkbox("💧 Liquidity Analysis", True)
        st.session_state['show_harmonic_patterns'] = st.sidebar.checkbox("🎵 Harmonic Patterns", True)

        # Display settings
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🎨 Display Settings")

        st.session_state['chart_theme'] = st.sidebar.selectbox(
            "Chart Theme", 
            ["plotly_dark", "plotly_white", "ggplot2", "seaborn"],
            index=0
        )

        st.session_state['lookback_ticks'] = st.sidebar.slider(
            "Tick Lookback Period", 100, 10000, 1000
        )

        st.session_state['lookback_bars'] = st.sidebar.slider(
            "Bar Lookback Period", 50, 2000, 500
        )

    def display_market_overview(self, symbol):
        """Display comprehensive market overview"""
        st.markdown("## 🌍 Market Overview & Real-Time Status")

        # Get latest data from multiple timeframes
        latest_data = {}

        # Get tick data if available
        if symbol in self.tick_data and self.tick_data[symbol]:
            for tf, df in self.tick_data[symbol].items():
                if len(df) > 0:
                    latest_data[f"tick_{tf}"] = df.iloc[-1]

        # Get bar data if available  
        if symbol in self.bar_data and self.bar_data[symbol]:
            for tf, df in self.bar_data[symbol].items():
                if len(df) > 0:
                    latest_data[f"bar_{tf}"] = df.iloc[-1]

        if not latest_data:
            st.warning("No data available for overview")
            return

        # Market status cards
        col1, col2, col3, col4, col5 = st.columns(5)

        # Get current price from tick data if available
        current_price = None
        if f"tick_tick" in latest_data:
            current_price = latest_data[f"tick_tick"]['mid_price']
        elif f"bar_1min" in latest_data:
            current_price = latest_data[f"bar_1min"]['close']

        with col1:
            if current_price:
                st.metric("💰 Current Price", f"{current_price:.4f}")
            else:
                st.metric("💰 Current Price", "N/A")

        with col2:
            # Spread analysis from tick data
            if f"tick_tick" in latest_data:
                spread = latest_data[f"tick_tick"]['spread_price']
                st.metric("📏 Spread", f"{spread:.5f}")
            else:
                st.metric("📏 Spread", "N/A")

        with col3:
            # Volatility from bar data
            if f"bar_1H" in latest_data:
                atr = latest_data[f"bar_1H"].get('ATR_14', 0)
                st.metric("📊 ATR (1H)", f"{atr:.4f}")
            else:
                st.metric("📊 ATR", "N/A")

        with col4:
            # RSI from bar data
            if f"bar_1H" in latest_data:
                rsi = latest_data[f"bar_1H"].get('RSI_14', 50)
                rsi_color = "🟢" if rsi < 30 else "🔴" if rsi > 70 else "🟡"
                st.metric("📈 RSI", f"{rsi_color} {rsi:.1f}")
            else:
                st.metric("📈 RSI", "N/A")

        with col5:
            # Market manipulation detection
            if f"tick_tick" in latest_data:
                spoofing = latest_data[f"tick_tick"]['spoofing_detected']
                layering = latest_data[f"tick_tick"]['layering_detected']
                manipulation_status = "🚨 DETECTED" if spoofing or layering else "✅ CLEAN"
                st.metric("🛡️ Manipulation", manipulation_status)
            else:
                st.metric("🛡️ Manipulation", "N/A")

    def display_microstructure_dashboard(self, symbol):
        """Display comprehensive microstructure analysis dashboard"""
        st.markdown("## 🔍 Advanced Microstructure Analysis")

        # Collapsible microstructure section
        with st.expander("📊 Tick-Level Microstructure Intelligence", expanded=True):
            if symbol not in self.tick_data or not self.tick_data[symbol]:
                st.warning("No tick data available for microstructure analysis")
                return

            # Select tick timeframe
            available_tick_tfs = list(self.tick_data[symbol].keys())
            selected_tick_tf = st.selectbox(
                "🕒 Select Tick Timeframe", 
                available_tick_tfs, 
                key="tick_tf_select"
            )

            if selected_tick_tf not in self.tick_data[symbol]:
                st.error(f"No data for {selected_tick_tf}")
                return

            tick_df = self.tick_data[symbol][selected_tick_tf]
            lookback_ticks = st.session_state.get('lookback_ticks', 1000)
            tick_df_display = tick_df.tail(lookback_ticks).copy()

            # Microstructure metrics overview
            st.markdown("### 🎯 Real-Time Microstructure Metrics")

            col1, col2, col3, col4, col5, col6 = st.columns(6)

            with col1:
                avg_spread = tick_df_display['spread_price'].mean()
                current_spread = tick_df_display['spread_price'].iloc[-1]
                spread_change = ((current_spread - avg_spread) / avg_spread) * 100
                st.metric("📏 Avg Spread", f"{avg_spread:.5f}", f"{spread_change:+.1f}%")

            with col2:
                tick_frequency = tick_df_display['tick_frequency'].mean()
                st.metric("⚡ Tick Frequency", f"{tick_frequency:.1f}/sec")

            with col3:
                spoofing_rate = (tick_df_display['spoofing_detected'].sum() / len(tick_df_display)) * 100
                st.metric("🚨 Spoofing Rate", f"{spoofing_rate:.2f}%")

            with col4:
                layering_rate = (tick_df_display['layering_detected'].sum() / len(tick_df_display)) * 100
                st.metric("🔄 Layering Rate", f"{layering_rate:.2f}%")

            with col5:
                momentum_ignition_rate = (tick_df_display['momentum_ignition'].sum() / len(tick_df_display)) * 100
                st.metric("🚀 Momentum Ignition", f"{momentum_ignition_rate:.2f}%")

            with col6:
                avg_flow_efficiency = tick_df_display['flow_efficiency'].mean()
                st.metric("💹 Flow Efficiency", f"{avg_flow_efficiency:.3f}")

            # Create microstructure charts
            self.create_microstructure_charts(tick_df_display, symbol, selected_tick_tf)

            # Manipulation detection analysis
            if st.session_state.get('show_manipulation_detection', True):
                self.display_manipulation_detection(tick_df_display)

            # Order flow analysis
            if st.session_state.get('show_order_flow', True):
                self.display_order_flow_analysis(tick_df_display)

        # Microstructure reports section
        if symbol in self.microstructure_data and self.microstructure_data[symbol]['txt_reports']:
            with st.expander("📄 Microstructure Analysis Reports", expanded=False):
                report_timestamps = list(self.microstructure_data[symbol]['txt_reports'].keys())
                selected_report = st.selectbox("Select Report", report_timestamps)

                if selected_report:
                    st.markdown("### 📋 Analysis Report")
                    report_content = self.microstructure_data[symbol]['txt_reports'][selected_report]
                    st.text_area("Report Content", report_content, height=300)

                    # JSON insights if available
                    if selected_report in self.microstructure_data[symbol]['json_reports']:
                        st.markdown("### 📊 JSON Insights")
                        json_data = self.microstructure_data[symbol]['json_reports'][selected_report]
                        st.json(json_data)

    def create_microstructure_charts(self, tick_df, symbol, timeframe):
        """Create comprehensive microstructure charts"""

        # Main microstructure chart
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=[
                f"{symbol} {timeframe} - Price & Spread Analysis",
                "Market Manipulation Detection", 
                "Order Flow & Momentum",
                "Liquidity & Efficiency Metrics"
            ],
            vertical_spacing=0.08,
            row_heights=[0.4, 0.2, 0.2, 0.2],
            shared_xaxes=True
        )

        # Price and spread
        fig.add_trace(go.Scatter(
            x=tick_df.index,
            y=tick_df['mid_price'],
            mode='lines',
            name='Mid Price',
            line=dict(color='blue', width=2)
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=tick_df.index,
            y=tick_df['bid'],
            mode='lines',
            name='Bid',
            line=dict(color='red', width=1),
            opacity=0.7
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=tick_df.index,
            y=tick_df['ask'], 
            mode='lines',
            name='Ask',
            line=dict(color='green', width=1),
            opacity=0.7
        ), row=1, col=1)

        # Manipulation detection
        manipulation_score = tick_df['spoofing_score'] + tick_df['layering_score']
        fig.add_trace(go.Scatter(
            x=tick_df.index,
            y=manipulation_score,
            mode='lines',
            name='Manipulation Score',
            line=dict(color='red', width=2)
        ), row=2, col=1)

        # Add manipulation alerts
        spoofing_alerts = tick_df[tick_df['spoofing_detected']]
        if not spoofing_alerts.empty:
            fig.add_trace(go.Scatter(
                x=spoofing_alerts.index,
                y=spoofing_alerts['spoofing_score'],
                mode='markers',
                name='Spoofing Alert',
                marker=dict(color='red', size=10, symbol='x')
            ), row=2, col=1)

        # Order flow and momentum
        fig.add_trace(go.Scatter(
            x=tick_df.index,
            y=tick_df['order_flow_imbalance'],
            mode='lines',
            name='Order Flow Imbalance',
            line=dict(color='purple', width=2)
        ), row=3, col=1)

        fig.add_trace(go.Scatter(
            x=tick_df.index,
            y=tick_df['momentum_score'],
            mode='lines',
            name='Momentum Score',
            line=dict(color='orange', width=2)
        ), row=3, col=1)

        # Liquidity and efficiency
        fig.add_trace(go.Scatter(
            x=tick_df.index,
            y=tick_df['flow_efficiency'],
            mode='lines',
            name='Flow Efficiency',
            line=dict(color='green', width=2)
        ), row=4, col=1)

        fig.add_trace(go.Scatter(
            x=tick_df.index,
            y=tick_df['spread_pct'] * 1000,  # Convert to basis points
            mode='lines',
            name='Spread (bps)',
            line=dict(color='brown', width=2)
        ), row=4, col=1)

        fig.update_layout(
            title=f"{symbol} {timeframe} Comprehensive Microstructure Analysis",
            template=st.session_state.get('chart_theme', 'plotly_dark'),
            height=1000,
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)

    def display_manipulation_detection(self, tick_df):
        """Display detailed manipulation detection analysis"""
        st.markdown("### 🚨 Market Manipulation Detection Analysis")

        # Manipulation statistics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<div class="manipulation-alert">', unsafe_allow_html=True)
            st.markdown("#### 🎭 Spoofing Analysis")
            spoofing_events = tick_df[tick_df['spoofing_detected']]
            st.markdown(f"**Events Detected**: {len(spoofing_events)}")
            if len(spoofing_events) > 0:
                avg_spoofing_score = spoofing_events['spoofing_score'].mean()
                max_spoofing_score = spoofing_events['spoofing_score'].max()
                st.markdown(f"**Avg Score**: {avg_spoofing_score:.3f}")
                st.markdown(f"**Max Score**: {max_spoofing_score:.3f}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="manipulation-alert">', unsafe_allow_html=True)
            st.markdown("#### 🔄 Layering Analysis")
            layering_events = tick_df[tick_df['layering_detected']]
            st.markdown(f"**Events Detected**: {len(layering_events)}")
            if len(layering_events) > 0:
                avg_layering_score = layering_events['layering_score'].mean()
                max_layering_score = layering_events['layering_score'].max()
                st.markdown(f"**Avg Score**: {avg_layering_score:.3f}")
                st.markdown(f"**Max Score**: {max_layering_score:.3f}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="manipulation-alert">', unsafe_allow_html=True)
            st.markdown("#### 🚀 Momentum Ignition")
            momentum_events = tick_df[tick_df['momentum_ignition']]
            st.markdown(f"**Events Detected**: {len(momentum_events)}")
            if len(momentum_events) > 0:
                avg_momentum_score = momentum_events['momentum_score'].mean()
                max_momentum_score = momentum_events['momentum_score'].max()
                st.markdown(f"**Avg Score**: {avg_momentum_score:.3f}")
                st.markdown(f"**Max Score**: {max_momentum_score:.3f}")
            st.markdown('</div>', unsafe_allow_html=True)

    def display_order_flow_analysis(self, tick_df):
        """Display detailed order flow analysis"""
        st.markdown("### 💹 Advanced Order Flow Analysis")

        col1, col2 = st.columns(2)

        with col1:
            # Order flow imbalance chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=tick_df.index,
                y=tick_df['order_flow_imbalance'],
                mode='lines',
                name='Order Flow Imbalance',
                line=dict(color='purple', width=2)
            ))

            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.update_layout(
                title="Order Flow Imbalance",
                template=st.session_state.get('chart_theme', 'plotly_dark'),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Volume flow analysis
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=tick_df.index,
                y=tick_df['volume_flow_imbalance'],
                mode='lines',
                name='Volume Flow Imbalance', 
                line=dict(color='orange', width=2)
            ))

            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.update_layout(
                title="Volume Flow Imbalance",
                template=st.session_state.get('chart_theme', 'plotly_dark'),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

    def display_multi_timeframe_analysis(self, symbol):
        """Display multi-timeframe analysis"""
        st.markdown("## ⏱️ Multi-Timeframe Analysis")

        if symbol not in self.bar_data or not self.bar_data[symbol]:
            st.warning("No bar data available for multi-timeframe analysis")
            return

        # Timeframe selection
        available_timeframes = list(self.bar_data[symbol].keys())
        selected_timeframes = st.multiselect(
            "Select Timeframes for Comparison",
            available_timeframes,
            default=available_timeframes[:3] if len(available_timeframes) >= 3 else available_timeframes
        )

        if not selected_timeframes:
            st.warning("Please select at least one timeframe")
            return

        # Multi-timeframe price chart
        fig = go.Figure()

        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

        for i, tf in enumerate(selected_timeframes):
            if tf in self.bar_data[symbol]:
                df = self.bar_data[symbol][tf]
                lookback = st.session_state.get('lookback_bars', 500)
                df_display = df.tail(lookback)

                fig.add_trace(go.Scatter(
                    x=df_display.index,
                    y=df_display['close'],
                    mode='lines',
                    name=f'{tf} Close',
                    line=dict(color=colors[i % len(colors)], width=2)
                ))

        fig.update_layout(
            title=f"{symbol} Multi-Timeframe Price Comparison",
            template=st.session_state.get('chart_theme', 'plotly_dark'),
            height=600,
            yaxis_title="Price"
        )

        st.plotly_chart(fig, use_container_width=True)

        # Multi-timeframe metrics comparison
        self.create_multi_timeframe_metrics(symbol, selected_timeframes)

    def create_multi_timeframe_metrics(self, symbol, timeframes):
        """Create multi-timeframe metrics comparison"""
        st.markdown("### 📊 Cross-Timeframe Metrics")

        metrics_data = []

        for tf in timeframes:
            if tf in self.bar_data[symbol]:
                df = self.bar_data[symbol][tf]
                if len(df) > 0:
                    latest = df.iloc[-1]

                    # Calculate metrics
                    rsi = latest.get('RSI_14', np.nan)
                    atr = latest.get('ATR_14', np.nan)
                    volume = latest.get('volume', np.nan)

                    # SMC signals
                    bullish_fvg = latest.get('SMC_fvg_bullish', False)
                    bearish_fvg = latest.get('SMC_fvg_bearish', False)
                    bullish_ob = latest.get('SMC_bullish_ob', False)
                    bearish_ob = latest.get('SMC_bearish_ob', False)

                    # Wyckoff signals
                    wyckoff_acc = latest.get('wyckoff_accumulation', False)
                    wyckoff_dist = latest.get('wyckoff_distribution', False)

                    metrics_data.append({
                        'Timeframe': tf,
                        'RSI': f"{rsi:.1f}" if not np.isnan(rsi) else "N/A",
                        'ATR': f"{atr:.4f}" if not np.isnan(atr) else "N/A",
                        'Volume': f"{volume:.0f}" if not np.isnan(volume) else "N/A",
                        'SMC Signals': f"{'🟢FVG ' if bullish_fvg else ''}{'🔴FVG ' if bearish_fvg else ''}{'🟢OB ' if bullish_ob else ''}{'🔴OB ' if bearish_ob else ''}",
                        'Wyckoff': f"{'🟢ACC' if wyckoff_acc else ''}{'🔴DIST' if wyckoff_dist else ''}"
                    })

        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True)

    def display_advanced_smc_analysis(self, symbol):
        """Display advanced Smart Money Concepts analysis"""
        st.markdown("## 🧠 Advanced Smart Money Concepts Analysis")

        if symbol not in self.bar_data or not self.bar_data[symbol]:
            st.warning("No bar data available for SMC analysis")
            return

        # Select timeframe for SMC analysis
        available_timeframes = list(self.bar_data[symbol].keys())
        selected_tf = st.selectbox(
            "🕒 Select Timeframe for SMC Analysis",
            available_timeframes,
            key="smc_tf_select"
        )

        if selected_tf not in self.bar_data[symbol]:
            st.error(f"No data for {selected_tf}")
            return

        df = self.bar_data[symbol][selected_tf]
        lookback = st.session_state.get('lookback_bars', 500)
        df_display = df.tail(lookback).copy()

        # SMC signals overview
        self.create_smc_signals_overview(df_display)

        # SMC price chart
        self.create_smc_price_chart(df_display, symbol, selected_tf)

        # Liquidity analysis
        if st.session_state.get('show_liquidity_analysis', True):
            self.create_liquidity_analysis(df_display)

    def create_smc_signals_overview(self, df):
        """Create SMC signals overview"""
        st.markdown("### 🎯 SMC Signals Overview")

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            bullish_fvgs = df['SMC_fvg_bullish'].sum() if 'SMC_fvg_bullish' in df.columns else 0
            bearish_fvgs = df['SMC_fvg_bearish'].sum() if 'SMC_fvg_bearish' in df.columns else 0
            st.markdown(f'<div class="smc-signal"><strong>Fair Value Gaps</strong><br>🟢 {bullish_fvgs} | 🔴 {bearish_fvgs}</div>', unsafe_allow_html=True)

        with col2:
            bullish_obs = df['SMC_bullish_ob'].sum() if 'SMC_bullish_ob' in df.columns else 0
            bearish_obs = df['SMC_bearish_ob'].sum() if 'SMC_bearish_ob' in df.columns else 0
            st.markdown(f'<div class="smc-signal"><strong>Order Blocks</strong><br>🟢 {bullish_obs} | 🔴 {bearish_obs}</div>', unsafe_allow_html=True)

        with col3:
            liquidity_grabs = df['SMC_liquidity_grab'].sum() if 'SMC_liquidity_grab' in df.columns else 0
            st.markdown(f'<div class="smc-signal"><strong>Liquidity Grabs</strong><br>⚡ {liquidity_grabs}</div>', unsafe_allow_html=True)

        with col4:
            bos_bullish = df['SMC_bos_bullish'].sum() if 'SMC_bos_bullish' in df.columns else 0
            bos_bearish = df['SMC_bos_bearish'].sum() if 'SMC_bos_bearish' in df.columns else 0
            st.markdown(f'<div class="smc-signal"><strong>Break of Structure</strong><br>🟢 {bos_bullish} | 🔴 {bos_bearish}</div>', unsafe_allow_html=True)

        with col5:
            # Current market bias
            latest = df.iloc[-1]
            premium_zone = latest.get('SMC_premium_zone', False)
            discount_zone = latest.get('SMC_discount_zone', False)
            equilibrium = latest.get('SMC_equilibrium', False)

            if premium_zone:
                bias = "🔴 PREMIUM"
            elif discount_zone:
                bias = "🟢 DISCOUNT"
            elif equilibrium:
                bias = "🟡 EQUILIBRIUM"
            else:
                bias = "⚪ NEUTRAL"

            st.markdown(f'<div class="smc-signal"><strong>Market Zone</strong><br>{bias}</div>', unsafe_allow_html=True)

    def create_smc_price_chart(self, df, symbol, timeframe):
        """Create SMC price chart with overlays"""
        fig = go.Figure()

        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="Price",
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4444'
        ))

        # SMC levels
        if 'SMC_range_high' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['SMC_range_high'],
                mode='lines',
                name='Range High',
                line=dict(color='red', width=2, dash='dash')
            ))

        if 'SMC_range_low' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['SMC_range_low'],
                mode='lines',
                name='Range Low',
                line=dict(color='green', width=2, dash='dash')
            ))

        if 'SMC_range_mid' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['SMC_range_mid'],
                mode='lines',
                name='Equilibrium',
                line=dict(color='yellow', width=1, dash='dot')
            ))

        # SMC signals
        if 'SMC_fvg_bullish' in df.columns:
            bullish_fvgs = df[df['SMC_fvg_bullish']]
            if not bullish_fvgs.empty:
                fig.add_trace(go.Scatter(
                    x=bullish_fvgs.index,
                    y=bullish_fvgs['low'],
                    mode='markers',
                    name='Bullish FVG',
                    marker=dict(symbol='triangle-up', color='lime', size=12)
                ))

        if 'SMC_fvg_bearish' in df.columns:
            bearish_fvgs = df[df['SMC_fvg_bearish']]
            if not bearish_fvgs.empty:
                fig.add_trace(go.Scatter(
                    x=bearish_fvgs.index,
                    y=bearish_fvgs['high'],
                    mode='markers',
                    name='Bearish FVG',
                    marker=dict(symbol='triangle-down', color='red', size=12)
                ))

        fig.update_layout(
            title=f"{symbol} {timeframe} - Smart Money Concepts Analysis",
            template=st.session_state.get('chart_theme', 'plotly_dark'),
            height=700,
            xaxis_rangeslider_visible=False
        )

        st.plotly_chart(fig, use_container_width=True)

    def create_liquidity_analysis(self, df):
        """Create liquidity analysis"""
        st.markdown("### 💧 Liquidity Analysis")

        col1, col2 = st.columns(2)

        with col1:
            # Liquidity grab events
            if 'SMC_liquidity_grab' in df.columns:
                liquidity_events = df[df['SMC_liquidity_grab']]

                st.markdown("#### ⚡ Recent Liquidity Grabs")
                if not liquidity_events.empty:
                    for idx, row in liquidity_events.tail(5).iterrows():
                        strength = row.get('SMC_liquidity_strength', 0)
                        st.markdown(f"""
                        **{idx.strftime('%Y-%m-%d %H:%M')}**  
                        Price: {row['close']:.4f} | Strength: {strength:.2f}
                        """)
                else:
                    st.info("No recent liquidity grabs detected")

        with col2:
            # Liquidity zones
            st.markdown("#### 🎯 Current Liquidity Zones")
            latest = df.iloc[-1]

            range_high = latest.get('SMC_range_high', 0)
            range_low = latest.get('SMC_range_low', 0)
            current_price = latest['close']

            if range_high > 0 and range_low > 0:
                range_position = ((current_price - range_low) / (range_high - range_low)) * 100

                st.markdown(f"""
                **Range High**: {range_high:.4f}  
                **Range Low**: {range_low:.4f}  
                **Current Position**: {range_position:.1f}%  
                **Distance to High**: {((range_high - current_price) / current_price) * 100:.2f}%  
                **Distance to Low**: {((current_price - range_low) / current_price) * 100:.2f}%
                """)

    def display_wyckoff_analysis(self, symbol):
        """Display comprehensive Wyckoff analysis"""
        st.markdown("## 📈 Comprehensive Wyckoff Analysis")

        if symbol not in self.bar_data or not self.bar_data[symbol]:
            st.warning("No bar data available for Wyckoff analysis")
            return

        # Select timeframe for Wyckoff analysis
        available_timeframes = list(self.bar_data[symbol].keys())
        selected_tf = st.selectbox(
            "🕒 Select Timeframe for Wyckoff Analysis",
            available_timeframes,
            key="wyckoff_tf_select"
        )

        if selected_tf not in self.bar_data[symbol]:
            st.error(f"No data for {selected_tf}")
            return

        df = self.bar_data[symbol][selected_tf]
        lookback = st.session_state.get('lookback_bars', 500)
        df_display = df.tail(lookback).copy()

        # Wyckoff phase analysis
        self.create_wyckoff_phase_analysis(df_display)

        # Wyckoff signals chart
        self.create_wyckoff_signals_chart(df_display, symbol, selected_tf)

        # Volume spread analysis
        self.create_volume_spread_analysis(df_display)

    def create_wyckoff_phase_analysis(self, df):
        """Create Wyckoff phase analysis"""
        st.markdown("### 📊 Wyckoff Phase Analysis")

        col1, col2, col3, col4 = st.columns(4)

        # Phase distribution
        accumulation_count = df['wyckoff_accumulation'].sum() if 'wyckoff_accumulation' in df.columns else 0
        distribution_count = df['wyckoff_distribution'].sum() if 'wyckoff_distribution' in df.columns else 0

        with col1:
            acc_pct = (accumulation_count / len(df)) * 100 if len(df) > 0 else 0
            st.markdown(f'<div class="wyckoff-phase accumulation">ACCUMULATION<br>{accumulation_count} bars ({acc_pct:.1f}%)</div>', unsafe_allow_html=True)

        with col2:
            dist_pct = (distribution_count / len(df)) * 100 if len(df) > 0 else 0
            st.markdown(f'<div class="wyckoff-phase distribution">DISTRIBUTION<br>{distribution_count} bars ({dist_pct:.1f}%)</div>', unsafe_allow_html=True)

        with col3:
            # Current phase determination
            latest = df.iloc[-1]
            current_acc = latest.get('wyckoff_accumulation', False)
            current_dist = latest.get('wyckoff_distribution', False)

            if current_acc:
                current_phase = "🟢 ACCUMULATION"
                phase_class = "accumulation"
            elif current_dist:
                current_phase = "🔴 DISTRIBUTION"
                phase_class = "distribution"
            else:
                current_phase = "🟡 NEUTRAL"
                phase_class = "markup"

            st.markdown(f'<div class="wyckoff-phase {phase_class}">CURRENT PHASE<br>{current_phase}</div>', unsafe_allow_html=True)

        with col4:
            # Wyckoff special events
            spring_count = df['wyckoff_spring'].sum() if 'wyckoff_spring' in df.columns else 0
            upthrust_count = df['wyckoff_upthrust'].sum() if 'wyckoff_upthrust' in df.columns else 0

            st.markdown(f'<div class="wyckoff-phase markup">SPECIAL EVENTS<br>Springs: {spring_count}<br>Upthrusts: {upthrust_count}</div>', unsafe_allow_html=True)

    def create_wyckoff_signals_chart(self, df, symbol, timeframe):
        """Create Wyckoff signals chart"""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=[
                f"{symbol} {timeframe} - Wyckoff Analysis",
                "Volume Spread Analysis"
            ],
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3],
            shared_xaxes=True
        )

        # Price chart with Wyckoff signals
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="Price"
        ), row=1, col=1)

        # Wyckoff accumulation zones
        if 'wyckoff_accumulation' in df.columns:
            acc_zones = df[df['wyckoff_accumulation']]
            if not acc_zones.empty:
                fig.add_trace(go.Scatter(
                    x=acc_zones.index,
                    y=acc_zones['close'],
                    mode='markers',
                    name='Accumulation',
                    marker=dict(symbol='circle', color='green', size=8)
                ), row=1, col=1)

        # Wyckoff distribution zones
        if 'wyckoff_distribution' in df.columns:
            dist_zones = df[df['wyckoff_distribution']]
            if not dist_zones.empty:
                fig.add_trace(go.Scatter(
                    x=dist_zones.index,
                    y=dist_zones['close'],
                    mode='markers',
                    name='Distribution',
                    marker=dict(symbol='circle', color='red', size=8)
                ), row=1, col=1)

        # Springs and upthrusts
        if 'wyckoff_spring' in df.columns:
            springs = df[df['wyckoff_spring']]
            if not springs.empty:
                fig.add_trace(go.Scatter(
                    x=springs.index,
                    y=springs['low'],
                    mode='markers',
                    name='Spring',
                    marker=dict(symbol='triangle-up', color='lime', size=15)
                ), row=1, col=1)

        if 'wyckoff_upthrust' in df.columns:
            upthrusts = df[df['wyckoff_upthrust']]
            if not upthrusts.empty:
                fig.add_trace(go.Scatter(
                    x=upthrusts.index,
                    y=upthrusts['high'],
                    mode='markers',
                    name='Upthrust',
                    marker=dict(symbol='triangle-down', color='red', size=15)
                ), row=1, col=1)

        # Volume spread ratio
        if 'wyckoff_vs_ratio' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['wyckoff_vs_ratio'],
                mode='lines',
                name='VS Ratio',
                line=dict(color='purple', width=2)
            ), row=2, col=1)

        if 'wyckoff_vs_ratio_ma' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['wyckoff_vs_ratio_ma'],
                mode='lines',
                name='VS Ratio MA',
                line=dict(color='orange', width=2)
            ), row=2, col=1)

        fig.update_layout(
            title=f"{symbol} {timeframe} Wyckoff Analysis",
            template=st.session_state.get('chart_theme', 'plotly_dark'),
            height=800,
            xaxis_rangeslider_visible=False
        )

        st.plotly_chart(fig, use_container_width=True)

    def create_volume_spread_analysis(self, df):
        """Create volume spread analysis"""
        st.markdown("### 📊 Volume Spread Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 💪 Effort vs Result Analysis")

            # Calculate effort vs result
            if 'wyckoff_effort' in df.columns and 'wyckoff_result' in df.columns:
                effort_result_df = df[['wyckoff_effort', 'wyckoff_result']].dropna()

                if not effort_result_df.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=effort_result_df.index,
                        y=effort_result_df['wyckoff_effort'],
                        mode='lines',
                        name='Effort (Volume)',
                        line=dict(color='blue', width=2)
                    ))

                    fig.add_trace(go.Scatter(
                        x=effort_result_df.index,
                        y=effort_result_df['wyckoff_result'],
                        mode='lines',
                        name='Result (Spread)',
                        line=dict(color='red', width=2)
                    ))

                    fig.update_layout(
                        title="Effort vs Result",
                        template=st.session_state.get('chart_theme', 'plotly_dark'),
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No effort vs result data available")
            else:
                st.info("Effort vs Result data not available")

        with col2:
            st.markdown("#### 🚫 No Demand/Supply Events")

            no_demand_count = df['wyckoff_no_demand'].sum() if 'wyckoff_no_demand' in df.columns else 0
            no_supply_count = df['wyckoff_no_supply'].sum() if 'wyckoff_no_supply' in df.columns else 0

            st.markdown(f"""
            **No Demand Events**: {no_demand_count}  
            **No Supply Events**: {no_supply_count}
            """)

            # Recent events
            if 'wyckoff_no_demand' in df.columns:
                recent_no_demand = df[df['wyckoff_no_demand']].tail(3)
                if not recent_no_demand.empty:
                    st.markdown("**Recent No Demand:**")
                    for idx, row in recent_no_demand.iterrows():
                        st.markdown(f"- {idx.strftime('%Y-%m-%d %H:%M')}: {row['close']:.4f}")

            if 'wyckoff_no_supply' in df.columns:
                recent_no_supply = df[df['wyckoff_no_supply']].tail(3)
                if not recent_no_supply.empty:
                    st.markdown("**Recent No Supply:**")
                    for idx, row in recent_no_supply.iterrows():
                        st.markdown(f"- {idx.strftime('%Y-%m-%d %H:%M')}: {row['close']:.4f}")

def main():
    """Main application entry point"""
    dashboard = UltimateMicrostructureZANFLOW()
    dashboard.create_main_dashboard()

if __name__ == "__main__":
    main()
