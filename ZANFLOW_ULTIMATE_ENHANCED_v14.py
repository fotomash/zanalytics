#!/usr/bin/env python3
"""
ZANFLOW v14 ULTIMATE Enhanced Trading Dashboard
- Symbol Scanner & Dropdown Selection
- Timeframe Selection
- Candlestick Chart as Main Chart
- Fixed Market Overview with Real Values
- Conclusive Analysis from TXT Files
- Multi-Timeframe Analysis
- No Data Loading Messages
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

class ZANFLOWUltimateEnhancedDashboard:
    def __init__(self, data_directory="./data"):
        """Initialize enhanced dashboard with symbol scanner"""
        self.data_dir = Path(data_directory)
        self.symbols_data = {}
        self.available_symbols = []
        self.available_timeframes = {}
        self.txt_reports = {}
        self.json_reports = {}

    def scan_available_symbols(self):
        """Scan data directory for available symbols and timeframes"""
        if not self.data_dir.exists():
            return

        # Look for symbol directories or CSV files
        symbol_pattern = {}

        # Method 1: Check for symbol directories
        for item in self.data_dir.iterdir():
            if item.is_dir():
                symbol = item.name
                timeframes = []
                for csv_file in item.glob("*.csv"):
                    # Extract timeframe from filename
                    if "_processed.csv" in csv_file.name:
                        parts = csv_file.stem.split("_")
                        if len(parts) >= 3:
                            timeframe = parts[2]  # Usually the timeframe part
                            timeframes.append(timeframe)
                if timeframes:
                    symbol_pattern[symbol] = list(set(timeframes))

        # Method 2: Check for CSV files directly in data directory
        csv_files = list(self.data_dir.glob("*_processed.csv"))
        for csv_file in csv_files:
            parts = csv_file.stem.split("_")
            if len(parts) >= 3:
                symbol = parts[0]  # First part is usually symbol
                timeframe = parts[2]  # Third part is usually timeframe

                if symbol not in symbol_pattern:
                    symbol_pattern[symbol] = []
                if timeframe not in symbol_pattern[symbol]:
                    symbol_pattern[symbol].append(timeframe)

        self.available_symbols = list(symbol_pattern.keys())
        self.available_timeframes = symbol_pattern

        return len(self.available_symbols) > 0

    def load_symbol_data(self, symbol, timeframe):
        """Load data for specific symbol and timeframe"""
        try:
            # Try different file path patterns
            possible_paths = [
                self.data_dir / symbol / f"{symbol}_M1_bars_{timeframe}_csv_processed.csv",
                self.data_dir / symbol / f"{symbol}_TICK_{timeframe}_csv_processed.csv",
                self.data_dir / f"{symbol}_M1_bars_{timeframe}_csv_processed.csv",
                self.data_dir / f"{symbol}_TICK_{timeframe}_csv_processed.csv",
                self.data_dir / f"{symbol}_{timeframe}_processed.csv"
            ]

            for path in possible_paths:
                if path.exists():
                    df = pd.read_csv(path)
                    # Handle timestamp column
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df.set_index('timestamp', inplace=True)
                    elif df.index.name != 'timestamp':
                        df.index = pd.to_datetime(df.index)

                    return df

            return None

        except Exception as e:
            st.error(f"Error loading data for {symbol} {timeframe}: {e}")
            return None

    def load_txt_report(self, symbol):
        """Load latest TXT report for symbol"""
        try:
            # Look for TXT files
            txt_patterns = [
                self.data_dir / symbol / f"*{symbol}*Analysis*Report*.txt",
                self.data_dir / f"*{symbol}*Analysis*Report*.txt",
                self.data_dir / symbol / "*.txt",
                self.data_dir / "*.txt"
            ]

            for pattern in txt_patterns:
                txt_files = glob.glob(str(pattern))
                if txt_files:
                    # Get the most recent file
                    latest_file = max(txt_files, key=os.path.getmtime)
                    with open(latest_file, 'r') as f:
                        return f.read()

            return None

        except Exception as e:
            return None

    def calculate_market_metrics(self, df):
        """Calculate real market metrics to fix Market Overview"""
        try:
            if df is None or len(df) == 0:
                return {
                    'spread': 'N/A',
                    'atr': 'N/A', 
                    'rsi': 'N/A',
                    'manipulation': 'N/A'
                }

            metrics = {}

            # Calculate spread
            if 'spread' in df.columns:
                metrics['spread'] = f"{df['spread'].iloc[-1]:.2f}"
            elif 'bid' in df.columns and 'ask' in df.columns:
                spread = df['ask'].iloc[-1] - df['bid'].iloc[-1]
                metrics['spread'] = f"{spread:.5f}"
            else:
                # Estimate spread from high-low
                recent_hl_spread = (df['high'] - df['low']).tail(20).mean()
                metrics['spread'] = f"{recent_hl_spread:.4f}"

            # Calculate ATR
            if 'atr' in df.columns:
                metrics['atr'] = f"{df['atr'].iloc[-1]:.4f}"
            elif 'atr_14' in df.columns:
                metrics['atr'] = f"{df['atr_14'].iloc[-1]:.4f}"
            else:
                # Calculate simple ATR
                high_low = df['high'] - df['low']
                high_close = abs(df['high'] - df['close'].shift(1))
                low_close = abs(df['low'] - df['close'].shift(1))
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                atr = true_range.rolling(14).mean().iloc[-1]
                metrics['atr'] = f"{atr:.4f}"

            # Calculate RSI
            if 'rsi' in df.columns:
                rsi_val = df['rsi'].iloc[-1]
                metrics['rsi'] = f"{'🟢' if rsi_val < 30 else '🔴' if rsi_val > 70 else '🟡'} {rsi_val:.1f}"
            elif 'rsi_14' in df.columns:
                rsi_val = df['rsi_14'].iloc[-1]
                metrics['rsi'] = f"{'🟢' if rsi_val < 30 else '🔴' if rsi_val > 70 else '🟡'} {rsi_val:.1f}"
            elif 'RSI_14' in df.columns:
                rsi_val = df['RSI_14'].iloc[-1]
                metrics['rsi'] = f"{'🟢' if rsi_val < 30 else '🔴' if rsi_val > 70 else '🟡'} {rsi_val:.1f}"
            else:
                # Calculate simple RSI
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                rsi_val = 100 - (100 / (1 + rs)).iloc[-1]
                metrics['rsi'] = f"{'🟢' if rsi_val < 30 else '🔴' if rsi_val > 70 else '🟡'} {rsi_val:.1f}"

            # Check for manipulation indicators
            manipulation_score = 0
            if 'spoofing_detected' in df.columns:
                manipulation_score += df['spoofing_detected'].tail(50).sum()
            if 'layering_detected' in df.columns:
                manipulation_score += df['layering_detected'].tail(50).sum()
            if 'momentum_ignition' in df.columns:
                manipulation_score += df['momentum_ignition'].tail(50).sum()

            if manipulation_score > 10:
                metrics['manipulation'] = "🔴 HIGH"
            elif manipulation_score > 5:
                metrics['manipulation'] = "🟡 MEDIUM"
            elif manipulation_score > 0:
                metrics['manipulation'] = "🟢 LOW"
            else:
                # Check for high volatility as proxy
                if len(df) > 20:
                    vol = df['close'].pct_change().tail(20).std()
                    vol_threshold = df['close'].pct_change().std()
                    if vol > vol_threshold * 2:
                        metrics['manipulation'] = "🟡 POSSIBLE"
                    else:
                        metrics['manipulation'] = "🟢 CLEAN"
                else:
                    metrics['manipulation'] = "🟡 UNKNOWN"

            return metrics

        except Exception as e:
            return {
                'spread': 'ERROR',
                'atr': 'ERROR', 
                'rsi': 'ERROR',
                'manipulation': 'ERROR'
            }

    def create_enhanced_dashboard(self):
        """Create the enhanced dashboard interface"""
        st.set_page_config(
            page_title="ZANFLOW v14 Ultimate Enhanced", 
            page_icon="🚀", 
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Enhanced CSS
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .metric-card {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 1rem;
            border-radius: 10px;
            border-left: 4px solid #667eea;
            margin: 0.5rem 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .market-overview {
            background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        .analysis-section {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            border-left: 3px solid #667eea;
        }
        .success-metric { color: #28a745; font-weight: bold; }
        .warning-metric { color: #ffc107; font-weight: bold; }
        .danger-metric { color: #dc3545; font-weight: bold; }
        </style>
        """, unsafe_allow_html=True)

        # Main header
        st.markdown("""
        <div class="main-header">
            <h1>🚀 ZANFLOW v14 Ultimate Enhanced Dashboard</h1>
            <p>Advanced Symbol Scanner • Multi-Timeframe Analysis • Real-Time Market Intelligence • Conclusive Analysis</p>
        </div>
        """, unsafe_allow_html=True)

        # Initialize and scan symbols (silently)
        if not hasattr(st.session_state, 'symbols_scanned'):
            with st.spinner("🔍 Scanning available symbols..."):
                if self.scan_available_symbols():
                    st.session_state.symbols_scanned = True
                else:
                    st.error("❌ No trading data found. Please ensure data files are in the ./data directory")
                    st.info("Expected structure: ./data/SYMBOL/SYMBOL_TIMEFRAME_processed.csv")
                    return

        # Sidebar controls
        self.create_enhanced_sidebar()

        # Main content
        if st.session_state.get('selected_symbol') and st.session_state.get('selected_timeframe'):
            self.display_enhanced_analysis()
        else:
            self.display_symbol_overview()

    def create_enhanced_sidebar(self):
        """Create enhanced sidebar with symbol and timeframe selection"""
        st.sidebar.markdown("# 🎛️ Trading Control Center")

        # Symbol selection
        if self.available_symbols:
            selected_symbol = st.sidebar.selectbox(
                "📈 Select Trading Symbol",
                [""] + sorted(self.available_symbols),
                key="selected_symbol",
                help="Choose from available symbols in your data directory"
            )

            if selected_symbol:
                # Timeframe selection
                available_tfs = self.available_timeframes.get(selected_symbol, [])
                if available_tfs:
                    selected_timeframe = st.sidebar.selectbox(
                        "⏱️ Select Timeframe",
                        [""] + sorted(available_tfs),
                        key="selected_timeframe",
                        help="Available timeframes for selected symbol"
                    )

                    if selected_timeframe:
                        # Load and display current data metrics
                        df = self.load_symbol_data(selected_symbol, selected_timeframe)
                        if df is not None:
                            # Calculate real metrics
                            metrics = self.calculate_market_metrics(df)

                            st.sidebar.markdown("---")
                            st.sidebar.markdown("### 🌍 Market Overview & Real-Time Status")

                            # Create market overview in sidebar
                            col1, col2 = st.sidebar.columns(2)
                            with col1:
                                st.markdown(f"""
                                📏 **Spread**  
                                {metrics['spread']}

                                📊 **ATR (14)**  
                                {metrics['atr']}
                                """)
                            with col2:
                                st.markdown(f"""
                                📈 **RSI**  
                                {metrics['rsi']}

                                🛡️ **Manipulation**  
                                {metrics['manipulation']}
                                """)

                            # Data info
                            current_price = df['close'].iloc[-1]
                            price_change = ((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100 if len(df) > 1 else 0

                            st.sidebar.markdown("---")
                            st.sidebar.markdown("### 📊 Current Market Data")

                            col1, col2 = st.sidebar.columns(2)
                            with col1:
                                st.metric("Price", f"{current_price:.4f}")
                            with col2:
                                st.metric("Change", f"{price_change:+.2f}%")

                            st.sidebar.info(f"""
                            🔢 **Bars**: {len(df):,}  
                            📅 **From**: {df.index.min().strftime('%Y-%m-%d %H:%M')}  
                            📅 **To**: {df.index.max().strftime('%Y-%m-%d %H:%M')}  
                            💹 **Range**: {df['low'].min():.4f} - {df['high'].max():.4f}
                            """)

        # Analysis options
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🔬 Analysis Configuration")

        st.session_state['show_candlestick'] = st.sidebar.checkbox("🕯️ Candlestick Chart", True)
        st.session_state['show_microstructure'] = st.sidebar.checkbox("🔍 Microstructure Analysis", True)
        st.session_state['show_txt_analysis'] = st.sidebar.checkbox("📄 TXT Report Analysis", True)
        st.session_state['show_multitimeframe'] = st.sidebar.checkbox("⏱️ Multi-Timeframe Analysis", True)
        st.session_state['show_smart_money'] = st.sidebar.checkbox("🧠 Smart Money Concepts", True)
        st.session_state['show_wyckoff'] = st.sidebar.checkbox("📈 Wyckoff Analysis", True)

        # Chart settings
        st.sidebar.markdown("### 📈 Chart Configuration")
        st.session_state['lookback_bars'] = st.sidebar.slider("Lookback Period", 50, 2000, 500)
        st.session_state['chart_theme'] = st.sidebar.selectbox("Chart Theme", ["plotly_dark", "plotly_white", "plotly"])

    def display_symbol_overview(self):
        """Display overview of available symbols"""
        st.markdown("## 🌍 Available Trading Symbols")

        if not self.available_symbols:
            st.warning("No symbols found. Please check your data directory structure.")
            return

        # Symbol statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Symbols", len(self.available_symbols))

        with col2:
            total_timeframes = sum(len(tfs) for tfs in self.available_timeframes.values())
            st.metric("Total Timeframes", total_timeframes)

        with col3:
            avg_timeframes = total_timeframes / len(self.available_symbols) if self.available_symbols else 0
            st.metric("Avg TFs per Symbol", f"{avg_timeframes:.1f}")

        with col4:
            # Count total data files
            total_files = len(list(self.data_dir.rglob("*_processed.csv")))
            st.metric("Data Files", total_files)

        # Symbol breakdown
        st.markdown("### 📊 Symbol Breakdown")

        symbol_data = []
        for symbol in sorted(self.available_symbols):
            timeframes = self.available_timeframes.get(symbol, [])
            symbol_data.append({
                'Symbol': symbol,
                'Available Timeframes': ', '.join(sorted(timeframes)),
                'TF Count': len(timeframes)
            })

        if symbol_data:
            symbols_df = pd.DataFrame(symbol_data)
            st.dataframe(symbols_df, use_container_width=True)

        # Quick access buttons
        st.markdown("### 🚀 Quick Access")
        cols = st.columns(min(len(self.available_symbols), 4))

        for i, symbol in enumerate(sorted(self.available_symbols)[:4]):
            with cols[i]:
                if st.button(f"📈 {symbol}", key=f"quick_{symbol}"):
                    st.session_state['selected_symbol'] = symbol
                    if self.available_timeframes[symbol]:
                        st.session_state['selected_timeframe'] = sorted(self.available_timeframes[symbol])[0]
                    st.rerun()

    def display_enhanced_analysis(self):
        """Display enhanced analysis for selected symbol and timeframe"""
        symbol = st.session_state['selected_symbol']
        timeframe = st.session_state['selected_timeframe']

        # Load data
        df = self.load_symbol_data(symbol, timeframe)
        if df is None:
            st.error(f"Failed to load data for {symbol} {timeframe}")
            return

        # Apply lookback
        lookback = st.session_state.get('lookback_bars', 500)
        df_display = df.tail(lookback).copy()

        st.markdown(f"# 🚀 {symbol} - {timeframe} Enhanced Analysis")

        # Enhanced Market Overview (Fixed)
        self.display_enhanced_market_overview(df_display, symbol)

        # Main candlestick chart
        if st.session_state.get('show_candlestick', True):
            self.create_enhanced_candlestick_chart(df_display, symbol, timeframe)

        # Multi-timeframe analysis
        if st.session_state.get('show_multitimeframe', True):
            self.create_multitimeframe_analysis(symbol)

        # TXT Report Analysis
        if st.session_state.get('show_txt_analysis', True):
            self.display_txt_report_analysis(symbol)

        # Other analysis sections
        col1, col2 = st.columns(2)

        with col1:
            if st.session_state.get('show_microstructure', True):
                self.create_microstructure_section(df_display)

        with col2:
            if st.session_state.get('show_smart_money', True):
                self.create_smart_money_section(df_display)

        if st.session_state.get('show_wyckoff', True):
            self.create_wyckoff_section(df_display)

    def display_enhanced_market_overview(self, df, symbol):
        """Display enhanced market overview with real values"""
        st.markdown('<div class="market-overview">', unsafe_allow_html=True)
        st.markdown("## 🌍 Market Overview & Real-Time Status")

        # Calculate real metrics
        metrics = self.calculate_market_metrics(df)

        # Main metrics row
        col1, col2, col3, col4, col5, col6 = st.columns(6)

        current_price = df['close'].iloc[-1]
        prev_price = df['close'].iloc[-2] if len(df) > 1 else current_price
        price_change = current_price - prev_price
        price_change_pct = (price_change / prev_price * 100) if prev_price != 0 else 0

        with col1:
            st.metric("💰 Price", f"{current_price:.4f}", f"{price_change:+.4f}")

        with col2:
            st.metric("📊 Change %", f"{price_change_pct:+.2f}%")

        with col3:
            st.metric("📏 Spread", metrics['spread'])

        with col4:
            st.metric("📊 ATR", metrics['atr'])

        with col5:
            st.metric("📈 RSI", metrics['rsi'])

        with col6:
            st.metric("🛡️ Manipulation", metrics['manipulation'])

        # Market regime analysis
        col1, col2, col3 = st.columns(3)

        with col1:
            # Trend analysis
            if len(df) >= 50:
                short_ma = df['close'].rolling(8).mean().iloc[-1]
                long_ma = df['close'].rolling(21).mean().iloc[-1]
                trend = "🟢 BULLISH" if short_ma > long_ma else "🔴 BEARISH"
            else:
                trend = "🟡 INSUFFICIENT DATA"
            st.metric("📈 Trend (8/21)", trend)

        with col2:
            # Volatility regime
            if len(df) >= 20:
                recent_vol = df['close'].pct_change().tail(20).std()
                historical_vol = df['close'].pct_change().std()
                vol_regime = "🔥 HIGH" if recent_vol > historical_vol * 1.5 else "❄️ LOW"
            else:
                vol_regime = "🟡 CALCULATING"
            st.metric("🔥 Volatility", vol_regime)

        with col3:
            # Volume regime
            if 'volume' in df.columns and len(df) >= 20:
                recent_vol = df['volume'].tail(20).mean()
                avg_vol = df['volume'].mean()
                vol_regime = "📈 HIGH" if recent_vol > avg_vol * 1.2 else "📉 LOW"
            else:
                vol_regime = "🟡 N/A"
            st.metric("📊 Volume", vol_regime)

        st.markdown('</div>', unsafe_allow_html=True)

    def create_enhanced_candlestick_chart(self, df, symbol, timeframe):
        """Create enhanced candlestick chart as main chart"""
        st.markdown("## 🕯️ Enhanced Candlestick Analysis")

        # Create subplot
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=[
                f"{symbol} {timeframe} - Price Action & Technical Analysis",
                "Volume Analysis", 
                "Momentum Indicators"
            ],
            vertical_spacing=0.05,
            row_heights=[0.6, 0.2, 0.2],
            shared_xaxes=True
        )

        # Main candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name="Price",
                increasing_line_color='#00ff88',
                decreasing_line_color='#ff4444',
                increasing_fillcolor='#00ff88',
                decreasing_fillcolor='#ff4444'
            ), row=1, col=1
        )

        # Add moving averages if available
        ma_colors = {
            'ema_8': '#ff6b6b', 'EMA_8': '#ff6b6b',
            'ema_21': '#4ecdc4', 'EMA_21': '#4ecdc4', 
            'ema_55': '#45b7d1', 'EMA_55': '#45b7d1',
            'sma_200': '#96ceb4', 'SMA_200': '#96ceb4'
        }

        for ma_col in df.columns:
            if ma_col in ma_colors:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[ma_col],
                    mode='lines', name=ma_col,
                    line=dict(color=ma_colors[ma_col], width=2)
                ), row=1, col=1)

        # Add Bollinger Bands if available
        bb_cols = [col for col in df.columns if 'BB_' in col]
        if len(bb_cols) >= 2:
            upper_col = [col for col in bb_cols if 'Upper' in col][0] if [col for col in bb_cols if 'Upper' in col] else bb_cols[0]
            lower_col = [col for col in bb_cols if 'Lower' in col][0] if [col for col in bb_cols if 'Lower' in col] else bb_cols[1]

            fig.add_trace(go.Scatter(
                x=df.index, y=df[upper_col],
                mode='lines', name='BB Upper',
                line=dict(color='gray', width=1, dash='dash'),
                showlegend=False
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=df.index, y=df[lower_col],
                mode='lines', name='BB Lower',
                line=dict(color='gray', width=1, dash='dash'),
                fill='tonexty', fillcolor='rgba(128,128,128,0.1)',
                showlegend=False
            ), row=1, col=1)

        # Volume chart
        if 'volume' in df.columns:
            colors = ['#00ff88' if close >= open_val else '#ff4444' 
                     for close, open_val in zip(df['close'], df['open'])]

            fig.add_trace(go.Bar(
                x=df.index, y=df['volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7
            ), row=2, col=1)

        # RSI
        rsi_cols = [col for col in df.columns if 'rsi' in col.lower() or 'RSI' in col]
        if rsi_cols:
            rsi_col = rsi_cols[0]
            fig.add_trace(go.Scatter(
                x=df.index, y=df[rsi_col],
                mode='lines', name=f'RSI',
                line=dict(color='purple', width=2)
            ), row=3, col=1)

            # RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1)
            fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.3, row=3, col=1)

        # Update layout
        fig.update_layout(
            title=f"{symbol} {timeframe} - Enhanced Candlestick Analysis",
            template=st.session_state.get('chart_theme', 'plotly_dark'),
            height=800,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )

        # Remove x-axis labels for upper subplots
        fig.update_xaxes(showticklabels=False, row=1, col=1)
        fig.update_xaxes(showticklabels=False, row=2, col=1)

        st.plotly_chart(fig, use_container_width=True)

    def create_multitimeframe_analysis(self, symbol):
        """Create multi-timeframe analysis"""
        st.markdown("## ⏱️ Multi-Timeframe Analysis")

        available_tfs = self.available_timeframes.get(symbol, [])
        if len(available_tfs) < 2:
            st.info("Multi-timeframe analysis requires multiple timeframes")
            return

        # Load data for multiple timeframes
        tf_data = {}
        for tf in available_tfs[:4]:  # Limit to 4 timeframes
            df = self.load_symbol_data(symbol, tf)
            if df is not None and len(df) > 0:
                tf_data[tf] = df

        if not tf_data:
            st.warning("No data available for multi-timeframe analysis")
            return

        # Create comparison metrics
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📊 Timeframe Comparison")

            comparison_data = []
            for tf, df in tf_data.items():
                if len(df) > 1:
                    current_price = df['close'].iloc[-1]
                    price_change = ((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100

                    # Calculate trend
                    if len(df) >= 20:
                        short_ma = df['close'].rolling(8).mean().iloc[-1]
                        long_ma = df['close'].rolling(21).mean().iloc[-1]
                        trend = "🟢" if short_ma > long_ma else "🔴"
                    else:
                        trend = "🟡"

                    comparison_data.append({
                        'Timeframe': tf,
                        'Current Price': f"{current_price:.4f}",
                        'Change %': f"{price_change:+.2f}%",
                        'Trend': trend,
                        'Bars': len(df)
                    })

            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)

        with col2:
            st.markdown("### 📈 Timeframe Alignment")

            # Check for alignment across timeframes
            trends = []
            for tf, df in tf_data.items():
                if len(df) >= 20:
                    short_ma = df['close'].rolling(8).mean().iloc[-1]
                    long_ma = df['close'].rolling(21).mean().iloc[-1]
                    trends.append(1 if short_ma > long_ma else -1)

            if trends:
                bullish_count = sum(1 for t in trends if t > 0)
                bearish_count = sum(1 for t in trends if t < 0)

                if bullish_count > bearish_count:
                    alignment = "🟢 BULLISH ALIGNMENT"
                    strength = f"{bullish_count}/{len(trends)} timeframes bullish"
                elif bearish_count > bullish_count:
                    alignment = "🔴 BEARISH ALIGNMENT"
                    strength = f"{bearish_count}/{len(trends)} timeframes bearish"
                else:
                    alignment = "🟡 MIXED SIGNALS"
                    strength = "No clear alignment"

                st.metric("Market Alignment", alignment)
                st.metric("Alignment Strength", strength)

        # Multi-timeframe chart
        self.create_multitimeframe_chart(tf_data, symbol)

    def create_multitimeframe_chart(self, tf_data, symbol):
        """Create multi-timeframe comparison chart"""
        try:
            fig = go.Figure()

            colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']

            for i, (tf, df) in enumerate(tf_data.items()):
                if len(df) > 0:
                    # Normalize prices to percentage change
                    normalized = ((df['close'] / df['close'].iloc[0]) - 1) * 100

                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=normalized,
                        mode='lines',
                        name=f"{tf}",
                        line=dict(color=colors[i % len(colors)], width=2)
                    ))

            fig.update_layout(
                title=f"{symbol} - Multi-Timeframe Performance Comparison",
                xaxis_title="Time",
                yaxis_title="Performance (%)",
                template=st.session_state.get('chart_theme', 'plotly_dark'),
                height=400,
                showlegend=True
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error creating multi-timeframe chart: {e}")

    def display_txt_report_analysis(self, symbol):
        """Display conclusive analysis from TXT files"""
        st.markdown("## 📄 Conclusive Analysis Report")

        txt_content = self.load_txt_report(symbol)

        if txt_content:
            # Parse key sections from the TXT report
            self.parse_and_display_txt_analysis(txt_content)
        else:
            st.info(f"No TXT analysis report found for {symbol}")

    def parse_and_display_txt_analysis(self, txt_content):
        """Parse and display key insights from TXT analysis"""
        try:
            # Split into sections
            sections = txt_content.split('----')

            for section in sections:
                if not section.strip():
                    continue

                lines = section.strip().split('\n')
                if not lines:
                    continue

                header = lines[0].strip()

                if 'MICROSTRUCTURE ANALYSIS' in header:
                    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
                    st.markdown("### 🔍 Microstructure Insights")
                    content = '\n'.join(lines[1:])
                    if 'Average Spread:' in content:
                        spread_match = re.search(r'Average Spread: ([\d.]+)', content)
                        if spread_match:
                            st.metric("Average Spread", f"{spread_match.group(1)} pips")
                    st.markdown(content)
                    st.markdown('</div>', unsafe_allow_html=True)

                elif 'MANIPULATION DETECTION' in header:
                    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
                    st.markdown("### 🛡️ Manipulation Analysis")
                    content = '\n'.join(lines[1:])

                    # Extract manipulation score
                    if 'Manipulation Activity Score:' in content:
                        score_match = re.search(r'Manipulation Activity Score: ([\d.]+)%', content)
                        if score_match:
                            score = float(score_match.group(1))
                            color_class = "danger-metric" if score > 30 else "warning-metric" if score > 15 else "success-metric"
                            st.markdown(f'<p class="{color_class}">Manipulation Score: {score}%</p>', unsafe_allow_html=True)

                    st.markdown(content)
                    st.markdown('</div>', unsafe_allow_html=True)

                elif 'TRADING STRATEGY RECOMMENDATIONS' in header:
                    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
                    st.markdown("### 🎯 Trading Recommendations")
                    content = '\n'.join(lines[1:])

                    # Parse recommendations
                    recommendations = []
                    for line in lines[1:]:
                        if line.strip().startswith('✅'):
                            recommendations.append(('success', line.strip()))
                        elif line.strip().startswith('⚠️'):
                            recommendations.append(('warning', line.strip()))
                        elif line.strip().startswith('📍'):
                            recommendations.append(('info', line.strip()))

                    for rec_type, rec_text in recommendations:
                        if rec_type == 'success':
                            st.success(rec_text)
                        elif rec_type == 'warning':
                            st.warning(rec_text)
                        else:
                            st.info(rec_text)

                    st.markdown('</div>', unsafe_allow_html=True)

                elif 'EXPERT COMMENTARY' in header:
                    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
                    st.markdown("### 🧠 Expert Commentary")
                    content = '\n'.join(lines[1:])

                    # Color-code different types of commentary
                    for line in lines[1:]:
                        line = line.strip()
                        if line.startswith('🚨'):
                            st.error(line)
                        elif line.startswith('📈'):
                            st.success(line)
                        elif line.startswith('🪤'):
                            st.warning(line)
                        elif line:
                            st.info(line)

                    st.markdown('</div>', unsafe_allow_html=True)

            # Show full report in expander
            with st.expander("📋 View Full Analysis Report"):
                st.text(txt_content)

        except Exception as e:
            st.error(f"Error parsing TXT analysis: {e}")
            with st.expander("📋 Raw Report Content"):
                st.text(txt_content)

    def create_microstructure_section(self, df):
        """Create microstructure analysis section"""
        st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
        st.markdown("### 🔍 Microstructure Analysis")

        # Check for tick-level data
        tick_columns = ['bid', 'ask', 'spread', 'volume', 'flags']
        has_tick_data = any(col in df.columns for col in tick_columns)

        if has_tick_data:
            if 'bid' in df.columns and 'ask' in df.columns:
                spread = df['ask'] - df['bid']

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Avg Spread", f"{spread.mean():.5f}")
                with col2:
                    st.metric("Max Spread", f"{spread.max():.5f}")
                with col3:
                    st.metric("Spread Volatility", f"{spread.std():.5f}")

            # Check for manipulation indicators
            manipulation_cols = [col for col in df.columns if any(x in col.lower() for x in 
                               ['spoofing', 'layering', 'stuffing', 'momentum_ignition'])]

            if manipulation_cols:
                st.markdown("#### 🛡️ Manipulation Detection")
                for col in manipulation_cols[:3]:  # Show top 3
                    if col in df.columns:
                        recent_count = df[col].tail(50).sum() if df[col].dtype in ['bool', 'int64'] else 0
                        st.metric(col.replace('_', ' ').title(), int(recent_count))

        else:
            # Basic microstructure from OHLCV
            if len(df) > 1:
                price_changes = df['close'].pct_change().dropna()

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Price Volatility", f"{price_changes.std():.6f}")
                with col2:
                    st.metric("Avg Price Change", f"{price_changes.mean():.6f}")
                with col3:
                    st.metric("Max Price Move", f"{abs(price_changes).max():.6f}")

        st.markdown('</div>', unsafe_allow_html=True)

    def create_smart_money_section(self, df):
        """Create Smart Money Concepts section"""
        st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
        st.markdown("### 🧠 Smart Money Concepts")

        # Look for SMC indicators
        smc_cols = [col for col in df.columns if any(x in col.lower() for x in 
                   ['fvg', 'order_block', 'structure', 'liquidity', 'sweep'])]

        if smc_cols:
            # Count recent SMC events
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 🟢 Bullish Signals")
                bullish_cols = [col for col in smc_cols if 'bullish' in col.lower()]
                for col in bullish_cols[:3]:
                    if col in df.columns:
                        count = df[col].tail(50).sum() if df[col].dtype in ['bool', 'int64'] else 0
                        st.metric(col.replace('_', ' ').title(), int(count))

            with col2:
                st.markdown("#### 🔴 Bearish Signals")
                bearish_cols = [col for col in smc_cols if 'bearish' in col.lower()]
                for col in bearish_cols[:3]:
                    if col in df.columns:
                        count = df[col].tail(50).sum() if df[col].dtype in ['bool', 'int64'] else 0
                        st.metric(col.replace('_', ' ').title(), int(count))

        else:
            # Basic SMC analysis
            if len(df) >= 20:
                # Simple structure analysis
                highs = df['high'].rolling(10).max()
                lows = df['low'].rolling(10).min()

                higher_highs = (df['high'] > highs.shift(1)).sum()
                lower_lows = (df['low'] < lows.shift(1)).sum()

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Higher Highs", int(higher_highs))
                with col2:
                    st.metric("Lower Lows", int(lower_lows))

        st.markdown('</div>', unsafe_allow_html=True)

    def create_wyckoff_section(self, df):
        """Create Wyckoff analysis section"""
        st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
        st.markdown("### 📈 Wyckoff Analysis")

        # Look for Wyckoff phases
        wyckoff_cols = [col for col in df.columns if 'wyckoff' in col.lower() or 'phase' in col.lower()]

        if wyckoff_cols:
            phase_col = wyckoff_cols[0]
            if phase_col in df.columns:
                current_phase = df[phase_col].iloc[-1] if not pd.isna(df[phase_col].iloc[-1]) else 0

                phase_names = {1: 'Accumulation', 2: 'Distribution', 3: 'Markup', 4: 'Markdown'}
                current_phase_name = phase_names.get(current_phase, 'Unknown')

                st.metric("Current Phase", current_phase_name)

                # Phase distribution
                if len(df) >= 50:
                    phase_counts = df[phase_col].tail(50).value_counts()

                    col1, col2, col3, col4 = st.columns(4)
                    for i, (col, phase) in enumerate(zip([col1, col2, col3, col4], [1, 2, 3, 4])):
                        count = phase_counts.get(phase, 0)
                        with col:
                            st.metric(phase_names[phase], int(count))

        else:
            # Basic Wyckoff analysis using volume and price
            if 'volume' in df.columns and len(df) >= 20:
                # Volume analysis for Wyckoff
                high_volume = df['volume'] > df['volume'].quantile(0.8)
                price_change = abs(df['close'].pct_change())
                low_price_change = price_change < price_change.quantile(0.2)

                # Effort vs Result
                effort_result = (high_volume & low_price_change).sum()

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("High Vol/Low Move", int(effort_result))
                with col2:
                    avg_volume = df['volume'].tail(20).mean()
                    historical_avg = df['volume'].mean()
                    vol_ratio = avg_volume / historical_avg if historical_avg > 0 else 1
                    st.metric("Volume Ratio", f"{vol_ratio:.2f}x")

        st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main application entry point"""
    dashboard = ZANFLOWUltimateEnhancedDashboard()
    dashboard.create_enhanced_dashboard()

if __name__ == "__main__":
    main()
