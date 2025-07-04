#!/usr/bin/env python3
"""
ZANFLOW v12 Ultimate Trading Dashboard
Comprehensive microstructure, SMC, Wyckoff, and top-down analysis
Reads processed data from convert_final_enhanced_smc_ULTIMATE.py
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
import sqlite3
from typing import Dict, List, Optional, Tuple, Any
import re
import base64
warnings.filterwarnings('ignore')

# --- Parquet scanning helper ---
def scan_parquet_files(data_dir):
    """Scan directory for Parquet files, returning (symbol, timeframe, rel_path) with folder or filename fallback."""
    files = []
    for f in Path(data_dir).rglob("*.parquet"):
        name = f.stem.upper()
        parent = f.parent.name.upper()
        # If filename looks like timeframe, parent is symbol
        if re.match(r"\d+[A-Z]+", name):
            symbol = parent
            timeframe = name
        # If folder looks like timeframe, filename is symbol
        elif re.match(r"\d+[A-Z]+", parent):
            symbol = name
            timeframe = parent
        # Fallback: try SYMBOL_TIMEFRAME in filename
        else:
            m = re.match(r"(.*?)_(\d+[a-zA-Z]+)$", name, re.IGNORECASE)
            if m:
                symbol, timeframe = m.group(1).upper(), m.group(2).upper()
            else:
                continue
        files.append((symbol, timeframe, f.relative_to(data_dir)))
    return files

class UltimateZANFLOWDashboard:
    def __init__(self, data_directory=None):
        # Pull path from .streamlit/secrets.toml if not passed
        if data_directory is None:
            data_directory = st.secrets.get("PARQUET_DATA_DIR", "./data")
        self.data_dir = Path(data_directory)
        self.pairs_data = {}
        self.analysis_reports = {}
        self.smc_analysis = {}
        self.wyckoff_analysis = {}
        self.microstructure_data = {}
        # Store latest .txt and .json insights per pair
        self.latest_txt_reports = {}
        self.latest_json_insights = {}

    def load_all_data(self):
        """Load all processed data silently"""
        # CSV loading logic removed - Parquet only mode
        pass

    def create_main_dashboard(self):
        """Create the ultimate dashboard interface"""
        st.set_page_config(
            page_title="ZANFLOW v12 Ultimate",
            page_icon="🚀",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # --- PATCH: HOME background and style ---
        def get_image_as_base64(path):
            try:
                with open(path, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode()
            except Exception:
                return None

        img_base64 = get_image_as_base64("./pages/image_af247b.jpg")
        if img_base64:
            st.markdown(f"""
            <style>
            [data-testid="stAppViewContainer"] > .main {{
                background-image: linear-gradient(rgba(0,0,0,0.8), rgba(0,0,0,0.8)), url(data:image/jpeg;base64,{img_base64});
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            .main .block-container {{
                background-color: rgba(0,0,0,0.025) !important;
            }}
            #MainMenu {{visibility: hidden;}}
            footer {{visibility: hidden;}}
            </style>
            """, unsafe_allow_html=True)

        # Remove old .main-header CSS block (now commented out)
        # st.markdown("""
        # <style>
        # .main-header {
        #     background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        #     padding: 1rem;
        #     border-radius: 10px;
        #     color: white;
        #     text-align: center;
        #     margin-bottom: 2rem;
        # }
        # .metric-card {
        #     background: #f0f2f6;
        #     padding: 1rem;
        #     border-radius: 8px;
        #     border-left: 4px solid #1e3c72;
        # }
        # .wyckoff-phase {
        #     padding: 0.5rem;
        #     border-radius: 5px;
        #     margin: 0.25rem 0;
        #     font-weight: bold;
        # }
        # .accumulation { background-color: #e8f5e8; color: #2e7d2e; }
        # .distribution { background-color: #ffe8e8; color: #cc0000; }
        # .markup { background-color: #e8f8ff; color: #0066cc; }
        # .markdown { background-color: #fff3e0; color: #e65100; }
        # </style>
        # """, unsafe_allow_html=True)

        # Dashboard main title - match HOME
        st.markdown(
            """
            <div style='text-align:center; margin-bottom:2rem;'>
                <span style='font-size:2rem; font-weight:700; color:#fff; letter-spacing:0.03em;'>
                    ZANFLOW Ultimate Dashboard
                </span>
                <br>
                <span style='font-size:1.1rem; color:#eee; font-weight:500;'>
                    Comprehensive SMC, Wyckoff & Microstructure Analytics
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Legacy loader kept for backward-compat, but no longer required
        try:
            self.load_all_data()
        except Exception:
            pass  # ignore if loader fails – we now rely on Parquet scanning

        # Sidebar controls
        self.create_sidebar_controls()

        # Main content area
        if (
            st.session_state.get('selected_pair') and
            st.session_state.get('selected_timeframe') and
            'df_to_use' in st.session_state
        ):
            self.display_ultimate_analysis()
        else:
            # Apply consistent background and style for HOME view
            self.display_market_overview()

    def create_sidebar_controls(self):
        """Create comprehensive sidebar controls for Parquet-based analysis"""
        st.sidebar.title("🎛️ Analysis Control Center")

        # Scan Parquet files and extract symbol/timeframe
        file_info = scan_parquet_files(self.data_dir)
        symbols = sorted({sym for sym, _, _ in file_info})
        if not symbols:
            st.sidebar.error("❌ No Parquet files found.")
            return

        selected_pair = st.sidebar.selectbox("📈 Select Symbol", symbols, key="selected_pair")

        available_timeframes = sorted({tf for sym, tf, _ in file_info if sym == selected_pair})
        if not available_timeframes:
            st.sidebar.error("❌ No timeframes found for selected symbol.")
            return

        selected_timeframe = st.sidebar.selectbox("⏱️ Select Timeframe", available_timeframes, key="selected_timeframe")

        # Find the actual Parquet file for symbol+tf
        try:
            rel_path = next(f for sym, tf, f in file_info if sym == selected_pair and tf == selected_timeframe)
        except StopIteration:
            st.sidebar.error("No data found for this symbol/timeframe.")
            return

        full_path = self.data_dir / rel_path
        df = pd.read_parquet(full_path)
        df.columns = [c.lower() for c in df.columns]  # Lowercase columns for robustness

        # SLIDER: bars to use
        max_bars = len(df)
        lookback = st.sidebar.slider("Lookback Period", min_value=20, max_value=max_bars, value=min(500, max_bars), key="lookback_bars")

        # Only store DataFrame in session for use in analysis
        st.session_state['df_to_use'] = df.tail(lookback)

        # Market status
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Market Status")

        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Price", f"{df['close'].iloc[-1]:.5f}")
        with col2:
            price_change = ((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100 if len(df) > 1 else 0
            st.metric("Change", f"{price_change:+.2f}%")

        st.sidebar.info(f"""
        🔢 **Bars:** {len(df):,}
        📅 **From:** {pd.to_datetime(df.index.min()).strftime('%Y-%m-%d')}
        📅 **To:** {pd.to_datetime(df.index.max()).strftime('%Y-%m-%d')}
        💹 **Range:** {df['low'].min():.5f} - {df['high'].max():.5f}
        """)

        # Analysis options
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🔬 Analysis Options")

        st.session_state['show_microstructure'] = st.sidebar.checkbox("🔍 Microstructure Analysis", True)
        st.session_state['show_smc'] = st.sidebar.checkbox("🧠 Smart Money Concepts", True)
        st.session_state['show_wyckoff'] = st.sidebar.checkbox("📈 Wyckoff Analysis", True)
        st.session_state['show_patterns'] = st.sidebar.checkbox("🎯 Pattern Recognition", True)
        st.session_state['show_volume'] = st.sidebar.checkbox("📊 Volume Analysis", True)
        # st.session_state['show_risk'] = st.sidebar.checkbox("⚠️ Risk Metrics", True)

        # Chart options
        st.sidebar.markdown("### 📈 Chart Settings")
        st.session_state['chart_theme'] = st.sidebar.selectbox("Chart Theme", ["plotly_dark", "plotly_white", "ggplot2"])

    def display_market_overview(self):
        """Display comprehensive market overview"""
        st.markdown("## 🌍 Market Overview & Analysis Summary")

        # Market statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("<div style='color:#ffeb3b; font-size:1.2rem;'>Currency Pairs</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='color:#fff; font-size:1.6rem; font-weight:bold;'>{len(self.pairs_data)}</div>", unsafe_allow_html=True)

        with col2:
            total_timeframes = sum(len(timeframes) for timeframes in self.pairs_data.values())
            st.markdown("<div style='color:#ffeb3b; font-size:1.2rem;'>Total Datasets</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='color:#fff; font-size:1.6rem; font-weight:bold;'>{total_timeframes}</div>", unsafe_allow_html=True)

        with col3:
            total_data_points = sum(
                len(df) for pair_data in self.pairs_data.values()
                for df in pair_data.values()
            )
            st.markdown("<div style='color:#ffeb3b; font-size:1.2rem;'>Total Bars</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='color:#fff; font-size:1.6rem; font-weight:bold;'>{total_data_points:,}</div>", unsafe_allow_html=True)

        with col4:
            avg_indicators = np.mean([
                len([col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']])
                for pair_data in self.pairs_data.values()
                for df in pair_data.values()
            ]) if self.pairs_data else 0
            st.markdown("<div style='color:#ffeb3b; font-size:1.2rem;'>Avg Indicators</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='color:#fff; font-size:1.6rem; font-weight:bold;'>{int(avg_indicators)}</div>", unsafe_allow_html=True)

        # Market heatmap
        st.markdown("### 🔥 Market Heatmap")
        self.create_market_heatmap()

        # Top movers
        st.markdown("### 📊 Top Movers Analysis")
        self.create_top_movers_analysis()

        # Correlation matrix
        st.markdown("### 🔗 Correlation Analysis")
        self.create_correlation_matrix()

    def display_home_page(self, *args, **kwargs):
        """Wrapper for backward compatibility"""
        return self.display_market_overview(*args, **kwargs)

    def create_market_heatmap(self):
        """Create market performance heatmap"""
        try:
            heatmap_data = []

            for pair, timeframes in self.pairs_data.items():
                pair_performance = {}

                for tf, df in timeframes.items():
                    if len(df) > 1:
                        performance = ((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100
                        pair_performance[tf] = performance

                if pair_performance:
                    heatmap_data.append({
                        'Pair': pair,
                        **pair_performance
                    })

            if heatmap_data:
                heatmap_df = pd.DataFrame(heatmap_data).set_index('Pair')

                fig = px.imshow(
                    heatmap_df.values,
                    x=heatmap_df.columns,
                    y=heatmap_df.index,
                    color_continuous_scale='RdYlGn',
                    aspect="auto",
                    title="Market Performance Heatmap (%)"
                )

                fig.update_layout(
                    template=st.session_state.get('chart_theme', 'plotly_dark'),
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error creating heatmap: {e}")

    def create_top_movers_analysis(self):
        """Analyze top moving pairs"""
        try:
            movers_data = []

            for pair, timeframes in self.pairs_data.items():
                if '1H' in timeframes:  # Use 1H for daily analysis
                    df = timeframes['1H']
                    if len(df) > 24:  # At least 24 hours of data
                        recent_24h = df.tail(24)
                        performance = ((recent_24h['close'].iloc[-1] / recent_24h['close'].iloc[0]) - 1) * 100
                        volatility = recent_24h['close'].pct_change().std() * np.sqrt(24) * 100

                        # Get additional metrics if available
                        rsi = df['rsi_14'].iloc[-1] if 'rsi_14' in df.columns else 50
                        atr = df['atr_14'].iloc[-1] if 'atr_14' in df.columns else 0

                        movers_data.append({
                            'Pair': pair,
                            'Performance_24h': performance,
                            'Volatility': volatility,
                            'RSI': rsi,
                            'ATR': atr,
                            'Current_Price': df['close'].iloc[-1]
                        })

            if movers_data:
                movers_df = pd.DataFrame(movers_data)

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### 📈 Top Gainers (24H)")
                    top_gainers = movers_df.nlargest(5, 'Performance_24h')
                    for _, row in top_gainers.iterrows():
                        st.markdown(f"""
                        **{row['Pair']}**: {row['Performance_24h']:+.2f}% | RSI: {row['RSI']:.1f}
                        """)

                with col2:
                    st.markdown("#### 📉 Top Losers (24H)")
                    top_losers = movers_df.nsmallest(5, 'Performance_24h')
                    for _, row in top_losers.iterrows():
                        st.markdown(f"""
                        **{row['Pair']}**: {row['Performance_24h']:+.2f}% | RSI: {row['RSI']:.1f}
                        """)

        except Exception as e:
            st.error(f"Error analyzing top movers: {e}")

    def create_correlation_matrix(self):
        """Create correlation matrix of pairs"""
        try:
            # Get price data for correlation
            price_data = {}

            for pair, timeframes in self.pairs_data.items():
                if '1H' in timeframes:  # Use 1H timeframe
                    df = timeframes['1H']
                    if len(df) > 100:
                        # Get returns
                        returns = df['close'].pct_change().dropna()
                        if len(returns) > 50:
                            price_data[pair] = returns.tail(100)  # Last 100 returns

            if len(price_data) >= 2:
                # Align data
                aligned_data = pd.DataFrame(price_data).dropna()

                if len(aligned_data) > 10:
                    # Calculate correlation
                    correlation_matrix = aligned_data.corr()

                    fig = px.imshow(
                        correlation_matrix.values,
                        x=correlation_matrix.columns,
                        y=correlation_matrix.index,
                        color_continuous_scale='RdBu',
                        aspect="auto",
                        title="Price Correlation Matrix"
                    )

                    fig.update_layout(
                        template=st.session_state.get('chart_theme', 'plotly_dark'),
                        height=500
                    )

                    st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error creating correlation matrix: {e}")

    def display_ultimate_analysis(self):
        """Display comprehensive analysis dashboard"""
        pair = st.session_state['selected_pair']
        timeframe = st.session_state['selected_timeframe']

        # Use Parquet session state only
        df = st.session_state['df_to_use']
        df_display = df.copy()

        st.markdown(f"# 🚀 {pair} {timeframe} - Ultimate Analysis")

        # Market status row
        self.display_market_status(df_display, pair)

        # Main price chart with comprehensive overlays
        self.create_ultimate_price_chart(df_display, pair, timeframe)

        # Analysis sections based on user selection
        if st.session_state.get('show_microstructure', True):
            self.create_microstructure_analysis(df_display)

        if st.session_state.get('show_smc', True):
            self.create_comprehensive_smc_analysis(df_display)

        if st.session_state.get('show_wyckoff', True):
            self.create_comprehensive_wyckoff_analysis(df_display)

        if st.session_state.get('show_patterns', True):
            self.create_pattern_analysis(df_display)

        # Technical analysis panels
        col1, col2 = st.columns(2)
        with col1:
            if st.session_state.get('show_volume', True):
                self.create_advanced_volume_analysis(df_display)
        # with col2:
        #     if st.session_state.get('show_risk', True):
        #         self.create_risk_analysis(df_display)

        # Latest TXT insights
        st.markdown("## 🧾 Latest Report Insights")

        if pair in self.latest_txt_reports:
            st.expander("📄 View Latest TXT Report").markdown(f"```text\n{self.latest_txt_reports[pair]}\n```")

        # Advanced analytics
        self.create_advanced_analytics_panel(df_display)

    def display_market_status(self, df, pair):
        """Display comprehensive market status"""
        col1, col2, col3, col4, col5, col6 = st.columns(6)

        current_price = df['close'].iloc[-1]
        price_change = df['close'].iloc[-1] - df['close'].iloc[-50] if len(df) > 50 else 0
        price_change_pct = (price_change / df['close'].iloc[-50]) * 100 if len(df) > 50 and df['close'].iloc[-50] != 0 else 0

        with col1:
            st.metric("Current Price", f"{current_price:.4f}", f"{price_change:+.4f}")

        with col2:
            st.metric("Change %", f"{price_change_pct:+.2f}%")

        with col3:
            atr = df['atr_14'].iloc[-1] if 'atr_14' in df.columns else 0
            try:
                if hasattr(atr, "__iter__") and not isinstance(atr, str):
                    # handle Series
                    if len(atr) == 1:
                        atr = float(atr.item())
                    else:
                        atr = float(atr.values[-1])
                atr_value = f"{float(atr):.4f}"
            except Exception:
                atr_value = "—"
            st.metric("ATR (14)", atr_value)

        with col4:
            rsi = df['rsi_14'].iloc[-1] if 'rsi_14' in df.columns else 50
            rsi_color = "🟢" if rsi < 30 else "🔴" if rsi > 70 else "🟡"
            st.metric("RSI (14)", f"{rsi_color} {rsi:.1f}")

        with col5:
            # Market regime detection
            if 'ema_8' in df.columns and 'ema_21' in df.columns:
                trend = "🟢 BULL" if df['ema_8'].iloc[-1] > df['ema_21'].iloc[-1] else "🔴 BEAR"
            else:
                trend = "🟡 UNKNOWN"
            st.metric("Trend", trend)

        with col6:
            volatility = df['close'].pct_change().std() if len(df) > 1 else 0
            vol_regime = "🔥 HIGH" if volatility > df['close'].pct_change().quantile(0.8) else "❄️ LOW"
            st.metric("Volatility", vol_regime)

    def create_ultimate_price_chart(self, df, pair, timeframe):
        """Create ultimate price chart with all overlays"""
        # Gold header above the chart
        st.markdown(
            f"<div style='text-align:center; font-size:1.12em; color:#ffe082; font-weight:700; margin-bottom:0.2em;'>"
            f"{pair} {timeframe} – Price Action & SMC Overlays"
            f"</div>",
            unsafe_allow_html=True
        )
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=[
                f"{pair} {timeframe} - Price Action & Smart Money Analysis",
                "Volume Profile",
                "Momentum Oscillators",
                "Market Structure"
            ],
            vertical_spacing=0.05,
            row_heights=[0.5, 0.2, 0.15, 0.15],
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
                increasing_line_color='lime',
                decreasing_line_color='red'
            ), row=1, col=1
        )

        # Moving averages with labels
        ma_colors = {'ema_8': '#ff6b6b', 'ema_21': '#4ecdc4', 'ema_55': '#45b7d1', 'sma_200': '#96ceb4'}
        for ma, color in ma_colors.items():
            if ma in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[ma],
                    mode='lines', name=ma.upper(),
                    line=dict(color=color, width=2)
                ), row=1, col=1)

        # Bollinger Bands
        if all(col in df.columns for col in ['BB_Upper_20', 'BB_Lower_20']):
            fig.add_trace(go.Scatter(
                x=df.index, y=df['BB_Upper_20'],
                mode='lines', name='BB Upper',
                line=dict(color='gray', width=1, dash='dash'),
                showlegend=False
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=df.index, y=df['BB_Lower_20'],
                mode='lines', name='BB Lower',
                line=dict(color='gray', width=1, dash='dash'),
                fill='tonexty', fillcolor='rgba(128,128,128,0.1)',
                showlegend=False
            ), row=1, col=1)

        # SMC Analysis overlays
        if st.session_state.get('show_smc', True):
            self.add_smc_overlays(fig, df, row=1)

        # Wyckoff Analysis overlays
        if st.session_state.get('show_wyckoff', True):
            self.add_wyckoff_overlays(fig, df, row=1)

        # Volume analysis
        if 'volume' in df.columns:
            colors = ['green' if close >= open_val else 'red'
                     for close, open_val in zip(df['close'], df['open'])]

            fig.add_trace(go.Bar(
                x=df.index, y=df['volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7
            ), row=2, col=1)

            # Volume moving average
            if 'volume_sma_20' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['volume_sma_20'],
                    mode='lines', name='Vol MA20',
                    line=dict(color='orange', width=2)
                ), row=2, col=1)

        # Momentum indicators
        if 'rsi_14' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df['rsi_14'],
                mode='lines', name='RSI 14',
                line=dict(color='purple', width=2)
            ), row=3, col=1)

            # RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1)

        # MACD
        if all(col in df.columns for col in ['MACD_12_26_9', 'MACDs_12_26_9']):
            fig.add_trace(go.Scatter(
                x=df.index, y=df['MACD_12_26_9'],
                mode='lines', name='MACD',
                line=dict(color='blue', width=2)
            ), row=4, col=1)

            fig.add_trace(go.Scatter(
                x=df.index, y=df['MACDs_12_26_9'],
                mode='lines', name='Signal',
                line=dict(color='red', width=2)
            ), row=4, col=1)

        # Update layout
        fig.update_layout(
            title="",  # Remove in-chart title
            template=st.session_state.get('chart_theme', 'plotly_dark'),
            paper_bgcolor='rgba(0,0,0,0.02)',
            plot_bgcolor='rgba(0,0,0,0.02)',
            showlegend=False,
            xaxis_rangeslider_visible=False,
            height=500,
            margin=dict(l=20, r=20, t=40, b=20)
        )

        st.plotly_chart(fig, use_container_width=True)

    def add_smc_overlays(self, fig, df, row=1):
        """Add Smart Money Concepts overlays"""
        # Fair Value Gaps
        if 'bullish_fvg' in df.columns:
            fvg_bullish = df[df['bullish_fvg'] == True]
            if not fvg_bullish.empty:
                fig.add_trace(go.Scatter(
                    x=fvg_bullish.index, y=fvg_bullish['low'],
                    mode='markers', name='Bullish FVG',
                    marker=dict(symbol='triangle-up', color='lime', size=12),
                    showlegend=True
                ), row=row, col=1)

        if 'bearish_fvg' in df.columns:
            fvg_bearish = df[df['bearish_fvg'] == True]
            if not fvg_bearish.empty:
                fig.add_trace(go.Scatter(
                    x=fvg_bearish.index, y=fvg_bearish['high'],
                    mode='markers', name='Bearish FVG',
                    marker=dict(symbol='triangle-down', color='red', size=12),
                    showlegend=True
                ), row=row, col=1)

        # Order Blocks
        if 'bullish_order_block' in df.columns:
            ob_bullish = df[df['bullish_order_block'] == True]
            if not ob_bullish.empty:
                fig.add_trace(go.Scatter(
                    x=ob_bullish.index, y=ob_bullish['low'],
                    mode='markers', name='Bullish OB',
                    marker=dict(symbol='square', color='lightgreen', size=10),
                    showlegend=True
                ), row=row, col=1)

        if 'bearish_order_block' in df.columns:
            ob_bearish = df[df['bearish_order_block'] == True]
            if not ob_bearish.empty:
                fig.add_trace(go.Scatter(
                    x=ob_bearish.index, y=ob_bearish['high'],
                    mode='markers', name='Bearish OB',
                    marker=dict(symbol='square', color='lightcoral', size=10),
                    showlegend=True
                ), row=row, col=1)

        # Structure breaks
        if 'structure_break' in df.columns:
            structure_breaks = df[df['structure_break'] == True]
            if not structure_breaks.empty:
                fig.add_trace(go.Scatter(
                    x=structure_breaks.index, y=structure_breaks['close'],
                    mode='markers', name='Structure Break',
                    marker=dict(symbol='x', color='yellow', size=15),
                    showlegend=True
                ), row=row, col=1)

    def add_wyckoff_overlays(self, fig, df, row=1):
        """Add Wyckoff analysis overlays"""
        # Wyckoff phases
        phase_colors = {
            1: 'blue',    # Accumulation
            2: 'red',     # Distribution  
            3: 'green',   # Markup
            4: 'orange'   # Markdown
        }

        if 'wyckoff_phase' in df.columns:
            for phase, color in phase_colors.items():
                phase_data = df[df['wyckoff_phase'] == phase]
                if not phase_data.empty:
                    phase_names = {1: 'Accumulation', 2: 'Distribution', 3: 'Markup', 4: 'Markdown'}
                    fig.add_trace(go.Scatter(
                        x=phase_data.index, y=phase_data['close'],
                        mode='markers', name=f'Wyckoff {phase_names[phase]}',
                        marker=dict(color=color, size=8, opacity=0.7),
                        showlegend=True
                    ), row=row, col=1)

    def create_microstructure_analysis(self, df):
        """Create comprehensive microstructure analysis"""
        st.markdown("## 🔍 Microstructure Analysis")

        # Bid-Ask analysis if available
        if 'bid' in df.columns and 'ask' in df.columns:
            col1, col2, col3, col4 = st.columns(4)

            spread = df['ask'] - df['bid']

            with col1:
                st.metric("Avg Spread", f"{spread.mean():.5f}")
            with col2:
                st.metric("Spread Volatility", f"{spread.std():.5f}")
            with col3:
                st.metric("Max Spread", f"{spread.max():.5f}")
            with col4:
                st.metric("Min Spread", f"{spread.min():.5f}")

        # Price impact analysis
        if len(df) > 1:
            price_changes = df['close'].pct_change().dropna()

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 📊 Price Change Distribution")
                fig = px.histogram(
                    price_changes,
                    nbins=50,
                    title="Price Change Distribution",
                    template=st.session_state.get('chart_theme', 'plotly_dark')
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("### 📈 Volatility Clustering")
                abs_returns = np.abs(price_changes)
                rolling_vol = abs_returns.rolling(20).mean()

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=rolling_vol.index,
                    y=rolling_vol.values,
                    mode='lines',
                    name='Rolling Volatility',
                    line=dict(color='orange')
                ))
                fig.update_layout(
                    title="20-Period Rolling Volatility",
                    template=st.session_state.get('chart_theme', 'plotly_dark')
                )
                st.plotly_chart(fig, use_container_width=True)

    def create_comprehensive_smc_analysis(self, df):
        """Create comprehensive Smart Money Concepts analysis"""
        st.markdown("## 🧠 Smart Money Concepts Analysis")

        # SMC metrics row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            bullish_fvgs = df['bullish_fvg'].sum() if 'bullish_fvg' in df.columns else 0
            bearish_fvgs = df['bearish_fvg'].sum() if 'bearish_fvg' in df.columns else 0
            st.metric("Fair Value Gaps", f"🟢{bullish_fvgs} | 🔴{bearish_fvgs}")

        with col2:
            bull_obs = df['bullish_order_block'].sum() if 'bullish_order_block' in df.columns else 0
            bear_obs = df['bearish_order_block'].sum() if 'bearish_order_block' in df.columns else 0
            st.metric("Order Blocks", f"🟢{bull_obs} | 🔴{bear_obs}")

        with col3:
            structure_breaks = df['structure_break'].sum() if 'structure_break' in df.columns else 0
            st.metric("Structure Breaks", structure_breaks)

        with col4:
            # Calculate SMC bias
            if bullish_fvgs + bull_obs > bearish_fvgs + bear_obs:
                bias = "🟢 BULLISH"
            elif bearish_fvgs + bear_obs > bullish_fvgs + bull_obs:
                bias = "🔴 BEARISH" 
            else:
                bias = "🟡 NEUTRAL"
            st.metric("SMC Bias", bias)

        # SMC detailed analysis
        tab1, tab2, tab3 = st.tabs(["📊 FVG Analysis", "🏗️ Order Blocks", "🔀 Market Structure"])

        with tab1:
            self.create_fvg_analysis(df)

        with tab2:
            self.create_order_block_analysis(df)

        with tab3:
            self.create_market_structure_analysis(df)

    def create_fvg_analysis(self, df):
        """Create Fair Value Gap analysis"""
        if 'bullish_fvg' in df.columns and 'bearish_fvg' in df.columns:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 🟢 Bullish Fair Value Gaps")
                bullish_fvgs = df[df['bullish_fvg'] == True]
                if not bullish_fvgs.empty:
                    for idx, row in bullish_fvgs.tail(5).iterrows():
                        label = idx.strftime('%Y-%m-%d %H:%M') if hasattr(idx, 'strftime') else str(idx)
                        st.markdown(f"""**{label}**  
Price: {row['close']:.4f} | Gap Size: {row.get('fvg_size', 0):.4f}
""")
                else:
                    st.info("No bullish FVGs detected in this period")

            with col2:
                st.markdown("#### 🔴 Bearish Fair Value Gaps")
                bearish_fvgs = df[df['bearish_fvg'] == True]
                if not bearish_fvgs.empty:
                    for idx, row in bearish_fvgs.tail(5).iterrows():
                        label = idx.strftime('%Y-%m-%d %H:%M') if hasattr(idx, 'strftime') else str(idx)
                        st.markdown(f"""**{label}**  
Price: {row['close']:.4f} | Gap Size: {row.get('fvg_size', 0):.4f}
""")
                else:
                    st.info("No bearish FVGs detected in this period")

    def create_order_block_analysis(self, df):
        """Create Order Block analysis"""
        if 'bullish_order_block' in df.columns and 'bearish_order_block' in df.columns:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 🟢 Bullish Order Blocks")
                bullish_obs = df[df['bullish_order_block'] == True]
                if not bullish_obs.empty:
                    for idx, row in bullish_obs.tail(5).iterrows():
                        body_size = abs(row['close'] - row['open']) / row['open'] * 100
                        label = idx.strftime('%Y-%m-%d %H:%M') if hasattr(idx, 'strftime') else str(idx)
                        st.markdown(f"""**{label}**  
Price: {row['close']:.4f} | Body: {body_size:.2f}%
""")
                else:
                    st.info("No bullish order blocks detected")

            with col2:
                st.markdown("#### 🔴 Bearish Order Blocks")
                bearish_obs = df[df['bearish_order_block'] == True]
                if not bearish_obs.empty:
                    for idx, row in bearish_obs.tail(5).iterrows():
                        body_size = abs(row['close'] - row['open']) / row['open'] * 100
                        label = idx.strftime('%Y-%m-%d %H:%M') if hasattr(idx, 'strftime') else str(idx)
                        st.markdown(f"""**{label}**  
Price: {row['close']:.4f} | Body: {body_size:.2f}%
""")
                else:
                    st.info("No bearish order blocks detected")

    def create_market_structure_analysis(self, df):
        """Create market structure analysis"""
        # Support and resistance levels
        if 'resistance_20' in df.columns and 'support_20' in df.columns:
            current_price = df['close'].iloc[-1]
            resistance = df['resistance_20'].iloc[-1]
            support = df['support_20'].iloc[-1]

            col1, col2, col3 = st.columns(3)

            with col1:
                dist_to_resistance = ((resistance - current_price) / current_price) * 100
                st.metric("Distance to Resistance", f"{dist_to_resistance:.2f}%")

            with col2:
                dist_to_support = ((current_price - support) / current_price) * 100
                st.metric("Distance to Support", f"{dist_to_support:.2f}%")

            with col3:
                range_position = ((current_price - support) / (resistance - support)) * 100
                st.metric("Range Position", f"{range_position:.1f}%")

        # Structure break analysis
        if 'higher_high' in df.columns and 'lower_low' in df.columns:
            recent_hh = df['higher_high'].tail(50).sum()
            recent_ll = df['lower_low'].tail(50).sum()

            st.markdown("#### 📊 Recent Structure Analysis (Last 50 bars)")
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Higher Highs", recent_hh)
            with col2:
                st.metric("Lower Lows", recent_ll)

            if recent_hh > recent_ll:
                st.success("🟢 Bullish structure: More higher highs than lower lows")
            elif recent_ll > recent_hh:
                st.error("🔴 Bearish structure: More lower lows than higher highs")
            else:
                st.warning("🟡 Neutral structure: Balanced highs and lows")

    def create_comprehensive_wyckoff_analysis(self, df):
        """Create comprehensive Wyckoff analysis"""
        st.markdown("## 📈 Wyckoff Analysis")

        # Wyckoff phase distribution
        if 'wyckoff_phase' in df.columns:
            phase_counts = df['wyckoff_phase'].value_counts()
            phase_names = {1: 'Accumulation', 2: 'Distribution', 3: 'Markup', 4: 'Markdown'}

            col1, col2, col3, col4 = st.columns(4)

            for i, (col, phase) in enumerate(zip([col1, col2, col3, col4], [1, 2, 3, 4])):
                count = phase_counts.get(phase, 0)
                percentage = (count / len(df)) * 100 if len(df) > 0 else 0

                with col:
                    phase_class = ['accumulation', 'distribution', 'markup', 'markdown'][i]
                    st.markdown(f"""
                    <div class="wyckoff-phase {phase_class}">
                        <strong>{phase_names[phase]}</strong><br>
                        {count} bars ({percentage:.1f}%)
                    </div>
                    """, unsafe_allow_html=True)

            # Current phase analysis
            current_phase = df['wyckoff_phase'].iloc[-1] if not pd.isna(df['wyckoff_phase'].iloc[-1]) else 0
            if current_phase in phase_names:
                st.markdown(f"### Current Phase: **{phase_names[current_phase]}**")

                # Phase-specific analysis
                self.create_phase_specific_analysis(df, current_phase)

        # Volume analysis for Wyckoff
        if 'volume' in df.columns:
            self.create_wyckoff_volume_analysis(df)

    def create_phase_specific_analysis(self, df, phase):
        """Create analysis specific to current Wyckoff phase"""
        phase_names = {1: 'Accumulation', 2: 'Distribution', 3: 'Markup', 4: 'Markdown'}
        phase_name = phase_names.get(phase, 'Unknown')

        if phase == 1:  # Accumulation
            st.markdown("""
            #### 🔵 Accumulation Phase Characteristics:
            - **Smart Money**: Quietly accumulating positions
            - **Price Action**: Sideways with periodic shakeouts
            - **Volume**: High on dips, low on rallies
            - **Strategy**: Look for long opportunities on tests of support
            """)

        elif phase == 2:  # Distribution
            st.markdown("""
            #### 🔴 Distribution Phase Characteristics:
            - **Smart Money**: Quietly distributing positions to retail
            - **Price Action**: Sideways with periodic false breakouts
            - **Volume**: High on rallies, declining on pullbacks
            - **Strategy**: Look for short opportunities on tests of resistance
            """)

        elif phase == 3:  # Markup
            st.markdown("""
            #### 🟢 Markup Phase Characteristics:
            - **Smart Money**: Adding to long positions
            - **Price Action**: Strong uptrend with pullbacks
            - **Volume**: Increasing on rallies
            - **Strategy**: Buy pullbacks to support, ride the trend
            """)

        elif phase == 4:  # Markdown
            st.markdown("""
            #### 🟠 Markdown Phase Characteristics:
            - **Smart Money**: Adding to short positions
            - **Price Action**: Strong downtrend with bounces
            - **Volume**: Increasing on declines
            - **Strategy**: Sell bounces to resistance, ride the trend down
            """)

    def create_wyckoff_volume_analysis(self, df):
        """Create Wyckoff-specific volume analysis"""
        st.markdown("### 📊 Volume Analysis for Wyckoff")

        if 'volume' in df.columns:
            volume = df['volume']
            price_change = df['close'].pct_change()

            # Effort vs Result analysis
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 💪 Effort vs Result")

                # High volume, low price movement (possible accumulation/distribution)
                high_vol_threshold = volume.quantile(0.8)
                low_price_movement = abs(price_change) < price_change.std() * 0.5

                effort_result_events = df[
                    (volume > high_vol_threshold) & low_price_movement
                ].tail(10)

                if not effort_result_events.empty:
                    for idx, row in effort_result_events.iterrows():
                        label = idx.strftime('%Y-%m-%d %H:%M') if hasattr(idx, 'strftime') else str(idx)
                        st.markdown(f"""
                        **{label}**: High volume ({row['volume']:.0f}) 
                        but small price move - Potential accumulation/distribution
                        """)
                else:
                    st.info("No significant effort vs result divergences detected")

            with col2:
                st.markdown("#### 🎯 Volume Climax Events")

                # Volume climax (extremely high volume)
                volume_climax_threshold = volume.quantile(0.95)
                climax_events = df[volume > volume_climax_threshold].tail(5)

                if not climax_events.empty:
                    for idx, row in climax_events.iterrows():
                        price_move = ((row['close'] - row['open']) / row['open']) * 100
                        label = idx.strftime('%Y-%m-%d %H:%M') if hasattr(idx, 'strftime') else str(idx)
                        st.markdown(f"""
                        **{label}**: Volume climax ({row['volume']:.0f})
                        with {price_move:+.2f}% price move
                        """)
                else:
                    st.info("No volume climax events detected")

    def create_pattern_analysis(self, df):
        """Create comprehensive pattern analysis"""
        st.markdown("## 🎯 Pattern Recognition Analysis")

        # Candlestick patterns
        pattern_columns = [col for col in df.columns if any(pattern in col.lower() for pattern in
                          ['doji', 'hammer', 'engulfing', 'shooting_star', 'morning_star', 'evening_star'])]

        if pattern_columns:
            st.markdown("### 🕯️ Candlestick Patterns")

            pattern_counts = {}
            for col in pattern_columns:
                # Count non-zero pattern occurrences
                count = (df[col] != 0).sum()
                if count > 0:
                    pattern_counts[col] = count

            if pattern_counts:
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### 🟢 Bullish Patterns")
                    bullish_patterns = {k: v for k, v in pattern_counts.items()
                                      if any(bp in k.lower() for bp in ['hammer', 'morning_star'])}
                    for pattern, count in bullish_patterns.items():
                        st.markdown(f"**{pattern.replace('_', ' ').title()}**: {count} occurrences")

                with col2:
                    st.markdown("#### 🔴 Bearish Patterns")
                    bearish_patterns = {k: v for k, v in pattern_counts.items()
                                      if any(bp in k.lower() for bp in ['shooting_star', 'evening_star'])}
                    for pattern, count in bearish_patterns.items():
                        st.markdown(f"**{pattern.replace('_', ' ').title()}**: {count} occurrences")

                # Show recent patterns
                st.markdown("#### 🔍 Recent Pattern Occurrences")
                recent_patterns = []
                for col in pattern_columns:
                    recent_data = df[df[col] != 0].tail(5)
                    for idx, row in recent_data.iterrows():
                        label = idx.strftime('%Y-%m-%d %H:%M') if hasattr(idx, 'strftime') else str(idx)
                        recent_patterns.append({
                            'Date': label,
                            'Pattern': col.replace('_', ' ').title(),
                            'Value': row[col],
                            'Price': row['close']
                        })

                if recent_patterns:
                    patterns_df = pd.DataFrame(recent_patterns).sort_values('Date', ascending=False)
                    st.dataframe(patterns_df.head(10), use_container_width=True)

    def create_advanced_volume_analysis(self, df):
        """Create advanced volume analysis"""
        st.markdown("### 📊 Advanced Volume Analysis")

        if 'volume' in df.columns:
            # Volume metrics
            col1, col2 = st.columns(2)

            with col1:
                avg_volume = df['volume'].mean()
                current_volume = df['volume'].iloc[-1]
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0

                st.metric("Current Volume Ratio", f"{volume_ratio:.2f}x")

                # Volume trend
                volume_trend = df['volume'].rolling(20).mean().pct_change().iloc[-1] * 100
                st.metric("Volume Trend (20)", f"{volume_trend:+.1f}%")

            with col2:
                # Price-Volume correlation
                if len(df) > 20:
                    price_change = df['close'].pct_change()
                    volume_change = df['volume'].pct_change()
                    correlation = price_change.corr(volume_change)
                    st.metric("Price-Volume Correlation", f"{correlation:.3f}")

                # Volume volatility
                volume_volatility = df['volume'].pct_change().std()
                st.metric("Volume Volatility", f"{volume_volatility:.3f}")

            # Volume profile
            self.create_volume_profile_chart(df)

    def create_volume_profile_chart(self, df):
        """Create volume profile chart"""
        try:
            # Simple volume profile
            price_bins = pd.cut(df['close'], bins=20)
            volume_profile = df.groupby(price_bins)['volume'].sum().sort_index()

            fig = go.Figure()

            # Convert intervals to midpoints for plotting
            bin_midpoints = [interval.mid for interval in volume_profile.index]

            fig.add_trace(go.Bar(
                x=volume_profile.values,
                y=bin_midpoints,
                orientation='h',
                name='Volume Profile',
                marker_color='rgba(55, 128, 191, 0.7)'
            ))

            fig.update_layout(
                title="Volume Profile",
                xaxis_title="Volume",
                yaxis_title="Price",
                template=st.session_state.get('chart_theme', 'plotly_dark'),
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error creating volume profile: {e}")


    def create_advanced_analytics_panel(self, df):
        """Create advanced analytics panel"""
        st.markdown("## 🔬 Advanced Analytics")

        tab1, tab2, tab3 = st.tabs(["📊 Statistical Analysis", "🤖 ML Features", "🎯 Signal Analysis"])

        with tab1:
            self.create_statistical_analysis(df)

        with tab2:
            self.create_ml_features_analysis(df)

        with tab3:
            self.create_signal_analysis(df)

    def create_statistical_analysis(self, df):
        """Create statistical analysis"""
        if len(df) > 20:
            returns = df['close'].pct_change().dropna()

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 📈 Return Statistics")
                st.markdown(f"""
                - **Mean Return**: {returns.mean():.6f}
                - **Std Deviation**: {returns.std():.6f}
                - **Min Return**: {returns.min():.6f}
                - **Max Return**: {returns.max():.6f}
                - **Sharpe Ratio**: {(returns.mean() / returns.std()) * np.sqrt(252):.3f}
                """)

            with col2:
                st.markdown("#### 📊 Distribution Analysis")

                # Normality test (simplified)
                from scipy import stats

                try:
                    jb_stat, jb_pvalue = stats.jarque_bera(returns)
                    st.markdown(f"""
                    - **Jarque-Bera Stat**: {jb_stat:.3f}
                    - **P-value**: {jb_pvalue:.6f}
                    - **Normal Distribution**: {'Yes' if jb_pvalue > 0.05 else 'No'}
                    - **Excess Kurtosis**: {returns.kurtosis():.3f}
                    - **Skewness**: {returns.skew():.3f}
                    """)
                except:
                    st.info("Statistical tests unavailable")

    def create_ml_features_analysis(self, df):
        """Create ML features analysis"""
        st.markdown("#### 🤖 Machine Learning Feature Analysis")

        # Technical indicators summary
        technical_indicators = [col for col in df.columns
                              if any(ind in col.upper() for ind in
                                   ['RSI', 'MACD', 'ATR', 'BB', 'EMA', 'SMA'])]

        if technical_indicators:
            st.markdown(f"**Available Technical Indicators**: {len(technical_indicators)}")

            # Recent indicator values
            indicator_data = []
            for indicator in technical_indicators[:10]:  # Show first 10
                # PATCH: Safe type and column check before accessing df[indicator].iat[-1]
                if (
                    isinstance(indicator, str)
                    and indicator in df.columns
                    and isinstance(df[indicator], pd.Series)
                ):
                    val = df[indicator].iat[-1]
                    if not pd.isna(val):
                        indicator_data.append({
                            'Indicator': indicator,
                            'Current Value': f"{val:.4f}",
                            'Mean': f"{df[indicator].mean():.4f}",
                            'Std': f"{df[indicator].std():.4f}"
                        })
                else:
                    continue

            if indicator_data:
                indicators_df = pd.DataFrame(indicator_data)
                st.dataframe(indicators_df, use_container_width=True)

        # Pattern features
        pattern_features = [col for col in df.columns
                          if any(pattern in col.upper() for pattern in
                               ['FVG', 'ORDER_BLOCK', 'WYCKOFF', 'STRUCTURE'])]

        if pattern_features:
            st.markdown(f"**Available Pattern Features**: {len(pattern_features)}")

            # Recent pattern activity
            recent_patterns = []
            for pattern in pattern_features:
                if pattern in df.columns:
                    recent_activity = df[pattern].tail(20).sum()
                    if isinstance(recent_activity, (int, float)) and recent_activity > 0:
                        recent_patterns.append({
                            'Pattern': pattern,
                            'Recent Activity (20 bars)': int(recent_activity)
                        })

            if recent_patterns:
                patterns_df = pd.DataFrame(recent_patterns)
                st.dataframe(patterns_df, use_container_width=True)

    def create_signal_analysis(self, df):
        """Create trading signal analysis"""
        st.markdown("#### 🎯 Trading Signal Analysis")

        # Look for signal columns
        signal_columns = [col for col in df.columns if 'signal' in col.lower()]

        if signal_columns:
            for signal_col in signal_columns:
                if signal_col in df.columns:
                    recent_signals = df[df[signal_col] != 0].tail(10)

                    if not recent_signals.empty:
                        st.markdown(f"**{signal_col.replace('_', ' ').title()}**")

                        for idx, row in recent_signals.iterrows():
                            signal_type = "🟢 BUY" if row[signal_col] > 0 else "🔴 SELL"
                            label = idx.strftime('%Y-%m-%d %H:%M') if hasattr(idx, 'strftime') else str(idx)
                            st.markdown(f"""
                            {label}: {signal_type} 
                            at {row['close']:.4f} (Strength: {abs(row[signal_col])})
                            """)

        # Composite signal if available
        if 'composite_signal' in df.columns:
            current_signal = df['composite_signal'].iloc[-1]
            signal_strength = abs(current_signal)
            signal_direction = "🟢 BULLISH" if current_signal > 0 else "🔴 BEARISH" if current_signal < 0 else "🟡 NEUTRAL"

            st.markdown(f"""
            ### Current Market Signal
            **Direction**: {signal_direction}  
            **Strength**: {signal_strength:.2f}  
            **Confidence**: {'High' if signal_strength >= 2 else 'Medium' if signal_strength >= 1 else 'Low'}
            """)

def main():
    """Main application entry point"""
    dashboard = UltimateZANFLOWDashboard()
    dashboard.create_main_dashboard()

if __name__ == "__main__":
    main()
