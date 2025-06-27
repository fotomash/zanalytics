
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime, timedelta
import warnings
import time
import glob
from typing import Optional

warnings.filterwarnings('ignore')

# Mapping of possible SMC column variants to unified keys
SMC_VARIANT_MAP = {
    'fvg_bullish': ['SMC_fvg_bullish', 'bullish_fvg', 'fvg_up'],
    'fvg_bearish': ['SMC_fvg_bearish', 'bearish_fvg', 'fvg_down'],
    'ob_bullish': ['SMC_bullish_ob', 'bullish_order_block', 'bullish_ob', 'ob_up'],
    'ob_bearish': ['SMC_bearish_ob', 'bearish_order_block', 'bearish_ob', 'ob_down'],
    'structure_break': ['structure_break', 'SMC_structure_break', 'bos']
}


def find_column(df: pd.DataFrame, variants: list) -> Optional[str]:
    """Return the first matching column from variants if present."""
    for col in variants:
        if col in df.columns:
            return col
    return None


def count_events(df: pd.DataFrame, col: str) -> int:
    """Count truthy values in a column, handling boolean and numeric types."""
    if col not in df.columns:
        return 0
    if df[col].dtype == bool:
        return int(df[col].sum())
    return int((df[col] != 0).sum())

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .smc-signal {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        color: white;
        font-weight: bold;
        margin: 0.25rem;
    }

    .wyckoff-phase {
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        color: white;
        font-weight: bold;
        margin: 0.25rem;
    }

    .wyckoff-phase.accumulation {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
    }

    .wyckoff-phase.distribution {
        background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%);
    }

    .wyckoff-phase.markup {
        background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%);
    }

    .risk-metric {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        padding: 0.8rem;
        border-radius: 6px;
        text-align: center;
        color: white;
        font-size: 0.9rem;
        margin: 0.2rem;
    }

    .stAlert > div {
        background-color: rgba(28, 131, 225, 0.1);
        border: 1px solid rgba(28, 131, 225, 0.2);
    }
</style>
""", unsafe_allow_html=True)

class UltimateTradingDashboard:
    def __init__(self):
        self.data_folder = "./data"
        self.tick_data = {}
        self.bar_data = {}

        # Initialize session state
        if 'last_update' not in st.session_state:
            st.session_state.last_update = datetime.now()
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = True
        if 'chart_theme' not in st.session_state:
            st.session_state.chart_theme = 'plotly_dark'
        if 'lookback_bars' not in st.session_state:
            st.session_state.lookback_bars = 500
        if 'selected_pair' not in st.session_state:
            st.session_state.selected_pair = None
        if 'show_liquidity_analysis' not in st.session_state:
            st.session_state.show_liquidity_analysis = True

    def load_data_sources(self):
        """Load all available data sources from the data folder"""
        data_sources = {}

        if not os.path.exists(self.data_folder):
            st.warning(f"Data folder '{self.data_folder}' not found. Please create it and add your data files.")
            return data_sources

        # Look for CSV and Parquet files
        file_patterns = ['*.csv', '*.parquet', '*.json']

        for pattern in file_patterns:
            files = glob.glob(os.path.join(self.data_folder, pattern))

            for file_path in files:
                try:
                    file_name = os.path.basename(file_path)
                    base_name = os.path.splitext(file_name)[0]

                    # Extract trading pair from filename (assuming format like EURUSD_H1.csv)
                    parts = base_name.split('_')
                    if len(parts) >= 2:
                        pair = parts[0]
                        timeframe = parts[1] if len(parts) > 1 else 'Unknown'
                    else:
                        pair = base_name
                        timeframe = 'Unknown'

                    # Load data based on file type
                    if file_path.endswith('.csv'):
                        df = pd.read_csv(file_path)
                    elif file_path.endswith('.parquet'):
                        df = pd.read_parquet(file_path)
                    elif file_path.endswith('.json'):
                        with open(file_path, 'r') as f:
                            json_data = json.load(f)
                        df = pd.DataFrame(json_data)

                    # Ensure datetime index
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df.set_index('timestamp', inplace=True)
                    elif 'datetime' in df.columns:
                        df['datetime'] = pd.to_datetime(df['datetime'])
                        df.set_index('datetime', inplace=True)
                    elif not isinstance(df.index, pd.DatetimeIndex):
                        try:
                            df.index = pd.to_datetime(df.index)
                        except:
                            df.reset_index(inplace=True)
                            df.index = pd.date_range(start='2023-01-01', periods=len(df), freq='1H')

                    # Store data
                    if pair not in data_sources:
                        data_sources[pair] = {}

                    # Determine if it's tick data or bar data
                    if 'bid' in df.columns and 'ask' in df.columns:
                        # Tick data
                        self.tick_data[pair] = df
                        data_sources[pair]['tick'] = df
                    else:
                        # Bar data
                        if pair not in self.bar_data:
                            self.bar_data[pair] = {}
                        self.bar_data[pair][timeframe] = df
                        data_sources[pair][timeframe] = df

                except Exception as e:
                    st.error(f"Error loading {file_path}: {str(e)}")

        return data_sources

    def create_main_dashboard(self):
        """Create the main dashboard interface"""
        st.title("🔥 ZANFLOW Ultimate Trading Analytics Dashboard")

        # Sidebar configuration
        with st.sidebar:
            st.markdown("## ⚙️ Dashboard Configuration")

            # Auto-refresh toggle
            st.session_state.auto_refresh = st.toggle("🔄 Auto Refresh", value=st.session_state.auto_refresh)

            # Chart theme
            st.session_state.chart_theme = st.selectbox(
                "🎨 Chart Theme",
                ['plotly_dark', 'plotly_white', 'seaborn', 'ggplot2'],
                index=0
            )

            # Lookback bars
            st.session_state.lookback_bars = st.slider(
                "📊 Lookback Bars",
                min_value=100,
                max_value=2000,
                value=st.session_state.lookback_bars,
                step=50
            )

            # Analysis options
            st.markdown("### 🔍 Analysis Options")
            st.session_state.show_liquidity_analysis = st.checkbox(
                "💧 Show Liquidity Analysis", 
                value=st.session_state.show_liquidity_analysis
            )

            # Manual refresh button
            if st.button("🔄 Refresh Data"):
                st.rerun()

        # Load data sources
        data_sources = self.load_data_sources()

        if not data_sources:
            st.warning("No data sources found. Please add CSV, Parquet, or JSON files to the ./data folder.")
            st.markdown("""
            ### 📁 Expected Data Structure

            Place your data files in the `./data` folder with these naming conventions:
            - `SYMBOL_TIMEFRAME.csv` (e.g., EURUSD_H1.csv, GBPUSD_D1.csv)
            - Files should contain OHLCV data with a timestamp column
            - For tick data, include 'bid' and 'ask' columns
            """)
            return

        # Trading pair selection
        available_pairs = list(data_sources.keys())
        if not st.session_state.selected_pair or st.session_state.selected_pair not in available_pairs:
            st.session_state.selected_pair = available_pairs[0]

        selected_pair = st.selectbox(
            "📈 Select Trading Pair",
            available_pairs,
            index=available_pairs.index(st.session_state.selected_pair),
            key="pair_selector"
        )
        st.session_state.selected_pair = selected_pair

        # Update timestamp
        st.session_state.last_update = datetime.now()
        st.info(f"📅 Last Update: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}")

        # Main analysis tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Overview", 
            "🧠 SMC Analysis", 
            "📈 Wyckoff Analysis",
            "🔍 Microstructure", 
            "⏱️ Multi-Timeframe",
            "⚠️ Risk Analytics"
        ])

        with tab1:
            self.create_overview_dashboard(selected_pair, data_sources)

        with tab2:
            self.display_advanced_smc_analysis(selected_pair)

        with tab3:
            self.display_wyckoff_analysis(selected_pair)

        with tab4:
            self.create_microstructure_analysis(selected_pair, data_sources)

        with tab5:
            self.display_multi_timeframe_analysis(selected_pair)

        with tab6:
            self.create_risk_analytics(selected_pair, data_sources)

    def create_overview_dashboard(self, selected_pair, data_sources):
        """Create overview dashboard"""
        st.markdown("## 📊 Market Overview")

        if selected_pair not in data_sources:
            st.error("No data available for selected pair")
            return

        pair_data = data_sources[selected_pair]

        # Market status indicators
        col1, col2, col3, col4 = st.columns(4)

        # Get latest bar data for metrics
        latest_data = None
        for timeframe, df in pair_data.items():
            if timeframe != 'tick' and len(df) > 0:
                latest_data = df.iloc[-1]
                break

        if latest_data is not None:
            with col1:
                current_price = latest_data.get('close', 0)
                st.markdown(f'<div class="metric-card"><h3>Current Price</h3><h2>{current_price:.5f}</h2></div>', unsafe_allow_html=True)

            with col2:
                daily_change = latest_data.get('close', 0) - latest_data.get('open', 0)
                change_pct = (daily_change / latest_data.get('open', 1)) * 100 if latest_data.get('open', 0) != 0 else 0
                color = "🟢" if daily_change >= 0 else "🔴"
                st.markdown(f'<div class="metric-card"><h3>Daily Change</h3><h2>{color} {change_pct:.2f}%</h2></div>', unsafe_allow_html=True)

            with col3:
                volume = latest_data.get('volume', 0)
                st.markdown(f'<div class="metric-card"><h3>Volume</h3><h2>{volume:.0f}</h2></div>', unsafe_allow_html=True)

            with col4:
                volatility = latest_data.get('ATR_14', 0)
                st.markdown(f'<div class="metric-card"><h3>Volatility (ATR)</h3><h2>{volatility:.5f}</h2></div>', unsafe_allow_html=True)

        # Multi-timeframe chart
        self.create_multi_timeframe_chart(selected_pair, pair_data)

        # Key market signals
        self.create_market_signals_overview(selected_pair, pair_data)

    def create_multi_timeframe_chart(self, pair, data):
        """Create multi-timeframe price chart"""
        st.markdown("### 📈 Multi-Timeframe Price Action")

        timeframes = [tf for tf in data.keys() if tf != 'tick']
        if not timeframes:
            st.warning("No timeframe data available")
            return

        # Select primary timeframe for main chart
        primary_tf = st.selectbox("Select Primary Timeframe", timeframes, key="overview_tf")

        if primary_tf not in data:
            return

        df = data[primary_tf]
        lookback = st.session_state.get('lookback_bars', 500)
        df_display = df.tail(lookback)

        # Create candlestick chart
        fig = go.Figure()

        fig.add_trace(go.Candlestick(
            x=df_display.index,
            open=df_display['open'],
            high=df_display['high'],
            low=df_display['low'],
            close=df_display['close'],
            name=f"{pair} {primary_tf}",
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4444'
        ))

        # Add moving averages if available
        if 'SMA_20' in df_display.columns:
            fig.add_trace(go.Scatter(
                x=df_display.index,
                y=df_display['SMA_20'],
                mode='lines',
                name='SMA 20',
                line=dict(color='orange', width=2)
            ))

        if 'SMA_50' in df_display.columns:
            fig.add_trace(go.Scatter(
                x=df_display.index,
                y=df_display['SMA_50'],
                mode='lines',
                name='SMA 50',
                line=dict(color='blue', width=2)
            ))

        # Add SMC and Wyckoff overlays
        self.add_smc_overlays(fig, df_display)
        self.add_wyckoff_overlays(fig, df_display)

        fig.update_layout(
            title=f"{pair} {primary_tf} - Comprehensive Analysis",
            template=st.session_state.get('chart_theme', 'plotly_dark'),
            height=600,
            xaxis_rangeslider_visible=False
        )

        st.plotly_chart(fig, use_container_width=True)

    def add_smc_overlays(self, fig, df):
        """Add Smart Money Concepts overlays"""
        # Fair Value Gaps
        b_fvg_col = find_column(df, SMC_VARIANT_MAP['fvg_bullish'])
        if b_fvg_col:
            fvg_bullish = df[df[b_fvg_col] == True]
            if not fvg_bullish.empty:
                fig.add_trace(go.Scatter(
                    x=fvg_bullish.index, y=fvg_bullish['low'],
                    mode='markers', name='Bullish FVG',
                    marker=dict(symbol='triangle-up', color='lime', size=12),
                    showlegend=True
                ))

        s_fvg_col = find_column(df, SMC_VARIANT_MAP['fvg_bearish'])
        if s_fvg_col:
            fvg_bearish = df[df[s_fvg_col] == True]
            if not fvg_bearish.empty:
                fig.add_trace(go.Scatter(
                    x=fvg_bearish.index, y=fvg_bearish['high'],
                    mode='markers', name='Bearish FVG',
                    marker=dict(symbol='triangle-down', color='red', size=12),
                    showlegend=True
                ))

        # Order Blocks
        ob_bull_col = find_column(df, SMC_VARIANT_MAP['ob_bullish'])
        if ob_bull_col:
            ob_bullish = df[df[ob_bull_col] == True]
            if not ob_bullish.empty:
                fig.add_trace(go.Scatter(
                    x=ob_bullish.index, y=ob_bullish['low'],
                    mode='markers', name='Bullish OB',
                    marker=dict(symbol='square', color='lightgreen', size=10),
                    showlegend=True
                ))

        ob_bear_col = find_column(df, SMC_VARIANT_MAP['ob_bearish'])
        if ob_bear_col:
            ob_bearish = df[df[ob_bear_col] == True]
            if not ob_bearish.empty:
                fig.add_trace(go.Scatter(
                    x=ob_bearish.index, y=ob_bearish['high'],
                    mode='markers', name='Bearish OB',
                    marker=dict(symbol='square', color='lightcoral', size=10),
                    showlegend=True
                ))

    def add_wyckoff_overlays(self, fig, df):
        """Add Wyckoff analysis overlays"""
        # Wyckoff phases
        if 'wyckoff_accumulation' in df.columns:
            acc_data = df[df['wyckoff_accumulation'] == True]
            if not acc_data.empty:
                fig.add_trace(go.Scatter(
                    x=acc_data.index, y=acc_data['close'],
                    mode='markers', name='Wyckoff Accumulation',
                    marker=dict(color='green', size=8, opacity=0.7),
                    showlegend=True
                ))

        if 'wyckoff_distribution' in df.columns:
            dist_data = df[df['wyckoff_distribution'] == True]
            if not dist_data.empty:
                fig.add_trace(go.Scatter(
                    x=dist_data.index, y=dist_data['close'],
                    mode='markers', name='Wyckoff Distribution',
                    marker=dict(color='red', size=8, opacity=0.7),
                    showlegend=True
                ))

    def create_market_signals_overview(self, pair, data):
        """Create market signals overview"""
        st.markdown("### 🎯 Key Market Signals")

        # Get the most recent data across timeframes
        signals_data = []

        for timeframe, df in data.items():
            if timeframe == 'tick' or len(df) == 0:
                continue

            latest = df.iloc[-1]

            # SMC signals
            b_fvg_col = find_column(df, SMC_VARIANT_MAP['fvg_bullish'])
            s_fvg_col = find_column(df, SMC_VARIANT_MAP['fvg_bearish'])
            ob_bull_col = find_column(df, SMC_VARIANT_MAP['ob_bullish'])
            ob_bear_col = find_column(df, SMC_VARIANT_MAP['ob_bearish'])

            bullish_fvg = latest.get(b_fvg_col, False) if b_fvg_col else False
            bearish_fvg = latest.get(s_fvg_col, False) if s_fvg_col else False
            bullish_ob = latest.get(ob_bull_col, False) if ob_bull_col else False
            bearish_ob = latest.get(ob_bear_col, False) if ob_bear_col else False

            # Wyckoff signals
            wyckoff_acc = latest.get('wyckoff_accumulation', False)
            wyckoff_dist = latest.get('wyckoff_distribution', False)

            # Technical indicators
            rsi = latest.get('RSI_14', np.nan)
            macd = latest.get('MACD_12_26_9', np.nan)

            signals_data.append({
                'Timeframe': timeframe,
                'SMC Bullish': '🟢' if bullish_fvg or bullish_ob else '',
                'SMC Bearish': '🔴' if bearish_fvg or bearish_ob else '',
                'Wyckoff': '🟢 ACC' if wyckoff_acc else '🔴 DIST' if wyckoff_dist else '',
                'RSI': f"{rsi:.1f}" if not np.isnan(rsi) else "N/A",
                'MACD': '🟢' if macd > 0 else '🔴' if macd < 0 else '' if not np.isnan(macd) else 'N/A'
            })

        if signals_data:
            signals_df = pd.DataFrame(signals_data)
            st.dataframe(signals_df, use_container_width=True)

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
            b_col = find_column(df, SMC_VARIANT_MAP['fvg_bullish'])
            s_col = find_column(df, SMC_VARIANT_MAP['fvg_bearish'])
            bullish_fvgs = count_events(df, b_col) if b_col else 0
            bearish_fvgs = count_events(df, s_col) if s_col else 0
            st.markdown(f'<div class="smc-signal"><strong>Fair Value Gaps</strong><br>🟢 {bullish_fvgs} | 🔴 {bearish_fvgs}</div>', unsafe_allow_html=True)

        with col2:
            ob_bull_col = find_column(df, SMC_VARIANT_MAP['ob_bullish'])
            ob_bear_col = find_column(df, SMC_VARIANT_MAP['ob_bearish'])
            bullish_obs = count_events(df, ob_bull_col) if ob_bull_col else 0
            bearish_obs = count_events(df, ob_bear_col) if ob_bear_col else 0
            st.markdown(f'<div class="smc-signal"><strong>Order Blocks</strong><br>🟢 {bullish_obs} | 🔴 {bearish_obs}</div>', unsafe_allow_html=True)

        with col3:
            liquidity_grabs = df['SMC_liquidity_grab'].sum() if 'SMC_liquidity_grab' in df.columns else 0
            st.markdown(f'<div class="smc-signal"><strong>Liquidity Grabs</strong><br>⚡ {liquidity_grabs}</div>', unsafe_allow_html=True)

        with col4:
            bos_bull_col = find_column(df, SMC_VARIANT_MAP.get('bos_bullish', ['SMC_bos_bullish']))
            bos_bear_col = find_column(df, SMC_VARIANT_MAP.get('bos_bearish', ['SMC_bos_bearish']))
            bos_bullish = count_events(df, bos_bull_col) if bos_bull_col else 0
            bos_bearish = count_events(df, bos_bear_col) if bos_bear_col else 0
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
            name="Price",
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4444'
        ), row=1, col=1)

        # Wyckoff phases
        if 'wyckoff_accumulation' in df.columns:
            acc_data = df[df['wyckoff_accumulation']]
            if not acc_data.empty:
                fig.add_trace(go.Scatter(
                    x=acc_data.index,
                    y=acc_data['close'],
                    mode='markers',
                    name='Accumulation',
                    marker=dict(color='green', size=10, symbol='circle')
                ), row=1, col=1)

        if 'wyckoff_distribution' in df.columns:
            dist_data = df[df['wyckoff_distribution']]
            if not dist_data.empty:
                fig.add_trace(go.Scatter(
                    x=dist_data.index,
                    y=dist_data['close'],
                    mode='markers',
                    name='Distribution',
                    marker=dict(color='red', size=10, symbol='circle')
                ), row=1, col=1)

        # Volume analysis
        if 'volume' in df.columns:
            colors = ['green' if close >= open_val else 'red' 
                     for close, open_val in zip(df['close'], df['open'])]

            fig.add_trace(go.Bar(
                x=df.index,
                y=df['volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7
            ), row=2, col=1)

        fig.update_layout(
            template=st.session_state.get('chart_theme', 'plotly_dark'),
            height=800,
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)

    def create_volume_spread_analysis(self, df):
        """Create volume spread analysis"""
        st.markdown("### 📊 Volume Spread Analysis")

        if 'volume' not in df.columns:
            st.warning("Volume data not available")
            return

        col1, col2 = st.columns(2)

        with col1:
            # Volume profile
            st.markdown("#### 📈 Volume Profile")

            # Calculate volume by price levels
            price_levels = np.linspace(df['low'].min(), df['high'].max(), 50)
            volume_profile = []

            for i in range(len(price_levels) - 1):
                level_volume = df[
                    (df['close'] >= price_levels[i]) & 
                    (df['close'] < price_levels[i + 1])
                ]['volume'].sum()
                volume_profile.append(level_volume)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=volume_profile,
                y=price_levels[:-1],
                orientation='h',
                name='Volume Profile',
                marker_color='blue',
                opacity=0.7
            ))

            fig.update_layout(
                title="Volume by Price Level",
                template=st.session_state.get('chart_theme', 'plotly_dark'),
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Volume analysis metrics
            st.markdown("#### 📊 Volume Metrics")

            avg_volume = df['volume'].mean()
            current_volume = df['volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0

            high_volume_days = len(df[df['volume'] > avg_volume * 1.5])
            low_volume_days = len(df[df['volume'] < avg_volume * 0.5])

            st.markdown(f"""
            **Average Volume**: {avg_volume:.0f}  
            **Current Volume**: {current_volume:.0f}  
            **Volume Ratio**: {volume_ratio:.2f}x  
            **High Volume Days**: {high_volume_days}  
            **Low Volume Days**: {low_volume_days}  
            """)

    def create_microstructure_analysis(self, symbol, data_sources):
        """Create microstructure analysis"""
        st.markdown("## 🔍 Microstructure Analysis")

        if symbol not in data_sources:
            st.warning("No data available for microstructure analysis")
            return

        pair_data = data_sources[symbol]

        # Check for tick data
        if 'tick' in pair_data:
            tick_df = pair_data['tick']
            self.create_tick_analysis(tick_df)
        else:
            st.info("No tick data available. Showing bar-based microstructure analysis.")
            # Use the finest timeframe available
            timeframes = [tf for tf in pair_data.keys() if tf != 'tick']
            if timeframes:
                finest_tf = timeframes[0]  # Assume first is finest
                df = pair_data[finest_tf]
                self.create_bar_microstructure_analysis(df)

    def create_tick_analysis(self, tick_df):
        """Create tick-level analysis"""
        st.markdown("### ⚡ Tick-Level Analysis")

        # Bid-Ask spread analysis
        if 'bid' in tick_df.columns and 'ask' in tick_df.columns:
            col1, col2, col3, col4 = st.columns(4)

            spread = tick_df['ask'] - tick_df['bid']

            with col1:
                st.metric("Avg Spread", f"{spread.mean():.5f}")
            with col2:
                st.metric("Spread Volatility", f"{spread.std():.5f}")
            with col3:
                st.metric("Max Spread", f"{spread.max():.5f}")
            with col4:
                st.metric("Min Spread", f"{spread.min():.5f}")

            # Spread chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=tick_df.index[-1000:],  # Last 1000 ticks
                y=spread.iloc[-1000:],
                mode='lines',
                name='Bid-Ask Spread',
                line=dict(color='orange', width=1)
            ))

            fig.update_layout(
                title="Bid-Ask Spread Evolution",
                template=st.session_state.get('chart_theme', 'plotly_dark'),
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

    def create_bar_microstructure_analysis(self, df):
        """Create bar-based microstructure analysis"""
        st.markdown("### 📊 Bar-Level Microstructure")

        # Price impact analysis
        if len(df) > 1:
            price_changes = df['close'].pct_change().dropna()

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 📈 Price Change Distribution")
                fig = px.histogram(
                    price_changes,
                    nbins=50,
                    title="Price Change Distribution",
                    template=st.session_state.get('chart_theme', 'plotly_dark')
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("#### 📊 Volatility Clustering")
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
                    b_fvg_col = find_column(df, SMC_VARIANT_MAP['fvg_bullish'])
                    s_fvg_col = find_column(df, SMC_VARIANT_MAP['fvg_bearish'])
                    ob_bull_col = find_column(df, SMC_VARIANT_MAP['ob_bullish'])
                    ob_bear_col = find_column(df, SMC_VARIANT_MAP['ob_bearish'])

                    bullish_fvg = latest.get(b_fvg_col, False) if b_fvg_col else False
                    bearish_fvg = latest.get(s_fvg_col, False) if s_fvg_col else False
                    bullish_ob = latest.get(ob_bull_col, False) if ob_bull_col else False
                    bearish_ob = latest.get(ob_bear_col, False) if ob_bear_col else False

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

    def create_risk_analytics(self, symbol, data_sources):
        """Create risk analytics dashboard"""
        st.markdown("## ⚠️ Risk Analytics")

        if symbol not in data_sources:
            st.warning("No data available for risk analysis")
            return

        pair_data = data_sources[symbol]

        # Get the primary timeframe data
        timeframes = [tf for tf in pair_data.keys() if tf != 'tick']
        if not timeframes:
            st.warning("No timeframe data available")
            return

        primary_tf = timeframes[0]
        df = pair_data[primary_tf]

        # Risk metrics
        self.create_risk_metrics(df, symbol)

        # VaR analysis
        self.create_var_analysis(df)

        # Correlation analysis if multiple pairs
        self.create_correlation_analysis(data_sources)

    def create_risk_metrics(self, df, symbol):
        """Create risk metrics"""
        st.markdown("### 📊 Risk Metrics")

        if len(df) < 20:
            st.warning("Insufficient data for risk analysis")
            return

        # Calculate returns
        returns = df['close'].pct_change().dropna()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            volatility = returns.std() * np.sqrt(252)  # Annualized
            st.markdown(f'<div class="risk-metric">Annual Volatility<br>{volatility:.2%}</div>', unsafe_allow_html=True)

        with col2:
            max_drawdown = self.calculate_max_drawdown(df['close'])
            st.markdown(f'<div class="risk-metric">Max Drawdown<br>{max_drawdown:.2%}</div>', unsafe_allow_html=True)

        with col3:
            sharpe_ratio = self.calculate_sharpe_ratio(returns)
            st.markdown(f'<div class="risk-metric">Sharpe Ratio<br>{sharpe_ratio:.2f}</div>', unsafe_allow_html=True)

        with col4:
            var_95 = np.percentile(returns, 5)
            st.markdown(f'<div class="risk-metric">VaR (95%)<br>{var_95:.2%}</div>', unsafe_allow_html=True)

    def calculate_max_drawdown(self, prices):
        """Calculate maximum drawdown"""
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak
        return drawdown.min()

    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.02):
        """Calculate Sharpe ratio"""
        excess_returns = returns.mean() * 252 - risk_free_rate
        volatility = returns.std() * np.sqrt(252)
        return excess_returns / volatility if volatility != 0 else 0

    def create_var_analysis(self, df):
        """Create VaR analysis"""
        st.markdown("### 📉 Value at Risk Analysis")

        returns = df['close'].pct_change().dropna()

        col1, col2 = st.columns(2)

        with col1:
            # VaR at different confidence levels
            var_levels = [90, 95, 99]
            var_data = []

            for level in var_levels:
                var_value = np.percentile(returns, 100 - level)
                var_data.append({
                    'Confidence Level': f"{level}%",
                    'VaR': f"{var_value:.4f}",
                    'VaR (%)': f"{var_value:.2%}"
                })

            var_df = pd.DataFrame(var_data)
            st.dataframe(var_df, use_container_width=True)

        with col2:
            # Returns distribution
            fig = px.histogram(
                returns,
                nbins=50,
                title="Returns Distribution",
                template=st.session_state.get('chart_theme', 'plotly_dark')
            )

            # Add VaR lines
            var_95 = np.percentile(returns, 5)
            fig.add_vline(x=var_95, line_dash="dash", line_color="red", 
                         annotation_text="VaR 95%")

            st.plotly_chart(fig, use_container_width=True)

    def create_correlation_analysis(self, data_sources):
        """Create correlation analysis"""
        st.markdown("### 🔗 Correlation Analysis")

        # Only show if multiple pairs available
        pairs = list(data_sources.keys())
        if len(pairs) < 2:
            st.info("Correlation analysis requires multiple trading pairs")
            return

        # Calculate correlation matrix
        price_data = {}

        for pair in pairs:
            pair_data = data_sources[pair]
            timeframes = [tf for tf in pair_data.keys() if tf != 'tick']
            if timeframes:
                df = pair_data[timeframes[0]]
                if len(df) > 0:
                    price_data[pair] = df['close']

        if len(price_data) >= 2:
            # Align data
            combined_df = pd.DataFrame(price_data)
            combined_df = combined_df.dropna()

            # Calculate returns
            returns_df = combined_df.pct_change().dropna()

            # Correlation matrix
            corr_matrix = returns_df.corr()

            # Plot correlation heatmap
            fig = px.imshow(
                corr_matrix,
                title="Returns Correlation Matrix",
                template=st.session_state.get('chart_theme', 'plotly_dark'),
                color_continuous_scale='RdBu',
                zmin=-1, zmax=1
            )

            st.plotly_chart(fig, use_container_width=True)

def main():
    st.set_page_config(
        page_title="ZANFLOW Ultimate Trading Dashboard",
        page_icon="🔥",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Auto-refresh functionality
    if st.session_state.get('auto_refresh', True):
        time.sleep(60)  # Wait 60 seconds
        st.rerun()

    dashboard = UltimateTradingDashboard()
    dashboard.create_main_dashboard()

if __name__ == "__main__":
    main()
