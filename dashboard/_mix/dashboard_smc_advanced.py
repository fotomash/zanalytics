
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
import glob
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Optional
import gzip

# Page config
st.set_page_config(
    page_title="ZANALYTICS SMC Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for SMC styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .smc-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        border-left: 5px solid #667eea;
        margin: 0.5rem 0;
    }
    .status-bullish { color: #00ff88; font-weight: bold; }
    .status-bearish { color: #ff4757; font-weight: bold; }
    .status-neutral { color: #ffa502; font-weight: bold; }
    .timeframe-selector {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Configuration class
class SMCConfig:
    def __init__(self):
        self.data_dir = "./data"
        self.max_bars = 1000  # Default max bars to load
        self.available_pairs = ["XAUUSD", "BTCUSD", "GBPUSD"]
        self.available_timeframes = ["1T", "5T", "15T", "30T", "1H"]
        self.timeframe_display = {
            "1T": "1 Minute",
            "5T": "5 Minutes", 
            "15T": "15 Minutes",
            "30T": "30 Minutes",
            "1H": "1 Hour"
        }

# Data loader class
class SMCDataLoader:
    def __init__(self, config: SMCConfig):
        self.config = config

    def scan_available_data(self) -> Dict:
        """Scan data directory for available pairs and timeframes"""
        available_data = {}

        # Check both root data dir and latest subfolder
        search_paths = [
            self.config.data_dir,
            os.path.join(self.config.data_dir, "latest")
        ]

        for base_path in search_paths:
            if not os.path.exists(base_path):
                continue

            for pair in self.config.available_pairs:
                pair_path = os.path.join(base_path, pair)
                if os.path.exists(pair_path):
                    if pair not in available_data:
                        available_data[pair] = {
                            "timeframes": [],
                            "analysis_reports": [],
                            "path": pair_path
                        }

                    # Find CSV files for each timeframe
                    for tf in self.config.available_timeframes:
                        csv_pattern = f"{pair}_M1_bars_COMPREHENSIVE_{tf}.csv"
                        csv_file = os.path.join(pair_path, csv_pattern)

                        if os.path.exists(csv_file):
                            file_size = os.path.getsize(csv_file) / (1024*1024)  # MB
                            available_data[pair]["timeframes"].append({
                                "timeframe": tf,
                                "file": csv_file,
                                "size_mb": round(file_size, 2)
                            })

                    # Find analysis reports
                    json_files = glob.glob(os.path.join(pair_path, "*.json"))
                    for json_file in json_files:
                        if "ANALYSIS_REPORT" in json_file or "SUMMARY" in json_file:
                            available_data[pair]["analysis_reports"].append(json_file)

        return available_data

    def load_csv_data(self, file_path: str, max_bars: int = None) -> pd.DataFrame:
        """Load CSV data with bar limit"""
        try:
            # Use max_bars from parameter or config
            limit = max_bars or self.config.max_bars

            # Load with nrows limit for performance
            df = pd.read_csv(file_path, sep=',', nrows=limit)

            # Parse timestamp
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')

            # Ensure we have OHLC columns
            required_cols = ['open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required_cols):
                st.error(f"Missing OHLC columns in {file_path}")
                return None

            # Sort by timestamp (most recent first, then reverse for chronological order)
            df = df.sort_index()

            return df.tail(limit)  # Get most recent N bars

        except Exception as e:
            st.error(f"Error loading {file_path}: {str(e)}")
            return None

    def load_analysis_json(self, file_path: str) -> Dict:
        """Load analysis JSON file"""
        try:
            # Handle gzipped files
            if file_path.endswith('.gz'):
                with gzip.open(file_path, 'rt') as f:
                    return json.load(f)
            else:
                with open(file_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            st.error(f"Error loading analysis {file_path}: {str(e)}")
            return {}

# SMC Chart generator
class SMCChartGenerator:
    def __init__(self):
        self.colors = {
            'bullish': '#00ff88',
            'bearish': '#ff4757',
            'neutral': '#ffa502',
            'order_block': '#9c88ff',
            'liquidity': '#ff6b6b',
            'support': '#4ecdc4',
            'resistance': '#ffe66d'
        }

    def create_smc_candlestick_chart(self, df: pd.DataFrame, analysis_data: Dict, pair: str, timeframe: str) -> go.Figure:
        """Create comprehensive SMC candlestick chart"""

        # Create subplots with secondary y-axis for volume
        fig = make_subplots(
            rows=3, cols=1,
            row_heights=[0.7, 0.15, 0.15],
            subplot_titles=[f"{pair} {timeframe} - Smart Money Concepts", "Volume", "RSI"],
            vertical_spacing=0.05,
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
                name=f"{pair} Price",
                increasing_line_color=self.colors['bullish'],
                decreasing_line_color=self.colors['bearish']
            ),
            row=1, col=1
        )

        # Add SMC elements if available in analysis
        self._add_smc_levels(fig, df, analysis_data, row=1)
        self._add_order_blocks(fig, df, analysis_data, row=1)
        self._add_liquidity_zones(fig, df, analysis_data, row=1)

        # Volume
        if 'volume' in df.columns:
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['volume'],
                    name="Volume",
                    marker_color=self.colors['neutral'],
                    opacity=0.7
                ),
                row=2, col=1
            )

        # RSI (calculate if not available)
        rsi_data = self._calculate_rsi(df['close'])
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=rsi_data,
                name="RSI",
                line=dict(color=self.colors['order_block'], width=2)
            ),
            row=3, col=1
        )

        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=3, col=1)

        # Layout updates
        fig.update_layout(
            title=f"{pair} {timeframe} - SMC Analysis Dashboard",
            height=800,
            showlegend=True,
            xaxis_rangeslider_visible=False,
            template="plotly_white"
        )

        # Update y-axes
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)
        fig.update_xaxes(title_text="Time", row=3, col=1)

        return fig

    def _add_smc_levels(self, fig: go.Figure, df: pd.DataFrame, analysis: Dict, row: int):
        """Add support/resistance levels"""
        if 'smc_levels' in analysis:
            levels = analysis['smc_levels']
            for level in levels.get('support_resistance', []):
                fig.add_hline(
                    y=level.get('price', 0),
                    line_dash="dash",
                    line_color=self.colors['support'] if level.get('type') == 'support' else self.colors['resistance'],
                    annotation_text=f"{level.get('type', 'Level')}: {level.get('price', 0):.2f}",
                    row=row, col=1
                )

    def _add_order_blocks(self, fig: go.Figure, df: pd.DataFrame, analysis: Dict, row: int):
        """Add order block rectangles"""
        if 'order_blocks' in analysis:
            for block in analysis['order_blocks']:
                fig.add_shape(
                    type="rect",
                    x0=block.get('start_time', df.index[0]),
                    x1=block.get('end_time', df.index[-1]),
                    y0=block.get('low', 0),
                    y1=block.get('high', 0),
                    fillcolor=self.colors['order_block'],
                    opacity=0.3,
                    line=dict(color=self.colors['order_block'], width=2),
                    row=row, col=1
                )

    def _add_liquidity_zones(self, fig: go.Figure, df: pd.DataFrame, analysis: Dict, row: int):
        """Add liquidity zones"""
        if 'liquidity_zones' in analysis:
            for zone in analysis['liquidity_zones']:
                fig.add_shape(
                    type="rect",
                    x0=zone.get('start_time', df.index[0]),
                    x1=zone.get('end_time', df.index[-1]),
                    y0=zone.get('low', 0),
                    y1=zone.get('high', 0),
                    fillcolor=self.colors['liquidity'],
                    opacity=0.2,
                    line=dict(color=self.colors['liquidity'], width=1, dash="dot"),
                    row=row, col=1
                )

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

# Main dashboard
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📈 ZANALYTICS SMC INTELLIGENCE</h1>
        <p>Advanced Smart Money Concepts Analysis Dashboard</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize components
    config = SMCConfig()
    loader = SMCDataLoader(config)
    chart_gen = SMCChartGenerator()

    # Sidebar controls
    st.sidebar.title("🎛️ SMC Controls")

    # Settings section
    with st.sidebar.expander("⚙️ Settings", expanded=True):
        max_bars = st.number_input(
            "Max Bars to Load",
            min_value=100,
            max_value=10000,
            value=config.max_bars,
            step=100,
            help="Limit bars for performance"
        )
        config.max_bars = max_bars

        auto_refresh = st.checkbox("Auto Refresh", value=False)
        show_analysis_data = st.checkbox("Show Analysis Data", value=True)

    # Scan available data
    available_data = loader.scan_available_data()

    if not available_data:
        st.error("No data found in ./data directory")
        st.info("Expected structure: ./data/{PAIR}/{PAIR}_M1_bars_COMPREHENSIVE_{TIMEFRAME}.csv")
        return

    # Pair selection
    st.sidebar.markdown("### 💱 Select Currency Pair")
    selected_pair = st.sidebar.selectbox(
        "Choose Pair:",
        options=list(available_data.keys()),
        key="pair_selector"
    )

    if selected_pair:
        pair_data = available_data[selected_pair]

        # Timeframe selection
        st.sidebar.markdown("### ⏰ Select Timeframe")
        available_tfs = [tf['timeframe'] for tf in pair_data['timeframes']]

        if available_tfs:
            selected_tf = st.sidebar.selectbox(
                "Choose Timeframe:",
                options=available_tfs,
                format_func=lambda x: f"{config.timeframe_display.get(x, x)} ({next((tf['size_mb'] for tf in pair_data['timeframes'] if tf['timeframe'] == x), 0):.1f}MB)",
                key="tf_selector"
            )

            # Load selected data
            tf_data = next((tf for tf in pair_data['timeframes'] if tf['timeframe'] == selected_tf), None)

            if tf_data:
                # Load CSV data
                with st.spinner(f"Loading {selected_pair} {selected_tf} data..."):
                    df = loader.load_csv_data(tf_data['file'], max_bars)

                if df is not None:
                    # Load analysis data
                    analysis_data = {}
                    if pair_data['analysis_reports']:
                        analysis_file = pair_data['analysis_reports'][0]  # Use first available
                        analysis_data = loader.load_analysis_json(analysis_file)

                    # Main metrics
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        current_price = df['close'].iloc[-1] if len(df) > 0 else 0
                        price_change = df['close'].iloc[-1] - df['close'].iloc[-2] if len(df) > 1 else 0
                        st.metric("Current Price", f"{current_price:.4f}", f"{price_change:+.4f}")

                    with col2:
                        high_24h = df['high'].max()
                        low_24h = df['low'].min()
                        st.metric("24H Range", f"{high_24h:.4f}", f"Low: {low_24h:.4f}")

                    with col3:
                        volatility = df['close'].pct_change().std() * 100
                        st.metric("Volatility", f"{volatility:.2f}%")

                    with col4:
                        total_bars = len(df)
                        st.metric("Bars Loaded", f"{total_bars:,}", f"of {max_bars:,} max")

                    # SMC Chart
                    st.subheader(f"📊 {selected_pair} {config.timeframe_display.get(selected_tf, selected_tf)} SMC Chart")

                    chart = chart_gen.create_smc_candlestick_chart(df, analysis_data, selected_pair, selected_tf)
                    st.plotly_chart(chart, use_container_width=True)

                    # Analysis data display
                    if show_analysis_data and analysis_data:
                        col1, col2 = st.columns(2)

                        with col1:
                            st.subheader("🎯 SMC Analysis")
                            if 'smc_analysis' in analysis_data:
                                smc = analysis_data['smc_analysis']

                                # Market structure
                                if 'market_structure' in smc:
                                    structure = smc['market_structure']
                                    trend = structure.get('trend', 'NEUTRAL')
                                    trend_class = f"status-{trend.lower()}"
                                    st.markdown(f"**Market Structure:** <span class='{trend_class}'>{trend}</span>", unsafe_allow_html=True)

                                # Order blocks
                                if 'order_blocks' in smc:
                                    st.markdown(f"**Order Blocks:** {len(smc['order_blocks'])} detected")

                                # Liquidity zones
                                if 'liquidity_zones' in smc:
                                    st.markdown(f"**Liquidity Zones:** {len(smc['liquidity_zones'])} active")

                        with col2:
                            st.subheader("📈 Technical Indicators")
                            if 'technical_indicators' in analysis_data:
                                indicators = analysis_data['technical_indicators']

                                for indicator, value in indicators.items():
                                    if isinstance(value, (int, float)):
                                        st.metric(indicator.upper(), f"{value:.2f}")
                                    else:
                                        st.text(f"{indicator.upper()}: {value}")

                    # Raw data preview
                    with st.expander("📋 Raw Data Preview"):
                        st.dataframe(df.tail(20), use_container_width=True)

                    # Analysis JSON preview
                    if analysis_data and show_analysis_data:
                        with st.expander("🔍 Analysis Data"):
                            st.json(analysis_data)

        else:
            st.warning(f"No timeframe data found for {selected_pair}")

    # Footer info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Data Info")
    st.sidebar.info(f"Found {len(available_data)} pairs with comprehensive analysis data")

    for pair, data in available_data.items():
        st.sidebar.markdown(f"**{pair}:** {len(data['timeframes'])} timeframes")

if __name__ == "__main__":
    main()
