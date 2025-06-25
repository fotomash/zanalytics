
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

# Custom CSS for enhanced SMC styling
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
    .wyckoff-card {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 1rem;
        border-radius: 10px;
        color: #2c3e50;
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
    .analysis-highlight {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Configuration class
class SMCConfig:
    def __init__(self):
        self.data_dir = "./data"
        self.max_bars = 1000
        self.available_pairs = ["XAUUSD", "BTCUSD", "GBPUSD"]
        self.available_timeframes = ["1T", "5T", "15T", "30T", "1H"]
        self.timeframe_display = {
            "1T": "1 Minute",
            "5T": "5 Minutes", 
            "15T": "15 Minutes",
            "30T": "30 Minutes",
            "1H": "1 Hour"
        }

# Enhanced Data loader with SMC/Wyckoff parsing
class EnhancedSMCDataLoader:
    def __init__(self, config: SMCConfig):
        self.config = config

    def scan_available_data(self) -> Dict:
        """Scan data directory for available pairs and timeframes"""
        available_data = {}

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
                            "summary_files": [],
                            "path": pair_path
                        }

                    # Find CSV files for each timeframe
                    for tf in self.config.available_timeframes:
                        csv_pattern = f"{pair}_M1_bars_COMPREHENSIVE_{tf}.csv"
                        csv_file = os.path.join(pair_path, csv_pattern)

                        if os.path.exists(csv_file):
                            file_size = os.path.getsize(csv_file) / (1024*1024)
                            available_data[pair]["timeframes"].append({
                                "timeframe": tf,
                                "file": csv_file,
                                "size_mb": round(file_size, 2),
                                "summary_file": os.path.join(pair_path, f"{pair}_M1_bars_SUMMARY_{tf}.json")
                            })

                    # Find analysis reports
                    analysis_report = os.path.join(pair_path, f"{pair}_M1_bars_ANALYSIS_REPORT.json")
                    if os.path.exists(analysis_report):
                        available_data[pair]["analysis_reports"].append(analysis_report)

                    # Find summary files
                    summary_files = glob.glob(os.path.join(pair_path, "*SUMMARY*.json"))
                    available_data[pair]["summary_files"].extend(summary_files)

        return available_data

    def load_csv_data(self, file_path: str, max_bars: int = None) -> pd.DataFrame:
        """Load CSV data with latest bars first"""
        try:
            limit = max_bars or self.config.max_bars

            # Read the entire file first to get proper sorting
            df = pd.read_csv(file_path, sep=',')

            # Parse timestamp
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')

            # Ensure OHLC columns
            required_cols = ['open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required_cols):
                st.error(f"Missing OHLC columns in {file_path}")
                return None

            # Sort by timestamp and get LATEST bars (most recent)
            df = df.sort_index()
            df = df.tail(limit)  # Get the most recent N bars

            return df

        except Exception as e:
            st.error(f"Error loading {file_path}: {str(e)}")
            return None

    def load_analysis_json(self, file_path: str) -> Dict:
        """Load analysis JSON file with enhanced parsing"""
        try:
            if file_path.endswith('.gz'):
                with gzip.open(file_path, 'rt') as f:
                    data = json.load(f)
            else:
                with open(file_path, 'r') as f:
                    data = json.load(f)

            return data
        except Exception as e:
            st.error(f"Error loading analysis {file_path}: {str(e)}")
            return {}

    def parse_smc_data(self, analysis_data: Dict) -> Dict:
        """Parse SMC-specific data from analysis"""
        smc_data = {
            'order_blocks': [],
            'liquidity_zones': [],
            'bos_points': [],
            'choch_points': [],
            'supply_demand_zones': [],
            'fair_value_gaps': [],
            'market_structure': {},
            'trend_analysis': {}
        }

        # Extract SMC data from various sections
        if 'smc_analysis' in analysis_data:
            smc_analysis = analysis_data['smc_analysis']

            # Order blocks
            if 'order_blocks' in smc_analysis:
                smc_data['order_blocks'] = smc_analysis['order_blocks']

            # Market structure
            if 'market_structure' in smc_analysis:
                smc_data['market_structure'] = smc_analysis['market_structure']

            # Liquidity analysis
            if 'liquidity_analysis' in smc_analysis:
                smc_data['liquidity_zones'] = smc_analysis['liquidity_analysis'].get('zones', [])

        # Extract from microstructure analysis
        if 'microstructure_analysis' in analysis_data:
            micro = analysis_data['microstructure_analysis']

            # Supply/Demand zones
            if 'supply_demand_zones' in micro:
                smc_data['supply_demand_zones'] = micro['supply_demand_zones']

            # Fair Value Gaps
            if 'fair_value_gaps' in micro:
                smc_data['fair_value_gaps'] = micro['fair_value_gaps']

        # Extract BOS/CHoCH from pattern analysis
        if 'pattern_analysis' in analysis_data:
            patterns = analysis_data['pattern_analysis']

            # Break of Structure points
            if 'bos_points' in patterns:
                smc_data['bos_points'] = patterns['bos_points']

            # Change of Character points  
            if 'choch_points' in patterns:
                smc_data['choch_points'] = patterns['choch_points']

        return smc_data

    def parse_wyckoff_data(self, analysis_data: Dict) -> Dict:
        """Parse Wyckoff-specific data from analysis"""
        wyckoff_data = {
            'phases': [],
            'accumulation_zones': [],
            'distribution_zones': [],
            'markup_phases': [],
            'markdown_phases': [],
            'spring_actions': [],
            'upthrusts': [],
            'current_phase': 'Unknown'
        }

        # Extract Wyckoff data
        if 'wyckoff_analysis' in analysis_data:
            wyckoff = analysis_data['wyckoff_analysis']

            # Current phase
            wyckoff_data['current_phase'] = wyckoff.get('current_phase', 'Unknown')

            # Phases
            if 'phases' in wyckoff:
                wyckoff_data['phases'] = wyckoff['phases']

            # Accumulation/Distribution zones
            if 'accumulation_zones' in wyckoff:
                wyckoff_data['accumulation_zones'] = wyckoff['accumulation_zones']

            if 'distribution_zones' in wyckoff:
                wyckoff_data['distribution_zones'] = wyckoff['distribution_zones']

            # Spring actions and upthrusts
            if 'spring_actions' in wyckoff:
                wyckoff_data['spring_actions'] = wyckoff['spring_actions']

            if 'upthrusts' in wyckoff:
                wyckoff_data['upthrusts'] = wyckoff['upthrusts']

        return wyckoff_data

# Enhanced SMC Chart generator with real data plotting
class EnhancedSMCChartGenerator:
    def __init__(self):
        self.colors = {
            'bullish': '#00ff88',
            'bearish': '#ff4757',
            'neutral': '#ffa502',
            'order_block_bullish': '#26de81',
            'order_block_bearish': '#fc5c65',
            'liquidity': '#45aaf2',
            'supply_zone': '#fd79a8',
            'demand_zone': '#00b894',
            'fair_value_gap': '#a29bfe',
            'wyckoff_accumulation': '#00cec9',
            'wyckoff_distribution': '#e17055',
            'bos': '#ffeaa7',
            'choch': '#fab1a0'
        }

    def create_enhanced_smc_chart(self, df: pd.DataFrame, smc_data: Dict, wyckoff_data: Dict, pair: str, timeframe: str) -> go.Figure:
        """Create comprehensive SMC chart with real analysis data"""

        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            row_heights=[0.7, 0.15, 0.15],
            subplot_titles=[
                f"{pair} {timeframe} - SMC & Wyckoff Analysis (Latest {len(df)} bars)",
                "Volume Profile", 
                "RSI & Momentum"
            ],
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

        # Add SMC elements
        self._add_order_blocks(fig, df, smc_data, row=1)
        self._add_liquidity_zones(fig, df, smc_data, row=1)
        self._add_supply_demand_zones(fig, df, smc_data, row=1)
        self._add_fair_value_gaps(fig, df, smc_data, row=1)
        self._add_bos_choch_points(fig, df, smc_data, row=1)

        # Add Wyckoff elements
        self._add_wyckoff_phases(fig, df, wyckoff_data, row=1)
        self._add_wyckoff_zones(fig, df, wyckoff_data, row=1)

        # Volume
        if 'volume' in df.columns:
            colors = ['red' if df['close'].iloc[i] < df['open'].iloc[i] else 'green' 
                     for i in range(len(df))]
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['volume'],
                    name="Volume",
                    marker_color=colors,
                    opacity=0.7
                ),
                row=2, col=1
            )

        # RSI and momentum
        rsi_data = self._calculate_rsi(df['close'])
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=rsi_data,
                name="RSI",
                line=dict(color=self.colors['order_block_bullish'], width=2)
            ),
            row=3, col=1
        )

        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1, annotation_text="Overbought")
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1, annotation_text="Oversold")
        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=3, col=1)

        # Layout updates
        fig.update_layout(
            title=f"{pair} {timeframe} - Enhanced SMC & Wyckoff Analysis",
            height=900,
            showlegend=True,
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            font=dict(size=12),
            hovermode='x unified'
        )

        # Update y-axes
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)
        fig.update_xaxes(title_text="Time", row=3, col=1)

        return fig

    def _add_order_blocks(self, fig: go.Figure, df: pd.DataFrame, smc_data: Dict, row: int):
        """Add order block rectangles"""
        for block in smc_data.get('order_blocks', []):
            if 'start_time' in block and 'end_time' in block:
                color = self.colors['order_block_bullish'] if block.get('type') == 'bullish' else self.colors['order_block_bearish']

                fig.add_shape(
                    type="rect",
                    x0=block['start_time'],
                    x1=block['end_time'],
                    y0=block.get('low', 0),
                    y1=block.get('high', 0),
                    fillcolor=color,
                    opacity=0.3,
                    line=dict(color=color, width=2),
                    row=row, col=1
                )

                # Add annotation
                fig.add_annotation(
                    x=block['start_time'],
                    y=block.get('high', 0),
                    text=f"OB-{block.get('type', 'Unknown').upper()}",
                    showarrow=True,
                    arrowhead=2,
                    row=row, col=1
                )

    def _add_liquidity_zones(self, fig: go.Figure, df: pd.DataFrame, smc_data: Dict, row: int):
        """Add liquidity zones"""
        for zone in smc_data.get('liquidity_zones', []):
            if 'level' in zone:
                fig.add_hline(
                    y=zone['level'],
                    line_dash="dot",
                    line_color=self.colors['liquidity'],
                    annotation_text=f"LIQ: {zone.get('type', 'Zone')}",
                    row=row, col=1
                )

    def _add_supply_demand_zones(self, fig: go.Figure, df: pd.DataFrame, smc_data: Dict, row: int):
        """Add supply and demand zones - FIXED VERSION"""
        supply_demand_data = smc_data.get('supply_demand_zones', [])

        # Handle both list and dict formats
        if isinstance(supply_demand_data, dict):
            # Original expected format
            supply_zones = supply_demand_data.get('supply_zones', [])
            demand_zones = supply_demand_data.get('demand_zones', [])
        elif isinstance(supply_demand_data, list):
            # Handle list format - separate by type
            supply_zones = [zone for zone in supply_demand_data if zone.get('type') == 'supply']
            demand_zones = [zone for zone in supply_demand_data if zone.get('type') == 'demand']
        else:
            supply_zones = []
            demand_zones = []

        # Add supply zones
        for zone in supply_zones:
            if 'start_time' in zone and 'end_time' in zone:
                fig.add_shape(
                    type="rect",
                    x0=zone['start_time'],
                    x1=zone['end_time'],
                    y0=zone.get('low', 0),
                    y1=zone.get('high', 0),
                    fillcolor=self.colors['supply_zone'],
                    opacity=0.2,
                    line=dict(color=self.colors['supply_zone'], width=1, dash="dash"),
                    row=row, col=1
                )
            elif 'level' in zone:
                # Handle simple level-based zones
                fig.add_hline(
                    y=zone['level'],
                    line_dash="dash",
                    line_color=self.colors['supply_zone'],
                    annotation_text="Supply Zone",
                    row=row, col=1
                )

        # Add demand zones
        for zone in demand_zones:
            if 'start_time' in zone and 'end_time' in zone:
                fig.add_shape(
                    type="rect",
                    x0=zone['start_time'],
                    x1=zone['end_time'],
                    y0=zone.get('low', 0),
                    y1=zone.get('high', 0),
                    fillcolor=self.colors['demand_zone'],
                    opacity=0.2,
                    line=dict(color=self.colors['demand_zone'], width=1, dash="dash"),
                    row=row, col=1
                )
            elif 'level' in zone:
                # Handle simple level-based zones
                fig.add_hline(
                    y=zone['level'],
                    line_dash="dash",
                    line_color=self.colors['demand_zone'],
                    annotation_text="Demand Zone",
                    row=row, col=1
                )
    
    def _add_fair_value_gaps(self, fig: go.Figure, df: pd.DataFrame, smc_data: Dict, row: int):
        """Add Fair Value Gaps (FVG)"""
        for fvg in smc_data.get('fair_value_gaps', []):
            if 'start_time' in fvg and 'end_time' in fvg:
                fig.add_shape(
                    type="rect",
                    x0=fvg['start_time'],
                    x1=fvg['end_time'],
                    y0=fvg.get('low', 0),
                    y1=fvg.get('high', 0),
                    fillcolor=self.colors['fair_value_gap'],
                    opacity=0.15,
                    line=dict(color=self.colors['fair_value_gap'], width=1),
                    row=row, col=1
                )

    def _add_bos_choch_points(self, fig: go.Figure, df: pd.DataFrame, smc_data: Dict, row: int):
        """Add Break of Structure and Change of Character points"""
        # BOS points
        for bos in smc_data.get('bos_points', []):
            if 'timestamp' in bos:
                fig.add_annotation(
                    x=bos['timestamp'],
                    y=bos.get('price', 0),
                    text="BOS",
                    showarrow=True,
                    arrowhead=4,
                    arrowcolor=self.colors['bos'],
                    bgcolor=self.colors['bos'],
                    bordercolor=self.colors['bos'],
                    row=row, col=1
                )

        # CHoCH points
        for choch in smc_data.get('choch_points', []):
            if 'timestamp' in choch:
                fig.add_annotation(
                    x=choch['timestamp'],
                    y=choch.get('price', 0),
                    text="CHoCH",
                    showarrow=True,
                    arrowhead=3,
                    arrowcolor=self.colors['choch'],
                    bgcolor=self.colors['choch'],
                    bordercolor=self.colors['choch'],
                    row=row, col=1
                )

    def _add_wyckoff_phases(self, fig: go.Figure, df: pd.DataFrame, wyckoff_data: Dict, row: int):
        """Add Wyckoff phase backgrounds"""
        for phase in wyckoff_data.get('phases', []):
            if 'start_time' in phase and 'end_time' in phase:
                phase_type = phase.get('phase', 'unknown').lower()

                if 'accumulation' in phase_type:
                    color = self.colors['wyckoff_accumulation']
                elif 'distribution' in phase_type:
                    color = self.colors['wyckoff_distribution']
                else:
                    color = self.colors['neutral']

                fig.add_vrect(
                    x0=phase['start_time'],
                    x1=phase['end_time'],
                    fillcolor=color,
                    opacity=0.1,
                    row=row, col=1,
                    annotation_text=phase.get('phase', 'Phase'),
                    annotation_position="top left"
                )

    def _add_wyckoff_zones(self, fig: go.Figure, df: pd.DataFrame, wyckoff_data: Dict, row: int):
        """Add Wyckoff accumulation/distribution zones"""
        # Accumulation zones
        for zone in wyckoff_data.get('accumulation_zones', []):
            if 'start_time' in zone and 'end_time' in zone:
                fig.add_shape(
                    type="rect",
                    x0=zone['start_time'],
                    x1=zone['end_time'],
                    y0=zone.get('low', 0),
                    y1=zone.get('high', 0),
                    fillcolor=self.colors['wyckoff_accumulation'],
                    opacity=0.25,
                    line=dict(color=self.colors['wyckoff_accumulation'], width=2),
                    row=row, col=1
                )

        # Distribution zones
        for zone in wyckoff_data.get('distribution_zones', []):
            if 'start_time' in zone and 'end_time' in zone:
                fig.add_shape(
                    type="rect",
                    x0=zone['start_time'],
                    x1=zone['end_time'],
                    y0=zone.get('low', 0),
                    y1=zone.get('high', 0),
                    fillcolor=self.colors['wyckoff_distribution'],
                    opacity=0.25,
                    line=dict(color=self.colors['wyckoff_distribution'], width=2),
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

# Main enhanced dashboard
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📈 ZANALYTICS SMC INTELLIGENCE</h1>
        <p>Enhanced Smart Money Concepts & Wyckoff Analysis Dashboard</p>
        <p><em>Plotting Real Analysis Data with Latest Bars First</em></p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize components
    config = SMCConfig()
    loader = EnhancedSMCDataLoader(config)
    chart_gen = EnhancedSMCChartGenerator()

    # Sidebar controls
    st.sidebar.title("🎛️ Enhanced SMC Controls")

    # Settings section
    with st.sidebar.expander("⚙️ Settings", expanded=True):
        max_bars = st.number_input(
            "Max Bars to Load (Latest First)",
            min_value=100,
            max_value=10000,
            value=config.max_bars,
            step=100,
            help="Shows the most recent N bars"
        )
        config.max_bars = max_bars

        show_smc_elements = st.checkbox("Show SMC Elements", value=True)
        show_wyckoff_elements = st.checkbox("Show Wyckoff Elements", value=True)
        show_analysis_data = st.checkbox("Show Analysis Data", value=True)

    # Scan available data
    available_data = loader.scan_available_data()

    if not available_data:
        st.error("No data found in ./data directory")
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
                # Load CSV data (latest bars first)
                with st.spinner(f"Loading latest {max_bars} bars for {selected_pair} {selected_tf}..."):
                    df = loader.load_csv_data(tf_data['file'], max_bars)

                if df is not None:
                    # Load analysis data
                    analysis_data = {}
                    if os.path.exists(tf_data['summary_file']):
                        analysis_data = loader.load_analysis_json(tf_data['summary_file'])
                    elif pair_data['analysis_reports']:
                        analysis_data = loader.load_analysis_json(pair_data['analysis_reports'][0])

                    # Parse SMC and Wyckoff data
                    smc_data = loader.parse_smc_data(analysis_data)
                    wyckoff_data = loader.parse_wyckoff_data(analysis_data)

                    # Main metrics with latest data
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        current_price = df['close'].iloc[-1] if len(df) > 0 else 0
                        price_change = df['close'].iloc[-1] - df['close'].iloc[-2] if len(df) > 1 else 0
                        change_pct = (price_change / df['close'].iloc[-2] * 100) if len(df) > 1 and df['close'].iloc[-2] != 0 else 0
                        st.metric("Latest Price", f"{current_price:.4f}", f"{price_change:+.4f} ({change_pct:+.2f}%)")

                    with col2:
                        period_high = df['high'].max()
                        period_low = df['low'].min()
                        st.metric("Period Range", f"{period_high:.4f}", f"Low: {period_low:.4f}")

                    with col3:
                        volatility = df['close'].pct_change().std() * 100
                        avg_volume = df['volume'].mean() if 'volume' in df.columns else 0
                        st.metric("Volatility", f"{volatility:.2f}%", f"Avg Vol: {avg_volume:,.0f}")

                    with col4:
                        total_bars = len(df)
                        latest_time = df.index[-1].strftime('%Y-%m-%d %H:%M') if len(df) > 0 else "N/A"
                        st.metric("Bars Loaded", f"{total_bars:,}", f"Latest: {latest_time}")

                    # SMC & Wyckoff Analysis cards
                    col1, col2 = st.columns(2)

                    with col1:
                        if smc_data['market_structure']:
                            st.markdown('<div class="smc-card">', unsafe_allow_html=True)
                            st.markdown("**🎯 SMC Market Structure**")
                            trend = smc_data['market_structure'].get('trend', 'NEUTRAL')
                            st.markdown(f"Trend: **{trend}**")
                            st.markdown(f"Order Blocks: **{len(smc_data['order_blocks'])}**")
                            st.markdown(f"Liquidity Zones: **{len(smc_data['liquidity_zones'])}**")
                            st.markdown('</div>', unsafe_allow_html=True)

                    with col2:
                        if wyckoff_data['current_phase'] != 'Unknown':
                            st.markdown('<div class="wyckoff-card">', unsafe_allow_html=True)
                            st.markdown("**📊 Wyckoff Analysis**")
                            st.markdown(f"Current Phase: **{wyckoff_data['current_phase']}**")
                            st.markdown(f"Accumulation Zones: **{len(wyckoff_data['accumulation_zones'])}**")
                            st.markdown(f"Distribution Zones: **{len(wyckoff_data['distribution_zones'])}**")
                            st.markdown('</div>', unsafe_allow_html=True)

                    # Enhanced SMC Chart with real analysis data
                    st.subheader(f"📊 {selected_pair} {config.timeframe_display.get(selected_tf, selected_tf)} - Enhanced SMC & Wyckoff Chart")

                    chart = chart_gen.create_enhanced_smc_chart(
                        df, 
                        smc_data if show_smc_elements else {}, 
                        wyckoff_data if show_wyckoff_elements else {}, 
                        selected_pair, 
                        selected_tf
                    )
                    st.plotly_chart(chart, use_container_width=True)

                    # Analysis insights
                    if show_analysis_data and analysis_data:
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.subheader("🎯 SMC Insights")
                            if smc_data['order_blocks']:
                                st.markdown("**Order Blocks Detected:**")
                                for i, ob in enumerate(smc_data['order_blocks'][:3]):  # Show first 3
                                    ob_type = ob.get('type', 'Unknown')
                                    strength = ob.get('strength', 0)
                                    st.markdown(f"• {ob_type.title()} OB (Strength: {strength:.2f})")

                            if smc_data['fair_value_gaps']:
                                st.markdown(f"**Fair Value Gaps:** {len(smc_data['fair_value_gaps'])}")

                        with col2:
                            st.subheader("📊 Wyckoff Status")
                            if wyckoff_data['current_phase'] != 'Unknown':
                                st.markdown(f"**Current Phase:** {wyckoff_data['current_phase']}")

                                if wyckoff_data['spring_actions']:
                                    st.markdown(f"**Spring Actions:** {len(wyckoff_data['spring_actions'])}")

                                if wyckoff_data['upthrusts']:
                                    st.markdown(f"**Upthrusts:** {len(wyckoff_data['upthrusts'])}")

                        with col3:
                            st.subheader("📈 Technical Summary")
                            if 'technical_indicators' in analysis_data:
                                indicators = analysis_data['technical_indicators']
                                for indicator, value in list(indicators.items())[:5]:  # Show first 5
                                    if isinstance(value, (int, float)):
                                        st.markdown(f"**{indicator.upper()}:** {value:.2f}")

                    # Raw data preview
                    with st.expander("📋 Latest Data Preview (Most Recent First)"):
                        # Show data in reverse order (most recent first)
                        display_df = df.iloc[::-1].head(20)
                        st.dataframe(display_df, use_container_width=True)

                    # Analysis JSON preview
                    if analysis_data and show_analysis_data:
                        with st.expander("🔍 Complete Analysis Data"):
                            st.json(analysis_data)

if __name__ == "__main__":
    main()
