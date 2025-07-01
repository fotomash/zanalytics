import streamlit as st
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime, timedelta
import warnings
import re
from pathlib import Path

warnings.filterwarnings('ignore')

from typing import Dict, Optional

# ========== Parquet File Scanning Utility ==========
def scan_parquet_files(parquet_dir):
    files = []
    for f in Path(parquet_dir).rglob("*.parquet"):
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
        files.append((symbol, timeframe, f.relative_to(parquet_dir)))
    return files

try:
    from utils.quantum_microstructure_analyzer import QuantumMicrostructureAnalyzer
except ImportError:
    # Fallback stub so the dashboard can still run if the full library isn't available.
    class QuantumMicrostructureAnalyzer:
        def __init__(self, config_path: str):
            self.config_path = config_path
            self.session_state = {}

# Import custom modules
from utils.data_processor import DataProcessor
from utils.timeframe_converter import TimeframeConverter
from utils.technical_analysis import TechnicalAnalysis

try:
    from utils.smc_analyzer import SMCAnalyzer
except ImportError:
    # Graceful fallback if the module isn't available
    class SMCAnalyzer:
        def analyze(self, df):
            return {}
from utils.wyckoff_analyzer import WyckoffAnalyzer
from utils.volume_profile import VolumeProfileAnalyzer
from components.chart_builder import ChartBuilder
from components.analysis_panel import AnalysisPanel


# ===================== QRT‑LEVEL QUANTUM ANALYZER =====================
class QRTQuantumAnalyzer(QuantumMicrostructureAnalyzer):
    """QRT‑level quantum microstructure analyzer with advanced Wyckoff and liquidity analysis"""

    def __init__(self, config_path: str):
        super().__init__(config_path)
        self.tiquidity_engine = TiquidityEngine()
        self.wyckoff_analyzer = WyckoffQuantumAnalyzer()

    def calculate_tiquidity_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced tick liquidity metrics"""
        # --- Ensure required columns exist for higher‑timeframe data ---
        if 'price_mid' not in df.columns:
            df['price_mid'] = (df['high'] + df['low']) / 2

        # Make sure a 'timestamp' column exists for downstream calculations/plots
        if 'timestamp' not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                # copy to avoid SettingWithCopy issues and preserve original index
                df = df.copy()
                df['timestamp'] = df.index
            elif 'datetime' in df.columns:
                df['timestamp'] = pd.to_datetime(df['datetime'])
            elif 'date' in df.columns:
                df['timestamp'] = pd.to_datetime(df['date'])
            elif 'time' in df.columns:
                df['timestamp'] = pd.to_datetime(df['time'])
            else:
                raise KeyError(
                    "DataFrame must contain a datetime index or a 'timestamp'/'datetime'/'date' column "
                    "for QRT analytics."
                )

        # Ensure 'timestamp' is a proper pandas datetime (handles tick‑data strings like '2025.06.29 19:21:36')
        if not np.issubdtype(df['timestamp'].dtype, np.datetime64):
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            if df['timestamp'].isna().all():
                raise ValueError("Failed to parse 'timestamp' column to datetime format.")

        # Provide a usable volume field for liquidity math
        if 'inferred_volume' not in df.columns:
            if 'real_volume' in df.columns and (df['real_volume'] > 0).any():
                df['inferred_volume'] = df['real_volume'].replace(0, np.nan).fillna(method='ffill').fillna(0)
            elif 'volume' in df.columns:
                df['inferred_volume'] = df['volume']
            else:
                df['inferred_volume'] = 1.0  # minimal placeholder

        # Fill missing OHLC columns with placeholder logic, prefer 'last' if present
        ohlc_cols = ['open', 'high', 'low', 'close']
        missing_ohlc = [col for col in ohlc_cols if col not in df.columns]
        if missing_ohlc:
            base_price = df['last'] if 'last' in df.columns else df['price_mid']
            df['open'] = base_price.shift().fillna(base_price)
            df['close'] = base_price
            df['high'] = pd.concat([df['open'], df['close']], axis=1).max(axis=1)
            df['low'] = pd.concat([df['open'], df['close']], axis=1).min(axis=1)

        # Cumulative Delta
        df['delta'] = (df['inferred_volume'] * np.where(df['price_mid'].diff() > 0, 1, -1))
        df['cumulative_delta'] = df['delta'].cumsum()

        # Delta Divergence
        df['price_norm'] = (df['price_mid'] - df['price_mid'].min()) / (df['price_mid'].max() - df['price_mid'].min())
        df['delta_norm'] = (df['cumulative_delta'] - df['cumulative_delta'].min()) / (
                    df['cumulative_delta'].max() - df['cumulative_delta'].min() + 1e-9)
        df['delta_divergence'] = df['price_norm'] - df['delta_norm']

        # Absorption Ratio
        df['absorption_ratio'] = df['inferred_volume'].rolling(20).sum() / (
                    abs(df['price_mid'].diff()).rolling(20).sum() + 1e-9)

        # Exhaustion Levels
        df['volume_ma'] = df['inferred_volume'].rolling(50).mean()
        df['exhaustion_score'] = pd.Series(
            np.where(
                (df['inferred_volume'] > df['volume_ma'] * 3) &
                (abs(df['price_mid'].diff()) < df['price_mid'].rolling(50).std() * 0.1),
                1, 0
            )
        ).rolling(10).sum().values

        # Liquidity Voids
        if 'tick_interval_ms' not in df.columns:
            # Approximate bar interval in milliseconds, robust to non-monotonic/duplicate timestamps
            df['tick_interval_ms'] = (
                df['timestamp'].diff()
                .dt.total_seconds()
                .abs()  # ensure positive
                .fillna(0)
                .mul(1000)
            )
        # Ensure .rolling() is only called on pandas Series, not numpy arrays
        # Robust: use pd.Series(df['tick_interval_ms']) to avoid accidental numpy
        df['tick_gap'] = pd.Series(df['tick_interval_ms']).rolling(10).max().values
        df['liquidity_void'] = np.where(df['tick_gap'] > df['tick_interval_ms'].mean() * 5, 1, 0)

        return df

    def detect_wyckoff_structures(self, df: pd.DataFrame) -> Dict:
        """Detect comprehensive Wyckoff patterns at QRT level"""
        patterns = {
            'accumulation': [],
            'distribution': [],
            'springs': [],
            'utads': [],
            'tests': [],
            'creek_jumps': []
        }

        # Volume profile for Wyckoff analysis
        df['volume_sma'] = df['inferred_volume'].rolling(50).mean()
        df['volume_ratio'] = df['inferred_volume'] / (df['volume_sma'] + 1e-9)

        # Detect Accumulation Structures
        for i in range(100, len(df) - 50):
            window = df.iloc[i - 100:i + 50]

            # Phase A - Selling Climax
            if self._detect_selling_climax(window[:50]):
                # Phase B - Building Cause
                if self._detect_accumulation_range(window[25:75]):
                    # Phase C - Spring
                    spring_idx = self._detect_spring(window[50:100])
                    if spring_idx is not None:
                        # Phase D - Markup
                        if self._detect_markup_beginning(window[75:]):
                            patterns['accumulation'].append({
                                'start': window.index[0],
                                'spring_time': window.index[50 + spring_idx],
                                'phase': 'complete',
                                'confidence': 0.85
                            })

        # Detect Micro‑Wyckoff Patterns (Tick Level)
        for i in range(20, len(df) - 10):
            micro_window = df.iloc[i - 10:i + 10]

            # Micro‑Spring Detection
            if (micro_window['price_mid'].iloc[10] < micro_window['price_mid'].iloc[:10].min() and
                    micro_window['price_mid'].iloc[-1] > micro_window['price_mid'].iloc[10] and
                    micro_window['volume_ratio'].iloc[10] > 2.5):
                velocity = (micro_window['price_mid'].iloc[-1] - micro_window['price_mid'].iloc[10]) / (
                            micro_window['tick_interval_ms'].iloc[10:].sum() + 1)

                patterns['springs'].append({
                    'type': 'micro_spring',
                    'timestamp': micro_window['timestamp'].iloc[10],
                    'low': micro_window['price_mid'].iloc[10],
                    'rejection_velocity': velocity,
                    'volume_surge': micro_window['volume_ratio'].iloc[10],
                    'delta_flip': micro_window['delta'].iloc[11:].sum(),
                    'confidence': min(velocity * 1000 * micro_window['volume_ratio'].iloc[10] / 3, 1.0)
                })

        return patterns

    def calculate_orderflow_footprint(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate order flow footprint for each price level"""
        df['price_level'] = (df['price_mid'] * 10000).round() / 10000

        footprint = df.groupby('price_level').agg({
            'inferred_volume': 'sum',
            'delta': 'sum',
            'timestamp': 'count'
        }).rename(columns={'timestamp': 'tick_count'})

        footprint['bid_volume'] = np.where(footprint['delta'] < 0, abs(footprint['delta']), 0)
        footprint['ask_volume'] = np.where(footprint['delta'] > 0, footprint['delta'], 0)
        footprint['imbalance'] = footprint['ask_volume'] - footprint['bid_volume']

        return footprint

    def detect_liquidity_patterns(self, df: pd.DataFrame) -> Dict:
        """Detect advanced liquidity patterns"""
        patterns = {
            'absorption_zones': [],
            'liquidity_voids': [],
            'imbalance_zones': [],
            'delta_divergences': []
        }

        high_absorption = df[df['absorption_ratio'] > df['absorption_ratio'].quantile(0.9)]
        for idx in high_absorption.index:
            if idx > 10 and idx < len(df) - 10:
                pre_move = abs(df.loc[idx - 10:idx - 1, 'price_mid'].pct_change().sum())
                post_move = abs(df.loc[idx:idx + 10, 'price_mid'].pct_change().sum())

                if post_move < pre_move * 0.3:
                    patterns['absorption_zones'].append({
                        'timestamp': df.loc[idx, 'timestamp'],
                        'price': df.loc[idx, 'price_mid'],
                        'absorption_ratio': df.loc[idx, 'absorption_ratio'],
                        'effectiveness': 1 - (post_move / (pre_move + 1e-9))
                    })

        divergence_points = df[abs(df['delta_divergence']) > 0.3]
        for idx in divergence_points.index:
            patterns['delta_divergences'].append({
                'timestamp': df.loc[idx, 'timestamp'],
                'price': df.loc[idx, 'price_mid'],
                'divergence': df.loc[idx, 'delta_divergence'],
                'type': 'bullish' if df.loc[idx, 'delta_divergence'] < -0.3 else 'bearish'
            })

        return patterns

    def create_qrt_dashboard(self, df: pd.DataFrame, selected_file: str):
        """Create QRT-level professional dashboard"""
        # [ Full Streamlit dashboard code from user snippet goes here ]
        # --- QRT Dashboard Streamlit Implementation ---
        import streamlit as st
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        st.title("🚀 QRT Quantum Microstructure Dashboard")
        st.markdown(f"**Instrument:** `{selected_file}`")
        st.divider()
        # Calculate metrics
        df = self.calculate_tiquidity_metrics(df)
        wyckoff = self.detect_wyckoff_structures(df)
        liquidity = self.detect_liquidity_patterns(df)
        footprint = self.calculate_orderflow_footprint(df)
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Last Price", f"${df['price_mid'].iloc[-1]:.5f}")
        with col2:
            st.metric("Cumulative Delta", f"{df['cumulative_delta'].iloc[-1]:,.0f}")
        with col3:
            st.metric("Absorption Ratio", f"{df['absorption_ratio'].iloc[-1]:.2f}")
        with col4:
            st.metric("Exhaustion Score", f"{df['exhaustion_score'].iloc[-1]:.2f}")
        # Chart
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                            row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(
            x=df['timestamp'], open=df['open'], high=df['high'],
            low=df['low'], close=df['close'], name="Price"), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['cumulative_delta'], name="Cumulative Delta", line=dict(color='orange')), row=2,
            col=1)
        fig.add_trace(go.Bar(
            x=df['timestamp'], y=df['inferred_volume'], name="Volume", marker_color='rgba(50,150,255,0.2)'), row=2,
            col=1)
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        # Wyckoff Patterns
        with st.expander("🔬 Wyckoff Pattern Detection"):
            st.write(wyckoff)
        # Liquidity Patterns
        with st.expander("💧 Liquidity Patterns"):
            st.write(liquidity)
        # Orderflow Footprint
        with st.expander("🦶 Orderflow Footprint"):
            st.dataframe(footprint.head(30))
        # Generate signals
        signals = self.generate_qrt_signals(df, wyckoff, liquidity)
        st.success(f"**QRT Trading Signal:** {signals.get('signal', 'N/A').upper()}")
        st.json(signals)
        # End QRT Dashboard

    def generate_qrt_signals(self, df: pd.DataFrame, wyckoff: Dict, liquidity: Dict) -> Dict:
        """Generate QRT-level trading signals"""
        # Signal logic: combine Wyckoff spring, absorption, and delta divergence
        signal = "neutral"
        reasons = []
        # Bullish: recent micro_spring + strong absorption + bullish delta divergence
        recent_springs = [s for s in wyckoff.get('springs', []) if s.get('confidence', 0) > 0.7]
        strong_abs = [a for a in liquidity.get('absorption_zones', []) if a.get('effectiveness', 0) > 0.5]
        bullish_div = [d for d in liquidity.get('delta_divergences', []) if d.get('type') == 'bullish']
        bearish_div = [d for d in liquidity.get('delta_divergences', []) if d.get('type') == 'bearish']
        if recent_springs and strong_abs and bullish_div:
            signal = "long"
            reasons.append("Micro spring, strong absorption, bullish delta divergence")
        elif bearish_div and not recent_springs and not strong_abs:
            signal = "short"
            reasons.append("Bearish delta divergence, no bullish absorption/spring")
        elif strong_abs:
            signal = "wait"
            reasons.append("Absorption but no clear reversal")
        else:
            signal = "neutral"
            reasons.append("No strong confluence")
        return {"signal": signal, "reasons": reasons, "springs": recent_springs, "absorption": strong_abs,
                "delta_div": bullish_div + bearish_div}


class TiquidityEngine:
    """Engine for tick liquidity analysis"""
    pass


class WyckoffQuantumAnalyzer:
    """Advanced Wyckoff pattern analyzer"""

    def detect_phases(self, df: pd.DataFrame) -> Dict:
        """Detect Wyckoff phases using volume and price action heuristics"""
        phases = []
        # Simple example: If cumulative delta rising and volume increasing, markup
        if df['cumulative_delta'].iloc[-1] > df['cumulative_delta'].iloc[0] and \
                df['inferred_volume'].rolling(50).mean().iloc[-1] > df['inferred_volume'].rolling(50).mean().iloc[0]:
            phases.append('Markup')
        elif df['cumulative_delta'].iloc[-1] < df['cumulative_delta'].iloc[0]:
            phases.append('Distribution')
        else:
            phases.append('Accumulation')
        return {"phases": phases}

    def detect_events(self, df: pd.DataFrame) -> list:
        """Detect key Wyckoff events (SC, AR, ST, Spring, UTAD)"""
        events = []
        # Example: Spring = local min with high volume
        min_idx = df['price_mid'].idxmin()
        if df['inferred_volume'].iloc[min_idx] > df['inferred_volume'].rolling(50).mean().iloc[min_idx] * 2:
            events.append({"type": "Spring", "price": df['price_mid'].iloc[min_idx]})
        return events


# =================== END QRT ANALYZER DEFINITIONS =====================

# Page configuration
st.set_page_config(
    page_title="Zanflow Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    div[data-testid="metric-container"] {
        background-color: rgba(28, 131, 225, 0.1);
        border: 1px solid rgba(28, 131, 225, 0.2);
        padding: 5px 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'df_to_use' not in st.session_state:
    st.session_state.df_to_use = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}

# Header
# st.title("🏛️ Zanflow Multi-Timeframe Analysis Dashboard")
st.markdown(
    "<small>Institutional-Grade Market Analysis with Smart Money Concepts & Wyckoff Methodology</small>",
    unsafe_allow_html=True
)

# Modernized Parquet-only sidebar
with st.sidebar:
    st.header("📁 Data Selection")
    # Set parquet_dir to the value from secrets.toml, fallback to './data' if not found
    parquet_dir = st.secrets.get("PARQUET_DATA_DIR", "./data")
    file_info = scan_parquet_files(parquet_dir)
    symbols = sorted({sym for sym, _, _ in file_info})
    if not symbols:
        st.warning("No Parquet files found in data folder.")
        st.info("Please place your Parquet files in the configured directory.")
        st.session_state.df_to_use = None
    else:
        selected_symbol = st.selectbox("Select Symbol", symbols)
        timeframes = sorted({tf for sym, tf, _ in file_info if sym == selected_symbol})
        selected_timeframe = st.selectbox("Select Timeframe", timeframes)
        rel_path = next(f for sym, tf, f in file_info if sym == selected_symbol and tf == selected_timeframe)
        full_path = Path(parquet_dir) / rel_path
        try:
            df = pd.read_parquet(full_path)
            df.columns = [c.lower() for c in df.columns]
            max_bars = len(df)
            bars_to_use = st.slider("Last N Bars", min_value=20, max_value=max_bars, value=min(500, max_bars))
            if st.button("📥 Load Data", type="primary"):
                st.session_state.df_to_use = df.tail(bars_to_use)
                st.session_state.analysis_results = {}
                st.success(f"Loaded {selected_symbol} {selected_timeframe} [{bars_to_use} bars]")
        except Exception as e:
            st.error(f"⚠️ File `{full_path.name}` is not a valid Parquet file or is corrupted. Please check your files.\n\nError: {e}")
            st.session_state.df_to_use = None

    # Analysis settings (only show if data loaded)
    if st.session_state.df_to_use is not None:
        st.divider()
        st.header("⚙️ Analysis Settings")
        # Analysis options
        st.subheader("Analysis Modules")
        show_smc = st.checkbox("Smart Money Concepts", value=True)
        show_wyckoff = st.checkbox("Wyckoff Analysis", value=True)
        show_volume_profile = st.checkbox("Volume Profile", value=True)
        show_indicators = st.checkbox("Technical Indicators", value=True)
        # Visualization options
        st.subheader("Visualization Options")
        chart_type = st.selectbox("Chart Type", ["Candlestick", "Heikin Ashi", "Line"])
        show_volume = st.checkbox("Show Volume", value=True)
        show_orderflow = st.checkbox("Show Order Flow", value=True)
        # Run analysis button
        if st.button("🔍 Run Analysis", type="primary"):
            with st.spinner("Running comprehensive analysis..."):
                try:
                    tf_data = st.session_state.df_to_use
                    # Initialize analyzers
                    smc = SMCAnalyzer() if show_smc else None
                    wyckoff = WyckoffAnalyzer() if show_wyckoff else None
                    volume_profile = VolumeProfileAnalyzer() if show_volume_profile else None
                    ta = TechnicalAnalysis() if show_indicators else None
                    results = {}
                    if smc:
                        results['smc'] = smc.analyze(tf_data)
                    if wyckoff:
                        results['wyckoff'] = wyckoff.analyze(tf_data)
                    if volume_profile:
                        results['volume_profile'] = volume_profile.analyze(tf_data)
                    if ta:
                        results['indicators'] = ta.calculate_all(tf_data)
                    st.session_state.analysis_results = results
                    st.success("✅ Analysis complete!")
                except Exception as e:
                    st.error(f"Analysis error: {str(e)}")

# Main content area
if st.session_state.df_to_use is not None:
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Chart Analysis", "📈 Market Structure", "📋 Reports", "🧠 QRT Dashboard"])

    with tab1:
        # Main chart analysis
        tf_data = st.session_state.df_to_use
        # Metrics row
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            current_price = tf_data['close'].iloc[-1]
            price_change = ((current_price - tf_data['close'].iloc[-2]) / tf_data['close'].iloc[-2]) * 100
            st.metric("Current Price", f"${current_price:.5f}", f"{price_change:+.2f}%")
        with col2:
            volume_col = 'volume' if 'volume' in tf_data.columns else 'tick_volume' if 'tick_volume' in tf_data.columns else None
            if volume_col:
                st.metric("24h Volume", f"{tf_data[volume_col].tail(24).sum():,.0f}")
            else:
                st.metric("24h Volume", "N/A")

        with col3:
            st.metric("Volatility", f"{tf_data['close'].pct_change().std() * 100:.2f}%")
        with col4:
            high_24h = tf_data['high'].tail(24).max()
            st.metric("24h High", f"${high_24h:.5f}")
        with col5:
            low_24h = tf_data['low'].tail(24).min()
            st.metric("24h Low", f"${low_24h:.5f}")

        # Main chart
        chart_builder = ChartBuilder()
        fig = chart_builder.create_main_chart(
            tf_data,
            "Current",
            chart_type=chart_type,
            show_volume=show_volume,
            analysis_results=st.session_state.analysis_results
        )
        st.plotly_chart(fig, use_container_width=True)

        # Analysis panel
        if st.session_state.analysis_results:
            analysis_panel = AnalysisPanel()
            analysis_panel.display_results(st.session_state.analysis_results)

    with tab2:
        # Market Structure Analysis
        st.header("Market Structure Analysis")
        tf_data = st.session_state.df_to_use
        # Structure visualization
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Smart Money Concepts")
            if 'smc' in st.session_state.analysis_results:
                smc_results = st.session_state.analysis_results['smc']
                # Display liquidity zones
                st.markdown("#### Liquidity Zones")
                for zone in smc_results.get('liquidity_zones', [])[:5]:
                    zone_type = "🔴 Sell-side" if zone['type'] == 'SSL' else "🟢 Buy-side"
                    st.write(f"{zone_type}: ${zone['level']:.5f} (Strength: {zone['strength']:.2f})")
                # Display order blocks
                st.markdown("#### Order Blocks")
                for ob in smc_results.get('order_blocks', [])[:5]:
                    ob_type = "🟢 Bullish" if ob['type'] == 'bullish' else "🔴 Bearish"
                    st.write(f"{ob_type}: ${ob['start']:.5f} - ${ob['end']:.5f}")
        with col2:
            st.subheader("Wyckoff Analysis")
            if 'wyckoff' in st.session_state.analysis_results:
                wyckoff_results = st.session_state.analysis_results['wyckoff']
                # Current phase
                st.markdown("#### Current Phase")
                phase = wyckoff_results.get('current_phase', 'Unknown')
                phase_emoji = {
                    'Accumulation': '📈',
                    'Markup': '🚀',
                    'Distribution': '📉',
                    'Markdown': '💥'
                }.get(phase, '❓')
                st.write(f"{phase_emoji} **{phase}**")
                # Key events
                st.markdown("#### Recent Events")
                for event in wyckoff_results.get('events', [])[:5]:
                    st.write(f"• {event['type']} at ${event['price']:.5f}")

    with tab3:
        # Reports and Export
        st.header("Analysis Reports")
        tf_data = st.session_state.df_to_use
        # Generate report
        if st.button("📄 Generate Full Report"):
            with st.spinner("Generating comprehensive report..."):
                # Create report content
                report = f"""
# Zanflow Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Market Overview
- **Current Price**: ${tf_data['close'].iloc[-1]:.5f}
- **24h Change**: {((tf_data['close'].iloc[-1] - tf_data['close'].iloc[-24]) / tf_data['close'].iloc[-24] * 100):.2f}%

## Analysis Summary
"""
                # Add analysis results to report
                if st.session_state.analysis_results:
                    for module, results in st.session_state.analysis_results.items():
                        report += f"\n### {module.upper()} Analysis\n"
                        report += str(results)[:500] + "...\n"
                # Display report
                st.text_area("Report Preview", report, height=400)
                # Download button
                st.download_button(
                    label="📥 Download Report",
                    data=report,
                    file_name=f"zanflow_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )

        # JSON Breakdown Commentary Expander
        with st.expander("🗂 JSON Breakdown Commentary"):
            st.markdown("The following is a human-readable explanation of the core analysis JSON output.")
            if st.session_state.analysis_results:
                for module, results in st.session_state.analysis_results.items():
                    st.subheader(f"{module.upper()} Commentary")
                    st.json(results)
            st.download_button(
                label="📥 Download Full JSON",
                data=json.dumps(st.session_state.analysis_results, indent=2),
                file_name=f"zanflow_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

    with tab4:
        # QRT Dashboard (shortcut)
        st.header("QRT Quantum Microstructure Dashboard")
        qrt_analyzer = QRTQuantumAnalyzer(config_path="./config/qrt_config.yaml")
        qrt_analyzer.create_qrt_dashboard(st.session_state.df_to_use, f"{selected_symbol} {selected_timeframe}")

else:
    # Welcome screen
    st.info("👈 Please select and load a Parquet data file from the sidebar to begin analysis")
    # Instructions
    with st.expander("📖 How to Use This Dashboard"):
        st.markdown('''
        1. **Load Data**: Place your Parquet files in the configured data folder and select from the sidebar
        2. **Configure Analysis**: Choose which analysis modules to run and visualization options
        3. **Run Analysis**: Click the "Run Analysis" button to process the data
        4. **Explore Results**: Navigate through the tabs to view different aspects of the analysis
        5. **Export Reports**: Generate and download comprehensive analysis reports

        ### Data Format
        Your Parquet file should have the following columns (case-insensitive, all will be lowercased):
        - Date/Time
        - Open
        - High
        - Low
        - Close
        - Volume

        ### Available Analysis Modules
        - **Smart Money Concepts**: Liquidity zones, order blocks, fair value gaps
        - **Wyckoff Analysis**: Market phases, accumulation/distribution patterns
        - **Volume Profile**: Point of control, value areas, volume nodes
        - **Technical Indicators**: Moving averages, RSI, MACD, Bollinger Bands
''')