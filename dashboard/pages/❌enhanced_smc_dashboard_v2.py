# enhanced_smc_dashboard_v3.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import yaml
import os
from typing import Dict, List, Tuple
from dataclasses import dataclass
from scipy.signal import find_peaks

# Import from utils folder
from utils.analysis_engines import SMCAnalyzer, WyckoffAnalyzer

# Parquet base directory (from secrets if present)
try:
    PARQUET_DATA_DIR = st.secrets["PARQUET_DATA_DIR"]
except Exception:
    PARQUET_DATA_DIR = "/Users/tom/Documents/_trade/_exports/_tick/out/parquet"

# Data classes for Wyckoff analysis
@dataclass
class WyckoffEvent:
    type: str
    timestamp: pd.Timestamp
    price: float
    description: str
    strength: float = 0.0

@dataclass
class VolumeProfile:
    poc_price: float
    poc_volume: float
    vah: float
    val: float
    value_area_pct: float
    profile: Dict[float, float]
    high_volume_nodes: List[float]
    low_volume_nodes: List[float]

# --- [CUSTOM_CSS remains the same] ---
CUSTOM_CSS = """
<style>
    /* Main container styling */
    .main {
    padding: 0rem 1rem;
    }
    
    /* Header styling */
    .dashboard-header {
    background: linear-gradient(135deg, #1e3d59 0%, #2e5266 100%);
    padding: 2rem;
    border-radius: 15px;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .dashboard-title {
    color: #ffc13b;
    font-size: 2.5rem;
    font-weight: 700;
    margin: 0;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .dashboard-subtitle {
    color: #f5f0e1;
    font-size: 1.2rem;
    margin-top: 0.5rem;
    }
    
    /* Card styling */
    .analysis-card {
    background: rgba(30, 61, 89, 0.3);
    border: 1px solid rgba(255, 193, 59, 0.3);
    border-radius: 10px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
    }
    
    .card-header {
    color: #ffc13b;
    font-size: 1.3rem;
    font-weight: 600;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    }
    
    .card-icon {
    margin-right: 0.5rem;
    font-size: 1.5rem;
    }
    
    /* Wyckoff specific styling */
    .wyckoff-phase {
    font-size: 1.5rem;
    font-weight: bold;
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
    margin-bottom: 1rem;
    }
    
    .phase-accumulation {
    background: rgba(0, 208, 132, 0.2);
    border: 2px solid #00d084;
    color: #00d084;
    }
    
    .phase-markup {
    background: rgba(0, 200, 255, 0.2);
    border: 2px solid #00c8ff;
    color: #00c8ff;
    }
    
    .phase-distribution {
    background: rgba(255, 56, 96, 0.2);
    border: 2px solid #ff3860;
    color: #ff3860;
    }
    
    .phase-markdown {
    background: rgba(255, 150, 50, 0.2);
    border: 2px solid #ff9632;
    color: #ff9632;
    }
</style>
"""


def scan_available_symbols(directory):
    """Scans the directory for available CSV files and extracts symbols."""
    symbols = []
    try:
        for file in os.listdir(directory):
            if file.endswith('_bars.csv') or file.endswith('_ticks.csv'):
                symbol = file.split('_')[0]
                if symbol not in symbols:
                    symbols.append(symbol)
        symbols.sort()
    except Exception as e:
        st.error(f"Error scanning directory: {e}")
    return symbols


# ----------- Parquet structure scan -----------
def scan_parquet_structure(base_dir: str) -> Dict[str, List[str]]:
    """
    Walk the parquet directory and build a mapping of {symbol: [timeframes]}.
    Expects files like  {SYMBOL}/{SYMBOL}_{TF}.parquet
    """
    mapping: Dict[str, List[str]] = {}
    base = Path(base_dir)
    if not base.exists():
        return mapping
    for symbol_dir in base.iterdir():
        if symbol_dir.is_dir():
            symbol = symbol_dir.name.upper()
            tfs = []
            for f in symbol_dir.glob(f"{symbol}_*.parquet"):
                # extract timeframe part e.g. BTCUSD_4h.parquet -> 4H
                tf = f.stem.replace(f"{symbol}_", "").upper()
                tfs.append(tf)
            if tfs:
                mapping[symbol] = sorted(list(set(tfs)))
    return mapping


@st.cache_data
def load_and_process_data(symbol: str, timeframe: str, base_dir: str):
    """
    Load parquet file {base_dir}/{symbol}/{symbol}_{timeframe}.parquet
    Returns (df, smc_results, wyckoff_results, filename)
    """
    file_path = Path(base_dir) / symbol / f"{symbol}_{timeframe}.parquet"
    if not file_path.exists():
        st.error(f"File not found: {file_path}")
        return None, None, None, None
    try:
        df = pd.read_parquet(file_path)
        # ensure timestamp index
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)

        # ensure volume column
        vol_cols = [c for c in df.columns if 'vol' in c.lower() or 'volume' in c.lower()]
        if vol_cols:
            df['volume'] = df[vol_cols[0]]

        smc_analyzer = SMCAnalyzer()
        wyckoff_analyzer = WyckoffAnalyzer()
        smc_results = smc_analyzer.analyze(df)
        wyckoff_results = wyckoff_analyzer.analyze(df)
        return df, smc_results, wyckoff_results, file_path.name
    except Exception as e:
        st.error(f"Error loading {file_path.name}: {e}")
        return None, None, None, None

class EnhancedWyckoffAnalysis:
    """Enhanced Wyckoff analysis module"""

    def __init__(self):
        self.config = self.load_wyckoff_config()

    def load_wyckoff_config(self):
        """Load Wyckoff-specific configuration"""
        return {
            'phase_detection': {'window': 120, 'volume_threshold': 1.2},
            'event_detection': {
                'volume_threshold': 1.5,
                'test_volume_ratio': 0.5,
                'spring_threshold': 0.998,
                'upthrust_threshold': 1.002
            },
            'volume_profiler': {
                'bins': 48,
                'value_area_percent': 0.7,
                'node_threshold': 0.15
            }
        }

    def create_wyckoff_chart(self, df, wyckoff_results):
        """Create comprehensive Wyckoff analysis chart"""
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.2, 0.15, 0.15],
            subplot_titles=['Wyckoff Price Action', 'Volume Analysis', 'Effort vs Result', 'Composite Operator']
        )

        # Price chart with Wyckoff events
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'],
            name='Price', increasing_line_color='#00d084', decreasing_line_color='#ff3860'
        ), row=1, col=1)

        # Add Wyckoff events
        for event in wyckoff_results.get('events', [])[:10]:
            event_time = pd.to_datetime(event['time'])
            event_type = event['type']

            # Event markers
            color = '#00d084' if event_type in ['Spring', 'PS', 'AR'] else '#ff3860'
            symbol = 'triangle-up' if event_type in ['Spring', 'PS'] else 'triangle-down'

            fig.add_trace(go.Scatter(
                x=[event_time],
                y=[event['price']],
                mode='markers+text',
                marker=dict(symbol=symbol, size=15, color=color),
                text=event_type,
                textposition="top center",
                showlegend=False
            ), row=1, col=1)

        # Trading ranges
        for tr in wyckoff_results.get('trading_ranges', [])[:3]:
            fig.add_shape(
                type="rect",
                x0=tr['start_time'], x1=tr['end_time'],
                y0=tr['low'], y1=tr['high'],
                fillcolor='rgba(156,39,176,0.1)',
                line=dict(color='#9c27b0', width=1, dash='dot'),
                layer='below',
                row=1, col=1
            )

        # Volume with color coding
        volume_patterns = wyckoff_results.get('volume_analysis', {})
        colors = []
        for i in range(len(df)):
            # Check if this bar is a volume surge or dry-up
            is_surge = any(p['index'] == i for p in volume_patterns.get('volume_surge', []))
            is_dryup = any(p['index'] == i for p in volume_patterns.get('volume_dry_up', []))

            if is_surge:
                colors.append('#9c27b0')  # Purple for surge
            elif is_dryup:
                colors.append('#ffc13b')  # Yellow for dry-up
            else:
                colors.append('#00d084' if df['close'].iloc[i] > df['open'].iloc[i] else '#ff3860')

        fig.add_trace(go.Bar(
            x=df.index, y=df['volume'],
            marker_color=colors,
            name='Volume'
        ), row=2, col=1)

        # Effort vs Result indicator
        effort_result = []
        for pattern in volume_patterns.get('effort_vs_result', []):
            if pattern['type'] == 'high_effort_low_result':
                effort_result.append({
                    'time': pd.to_datetime(pattern['time']),
                    'value': pattern['volume_ratio']
                })

        if effort_result:
            fig.add_trace(go.Scatter(
                x=[e['time'] for e in effort_result],
                y=[e['value'] for e in effort_result],
                mode='markers+lines',
                marker=dict(size=8, color='orange'),
                name='High Effort/Low Result'
            ), row=3, col=1)

        # Composite Operator activity
        co_analysis = wyckoff_results.get('composite_operator', {})

        # Accumulation signs
        for sign in co_analysis.get('accumulation_signs', []):
            if 'range' in sign:
                fig.add_annotation(
                    x=sign['range']['start_time'],
                    y=0.5,
                    text="ACC",
                    showarrow=True,
                    arrowcolor='green',
                    row=4, col=1
                )

        # Distribution signs
        for sign in co_analysis.get('distribution_signs', []):
            fig.add_annotation(
                x=sign['time'],
                y=0.5,
                text="DIST",
                showarrow=True,
                arrowcolor='red',
                row=4, col=1
            )

        fig.update_layout(
            template='plotly_dark',
            height=1000,
            showlegend=False,
            xaxis_rangeslider_visible=False
        )

        return fig

    def create_volume_profile_chart(self, vp_data):
        """Create volume profile visualization"""
        if not vp_data or 'profile' not in vp_data:
            return None

        profile = vp_data['profile']
        prices = list(profile.keys())
        volumes = list(profile.values())

        fig = go.Figure()

        # Volume profile bars
        fig.add_trace(go.Bar(
            x=volumes,
            y=prices,
            orientation='h',
            marker_color='rgba(156,39,176,0.5)',
            name='Volume Profile'
        ))

        # POC line
        fig.add_hline(
            y=vp_data['poc']['price'],
            line_color='#9c27b0',
            line_width=3,
            annotation_text=f"POC: ${vp_data['poc']['price']:.2f}"
        )

        # Value Area
        fig.add_hrect(
            y0=vp_data['val'],
            y1=vp_data['vah'],
            fillcolor='rgba(156,39,176,0.2)',
            line_width=0,
            annotation_text=f"Value Area ({vp_data['value_area_pct']:.1f}%)",
            annotation_position="right"
        )

        # High volume nodes
        for node in vp_data.get('high_volume_nodes', []):
            fig.add_hline(
                y=node,
                line_color='green',
                line_width=1,
                line_dash='dot'
            )

        # Low volume nodes
        for node in vp_data.get('low_volume_nodes', []):
            fig.add_hline(
                y=node,
                line_color='red',
                line_width=1,
                line_dash='dot'
            )

        fig.update_layout(
            template='plotly_dark',
            height=600,
            title="Volume Profile Analysis",
            xaxis_title="Volume",
            yaxis_title="Price"
        )

        return fig

class EnhancedSMCDashboard:
    def __init__(self):
        self.config = self.load_config()
        self.symbol_timeframes = scan_parquet_structure(PARQUET_DATA_DIR)
        self.available_symbols = list(self.symbol_timeframes.keys())
        self.wyckoff_module = EnhancedWyckoffAnalysis()

    def load_config(self):
        """Load visualization configuration."""
        return {
            "visualization": {
                "chart": {"background": "rgba(17, 24, 39, 1)", "grid_color": "rgba(255,255,255,0.1)"},
                "smc_colors": {
                    "bullish_ob": "rgba(0, 208, 132, 0.2)", "bullish_ob_border": "#00d084",
                    "bearish_ob": "rgba(255, 56, 96, 0.2)", "bearish_ob_border": "#ff3860",
                    "fvg_bull": "rgba(0, 200, 255, 0.15)", "fvg_bull_border": "#00c8ff",
                    "fvg_bear": "rgba(255, 150, 50, 0.15)", "fvg_bear_border": "#ff9632",
                }
            }
        }

    def run(self):
        st.set_page_config(page_title="ZANFLOW Dashboard", page_icon="🎯", layout="wide")
        st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

        # Sidebar with symbol, timeframe, and bars slider
        with st.sidebar:
            st.title("📊 Dashboard Controls")

            if self.available_symbols:
                selected_symbol = st.selectbox("Select Symbol", self.available_symbols, index=0)
                tfs = self.symbol_timeframes.get(selected_symbol, [])
                selected_tf = st.selectbox("Select Timeframe", tfs, index=0) if tfs else None
                bars_slider = st.slider("Bars to Display", 100, 5000, 500, step=100)
                st.info(f"📁 Data Dir: `{PARQUET_DATA_DIR}`")
            else:
                st.error("No parquet data found")
                selected_symbol = selected_tf = None
                bars_slider = 500

        self.bars_slider = bars_slider

        self.render_header()

        if selected_symbol and selected_tf:
            # Load data for selected symbol/timeframe
            df, smc_results, wyckoff_results, filename = load_and_process_data(
                selected_symbol, selected_tf, PARQUET_DATA_DIR
            )

            if df is not None and smc_results is not None and wyckoff_results is not None:
                st.success(f"✅ Loaded: {filename}")
                self.render_main_content(df, smc_results, wyckoff_results, selected_symbol)
            else:
                st.warning(f"Could not load data for {selected_symbol} {selected_tf}")
        else:
            st.warning("Please check your parquet data directory/configuration")

    def render_header(self):
        st.markdown("""
        <div class="dashboard-header">
            <h1 class="dashboard-title">🎯 ZANFLOW v12 Dashboard</h1>
            <p class="dashboard-subtitle">Institutional-Grade SMC & Wyckoff Analysis</p>
        </div>
        """, unsafe_allow_html=True)

    def render_main_content(self, df, smc_results, wyckoff_results, symbol):
        tab1, tab2, tab3 = st.tabs(["📊 Advanced Chart", "🏦 SMC Analysis", "🎭 Wyckoff Analysis"])

        with tab1:
            self.render_advanced_chart(df, smc_results, wyckoff_results, symbol)

        with tab2:
            self.render_smc_analysis_panel(smc_results)

        with tab3:
            self.render_wyckoff_analysis(df, wyckoff_results)

    def render_advanced_chart(self, df, smc_results, wyckoff_results, symbol):
        """Render the main chart with both SMC and Wyckoff overlays"""
        st.markdown(f"<div class='analysis-card'><div class='card-header'><span class='card-icon'>📈</span>Advanced Price Action - {symbol}</div></div>", unsafe_allow_html=True)

        # Limit data for plotting performance
        plot_df = df.tail(self.bars_slider)

        fig = self.create_advanced_chart(plot_df, smc_results, wyckoff_results)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})

    def create_advanced_chart(self, df, smc_results, wyckoff_results):
        """Create the main chart with all elements"""
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.8, 0.2])

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'],
            name='Price', increasing_line_color='#00d084', decreasing_line_color='#ff3860'
        ), row=1, col=1)

        # SMC elements
        self.add_order_blocks(fig, smc_results['order_blocks'], df.index.min(), df.index.max())
        self.add_fair_value_gaps(fig, smc_results['fair_value_gaps'], df.index.min(), df.index.max())
        
        # Add key Wyckoff events
        for event in wyckoff_results.get('events', [])[:5]:
            event_time = pd.to_datetime(event['time'])
            if df.index.min() <= event_time <= df.index.max():
                fig.add_annotation(
                    x=event_time,
                    y=event['price'],
                    text=event['type'],
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor='#9c27b0',
                    font=dict(color='#9c27b0', size=10)
                )
        
        # Volume
        colors = ['#00d084' if row['close'] >= row['open'] else '#ff3860' for index, row in df.iterrows()]
        fig.add_trace(go.Bar(x=df.index, y=df['volume'], name='Volume', marker_color=colors, opacity=0.7), row=2, col=1)

        fig.update_layout(
            template='plotly_dark', height=800, showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20),
            plot_bgcolor=self.config['visualization']['chart']['background'],
            paper_bgcolor=self.config['visualization']['chart']['background'],
            xaxis_rangeslider_visible=False
        )
        return fig

    def add_order_blocks(self, fig, order_blocks, view_start, view_end):
        """Draws order blocks found by the analyzer."""
        for ob in order_blocks:
            ob_time = pd.to_datetime(ob['time'])
            if view_start <= ob_time <= view_end:
                color_config = self.config['visualization']['smc_colors']
                if ob['type'] == 'bullish':
                    fill_color = color_config['bullish_ob']
                    line_color = color_config['bullish_ob_border']
                else:
                    fill_color = color_config['bearish_ob']
                    line_color = color_config['bearish_ob_border']

                fig.add_shape(
                    type="rect",
                    x0=ob_time,
                    x1=ob_time + pd.Timedelta(minutes=15),
                    y0=ob['start'],
                    y1=ob['end'],
                    fillcolor=fill_color,
                    line=dict(color=line_color, width=1),
                    layer='below'
                )

    def add_fair_value_gaps(self, fig, fvgs, view_start, view_end):
        """Draws fair value gaps found by the analyzer."""
        for fvg in fvgs:
            fvg_time = pd.to_datetime(fvg['time'])
            if not fvg['filled'] and view_start <= fvg_time <= view_end:
                color_config = self.config['visualization']['smc_colors']
                if fvg['type'] == 'bullish':
                    fill_color = color_config['fvg_bull']
                else:
                    fill_color = color_config['fvg_bear']
                
                fig.add_shape(
                    type="rect",
                    x0=fvg_time,
                    x1=fvg_time + pd.Timedelta(hours=2),
                    y0=fvg['bottom'],
                    y1=fvg['top'],
                    fillcolor=fill_color,
                    line=dict(width=0),
                    layer='below'
                )
    
    def render_smc_analysis_panel(self, smc_results):
        """Render SMC analysis panel"""
        st.markdown("<div class='analysis-card'><div class='card-header'><span class='card-icon'>🏦</span>SMC Analysis Summary</div></div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📦 Order Blocks")
            ob_df = pd.DataFrame(smc_results['order_blocks'])
            if not ob_df.empty:
                ob_df = ob_df.sort_values('time', ascending=False).head(20)
                st.dataframe(
                    ob_df[['time', 'type', 'start', 'end', 'strength']].round(2),
                    use_container_width=True
                )
            else:
                st.write("No significant order blocks detected.")

        with col2:
            st.subheader("🔲 Fair Value Gaps")
            fvg_df = pd.DataFrame(smc_results['fair_value_gaps'])
            if not fvg_df.empty:
                fvg_df = fvg_df.sort_values('time', ascending=False).head(20)
                st.dataframe(
                    fvg_df[['time', 'type', 'top', 'bottom', 'size', 'filled']].round(2),
                    use_container_width=True
                )
            else:
                st.write("No significant fair value gaps detected.")
    
    def render_wyckoff_analysis(self, df, wyckoff_results):
        """Render complete Wyckoff analysis tab"""
        st.markdown("<div class='analysis-card'><div class='card-header'><span class='card-icon'>🎭</span>Wyckoff Method Analysis</div></div>", unsafe_allow_html=True)
        
        # Current Phase
        phase = wyckoff_results.get('current_phase', 'Unknown')
        phase_class = f"phase-{phase.lower().replace(' ', '-')}"
        
        st.markdown(f"""
        <div class="wyckoff-phase {phase_class}">
            Current Phase: {phase}
        </div>
        """, unsafe_allow_html=True)
        
        # Create columns for analysis
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📍 Wyckoff Events")
            events = wyckoff_results.get('events', [])
            if events:
                for event in events[:10]:
                    event_time = pd.to_datetime(event['time'])
                    st.markdown(f"**{event['type']}** - {event['description']}")
                    st.caption(f"Price: ${event['price']:.2f} | {event_time.strftime('%Y-%m-%d %H:%M')}")
            else:
                st.write("No significant Wyckoff events detected.")
        
        with col2:
            st.subheader("🔄 Springs & Upthrusts")
            spring_upthrust = wyckoff_results.get('spring_upthrust', [])
            if spring_upthrust:
                for event in spring_upthrust[:5]:
                    event_type = event['type']
                    icon = "🟢" if event_type == 'Spring' else "🔴"
                    st.markdown(f"{icon} **{event_type}** at ${event.get('spring_low', event.get('upthrust_high', 0)):.2f}")
                    if event.get('reversal_confirmed'):
                        st.caption("✅ Reversal confirmed")
            else:
                st.write("No springs or upthrusts detected.")
        
        with col3:
            st.subheader("💪 SOS/SOW")
            sos_sow = wyckoff_results.get('sos_sow', [])
            if sos_sow:
                for signal in sos_sow[:5]:
                    signal_type = signal['type']
                    icon = "📈" if signal_type == 'SOS' else "📉"
                    strength = signal.get('strength', signal.get('weakness', 0))
                    st.markdown(f"{icon} **{signal_type}** - Strength: {strength:.2%}")
                    st.caption(f"Price: ${signal['price']:.2f}")
            else:
                st.write("No SOS/SOW signals detected.")
        
        # Wyckoff Chart
        st.subheader("📊 Wyckoff Price Action Analysis")
        wyckoff_fig = self.wyckoff_module.create_wyckoff_chart(df.tail(500), wyckoff_results)
        st.plotly_chart(wyckoff_fig, use_container_width=True)
        
        # Volume Profile Analysis
        st.subheader("📊 Volume Profile")
        volume_profile = wyckoff_results.get('volume_profile', {})
        
        if volume_profile:
            vp_col1, vp_col2 = st.columns([2, 1])
            
            with vp_col1:
                vp_fig = self.wyckoff_module.create_volume_profile_chart(volume_profile)
                if vp_fig:
                    st.plotly_chart(vp_fig, use_container_width=True)
            
            with vp_col2:
                st.markdown("### Key Levels")
                st.metric("POC", f"${volume_profile['poc']['price']:.2f}")
                st.metric("VAH", f"${volume_profile['vah']:.2f}")
                st.metric("VAL", f"${volume_profile['val']:.2f}")
                st.metric("Value Area %", f"{volume_profile['value_area_pct']:.1f}%")
                
                if 'high_volume_nodes' in volume_profile:
                    st.markdown("**High Volume Nodes:**")
                    for node in volume_profile['high_volume_nodes'][:3]:
                        st.caption(f"${node:.2f}")
        
        # Trading Ranges
        st.subheader("📏 Trading Ranges")
        ranges = wyckoff_results.get('trading_ranges', [])
        if ranges:
            range_df = pd.DataFrame(ranges)
            if not range_df.empty:
                st.dataframe(
                    range_df[['start_time', 'end_time', 'high', 'low', 'duration']].head(10),
                    use_container_width=True
                )
        
        # Volume Patterns
        st.subheader("📊 Volume Pattern Analysis")
        volume_analysis = wyckoff_results.get('volume_analysis', {})
        
        pattern_col1, pattern_col2, pattern_col3 = st.columns(3)
        
        with pattern_col1:
            st.markdown("**Effort vs Result**")
            effort_patterns = volume_analysis.get('effort_vs_result', [])
            if effort_patterns:
                for pattern in effort_patterns[:3]:
                    st.caption(f"High Effort/Low Result at {pattern['time']}")
                    st.caption(f"Volume Ratio: {pattern['volume_ratio']:.2f}")
        
        with pattern_col2:
            st.markdown("**Volume Dry-ups**")
            dryups = volume_analysis.get('volume_dry_up', [])
            if dryups:
                st.metric("Recent Dry-ups", len(dryups))
                st.caption("Potential end of move")
        
        with pattern_col3:
            st.markdown("**Volume Surges**")
            surges = volume_analysis.get('volume_surge', [])
            if surges:
                st.metric("Recent Surges", len(surges))
                for surge in surges[:2]:
                    st.caption(f"{surge['price_direction']} @ {surge['volume_ratio']:.1f}x")


if __name__ == "__main__":
    dashboard = EnhancedSMCDashboard()
    dashboard.run()