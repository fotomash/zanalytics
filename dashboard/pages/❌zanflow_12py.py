import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import ta
import json
from pathlib import Path
import toml
import yaml
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set page config for full dashboard experience
st.set_page_config(
    page_title="ZANFLOW v12 Advanced Trading Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    theme="dark"
)

# Custom CSS for dark theme optimization
st.markdown("""
<style>
    .main > div {
        padding: 1rem;
    }
    .stPlotlyChart {
        background-color: transparent;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .signal-green {
        color: #00ff88;
        font-weight: bold;
    }
    .signal-red {
        color: #ff4444;
        font-weight: bold;
    }
    .signal-yellow {
        color: #ffaa00;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class ZANFLOWDataProcessor:
    """Core data processing engine for ZANFLOW v12 dashboard"""
    
    def __init__(self):
        self.data = None
        self.enriched_data = None
        
    def load_mt5_data(self, file_path: str) -> pd.DataFrame:
        """Load MT5 tab-separated data"""
        try:
            # Handle different MT5 export formats
            for sep in ['\t', ',', ';']:
                try:
                    df = pd.read_csv(file_path, sep=sep)
                    if len(df.columns) > 1:  # Valid separation found
                        break
                except:
                    continue
            
            # Standardize column names
            column_mapping = {
                'time': 'timestamp',
                'Time': 'timestamp',
                'Date': 'date',
                'Time': 'time',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'Tick Volume': 'tick_volume'
            }
            
            df = df.rename(columns=column_mapping)
            
            # Handle timestamp creation
            if 'timestamp' not in df.columns:
                if 'date' in df.columns and 'time' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str))
                else:
                    st.error("No valid timestamp column found")
                    return None
            else:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Ensure required columns exist
            required_cols = ['open', 'high', 'low', 'close']
            for col in required_cols:
                if col not in df.columns:
                    st.error(f"Missing required column: {col}")
                    return None
            
            # Add volume if missing
            if 'volume' not in df.columns:
                df['volume'] = df.get('tick_volume', 100)
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None
    
    def enrich_with_zanflow_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all ZANFLOW v12 indicators and analysis"""
        enriched = df.copy()
        
        # Basic technical indicators
        enriched = self.add_basic_indicators(enriched)
        
        # Wyckoff phase analysis
        enriched = self.add_wyckoff_analysis(enriched)
        
        # SMC structure detection
        enriched = self.add_smc_analysis(enriched)
        
        # Microstructure analysis
        enriched = self.add_microstructure_analysis(enriched)
        
        # Liquidity analysis
        enriched = self.add_liquidity_analysis(enriched)
        
        # Session analysis
        enriched = self.add_session_analysis(enriched)
        
        return enriched
    
    def add_basic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic technical indicators"""
        # Moving averages
        for period in [20, 50, 100, 200]:
            df[f'sma_{period}'] = ta.trend.sma_indicator(df['close'], window=period)
            df[f'ema_{period}'] = ta.trend.ema_indicator(df['close'], window=period)
        
        # ATR
        df['atr_14'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
        
        # RSI
        df['rsi_14'] = ta.momentum.rsi(df['close'], window=14)
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_hist'] = macd.macd_diff()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        
        return df
    
    def add_wyckoff_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Wyckoff phase detection"""
        # Simplified Wyckoff phase detection
        df['price_change'] = df['close'].diff()
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Phase detection logic (simplified)
        conditions = [
            (df['volume_ratio'] > 1.5) & (df['price_change'] > 0),  # Accumulation
            (df['volume_ratio'] > 1.5) & (df['price_change'] < 0),  # Distribution
            (df['volume_ratio'] < 0.8),  # Consolidation
        ]
        
        choices = ['Accumulation', 'Distribution', 'Consolidation']
        df['wyckoff_phase'] = np.select(conditions, choices, default='Neutral')
        
        return df
    
    def add_smc_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Smart Money Concepts analysis"""
        # Structure breaks
        df['swing_high'] = self.detect_swing_points(df['high'], True)
        df['swing_low'] = self.detect_swing_points(df['low'], False)
        
        # CHoCH and BoS detection
        df['choch'] = self.detect_structure_breaks(df, 'choch')
        df['bos'] = self.detect_structure_breaks(df, 'bos')
        
        # Fair Value Gaps
        df['fvg_bullish'], df['fvg_bearish'] = self.detect_fvg(df)
        
        # Order blocks
        df['ob_bullish'], df['ob_bearish'] = self.detect_order_blocks(df)
        
        return df
    
    def add_microstructure_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add microstructure analysis"""
        # Price velocity
        df['price_velocity'] = df['close'].diff() / df['close'].shift(1) * 100
        
        # Volatility
        df['volatility'] = df['high'] - df['low']
        df['volatility_pct'] = df['volatility'] / df['close'] * 100
        
        # Body size analysis
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
        df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
        
        # Manipulation markers
        df['spread_spike'] = self.detect_spread_spikes(df)
        df['stop_hunt'] = self.detect_stop_hunts(df)
        df['liquidity_sweep'] = self.detect_liquidity_sweeps(df)
        
        return df
    
    def add_liquidity_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add liquidity analysis"""
        # Volume profile
        df['volume_profile'] = self.calculate_volume_profile(df)
        
        # Liquidity levels
        df['resistance_level'] = df['high'].rolling(20).max()
        df['support_level'] = df['low'].rolling(20).min()
        
        # Inducement analysis
        df['high_inducement'] = self.detect_inducement(df, 'high')
        df['low_inducement'] = self.detect_inducement(df, 'low')
        
        return df
    
    def add_session_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add session-based analysis"""
        df['hour'] = df['timestamp'].dt.hour
        df['session'] = df['hour'].apply(self.classify_session)
        
        # Session highs/lows
        df['session_high'] = df.groupby(df['timestamp'].dt.date)['high'].transform('max')
        df['session_low'] = df.groupby(df['timestamp'].dt.date)['low'].transform('min')
        
        return df
    
    def detect_swing_points(self, series: pd.Series, is_high: bool, window: int = 5) -> pd.Series:
        """Detect swing highs/lows"""
        swings = pd.Series(False, index=series.index)
        
        for i in range(window, len(series) - window):
            if is_high:
                if series.iloc[i] == series.iloc[i-window:i+window+1].max():
                    swings.iloc[i] = True
            else:
                if series.iloc[i] == series.iloc[i-window:i+window+1].min():
                    swings.iloc[i] = True
        
        return swings
    
    def detect_structure_breaks(self, df: pd.DataFrame, break_type: str) -> pd.Series:
        """Detect CHoCH and BoS"""
        # Simplified structure break detection
        breaks = pd.Series(False, index=df.index)
        
        # This would be more complex in production
        high_breaks = df['close'] > df['high'].shift(1).rolling(10).max()
        low_breaks = df['close'] < df['low'].shift(1).rolling(10).min()
        
        if break_type == 'choch':
            breaks = high_breaks | low_breaks
        else:  # bos
            breaks = (high_breaks & (df['close'].shift(1) < df['close'].shift(2))) | \
                    (low_breaks & (df['close'].shift(1) > df['close'].shift(2)))
        
        return breaks
    
    def detect_fvg(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Detect Fair Value Gaps"""
        bullish_fvg = pd.Series(False, index=df.index)
        bearish_fvg = pd.Series(False, index=df.index)
        
        for i in range(2, len(df)):
            # Bullish FVG: previous low > current high
            if i > 0 and df['low'].iloc[i-1] > df['high'].iloc[i+1]:
                bullish_fvg.iloc[i] = True
            
            # Bearish FVG: previous high < current low
            if i > 0 and df['high'].iloc[i-1] < df['low'].iloc[i+1]:
                bearish_fvg.iloc[i] = True
        
        return bullish_fvg, bearish_fvg
    
    def detect_order_blocks(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Detect Order Blocks"""
        # Simplified order block detection
        bullish_ob = df['volume'] > df['volume'].rolling(20).mean() * 1.5
        bearish_ob = df['volume'] > df['volume'].rolling(20).mean() * 1.5
        
        return bullish_ob & (df['close'] > df['open']), bearish_ob & (df['close'] < df['open'])
    
    def detect_spread_spikes(self, df: pd.DataFrame) -> pd.Series:
        """Detect spread spikes"""
        spread = df['high'] - df['low']
        avg_spread = spread.rolling(20).mean()
        return spread > avg_spread * 2
    
    def detect_stop_hunts(self, df: pd.DataFrame) -> pd.Series:
        """Detect stop hunt patterns"""
        # High with long upper shadow followed by reversal
        long_upper_shadow = df['upper_shadow'] > df['body_size'] * 2
        reversal = df['close'] < df['open']
        return long_upper_shadow & reversal
    
    def detect_liquidity_sweeps(self, df: pd.DataFrame) -> pd.Series:
        """Detect liquidity sweeps"""
        recent_high = df['high'].rolling(20).max()
        recent_low = df['low'].rolling(20).min()
        
        high_sweep = df['high'] > recent_high.shift(1)
        low_sweep = df['low'] < recent_low.shift(1)
        
        return high_sweep | low_sweep
    
    def calculate_volume_profile(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volume profile"""
        return df['volume'].rolling(20).sum()
    
    def detect_inducement(self, df: pd.DataFrame, level_type: str) -> pd.Series:
        """Detect inducement patterns"""
        if level_type == 'high':
            return df['high'] > df['high'].rolling(10).max().shift(1)
        else:
            return df['low'] < df['low'].rolling(10).min().shift(1)
    
    def classify_session(self, hour: int) -> str:
        """Classify trading session"""
        if 0 <= hour < 8:
            return 'Asian'
        elif 8 <= hour < 16:
            return 'London'
        elif 16 <= hour < 24:
            return 'New York'
        else:
            return 'Off Hours'

class ZANFLOWDashboard:
    """Main dashboard class"""
    
    def __init__(self):
        self.processor = ZANFLOWDataProcessor()
        self.setup_sidebar()
    
    def setup_sidebar(self):
        """Setup sidebar controls"""
        st.sidebar.title("🚀 ZANFLOW v12 Controls")
        
        # File upload
        st.sidebar.subheader("📊 Data Input")
        uploaded_file = st.sidebar.file_uploader(
            "Upload MT5 CSV/TSV File",
            type=['csv', 'tsv', 'txt'],
            help="Upload your MT5 exported data file"
        )
        
        if uploaded_file:
            # Load and process data
            with st.spinner("Loading and enriching data..."):
                df = self.processor.load_mt5_data(uploaded_file)
                if df is not None:
                    self.data = self.processor.enrich_with_zanflow_indicators(df)
                    st.sidebar.success(f"✅ Loaded {len(self.data)} rows")
        
        # Analysis controls
        st.sidebar.subheader("🔍 Analysis Controls")
        
        self.timeframe = st.sidebar.selectbox(
            "Primary Timeframe",
            ["1m", "5m", "15m", "1h", "4h", "1d"],
            index=2
        )
        
        self.lookback_periods = st.sidebar.slider(
            "Lookback Periods",
            min_value=50,
            max_value=500,
            value=200,
            step=50
        )
        
        # Strategy filters
        st.sidebar.subheader("⚡ Strategy Filters")
        
        self.show_wyckoff = st.sidebar.checkbox("Show Wyckoff Analysis", True)
        self.show_smc = st.sidebar.checkbox("Show SMC Structures", True)
        self.show_liquidity = st.sidebar.checkbox("Show Liquidity Analysis", True)
        self.show_microstructure = st.sidebar.checkbox("Show Microstructure", True)
        
        # Alert thresholds
        st.sidebar.subheader("🚨 Alert Thresholds")
        
        self.volatility_threshold = st.sidebar.slider(
            "Volatility Spike Threshold (%)",
            min_value=0.1,
            max_value=5.0,
            value=1.0,
            step=0.1
        )
        
        self.volume_threshold = st.sidebar.slider(
            "Volume Spike Multiplier",
            min_value=1.5,
            max_value=5.0,
            value=2.0,
            step=0.1
        )
        
        # Session timing
        st.sidebar.subheader("🕒 Session Timing")
        
        self.london_session = st.sidebar.checkbox("London Session Focus", True)
        self.ny_session = st.sidebar.checkbox("NY Session Focus", True)
        self.asian_session = st.sidebar.checkbox("Asian Session Focus", False)
    
    def render_header(self):
        """Render dashboard header"""
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.title("📈 ZANFLOW v12 Advanced Trading Dashboard")
            st.markdown("*Comprehensive Wyckoff & SMC Analysis with Microstructure Intelligence*")
        
        with col2:
            if hasattr(self, 'data') and self.data is not None:
                st.metric(
                    "Data Points",
                    f"{len(self.data):,}",
                    delta=f"Last: {self.data['timestamp'].max().strftime('%H:%M')}"
                )
        
        with col3:
            if hasattr(self, 'data') and self.data is not None:
                current_price = self.data['close'].iloc[-1]
                price_change = self.data['close'].iloc[-1] - self.data['close'].iloc[-2]
                st.metric(
                    "Current Price",
                    f"{current_price:.4f}",
                    delta=f"{price_change:+.4f}"
                )
    
    def render_main_chart(self):
        """Render main price chart with all markers"""
        if not hasattr(self, 'data') or self.data is None:
            st.warning("Please upload data to view charts")
            return
        
        # Limit data for performance
        display_data = self.data.tail(self.lookback_periods)
        
        fig = go.Figure()
        
        # Main price line
        fig.add_trace(go.Scatter(
            x=display_data['timestamp'],
            y=display_data['close'],
            name='Price',
            line=dict(color='white', width=2),
            showlegend=True
        ))
        
        # Add manipulation markers
        if self.show_microstructure:
            # Spread spikes
            spread_spikes = display_data[display_data['spread_spike']]
            if not spread_spikes.empty:
                fig.add_trace(go.Scatter(
                    x=spread_spikes['timestamp'],
                    y=spread_spikes['high'],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up',
                        size=8,
                        color='magenta',
                        line=dict(width=1, color='white')
                    ),
                    name='Spread Spike',
                    showlegend=True
                ))
            
            # Stop hunts
            stop_hunts = display_data[display_data['stop_hunt']]
            if not stop_hunts.empty:
                fig.add_trace(go.Scatter(
                    x=stop_hunts['timestamp'],
                    y=stop_hunts['high'],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-down',
                        size=8,
                        color='yellow',
                        line=dict(width=1, color='white')
                    ),
                    name='Stop Hunt',
                    showlegend=True
                ))
            
            # Liquidity sweeps
            sweeps = display_data[display_data['liquidity_sweep']]
            if not sweeps.empty:
                fig.add_trace(go.Scatter(
                    x=sweeps['timestamp'],
                    y=sweeps['close'],
                    mode='markers',
                    marker=dict(
                        symbol='diamond',
                        size=10,
                        color='cyan',
                        line=dict(width=1, color='white')
                    ),
                    name='Liquidity Sweep',
                    showlegend=True
                ))
        
        # SMC structures
        if self.show_smc:
            # CHoCH
            choch_points = display_data[display_data['choch']]
            if not choch_points.empty:
                fig.add_trace(go.Scatter(
                    x=choch_points['timestamp'],
                    y=choch_points['close'],
                    mode='markers',
                    marker=dict(
                        symbol='star',
                        size=12,
                        color='lime',
                        line=dict(width=2, color='white')
                    ),
                    name='CHoCH',
                    showlegend=True
                ))
            
            # BoS
            bos_points = display_data[display_data['bos']]
            if not bos_points.empty:
                fig.add_trace(go.Scatter(
                    x=bos_points['timestamp'],
                    y=bos_points['close'],
                    mode='markers',
                    marker=dict(
                        symbol='hexagon',
                        size=12,
                        color='orange',
                        line=dict(width=2, color='white')
                    ),
                    name='BoS',
                    showlegend=True
                ))
        
        # Wyckoff phases
        if self.show_wyckoff:
            phase_colors = {
                'Accumulation': 'green',
                'Distribution': 'red',
                'Consolidation': 'gray',
                'Neutral': 'blue'
            }
            
            for phase, color in phase_colors.items():
                phase_data = display_data[display_data['wyckoff_phase'] == phase]
                if not phase_data.empty:
                    fig.add_trace(go.Scatter(
                        x=phase_data['timestamp'],
                        y=phase_data['close'],
                        mode='markers',
                        marker=dict(
                            symbol='square',
                            size=6,
                            color=color,
                            opacity=0.6
                        ),
                        name=f'Wyckoff: {phase}',
                        showlegend=True
                    ))
        
        fig.update_layout(
            title=f"Price Action with Microstructure Manipulation Markers ({self.timeframe})",
            template='plotly_dark',
            height=500,
            xaxis_title="Time",
            yaxis_title="Price",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_analysis_panels(self):
        """Render analysis panels"""
        if not hasattr(self, 'data') or self.data is None:
            return
        
        display_data = self.data.tail(self.lookback_periods)
        
        # Create 2x2 subplot layout
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Spread Analysis (Volatility %)',
                'Volume Analysis',
                'Microstructure Metrics',
                'Price Distribution Analysis'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Spread/Volatility analysis
        fig.add_trace(
            go.Scatter(
                x=display_data['timestamp'],
                y=display_data['volatility_pct'],
                name='Volatility %',
                line=dict(color='yellow')
            ),
            row=1, col=1
        )
        
        # Volume analysis
        fig.add_trace(
            go.Bar(
                x=display_data['timestamp'],
                y=display_data['volume'],
                name='Volume',
                marker_color='cyan',
                opacity=0.7
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=display_data['timestamp'],
                y=display_data['volume_ratio'],
                name='Volume Ratio',
                line=dict(color='orange')
            ),
            row=1, col=2
        )
        
        # Microstructure metrics
        fig.add_trace(
            go.Scatter(
                x=display_data['timestamp'],
                y=display_data['price_velocity'],
                name='Price Velocity',
                line=dict(color='purple')
            ),
            row=2, col=1
        )
        
        # Price distribution
        fig.add_trace(
            go.Histogram(
                x=display_data['price_change'],
                name='Price Changes',
                marker_color='lightblue',
                opacity=0.7,
                nbinsx=50
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            template='plotly_dark',
            height=600,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_wyckoff_smc_panels(self):
        """Render Wyckoff and SMC analysis panels"""
        if not hasattr(self, 'data') or self.data is None:
            return
        
        display_data = self.data.tail(self.lookback_periods)
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(
                'Wyckoff Phase Analysis',
                'Smart Money Concepts - Fair Value Gaps & Order Blocks'
            ),
            vertical_spacing=0.1
        )
        
        # Wyckoff phase plot
        fig.add_trace(
            go.Scatter(
                x=display_data['timestamp'],
                y=display_data['close'],
                name='Price',
                line=dict(color='white')
            ),
            row=1, col=1
        )
        
        # Color background by Wyckoff phase
        phase_colors = {
            'Accumulation': 'rgba(0, 255, 0, 0.1)',
            'Distribution': 'rgba(255, 0, 0, 0.1)',
            'Consolidation': 'rgba(128, 128, 128, 0.1)',
            'Neutral': 'rgba(0, 0, 255, 0.05)'
        }
        
        # SMC structures plot
        fig.add_trace(
            go.Scatter(
                x=display_data['timestamp'],
                y=display_data['close'],
                name='Price',
                line=dict(color='white'),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Add FVG markers
        bullish_fvg = display_data[display_data['fvg_bullish']]
        bearish_fvg = display_data[display_data['fvg_bearish']]
        
        if not bullish_fvg.empty:
            fig.add_trace(
                go.Scatter(
                    x=bullish_fvg['timestamp'],
                    y=bullish_fvg['low'],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up',
                        size=8,
                        color='lime'
                    ),
                    name='Bullish FVG'
                ),
                row=2, col=1
            )
        
        if not bearish_fvg.empty:
            fig.add_trace(
                go.Scatter(
                    x=bearish_fvg['timestamp'],
                    y=bearish_fvg['high'],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-down',
                        size=8,
                        color='red'
                    ),
                    name='Bearish FVG'
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            template='plotly_dark',
            height=700,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_inducement_analysis(self):
        """Render inducement analysis panel"""
        if not hasattr(self, 'data') or self.data is None:
            return
        
        display_data = self.data.tail(self.lookback_periods)
        
        fig = go.Figure()
        
        # Base price
        fig.add_trace(go.Scatter(
            x=display_data['timestamp'],
            y=display_data['close'],
            name='Price',
            line=dict(color='white', width=1),
            opacity=0.8
        ))
        
        # Support and resistance levels
        fig.add_trace(go.Scatter(
            x=display_data['timestamp'],
            y=display_data['resistance_level'],
            name='Resistance',
            line=dict(color='red', dash='dot'),
            opacity=0.6
        ))
        
        fig.add_trace(go.Scatter(
            x=display_data['timestamp'],
            y=display_data['support_level'],
            name='Support',
            line=dict(color='green', dash='dot'),
            opacity=0.6
        ))
        
        # Inducement markers
        high_inducement = display_data[display_data['high_inducement']]
        low_inducement = display_data[display_data['low_inducement']]
        
        if not high_inducement.empty:
            fig.add_trace(go.Scatter(
                x=high_inducement['timestamp'],
                y=high_inducement['high'],
                mode='markers',
                marker=dict(
                    symbol='triangle-down',
                    size=12,
                    color='red',
                    line=dict(width=2, color='white')
                ),
                name='High Inducement'
            ))
        
        if not low_inducement.empty:
            fig.add_trace(go.Scatter(
                x=low_inducement['timestamp'],
                y=low_inducement['low'],
                mode='markers',
                marker=dict(
                    symbol='triangle-up',
                    size=12,
                    color='green',
                    line=dict(width=2, color='white')
                ),
                name='Low Inducement'
            ))
        
        fig.update_layout(
            title="Inducement Analysis - Institutional Order Flow",
            template='plotly_dark',
            height=400,
            xaxis_title="Time",
            yaxis_title="Price",
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_signal_dashboard(self):
        """Render trading signals dashboard"""
        if not hasattr(self, 'data') or self.data is None:
            return
        
        st.subheader("🎯 Trading Signals & Confluence")
        
        latest_data = self.data.tail(10)
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Signal strength metrics
        with col1:
            wyckoff_signal = latest_data['wyckoff_phase'].iloc[-1]
            signal_color = 'green' if wyckoff_signal == 'Accumulation' else 'red' if wyckoff_signal == 'Distribution' else 'yellow'
            st.markdown(f"**Wyckoff Phase:** <span class='signal-{signal_color}'>{wyckoff_signal}</span>", unsafe_allow_html=True)
        
        with col2:
            smc_signals = (latest_data['choch'].sum() + latest_data['bos'].sum())
            st.markdown(f"**SMC Breaks:** <span class='signal-yellow'>{smc_signals}</span>", unsafe_allow_html=True)
        
        with col3:
            liquidity_events = latest_data['liquidity_sweep'].sum()
            st.markdown(f"**Liquidity Events:** <span class='signal-yellow'>{liquidity_events}</span>", unsafe_allow_html=True)
        
        with col4:
            micro_events = (latest_data['spread_spike'].sum() + latest_data['stop_hunt'].sum())
            st.markdown(f"**Micro Events:** <span class='signal-yellow'>{micro_events}</span>", unsafe_allow_html=True)
        
        # Generate trade signals
        self.generate_trade_signals(latest_data)
    
    def generate_trade_signals(self, data: pd.DataFrame):
        """Generate ZANFLOW trade signals"""
        st.subheader("📊 ZANFLOW v12 Trade Analysis")
        
        # Signal confluence analysis
        confluence_score = 0
        signals = []
        
        # Wyckoff confluence
        if data['wyckoff_phase'].iloc[-1] == 'Accumulation':
            confluence_score += 30
            signals.append("✅ Wyckoff Accumulation Phase Active")
        elif data['wyckoff_phase'].iloc[-1] == 'Distribution':
            confluence_score += 30
            signals.append("🔻 Wyckoff Distribution Phase Active")
        
        # SMC confluence
        if data['choch'].iloc[-1] or data['bos'].iloc[-1]:
            confluence_score += 25
            signals.append("✅ SMC Structure Break Confirmed")
        
        # Liquidity confluence
        if data['liquidity_sweep'].iloc[-1]:
            confluence_score += 20
            signals.append("⚡ Liquidity Sweep Detected")
        
        # Microstructure confluence
        if data['spread_spike'].iloc[-1] or data['stop_hunt'].iloc[-1]:
            confluence_score += 15
            signals.append("🎯 Microstructure Event Active")
        
        # Volume confluence
        if data['volume_ratio'].iloc[-1] > 1.5:
            confluence_score += 10
            signals.append("📈 Volume Spike Confirmed")
        
        # Display confluence score
        col1, col2 = st.columns([1, 2])
        
        with col1:
            score_color = 'green' if confluence_score >= 70 else 'yellow' if confluence_score >= 40 else 'red'
            st.markdown(f"### Confluence Score: <span class='signal-{score_color}'>{confluence_score}%</span>", unsafe_allow_html=True)
        
        with col2:
            for signal in signals:
                st.markdown(f"• {signal}")
        
        # Trade recommendation
        if confluence_score >= 70:
            st.success("🚀 HIGH PROBABILITY SETUP - Consider entry with proper risk management")
        elif confluence_score >= 40:
            st.warning("⚠️ MODERATE SETUP - Wait for additional confirmation")
        else:
            st.error("❌ LOW PROBABILITY - Avoid entry, wait for better setup")
    
    def run(self):
        """Main dashboard execution"""
        self.render_header()
        
        if hasattr(self, 'data') and self.data is not None:
            # Main chart
            self.render_main_chart()
            
            # Analysis panels
            self.render_analysis_panels()
            
            # Wyckoff and SMC panels
            if self.show_wyckoff or self.show_smc:
                self.render_wyckoff_smc_panels()
            
            # Inducement analysis
            if self.show_liquidity:
                self.render_inducement_analysis()
            
            # Signal dashboard
            self.render_signal_dashboard()
            
            # Export functionality
            st.subheader("📁 Export & Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📊 Export Analysis"):
                    csv = self.data.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"zanflow_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if st.button("📈 Generate Report"):
                    # Generate summary report
                    report = self.generate_summary_report()
                    st.download_button(
                        label="Download Report",
                        data=report,
                        file_name=f"zanflow_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
            
            with col3:
                if st.button("🔄 Refresh Analysis"):
                    st.experimental_rerun()
        
        else:
            st.info("👆 Please upload your MT5 data file using the sidebar to begin analysis")
            
            # Show sample data format
            st.subheader("📋 Expected Data Format")
            sample_data = pd.DataFrame({
                'timestamp': ['2025-06-26 08:00:00', '2025-06-26 08:01:00'],
                'open': [3334.86, 3335.21],
                'high': [3335.63, 3335.46],
                'low': [3333.24, 3334.50],
                'close': [3333.82, 3335.44],
                'volume': [476, 628]
            })
            st.dataframe(sample_data)
    
    def generate_summary_report(self) -> str:
        """Generate markdown summary report"""
        if not hasattr(self, 'data') or self.data is None:
            return "No data available for report generation."
        
        latest = self.data.iloc[-1]
        
        report = f"""# ZANFLOW v12 Trading Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Market Overview
- **Current Price:** {latest['close']:.4f}
- **24h Change:** {(latest['close'] - self.data['close'].iloc[-24]):.4f} ({((latest['close'] - self.data['close'].iloc[-24])/self.data['close'].iloc[-24]*100):+.2f}%)
- **Volatility (ATR):** {latest['atr_14']:.4f}
- **Volume:** {latest['volume']:,.0f}

## Wyckoff Analysis
- **Current Phase:** {latest['wyckoff_phase']}
- **Volume Ratio:** {latest['volume_ratio']:.2f}x average

## SMC Analysis
- **Structure Breaks (Last 20):** CHoCH: {self.data['choch'].tail(20).sum()}, BoS: {self.data['bos'].tail(20).sum()}
- **FVG Activity:** Bullish: {self.data['fvg_bullish'].tail(20).sum()}, Bearish: {self.data['fvg_bearish'].tail(20).sum()}

## Microstructure Events
- **Liquidity Sweeps (Last 20):** {self.data['liquidity_sweep'].tail(20).sum()}
- **Stop Hunts (Last 20):** {self.data['stop_hunt'].tail(20).sum()}
- **Spread Spikes (Last 20):** {self.data['spread_spike'].tail(20).sum()}

## Trading Recommendations
Based on current confluence analysis, the market shows:
{self.get_market_bias()}

---
*Report generated by ZANFLOW v12 Advanced Trading Dashboard*
"""
        return report
    
    def get_market_bias(self) -> str:
        """Get current market bias"""
        latest = self.data.tail(10)
        
        if latest['wyckoff_phase'].iloc[-1] == 'Accumulation':
            return "• Bullish bias supported by Wyckoff accumulation phase"
        elif latest['wyckoff_phase'].iloc[-1] == 'Distribution':
            return "• Bearish bias indicated by Wyckoff distribution phase"
        else:
            return "• Neutral bias - await clearer directional signals"

# Run the dashboard
if __name__ == "__main__":
    dashboard = ZANFLOWDashboard()
    dashboard.run()

    # zanflow_ultimate_dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
import os
import warnings
from typing import Dict, List, Tuple

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')

# --- Page Configuration (Must be the first Streamlit command) ---
st.set_page_config(
    page_title="Zanflow Institutional Dashboard",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for a Professional Look (Merged from your files) ---
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .metric-card {
        background: #1E293B; /* Dark blue-gray */
        border-left: 5px solid #4A90E2; /* Accent color */
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        color: #E2E8F0; /* Light text */
    }
    .metric-card h4 {
        color: #94A3B8; /* Lighter gray for titles */
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    .metric-card .value {
        font-size: 1.5rem;
        font-weight: bold;
    }
    .signal-green { color: #34D399; }
    .signal-red { color: #F87171; }
    .signal-yellow { color: #FBBF24; }
    .confluence-score {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


class ZANFLOWAnalyzer:
    """
    Core analysis engine for ZANFLOW v12.
    Handles data loading, enrichment, and strategy logic implementation.
    """

    @st.cache_data
    def load_and_process_data(_self, file_path: str) -> pd.DataFrame:
        """
        Loads and processes MT5 data, handling tab separation and column mapping.
        Specifically uses 'tickvol' for volume analysis as requested.
        """
        try:
            # MT5 files are tab-separated
            df = pd.read_csv(file_path, sep='\t')

            # Standardize column names to handle variations
            column_mapping = {
                'time': 'timestamp', 'Time': 'timestamp',
                'open': 'open', 'Open': 'open',
                'high': 'high', 'High': 'high',
                'low': 'low', 'Low': 'low',
                'close': 'close', 'Close': 'close',
                'tickvol': 'volume', 'Tick Volume': 'volume', # IMPORTANT: Map tickvol to 'volume'
                'spread': 'spread', 'Spread': 'spread'
            }
            df = df.rename(columns=lambda c: column_mapping.get(c.strip(), c.strip()))

            # Ensure timestamp is a datetime object
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            else:
                st.error("Timestamp column not found. Please ensure your file has a 'time' or 'timestamp' column.")
                return None

            # Ensure all required OHLCV columns are present
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    st.error(f"Missing required column: '{col}'. Please check your MT5 export.")
                    return None

            # Sort by timestamp to ensure correct chronological order
            df = df.sort_values('timestamp').reset_index(drop=True)
            return df

        except Exception as e:
            st.error(f"Error loading data: {e}. Please ensure it's a valid tab-separated MT5 export.")
            return None

    def enrich_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies all ZANFLOW v12 analytical layers to the dataframe.
        """
        if df is None:
            return None
        
        df = self.add_technical_indicators(df)
        df = self.add_wyckoff_analysis(df)
        df = self.add_smc_analysis(df)
        df = self.add_microstructure_analysis(df)
        df = self.add_session_analysis(df)
        
        return df

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df['atr_14'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
        df['rsi_14'] = ta.momentum.rsi(df['close'], window=14)
        return df

    def add_wyckoff_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = (df['volume'] / df['volume_sma_20']).fillna(1.0)
        
        conditions = [
            (df['volume_ratio'] > 1.8) & (df['close'] > df['open']),
            (df['volume_ratio'] > 1.8) & (df['close'] < df['open']),
            (df['volume_ratio'] < 0.7)
        ]
        choices = ['Accumulation', 'Distribution', 'Consolidation']
        df['wyckoff_phase'] = np.select(conditions, choices, default='Neutral')
        
        df['spring_utad'] = self.detect_springs_utads(df)
        return df

    def detect_springs_utads(self, df: pd.DataFrame) -> pd.Series:
        is_spring_utad = pd.Series(False, index=df.index)
        for i in range(20, len(df)):
            is_volume_spike = df['volume_ratio'].iloc[i] > 2.0
            body_size = abs(df['close'].iloc[i] - df['open'].iloc[i])
            upper_shadow = df['high'].iloc[i] - max(df['open'].iloc[i], df['close'].iloc[i])
            lower_shadow = min(df['open'].iloc[i], df['close'].iloc[i]) - df['low'].iloc[i]
            
            is_spring = (df['low'].iloc[i] < df['low'].iloc[i-20:i].min()) and (lower_shadow > body_size * 1.5)
            is_utad = (df['high'].iloc[i] > df['high'].iloc[i-20:i].max()) and (upper_shadow > body_size * 1.5)
            
            if is_volume_spike and (is_spring or is_utad):
                is_spring_utad.iloc[i] = True
        return is_spring_utad

    def add_smc_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        df['choch_bos'] = self.detect_structure_breaks(df)
        df['fvg_bullish'], df['fvg_bearish'] = self.detect_fvg(df)
        df['liquidity_sweep'] = self.detect_liquidity_sweeps(df)
        return df

    def detect_structure_breaks(self, df: pd.DataFrame) -> pd.Series:
        breaks = pd.Series(False, index=df.index)
        for i in range(20, len(df)):
            recent_high = df['high'].iloc[i-20:i].max()
            recent_low = df['low'].iloc[i-20:i].min()
            if df['close'].iloc[i] > recent_high or df['close'].iloc[i] < recent_low:
                breaks.iloc[i] = True
        return breaks

    def detect_fvg(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        bullish_fvg = (df['low'] > df['high'].shift(2)) & (df['close'] > df['open'])
        bearish_fvg = (df['high'] < df['low'].shift(2)) & (df['close'] < df['open'])
        return bullish_fvg, bearish_fvg

    def detect_liquidity_sweeps(self, df: pd.DataFrame) -> pd.Series:
        high_sweep = df['high'] > df['high'].rolling(20).max().shift(1)
        low_sweep = df['low'] < df['low'].rolling(20).min().shift(1)
        return high_sweep | low_sweep

    def add_microstructure_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        df['tick_volatility'] = df['close'].rolling(20).std()
        df['spread_pips'] = df.get('spread', 0)
        df['spread_spike'] = df['spread_pips'] > df['spread_pips'].rolling(20).mean() * 2.0
        return df

    def add_session_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        df['hour_utc'] = df['timestamp'].dt.hour
        conditions = [
            (df['hour_utc'] >= 0) & (df['hour_utc'] < 7),
            (df['hour_utc'] >= 7) & (df['hour_utc'] < 12),
            (df['hour_utc'] >= 12) & (df['hour_utc'] < 16),
            (df['hour_utc'] >= 16)
        ]
        choices = ['Asian', 'London', 'New York', 'Post-Session']
        df['session'] = np.select(conditions, choices, default='Off')
        return df

def display_main_chart(df: pd.DataFrame, lookback: int):
    display_data = df.tail(lookback)
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=display_data['timestamp'],
        open=display_data['open'],
        high=display_data['high'],
        low=display_data['low'],
        close=display_data['close'],
        name='Price'
    ))

    markers = {
        'Liquidity Sweep': (display_data[display_data['liquidity_sweep']], 'cyan', 'diamond'),
        'Spring/UTAD': (display_data[display_data['spring_utad']], 'lime', 'star'),
        'CHoCH/BoS': (display_data[display_data['choch_bos']], 'yellow', 'cross'),
    }
    for name, (data, color, symbol) in markers.items():
        if not data.empty:
            fig.add_trace(go.Scatter(x=data['timestamp'], y=data['high'] * 1.0005, mode='markers', 
                                     marker=dict(color=color, symbol=symbol, size=10), name=name))

    fig.update_layout(title="Price Action with ZANFLOW v12 Event Markers", template='plotly_dark', height=600, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

def display_indicator_panels(df: pd.DataFrame, lookback: int):
    display_data = df.tail(lookback)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.6, 0.4])
    
    fig.add_trace(go.Bar(x=display_data['timestamp'], y=display_data['volume'], name='Tick Volume', marker_color='cyan', opacity=0.5), row=1, col=1)
    fig.add_trace(go.Scatter(x=display_data['timestamp'], y=display_data['volume_sma_20'], name='Volume SMA(20)', line=dict(color='orange')), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=display_data['timestamp'], y=display_data['rsi_14'], name='RSI', line=dict(color='purple')), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", row=2, col=1, line_color="red", opacity=0.5)
    fig.add_hline(y=30, line_dash="dot", row=2, col=1, line_color="green", opacity=0.5)

    fig.update_layout(template='plotly_dark', height=400, showlegend=False)
    fig.update_yaxes(title_text="Volume", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

def calculate_confluence_score(df: pd.DataFrame) -> Tuple[int, List[str]]:
    if df is None or len(df) < 20:
        return 0, []
    
    latest = df.iloc[-1]
    score = 0
    signals = []

    if latest['wyckoff_phase'] in ['Accumulation', 'Distribution']:
        score += 30
        signals.append(f"✅ Wyckoff Phase: {latest['wyckoff_phase']}")
    
    if latest['spring_utad']:
        score += 25
        signals.append("⚡️ Wyckoff Spring/UTAD Event")

    if latest['choch_bos']:
        score += 20
        signals.append("✅ Market Structure Break (CHoCH/BoS)")

    if latest['liquidity_sweep']:
        score += 15
        signals.append("🎯 Liquidity Sweep Detected")

    if latest['volume_ratio'] > 1.8:
        score += 10
        signals.append("📈 High Volume Confirmation")

    return min(score, 100), signals

def main():
    st.title("ZANFLOW v12 Institutional Dashboard")
    
    analyzer = ZANFLOWAnalyzer()
    
    with st.sidebar:
        st.header("⚙️ Controls")
        
        # Auto-scan for data files
        data_dir = "./data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        available_files = [f for f in os.listdir(data_dir) if f.endswith(('.csv', '.txt'))]
        
        if not available_files:
            st.warning("No data files found in ./data folder.")
            st.info("Please place your tab-separated MT5 files in a folder named 'data' in the same directory as this script.")
            return
            
        selected_file = st.selectbox("Select Data File", available_files)
        lookback = st.slider("Lookback Periods", 50, 1000, 300, 50)

    if selected_file:
        file_path = os.path.join(data_dir, selected_file)
        data = analyzer.load_and_process_data(file_path)
        
        if data is not None:
            with st.spinner("Applying ZANFLOW v12 analysis..."):
                enriched_data = analyzer.enrich_data(data)
            
            score, signals = calculate_confluence_score(enriched_data)
            
            # --- Dashboard Display ---
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Confluence Score</h4>
                    <div class="value confluence-score signal-{'green' if score >= 70 else 'yellow' if score >= 40 else 'red'}">{score}%</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Current Price</h4>
                    <div class="value">{enriched_data['close'].iloc[-1]:.3f}</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Wyckoff Phase</h4>
                    <div class="value">{enriched_data['wyckoff_phase'].iloc[-1]}</div>
                </div>
                """, unsafe_allow_html=True)

            if signals:
                st.subheader("Key Signals Detected:")
                for s in signals:
                    st.markdown(f"- {s}")
            
            # --- Tabs for Organized View ---
            tab_main, tab_indicators = st.tabs(["📈 Main Chart & Events", "📊 Indicators"])

            with tab_main:
                display_main_chart(enriched_data, lookback)
            
            with tab_indicators:
                display_indicator_panels(enriched_data, lookback)

if __name__ == "__main__":
    main()