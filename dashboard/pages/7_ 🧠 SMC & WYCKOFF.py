# enhanced_smc_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import yaml
from pathlib import Path
from datetime import datetime, timedelta, time
import pytz
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import threading
import time as time_module

# Custom CSS for beautiful UI
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
    
    /* Metric styling */
    .metric-container {
        background: rgba(255, 193, 59, 0.1);
        border: 1px solid rgba(255, 193, 59, 0.3);
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #ffc13b;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #f5f0e1;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Level badges */
    .level-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    
    .badge-bullish {
        background: rgba(0, 208, 132, 0.2);
        border: 1px solid #00d084;
        color: #00ff00;
    }
    
    .badge-bearish {
        background: rgba(255, 56, 96, 0.2);
        border: 1px solid #ff3860;
        color: #ff6b6b;
    }
    
    .badge-neutral {
        background: rgba(255, 221, 87, 0.2);
        border: 1px solid #ffdd57;
        color: #ffdd57;
    }
    
    /* Table styling */
    .styled-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 1rem;
    }
    
    .styled-table th {
        background: rgba(30, 61, 89, 0.5);
        color: #ffc13b;
        padding: 0.8rem;
        text-align: left;
        border-bottom: 2px solid #ffc13b;
    }
    
    .styled-table td {
        padding: 0.6rem;
        border-bottom: 1px solid rgba(255, 193, 59, 0.2);
        color: #f5f0e1;
    }
    
    /* Alert styling */
    .alert-poi {
        background: linear-gradient(135deg, rgba(255, 193, 59, 0.2) 0%, rgba(255, 193, 59, 0.1) 100%);
        border-left: 4px solid #ffc13b;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .dashboard-title {
            font-size: 2rem;
        }
        .card-header {
            font-size: 1.1rem;
        }
    }
</style>
"""

@dataclass
class PointOfInterest:
    """Enhanced POI with visual properties"""
    price: float
    type: str
    timestamp: datetime
    strength: float
    touches: int
    description: str
    color: str
    icon: str
    mitigated: bool = False
    entry_zone: Tuple[float, float] = None

class EnhancedSMCDashboard:
    """Enhanced SMC Dashboard with beautiful visualizations"""
    
    def __init__(self):
        self.load_config()
        self.scan_available_symbols()
        self.setup_state()

    def load_config(self):
        """Load enhanced configuration"""
        config_path = Path(st.secrets.get("processed_data_directory", ".")) / "enhanced_smc_dashboard_config.yaml"
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            st.warning(f"Configuration file not found at {config_path}. Using default settings.")
            self.config = {}

        # Inject fallback defaults for visualization keys
        if not self.config:
            self.config = {}

        default_config = {
            "visualization": {
                "chart": {
                    "background": "rgba(0,0,0,0)",
                    "grid_color": "rgba(255,255,255,0.1)"
                },
                "smc_colors": {
                    "bullish_ob": "rgba(0,208,132,0.30)",
                    "bullish_ob_border": "#00d084",
                    "fvg_bull": "rgba(0,255,255,0.20)"
                },
                "markers": {
                    "poi_size": 14
                }
            }
        }

        def deep_merge(d1, d2):
            for k, v in d2.items():
                if isinstance(v, dict):
                    d1[k] = deep_merge(d1.get(k, {}), v)
                else:
                    d1.setdefault(k, v)
            return d1

        self.config = deep_merge(self.config, default_config)

    def scan_available_symbols(self):
        """Scan ./data for CSV files and extract unique symbols"""
        data_dir = Path("./data")
        symbols = set()
        if data_dir.exists() and data_dir.is_dir():
            for file in data_dir.glob("*.csv"):
                symbol = file.stem
                symbols.add(symbol)
        self.available_symbols = sorted(symbols)

    def setup_state(self):
        """Initialize session state"""
        if 'theme' not in st.session_state:
            st.session_state.theme = 'dark'
        if 'selected_timeframe' not in st.session_state:
            st.session_state.selected_timeframe = '15m'
        if 'show_poi_labels' not in st.session_state:
            st.session_state.show_poi_labels = True

    def run(self):
        """Main dashboard entry point"""
        st.set_page_config(
            page_title="SMC Wyckoff Dashboard",
            page_icon="🎯",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Apply custom CSS
        st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

        # Symbol selector
        selected_symbol = st.selectbox("📈 Select Trading Symbol", self.available_symbols)
        st.session_state.selected_symbol = selected_symbol

        # Header
        self.render_header()

        # Main content area
        self.render_main_content()

    def render_header(self):
        """Render beautiful header"""
        st.markdown("""
            <div class="dashboard-header">
                <h1 class="dashboard-title">🎯 SMC Wyckoff Dashboard</h1>
                <p class="dashboard-subtitle">Advanced Market Structure & Wyckoff Analysis</p>
            </div>
        """, unsafe_allow_html=True)

        # Quick stats row
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            self.render_metric("Market Bias", "BULLISH", "badge-bullish")
        with col2:
            self.render_metric("Active POIs", "12", "badge-neutral")
        # Remove Win Rate metric (col3)
        with col3:
            pass
        with col4:
            self.render_metric("Wyckoff Phase", "MARKUP", "badge-bullish")
        with col5:
            self.render_metric("Risk Score", "LOW", "badge-bullish")
            
    def render_metric(self, label: str, value: str, badge_class: str):
        """Render a styled metric"""
        st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">{label}</div>
                <div class="metric-value {badge_class}">{value}</div>
            </div>
        """, unsafe_allow_html=True)
        
    def render_main_content(self):
        """Render main content area"""
        # Create tabs with icons
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Advanced Price Action",
            "🏦 Smart Money Concepts",
            "🎭 Wyckoff Analysis",
            "⚡ Points of Interest",
            "🎯 Trade Setups"
        ])
        
        with tab1:
            self.render_advanced_chart()
            
        with tab2:
            self.render_smc_analysis()
            
        with tab3:
            self.render_wyckoff_analysis()
            
        with tab4:
            self.render_poi_analysis()
            
        with tab5:
            self.render_trade_setups()
            
    def render_advanced_chart(self):
        """Render advanced price action chart with all markups"""
        st.markdown("""
            <div class="analysis-card">
                <div class="card-header">
                    <span class="card-icon">📈</span>
                    Advanced Price Action Analysis
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Create sophisticated chart
        fig = self.create_advanced_chart()
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
        
    def create_advanced_chart(self) -> go.Figure:
        """Create advanced chart with all SMC markups"""
        # Create figure with subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=[0.7, 0.15, 0.15],
            subplot_titles=('', '', ''),
            specs=[[{"secondary_y": True}],
                   [{"secondary_y": False}],
                   [{"secondary_y": False}]]
        )

        # Load real data from CSV based on selected symbol
        symbol_file = Path("./data") / f"{st.session_state.selected_symbol}.csv"
        if not symbol_file.exists():
            st.error(f"Data file for {st.session_state.selected_symbol} not found.")
            return go.Figure()

        # Robust loading logic to handle CSVs, supporting MT5 tab-separated exports
        try:
            # Try reading as tab-separated (MT5) first
            df = pd.read_csv(symbol_file, sep="\t")
            if "timestamp" not in df.columns and {"Date", "Time"} <= set(df.columns):
                df["timestamp"] = pd.to_datetime(df["Date"] + " " + df["Time"])
                df.drop(columns=["Date", "Time"], inplace=True)
        except Exception:
            # Fallback to comma-separated
            try:
                df = pd.read_csv(symbol_file)
                if "timestamp" not in df.columns:
                    st.warning(f"'timestamp' column not found in {symbol_file.name}.")
                    return go.Figure()
            except Exception as e:
                st.warning(f"Failed to load CSV: {e}")
                return go.Figure()

        # Normalize column names for MT5 conventions
        df.rename(columns=lambda x: x.strip().lower(), inplace=True)
        df.rename(columns={"tickvol": "volume"}, inplace=True)

        # Sanity check for required columns
        required = ["timestamp", "open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            st.warning(f"Missing columns {missing} in {symbol_file.name}")
            return go.Figure()

        # Parse timestamp to datetime if not already
        if not np.issubdtype(df["timestamp"].dtype, np.datetime64):
            try:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            except Exception:
                st.warning("Could not parse 'timestamp' column as datetime.")
                return go.Figure()

        df.sort_values("timestamp", inplace=True)
        df = df.tail(200)  # Limit to recent 200 rows for plotting

        # Robust check for empty or missing data
        if df.empty:
            st.warning("No valid data available to plot for the selected symbol.")
            return go.Figure()

        # Only proceed if DataFrame is valid and non-empty
        dates = df["timestamp"]
        prices = df["close"]

        # Main candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=dates,
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name='Price',
                increasing_line_color='#00d084',
                decreasing_line_color='#ff3860'
            ),
            row=1, col=1, secondary_y=False
        )

        # Add Order Blocks
        self.add_order_blocks(fig, dates, prices)

        # Add Fair Value Gaps
        self.add_fair_value_gaps(fig, dates, prices)

        # Add Liquidity Zones
        self.add_liquidity_zones(fig, dates, prices)

        # Add POIs
        self.add_points_of_interest(fig, dates, prices)

        # Add Market Structure
        self.add_market_structure(fig, dates, prices)

        # Volume with color coding
        colors = ['#00d084' if df["close"].iloc[i] >= df["open"].iloc[i] else '#ff3860' for i in range(len(df))]
        fig.add_trace(
            go.Bar(
                x=dates,
                y=df["volume"],
                name='Volume',
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )

        # RSI calculation
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(50)  # fill NaNs

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=rsi,
                name='RSI',
                line=dict(color='#ffc13b', width=2)
            ),
            row=3, col=1
        )

        # Add RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

        # Update layout
        fig.update_layout(
            template='plotly_dark',
            height=800,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(0,0,0,0.5)"
            ),
            margin=dict(l=0, r=0, t=30, b=0),
            plot_bgcolor=self.config['visualization']['chart']['background'],
            paper_bgcolor=self.config['visualization']['chart']['background'],
            font=dict(color='#f5f0e1')
        )

        # Update axes
        fig.update_xaxes(
            gridcolor=self.config['visualization']['chart']['grid_color'],
            showgrid=True,
            zeroline=False
        )
        fig.update_yaxes(
            gridcolor=self.config['visualization']['chart']['grid_color'],
            showgrid=True,
            zeroline=False
        )

        return fig
        
    def add_order_blocks(self, fig: go.Figure, dates: pd.DatetimeIndex, prices: np.ndarray):
        """Add order blocks to chart"""
        # Bullish Order Block
        ob_start = 50
        ob_end = 70
        ob_high = prices[ob_start:ob_end].max() + 1
        ob_low = prices[ob_start:ob_end].min() - 0.5
        
        fig.add_shape(
            type="rect",
            x0=dates[ob_start], x1=dates[ob_end],
            y0=ob_low, y1=ob_high,
            fillcolor=self.config['visualization']['smc_colors']['bullish_ob'],
            line=dict(color=self.config['visualization']['smc_colors']['bullish_ob_border'], width=2),
            row=1, col=1
        )
        
        # Add annotation
        fig.add_annotation(
            x=dates[ob_start + 10],
            y=ob_high + 0.5,
            text="Bullish OB",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#00ff00",
            font=dict(color="#00ff00", size=12),
            row=1, col=1
        )
        
    def add_fair_value_gaps(self, fig: go.Figure, dates: pd.DatetimeIndex, prices: np.ndarray):
        """Add Fair Value Gaps to chart"""
        # FVG example
        fvg_index = 100
        fvg_high = prices[fvg_index] + 0.8
        fvg_low = prices[fvg_index] - 0.8
        
        fig.add_shape(
            type="rect",
            x0=dates[fvg_index - 5], x1=dates[fvg_index + 5],
            y0=fvg_low, y1=fvg_high,
            fillcolor=self.config['visualization']['smc_colors']['fvg_bull'],
            line=dict(width=0),
            row=1, col=1
        )
        
        # FVG label
        fig.add_annotation(
            x=dates[fvg_index],
            y=fvg_high + 0.3,
            text="FVG",
            showarrow=False,
            font=dict(color="#00ffff", size=10),
            bgcolor="rgba(0,0,0,0.5)",
            row=1, col=1
        )
        
    def add_liquidity_zones(self, fig: go.Figure, dates: pd.DatetimeIndex, prices: np.ndarray):
        """Add liquidity zones"""
        # Equal highs liquidity
        liq_level = prices.max() + 1
        
        fig.add_hline(
            y=liq_level,
            line_dash="dot",
            line_color="#ffff00",
            line_width=2,
            annotation_text="Liquidity Pool",
            annotation_position="right",
            row=1, col=1
        )
        
        # Add liquidity grab visualization
        grab_index = 150
        fig.add_trace(
            go.Scatter(
                x=[dates[grab_index]],
                y=[liq_level + 0.5],
                mode='markers',
                marker=dict(
                    symbol='triangle-down',
                    size=20,
                    color='#ffff00',
                    line=dict(width=2, color='#ffffff')
                ),
                name='Liquidity Grab',
                showlegend=True
            ),
            row=1, col=1
        )
        
    def add_points_of_interest(self, fig: go.Figure, dates: pd.DatetimeIndex, prices: np.ndarray):
        """Add Points of Interest"""
        # Define POIs
        pois = [
            PointOfInterest(
                price=prices[30],
                type='Demand Zone',
                timestamp=dates[30],
                strength=0.9,
                touches=3,
                description='Strong Demand Zone',
                color='#00ff00',
                icon='⬆️'
            ),
            PointOfInterest(
                price=prices[120],
                type='Supply Zone',
                timestamp=dates[120],
                strength=0.85,
                touches=4,
                description='Key Supply Zone',
                color='#ff0000',
                icon='⬇️'
            )
        ]
        
        for poi in pois:
            # Add POI marker
            fig.add_trace(
                go.Scatter(
                    x=[poi.timestamp],
                    y=[poi.price],
                    mode='markers+text',
                    marker=dict(
                        symbol='hexagram',
                        size=self.config['visualization']['markers']['poi_size'],
                        color=poi.color,
                        line=dict(width=2, color='white')
                    ),
                    text=[poi.icon],
                    textposition='top center',
                    name=poi.type,
                    showlegend=True
                ),
                row=1, col=1
            )
            
            # Add POI zone
            zone_width = 0.5
            fig.add_shape(
                type="rect",
                x0=dates.iloc[0], x1=dates.iloc[-1],
                y0=poi.price - zone_width, y1=poi.price + zone_width,
                fillcolor=poi.color,
                opacity=0.1,
                line=dict(width=0),
                row=1, col=1
            )
            
    def add_market_structure(self, fig: go.Figure, dates: pd.DatetimeIndex, prices: np.ndarray):
        """Add market structure elements"""
        # Add swing highs and lows
        swing_highs = [20, 60, 100, 140, 180]
        swing_lows = [10, 40, 80, 120, 160]
        
        # Plot swing highs
        for idx in swing_highs:
            if idx < len(dates):
                fig.add_trace(
                    go.Scatter(
                        x=[dates[idx]],
                        y=[prices[idx] + 1],
                        mode='markers+text',
                        marker=dict(
                            symbol='triangle-down',
                            size=15,
                            color='#ff3860'
                        ),
                        text=['HH' if idx > 60 else 'LH'],
                        textposition='top center',
                        textfont=dict(color='#ff3860', size=10),
                        showlegend=False
                    ),
                    row=1, col=1
                )
                
        # Plot swing lows
        for idx in swing_lows:
            if idx < len(dates):
                fig.add_trace(
                    go.Scatter(
                        x=[dates[idx]],
                        y=[prices[idx] - 1],
                        mode='markers+text',
                        marker=dict(
                            symbol='triangle-up',
                            size=15,
                            color='#00d084'
                        ),
                        text=['HL' if idx > 40 else 'LL'],
                        textposition='bottom center',
                        textfont=dict(color='#00d084', size=10),
                        showlegend=False
                    ),
                    row=1, col=1
                )
                
        # Add trend lines
        if len(swing_lows) > 1:
            fig.add_trace(
                go.Scatter(
                    x=[dates[swing_lows[0]], dates[swing_lows[-1]]],
                    y=[prices[swing_lows[0]], prices[swing_lows[-1]]],
                    mode='lines',
                    line=dict(color='#00d084', width=2, dash='dash'),
                    name='Support Trend',
                    showlegend=True
                ),
                row=1, col=1
            )
            
    def render_smc_analysis(self):
        """Render Smart Money Concepts analysis section"""
        st.markdown("""
            <div class="analysis-card">
                <div class="card-header">
                    <span class="card-icon">🏦</span>
                    Smart Money Concepts Analysis
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Create columns for different SMC concepts
        col1, col2 = st.columns(2)
        
        with col1:
            self.render_order_blocks_panel()
            self.render_fair_value_gaps_panel()
            
        with col2:
            self.render_liquidity_analysis_panel()
            self.render_inducement_panel()
            
    def render_order_blocks_panel(self):
        """Render order blocks panel"""
        st.markdown("""
            <div class="analysis-card">
                <h4 style="color: #ffc13b;">📦 Order Blocks</h4>
            </div>
        """, unsafe_allow_html=True)

        try:
            ob_path = f"./processed/{st.session_state.selected_symbol}_order_blocks.csv"
            ob_df = pd.read_csv(ob_path)
            # Ensure 'timestamp' is parsed if present
            if 'timestamp' in ob_df.columns:
                ob_df['timestamp'] = pd.to_datetime(ob_df['timestamp'])
            else:
                st.warning(f"'timestamp' column not found in {ob_path}. Time-based sorting or display may be affected.")
            st.dataframe(
                ob_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Type": st.column_config.TextColumn("Type", width="medium"),
                    "Status": st.column_config.TextColumn("Status", help="Order block status")
                }
            )
        except Exception as e:
            st.warning(f"Failed to load Order Blocks: {e}")
        
    def render_fair_value_gaps_panel(self):
        """Render FVG panel"""
        st.markdown("""
            <div class="analysis-card">
                <h4 style="color: #ffc13b;">🌊 Fair Value Gaps</h4>
            </div>
        """, unsafe_allow_html=True)
        
        # FVG metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Active FVGs", "5", "2 new")
        with col2:
            st.metric("8AM FVGs", "2", "1 filled")
        with col3:
            st.metric("Avg Size", "12.5 pips", "+2.3")
            
        # Alert for new FVG
        st.markdown("""
            <div class="alert-poi">
                <strong>🚨 New FVG Detected!</strong><br>
                Bullish FVG formed at $2,655.00 - $2,657.50<br>
                <small>Created 15 minutes ago</small>
            </div>
        """, unsafe_allow_html=True)
        
    def render_liquidity_analysis_panel(self):
        """Render liquidity analysis panel"""
        st.markdown("""
            <div class="analysis-card">
                <h4 style="color: #ffc13b;">💧 Liquidity Analysis</h4>
            </div>
        """, unsafe_allow_html=True)
        
        # Liquidity levels
        liq_levels = {
            'Buy Side': ['$2,685.00', '$2,690.50', '$2,695.00'],
            'Sell Side': ['$2,640.00', '$2,635.50', '$2,630.00'],
            'Strength': ['High', 'Medium', 'High']
        }
        
        # Visual representation
        st.markdown("""
            <div style="background: rgba(255,255,0,0.1); padding: 1rem; border-radius: 8px;">
                <strong>🎯 Next Liquidity Targets:</strong><br>
                ⬆️ Buy Side: $2,685.00 (High probability)<br>
                ⬇️ Sell Side: $2,640.00 (Medium probability)
            </div>
        """, unsafe_allow_html=True)
        
    def render_inducement_panel(self):
        """Render inducement panel"""
        st.markdown("""
            <div class="analysis-card">
                <h4 style="color: #ffc13b;">🎣 Inducement & Sweeps</h4>
            </div>
        """, unsafe_allow_html=True)
        
        # Recent sweeps
        sweep_data = {
            'Time': ['10:30', '09:15', '08:45'],
            'Level': ['$2,680', '$2,645', '$2,690'],
            'Type': ['Buy Stop Hunt', 'Sell Stop Hunt', 'Buy Stop Hunt'],
            'Result': ['Reversed ✅', 'Continued ❌', 'Reversed ✅']
        }
        
        df = pd.DataFrame(sweep_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
    def render_wyckoff_analysis(self):
        """Render Wyckoff analysis section"""
        st.markdown("""
            <div class="analysis-card">
                <div class="card-header">
                    <span class="card-icon">🎭</span>
                    Wyckoff Method Analysis
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Wyckoff phase diagram
        self.render_wyckoff_phase_diagram()
        
        # Volume analysis
        self.render_volume_analysis()
        
    def render_wyckoff_phase_diagram(self):
        """Render Wyckoff phase visualization"""
        col1, col2, col3, col4 = st.columns(4)
        
        phases = [
            ("Accumulation", "🟢", col1, "Phase B"),
            ("Markup", "🔵", col2, "Early"),
            ("Distribution", "🔴", col3, "Not Active"),
            ("Markdown", "🟠", col4, "Not Active")
        ]
        
        for phase, emoji, col, status in phases:
            with col:
                active = "Active" in status or "Phase" in status or "Early" in status
                bg_color = "rgba(0,208,132,0.2)" if active else "rgba(128,128,128,0.1)"
                
                st.markdown(f"""
                    <div style="background: {bg_color}; padding: 1rem; border-radius: 8px; text-align: center;">
                        <div style="font-size: 2rem;">{emoji}</div>
                        <div style="font-weight: bold; color: #ffc13b;">{phase}</div>
                        <div style="font-size: 0.9rem; color: #f5f0e1;">{status}</div>
                    </div>
                """, unsafe_allow_html=True)
                
    def render_volume_analysis(self):
        """Render volume analysis section"""
        st.markdown("""
            <div class="analysis-card" style="margin-top: 2rem;">
                <h4 style="color: #ffc13b;">📊 Volume Spread Analysis</h4>
            </div>
        """, unsafe_allow_html=True)
        
        # Volume metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Effort vs Result", "Low Volume Rally", "⚠️")
        with col2:
            st.metric("Volume Trend", "Decreasing", "📉")
        with col3:
            st.metric("Climactic Action", "None", "➖")
        with col4:
            st.metric("Supply Test", "Passed", "✅")
            
        # Wyckoff events timeline
        st.markdown("""
            <div class="analysis-card" style="margin-top: 1rem;">
                <h5 style="color: #f5f0e1;">Recent Wyckoff Events:</h5>
                <ul style="color: #f5f0e1;">
                    <li>✅ Spring detected at $2,635 (2 hours ago)</li>
                    <li>✅ Test of spring successful (1 hour ago)</li>
                    <li>⏳ Awaiting Sign of Strength (SOS)</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
    def render_poi_analysis(self):
        """Render Points of Interest analysis"""
        st.markdown("""
            <div class="analysis-card">
                <div class="card-header">
                    <span class="card-icon">⚡</span>
                    Points of Interest (POI) Analysis
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # POI map
        self.render_poi_map()
        
        # POI details
        self.render_poi_details()
        
    def render_poi_map(self):
        """Render visual POI map"""
        # Create a simple POI visualization
        fig = go.Figure()
        
        # Current price line
        current_price = 2665
        fig.add_hline(
            y=current_price,
            line_dash="solid",
            line_color="white",
            line_width=2,
            annotation_text=f"Current Price: ${current_price}",
            annotation_position="right"
        )
        
        # POI levels
        pois = [
            {"price": 2680, "type": "Resistance", "strength": "Strong", "color": "#ff3860"},
            {"price": 2672, "type": "Supply", "strength": "Medium", "color": "#ffdd57"},
            {"price": 2658, "type": "Demand", "strength": "Strong", "color": "#00d084"},
            {"price": 2645, "type": "Support", "strength": "Strong", "color": "#00d084"},
        ]
        
        for poi in pois:
            fig.add_shape(
                type="rect",
                x0=0, x1=1,
                y0=poi["price"] - 1, y1=poi["price"] + 1,
                xref="paper",
                fillcolor=poi["color"],
                opacity=0.3,
                line=dict(width=0)
            )
            
            fig.add_annotation(
                x=0.5,
                y=poi["price"],
                text=f"{poi['type']} ({poi['strength']})",
                xref="paper",
                showarrow=False,
                font=dict(color="white", size=12),
                bgcolor=poi["color"],
                borderpad=4
            )
        
        fig.update_layout(
            template='plotly_dark',
            height=400,
            showlegend=False,
            xaxis=dict(visible=False),
            yaxis=dict(
                title="Price Levels",
                range=[2640, 2685],
                tickformat="$,.0f"
            ),
            plot_bgcolor="rgba(0,0,0,0.5)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    def render_poi_details(self):
        """Render POI details table"""
        poi_data = {
            'Level': ['$2,680', '$2,672', '$2,658', '$2,645'],
            'Type': ['Resistance', 'Supply Zone', 'Demand Zone', 'Support'],
            'Touches': [5, 3, 4, 6],
            'Last Test': ['2h ago', '5h ago', '1d ago', '2d ago'],
            'Entry Zone': ['$2,678-2,682', '$2,670-2,674', '$2,656-2,660', '$2,643-2,647'],
            'Risk/Reward': ['1:3.5', '1:2.8', '1:4.2', '1:3.8']
        }
        
        df = pd.DataFrame(poi_data)
        
        # Style the dataframe
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Risk/Reward": st.column_config.TextColumn(
                    "R:R",
                    help="Risk to Reward ratio for this POI"
                ),
                "Entry Zone": st.column_config.TextColumn(
                    "Entry Zone",
                    help="Optimal entry range"
                )
            }
        )
        
    def render_trade_setups(self):
        """Render current trade setups"""
        st.markdown("""
            <div class="analysis-card">
                <div class="card-header">
                    <span class="card-icon">🎯</span>
                    Active Trade Setups
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Load trade setups from CSV file
        try:
            setup_path = f"./processed/{st.session_state.selected_symbol}_trade_setups.csv"
            setups_df = pd.read_csv(setup_path)
            for _, row in setups_df.iterrows():
                targets = [t.strip() for t in row["Targets"].split(",")]
                setup = {
                    "name": row["Name"],
                    "entry": row["Entry"],
                    "stop": row["Stop"],
                    "targets": targets,
                    "rr": row["RR"],
                    "confidence": row["Confidence"],
                    "status": row["Status"],
                    "color": row.get("Color", "#ffc13b")
                }
                self.render_trade_setup_card(setup)
        except Exception as e:
            st.warning(f"Failed to load trade setups: {e}")
            
    def render_trade_setup_card(self, setup: Dict):
        """Render individual trade setup card"""
        status_color = "#00d084" if setup["status"] == "Active" else "#ffdd57"
        
        st.markdown(f"""
            <div class="analysis-card" style="border-left: 4px solid {setup['color']};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h4 style="color: {setup['color']}; margin: 0;">{setup['name']}</h4>
                    <span class="level-badge" style="background: {status_color}20; border-color: {status_color}; color: {status_color};">
                        {setup['status']}
                    </span>
                </div>
                <div style="margin-top: 1rem; display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                    <div>
                        <strong style="color: #f5f0e1;">Entry Zone:</strong> {setup['entry']}<br>
                        <strong style="color: #f5f0e1;">Stop Loss:</strong> {setup['stop']}<br>
                        <strong style="color: #f5f0e1;">R:R Ratio:</strong> {setup['rr']}
                    </div>
                    <div>
                        <strong style="color: #f5f0e1;">Targets:</strong><br>
                        {'<br>'.join([f"TP{i+1}: {target}" for i, target in enumerate(setup['targets'])])}
                    </div>
                </div>
                <div style="margin-top: 1rem;">
                    <div style="background: rgba(255,193,59,0.1); padding: 0.5rem; border-radius: 4px;">
                        <strong>Confidence:</strong> {setup['confidence']}
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

# Run the dashboard
if __name__ == "__main__":
    dashboard = EnhancedSMCDashboard()
    dashboard.run()