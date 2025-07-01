#!/usr/bin/env python3
"""
NCOS v11 Enhanced Analysis Dashboard
Intelligent Commentary System for Enriched Market Data
Integrates with TOML configuration and JSON analysis files
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import json
import toml
import os
from pathlib import Path
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Optional, Any
import logging

# Configure warnings and logging
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION AND SETUP
# ============================================================================

def load_toml_config():
    """Load configuration from Streamlit secrets (TOML)"""
    try:
        config = {
            'json_dir': st.secrets.get('JSONdir', './data'),
            'parquet_dir': st.secrets.get('PARQUET_DATA_DIR', './data/parquet'),
            'raw_data_dir': st.secrets.get('raw_data_directory', './data/raw'),
            'enriched_data_dir': st.secrets.get('enriched_data', './data/enriched')
        }
        return config
    except Exception as e:
        st.error(f"Error loading TOML configuration: {e}")
        return {
            'json_dir': './data',
            'parquet_dir': './data/parquet', 
            'raw_data_dir': './data/raw',
            'enriched_data_dir': './data/enriched'
        }

def setup_page_config():
    """Configure Streamlit page with professional styling"""
    st.set_page_config(
        page_title="NCOS v11 Enhanced Analysis Dashboard",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for professional styling
    st.markdown("""
        <style>
        .main {
            background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
            color: #ffffff;
        }
        
        .header-style {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            text-align: center;
            box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        }
        
        .metric-card {
            background: rgba(255, 255, 255, 0.1);
            padding: 1rem;
            border-radius: 10px;
            border-left: 4px solid #667eea;
            margin-bottom: 1rem;
            backdrop-filter: blur(10px);
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }
        
        .commentary-box {
            background: linear-gradient(145deg, #1e1e2e, #2a2d47);
            padding: 1.5rem;
            border-radius: 12px;
            margin: 1rem 0;
            border-left: 5px solid #ffd700;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.2);
        }
        
        .signal-alert {
            background: linear-gradient(45deg, #ff6b6b, #ee5a52);
            color: white;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            font-weight: 600;
            text-align: center;
            animation: pulse 2s infinite;
        }
        
        .midas-signal {
            background: linear-gradient(45deg, #ffd700, #ffed4e);
            color: #000;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            font-weight: 600;
            text-align: center;
            box-shadow: 0 4px 15px rgba(255, 215, 0, 0.4);
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.7; }
            100% { opacity: 1; }
        }
        
        .session-indicator {
            display: inline-block;
            padding: 0.3rem 0.8rem;
            border-radius: 15px;
            font-weight: 600;
            margin: 0.2rem;
            font-size: 0.8rem;
        }
        
        .london-killzone { background: #ff6b6b; color: white; }
        .us-session { background: #4ecdc4; color: white; }
        .asian-session { background: #45b7d1; color: white; }
        </style>
    """, unsafe_allow_html=True)

# ============================================================================
# DATA LOADING AND DISCOVERY
# ============================================================================

@st.cache_data(ttl=300)  # Cache for 5 minutes
def discover_json_files(json_dir: str) -> Dict[str, List[str]]:
    """Discover available JSON files in the configured directory"""
    discovered_files = {
        'comprehensive_files': [],
        'symbol_files': {},
        'metadata_files': [],
        'symbols': set(),
        'timeframes': set()
    }
    
    json_path = Path(json_dir)
    if not json_path.exists():
        st.warning(f"JSON directory not found: {json_dir}")
        return discovered_files
    
    # Scan for JSON files
    for file_path in json_path.rglob('*.json'):
        file_name = file_path.name
        discovered_files['comprehensive_files'].append(str(file_path))
        
        # Extract symbol and timeframe from filename
        if '_comprehensive.json' in file_name:
            symbol = file_name.replace('_comprehensive.json', '').upper()
            discovered_files['symbols'].add(symbol)
        
        # Check for metadata
        if 'metadata' in file_name.lower() or 'analysis' in file_name.lower():
            discovered_files['metadata_files'].append(str(file_path))
    
    # Convert sets to sorted lists
    discovered_files['symbols'] = sorted(list(discovered_files['symbols']))
    
    return discovered_files

def load_json_analysis(file_path: str) -> Optional[Dict]:
    """Load and validate JSON analysis file"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if not data:
            st.warning(f"Empty JSON file: {file_path}")
            return None
        
        return data
        
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON format in {file_path}: {e}")
        return None
    except Exception as e:
        st.error(f"Error loading {file_path}: {e}")
        return None

def extract_timeframe_data(analysis_data: Dict) -> Dict[str, pd.DataFrame]:
    """Extract timeframe data from comprehensive analysis"""
    timeframe_data = {}
    
    for key, value in analysis_data.items():
        # Skip metadata and analysis sections
        if key in ['analysis_metadata', 'file_path', 'analysis_timestamp', 'basic_stats']:
            continue
        
        # Check if this looks like timeframe data
        if isinstance(value, list) and value:
            try:
                df = pd.DataFrame(value)
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                    timeframe_data[key.upper()] = df
            except Exception as e:
                logger.warning(f"Could not convert {key} to DataFrame: {e}")
    
    return timeframe_data

# ============================================================================
# INTELLIGENT COMMENTARY SYSTEM
# ============================================================================

class NCOSCommentaryEngine:
    """Advanced commentary engine for NCOS analysis"""
    
    def __init__(self):
        self.midas_signals = []
        self.smc_signals = []
        self.wyckoff_phases = []
        self.session_analysis = {}
        self.market_structure = {}
    
    def generate_comprehensive_commentary(self, analysis_data: Dict, symbol: str) -> Dict[str, str]:
        """Generate intelligent commentary based on analysis data"""
        commentary = {
            'overview': self._generate_overview(analysis_data, symbol),
            'midas_analysis': self._analyze_midas_signals(analysis_data),
            'smc_analysis': self._analyze_smc_structure(analysis_data),
            'session_analysis': self._analyze_sessions(analysis_data),
            'risk_assessment': self._assess_risk(analysis_data),
            'trading_opportunities': self._identify_opportunities(analysis_data),
            'market_structure': self._analyze_market_structure(analysis_data)
        }
        
        return commentary
    
    def _generate_overview(self, data: Dict, symbol: str) -> str:
        """Generate market overview commentary"""
        basic_stats = data.get('basic_stats', {})
        
        if not basic_stats:
            return f"**{symbol} Market Overview**: Analysis data loaded successfully. Comprehensive microstructure analysis available."
        
        price_change = basic_stats.get('price_change_pct', 0)
        volatility = basic_stats.get('volatility', 0)
        
        direction = "bullish momentum" if price_change > 0 else "bearish pressure"
        vol_level = "high" if volatility > 0.02 else "moderate" if volatility > 0.01 else "low"
        
        return f"""
        **{symbol} Market Overview**
        
        📊 **Price Action**: Currently showing {direction} with {abs(price_change):.2f}% movement
        📈 **Volatility**: {vol_level.title()} volatility environment ({volatility:.3f})
        🎯 **Analysis Status**: Full NCOS v11 microstructure analysis completed
        ⚡ **Data Quality**: Enhanced with 70+ technical indicators and SMC analysis
        """
    
    def _analyze_midas_signals(self, data: Dict) -> str:
        """Analyze Midas Model signals for Gold trading"""
        commentary = "**🥇 Midas Model Analysis (Gold-Specific)**\n\n"
        
        # Check for Midas signals in the data
        midas_found = False
        for timeframe, tf_data in data.items():
            if isinstance(tf_data, list) and tf_data:
                df = pd.DataFrame(tf_data)
                if 'midas_signal' in df.columns:
                    signals = df[df['midas_signal'] == True]
                    if not signals.empty:
                        midas_found = True
                        latest_signal = signals.iloc[-1]
                        signal_type = latest_signal.get('midas_setup_type', 'Unknown')
                        
                        commentary += f"""
                        🎯 **Active Midas Signal Detected** ({timeframe})
                        - **Setup Type**: {signal_type}
                        - **Entry Price**: {latest_signal.get('midas_entry', 'N/A')}
                        - **Stop Loss**: {latest_signal.get('midas_stop', 'N/A')}
                        - **Target**: {latest_signal.get('midas_target', 'N/A')}
                        
                        💡 **Interpretation**: Midas Model identifies mechanical setups at 8PM/9PM NY time.
                        This signal suggests institutional liquidity sweep followed by market structure shift.
                        """
        
        if not midas_found:
            commentary += """
            ⏰ **No Active Midas Signals**
            
            The Midas Model operates on specific timing (8PM & 9PM NY time) and looks for:
            - Liquidity sweeps beyond 15-minute highs/lows
            - Market structure shifts with displacement
            - Fair Value Gap formation after institutional moves
            
            📍 **Next Window**: Monitor upcoming 8PM/9PM NY sessions for potential setups
            """
        
        return commentary
    
    def _analyze_smc_structure(self, data: Dict) -> str:
        """Analyze Smart Money Concepts structure"""
        commentary = "**🧠 Smart Money Concepts (SMC) Analysis**\n\n"
        
        # Look for SMC analysis data
        smc_data = data.get('smc_analysis', {})
        
        if smc_data:
            # Market structure
            market_structure = smc_data.get('market_structure', {})
            trend = market_structure.get('trend', 'Unknown')
            
            commentary += f"📈 **Market Trend**: {trend}\n\n"
            
            # Fair Value Gaps
            fvgs = smc_data.get('fair_value_gaps', [])
            if fvgs:
                commentary += f"🎯 **Fair Value Gaps**: {len(fvgs)} FVGs identified\n"
                recent_fvg = fvgs[-1] if fvgs else {}
                fvg_type = recent_fvg.get('type', 'Unknown')
                commentary += f"- **Latest FVG**: {fvg_type} at index {recent_fvg.get('index', 'N/A')}\n\n"
            
            # Order Blocks
            order_blocks = smc_data.get('order_blocks', [])
            if order_blocks:
                commentary += f"📦 **Order Blocks**: {len(order_blocks)} identified\n"
                recent_ob = order_blocks[-1] if order_blocks else {}
                ob_type = recent_ob.get('type', 'Unknown')
                commentary += f"- **Latest OB**: {ob_type} (strength: {recent_ob.get('strength', 0):.3f})\n\n"
            
            # Liquidity Zones
            liquidity_zones = smc_data.get('liquidity_zones', [])
            if liquidity_zones:
                commentary += f"💧 **Liquidity Zones**: {len(liquidity_zones)} zones detected\n\n"
            
            # Displacement Analysis
            displacement = smc_data.get('displacement', [])
            if displacement:
                recent_disp = displacement[-1] if displacement else {}
                disp_direction = recent_disp.get('direction', 'Unknown')
                commentary += f"⚡ **Recent Displacement**: {disp_direction} with {recent_disp.get('magnitude', 0):.4f} magnitude\n\n"
        
        else:
            commentary += "No SMC analysis data available in current dataset.\n\n"
        
        commentary += """
        💡 **SMC Interpretation**:
        - **FVGs**: Imbalances that price may return to fill
        - **Order Blocks**: Areas of institutional accumulation/distribution
        - **Displacement**: Sharp moves indicating smart money activity
        - **Liquidity Zones**: Areas where stops are likely clustered
        """
        
        return commentary
    
    def _analyze_sessions(self, data: Dict) -> str:
        """Analyze trading session data"""
        commentary = "**🌍 Session Analysis**\n\n"
        
        session_found = False
        london_killzone_active = False
        
        # Check for session data in timeframes
        for timeframe, tf_data in data.items():
            if isinstance(tf_data, list) and tf_data:
                df = pd.DataFrame(tf_data)
                
                if 'session' in df.columns:
                    session_found = True
                    sessions = df['session'].value_counts()
                    commentary += f"**{timeframe} Session Distribution**:\n"
                    for session, count in sessions.items():
                        commentary += f"- {session}: {count} bars\n"
                
                if 'is_london_killzone' in df.columns:
                    killzone_bars = df['is_london_killzone'].sum()
                    if killzone_bars > 0:
                        london_killzone_active = True
                        commentary += f"\n🔥 **London Kill Zone Active**: {killzone_bars} bars in {timeframe}\n"
        
        if london_killzone_active:
            commentary += """
            **🇬🇧 London Kill Zone Analysis**:
            - **Timing**: 7:00-13:00 UTC (London Session)
            - **Characteristics**: High liquidity injection, new trend formation
            - **Strategy**: Monitor for liquidity sweeps and reversals
            - **Daily Extremes**: High/low often established in this session
            """
        
        if not session_found:
            commentary += """
            **Session Information**: No session data available in current analysis.
            
            **Key Sessions**:
            - **Asian**: 22:00-07:00 UTC (Lower volatility)
            - **London**: 07:00-16:00 UTC (High volatility)
            - **New York**: 13:00-22:00 UTC (Overlaps with London)
            """
        
        return commentary
    
    def _assess_risk(self, data: Dict) -> str:
        """Generate risk assessment commentary"""
        commentary = "**⚠️ Risk Assessment**\n\n"
        
        risk_metrics = data.get('risk_metrics', {})
        
        if risk_metrics:
            volatility = risk_metrics.get('std_dev', 0)
            max_dd = risk_metrics.get('max_drawdown', 0)
            var_95 = risk_metrics.get('value_at_risk_95', 0)
            
            vol_level = "HIGH" if volatility > 0.03 else "MODERATE" if volatility > 0.015 else "LOW"
            
            commentary += f"""
            📊 **Risk Metrics**:
            - **Volatility Level**: {vol_level} ({volatility:.4f})
            - **Maximum Drawdown**: {max_dd:.2%}
            - **VaR (95%)**: {var_95:.4f}
            
            **Risk Interpretation**:
            """
            
            if volatility > 0.03:
                commentary += "- **HIGH VOLATILITY**: Exercise caution, reduce position sizes\n"
            elif volatility > 0.015:
                commentary += "- **MODERATE VOLATILITY**: Standard risk management applies\n"
            else:
                commentary += "- **LOW VOLATILITY**: May indicate range-bound conditions\n"
        
        else:
            commentary += """
            **Risk Metrics**: No risk calculation data available.
            
            **General Risk Guidelines**:
            - **Position Sizing**: Never risk more than 1-2% per trade
            - **Stop Losses**: Always use stops, especially with SMC setups
            - **Correlation**: Monitor correlation with other positions
            """
        
        return commentary
    
    def _identify_opportunities(self, data: Dict) -> str:
        """Identify trading opportunities"""
        commentary = "**🎯 Trading Opportunities**\n\n"
        
        opportunities = []
        
        # Check for various signal types
        for timeframe, tf_data in data.items():
            if isinstance(tf_data, list) and tf_data:
                df = pd.DataFrame(tf_data)
                
                # Midas signals
                if 'midas_signal' in df.columns and df['midas_signal'].any():
                    opportunities.append(f"🥇 **Midas Signal** in {timeframe}")
                
                # FVG signals
                if any(col.startswith('fvg_') for col in df.columns):
                    opportunities.append(f"🎯 **FVG Opportunity** in {timeframe}")
                
                # Check for recent breakouts or patterns
                if 'breakout_up' in df.columns and df['breakout_up'].tail(5).any():
                    opportunities.append(f"📈 **Bullish Breakout** in {timeframe}")
                
                if 'breakout_down' in df.columns and df['breakout_down'].tail(5).any():
                    opportunities.append(f"📉 **Bearish Breakout** in {timeframe}")
        
        if opportunities:
            commentary += "**🔍 Active Opportunities**:\n"
            for opp in opportunities[:5]:  # Limit to top 5
                commentary += f"- {opp}\n"
            
            commentary += "\n**💡 Opportunity Analysis**:\n"
            commentary += "- Confirm signals across multiple timeframes\n"
            commentary += "- Wait for London Kill Zone or NY session for execution\n"
            commentary += "- Ensure proper risk-reward ratio (minimum 1:2)\n"
        
        else:
            commentary += """
            **📊 Market Status**: No immediate high-probability setups identified.
            
            **⏰ What to Monitor**:
            - Upcoming session opens (London/NY)
            - Liquidity sweep setups near key levels
            - FVG formations after displacement moves
            - Wyckoff accumulation/distribution phases
            """
        
        return commentary
    
    def _analyze_market_structure(self, data: Dict) -> str:
        """Analyze overall market structure"""
        commentary = "**🏗️ Market Structure Analysis**\n\n"
        
        # Look for advanced analytics
        advanced_analytics = data.get('advanced_analytics', {})
        
        if advanced_analytics:
            trend_strength = advanced_analytics.get('trend_strength', 0)
            efficiency_ratio = advanced_analytics.get('efficiency_ratio', 0)
            
            trend_level = "STRONG" if trend_strength > 0.7 else "MODERATE" if trend_strength > 0.4 else "WEAK"
            
            commentary += f"""
            **📈 Structure Metrics**:
            - **Trend Strength**: {trend_level} ({trend_strength:.3f})
            - **Efficiency Ratio**: {efficiency_ratio:.3f}
            
            **Interpretation**:
            """
            
            if trend_strength > 0.7:
                commentary += "- **TRENDING MARKET**: Strong directional bias, follow trend\n"
            elif trend_strength > 0.4:
                commentary += "- **MIXED CONDITIONS**: Some trend present, be selective\n"
            else:
                commentary += "- **RANGING MARKET**: Weak trend, focus on support/resistance\n"
        
        commentary += """
        
        **🧭 Structure Guidelines**:
        - **Uptrend**: Higher highs + higher lows
        - **Downtrend**: Lower highs + lower lows  
        - **Range**: Respect support/resistance levels
        - **Break of Structure**: Confirms trend change
        """
        
        return commentary

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_enhanced_chart(df: pd.DataFrame, symbol: str, timeframe: str, commentary: Dict) -> go.Figure:
    """Create enhanced chart with commentary integration"""
    
    # Create subplots
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=(
            f'{symbol} {timeframe} - Enhanced Analysis',
            'Volume Profile',
            'Technical Indicators',
            'Session & Signals'
        ),
        row_heights=[0.5, 0.2, 0.2, 0.1]
    )
    
    # Main candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'] if 'open' in df.columns else df['Open'],
            high=df['high'] if 'high' in df.columns else df['High'],
            low=df['low'] if 'low' in df.columns else df['Low'],
            close=df['close'] if 'close' in df.columns else df['Close'],
            name=f'{symbol} {timeframe}',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ),
        row=1, col=1
    )
    
    # Add moving averages if available
    if 'sma_20' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['sma_20'],
                mode='lines', name='SMA 20',
                line=dict(color='orange', width=1)
            ),
            row=1, col=1
        )
    
    if 'ema_50' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['ema_50'],
                mode='lines', name='EMA 50',
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )
    
    # Add Midas signals if present
    if 'midas_signal' in df.columns:
        midas_signals = df[df['midas_signal'] == True]
        if not midas_signals.empty:
            close_col = 'close' if 'close' in df.columns else 'Close'
            fig.add_trace(
                go.Scatter(
                    x=midas_signals.index,
                    y=midas_signals[close_col],
                    mode='markers',
                    marker=dict(symbol='star', size=15, color='gold'),
                    name='Midas Signal',
                    hovertemplate='<b>Midas Signal</b><br>Price: %{y}<br>Time: %{x}<extra></extra>'
                ),
                row=1, col=1
            )
    
    # Add session highlights if available
    if 'is_london_killzone' in df.columns:
        killzone_periods = df[df['is_london_killzone'] == True]
        if not killzone_periods.empty:
            for idx in killzone_periods.index:
                fig.add_vrect(
                    x0=idx, x1=idx + timedelta(hours=1),
                    fillcolor="rgba(255, 215, 0, 0.2)",
                    layer="below", line_width=0,
                    row=1, col=1
                )
    
    # Volume chart
    volume_col = 'volume' if 'volume' in df.columns else 'Volume' if 'Volume' in df.columns else None
    if volume_col:
        close_col = 'close' if 'close' in df.columns else 'Close'
        open_col = 'open' if 'open' in df.columns else 'Open'
        colors = ['red' if close < open_price else 'green' 
                 for close, open_price in zip(df[close_col], df[open_col])]
        
        fig.add_trace(
            go.Bar(
                x=df.index, y=df[volume_col],
                name='Volume',
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )
    
    # Technical indicators
    if 'rsi' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['rsi'],
                mode='lines', name='RSI',
                line=dict(color='purple')
            ),
            row=3, col=1
        )
        
        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    # Session indicators
    if 'session' in df.columns:
        session_numeric = df['session'].map({'AS': 1, 'EU': 2, 'US': 3}).fillna(0)
        fig.add_trace(
            go.Scatter(
                x=df.index, y=session_numeric,
                mode='markers', name='Sessions',
                marker=dict(size=4, color=session_numeric, colorscale='Viridis')
            ),
            row=4, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=f'{symbol} {timeframe} - NCOS v11 Enhanced Analysis',
        template='plotly_dark',
        height=800,
        showlegend=True,
        font=dict(color='white'),
        paper_bgcolor='rgba(15, 15, 35, 0.95)',
        plot_bgcolor='rgba(15, 15, 35, 0.95)'
    )
    
    # Update axes
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])
    fig.update_yaxes(title_text="Session", row=4, col=1)
    
    return fig

def create_analysis_summary_chart(analysis_data: Dict) -> go.Figure:
    """Create summary analysis chart"""
    fig = go.Figure()
    
    # Create a summary metrics visualization
    metrics = []
    values = []
    
    # Extract key metrics
    basic_stats = analysis_data.get('basic_stats', {})
    if basic_stats:
        metrics.extend(['Price Change %', 'Volatility', 'Volume Avg'])
        values.extend([
            basic_stats.get('price_change_pct', 0),
            basic_stats.get('volatility', 0) * 100,
            basic_stats.get('avg_volume', 0) / 1000  # Scale volume
        ])
    
    # Add risk metrics if available
    risk_metrics = analysis_data.get('risk_metrics', {})
    if risk_metrics:
        metrics.extend(['Max Drawdown %', 'VaR 95%'])
        values.extend([
            abs(risk_metrics.get('max_drawdown', 0)) * 100,
            abs(risk_metrics.get('value_at_risk_95', 0)) * 100
        ])
    
    if metrics and values:
        fig.add_trace(go.Bar(
            x=metrics,
            y=values,
            marker_color=['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe'],
            text=[f'{v:.2f}' for v in values],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Analysis Summary Metrics",
            template='plotly_dark',
            font=dict(color='white'),
            paper_bgcolor='rgba(15, 15, 35, 0.95)',
            height=400
        )
    
    return fig

# ============================================================================
# MAIN STREAMLIT APPLICATION
# ============================================================================

def main():
    """Main Streamlit application"""
    
    # Setup page configuration
    setup_page_config()
    
    # Professional header
    st.markdown("""
        <div class="header-style">
            <h1>🧠 NCOS v11 Enhanced Analysis Dashboard</h1>
            <p><strong>Intelligent Commentary System for Enriched Market Data</strong></p>
            <p>Advanced SMC • Midas Model • Wyckoff • London Kill Zone Analysis</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load TOML configuration
    config = load_toml_config()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Display current configuration
        with st.expander("TOML Configuration"):
            st.json(config)
        
        st.markdown("---")
        
        # Analysis options
        st.header("📊 Analysis Options")
        
        enable_midas = st.checkbox("Midas Model Analysis", value=True, help="Gold-specific mechanical trading strategy")
        enable_smc = st.checkbox("Smart Money Concepts", value=True, help="Institutional flow analysis")
        enable_sessions = st.checkbox("Session Analysis", value=True, help="London Kill Zone and session timing")
        enable_commentary = st.checkbox("AI Commentary", value=True, help="Intelligent market commentary")
        
        st.markdown("---")
        
        # Display options
        st.header("📈 Display Options")
        
        show_all_timeframes = st.checkbox("All Timeframes", value=False)
        max_timeframes = st.slider("Max Timeframes", 1, 10, 5)
    
    # Discover available JSON files
    with st.spinner("Scanning JSON directory for analysis files..."):
        discovered_files = discover_json_files(config['json_dir'])
    
    # Display discovery results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("JSON Files Found", len(discovered_files['comprehensive_files']))
    with col2:
        st.metric("Symbols Available", len(discovered_files['symbols']))
    with col3:
        st.metric("Metadata Files", len(discovered_files['metadata_files']))
    
    # File selection
    if discovered_files['comprehensive_files']:
        st.subheader("📂 Select Analysis File")
        
        # Symbol-based selection
        if discovered_files['symbols']:
            selected_symbol = st.selectbox(
                "Select Symbol",
                options=discovered_files['symbols'],
                help="Choose symbol for analysis"
            )
            
            # Find matching files for the symbol
            matching_files = [f for f in discovered_files['comprehensive_files'] 
                            if selected_symbol.lower() in f.lower()]
            
            if matching_files:
                selected_file = st.selectbox(
                    "Select Analysis File",
                    options=matching_files,
                    format_func=lambda x: Path(x).name
                )
            else:
                st.warning(f"No analysis files found for {selected_symbol}")
                return
        else:
            # Direct file selection if no symbols detected
            selected_file = st.selectbox(
                "Select Analysis File",
                options=discovered_files['comprehensive_files'],
                format_func=lambda x: Path(x).name
            )
            selected_symbol = "UNKNOWN"
    
    else:
        st.warning(f"No JSON analysis files found in: {config['json_dir']}")
        st.info("Run the ncOS enhanced analyzer to generate analysis files first:")
        st.code("""
        python ncos_enhanced_analyzer.py \\
            --file XAUUSD_M1_bars.csv \\
            --timeframes_parquet M1,M5,M15,M30,H1,H4,D1 \\
            --max_candles 500 \\
            --output_dir ./midas_analysis
        """)
        return
    
    # Load and analyze selected file
    if st.button("🚀 Load Analysis", type="primary"):
        with st.spinner("Loading and analyzing data..."):
            
            # Load JSON analysis
            analysis_data = load_json_analysis(selected_file)
            
            if not analysis_data:
                st.error("Failed to load analysis data")
                return
            
            # Extract timeframe data
            timeframe_data = extract_timeframe_data(analysis_data)
            
            # Initialize commentary engine
            commentary_engine = NCOSCommentaryEngine()
            
            # Generate comprehensive commentary
            if enable_commentary:
                commentary = commentary_engine.generate_comprehensive_commentary(
                    analysis_data, selected_symbol
                )
            else:
                commentary = {}
            
            # Display analysis results
            st.subheader(f"📊 Analysis Results: {selected_symbol}")
            
            # Key metrics
            basic_stats = analysis_data.get('basic_stats', {})
            if basic_stats:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Price Change",
                        f"{basic_stats.get('price_change_pct', 0):.2f}%",
                        delta=f"{basic_stats.get('price_change', 0):.2f}"
                    )
                
                with col2:
                    st.metric(
                        "Volatility",
                        f"{basic_stats.get('volatility', 0):.3f}"
                    )
                
                with col3:
                    st.metric(
                        "Avg Volume",
                        f"{basic_stats.get('avg_volume', 0):,.0f}"
                    )
                
                with col4:
                    duration = basic_stats.get('duration_hours', 0)
                    st.metric(
                        "Duration",
                        f"{duration:.1f}h"
                    )
            
            # Commentary sections
            if enable_commentary and commentary:
                
                # Overview
                if commentary.get('overview'):
                    st.markdown(f"""
                        <div class="commentary-box">
                            {commentary['overview']}
                        </div>
                    """, unsafe_allow_html=True)
                
                # Midas analysis for Gold
                if enable_midas and selected_symbol in ['XAUUSD', 'GOLD', 'XAU'] and commentary.get('midas_analysis'):
                    st.markdown("""
                        <div class="midas-signal">
                            🥇 MIDAS MODEL ANALYSIS (GOLD)
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                        <div class="commentary-box">
                            {commentary['midas_analysis']}
                        </div>
                    """, unsafe_allow_html=True)
                
                # SMC Analysis
                if enable_smc and commentary.get('smc_analysis'):
                    st.markdown(f"""
                        <div class="commentary-box">
                            {commentary['smc_analysis']}
                        </div>
                    """, unsafe_allow_html=True)
                
                # Session Analysis
                if enable_sessions and commentary.get('session_analysis'):
                    st.markdown(f"""
                        <div class="commentary-box">
                            {commentary['session_analysis']}
                        </div>
                    """, unsafe_allow_html=True)
                
                # Trading Opportunities
                if commentary.get('trading_opportunities'):
                    st.markdown(f"""
                        <div class="signal-alert">
                            {commentary['trading_opportunities']}
                        </div>
                    """, unsafe_allow_html=True)
                
                # Risk Assessment
                if commentary.get('risk_assessment'):
                    st.markdown(f"""
                        <div class="commentary-box">
                            {commentary['risk_assessment']}
                        </div>
                    """, unsafe_allow_html=True)
            
            # Timeframe charts
            if timeframe_data:
                st.subheader("📈 Timeframe Analysis")
                
                # Limit timeframes if requested
                displayed_timeframes = list(timeframe_data.keys())
                if not show_all_timeframes:
                    displayed_timeframes = displayed_timeframes[:max_timeframes]
                
                for tf in displayed_timeframes:
                    df = timeframe_data[tf]
                    
                    st.markdown(f"### {tf} Timeframe")
                    
                    # Create chart
                    chart = create_enhanced_chart(df, selected_symbol, tf, commentary)
                    st.plotly_chart(chart, use_container_width=True)
                    
                    # Data summary
                    with st.expander(f"{tf} Data Summary"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Basic Info**")
                            st.write(f"- Bars: {len(df)}")
                            st.write(f"- Columns: {len(df.columns)}")
                            st.write(f"- Period: {df.index.min()} to {df.index.max()}")
                        
                        with col2:
                            st.write("**Available Indicators**")
                            indicators = [col for col in df.columns if any(x in col.lower() 
                                        for x in ['sma', 'ema', 'rsi', 'macd', 'atr', 'midas', 'fvg'])]
                            for indicator in indicators[:10]:  # Show first 10
                                st.write(f"- {indicator}")
                            if len(indicators) > 10:
                                st.write(f"... and {len(indicators) - 10} more")
            
            # Analysis summary chart
            st.subheader("📊 Analysis Summary")
            summary_chart = create_analysis_summary_chart(analysis_data)
            st.plotly_chart(summary_chart, use_container_width=True)
            
            # Raw data viewer
            with st.expander("🔍 Raw Analysis Data"):
                st.json(analysis_data)
            
            # Export options
            st.subheader("💾 Export Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📄 Export Commentary"):
                    commentary_text = "\n\n".join([
                        f"# {key.replace('_', ' ').title()}\n\n{value}"
                        for key, value in commentary.items() if value
                    ])
                    
                    st.download_button(
                        label="Download Commentary",
                        data=commentary_text,
                        file_name=f"{selected_symbol}_commentary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
            
            with col2:
                if st.button("📊 Export Analysis Data"):
                    analysis_json = json.dumps(analysis_data, indent=2, default=str)
                    
                    st.download_button(
                        label="Download Analysis",
                        data=analysis_json,
                        file_name=f"{selected_symbol}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; padding: 1rem; color: #888;">
            <p><strong>NCOS v11 Enhanced Analysis Dashboard</strong></p>
            <p>Intelligent Commentary • SMC Analysis • Midas Model • Wyckoff Method</p>
            <p>Built for institutional-grade market analysis</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()