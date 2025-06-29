# Stage 5: Streamlit Dashboard for Trading Analysis
# zanalytics_dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
from pathlib import Path
import asyncio
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass
import altair as alt
from streamlit_autorefresh import st_autorefresh
import time

# Configure page
st.set_page_config(
    page_title="Zanalytics Trading Dashboard",
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
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    div[data-testid="metric-container"] {
        background-color: rgba(28, 131, 225, 0.1);
        border: 1px solid rgba(28, 131, 225, 0.2);
        padding: 5% 5% 5% 10%;
        border-radius: 5px;
        color: rgb(30, 103, 119);
        overflow-wrap: break-word;
    }
    .signal-buy {
        background-color: #28a745;
        color: white;
        padding: 5px 10px;
        border-radius: 3px;
    }
    .signal-sell {
        background-color: #dc3545;
        color: white;
        padding: 5px 10px;
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()
if 'selected_symbol' not in st.session_state:
    st.session_state.selected_symbol = "BTC/USDT"
if 'selected_timeframe' not in st.session_state:
    st.session_state.selected_timeframe = "1h"
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = False

class DashboardData:
    """Manages dashboard data loading and caching"""

    def __init__(self):
        self.data_dir = Path("./enriched")
        self.signals_dir = Path("./signals")
        self.integrated_dir = Path("./integrated_results")
        self.llm_dir = Path("./llm_outputs")

    @st.cache_data(ttl=60)
    def load_market_data(_self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Load market data for symbol and timeframe"""
        try:
            pattern = f"{symbol.replace('/', '_')}_{timeframe}_*.csv"
            files = list(_self.data_dir.glob(pattern))

            if not files:
                return None

            # Get most recent file
            latest_file = max(files, key=lambda x: x.stat().st_mtime)
            df = pd.read_csv(latest_file, index_col=0, parse_dates=True)

            return df
        except Exception as e:
            st.error(f"Error loading market data: {e}")
            return None

    @st.cache_data(ttl=60)
    def load_signals(_self, symbol: str, timeframe: str) -> List[Dict[str, Any]]:
        """Load trading signals"""
        try:
            pattern = f"{symbol.replace('/', '_')}_{timeframe}_*_signals.json"
            files = list(_self.signals_dir.glob(pattern))

            if not files:
                return []

            # Get most recent file
            latest_file = max(files, key=lambda x: x.stat().st_mtime)

            with open(latest_file, 'r') as f:
                data = json.load(f)

            return data.get("signals", [])
        except Exception as e:
            st.error(f"Error loading signals: {e}")
            return []

    @st.cache_data(ttl=60)
    def load_integrated_analysis(_self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """Load integrated analysis results"""
        try:
            pattern = f"{symbol.replace('/', '_')}_{timeframe}_*_integrated.json"
            files = list(_self.integrated_dir.glob(pattern))

            if not files:
                return None

            latest_file = max(files, key=lambda x: x.stat().st_mtime)

            with open(latest_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading integrated analysis: {e}")
            return None

    def get_available_symbols(_self) -> List[str]:
        """Get list of available symbols"""
        symbols = set()

        for file in _self.data_dir.glob("*_*.csv"):
            parts = file.stem.split('_')
            if len(parts) >= 2:
                symbol = parts[0].replace('-', '/')
                symbols.add(symbol)

        return sorted(list(symbols))

    def get_available_timeframes(_self, symbol: str) -> List[str]:
        """Get available timeframes for a symbol"""
        timeframes = set()
        pattern = f"{symbol.replace('/', '_')}_*.csv"

        for file in _self.data_dir.glob(pattern):
            parts = file.stem.split('_')
            if len(parts) >= 2:
                timeframes.add(parts[1])

        return sorted(list(timeframes))

def create_price_chart(df: pd.DataFrame, signals: List[Dict[str, Any]] = None) -> go.Figure:
    """Create interactive price chart with indicators"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=("Price & Indicators", "Volume", "RSI")
    )

    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="Price",
            showlegend=False
        ),
        row=1, col=1
    )

    # Moving averages
    for ma_col in df.columns:
        if ma_col.startswith('sma_') or ma_col.startswith('ema_'):
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[ma_col],
                    name=ma_col.upper(),
                    line=dict(width=1)
                ),
                row=1, col=1
            )

    # Bollinger Bands
    if 'bb_upper' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['bb_upper'],
                name='BB Upper',
                line=dict(color='gray', dash='dash'),
                showlegend=False
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['bb_lower'],
                name='BB Lower',
                line=dict(color='gray', dash='dash'),
                fill='tonexty',
                fillcolor='rgba(128,128,128,0.1)',
                showlegend=False
            ),
            row=1, col=1
        )

    # Add signals to chart
    if signals:
        buy_signals = [s for s in signals if 'buy' in s.get('signal_type', '').lower()]
        sell_signals = [s for s in signals if 'sell' in s.get('signal_type', '').lower()]

        if buy_signals:
            fig.add_trace(
                go.Scatter(
                    x=[datetime.fromisoformat(s['timestamp']) for s in buy_signals],
                    y=[s['entry_price'] for s in buy_signals],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up',
                        size=12,
                        color='green'
                    ),
                    name='Buy Signals',
                    text=[f"Buy: {s.get('confidence', 0)*100:.1f}%" for s in buy_signals],
                    hovertemplate='%{text}<br>Price: %{y:.4f}<extra></extra>'
                ),
                row=1, col=1
            )

        if sell_signals:
            fig.add_trace(
                go.Scatter(
                    x=[datetime.fromisoformat(s['timestamp']) for s in sell_signals],
                    y=[s['entry_price'] for s in sell_signals],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-down',
                        size=12,
                        color='red'
                    ),
                    name='Sell Signals',
                    text=[f"Sell: {s.get('confidence', 0)*100:.1f}%" for s in sell_signals],
                    hovertemplate='%{text}<br>Price: %{y:.4f}<extra></extra>'
                ),
                row=1, col=1
            )

    # Volume
    colors = ['red' if row['close'] < row['open'] else 'green' for idx, row in df.iterrows()]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['volume'],
            name='Volume',
            marker_color=colors,
            showlegend=False
        ),
        row=2, col=1
    )

    # RSI
    if 'rsi' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['rsi'],
                name='RSI',
                line=dict(color='purple'),
                showlegend=False
            ),
            row=3, col=1
        )

        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

    # Update layout
    fig.update_layout(
        title=f"Price Chart with Technical Indicators",
        xaxis_title="Time",
        yaxis_title="Price",
        height=800,
        template="plotly_white",
        hovermode='x unified'
    )

    fig.update_xaxes(rangeslider_visible=False)

    return fig

def display_signal_card(signal: Dict[str, Any]):
    """Display a signal as a card"""
    signal_type = signal.get('signal_type', 'unknown').lower()
    priority = signal.get('priority', 1)
    confidence = signal.get('confidence', 0)

    # Color coding
    if 'buy' in signal_type:
        color = "green"
        emoji = "📈"
    elif 'sell' in signal_type:
        color = "red"
        emoji = "📉"
    else:
        color = "gray"
        emoji = "↔️"

    # Priority badge
    priority_badges = {
        1: "🟢 Low",
        2: "🟡 Medium", 
        3: "🟠 High",
        4: "🔴 Critical"
    }

    with st.container():
        col1, col2, col3 = st.columns([2, 3, 2])

        with col1:
            st.markdown(f"### {emoji} {signal_type.upper()}")
            st.markdown(f"**Priority:** {priority_badges.get(priority, '⚪ Unknown')}")

        with col2:
            st.metric("Entry Price", f"${signal.get('entry_price', 0):.4f}")
            st.metric("Stop Loss", f"${signal.get('stop_loss', 0):.4f}")

            targets = signal.get('take_profit_targets', [])
            if targets:
                st.metric("Target 1", f"${targets[0]:.4f}")

        with col3:
            st.metric("Confidence", f"{confidence*100:.1f}%")
            st.metric("Risk/Reward", f"{signal.get('risk_reward_ratio', 0):.2f}")
            st.metric("Position Size", f"{signal.get('position_size_suggestion', 0)*100:.1f}%")

        # Reasoning
        reasons = signal.get('reasoning', [])
        if reasons:
            with st.expander("Signal Reasoning"):
                for reason in reasons:
                    st.write(f"• {reason}")

def display_market_overview(analysis: Dict[str, Any]):
    """Display market overview section"""
    st.header("📊 Market Overview")

    consensus = analysis.get('consensus', {})
    metadata = analysis.get('metadata', {})

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        sentiment = consensus.get('overall_sentiment', 'neutral')
        sentiment_emoji = {"bullish": "🐂", "bearish": "🐻", "neutral": "➖"}
        st.metric(
            "Market Sentiment",
            sentiment.title(),
            sentiment_emoji.get(sentiment, "")
        )

    with col2:
        confidence = consensus.get('confidence', 0)
        delta = "High" if confidence > 0.7 else "Low" if confidence < 0.3 else "Medium"
        st.metric(
            "Analysis Confidence",
            f"{confidence*100:.1f}%",
            delta
        )

    with col3:
        st.metric(
            "Active Analyzers",
            metadata.get('successful_analyses', 0),
            f"of {metadata.get('total_analyzers', 0)}"
        )

    with col4:
        signal_count = len(consensus.get('signals', []))
        st.metric(
            "Signals Generated",
            signal_count,
            "Ready" if signal_count > 0 else "None"
        )

def display_technical_analysis(df: pd.DataFrame):
    """Display technical analysis indicators"""
    st.header("🔧 Technical Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Trend Indicators")

        # Current price vs MAs
        current_price = df['close'].iloc[-1]

        metrics = []
        for col in df.columns:
            if col.startswith(('sma_', 'ema_')):
                ma_value = df[col].iloc[-1]
                if pd.notna(ma_value):
                    diff_pct = ((current_price - ma_value) / ma_value) * 100
                    metrics.append({
                        'Indicator': col.upper(),
                        'Value': f"${ma_value:.2f}",
                        'Difference': f"{diff_pct:+.2f}%",
                        'Status': '🟢' if diff_pct > 0 else '🔴'
                    })

        if metrics:
            metrics_df = pd.DataFrame(metrics)
            st.dataframe(metrics_df, hide_index=True)

    with col2:
        st.subheader("Momentum Indicators")

        # RSI
        if 'rsi' in df.columns:
            rsi_value = df['rsi'].iloc[-1]

            # Create RSI gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=rsi_value,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "RSI"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "lightgray"},
                        {'range': [70, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)

        # MACD
        if all(col in df.columns for col in ['macd', 'macd_signal']):
            macd_val = df['macd'].iloc[-1]
            signal_val = df['macd_signal'].iloc[-1]
            hist_val = macd_val - signal_val

            st.metric("MACD", f"{macd_val:.4f}")
            st.metric("MACD Signal", f"{signal_val:.4f}")
            st.metric("MACD Histogram", f"{hist_val:.4f}", 
                     "Bullish" if hist_val > 0 else "Bearish")

def display_pattern_analysis(analysis: Dict[str, Any]):
    """Display detected patterns"""
    st.header("🎯 Pattern Detection")

    all_patterns = []

    # Collect patterns from all analyzers
    for analyzer_name, result in analysis.get('individual_analyses', {}).items():
        if not result.get('errors') and 'results' in result:
            patterns = result['results'].get('patterns', [])
            if isinstance(patterns, list):
                for pattern in patterns:
                    if isinstance(pattern, dict):
                        pattern['source'] = analyzer_name
                        all_patterns.append(pattern)

    if all_patterns:
        # Sort by confidence
        all_patterns.sort(key=lambda x: x.get('confidence', 0), reverse=True)

        # Display top patterns
        for i, pattern in enumerate(all_patterns[:5]):
            col1, col2, col3 = st.columns([3, 2, 1])

            with col1:
                st.write(f"**{pattern.get('name', 'Unknown Pattern')}**")
                st.caption(f"Source: {pattern['source']}")

            with col2:
                confidence = pattern.get('confidence', 0)
                st.progress(confidence)
                st.caption(f"Confidence: {confidence*100:.1f}%")

            with col3:
                pattern_type = pattern.get('type', 'neutral').lower()
                if 'bullish' in pattern_type:
                    st.success("Bullish")
                elif 'bearish' in pattern_type:
                    st.error("Bearish")
                else:
                    st.info("Neutral")
    else:
        st.info("No patterns detected in the current analysis")

def display_risk_metrics(analysis: Dict[str, Any], df: pd.DataFrame):
    """Display risk assessment metrics"""
    st.header("⚠️ Risk Assessment")

    col1, col2 = st.columns(2)

    with col1:
        # Volatility metrics
        st.subheader("Volatility Analysis")

        if 'volatility' in df.columns:
            current_vol = df['volatility'].iloc[-1]
            avg_vol = df['volatility'].mean()

            vol_fig = go.Figure()
            vol_fig.add_trace(go.Scatter(
                x=df.index[-100:],
                y=df['volatility'].iloc[-100:],
                mode='lines',
                name='Volatility',
                fill='tozeroy'
            ))
            vol_fig.add_hline(y=avg_vol, line_dash="dash", 
                            annotation_text="Average")
            vol_fig.update_layout(
                title="Volatility Trend",
                height=300,
                showlegend=False
            )
            st.plotly_chart(vol_fig, use_container_width=True)

            st.metric("Current Volatility", f"{current_vol*100:.2f}%")
            st.metric("Average Volatility", f"{avg_vol*100:.2f}%")

    with col2:
        # Risk summary from analysis
        st.subheader("Risk Summary")

        consensus = analysis.get('consensus', {})

        # Extract risks
        risks = []
        for analyzer_name, result in analysis.get('individual_analyses', {}).items():
            if 'risks' in result.get('results', {}):
                risks.extend(result['results']['risks'])

        if risks:
            for risk in risks[:5]:
                st.warning(f"• {risk}")
        else:
            st.success("No significant risks identified")

        # Key levels
        key_levels = consensus.get('key_levels', {})
        if key_levels:
            st.subheader("Key Price Levels")

            current_price = df['close'].iloc[-1]

            # Support levels
            if 'support' in key_levels and key_levels['support']:
                nearest_support = min(key_levels['support'], 
                                    key=lambda x: abs(x - current_price))
                st.metric("Nearest Support", f"${nearest_support:.2f}",
                         f"{((current_price - nearest_support)/current_price)*100:.2f}% above")

            # Resistance levels
            if 'resistance' in key_levels and key_levels['resistance']:
                nearest_resistance = min(key_levels['resistance'], 
                                       key=lambda x: abs(x - current_price))
                st.metric("Nearest Resistance", f"${nearest_resistance:.2f}",
                         f"{((nearest_resistance - current_price)/current_price)*100:.2f}% below")

def main():
    """Main dashboard application"""
    st.title("🚀 Zanalytics Trading Dashboard")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # Data loader
        data_loader = DashboardData()

        # Symbol selection
        available_symbols = data_loader.get_available_symbols()
        if available_symbols:
            st.session_state.selected_symbol = st.selectbox(
                "Select Symbol",
                available_symbols,
                index=available_symbols.index(st.session_state.selected_symbol) 
                if st.session_state.selected_symbol in available_symbols else 0
            )

            # Timeframe selection
            available_timeframes = data_loader.get_available_timeframes(
                st.session_state.selected_symbol
            )
            if available_timeframes:
                st.session_state.selected_timeframe = st.selectbox(
                    "Select Timeframe",
                    available_timeframes,
                    index=available_timeframes.index(st.session_state.selected_timeframe)
                    if st.session_state.selected_timeframe in available_timeframes else 0
                )

        # Auto-refresh
        st.session_state.auto_refresh = st.checkbox(
            "Auto-refresh (60s)",
            value=st.session_state.auto_refresh
        )

        if st.session_state.auto_refresh:
            st_autorefresh(interval=60000, key="datarefresh")

        # Manual refresh
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.session_state.last_update = datetime.now()
            st.rerun()

        # Last update time
        st.caption(f"Last update: {st.session_state.last_update.strftime('%H:%M:%S')}")

        st.divider()

        # Display options
        st.subheader("Display Options")
        show_patterns = st.checkbox("Show Patterns", value=True)
        show_indicators = st.checkbox("Show Indicators", value=True)
        show_risk = st.checkbox("Show Risk Metrics", value=True)

    # Main content
    if available_symbols:
        # Load data
        df = data_loader.load_market_data(
            st.session_state.selected_symbol,
            st.session_state.selected_timeframe
        )

        signals = data_loader.load_signals(
            st.session_state.selected_symbol,
            st.session_state.selected_timeframe
        )

        analysis = data_loader.load_integrated_analysis(
            st.session_state.selected_symbol,
            st.session_state.selected_timeframe
        )

        if df is not None:
            # Display market overview
            if analysis:
                display_market_overview(analysis)
                st.divider()

            # Main chart
            st.header(f"📈 {st.session_state.selected_symbol} - {st.session_state.selected_timeframe}")
            fig = create_price_chart(df, signals)
            st.plotly_chart(fig, use_container_width=True)

            # Trading signals
            if signals:
                st.divider()
                st.header("🎯 Active Trading Signals")

                # Filter active signals
                active_signals = [s for s in signals 
                                if not s.get('expiry') or 
                                datetime.fromisoformat(s['expiry']) > datetime.now()]

                if active_signals:
                    for signal in active_signals[:3]:  # Show top 3
                        display_signal_card(signal)
                        st.divider()
                else:
                    st.info("No active signals at the moment")

            # Technical Analysis
            if show_indicators:
                st.divider()
                display_technical_analysis(df)

            # Pattern Analysis
            if show_patterns and analysis:
                st.divider()
                display_pattern_analysis(analysis)

            # Risk Metrics
            if show_risk and analysis:
                st.divider()
                display_risk_metrics(analysis, df)

            # Raw Data Export
            with st.expander("📊 Export Raw Data"):
                col1, col2, col3 = st.columns(3)

                with col1:
                    csv = df.to_csv()
                    st.download_button(
                        label="Download Market Data (CSV)",
                        data=csv,
                        file_name=f"{st.session_state.selected_symbol}_{st.session_state.selected_timeframe}_data.csv",
                        mime="text/csv"
                    )

                with col2:
                    if signals:
                        signals_json = json.dumps(signals, indent=2)
                        st.download_button(
                            label="Download Signals (JSON)",
                            data=signals_json,
                            file_name=f"{st.session_state.selected_symbol}_{st.session_state.selected_timeframe}_signals.json",
                            mime="application/json"
                        )

                with col3:
                    if analysis:
                        analysis_json = json.dumps(analysis, indent=2)
                        st.download_button(
                            label="Download Analysis (JSON)",
                            data=analysis_json,
                            file_name=f"{st.session_state.selected_symbol}_{st.session_state.selected_timeframe}_analysis.json",
                            mime="application/json"
                        )
        else:
            st.warning(f"No data available for {st.session_state.selected_symbol} - {st.session_state.selected_timeframe}")
    else:
        st.info("No data files found. Please run the data pipeline first.")

        with st.expander("🚀 Getting Started"):
            st.markdown("""
            1. **Run the data pipeline**: `python zanalytics_data_pipeline.py`
            2. **Run the integration engine**: `python zanalytics_integration.py`
            3. **Generate signals**: `python zanalytics_signal_generator.py`
            4. **Refresh this dashboard**

            The dashboard will automatically detect and display available data.
            """)

if __name__ == "__main__":
    main()
