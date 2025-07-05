# DEPRECATED: This module is retained for reference only.
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import asyncio
import aiohttp
import websockets
import json
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional
import redis
from collections import deque
import threading

# Page configuration
st.set_page_config(
    page_title="Trading Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stMetric {
        background-color: #1e1e1e;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #333;
    }
    .metric-container {
        display: flex;
        justify-content: space-between;
        margin-bottom: 20px;
    }
    .signal-box {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        font-weight: bold;
    }
    .buy-signal {
        background-color: #1f4e1f;
        border: 2px solid #4caf50;
    }
    .sell-signal {
        background-color: #4e1f1f;
        border: 2px solid #f44336;
    }
    .neutral-signal {
        background-color: #3e3e1f;
        border: 2px solid #ff9800;
    }
</style>
""", unsafe_allow_html=True)

class TradingDashboard:
    def __init__(self):
        self.api_base_url = "http://localhost:8000"
        self.ws_url = "ws://localhost:8000/ws"
        self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.tick_buffer = deque(maxlen=1000)
        self.ws_connection = None
        self.running = False

    async def connect_websocket(self):
        """Connect to WebSocket for real-time updates"""
        try:
            self.ws_connection = await websockets.connect(self.ws_url)
            self.running = True

            # Listen for updates
            async for message in self.ws_connection:
                data = json.loads(message)
                if data['type'] == 'tick':
                    self.tick_buffer.append(data['data'])

        except Exception as e:
            st.error(f"WebSocket connection error: {e}")
            self.running = False

    async def fetch_market_snapshot(self, symbol: str) -> Dict:
        """Fetch current market snapshot"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.api_base_url}/api/v1/market/snapshot/{symbol}") as response:
                if response.status == 200:
                    return await response.json()
                return None

    async def fetch_microstructure_analysis(self, symbol: str) -> Dict:
        """Fetch microstructure analysis"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.api_base_url}/api/v1/analysis/microstructure/{symbol}") as response:
                if response.status == 200:
                    return await response.json()
                return None

    async def fetch_trading_signals(self, symbol: str) -> Dict:
        """Fetch trading signals"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.api_base_url}/api/v1/signals/{symbol}") as response:
                if response.status == 200:
                    return await response.json()
                return None

    def create_price_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create interactive price chart with indicators"""
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=('Price & Indicators', 'Volume', 'RSI')
        )

        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df['timestamp'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='OHLC'
            ),
            row=1, col=1
        )

        # Add moving averages
        if 'sma_20' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['sma_20'],
                    name='SMA 20',
                    line=dict(color='orange', width=1)
                ),
                row=1, col=1
            )

        if 'sma_50' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['sma_50'],
                    name='SMA 50',
                    line=dict(color='blue', width=1)
                ),
                row=1, col=1
            )

        # Bollinger Bands
        if all(col in df.columns for col in ['bb_upper', 'bb_middle', 'bb_lower']):
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['bb_upper'],
                    name='BB Upper',
                    line=dict(color='gray', width=1, dash='dash')
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['bb_lower'],
                    name='BB Lower',
                    line=dict(color='gray', width=1, dash='dash'),
                    fill='tonexty',
                    fillcolor='rgba(128, 128, 128, 0.1)'
                ),
                row=1, col=1
            )

        # Volume bars
        colors = ['red' if close < open else 'green' 
                 for close, open in zip(df['close'], df['open'])]

        fig.add_trace(
            go.Bar(
                x=df['timestamp'],
                y=df['volume'],
                name='Volume',
                marker_color=colors
            ),
            row=2, col=1
        )

        # RSI
        if 'rsi' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['rsi'],
                    name='RSI',
                    line=dict(color='purple', width=2)
                ),
                row=3, col=1
            )

            # RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

        # Update layout
        fig.update_layout(
            title='Price Analysis',
            xaxis_title='Time',
            yaxis_title='Price',
            template='plotly_dark',
            height=800,
            showlegend=True,
            legend=dict(x=0, y=1)
        )

        fig.update_xaxes(rangeslider_visible=False)

        return fig

    def create_microstructure_chart(self, data: Dict) -> go.Figure:
        """Create microstructure analysis chart"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Spread Analysis', 'Order Flow Imbalance', 
                          'Liquidity Score', 'Market Toxicity'),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )

        timestamps = [datetime.fromisoformat(t) for t in data['timestamps']]

        # Spread analysis
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=data['effective_spread'],
                name='Effective Spread',
                line=dict(color='cyan')
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=data['realized_spread'],
                name='Realized Spread',
                line=dict(color='magenta')
            ),
            row=1, col=1
        )

        # Order flow imbalance
        fig.add_trace(
            go.Bar(
                x=timestamps,
                y=data['order_flow_imbalance'],
                name='OFI',
                marker_color=['red' if x < 0 else 'green' for x in data['order_flow_imbalance']]
            ),
            row=1, col=2
        )

        # Liquidity score
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=data['liquidity_score'],
                name='Liquidity',
                line=dict(color='blue'),
                fill='tozeroy'
            ),
            row=2, col=1
        )

        # Toxicity score
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=data['toxicity_score'],
                name='Toxicity',
                line=dict(color='red'),
                fill='tozeroy'
            ),
            row=2, col=2
        )

        fig.update_layout(
            title='Microstructure Analysis',
            template='plotly_dark',
            height=600,
            showlegend=True
        )

        return fig

    def create_signal_gauge(self, signal_strength: float) -> go.Figure:
        """Create signal strength gauge"""
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=signal_strength,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Signal Strength"},
            delta={'reference': 0},
            gauge={
                'axis': {'range': [-100, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [-100, -50], 'color': "darkred"},
                    {'range': [-50, -20], 'color': "red"},
                    {'range': [-20, 20], 'color': "gray"},
                    {'range': [20, 50], 'color': "lightgreen"},
                    {'range': [50, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': signal_strength
                }
            }
        ))

        fig.update_layout(
            template='plotly_dark',
            height=300
        )

        return fig

def main():
    st.title("🚀 Advanced Trading Analytics Dashboard")

    # Initialize dashboard
    if 'dashboard' not in st.session_state:
        st.session_state.dashboard = TradingDashboard()

    dashboard = st.session_state.dashboard

    # Sidebar
    with st.sidebar:
        st.header("Configuration")

        symbol = st.selectbox(
            "Select Symbol",
            ["XAUUSD", "EURUSD", "GBPUSD", "BTCUSD"],
            index=0
        )

        timeframe = st.selectbox(
            "Timeframe",
            ["1min", "5min", "15min", "30min", "1h", "4h", "1d"],
            index=2
        )

        st.divider()

        # Real-time toggle
        enable_realtime = st.checkbox("Enable Real-time Updates", value=True)
        update_interval = st.slider("Update Interval (seconds)", 1, 10, 3)

        st.divider()

        # Analysis options
        st.subheader("Analysis Options")
        show_microstructure = st.checkbox("Microstructure Analysis", value=True)
        show_signals = st.checkbox("Trading Signals", value=True)
        show_market_regime = st.checkbox("Market Regime", value=True)

    # Main content area
    if enable_realtime:
        # Create placeholder for real-time updates
        placeholder = st.empty()

        while True:
            with placeholder.container():
                # Fetch data asynchronously
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                # Market snapshot
                snapshot = loop.run_until_complete(
                    dashboard.fetch_market_snapshot(symbol)
                )

                if snapshot:
                    # Display metrics
                    col1, col2, col3, col4, col5 = st.columns(5)

                    with col1:
                        st.metric(
                            "Price",
                            f"${snapshot['current_price']:,.2f}",
                            f"{snapshot['price_change_pct']:.2%}"
                        )

                    with col2:
                        st.metric(
                            "Bid/Ask",
                            f"{snapshot['bid']:.2f}/{snapshot['ask']:.2f}",
                            f"Spread: {snapshot['spread']:.4f}"
                        )

                    with col3:
                        st.metric(
                            "Volume",
                            f"{snapshot['volume']:,.0f}",
                            f"{snapshot['volume_change_pct']:.1%}"
                        )

                    with col4:
                        st.metric(
                            "RSI",
                            f"{snapshot['rsi']:.1f}",
                            "Overbought" if snapshot['rsi'] > 70 else "Oversold" if snapshot['rsi'] < 30 else "Neutral"
                        )

                    with col5:
                        st.metric(
                            "Volatility",
                            f"{snapshot['volatility']:.2%}",
                            snapshot['volatility_regime']
                        )

                    st.divider()

                    # Main charts
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        # Price chart
                        if 'price_data' in snapshot:
                            df = pd.DataFrame(snapshot['price_data'])
                            fig = dashboard.create_price_chart(df)
                            st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        # Trading signals
                        if show_signals:
                            signals = loop.run_until_complete(
                                dashboard.fetch_trading_signals(symbol)
                            )

                            if signals:
                                st.subheader("Trading Signals")

                                # Signal gauge
                                fig = dashboard.create_signal_gauge(signals['composite_signal'])
                                st.plotly_chart(fig, use_container_width=True)

                                # Signal details
                                signal_class = "buy-signal" if signals['composite_signal'] > 20 else "sell-signal" if signals['composite_signal'] < -20 else "neutral-signal"

                                st.markdown(f"""
                                <div class="signal-box {signal_class}">
                                    <h4>{signals['recommendation']}</h4>
                                    <p>Confidence: {signals['confidence']:.1%}</p>
                                    <p>Risk Level: {signals['risk_level']}</p>
                                </div>
                                """, unsafe_allow_html=True)

                                # Entry/Exit points
                                if signals['entry_price']:
                                    st.info(f"Entry: ${signals['entry_price']:,.2f}")
                                    st.success(f"Target: ${signals['target_price']:,.2f}")
                                    st.error(f"Stop Loss: ${signals['stop_loss']:,.2f}")

                    # Microstructure analysis
                    if show_microstructure:
                        st.divider()
                        st.subheader("Microstructure Analysis")

                        micro_data = loop.run_until_complete(
                            dashboard.fetch_microstructure_analysis(symbol)
                        )

                        if micro_data:
                            fig = dashboard.create_microstructure_chart(micro_data)
                            st.plotly_chart(fig, use_container_width=True)

                            # Microstructure metrics
                            col1, col2, col3, col4 = st.columns(4)

                            with col1:
                                st.metric(
                                    "Effective Spread",
                                    f"{micro_data['current_effective_spread']:.6f}",
                                    f"{micro_data['spread_change']:.2%}"
                                )

                            with col2:
                                st.metric(
                                    "Price Impact",
                                    f"{micro_data['price_impact']:.6f}",
                                    "High" if micro_data['price_impact'] > 0.001 else "Low"
                                )

                            with col3:
                                st.metric(
                                    "Liquidity Score",
                                    f"{micro_data['liquidity_score']:.2f}",
                                    "Good" if micro_data['liquidity_score'] > 0.7 else "Poor"
                                )

                            with col4:
                                st.metric(
                                    "Toxicity",
                                    f"{micro_data['toxicity_score']:.2f}",
                                    "High" if micro_data['toxicity_score'] > 0.5 else "Low"
                                )

                    # Market regime
                    if show_market_regime:
                        st.divider()
                        st.subheader("Market Regime Analysis")

                        col1, col2 = st.columns(2)

                        with col1:
                            st.info(f"Current Regime: **{snapshot.get('market_regime', 'Unknown')}**")
                            st.info(f"Trend Strength: **{snapshot.get('trend_strength', 0):.2f}**")

                        with col2:
                            st.info(f"Volatility Regime: **{snapshot.get('volatility_regime', 'Normal')}**")
                            st.info(f"Market Phase: **{snapshot.get('market_phase', 'Unknown')}**")

                loop.close()

            # Wait before next update
            time.sleep(update_interval)

            # Check if real-time is still enabled
            if not enable_realtime:
                break
    else:
        st.info("Real-time updates are disabled. Enable them in the sidebar to see live data.")

if __name__ == "__main__":
    main()
