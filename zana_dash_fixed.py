import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')
from core.data.manager import get_data_manager

# Page configuration
st.set_page_config(
    page_title="ZanFlow Trading Dashboard",
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
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_pair' not in st.session_state:
    st.session_state.current_pair = 'XAUUSD'
if 'timeframe' not in st.session_state:
    st.session_state.timeframe = '5m'
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}

class TradingDashboard:
    def __init__(self):
        self.data_dir = "./data"
        self.pairs = self.get_available_pairs()
        self.timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
        self.dm = get_data_manager()

    def get_available_pairs(self):
        """Get list of available trading pairs from data directory"""
        pairs = []
        if os.path.exists(self.data_dir):
            for item in os.listdir(self.data_dir):
                if os.path.isdir(os.path.join(self.data_dir, item)):
                    pairs.append(item)
        return sorted(pairs) if pairs else ['XAUUSD']

    def load_data(self, pair: str, timeframe: str) -> pd.DataFrame:
        """Load data for specific pair and timeframe using DataManager."""

        try:
            df = self.dm.get_data(
                data_type="ohlc_data",
                symbol=pair,
                timeframe=timeframe,
                format="csv",
            )
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return pd.DataFrame()

    def calculate_smc_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Smart Money Concepts features"""
        if df.empty:
            return df

        # Order blocks
        df['order_block_bull'] = ((df['close'] > df['open']) & 
                                  (df['low'] < df['low'].shift(1)) & 
                                  (df['close'] > df['high'].shift(1)))

        df['order_block_bear'] = ((df['close'] < df['open']) & 
                                  (df['high'] > df['high'].shift(1)) & 
                                  (df['close'] < df['low'].shift(1)))

        # Fair Value Gaps
        df['fvg_bull'] = ((df['low'] > df['high'].shift(2)) & 
                          (df['low'].shift(1) > df['high'].shift(2)))

        df['fvg_bear'] = ((df['high'] < df['low'].shift(2)) & 
                          (df['high'].shift(1) < df['low'].shift(2)))

        # Liquidity levels
        df['liquidity_high'] = df['high'].rolling(window=20).max()
        df['liquidity_low'] = df['low'].rolling(window=20).min()

        # Break of Structure
        df['bos_bull'] = (df['high'] > df['high'].shift(1).rolling(window=10).max())
        df['bos_bear'] = (df['low'] < df['low'].shift(1).rolling(window=10).min())

        return df

    def calculate_wyckoff_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Wyckoff analysis features"""
        if df.empty:
            return df

        # Volume analysis
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']

        # Price spread
        df['spread'] = df['high'] - df['low']
        df['spread_ma'] = df['spread'].rolling(window=20).mean()

        # Effort vs Result
        df['effort_result'] = df['volume'] / (df['spread'] + 0.0001)

        # Accumulation/Distribution
        df['acc_dist'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 0.0001) * df['volume']
        df['acc_dist_cum'] = df['acc_dist'].cumsum()

        # Spring/Upthrust detection
        df['potential_spring'] = ((df['low'] < df['low'].rolling(window=20).min()) & 
                                  (df['close'] > df['open']) & 
                                  (df['volume'] > df['volume_ma']))

        df['potential_upthrust'] = ((df['high'] > df['high'].rolling(window=20).max()) & 
                                    (df['close'] < df['open']) & 
                                    (df['volume'] > df['volume_ma']))

        return df

    def create_main_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create main price chart with indicators"""
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=('Price Action', 'Volume', 'Indicators')
        )

        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price',
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350'
            ),
            row=1, col=1
        )

        # SMC Features
        if 'order_block_bull' in df.columns:
            bull_ob = df[df['order_block_bull']]
            if not bull_ob.empty:
                fig.add_trace(
                    go.Scatter(
                        x=bull_ob.index,
                        y=bull_ob['low'],
                        mode='markers',
                        marker=dict(symbol='triangle-up', size=10, color='green'),
                        name='Bullish OB'
                    ),
                    row=1, col=1
                )

        if 'order_block_bear' in df.columns:
            bear_ob = df[df['order_block_bear']]
            if not bear_ob.empty:
                fig.add_trace(
                    go.Scatter(
                        x=bear_ob.index,
                        y=bear_ob['high'],
                        mode='markers',
                        marker=dict(symbol='triangle-down', size=10, color='red'),
                        name='Bearish OB'
                    ),
                    row=1, col=1
                )

        # Volume
        colors = ['green' if close >= open else 'red' 
                 for close, open in zip(df['close'], df['open'])]

        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['volume'],
                marker_color=colors,
                name='Volume',
                showlegend=False
            ),
            row=2, col=1
        )

        # Volume MA
        if 'volume_ma' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['volume_ma'],
                    line=dict(color='orange', width=2),
                    name='Volume MA'
                ),
                row=2, col=1
            )

        # Wyckoff Accumulation/Distribution
        if 'acc_dist_cum' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['acc_dist_cum'],
                    line=dict(color='purple', width=2),
                    name='Acc/Dist'
                ),
                row=3, col=1
            )

        # Update layout
        fig.update_layout(
            title=f"{st.session_state.current_pair} - {st.session_state.timeframe}",
            xaxis_rangeslider_visible=False,
            height=800,
            showlegend=True,
            legend=dict(x=0, y=1, orientation='h'),
            margin=dict(l=0, r=0, t=30, b=0)
        )

        fig.update_xaxes(title_text="Time", row=3, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="A/D", row=3, col=1)

        return fig

    def calculate_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate key metrics"""
        if df.empty:
            return {}

        current_price = df['close'].iloc[-1]
        prev_close = df['close'].iloc[-2] if len(df) > 1 else current_price

        metrics = {
            'current_price': current_price,
            'change_pct': ((current_price - prev_close) / prev_close * 100) if prev_close != 0 else 0,
            'volume': df['volume'].iloc[-1],
            'high_24h': df['high'].tail(288).max() if len(df) > 288 else df['high'].max(),
            'low_24h': df['low'].tail(288).min() if len(df) > 288 else df['low'].min(),
            'avg_volume': df['volume'].mean(),
            'volatility': df['close'].pct_change().std() * 100
        }

        return metrics

    def run(self):
        """Run the dashboard"""
        st.title("🚀 ZanFlow Trading Dashboard")

        # Sidebar
        with st.sidebar:
            st.header("Settings")

            # Pair selection
            selected_pair = st.selectbox(
                "Trading Pair",
                self.pairs,
                index=self.pairs.index(st.session_state.current_pair) if st.session_state.current_pair in self.pairs else 0
            )
            st.session_state.current_pair = selected_pair

            # Timeframe selection
            selected_tf = st.selectbox(
                "Timeframe",
                self.timeframes,
                index=self.timeframes.index(st.session_state.timeframe) if st.session_state.timeframe in self.timeframes else 1
            )
            st.session_state.timeframe = selected_tf

            # Analysis options
            st.subheader("Analysis Options")
            show_smc = st.checkbox("Show SMC Features", value=True)
            show_wyckoff = st.checkbox("Show Wyckoff Analysis", value=True)
            show_volume_profile = st.checkbox("Show Volume Profile", value=False)

            # Refresh button
            if st.button("🔄 Refresh Data"):
                st.rerun()

        # Load data
        df = self.load_data(st.session_state.current_pair, st.session_state.timeframe)

        if df.empty:
            st.error("No data available for the selected pair and timeframe")
            return

        # Calculate features
        if show_smc:
            df = self.calculate_smc_features(df)
        if show_wyckoff:
            df = self.calculate_wyckoff_features(df)

        # Display metrics
        metrics = self.calculate_metrics(df)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Current Price",
                f"${metrics.get('current_price', 0):.2f}",
                f"{metrics.get('change_pct', 0):.2f}%"
            )
        with col2:
            st.metric(
                "24h High",
                f"${metrics.get('high_24h', 0):.2f}"
            )
        with col3:
            st.metric(
                "24h Low",
                f"${metrics.get('low_24h', 0):.2f}"
            )
        with col4:
            st.metric(
                "Volume",
                f"{metrics.get('volume', 0):,.0f}",
                f"Avg: {metrics.get('avg_volume', 0):,.0f}"
            )

        # Main chart
        st.subheader("Price Chart")
        fig = self.create_main_chart(df)
        st.plotly_chart(fig, use_container_width=True)

        # Additional analysis sections
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Recent Signals")

            # Check for recent signals
            recent_data = df.tail(10)
            signals = []

            if 'order_block_bull' in recent_data.columns:
                bull_signals = recent_data[recent_data['order_block_bull']]
                for idx, row in bull_signals.iterrows():
                    signals.append(f"🟢 Bullish OB at {idx.strftime('%Y-%m-%d %H:%M')}")

            if 'order_block_bear' in recent_data.columns:
                bear_signals = recent_data[recent_data['order_block_bear']]
                for idx, row in bear_signals.iterrows():
                    signals.append(f"🔴 Bearish OB at {idx.strftime('%Y-%m-%d %H:%M')}")

            if signals:
                for signal in signals[-5:]:  # Show last 5 signals
                    st.write(signal)
            else:
                st.write("No recent signals")

        with col2:
            st.subheader("Market Structure")

            # Determine trend
            sma_20 = df['close'].rolling(20).mean()
            sma_50 = df['close'].rolling(50).mean()

            if len(df) >= 50:
                current_sma20 = sma_20.iloc[-1]
                current_sma50 = sma_50.iloc[-1]

                if current_sma20 > current_sma50:
                    st.write("📈 **Trend:** Bullish")
                else:
                    st.write("📉 **Trend:** Bearish")

                st.write(f"**Volatility:** {metrics.get('volatility', 0):.2f}%")

                # Volume analysis
                if 'volume_ratio' in df.columns:
                    current_vol_ratio = df['volume_ratio'].iloc[-1]
                    st.write(f"**Volume Ratio:** {current_vol_ratio:.2f}x")

# Run the dashboard
if __name__ == "__main__":
    dashboard = TradingDashboard()
    dashboard.run()
