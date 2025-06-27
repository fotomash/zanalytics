# real_data_zanalytics_dashboard.py - REAL DATA Streamlit Dashboard
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Page config
st.set_page_config(
    page_title="ZANALYTICS Intelligence Dashboard - LIVE DATA",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f4e79 0%, #2d7dd2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2d7dd2;
    }
    .analysis-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .bullish { color: #28a745; font-weight: bold; }
    .bearish { color: #dc3545; font-weight: bold; }
    .neutral { color: #6c757d; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

class RealDataDashboard:
    def __init__(self):
        self.data_cache = {}
        self.analysis_data = None
        self.ohlc_data = None
        self.tick_data = None
        self.load_real_data()

    def load_real_data(self):
        """Load actual data files"""
        try:
            # Load analysis JSON
            analysis_files = ["analysis_20250623_205333.json"]
            for file in analysis_files:
                if os.path.exists(file):
                    with open(file, 'r') as f:
                        self.analysis_data = json.load(f)
                    st.success(f"✅ Loaded analysis data: {file}")
                    break

            # Load OHLC data
            ohlc_files = ["XAUUSD_M1_500bars_20250623.csv"]
            for file in ohlc_files:
                if os.path.exists(file):
                    # Try both comma and tab separators
                    try:
                        self.ohlc_data = pd.read_csv(file, sep=',')
                    except:
                        self.ohlc_data = pd.read_csv(file, sep='\t')

                    # Convert timestamp
                    if 'timestamp' in self.ohlc_data.columns:
                        self.ohlc_data['timestamp'] = pd.to_datetime(self.ohlc_data['timestamp'])
                        self.ohlc_data.set_index('timestamp', inplace=True)

                    st.success(f"✅ Loaded OHLC data: {file} ({len(self.ohlc_data)} bars)")
                    break

            # Load tick data
            tick_files = ["XAUUSD_TICK.csv"]
            for file in tick_files:
                if os.path.exists(file):
                    try:
                        self.tick_data = pd.read_csv(file, sep=',')
                    except:
                        self.tick_data = pd.read_csv(file, sep='\t')

                    # Convert timestamp
                    if 'timestamp' in self.tick_data.columns:
                        self.tick_data['timestamp'] = pd.to_datetime(self.tick_data['timestamp'])

                    st.success(f"✅ Loaded tick data: {file} ({len(self.tick_data)} ticks)")
                    break

        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")

    def render_header(self):
        """Render dashboard header with real data info"""
        st.markdown("""
        <div class="main-header">
            <h1>🚀 ZANALYTICS INTELLIGENCE DASHBOARD - LIVE DATA</h1>
            <p>Real XAUUSD Trading Data Analysis & Intelligence</p>
        </div>
        """, unsafe_allow_html=True)

        if self.analysis_data:
            data_info = self.analysis_data.get('data_info', {})
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "📊 Data Source", 
                    data_info.get('data_file', 'Unknown'),
                    delta="Real MT5 Export"
                )

            with col2:
                total_bars = data_info.get('total_bars', 0)
                st.metric(
                    "📈 Total Bars", 
                    total_bars,
                    delta=f"{len(self.analysis_data.get('timeframes_analyzed', []))} Timeframes"
                )

            with col3:
                price_range = data_info.get('price_range', {})
                price_move = price_range.get('high', 0) - price_range.get('low', 0)
                st.metric(
                    "💰 Price Range", 
                    f"{price_move:.2f}",
                    delta=f"High: {price_range.get('high', 0)}"
                )

            with col4:
                latest_close = price_range.get('latest_close', 0)
                st.metric(
                    "🎯 Latest Close", 
                    f"{latest_close:.2f}",
                    delta="Live Price"
                )

    def render_real_ohlc_chart(self):
        """Render real OHLC candlestick chart"""
        st.subheader("📈 Real XAUUSD OHLC Chart")

        if self.ohlc_data is not None:
            col1, col2 = st.columns([3, 1])

            with col1:
                # Create candlestick chart with real data
                fig = go.Figure(data=go.Candlestick(
                    x=self.ohlc_data.index,
                    open=self.ohlc_data['open'],
                    high=self.ohlc_data['high'],
                    low=self.ohlc_data['low'],
                    close=self.ohlc_data['close'],
                    name="XAUUSD M1"
                ))

                # Add support/resistance levels from analysis
                if self.analysis_data:
                    smc_data = self.analysis_data.get('results', {}).get('1m', {}).get('smc', {})

                    # Add resistance levels
                    resistance_levels = smc_data.get('order_blocks', {}).get('resistance_levels', [])
                    for level in resistance_levels[:3]:  # Show top 3
                        fig.add_hline(
                            y=level, 
                            line_dash="dash", 
                            line_color="red",
                            annotation_text=f"R: {level:.2f}"
                        )

                    # Add support levels  
                    support_levels = smc_data.get('order_blocks', {}).get('support_levels', [])
                    for level in support_levels[:3]:  # Show top 3
                        fig.add_hline(
                            y=level, 
                            line_dash="dash", 
                            line_color="green",
                            annotation_text=f"S: {level:.2f}"
                        )

                fig.update_layout(
                    title="Real XAUUSD M1 Chart with SMC Levels",
                    xaxis_title="Time",
                    yaxis_title="Price",
                    height=600,
                    showlegend=True
                )

                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Real price statistics
                st.markdown("### 📊 Price Stats")

                latest_bar = self.ohlc_data.iloc[-1]

                st.markdown(f"""
                <div class="analysis-card">
                    <h4>Latest Bar</h4>
                    <p><strong>Open:</strong> {latest_bar['open']:.2f}</p>
                    <p><strong>High:</strong> {latest_bar['high']:.2f}</p>
                    <p><strong>Low:</strong> {latest_bar['low']:.2f}</p>
                    <p><strong>Close:</strong> {latest_bar['close']:.2f}</p>
                    <p><strong>Volume:</strong> {latest_bar.get('volume', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)

                # Price changes
                if len(self.ohlc_data) > 1:
                    prev_close = self.ohlc_data.iloc[-2]['close']
                    price_change = latest_bar['close'] - prev_close
                    change_pct = (price_change / prev_close) * 100

                    change_color = "bullish" if price_change > 0 else "bearish"
                    change_symbol = "📈" if price_change > 0 else "📉"

                    st.markdown(f"""
                    <div class="analysis-card">
                        <h4>{change_symbol} Price Change</h4>
                        <p class="{change_color}">
                            <strong>{price_change:+.2f}</strong><br>
                            <strong>({change_pct:+.2f}%)</strong>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.error("❌ No OHLC data available")

    def render_real_analysis_results(self):
        """Render real analysis results from JSON"""
        st.subheader("🤖 Real Analysis Results")

        if self.analysis_data:
            results = self.analysis_data.get('results', {})

            # Timeframe tabs
            timeframes = list(results.keys())
            tabs = st.tabs([f"📊 {tf.upper()}" for tf in timeframes])

            for i, (tf, tab) in enumerate(zip(timeframes, tabs)):
                with tab:
                    tf_data = results[tf]

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        # Pattern analysis
                        st.markdown("### 🎯 Pattern Recognition")
                        patterns = tf_data.get('patterns', {}).get('counts', {})

                        for pattern_name, pattern_data in patterns.items():
                            if isinstance(pattern_data, dict):
                                total = pattern_data.get('total', 0)
                                bullish = pattern_data.get('bullish', 0)
                                bearish = pattern_data.get('bearish', 0)
                                last_signal = pattern_data.get('last_signal', 0)

                                signal_color = "🟢" if last_signal > 0 else "🔴" if last_signal < 0 else "⚪"

                                st.markdown(f"""
                                <div class="metric-card">
                                    <h5>{signal_color} {pattern_name.title()}</h5>
                                    <p>Total: {total} | Bull: {bullish} | Bear: {bearish}</p>
                                    <p>Last Signal: {last_signal}</p>
                                </div>
                                """, unsafe_allow_html=True)

                    with col2:
                        # Wyckoff analysis
                        st.markdown("### 🔄 Wyckoff Analysis")
                        wyckoff = tf_data.get('wyckoff', {})

                        current_phase = wyckoff.get('current_phase', 'unknown')
                        phases = wyckoff.get('phases', {})

                        phase_colors = {
                            'accumulation': '#28a745',
                            'distribution': '#dc3545', 
                            'neutral': '#6c757d'
                        }

                        st.markdown(f"""
                        <div class="analysis-card">
                            <h4>Current Phase</h4>
                            <p style="color: {phase_colors.get(current_phase, '#000')}; font-size: 1.2em;">
                                <strong>{current_phase.upper()}</strong>
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                        # Phase distribution chart
                        if phases:
                            fig_wyckoff = go.Figure(data=go.Pie(
                                labels=list(phases.keys()),
                                values=list(phases.values()),
                                hole=0.4,
                                marker_colors=['#28a745', '#dc3545', '#6c757d']
                            ))

                            fig_wyckoff.update_layout(
                                title=f"Wyckoff Phases ({tf})",
                                height=300
                            )

                            st.plotly_chart(fig_wyckoff, use_container_width=True)

                    with col3:
                        # Technical indicators
                        st.markdown("### 📊 Technical Indicators")
                        indicators = tf_data.get('indicators', {})

                        for indicator, value in indicators.items():
                            if value is not None:
                                # Color coding for RSI
                                if indicator == 'rsi':
                                    if value > 70:
                                        color_class = "bearish"
                                        signal = "🔴 Overbought"
                                    elif value < 30:
                                        color_class = "bullish"
                                        signal = "🟢 Oversold"
                                    else:
                                        color_class = "neutral"
                                        signal = "⚪ Neutral"

                                    st.markdown(f"""
                                    <div class="metric-card">
                                        <h5>RSI</h5>
                                        <p class="{color_class}">{value:.2f}</p>
                                        <p>{signal}</p>
                                    </div>
                                    """, unsafe_allow_html=True)

                                elif indicator == 'atr':
                                    st.markdown(f"""
                                    <div class="metric-card">
                                        <h5>ATR</h5>
                                        <p>{value:.2f}</p>
                                        <p>Volatility Measure</p>
                                    </div>
                                    """, unsafe_allow_html=True)

                                else:
                                    st.markdown(f"""
                                    <div class="metric-card">
                                        <h5>{indicator.upper()}</h5>
                                        <p>{value:.2f}</p>
                                    </div>
                                    """, unsafe_allow_html=True)

                        # Trend analysis
                        trend = tf_data.get('trend_analysis', {})
                        direction = trend.get('direction', 'unknown')

                        trend_colors = {
                            'strong_bullish': '#28a745',
                            'strong_bearish': '#dc3545',
                            'bullish': '#90ee90',
                            'bearish': '#ffcccb',
                            'neutral': '#6c757d'
                        }

                        trend_icons = {
                            'strong_bullish': '🚀',
                            'strong_bearish': '🔻',
                            'bullish': '📈',
                            'bearish': '📉',
                            'neutral': '➡️'
                        }

                        st.markdown(f"""
                        <div class="analysis-card">
                            <h4>Trend Direction</h4>
                            <p style="color: {trend_colors.get(direction, '#000')}; font-size: 1.1em;">
                                {trend_icons.get(direction, '❓')} <strong>{direction.upper()}</strong>
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.error("❌ No analysis data available")

    def render_tick_analysis(self):
        """Render tick data analysis"""
        st.subheader("🔬 Real Tick Data Analysis")

        if self.tick_data is not None:
            col1, col2 = st.columns(2)

            with col1:
                # Spread analysis
                if 'bid' in self.tick_data.columns and 'ask' in self.tick_data.columns:
                    self.tick_data['spread'] = self.tick_data['ask'] - self.tick_data['bid']
                    self.tick_data['mid_price'] = (self.tick_data['bid'] + self.tick_data['ask']) / 2

                    # Last 100 ticks for performance
                    recent_ticks = self.tick_data.tail(100).copy()

                    fig_spread = go.Figure()
                    fig_spread.add_trace(go.Scatter(
                        x=recent_ticks.index,
                        y=recent_ticks['spread'],
                        mode='lines',
                        name='Spread',
                        line=dict(color='#dc3545', width=1)
                    ))

                    fig_spread.update_layout(
                        title="Real Spread Analysis (Last 100 Ticks)",
                        xaxis_title="Tick Index",
                        yaxis_title="Spread",
                        height=400
                    )

                    st.plotly_chart(fig_spread, use_container_width=True)

                    # Spread statistics
                    avg_spread = recent_ticks['spread'].mean()
                    max_spread = recent_ticks['spread'].max()
                    min_spread = recent_ticks['spread'].min()

                    st.markdown(f"""
                    <div class="analysis-card">
                        <h4>📊 Spread Statistics</h4>
                        <p><strong>Average:</strong> {avg_spread:.4f}</p>
                        <p><strong>Maximum:</strong> {max_spread:.4f}</p>
                        <p><strong>Minimum:</strong> {min_spread:.4f}</p>
                        <p><strong>Ticks Analyzed:</strong> {len(recent_ticks)}</p>
                    </div>
                    """, unsafe_allow_html=True)

            with col2:
                # Price movement analysis
                if 'mid_price' in self.tick_data.columns:
                    recent_ticks['price_change'] = recent_ticks['mid_price'].diff()

                    fig_price = go.Figure()
                    fig_price.add_trace(go.Scatter(
                        x=recent_ticks.index,
                        y=recent_ticks['mid_price'],
                        mode='lines',
                        name='Mid Price',
                        line=dict(color='#2d7dd2', width=2)
                    ))

                    fig_price.update_layout(
                        title="Real Price Movement (Last 100 Ticks)",
                        xaxis_title="Tick Index", 
                        yaxis_title="Mid Price",
                        height=400
                    )

                    st.plotly_chart(fig_price, use_container_width=True)

                    # Price statistics
                    price_vol = recent_ticks['price_change'].std()
                    price_range = recent_ticks['mid_price'].max() - recent_ticks['mid_price'].min()

                    st.markdown(f"""
                    <div class="analysis-card">
                        <h4>💰 Price Statistics</h4>
                        <p><strong>Price Volatility:</strong> {price_vol:.4f}</p>
                        <p><strong>Price Range:</strong> {price_range:.2f}</p>
                        <p><strong>Latest Mid:</strong> {recent_ticks['mid_price'].iloc[-1]:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.error("❌ No tick data available")

    def render_sidebar(self):
        """Render sidebar with real data controls"""
        with st.sidebar:
            st.markdown("### 🔧 Real Data Dashboard")

            # Data source info
            if self.analysis_data:
                st.markdown("### 📊 Data Sources")
                st.success("✅ Analysis JSON loaded")

                analysis_time = self.analysis_data.get('analysis_timestamp', 'Unknown')
                st.info(f"Analysis Time: {analysis_time[:19]}")

            if self.ohlc_data is not None:
                st.success(f"✅ OHLC data: {len(self.ohlc_data)} bars")

            if self.tick_data is not None:
                st.success(f"✅ Tick data: {len(self.tick_data)} ticks")

            st.markdown("---")

            # Refresh controls
            if st.button("🔄 Reload Data"):
                self.load_real_data()
                st.rerun()

            if st.button("📊 Export Analysis"):
                if self.analysis_data:
                    st.download_button(
                        "💾 Download Analysis JSON",
                        data=json.dumps(self.analysis_data, indent=2),
                        file_name=f"zanalytics_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )

            st.markdown("---")

            # File status
            st.markdown("### 📁 File Status")

            files_to_check = [
                "analysis_20250623_205333.json",
                "XAUUSD_M1_500bars_20250623.csv", 
                "XAUUSD_TICK.csv"
            ]

            for file in files_to_check:
                if os.path.exists(file):
                    st.success(f"✅ {file}")
                else:
                    st.error(f"❌ {file}")

    def run(self):
        """Main dashboard runner"""
        self.render_header()
        self.render_sidebar()

        # Main content
        self.render_real_ohlc_chart()

        st.markdown("---")

        self.render_real_analysis_results()

        st.markdown("---")

        self.render_tick_analysis()

def main():
    """Main entry point"""
    dashboard = RealDataDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()
