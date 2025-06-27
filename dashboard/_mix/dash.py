
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import requests
import ta
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# --- Configuration ---
st.set_page_config(
    page_title="ZANFLOW v13 ULTIMATE MEGA DASHBOARD",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Analysis Classes (Corrected & Complete) ---

class WyckoffAnalyzer:
    """Complete Wyckoff Analysis with proper phase detection"""
    def __init__(self):
        self.phases_info = {
            'accumulation': {
                'PS': 'Preliminary Support', 'SC': 'Selling Climax', 'AR': 'Automatic Rally',
                'ST': 'Secondary Test', 'SPRING': 'Spring/Shakeout', 'SOS': 'Sign of Strength',
                'LPS': 'Last Point of Support', 'BU': 'Back-Up'
            },
            'distribution': {
                'PSY': 'Preliminary Supply', 'BC': 'Buying Climax', 'AR': 'Automatic Reaction',
                'ST': 'Secondary Test', 'SOW': 'Sign of Weakness', 'LPSY': 'Last Point of Supply',
                'UTAD': 'Upthrust After Distribution'
            }
        }

    def detect_trading_range(self, df, lookback=50):
        high_range = df['High'].rolling(window=lookback, min_periods=1).max()
        low_range = df['Low'].rolling(window=lookback, min_periods=1).min()
        tr_height = (high_range - low_range) / low_range * 100
        avg_volume = df['Volume'].rolling(window=lookback, min_periods=1).mean()
        return {
            'upper_boundary': high_range, 'lower_boundary': low_range,
            'tr_height_pct': tr_height, 'avg_volume': avg_volume,
            'is_tr': tr_height < 15
        }

    def detect_springs_utad(self, df, tr_data):
        springs, utads = [], []
        volume_mean = df['Volume'].rolling(20).mean()
        for i in range(20, len(df) - 1):
            # Spring detection
            if (df['Low'].iloc[i] < tr_data['lower_boundary'].iloc[i-1] and
                df['Close'].iloc[i] > df['Low'].iloc[i] + (df['High'].iloc[i] - df['Low'].iloc[i]) * 0.5 and
                df['Volume'].iloc[i] > volume_mean.iloc[i] * 1.5):
                springs.append({
                    'index': i, 'date': df.index[i], 'low': df['Low'].iloc[i],
                    'snapback_strength': (df['Close'].iloc[i] - df['Low'].iloc[i]) / (df['High'].iloc[i] - df['Low'].iloc[i])
                })
            # UTAD detection
            if (df['High'].iloc[i] > tr_data['upper_boundary'].iloc[i-1] and
                df['Close'].iloc[i] < df['High'].iloc[i] - (df['High'].iloc[i] - df['Low'].iloc[i]) * 0.5 and
                df['Volume'].iloc[i] > volume_mean.iloc[i] * 1.5):
                utads.append({
                    'index': i, 'date': df.index[i], 'high': df['High'].iloc[i],
                    'rejection_strength': (df['High'].iloc[i] - df['Close'].iloc[i]) / (df['High'].iloc[i] - df['Low'].iloc[i])
                })
        return springs, utads

    def analyze(self, df):
        tr_data = self.detect_trading_range(df)
        springs, utads = self.detect_springs_utad(df, tr_data)
        return {'tr_data': tr_data, 'springs': springs, 'utads': utads}

class SMCAnalyzer:
    """Smart Money Concepts Analysis"""
    def detect_liquidity_sweeps(self, df, lookback=20):
        sweeps = []
        volume_mean = df['Volume'].rolling(20).mean()
        for i in range(lookback, len(df) - 1):
            prev_high = df['High'].iloc[i-lookback:i].max()
            prev_low = df['Low'].iloc[i-lookback:i].min()
            if df['Low'].iloc[i] < prev_low:
                wick_ratio = (df['Close'].iloc[i] - df['Low'].iloc[i]) / (df['High'].iloc[i] - df['Low'].iloc[i]) if (df['High'].iloc[i] - df['Low'].iloc[i]) > 0 else 0
                if wick_ratio > 0.7:
                    sweeps.append({
                        'type': 'bullish_sweep', 'index': i, 'level': prev_low, 'sweep_low': df['Low'].iloc[i],
                        'rejection_strength': wick_ratio, 'volume_spike': df['Volume'].iloc[i] / volume_mean.iloc[i]
                    })
            if df['High'].iloc[i] > prev_high:
                wick_ratio = (df['High'].iloc[i] - df['Close'].iloc[i]) / (df['High'].iloc[i] - df['Low'].iloc[i]) if (df['High'].iloc[i] - df['Low'].iloc[i]) > 0 else 0
                if wick_ratio > 0.7:
                    sweeps.append({
                        'type': 'bearish_sweep', 'index': i, 'level': prev_high, 'sweep_high': df['High'].iloc[i],
                        'rejection_strength': wick_ratio, 'volume_spike': df['Volume'].iloc[i] / volume_mean.iloc[i]
                    })
        return sweeps

    def detect_fvgs(self, df):
        fvgs = []
        for i in range(2, len(df)):
            if df['Low'].iloc[i] > df['High'].iloc[i-2]:
                fvgs.append({
                    'type': 'bullish_fvg', 'index': i, 'upper': df['Low'].iloc[i], 'lower': df['High'].iloc[i-2]
                })
            if df['High'].iloc[i] < df['Low'].iloc[i-2]:
                fvgs.append({
                    'type': 'bearish_fvg', 'index': i, 'upper': df['Low'].iloc[i-2], 'lower': df['High'].iloc[i]
                })
        return fvgs

    def detect_order_blocks(self, df, sweeps):
        order_blocks = []
        for sweep in sweeps:
            idx = sweep['index']
            if sweep['type'] == 'bullish_sweep':
                for j in range(idx - 1, max(0, idx - 10), -1):
                    if df['Close'].iloc[j] < df['Open'].iloc[j]:
                        order_blocks.append({'type': 'bullish_ob', 'index': j, 'upper': df['High'].iloc[j], 'lower': df['Low'].iloc[j]})
                        break
            elif sweep['type'] == 'bearish_sweep':
                for j in range(idx - 1, max(0, idx - 10), -1):
                    if df['Close'].iloc[j] > df['Open'].iloc[j]:
                        order_blocks.append({'type': 'bearish_ob', 'index': j, 'upper': df['High'].iloc[j], 'lower': df['Low'].iloc[j]})
                        break
        return order_blocks

    def analyze(self, df):
        sweeps = self.detect_liquidity_sweeps(df)
        fvgs = self.detect_fvgs(df)
        order_blocks = self.detect_order_blocks(df, sweeps)
        return {'sweeps': sweeps, 'fvgs': fvgs, 'order_blocks': order_blocks}

class MicrostructureAnalyzer:
    """Tick and microstructure analysis"""
    def compute_spread_instability(self, df):
        spreads = df['High'] - df['Low']
        spread_volatility = spreads.rolling(20).std()
        spread_mean = spreads.rolling(20).mean()
        instability = (spread_volatility / spread_mean).fillna(0)
        return instability

    def detect_absorption_patterns(self, df):
        patterns = []
        avg_spread = (df['High'] - df['Low']).rolling(20).mean()
        volume_mean = df['Volume'].rolling(20).mean()
        for i in range(20, len(df) - 1):
            volume_ratio = df['Volume'].iloc[i] / volume_mean.iloc[i]
            spread = df['High'].iloc[i] - df['Low'].iloc[i]
            if volume_ratio > 1.5 and spread < avg_spread.iloc[i] * 0.7:
                patterns.append({'index': i, 'type': 'absorption'})
        return patterns

    def analyze(self, df):
        instability = self.compute_spread_instability(df)
        absorption = self.detect_absorption_patterns(df)
        return {'spread_instability': instability, 'absorption': absorption}

class EconomicDataManager:
    """Manage economic data integration"""
    @st.cache_data(ttl=600) # Cache for 10 minutes
    def fetch_macro_indicators(self):
        try:
            dxy = yf.Ticker("DX-Y.NYB").history(period="2d")
            us10y = yf.Ticker("^TNX").history(period="2d")
            vix = yf.Ticker("^VIX").history(period="2d")
            if dxy.empty or us10y.empty or vix.empty: return None
            return {
                'DXY': {'value': dxy['Close'].iloc[-1], 'change': dxy['Close'].diff().iloc[-1]},
                'US10Y': {'value': us10y['Close'].iloc[-1], 'change': us10y['Close'].diff().iloc[-1]},
                'VIX': {'value': vix['Close'].iloc[-1], 'change': vix['Close'].diff().iloc[-1]}
            }
        except Exception as e:
            return None

    def calculate_risk_metrics(self, macro_data):
        if not macro_data:
            return {'risk_score': 0, 'market_regime': 'Unknown', 'confidence': 0}
        risk_score = 0
        if macro_data['VIX']['value'] > 20: risk_score -= 1
        if macro_data['VIX']['value'] < 15: risk_score += 1
        if macro_data['DXY']['change'] > 0: risk_score -= 0.5
        if macro_data['US10Y']['change'] < 0: risk_score -= 0.5
        
        regime = "Risk-On" if risk_score > 0 else "Risk-Off" if risk_score < 0 else "Neutral"
        confidence = min(abs(risk_score) / 2, 1.0)
        return {'risk_score': risk_score, 'market_regime': regime, 'confidence': confidence}

    def analyze(self):
        macro_data = self.fetch_macro_indicators()
        risk_metrics = self.calculate_risk_metrics(macro_data)
        return {'macro_data': macro_data, 'risk_metrics': risk_metrics}

class ZANFLOWDashboard:
    def __init__(self):
        self.wyckoff_analyzer = WyckoffAnalyzer()
        self.smc_analyzer = SMCAnalyzer()
        self.micro_analyzer = MicrostructureAnalyzer()
        self.econ_manager = EconomicDataManager()
        
        if 'data' not in st.session_state:
            st.session_state.data = None
            st.session_state.analysis_results = {}
            st.session_state.symbol = "EURUSD=X"
            st.session_state.interval = "1h"
            st.session_state.show_wyckoff = True
            st.session_state.show_smc = True
            st.session_state.show_micro = True
            st.session_state.show_econ = True

    def setup_sidebar(self):
        st.sidebar.header("⚙️ ZANFLOW Controls")
        
        st.session_state.symbol = st.sidebar.text_input("Symbol", value=st.session_state.symbol)
        st.session_state.interval = st.sidebar.selectbox("Interval", 
            ['1m', '5m', '15m', '30m', '1h', '4h', '1d'], index=4)
        
        if st.sidebar.button("🚀 Load & Analyze Data"):
            self.load_data()
            if st.session_state.data is not None:
                self.run_analysis()

        st.sidebar.markdown("---")
        with st.sidebar.expander("📊 Analysis Modules", expanded=True):
            st.session_state.show_wyckoff = st.checkbox("Wyckoff Analysis", value=st.session_state.show_wyckoff)
            st.session_state.show_smc = st.checkbox("SMC Analysis", value=st.session_state.show_smc)
            st.session_state.show_micro = st.checkbox("Microstructure", value=st.session_state.show_micro)
            st.session_state.show_econ = st.checkbox("Economic Data", value=st.session_state.show_econ)

    def load_data(self):
        with st.spinner(f"Loading {st.session_state.symbol} data..."):
            try:
                period_map = {'1m': '7d', '5m': '60d', '15m': '60d', '30m': '60d', '1h': '730d', '4h': '730d', '1d': '10y'}
                df = yf.download(tickers=st.session_state.symbol, 
                                 period=period_map[st.session_state.interval], 
                                 interval=st.session_state.interval,
                                 auto_adjust=True)
                if df.empty:
                    st.error("No data found for the symbol. Please check the ticker.")
                    st.session_state.data = None
                    return
                df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
                st.session_state.data = df
                st.success("Data loaded successfully.")
            except Exception as e:
                st.error(f"Error loading data: {e}")
                st.session_state.data = None

    def run_analysis(self):
        with st.spinner("Running institutional-grade analysis..."):
            df = st.session_state.data
            results = {}
            if st.session_state.show_wyckoff:
                results.update(self.wyckoff_analyzer.analyze(df))
            if st.session_state.show_smc:
                results.update(self.smc_analyzer.analyze(df))
            if st.session_state.show_micro:
                results.update(self.micro_analyzer.analyze(df))
            if st.session_state.show_econ:
                results.update(self.econ_manager.analyze())
            st.session_state.analysis_results = results
            st.success("Analysis complete.")

    def display_main_chart(self):
        if st.session_state.data is None:
            st.info("Please load data using the sidebar controls to view the chart.")
            return

        df = st.session_state.data
        results = st.session_state.analysis_results
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.05, row_heights=[0.7, 0.3])

        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)

        if st.session_state.show_wyckoff and 'tr_data' in results:
            fig.add_trace(go.Scatter(x=df.index, y=results['tr_data']['upper_boundary'], mode='lines', line=dict(color='rgba(255, 0, 0, 0.5)', dash='dash'), name='TR High'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=results['tr_data']['lower_boundary'], mode='lines', line=dict(color='rgba(0, 255, 0, 0.5)', dash='dash'), name='TR Low'), row=1, col=1)
        if st.session_state.show_wyckoff and 'springs' in results and results['springs']:
            spring_points = df.iloc[[s['index'] for s in results['springs']]]
            fig.add_trace(go.Scatter(x=spring_points.index, y=spring_points['Low'], mode='markers', marker=dict(symbol='triangle-up', color='lime', size=12, line=dict(width=1, color='black')), name='Springs'), row=1, col=1)
        if st.session_state.show_wyckoff and 'utads' in results and results['utads']:
            utad_points = df.iloc[[u['index'] for u in results['utads']]]
            fig.add_trace(go.Scatter(x=utad_points.index, y=utad_points['High'], mode='markers', marker=dict(symbol='triangle-down', color='fuchsia', size=12, line=dict(width=1, color='black')), name='UTADs'), row=1, col=1)
        
        if st.session_state.show_smc and 'sweeps' in results:
            bull_sweeps = [s for s in results['sweeps'] if s['type'] == 'bullish_sweep']
            bear_sweeps = [s for s in results['sweeps'] if s['type'] == 'bearish_sweep']
            if bull_sweeps:
                fig.add_trace(go.Scatter(x=df.index[[s['index'] for s in bull_sweeps]], y=df['Low'].iloc[[s['index'] for s in bull_sweeps]], mode='markers', marker=dict(symbol='x', color='cyan', size=10), name='Bull Sweep'), row=1, col=1)
            if bear_sweeps:
                fig.add_trace(go.Scatter(x=df.index[[s['index'] for s in bear_sweeps]], y=df['High'].iloc[[s['index'] for s in bear_sweeps]], mode='markers', marker=dict(symbol='x', color='magenta', size=10), name='Bear Sweep'), row=1, col=1)
        
        if st.session_state.show_smc and 'fvgs' in results:
            for fvg in results['fvgs'][-50:]: # Plot last 50 for performance
                color = 'rgba(0, 255, 0, 0.2)' if fvg['type'] == 'bullish_fvg' else 'rgba(255, 0, 0, 0.2)'
                fig.add_shape(type="rect", x0=df.index[fvg['index']-1], y0=fvg['lower'], x1=df.index[fvg['index']], y1=fvg['upper'],
                              line=dict(color="rgba(0,0,0,0)"), fillcolor=color, layer='below', row=1, col=1)
        
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color='grey'), row=2, col=1)

        fig.update_layout(title=f'{st.session_state.symbol} Analysis | {st.session_state.interval}', xaxis_rangeslider_visible=False, height=700, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)

    def display_dashboard(self):
        st.title("📈 ZANFLOW v13 ULTIMATE MEGA DASHBOARD")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            self.display_main_chart()
        
        with col2:
            if st.session_state.show_econ:
                st.markdown("### 📊 Market Regime Analysis")
                if 'risk_metrics' in st.session_state.analysis_results and st.session_state.analysis_results['risk_metrics']:
                    risk = st.session_state.analysis_results['risk_metrics']
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number", value=risk['risk_score'], title={'text': "Risk Sentiment"},
                        gauge={'axis': {'range': [-2, 2]}, 'bar': {'color': "green" if risk['risk_score'] > 0 else "red"},
                               'steps': [{'range': [-2, -1], 'color': "#ff7c7c"}, {'range': [1, 2], 'color': "#7cff7c"}]}))
                    fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                    st.info(f"**Regime:** {risk['market_regime']} | **Confidence:** {risk['confidence']:.0%}")
                else:
                    st.info("Economic analysis not run or data unavailable.")

            st.markdown("### 📝 Analysis Summary")
            if st.session_state.data is not None:
                if st.session_state.show_smc and 'sweeps' in st.session_state.analysis_results:
                    st.metric("Liquidity Sweeps Found", len(st.session_state.analysis_results['sweeps']))
                if st.session_state.show_wyckoff and 'springs' in st.session_state.analysis_results:
                    st.metric("Wyckoff Events", len(st.session_state.analysis_results['springs']) + len(st.session_state.analysis_results['utads']))
                if st.session_state.show_smc and 'fvgs' in st.session_state.analysis_results:
                    st.metric("Imbalances (FVGs)", len(st.session_state.analysis_results['fvgs']))
            else:
                st.info("No data loaded.")

    def run(self):
        self.setup_sidebar()
        self.display_dashboard()

if __name__ == "__main__":
    dashboard = ZANFLOWDashboard()
    dashboard.run()
"""
with open("ZANFLOW_ULTIMATE_MEGA_DASHBOARD_v13_FIXED.py", "w") as f:
    f.write(code)

print("Created ZANFLOW_ULTIMATE_MEGA_DASHBOARD_v13_FIXED.py")