#!/usr/bin/env python3
"""
ZanFlow Ultimate Trading Dashboard v2.0
This dashboard is a powerful viewer for pre-enriched data files from the ZANZIBAR system,
featuring a multi-page layout with tabbed sub-pages for focused analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from pathlib import Path
import re
from typing import Dict, List
import warnings

warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(
    page_title="ZanFlow Ultimate Trading Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.2rem;
    }
    .metric-card {
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)

# --- Analysis Classes (as provided by the user) ---
# These classes will be used to enrich data on-the-fly if it hasn't been pre-processed.

class MicrostructureAnalysis:
    @staticmethod
    def analyze(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        print("🔬 Running Microstructure Analysis...")
        try:
            if 'mid' not in df.columns and 'bid' in df.columns and 'ask' in df.columns:
                df['mid'] = (df['bid'] + df['ask']) / 2
            
            df = MicrostructureAnalysis._calculate_spread_dynamics(df)
            df = MicrostructureAnalysis._analyze_order_flow(df)
            df = MicrostructureAnalysis._detect_stop_hunts(df)
            df = MicrostructureAnalysis._detect_quote_stuffing(df)
            df = MicrostructureAnalysis._detect_spoofing_and_layering(df)
            df = MicrostructureAnalysis._calculate_manipulation_score(df)
        except Exception as e:
            print(f"  ⚠️ Error in Microstructure Analysis: {e}")
        return df
    
    @staticmethod
    def _calculate_spread_dynamics(df: pd.DataFrame) -> pd.DataFrame:
        if 'bid' in df.columns and 'ask' in df.columns:
            df['micro_spread'] = df['ask'] - df['bid']
            df['micro_spread_ma'] = df['micro_spread'].rolling(window=20).mean()
            df['micro_spread_std'] = df['micro_spread'].rolling(window=20).std()
            df['micro_spread_zscore'] = (df['micro_spread'] - df['micro_spread_ma']) / df['micro_spread_std'].replace(0, np.nan)
        return df
    
    @staticmethod
    def _analyze_order_flow(df: pd.DataFrame) -> pd.DataFrame:
        if 'mid' in df.columns:
            df['micro_price_change'] = df['mid'].diff()
            df['micro_buy_pressure'] = (df['micro_price_change'] > 0).astype(int)
            df['micro_sell_pressure'] = (df['micro_price_change'] < 0).astype(int)
            window = 20
            df['micro_buy_volume'] = df['micro_buy_pressure'].rolling(window=window).sum()
            df['micro_sell_volume'] = df['micro_sell_pressure'].rolling(window=window).sum()
            df['micro_total_ticks'] = df['micro_buy_volume'] + df['micro_sell_volume']
            df['micro_order_flow_imbalance'] = np.where(df['micro_total_ticks'] > 0, (df['micro_buy_volume'] - df['micro_sell_volume']) / df['micro_total_ticks'], 0)
        return df
    
    @staticmethod
    def _detect_stop_hunts(df: pd.DataFrame) -> pd.DataFrame:
        df['micro_stop_hunt'] = False
        window = 20
        if 'high' in df.columns and 'low' in df.columns and 'mid' in df.columns:
            for i in range(window, len(df) - 5):
                segment = df.iloc[i-window:i]
                future_segment = df.iloc[i+1:i+6]
                recent_high = segment['high'].max()
                recent_low = segment['low'].min()
                if df['high'].iloc[i] > recent_high and future_segment['mid'].min() < recent_high:
                    df.loc[df.index[i], 'micro_stop_hunt'] = True
                elif df['low'].iloc[i] < recent_low and future_segment['mid'].max() > recent_low:
                    df.loc[df.index[i], 'micro_stop_hunt'] = True
        return df

    @staticmethod
    def _detect_quote_stuffing(df: pd.DataFrame) -> pd.DataFrame:
        if 'timestamp' in df.columns:
            df['timestamp_sec'] = pd.to_datetime(df['timestamp']).dt.floor('S')
            quote_freq = df.groupby('timestamp_sec')['timestamp'].count()
            high_freq_seconds = quote_freq[quote_freq > 20].index # Threshold for stuffing
            df['micro_quote_stuffing'] = df['timestamp_sec'].isin(high_freq_seconds)
        return df

    @staticmethod
    def _detect_spoofing_and_layering(df: pd.DataFrame) -> pd.DataFrame:
        df['micro_spoofing_layering'] = False
        if 'micro_spread_std' in df.columns and not df['micro_spread_std'].isnull().all():
            spread_spike = df['micro_spread'] > (df['micro_spread_ma'] + 2 * df['micro_spread_std'])
            for i in range(10, len(df) - 5):
                if spread_spike.iloc[i] and df['micro_spread'].iloc[i+1:i+6].min() < df['micro_spread_ma'].iloc[i]:
                    df.loc[df.index[i], 'micro_spoofing_layering'] = True
        return df

    @staticmethod
    def _calculate_manipulation_score(df: pd.DataFrame) -> pd.DataFrame:
        score = pd.Series(0, index=df.index)
        if 'micro_stop_hunt' in df.columns: score += df['micro_stop_hunt'].astype(int) * 0.4
        if 'micro_spoofing_layering' in df.columns: score += df['micro_spoofing_layering'].astype(int) * 0.4
        if 'micro_quote_stuffing' in df.columns: score += df['micro_quote_stuffing'].astype(int) * 0.2
        df['micro_manipulation_score'] = score
        return df

class SMCAnalysis:
    @staticmethod
    def analyze(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        print("💡 Running SMC Analysis...")
        try:
            df = SMCAnalysis._identify_market_structure(df)
            df = SMCAnalysis._find_order_blocks(df)
            df = SMCAnalysis._detect_fair_value_gaps(df)
        except Exception as e:
            print(f"  ⚠️ Error in SMC Analysis: {e}")
        return df

    @staticmethod
    def _identify_market_structure(df: pd.DataFrame) -> pd.DataFrame:
        window = 10
        df['SMC_swing_high'] = (df['high'] == df['high'].rolling(window*2+1, center=True).max())
        df['SMC_swing_low'] = (df['low'] == df['low'].rolling(window*2+1, center=True).min())
        df['SMC_structure'] = 'neutral'
        swing_highs = df[df['SMC_swing_high']]['high'].dropna()
        swing_lows = df[df['SMC_swing_low']]['low'].dropna()
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            if swing_highs.iloc[-1] > swing_highs.iloc[-2] and swing_lows.iloc[-1] > swing_lows.iloc[-2]:
                df['SMC_structure'] = 'bullish'
            elif swing_highs.iloc[-1] < swing_highs.iloc[-2] and swing_lows.iloc[-1] < swing_lows.iloc[-2]:
                df['SMC_structure'] = 'bearish'
        return df

    @staticmethod
    def _find_order_blocks(df: pd.DataFrame) -> pd.DataFrame:
        df['SMC_bullish_ob'] = False
        df['SMC_bearish_ob'] = False
        if 'open' in df.columns and 'close' in df.columns:
            body_size = abs(df['close'] - df['open'])
            is_bullish_candle = df['close'] > df['open']
            is_bearish_candle = df['close'] < df['open']
            df['SMC_bullish_ob'] = is_bearish_candle & is_bullish_candle.shift(-1) & (body_size.shift(-1) > body_size.rolling(10).mean().shift(-1))
            df['SMC_bearish_ob'] = is_bullish_candle & is_bearish_candle.shift(-1) & (body_size.shift(-1) > body_size.rolling(10).mean().shift(-1))
        return df

    @staticmethod
    def _detect_fair_value_gaps(df: pd.DataFrame) -> pd.DataFrame:
        df['SMC_fvg_bullish'] = False
        df['SMC_fvg_bearish'] = False
        for i in range(2, len(df)):
            if df['low'].iloc[i-2] > df['high'].iloc[i]:
                df.loc[df.index[i-1], 'SMC_fvg_bullish'] = True
            if df['high'].iloc[i-2] < df['low'].iloc[i]:
                df.loc[df.index[i-1], 'SMC_fvg_bearish'] = True
        return df

class WyckoffAnalysis:
    @staticmethod
    def analyze(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        print("📈 Running Wyckoff Analysis...")
        try:
            df = WyckoffAnalysis._detect_phases(df)
            df = WyckoffAnalysis._detect_spring_upthrust(df)
            df = WyckoffAnalysis._identify_wyckoff_events(df)
        except Exception as e:
            print(f"  ⚠️ Error in Wyckoff Analysis: {e}")
        return df

    @staticmethod
    def _detect_phases(df: pd.DataFrame) -> pd.DataFrame:
        window = 50
        df['wyckoff_accumulation'] = False
        df['wyckoff_distribution'] = False
        volatility = (df['high'] - df['low']).rolling(window).mean()
        volume_ma = df['volume'].rolling(window).mean() if 'volume' in df.columns else pd.Series(0, index=df.index)
        is_ranging = volatility < volatility.rolling(window*2).mean()
        is_high_volume = df.get('volume', pd.Series(0, index=df.index)) > volume_ma * 1.5
        df['wyckoff_accumulation'] = is_ranging & is_high_volume
        df['wyckoff_distribution'] = is_ranging & is_high_volume & (df['close'] > df['close'].rolling(window).mean())
        return df

    @staticmethod
    def _detect_spring_upthrust(df: pd.DataFrame) -> pd.DataFrame:
        df['wyckoff_spring'] = False
        df['wyckoff_upthrust'] = False
        support = df['low'].rolling(50).min().shift(1)
        resistance = df['high'].rolling(50).max().shift(1)
        df['wyckoff_spring'] = (df['low'] < support) & (df['close'] > support)
        df['wyckoff_upthrust'] = (df['high'] > resistance) & (df['close'] < resistance)
        return df

    @staticmethod
    def _identify_wyckoff_events(df: pd.DataFrame) -> pd.DataFrame:
        df['wyckoff_phase'] = 'neutral'
        if 'wyckoff_accumulation' in df.columns: df.loc[df['wyckoff_accumulation'], 'wyckoff_phase'] = 'accumulation'
        if 'wyckoff_distribution' in df.columns: df.loc[df['wyckoff_distribution'], 'wyckoff_phase'] = 'distribution'
        if 'wyckoff_spring' in df.columns: df.loc[df['wyckoff_spring'], 'wyckoff_phase'] = 'spring'
        if 'wyckoff_upthrust' in df.columns: df.loc[df['wyckoff_upthrust'], 'wyckoff_phase'] = 'upthrust'
        return df

# --- Data Loading & Caching ---

@st.cache_data(ttl=300)
def load_and_cache_data(data_folder="./data"):
    """Scans for data, loads it, and caches the result."""
    data_files = {}
    data_path = Path(data_folder)
    if not data_path.exists():
        return data_files

    for file_path in data_path.rglob('*.csv'):
        try:
            match = re.search(r'([A-Z]+(?:_[A-Z]+)*)_(M\d+|H\d+|D\d+|W\d+)_bars_COMPREHENSIVE_(\w+)\.csv', file_path.name, re.IGNORECASE)
            if match:
                pair_name, _, timeframe = match.groups()
            else:
                parts = file_path.stem.split('_')
                pair_name = parts[0]
                timeframe = parts[1] if len(parts) > 1 else 'Unknown'

            df = pd.read_csv(file_path)
            time_col = next((col for col in df.columns if 'time' in col.lower()), 'timestamp')
            df.rename(columns={time_col: 'timestamp'}, inplace=True)
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df.dropna(subset=['timestamp'], inplace=True)
            df = df.sort_values('timestamp').reset_index(drop=True)

            for col in ['open', 'high', 'low', 'close']:
                if col not in df.columns:
                    df[col] = df.get('mid', 0)

            if pair_name not in data_files:
                data_files[pair_name] = {}
            data_files[pair_name][timeframe] = df
        except Exception as e:
            print(f"Error loading {file_path.name}: {e}")
    return data_files

@st.cache_data(ttl=300)
def run_analysis(_df: pd.DataFrame):
    """Runs all analysis modules on a dataframe and caches the result."""
    df = _df.copy()
    df = MicrostructureAnalysis.analyze(df)
    df = SMCAnalysis.analyze(df)
    df = WyckoffAnalysis.analyze(df)
    return df

# --- UI Components ---

def create_main_plot(df: pd.DataFrame, pair_name: str, timeframe: str) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.8, 0.2])
    fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'), row=1, col=1)
    if 'volume' in df.columns:
        colors = ['green' if c >= o else 'red' for c, o in zip(df['close'], df['open'])]
        fig.add_trace(go.Bar(x=df['timestamp'], y=df['volume'], name='Volume', marker_color=colors, showlegend=False), row=2, col=1)
    fig.update_layout(height=700, template=st.session_state.chart_theme, xaxis_rangeslider_visible=False, hovermode='x unified', title=f"{pair_name} [{timeframe}] Price Action")
    return fig

def create_smc_plot(df: pd.DataFrame, pair_name: str) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.8, 0.2],
                        subplot_titles=(f"{pair_name} - SMC Events", "Order Flow Imbalance"))
    fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price', showlegend=False), row=1, col=1)
    if 'SMC_bullish_ob' in df.columns and df['SMC_bullish_ob'].sum() > 0:
        fig.add_trace(go.Scatter(x=df[df['SMC_bullish_ob']]['timestamp'], y=df[df['SMC_bullish_ob']]['low'], mode='markers', name='Bullish OB', marker=dict(symbol='square', size=10, color='green')), row=1, col=1)
    if 'SMC_bearish_ob' in df.columns and df['SMC_bearish_ob'].sum() > 0:
        fig.add_trace(go.Scatter(x=df[df['SMC_bearish_ob']]['timestamp'], y=df[df['SMC_bearish_ob']]['high'], mode='markers', name='Bearish OB', marker=dict(symbol='square', size=10, color='red')), row=1, col=1)
    if 'SMC_fvg_bullish' in df.columns and df['SMC_fvg_bullish'].sum() > 0:
        bullish_fvgs = df[df['SMC_fvg_bullish']]
        for idx, row in bullish_fvgs.iterrows():
            if idx >= 2:
                fig.add_shape(type="rect", x0=df.loc[idx-2, 'timestamp'], x1=df.loc[idx, 'timestamp'], y0=df.loc[idx, 'high'], y1=df.loc[idx-2, 'low'], fillcolor="green", opacity=0.2, line_width=0, row=1, col=1)
    if 'micro_order_flow_imbalance' in df.columns:
        colors = ['red' if x < 0 else 'green' for x in df['micro_order_flow_imbalance']]
        fig.add_trace(go.Bar(x=df['timestamp'], y=df['micro_order_flow_imbalance'], name='Order Flow', marker_color=colors, showlegend=False), row=2, col=1)
    fig.update_layout(height=800, template=st.session_state.chart_theme, xaxis_rangeslider_visible=False, hovermode='x unified')
    return fig

def generate_commentary(df: pd.DataFrame, pair_name: str, analysis_type: str) -> str:
    commentary = [f"### {pair_name} {analysis_type} Analysis"]
    if df.empty: return "\n".join(commentary)
    
    if analysis_type == "Microstructure":
        if 'micro_spread' in df.columns: commentary.append(f"- **Average Spread**: {df['micro_spread'].mean():.4f}")
        if 'micro_manipulation_score' in df.columns:
            avg_manipulation = df['micro_manipulation_score'].mean()
            commentary.append(f"- **Average Manipulation Score**: {avg_manipulation:.3f}")
            if avg_manipulation > 0.3: commentary.append("  - ⚠️ **High manipulation detected**.")
    
    elif analysis_type == "SMC & Wyckoff":
        if 'SMC_structure' in df.columns: commentary.append(f"- **Market Structure**: {df['SMC_structure'].iloc[-1].upper()}")
        if 'wyckoff_phase' in df.columns:
            dominant_phase = df['wyckoff_phase'].value_counts().idxmax() if not df['wyckoff_phase'].value_counts().empty else 'neutral'
            commentary.append(f"- **Dominant Wyckoff Phase**: {dominant_phase.upper()}")
        if 'SMC_bullish_ob' in df.columns: commentary.append(f"- **Bullish/Bearish OBs**: {df['SMC_bullish_ob'].sum()} / {df['SMC_bearish_ob'].sum()}")
        if 'SMC_fvg_bullish' in df.columns: commentary.append(f"- **Bullish/Bearish FVGs**: {df['SMC_fvg_bullish'].sum()} / {df['SMC_fvg_bearish'].sum()}")
            
    return "\n".join(commentary)

# --- Main App UI Pages ---

class ZanflowDashboard:
    def __init__(self):
        self.initialize_session_state()
        self.data_files = load_and_cache_data()

    def initialize_session_state(self):
        defaults = {
            'current_page': '🏠 Home',
            'selected_pair': None,
            'selected_timeframe': None,
            'lookback_bars': 500,
            'chart_theme': 'plotly_dark'
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def run(self):
        st.sidebar.title("🧭 Navigation")
        pages = {
            "🏠 Home": self.home_page,
            "🔬 Microstructure": self.microstructure_page,
            "🎯 SMC & Wyckoff": self.smc_wyckoff_page,
            "⚙️ Settings": self.settings_page
        }
        st.session_state.current_page = st.sidebar.radio("Go to", list(pages.keys()), key="main_nav")
        
        page_function = pages[st.session_state.current_page]
        page_function()
        
        st.sidebar.markdown("---")
        st.sidebar.info("ZanFlow Ultimate Trading Dashboard v2.0")
        st.sidebar.caption("Data loaded from local files.")

    def home_page(self):
        st.title("🚀 ZanFlow Ultimate Trading Dashboard")
        st.markdown("Welcome to the central hub for the ZANZIBAR trading system. All enriched data sources are loaded and ready for analysis.")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📊 Available Pairs & Timeframes")
            if self.data_files:
                for pair, timeframes in self.data_files.items():
                    st.markdown(f"- **{pair}**: {', '.join(timeframes.keys())}")
            else:
                st.warning("No data files found.")
        with col2:
            st.markdown("### 📖 Analysis Modules")
            st.markdown("- **🔬 Microstructure**: Order flow, manipulation, etc.\n- **🎯 SMC & Wyckoff**: Institutional concepts.")
        st.info("📁 Select a page from the sidebar to begin analysis.")

    def _display_analysis_page(self, page_title: str, plot_function, commentary_function):
        st.title(page_title)
        if not self.data_files:
            st.warning("No data files found. Please add your enriched CSV files to the `./data` folder.")
            return

        available_pairs = list(self.data_files.keys())
        if not st.session_state.selected_pair or st.session_state.selected_pair not in available_pairs:
            st.session_state.selected_pair = available_pairs[0] if available_pairs else None
        
        selected_pair = st.selectbox("Select Trading Pair", available_pairs, index=available_pairs.index(st.session_state.selected_pair) if st.session_state.selected_pair in available_pairs else 0, key=f"sb_pair_{page_title}")
        st.session_state.selected_pair = selected_pair
        
        if selected_pair:
            available_tfs = list(self.data_files[selected_pair].keys())
            if not st.session_state.selected_timeframe or st.session_state.selected_timeframe not in available_tfs:
                st.session_state.selected_timeframe = available_tfs[0] if available_tfs else None
            
            selected_tf = st.selectbox("Select Timeframe", available_tfs, index=available_tfs.index(st.session_state.selected_timeframe) if st.session_state.selected_timeframe in available_tfs else 0, key=f"sb_tf_{page_title}")
            st.session_state.selected_timeframe = selected_tf
            
            if selected_tf:
                df_raw = self.data_files[selected_pair][selected_tf]
                with st.spinner(f"Running {page_title} analysis..."):
                    df_analyzed = run_analysis(df_raw)

                tab1, tab2 = st.tabs(["📊 Chart & Metrics", "📝 Detailed Commentary"])
                with tab1:
                    st.plotly_chart(plot_function(df_analyzed, selected_pair), use_container_width=True)
                with tab2:
                    st.markdown(generate_commentary(df_analyzed, selected_pair, page_title.split(" ")[1]))

    def microstructure_page(self):
        self._display_analysis_page("🔬 Microstructure Analysis", create_main_plot, generate_commentary)

    def smc_wyckoff_page(self):
        self._display_analysis_page("🎯 SMC & Wyckoff Analysis", create_smc_plot, generate_commentary)

    def settings_page(self):
        st.title("⚙️ Settings & Configuration")
        st.info("This dashboard is a viewer for pre-analyzed data. Analysis parameters should be configured in your enrichment scripts.")
        st.text_input("Data Folder Path", value="./data", disabled=True)
        st.session_state.lookback_bars = st.slider("Chart Lookback (Bars)", 100, 2000, st.session_state.get('lookback_bars', 500))
        st.session_state.chart_theme = st.selectbox("Chart Theme", ['plotly_dark', 'plotly_white', 'ggplot2', 'seaborn'], index=['plotly_dark', 'plotly_white', 'ggplot2', 'seaborn'].index(st.session_state.get('chart_theme', 'plotly_dark')))
        st.success("Settings updated.")

# --- Main App Execution ---
if __name__ == "__main__":
    app = ZanflowDashboard()
    app.run()
