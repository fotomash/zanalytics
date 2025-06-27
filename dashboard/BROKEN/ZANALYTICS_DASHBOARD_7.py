#!/usr/bin/env python3
"""
ZANFLOW ULTIMATE MEGA DASHBOARD (v16)
--------------------------------------
The definitive, all-in-one trading analysis platform. This version merges the features 
from all previous iterations into a single, cohesive, and powerful tool.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from pathlib import Path
from datetime import datetime, timedelta
import warnings
import os
import glob
from typing import Dict, List, Optional, Tuple, Any
import re
try:
    from scipy import stats
except ImportError:
    stats = None

warnings.filterwarnings('ignore')

class ZanflowMegaDashboard:
    def __init__(self, data_directory="."):
        self.data_dir = Path(data_directory)
        self.symbols_data = {}
        self.available_symbols = []
        self.available_timeframes = {}
        self.txt_reports = {}

    def scan_available_symbols(self):
        if not self.data_dir.exists():
            return False
        self.symbols_data = {}
        csv_files = list(self.data_dir.rglob("*_processed.csv"))
        if not csv_files:
            return False
        for file_path in csv_files:
            try:
                stem = file_path.stem
                base_name = stem.replace('_csv_processed', '').replace('_processed', '')
                parts = base_name.split('_')
                if len(parts) < 2: continue
                symbol = parts[0]
                timeframe = parts[-1]
                if symbol not in self.symbols_data:
                    self.symbols_data[symbol] = {}
                if timeframe not in self.symbols_data[symbol]:
                    self.symbols_data[symbol][timeframe] = file_path
            except Exception:
                continue
        self.available_symbols = list(self.symbols_data.keys())
        self.available_timeframes = {s: list(t.keys()) for s, t in self.symbols_data.items()}
        return len(self.available_symbols) > 0
    
    def load_symbol_data(self, symbol, timeframe):
        try:
            file_path = self.symbols_data.get(symbol, {}).get(timeframe)
            if file_path and file_path.exists():
                df = pd.read_csv(file_path)
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                elif df.columns[0] == 'Unnamed: 0':
                    df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
                    df.set_index(df.columns[0], inplace=True)
                return df
            return None
        except Exception as e:
            st.error(f"Error loading data for {symbol} {timeframe}: {e}")
            return None

    def load_txt_report(self, symbol):
        try:
            txt_files = glob.glob(str(self.data_dir / f"*{symbol}*Report*.txt"))
            if txt_files:
                latest_file = max(txt_files, key=os.path.getmtime)
                with open(latest_file, 'r') as f:
                    return f.read()
            return None
        except Exception:
            return None

    def create_mega_dashboard(self):
        st.set_page_config(
            page_title="ZANFLOW ULTIMATE MEGA DASHBOARD", 
            page_icon="👑", 
            layout="wide",
            initial_sidebar_state="expanded"
        )
        st.markdown('''
        <style>
            .main-header { background: linear-gradient(135deg, #1F2937 0%, #111827 100%); padding: 2rem; border-radius: 15px; color: white; text-align: center; margin-bottom: 2rem; box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37); }
            .stTabs [data-baseweb="tab-list"] { gap: 24px; }
            .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #F3F4F6; border-radius: 8px; gap: 8px; }
            .stTabs [aria-selected="true"] { background-color: #4B5563; color: white; }
        </style>
        ''', unsafe_allow_html=True)
        st.markdown('<div class="main-header"><h1>👑 ZANFLOW ULTIMATE MEGA DASHBOARD</h1><p>The Definitive, All-in-One Trading Analysis Platform</p></div>', unsafe_allow_html=True)
        if not hasattr(st.session_state, 'symbols_scanned'):
            with st.spinner("🔥 Initializing ZANFLOW Engine... Scanning data sources..."):
                if self.scan_available_symbols():
                    st.session_state.symbols_scanned = True
                else:
                    st.error("❌ No trading data found. Please upload your '*_processed.csv' files.")
                    return
        self.create_sidebar()
        if st.session_state.get('selected_symbol') and st.session_state.get('selected_timeframe'):
            self.display_symbol_analysis_view()
        else:
            self.display_global_market_overview()

    def create_sidebar(self):
        st.sidebar.markdown("# 🎛️ Control Center")
        if self.available_symbols:
            selected_symbol = st.sidebar.selectbox("📈 Select Symbol", [""] + sorted(self.available_symbols), key="selected_symbol")
            if selected_symbol:
                available_tfs = self.available_timeframes.get(selected_symbol, [])
                selected_timeframe = st.sidebar.selectbox("⏱️ Select Timeframe", [""] + sorted(available_tfs), key="selected_timeframe")
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🔬 Analysis Sections")
        st.session_state['show_multitimeframe'] = st.sidebar.checkbox("Multi-Timeframe", True)
        st.session_state['show_microstructure'] = st.sidebar.checkbox("Microstructure & SMC", True)
        st.session_state['show_wyckoff'] = st.sidebar.checkbox("Wyckoff & Patterns", True)
        st.session_state['show_economics'] = st.sidebar.checkbox("Economic Data", True)
        st.session_state['show_risk'] = st.sidebar.checkbox("Risk & Volume", True)
        st.session_state['show_advanced'] = st.sidebar.checkbox("🔬 Advanced Analytics", True)
        st.session_state['show_report'] = st.sidebar.checkbox("📄 Conclusive Report", True)
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📈 Chart Settings")
        st.session_state['lookback_bars'] = st.sidebar.slider("Lookback Period", 50, 2000, 500)
        st.session_state['chart_theme'] = st.sidebar.selectbox("Chart Theme", ["plotly_dark", "plotly_white", "plotly"])

    def display_symbol_analysis_view(self):
        symbol = st.session_state['selected_symbol']
        timeframe = st.session_state['selected_timeframe']
        df = self.load_symbol_data(symbol, timeframe)
        if df is None:
            return
        lookback = st.session_state.get('lookback_bars', 500)
        df_display = df.tail(lookback).copy()
        self.display_realtime_status_bar(df_display)
        self.create_main_price_chart(df_display, symbol, timeframe)
        self.create_analysis_tabs(df_display, symbol)

    def create_analysis_tabs(self, df, symbol):
        tabs_to_show = []
        if st.session_state.get('show_multitimeframe'): tabs_to_show.append("⏱️ Multi-Timeframe")
        if st.session_state.get('show_microstructure'): tabs_to_show.append("🔬 Microstructure & SMC")
        if st.session_state.get('show_wyckoff'): tabs_to_show.append("📈 Wyckoff & Patterns")
        if st.session_state.get('show_economics'): tabs_to_show.append("🌍 Economic Data")
        if st.session_state.get('show_risk'): tabs_to_show.append("⚠️ Risk & Volume")
        if st.session_state.get('show_advanced'): tabs_to_show.append("🤖 Advanced Analytics")
        if st.session_state.get('show_report'): tabs_to_show.append("📄 Conclusive Report")
        if not tabs_to_show:
            return
        tabs = st.tabs(tabs_to_show)
        tab_map = {name: tab for name, tab in zip(tabs_to_show, tabs)}
        if "⏱️ Multi-Timeframe" in tab_map:
            with tab_map["⏱️ Multi-Timeframe"]: self.create_multitimeframe_analysis(symbol)
        if "🔬 Microstructure & SMC" in tab_map:
            with tab_map["🔬 Microstructure & SMC"]: self.create_comprehensive_smc_analysis(df)
        if "📈 Wyckoff & Patterns" in tab_map:
            with tab_map["📈 Wyckoff & Patterns"]: 
                self.create_comprehensive_wyckoff_analysis(df)
                self.create_pattern_analysis(df)
        if "🌍 Economic Data" in tab_map:
            with tab_map["🌍 Economic Data"]: self.create_economic_data_section()
        if "⚠️ Risk & Volume" in tab_map:
            with tab_map["⚠️ Risk & Volume"]: 
                self.create_risk_analysis(df)
                self.create_advanced_volume_analysis(df)
        if "🤖 Advanced Analytics" in tab_map:
            with tab_map["🤖 Advanced Analytics"]: self.create_advanced_analytics_panel(df)
        if "📄 Conclusive Report" in tab_map:
            with tab_map["📄 Conclusive Report"]: self.display_txt_report_analysis(symbol)

    def create_main_price_chart(self, df, symbol, timeframe):
        st.markdown(f"### {symbol} | {timeframe} Price Action")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.8, 0.2])
        fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="Price"), row=1, col=1)
        ma_colors = {'EMA_8': '#ff6b6b', 'EMA_21': '#4ecdc4', 'SMA_200': '#96ceb4'}
        for ma, color in ma_colors.items():
            if ma in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df[ma], mode='lines', name=ma, line=dict(color=color, width=1.5)), row=1, col=1)
        if 'volume' in df.columns:
            colors = ['#26A69A' if c >= o else '#EF5350' for c, o in zip(df['close'], df['open'])]
            fig.add_trace(go.Bar(x=df.index, y=df['volume'], name='Volume', marker_color=colors, opacity=0.7), row=2, col=1)
        fig.update_layout(
            template=st.session_state.get('chart_theme', 'plotly_dark'),
            height=600,
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    def display_realtime_status_bar(self, df):
        metrics = self.calculate_market_metrics(df)
        latest = df.iloc[-1]
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("💰 Price", f"{latest['close']:.4f}", f"{latest['close'] - df.iloc[-2]['close']:.4f}")
        col2.metric("📏 Spread", metrics.get('spread', 'N/A'))
        col3.metric("📊 ATR", metrics.get('atr', 'N/A'))
        col4.metric("📈 RSI", metrics.get('rsi', 'N/A'))
        col5.metric("🛡️ Manipulation", metrics.get('manipulation', 'N/A'))
        st.markdown("---")

    def display_global_market_overview(self):
        st.markdown("## 🌍 Global Market Overview")
        overview_data = {}
        for symbol in self.available_symbols:
            if '1H' in self.available_timeframes.get(symbol, []):
                df = self.load_symbol_data(symbol, '1H')
                if df is not None:
                    overview_data[symbol] = df
        if not overview_data:
            st.warning("Cannot generate Global Overview. Requires '1H' data for at least one symbol.")
            return
        tab1, tab2, tab3 = st.tabs(["🔥 Market Heatmap", "📊 Top Movers", "🔗 Correlation Matrix"])
        with tab1: self.create_market_heatmap(overview_data)
        with tab2: self.create_top_movers_analysis(overview_data)
        with tab3: self.create_correlation_matrix(overview_data)

    def create_market_heatmap(self, overview_data):
        heatmap_data = []
        for symbol, df in overview_data.items():
            if len(df) > 24:
                performance = ((df['close'].iloc[-1] / df['close'].iloc[-24]) - 1) * 100
                heatmap_data.append({'Symbol': symbol, 'Performance (24H)': performance})
        if heatmap_data:
            heatmap_df = pd.DataFrame(heatmap_data).set_index('Symbol')
            fig = px.imshow(heatmap_df, color_continuous_scale='RdYlGn', aspect="auto", title="24H Market Performance Heatmap (%)")
            st.plotly_chart(fig, use_container_width=True)

    def create_top_movers_analysis(self, overview_data):
        movers_data = []
        for symbol, df in overview_data.items():
            if len(df) > 24:
                performance = ((df['close'].iloc[-1] / df['close'].iloc[-24]) - 1) * 100
                volatility = df['close'].pct_change().tail(24).std() * 100
                movers_data.append({'Symbol': symbol, 'Performance': performance, 'Volatility': volatility})
        if movers_data:
            movers_df = pd.DataFrame(movers_data)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 📈 Top Gainers (24H)")
                st.dataframe(movers_df.nlargest(5, 'Performance'))
            with col2:
                st.markdown("#### 📉 Top Losers (24H)")
                st.dataframe(movers_df.nsmallest(5, 'Performance'))

    def create_correlation_matrix(self, overview_data):
        returns_data = pd.DataFrame({symbol: df['close'].pct_change() for symbol, df in overview_data.items()}).dropna()
        if len(returns_data) > 50:
            correlation_matrix = returns_data.corr()
            fig = px.imshow(correlation_matrix, color_continuous_scale='RdBu_r', title="Symbol Correlation Matrix (1H Returns)")
            st.plotly_chart(fig, use_container_width=True)

    def create_multitimeframe_analysis(self, symbol):
        st.subheader("Multi-Timeframe Trend Alignment")
        available_tfs = self.available_timeframes.get(symbol, [])
        if len(available_tfs) < 2:
            st.info("Requires at least two timeframes for analysis.")
            return
        trends = []
        for tf in available_tfs:
            df = self.load_symbol_data(symbol, tf)
            if df is not None and len(df) >= 21:
                short_ma = df['close'].rolling(8).mean().iloc[-1]
                long_ma = df['close'].rolling(21).mean().iloc[-1]
                trends.append({'Timeframe': tf, 'Trend (8/21 EMA)': "🟢 Bullish" if short_ma > long_ma else "🔴 Bearish"})
        if trends:
            st.dataframe(pd.DataFrame(trends), use_container_width=True)

    def create_comprehensive_smc_analysis(self, df):
        st.subheader("Smart Money Concepts & Microstructure")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🟢 Bullish Signals (Last 50 Bars)")
            if 'bullish_fvg' in df.columns: st.metric("Fair Value Gaps", int(df['bullish_fvg'].tail(50).sum()))
            if 'bullish_order_block' in df.columns: st.metric("Order Blocks", int(df['bullish_order_block'].tail(50).sum()))
        with col2:
            st.markdown("#### 🔴 Bearish Signals (Last 50 Bars)")
            if 'bearish_fvg' in df.columns: st.metric("Fair Value Gaps", int(df['bearish_fvg'].tail(50).sum()))
            if 'bearish_order_block' in df.columns: st.metric("Order Blocks", int(df['bearish_order_block'].tail(50).sum()))

    def create_comprehensive_wyckoff_analysis(self, df):
        st.subheader("Wyckoff Analysis")
        if 'wyckoff_phase' in df.columns:
            phase_counts = df['wyckoff_phase'].value_counts(normalize=True) * 100
            phase_names = {1: 'Accumulation', 2: 'Distribution', 3: 'Markup', 4: 'Markdown'}
            phase_data = {phase_names.get(p, 'Unknown'): f"{v:.1f}%" for p, v in phase_counts.items()}
            st.json(phase_data)
            st.metric("Current Phase", phase_names.get(df['wyckoff_phase'].iloc[-1], "Unknown"))

    def create_pattern_analysis(self, df):
        st.subheader("Candlestick Pattern Recognition")
        pattern_cols = [c for c in df.columns if any(p in c.lower() for p in ['doji', 'hammer', 'engulfing'])]
        if pattern_cols:
            recent_patterns = df[pattern_cols].tail(50).sum().astype(int)
            st.dataframe(recent_patterns[recent_patterns > 0].rename("Count in last 50 bars"))
        else:
            st.info("No candlestick pattern data found.")

    def create_economic_data_section(self):
        st.subheader("Economic Data & Events")
        st.info("This section is a placeholder for economic data integration.")
        st.text_input("Enter Finnhub API Key (Not stored)", type="password")
        st.markdown("Features to be added: Upcoming economic events, key rates, market sentiment indices.")

    def create_risk_analysis(self, df):
        st.subheader("Risk Analysis")
        if len(df) > 1:
            returns = df['close'].pct_change().dropna()
            col1, col2, col3 = st.columns(3)
            col1.metric("Annualized Volatility", f"{returns.std() * np.sqrt(252) * 100:.2f}%")
            col2.metric("Value at Risk (95%)", f"{returns.quantile(0.05) * 100:.2f}%")
            cumulative_returns = (1 + returns).cumprod()
            drawdown = (cumulative_returns / cumulative_returns.cummax() - 1) * 100
            col3.metric("Max Drawdown", f"{drawdown.min():.2f}%")
            fig = px.area(drawdown, title="Portfolio Drawdown (%)", labels={'value': 'Drawdown (%)', 'index': 'Date'})
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    def create_advanced_volume_analysis(self, df):
        st.subheader("Volume Analysis")
        if 'volume' in df.columns:
            try:
                price_bins = pd.cut(df['close'], bins=30)
                volume_profile = df.groupby(price_bins)['volume'].sum()
                bin_midpoints = [interval.mid for interval in volume_profile.index.categories]
                fig = go.Figure(go.Bar(x=volume_profile.values, y=bin_midpoints, orientation='h'))
                fig.update_layout(title="Volume Profile", yaxis_title="Price", xaxis_title="Volume")
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.warning("Could not generate Volume Profile.")

    def display_txt_report_analysis(self, symbol):
        st.subheader("Conclusive Analysis Report")
        txt_content = self.load_txt_report(symbol)
        if txt_content:
            st.text_area("Latest Report", txt_content, height=400)
        else:
            st.info(f"No analysis report found for {symbol}.")

    def create_advanced_analytics_panel(self, df):
        st.subheader("Advanced Analytics Engine")
        if stats is None:
            st.error("Statistical analysis disabled because 'scipy' is not installed.")
            return
        adv_tab1, adv_tab2, adv_tab3 = st.tabs(["📊 Statistical Analysis", "🤖 ML Features", "🎯 Signal Analysis"])
        with adv_tab1: self.create_statistical_analysis(df)
        with adv_tab2: self.create_ml_features_analysis(df)
        with adv_tab3: self.create_signal_analysis(df)

    def create_statistical_analysis(self, df):
        st.markdown("#### Return Distribution Statistics")
        if len(df) > 20:
            returns = df['close'].pct_change().dropna()
            if returns.empty:
                st.warning("Not enough data for statistical analysis.")
                return
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Sharpe Ratio (Annualized)", f"{(returns.mean() / returns.std()) * np.sqrt(252):.3f}")
                st.metric("Skewness", f"{returns.skew():.3f}")
                st.metric("Kurtosis", f"{returns.kurtosis():.3f}")
            with col2:
                jb_stat, jb_pvalue = stats.jarque_bera(returns)
                st.metric("Jarque-Bera Stat", f"{jb_stat:.3f}")
                st.metric("Normality P-value", f"{jb_pvalue:.3f}")
                st.success("Distribution is Normal" if jb_pvalue > 0.05 else "Distribution is NOT Normal")

    def create_ml_features_analysis(self, df):
        st.markdown("#### Machine Learning Feature Summary")
        tech_indicators = [c for c in df.columns if any(i in c.upper() for i in ['RSI', 'MACD', 'ATR', 'BB', 'EMA', 'SMA'])]
        pattern_features = [c for c in df.columns if any(p in c.upper() for p in ['FVG', 'ORDER_BLOCK', 'WYCKOFF', 'STRUCTURE'])]
        st.metric("Available Technical Indicators", len(tech_indicators))
        st.metric("Available Pattern Features", len(pattern_features))
        with st.expander("View Feature List"):
            st.json({'Technical Indicators': tech_indicators, 'Pattern Features': pattern_features})

    def create_signal_analysis(self, df):
        st.markdown("#### Trading Signal Analysis")
        signal_cols = [c for c in df.columns if 'signal' in c.lower()]
        if signal_cols:
            st.markdown("##### Recent Signals")
            for col in signal_cols:
                recent_signals = df[df[col] != 0].tail(5)
                if not recent_signals.empty:
                    st.markdown(f"**{col.replace('_', ' ').title()}**")
                    for idx, row in recent_signals.iterrows():
                        st.text(f"{idx.strftime('%Y-%m-%d %H:%M')}: {'BUY' if row[col] > 0 else 'SELL'} at {row['close']:.4f}")
        else:
            st.info("No signal columns found in the data.")

    def calculate_market_metrics(self, df):
        try:
            if df is None or len(df) < 2: return {}
            metrics = {}
            latest = df.iloc[-1]
            if 'spread' in df.columns: metrics['spread'] = f"{latest['spread']:.2f}"
            else: metrics['spread'] = f"{(df['high'] - df['low']).tail(20).mean():.4f}"
            atr_val = next((latest[c] for c in ['atr', 'atr_14', 'ATR_14'] if c in df.columns and pd.notna(latest[c])), np.nan)
            if pd.isna(atr_val):
                tr = pd.concat([df['high'] - df['low'], abs(df['high'] - df['close'].shift()), abs(df['low'] - df['close'].shift())], axis=1).max(axis=1)
                atr_val = tr.rolling(14).mean().iloc[-1]
            metrics['atr'] = f"{atr_val:.4f}"
            rsi_val = next((latest[c] for c in ['rsi', 'rsi_14', 'RSI_14'] if c in df.columns and pd.notna(latest[c])), np.nan)
            if pd.isna(rsi_val) and len(df) > 14:
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                if loss.iloc[-1] != 0:
                    rs = gain.iloc[-1] / loss.iloc[-1]
                    rsi_val = 100 - (100 / (1 + rs))
            if pd.notna(rsi_val):
                color = "🟢" if rsi_val < 30 else "🔴" if rsi_val > 70 else "🟡"
                metrics['rsi'] = f"{color} {rsi_val:.1f}"
            else: metrics['rsi'] = "N/A"
            score = sum(df[c].tail(50).sum() for c in ['spoofing_detected', 'layering_detected'] if c in df.columns)
            if score > 10: metrics['manipulation'] = "🔴 HIGH"
            elif score > 5: metrics['manipulation'] = "🟡 MEDIUM"
            else: metrics['manipulation'] = "🟢 LOW"
            return metrics
        except Exception:
            return {'spread': 'ERR', 'atr': 'ERR', 'rsi': 'ERR', 'manipulation': 'ERR'}

def main():
    dashboard = ZanflowMegaDashboard()
    dashboard.create_mega_dashboard()

if __name__ == "__main__":
    main()
