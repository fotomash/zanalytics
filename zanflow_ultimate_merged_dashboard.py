#!/usr/bin/env python3
"""
ZANFLOW ULTIMATE MERGED DASHBOARD v14
The Most Comprehensive Trading Analysis Platform
Combines: Microstructure + SMC + Wyckoff + Patterns + ML + Risk + Multi-Timeframe Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
import glob
from pathlib import Path
from datetime import datetime, timedelta
import warnings
import re
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats
warnings.filterwarnings('ignore')

class UltimateTradingDashboard:
    def __init__(self, data_directory="./data"):
        """Initialize the ultimate trading dashboard"""
        self.data_dir = Path(data_directory)
        self.pairs_data = {}
        self.tick_data = {}
        self.bar_data = {}
        self.analysis_reports = {}
        self.microstructure_data = {}
        self.smc_features = {}

        # Configuration
        self.max_bars = 2000
        self.max_ticks = 5000
        self.supported_pairs = ["XAUUSD", "BTCUSD", "EURUSD", "GBPUSD", "USDJPY"]
        self.timeframes = ["1min", "5min", "15min", "30min", "1H", "4H", "1D"]

        # Color scheme
        self.colors = {
            'bullish': '#26de81',
            'bearish': '#fc5c65', 
            'bullish_ob': 'rgba(38, 222, 129, 0.3)',
            'bearish_ob': 'rgba(252, 92, 101, 0.3)',
            'fvg_bull': 'rgba(162, 155, 254, 0.4)',
            'fvg_bear': 'rgba(253, 121, 168, 0.4)',
            'liquidity': '#45aaf2',
            'supply': 'rgba(255, 107, 107, 0.2)',
            'demand': 'rgba(46, 213, 115, 0.2)',
            'wyckoff_acc': '#00cec9',
            'wyckoff_dist': '#e17055',
            'manipulation': '#ff6b6b'
        }

    def scan_all_data_sources(self):
        """Scan all possible data sources and formats"""
        data_sources = {
            'comprehensive_files': {},
            'summary_files': {},
            'tick_files': {},
            'microstructure_files': {},
            'pair_directories': {}
        }

        # Scan for CSV files (comprehensive data)
        csv_patterns = [
            "*_COMPREHENSIVE_*.csv",
            "*_csv_processed.csv", 
            "*TICK*.csv",
            "*_bars_*.csv"
        ]

        csv_files = []
        for pattern in csv_patterns:
            csv_files.extend(glob.glob(os.path.join(str(self.data_dir), "**", pattern), recursive=True))
            csv_files.extend(glob.glob(pattern, recursive=False))  # Current directory

        # Scan for JSON files (summary data)
        json_patterns = [
            "*_ANALYSIS_REPORT.json",
            "*Microstructure_Analysis*.json",
            "*summary*.json"
        ]

        json_files = []
        for pattern in json_patterns:
            json_files.extend(glob.glob(os.path.join(str(self.data_dir), "**", pattern), recursive=True))
            json_files.extend(glob.glob(pattern, recursive=False))  # Current directory

        # Categorize files by pair
        for pair in self.supported_pairs:
            data_sources['comprehensive_files'][pair] = {}
            data_sources['summary_files'][pair] = {}
            data_sources['tick_files'][pair] = {}
            data_sources['microstructure_files'][pair] = {}

            # Comprehensive CSV files
            pair_csvs = [f for f in csv_files if pair in f and "COMPREHENSIVE" in f]
            for csv_file in pair_csvs:
                # Extract timeframe from filename
                for tf in self.timeframes:
                    if tf in csv_file:
                        data_sources['comprehensive_files'][pair][tf] = csv_file
                        break

            # Tick data files
            tick_csvs = [f for f in csv_files if pair in f and ("TICK" in f or "tick" in f)]
            for tick_file in tick_csvs:
                data_sources['tick_files'][pair][os.path.basename(tick_file)] = tick_file

            # Bar data files
            bar_csvs = [f for f in csv_files if pair in f and ("bars" in f or any(tf in f for tf in self.timeframes))]
            for bar_file in bar_csvs:
                for tf in self.timeframes:
                    if tf in bar_file:
                        data_sources['comprehensive_files'][pair][tf] = bar_file
                        break

            # Summary JSON files
            pair_jsons = [f for f in json_files if pair in f]
            for json_file in pair_jsons:
                if "Microstructure" in json_file:
                    data_sources['microstructure_files'][pair][os.path.basename(json_file)] = json_file
                else:
                    data_sources['summary_files'][pair][os.path.basename(json_file)] = json_file

        return data_sources

    def load_comprehensive_data(self, file_path, max_records=None):
        """Load comprehensive trading data from CSV"""
        try:
            # Try different separators
            for sep in [',', '\t', ';']:
                try:
                    df = pd.read_csv(file_path, sep=sep)
                    if len(df.columns) > 4:  # Should have OHLC + more
                        break
                except:
                    continue

            # Handle different index formats
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            elif df.index.dtype == 'object':
                try:
                    df.index = pd.to_datetime(df.index)
                except:
                    pass

            # Limit records if specified
            if max_records and len(df) > max_records:
                df = df.tail(max_records)

            # Ensure required columns exist
            required_cols = ['open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required_cols):
                return None

            return df

        except Exception as e:
            st.warning(f"Error loading {file_path}: {str(e)}")
            return None

    def load_json_data(self, file_path):
        """Load JSON analysis data"""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.warning(f"Error loading JSON {file_path}: {str(e)}")
            return {}

    def extract_smc_features(self, df, summary_data=None):
        """Extract comprehensive SMC features"""
        features = {
            'order_blocks': {'bullish': [], 'bearish': []},
            'liquidity_zones': [],
            'fair_value_gaps': {'bullish': [], 'bearish': []},
            'supply_demand_zones': {'supply': [], 'demand': []},
            'bos_choch': {'bos': [], 'choch': []},
            'market_structure': {
                'trend': 'neutral',
                'strength': 50,
                'higher_highs': 0,
                'lower_lows': 0
            },
            'signals': []
        }

        # Extract from dataframe columns
        if 'bullish_fvg' in df.columns:
            fvg_bullish = df[df['bullish_fvg'] == True]
            for idx, row in fvg_bullish.iterrows():
                features['fair_value_gaps']['bullish'].append({
                    'timestamp': idx,
                    'price': row['close'],
                    'size': row.get('fvg_size', 0)
                })

        if 'bearish_fvg' in df.columns:
            fvg_bearish = df[df['bearish_fvg'] == True]
            for idx, row in fvg_bearish.iterrows():
                features['fair_value_gaps']['bearish'].append({
                    'timestamp': idx,
                    'price': row['close'],
                    'size': row.get('fvg_size', 0)
                })

        # Order blocks
        if 'bullish_order_block' in df.columns:
            ob_bullish = df[df['bullish_order_block'] == True]
            for idx, row in ob_bullish.iterrows():
                features['order_blocks']['bullish'].append({
                    'timestamp': idx,
                    'high': row['high'],
                    'low': row['low'],
                    'strength': row.get('ob_strength', 1)
                })

        if 'bearish_order_block' in df.columns:
            ob_bearish = df[df['bearish_order_block'] == True]
            for idx, row in ob_bearish.iterrows():
                features['order_blocks']['bearish'].append({
                    'timestamp': idx,
                    'high': row['high'],
                    'low': row['low'],
                    'strength': row.get('ob_strength', 1)
                })

        # Market structure analysis
        if 'higher_high' in df.columns and 'lower_low' in df.columns:
            features['market_structure']['higher_highs'] = df['higher_high'].sum()
            features['market_structure']['lower_lows'] = df['lower_low'].sum()

            if features['market_structure']['higher_highs'] > features['market_structure']['lower_lows']:
                features['market_structure']['trend'] = 'bullish'
                features['market_structure']['strength'] = 70
            elif features['market_structure']['lower_lows'] > features['market_structure']['higher_highs']:
                features['market_structure']['trend'] = 'bearish'
                features['market_structure']['strength'] = 70

        # Add summary data if available
        if summary_data:
            for key in features.keys():
                if key in summary_data:
                    if isinstance(summary_data[key], dict):
                        features[key].update(summary_data[key])
                    elif isinstance(summary_data[key], list):
                        features[key].extend(summary_data[key])

        return features

    def create_main_dashboard(self):
        """Create the main dashboard interface"""
        st.set_page_config(
            page_title="ZANFLOW Ultimate v14", 
            page_icon="🚀", 
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Enhanced CSS styling
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 20px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }
        .feature-card {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            padding: 1.5rem;
            border-radius: 15px;
            color: white;
            margin: 0.5rem 0;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }
        .metric-premium {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            padding: 1.2rem;
            border-radius: 12px;
            color: white;
            text-align: center;
            margin: 0.5rem 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .smc-alert {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            margin: 1rem 0;
            animation: pulse 2s infinite;
        }
        .wyckoff-indicator {
            background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
            padding: 0.8rem;
            border-radius: 8px;
            color: #2c3e50;
            margin: 0.3rem 0;
            font-weight: bold;
            text-align: center;
        }
        .manipulation-warning {
            background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
            padding: 1rem;
            border-radius: 10px;
            border: 2px solid #ff6b6b;
            margin: 1rem 0;
            animation: blink 1s infinite;
        }
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.02); }
            100% { transform: scale(1); }
        }
        @keyframes blink {
            0%, 50% { opacity: 1; }
            51%, 100% { opacity: 0.7; }
        }
        .data-source-badge {
            background: #2c3e50;
            color: #ecf0f1;
            padding: 0.3rem 0.8rem;
            border-radius: 15px;
            font-size: 0.8rem;
            display: inline-block;
            margin: 0.2rem;
        }
        .analysis-section {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        </style>
        """, unsafe_allow_html=True)

        # Main header
        st.markdown("""
        <div class="main-header">
        <h1>🚀 ZANFLOW Ultimate Trading Intelligence v14</h1>
        <p><strong>The Most Advanced Trading Analysis Platform</strong></p>
        <p>Microstructure • Smart Money Concepts • Wyckoff • ML • Risk Analytics • Multi-Timeframe</p>
        </div>
        """, unsafe_allow_html=True)

        # Initialize and load data
        with st.spinner("🔄 Scanning all data sources..."):
            data_sources = self.scan_all_data_sources()

        # Sidebar configuration
        self.create_enhanced_sidebar(data_sources)

        # Main content based on selection
        if st.session_state.get('analysis_mode') == 'overview':
            self.display_market_overview(data_sources)
        elif st.session_state.get('analysis_mode') == 'pair_analysis':
            self.display_pair_analysis(data_sources)
        elif st.session_state.get('analysis_mode') == 'microstructure':
            self.display_microstructure_analysis(data_sources)
        elif st.session_state.get('analysis_mode') == 'smc_deep':
            self.display_smc_deep_analysis(data_sources)
        elif st.session_state.get('analysis_mode') == 'wyckoff_analysis':
            self.display_wyckoff_analysis(data_sources)
        elif st.session_state.get('analysis_mode') == 'risk_analytics':
            self.display_risk_analytics(data_sources)
        else:
            self.display_welcome_screen(data_sources)

    def create_enhanced_sidebar(self, data_sources):
        """Create enhanced sidebar with all controls"""
        st.sidebar.title("🎛️ Ultimate Control Center")

        # Data source summary
        st.sidebar.markdown("### 📊 Data Sources Found")
        total_comprehensive = sum(len(files) for files in data_sources['comprehensive_files'].values())
        total_tick = sum(len(files) for files in data_sources['tick_files'].values())
        total_microstructure = sum(len(files) for files in data_sources['microstructure_files'].values())

        st.sidebar.markdown(f"""
        <div class="data-source-badge">📈 Comprehensive: {total_comprehensive}</div>
        <div class="data-source-badge">⚡ Tick Data: {total_tick}</div>
        <div class="data-source-badge">🔬 Microstructure: {total_microstructure}</div>
        """, unsafe_allow_html=True)

        # Analysis mode selection
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🔍 Analysis Mode")

        analysis_modes = {
            'overview': '🌍 Market Overview',
            'pair_analysis': '📊 Pair Analysis', 
            'microstructure': '🔬 Microstructure',
            'smc_deep': '🧠 SMC Deep Dive',
            'wyckoff_analysis': '📈 Wyckoff Analysis',
            'risk_analytics': '⚠️ Risk Analytics'
        }

        selected_mode = st.sidebar.radio(
            "Select Analysis Mode",
            list(analysis_modes.keys()),
            format_func=lambda x: analysis_modes[x],
            key="analysis_mode"
        )

        # Pair selection (for single pair analysis)
        if selected_mode in ['pair_analysis', 'microstructure', 'smc_deep', 'wyckoff_analysis']:
            st.sidebar.markdown("---")
            st.sidebar.markdown("### 📈 Pair Selection")

            available_pairs = [pair for pair in self.supported_pairs 
                             if data_sources['comprehensive_files'].get(pair)]

            if available_pairs:
                selected_pair = st.sidebar.selectbox(
                    "Select Trading Pair",
                    available_pairs,
                    key="selected_pair"
                )

                # Timeframe selection
                available_timeframes = list(data_sources['comprehensive_files'][selected_pair].keys())
                if available_timeframes:
                    selected_timeframe = st.sidebar.selectbox(
                        "Select Timeframe",
                        available_timeframes,
                        key="selected_timeframe"
                    )
            else:
                st.sidebar.error("No pairs with data found")

        # Feature toggles
        st.sidebar.markdown("---")
        st.sidebar.markdown("### ⚙️ Feature Toggles")

        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.session_state['show_smc'] = st.checkbox("SMC Analysis", True)
            st.session_state['show_wyckoff'] = st.checkbox("Wyckoff", True)
            st.session_state['show_patterns'] = st.checkbox("Patterns", True)
            st.session_state['show_volume'] = st.checkbox("Volume", True)

        with col2:
            st.session_state['show_microstructure'] = st.checkbox("Microstructure", True)
            st.session_state['show_manipulation'] = st.checkbox("Manipulation", True)
            st.session_state['show_risk'] = st.checkbox("Risk Metrics", True)
            st.session_state['show_ml'] = st.checkbox("ML Features", True)

        # Display settings
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🎨 Display Settings")

        st.session_state['chart_theme'] = st.sidebar.selectbox(
            "Chart Theme",
            ["plotly_dark", "plotly_white", "ggplot2", "seaborn"],
            key="theme_select"
        )

        st.session_state['max_bars'] = st.sidebar.slider(
            "Max Bars to Display",
            100, 3000, self.max_bars,
            key="bars_slider"
        )

        # Export options
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 💾 Quick Actions")

        if st.sidebar.button("🔄 Refresh Data", key="refresh_all"):
            st.rerun()

        if st.sidebar.button("📊 Export Analysis", key="export_analysis"):
            st.session_state['show_export'] = True

    def display_welcome_screen(self, data_sources):
        """Display welcome screen with data overview"""
        st.markdown("## 🎯 Welcome to ZANFLOW Ultimate v14")

        # Feature overview
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div class="feature-card">
            <h3>🔬 Advanced Microstructure</h3>
            <ul>
            <li>Tick-level analysis</li>
            <li>Manipulation detection</li>
            <li>Order flow analysis</li>
            <li>Liquidity mapping</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="feature-card">
            <h3>🧠 Smart Money Concepts</h3>
            <ul>
            <li>Order blocks detection</li>
            <li>Fair value gaps (FVG)</li>
            <li>BOS/CHoCH analysis</li>
            <li>Supply/demand zones</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="feature-card">
            <h3>📈 Wyckoff Analysis</h3>
            <ul>
            <li>Accumulation/Distribution</li>
            <li>Volume spread analysis</li>
            <li>Effort vs Result</li>
            <li>Phase detection</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

        # Data overview
        st.markdown("## 📊 Data Overview")

        # Create data summary
        data_summary = []
        for pair in self.supported_pairs:
            comprehensive_count = len(data_sources['comprehensive_files'].get(pair, {}))
            tick_count = len(data_sources['tick_files'].get(pair, {}))
            microstructure_count = len(data_sources['microstructure_files'].get(pair, {}))

            if comprehensive_count > 0 or tick_count > 0 or microstructure_count > 0:
                data_summary.append({
                    'Pair': pair,
                    'Comprehensive Files': comprehensive_count,
                    'Tick Files': tick_count,
                    'Microstructure Files': microstructure_count,
                    'Total': comprehensive_count + tick_count + microstructure_count
                })

        if data_summary:
            summary_df = pd.DataFrame(data_summary)
            st.dataframe(summary_df, use_container_width=True)
        else:
            st.warning("No trading data found. Please ensure data files are in the ./data directory or current folder.")

        # Quick start guide
        st.markdown("## 🚀 Quick Start Guide")

        st.markdown("""
        1. **📊 Data Setup**: Place your CSV/JSON files in the `./data` directory or current folder
        2. **🔍 Analysis Mode**: Select your preferred analysis mode from the sidebar
        3. **📈 Pair Selection**: Choose the trading pair and timeframe you want to analyze
        4. **⚙️ Features**: Toggle the analysis features you want to see
        5. **🎨 Customize**: Adjust display settings and chart themes
        6. **📊 Analyze**: Explore the comprehensive analysis results
        """)

    def display_market_overview(self, data_sources):
        """Display comprehensive market overview"""
        st.markdown("## 🌍 Market Overview & Intelligence")

        # Load data for all available pairs
        market_data = {}
        for pair in self.supported_pairs:
            if data_sources['comprehensive_files'].get(pair):
                # Get the highest timeframe available
                timeframes = data_sources['comprehensive_files'][pair]
                if timeframes:
                    latest_tf = list(timeframes.keys())[-1]
                    file_path = timeframes[latest_tf]
                    df = self.load_comprehensive_data(file_path, max_records=500)
                    if df is not None:
                        market_data[pair] = df

        if not market_data:
            st.warning("No market data available for overview")
            return

        # Market metrics
        self.create_market_metrics(market_data)

        # Market heatmap
        self.create_market_heatmap(market_data)

        # Top movers and correlation
        col1, col2 = st.columns(2)
        with col1:
            self.create_top_movers(market_data)
        with col2:
            self.create_correlation_matrix(market_data)

        # Market sentiment analysis
        self.create_market_sentiment(market_data)

    def create_market_metrics(self, market_data):
        """Create market overview metrics"""
        st.markdown("### 📊 Market Metrics")

        col1, col2, col3, col4, col5 = st.columns(5)

        # Calculate aggregate metrics
        total_pairs = len(market_data)
        total_bars = sum(len(df) for df in market_data.values())

        # Average volatility
        avg_volatility = np.mean([
            df['close'].pct_change().std() * np.sqrt(252) * 100
            for df in market_data.values() if len(df) > 1
        ]) if market_data else 0

        # Trending pairs
        trending_pairs = sum(1 for df in market_data.values() 
                           if len(df) > 20 and df['close'].iloc[-1] > df['close'].iloc[-20])

        # High volume pairs
        high_vol_pairs = sum(1 for df in market_data.values()
                           if 'volume' in df.columns and len(df) > 1 and 
                           df['volume'].iloc[-1] > df['volume'].mean())

        with col1:
            st.markdown(f"""
            <div class="metric-premium">
            <h3>📈 Active Pairs</h3>
            <h2>{total_pairs}</h2>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-premium">
            <h3>📊 Total Bars</h3>
            <h2>{total_bars:,}</h2>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-premium">
            <h3>📈 Trending</h3>
            <h2>{trending_pairs}/{total_pairs}</h2>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="metric-premium">
            <h3>🔥 Avg Volatility</h3>
            <h2>{avg_volatility:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)

        with col5:
            st.markdown(f"""
            <div class="metric-premium">
            <h3>📊 High Volume</h3>
            <h2>{high_vol_pairs}/{total_pairs}</h2>
            </div>
            """, unsafe_allow_html=True)

    def create_market_heatmap(self, market_data):
        """Create market performance heatmap"""
        st.markdown("### 🔥 Market Performance Heatmap")

        try:
            performance_data = []

            for pair, df in market_data.items():
                if len(df) > 1:
                    # Different timeframe performances
                    perfs = {}

                    # 1 day (24 bars for hourly data)
                    if len(df) >= 24:
                        perfs['1D'] = ((df['close'].iloc[-1] / df['close'].iloc[-24]) - 1) * 100

                    # 1 week (168 bars for hourly data)
                    if len(df) >= 168:
                        perfs['1W'] = ((df['close'].iloc[-1] / df['close'].iloc[-168]) - 1) * 100

                    # 1 month (720 bars for hourly data)
                    if len(df) >= 720:
                        perfs['1M'] = ((df['close'].iloc[-1] / df['close'].iloc[-720]) - 1) * 100

                    # Overall trend
                    perfs['Overall'] = ((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100

                    if perfs:
                        performance_data.append({'Pair': pair, **perfs})

            if performance_data:
                perf_df = pd.DataFrame(performance_data).set_index('Pair')

                fig = px.imshow(
                    perf_df.values,
                    x=perf_df.columns,
                    y=perf_df.index,
                    color_continuous_scale='RdYlGn',
                    aspect="auto",
                    title="Performance Heatmap (%)",
                    text_auto=True
                )

                fig.update_layout(
                    template=st.session_state.get('chart_theme', 'plotly_dark'),
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error creating heatmap: {e}")

    def create_top_movers(self, market_data):
        """Create top movers analysis"""
        st.markdown("#### 📈 Top Movers")

        movers = []
        for pair, df in market_data.items():
            if len(df) > 24:  # Need at least 24 periods
                recent_performance = ((df['close'].iloc[-1] / df['close'].iloc[-24]) - 1) * 100
                volatility = df['close'].pct_change().tail(24).std() * 100

                movers.append({
                    'Pair': pair,
                    'Performance': recent_performance,
                    'Volatility': volatility,
                    'Current': df['close'].iloc[-1]
                })

        if movers:
            movers_df = pd.DataFrame(movers)

            # Top gainers
            st.markdown("**🟢 Top Gainers**")
            top_gainers = movers_df.nlargest(3, 'Performance')
            for _, row in top_gainers.iterrows():
                st.markdown(f"**{row['Pair']}**: {row['Performance']:+.2f}% | Vol: {row['Volatility']:.2f}%")

            # Top losers
            st.markdown("**🔴 Top Losers**")
            top_losers = movers_df.nsmallest(3, 'Performance')
            for _, row in top_losers.iterrows():
                st.markdown(f"**{row['Pair']}**: {row['Performance']:+.2f}% | Vol: {row['Volatility']:.2f}%")

    def create_correlation_matrix(self, market_data):
        """Create correlation matrix"""
        st.markdown("#### 🔗 Price Correlations")

        try:
            # Get returns for correlation
            returns_data = {}
            for pair, df in market_data.items():
                if len(df) > 50:
                    returns = df['close'].pct_change().dropna().tail(100)
                    if len(returns) > 10:
                        returns_data[pair] = returns

            if len(returns_data) >= 2:
                corr_df = pd.DataFrame(returns_data).corr()

                fig = px.imshow(
                    corr_df.values,
                    x=corr_df.columns,
                    y=corr_df.index,
                    color_continuous_scale='RdBu',
                    aspect="auto",
                    title="Correlation Matrix",
                    text_auto=True
                )

                fig.update_layout(
                    template=st.session_state.get('chart_theme', 'plotly_dark'),
                    height=300
                )

                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error creating correlation matrix: {e}")

    def create_market_sentiment(self, market_data):
        """Create market sentiment analysis"""
        st.markdown("### 🎯 Market Sentiment Analysis")

        sentiment_data = []

        for pair, df in market_data.items():
            if len(df) > 50:
                # RSI sentiment
                if 'rsi_14' in df.columns:
                    current_rsi = df['rsi_14'].iloc[-1]
                    if current_rsi > 70:
                        rsi_sentiment = "Overbought"
                    elif current_rsi < 30:
                        rsi_sentiment = "Oversold"
                    else:
                        rsi_sentiment = "Neutral"
                else:
                    rsi_sentiment = "N/A"

                # Trend sentiment
                if 'ema_8' in df.columns and 'ema_21' in df.columns:
                    trend_sentiment = "Bullish" if df['ema_8'].iloc[-1] > df['ema_21'].iloc[-1] else "Bearish"
                else:
                    # Simple trend analysis
                    trend_sentiment = "Bullish" if df['close'].iloc[-1] > df['close'].iloc[-20] else "Bearish"

                # Volume sentiment
                if 'volume' in df.columns:
                    avg_volume = df['volume'].tail(20).mean()
                    current_volume = df['volume'].iloc[-1]
                    volume_sentiment = "High" if current_volume > avg_volume * 1.2 else "Normal"
                else:
                    volume_sentiment = "N/A"

                sentiment_data.append({
                    'Pair': pair,
                    'RSI Signal': rsi_sentiment,
                    'Trend': trend_sentiment,
                    'Volume': volume_sentiment,
                    'Price': f"{df['close'].iloc[-1]:.4f}"
                })

        if sentiment_data:
            sentiment_df = pd.DataFrame(sentiment_data)
            st.dataframe(sentiment_df, use_container_width=True)

    def display_pair_analysis(self, data_sources):
        """Display comprehensive pair analysis"""
        pair = st.session_state.get('selected_pair')
        timeframe = st.session_state.get('selected_timeframe')

        if not pair or not timeframe:
            st.warning("Please select a pair and timeframe from the sidebar")
            return

        # Load data
        file_path = data_sources['comprehensive_files'][pair][timeframe]
        df = self.load_comprehensive_data(file_path, st.session_state.get('max_bars', self.max_bars))

        if df is None:
            st.error(f"Failed to load data for {pair} {timeframe}")
            return

        # Load summary data if available
        summary_data = {}
        if data_sources['summary_files'][pair]:
            summary_file = list(data_sources['summary_files'][pair].values())[0]
            summary_data = self.load_json_data(summary_file)

        # Extract SMC features
        smc_features = self.extract_smc_features(df, summary_data)

        st.markdown(f"# 🚀 {pair} {timeframe} - Comprehensive Analysis")

        # Market status
        self.display_market_status(df, pair)

        # Main price chart
        self.create_ultimate_price_chart(df, smc_features, pair, timeframe)

        # Analysis sections
        col1, col2 = st.columns(2)

        with col1:
            if st.session_state.get('show_smc', True):
                self.create_smc_analysis_section(df, smc_features)

            if st.session_state.get('show_volume', True):
                self.create_volume_analysis_section(df)

        with col2:
            if st.session_state.get('show_wyckoff', True):
                self.create_wyckoff_analysis_section(df)

            if st.session_state.get('show_risk', True):
                self.create_risk_analysis_section(df)

    def display_market_status(self, df, pair):
        """Display current market status"""
        col1, col2, col3, col4, col5, col6 = st.columns(6)

        current_price = df['close'].iloc[-1]
        prev_price = df['close'].iloc[-2] if len(df) > 1 else current_price
        price_change = current_price - prev_price
        price_change_pct = (price_change / prev_price * 100) if prev_price != 0 else 0

        with col1:
            st.metric("💰 Current Price", f"{current_price:.4f}", f"{price_change:+.4f}")

        with col2:
            st.metric("📊 Change %", f"{price_change_pct:+.2f}%")

        with col3:
            atr = df['atr_14'].iloc[-1] if 'atr_14' in df.columns else 0
            st.metric("📈 ATR(14)", f"{atr:.4f}")

        with col4:
            rsi = df['rsi_14'].iloc[-1] if 'rsi_14' in df.columns else 50
            rsi_status = "🟢" if rsi < 30 else "🔴" if rsi > 70 else "🟡"
            st.metric("📊 RSI", f"{rsi_status} {rsi:.1f}")

        with col5:
            # Trend analysis
            if 'ema_8' in df.columns and 'ema_21' in df.columns:
                trend = "🟢 BULL" if df['ema_8'].iloc[-1] > df['ema_21'].iloc[-1] else "🔴 BEAR"
            else:
                trend = "🟡 UNKNOWN"
            st.metric("📈 Trend", trend)

        with col6:
            # Volatility regime
            volatility = df['close'].pct_change().std() * 100
            vol_regime = "🔥 HIGH" if volatility > 1.0 else "❄️ LOW"
            st.metric("🌊 Volatility", vol_regime)

    def create_ultimate_price_chart(self, df, smc_features, pair, timeframe):
        """Create the ultimate price chart with all overlays"""
        st.markdown("### 📊 Ultimate Price Analysis")

        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=[
                f"{pair} {timeframe} - Price Action & Smart Money Analysis",
                "Volume Analysis", 
                "Technical Indicators"
            ],
            vertical_spacing=0.08,
            row_heights=[0.6, 0.2, 0.2],
            shared_xaxes=True
        )

        # Main candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name="Price",
                increasing_line_color=self.colors['bullish'],
                decreasing_line_color=self.colors['bearish']
            ), row=1, col=1
        )

        # Add moving averages
        ma_lines = ['ema_8', 'ema_21', 'ema_55', 'sma_200']
        ma_colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']

        for ma, color in zip(ma_lines, ma_colors):
            if ma in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[ma],
                    mode='lines', name=ma.upper(),
                    line=dict(color=color, width=2)
                ), row=1, col=1)

        # Add SMC overlays
        if st.session_state.get('show_smc', True):
            self.add_smc_overlays_to_chart(fig, df, smc_features, row=1)

        # Volume analysis
        if 'volume' in df.columns:
            colors = [self.colors['bullish'] if close >= open_val else self.colors['bearish'] 
                     for close, open_val in zip(df['close'], df['open'])]

            fig.add_trace(go.Bar(
                x=df.index, y=df['volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7
            ), row=2, col=1)

            # Volume MA
            if 'volume_sma_20' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['volume_sma_20'],
                    mode='lines', name='Vol MA20',
                    line=dict(color='orange', width=2)
                ), row=2, col=1)

        # Technical indicators
        if 'rsi_14' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df['rsi_14'],
                mode='lines', name='RSI 14',
                line=dict(color='purple', width=2)
            ), row=3, col=1)

            # RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1)

        # Update layout
        fig.update_layout(
            title=f"{pair} {timeframe} Ultimate Analysis",
            template=st.session_state.get('chart_theme', 'plotly_dark'),
            height=900,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )

        st.plotly_chart(fig, use_container_width=True)

    def add_smc_overlays_to_chart(self, fig, df, smc_features, row=1):
        """Add SMC overlays to the chart"""
        # Fair Value Gaps - Bullish
        for fvg in smc_features['fair_value_gaps']['bullish']:
            fig.add_trace(go.Scatter(
                x=[fvg['timestamp']],
                y=[fvg['price']],
                mode='markers',
                name='Bullish FVG',
                marker=dict(symbol='triangle-up', color='lime', size=12),
                showlegend=False
            ), row=row, col=1)

        # Fair Value Gaps - Bearish
        for fvg in smc_features['fair_value_gaps']['bearish']:
            fig.add_trace(go.Scatter(
                x=[fvg['timestamp']],
                y=[fvg['price']],
                mode='markers',
                name='Bearish FVG',
                marker=dict(symbol='triangle-down', color='red', size=12),
                showlegend=False
            ), row=row, col=1)

        # Order Blocks - Bullish
        for ob in smc_features['order_blocks']['bullish']:
            fig.add_trace(go.Scatter(
                x=[ob['timestamp']],
                y=[ob['low']],
                mode='markers',
                name='Bullish OB',
                marker=dict(symbol='square', color='lightgreen', size=10),
                showlegend=False
            ), row=row, col=1)

        # Order Blocks - Bearish
        for ob in smc_features['order_blocks']['bearish']:
            fig.add_trace(go.Scatter(
                x=[ob['timestamp']],
                y=[ob['high']],
                mode='markers',
                name='Bearish OB',
                marker=dict(symbol='square', color='lightcoral', size=10),
                showlegend=False
            ), row=row, col=1)

    def create_smc_analysis_section(self, df, smc_features):
        """Create SMC analysis section"""
        st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
        st.markdown("#### 🧠 Smart Money Concepts Analysis")

        # SMC metrics
        col1, col2 = st.columns(2)

        with col1:
            bullish_fvgs = len(smc_features['fair_value_gaps']['bullish'])
            bearish_fvgs = len(smc_features['fair_value_gaps']['bearish'])
            st.metric("Fair Value Gaps", f"🟢{bullish_fvgs} | 🔴{bearish_fvgs}")

        with col2:
            bullish_obs = len(smc_features['order_blocks']['bullish'])
            bearish_obs = len(smc_features['order_blocks']['bearish'])
            st.metric("Order Blocks", f"🟢{bullish_obs} | 🔴{bearish_obs}")

        # Market structure
        trend = smc_features['market_structure']['trend']
        strength = smc_features['market_structure']['strength']

        if trend == 'bullish':
            st.markdown(f"""
            <div class="smc-alert">
            <strong>🟢 BULLISH STRUCTURE DETECTED</strong><br>
            Strength: {strength}% | Higher Highs: {smc_features['market_structure']['higher_highs']}
            </div>
            """, unsafe_allow_html=True)
        elif trend == 'bearish':
            st.markdown(f"""
            <div class="smc-alert" style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);">
            <strong>🔴 BEARISH STRUCTURE DETECTED</strong><br>
            Strength: {strength}% | Lower Lows: {smc_features['market_structure']['lower_lows']}
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    def create_volume_analysis_section(self, df):
        """Create volume analysis section"""
        if 'volume' not in df.columns:
            return

        st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
        st.markdown("#### 📊 Volume Analysis")

        volume = df['volume']
        avg_volume = volume.mean()
        current_volume = volume.iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Current Volume Ratio", f"{volume_ratio:.2f}x")

            # Volume trend
            volume_trend = volume.rolling(20).mean().pct_change().iloc[-1] * 100
            st.metric("Volume Trend (20)", f"{volume_trend:+.1f}%")

        with col2:
            # Price-Volume correlation
            if len(df) > 20:
                price_change = df['close'].pct_change()
                volume_change = volume.pct_change()
                correlation = price_change.corr(volume_change)
                st.metric("Price-Volume Correlation", f"{correlation:.3f}")

            # Volume volatility
            volume_volatility = volume.pct_change().std()
            st.metric("Volume Volatility", f"{volume_volatility:.3f}")

        st.markdown('</div>', unsafe_allow_html=True)

    def create_wyckoff_analysis_section(self, df):
        """Create Wyckoff analysis section"""
        st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
        st.markdown("#### 📈 Wyckoff Analysis")

        # Check for Wyckoff columns
        wyckoff_cols = [col for col in df.columns if 'wyckoff' in col.lower()]

        if wyckoff_cols:
            # Wyckoff phase analysis
            if 'wyckoff_phase' in df.columns:
                current_phase = df['wyckoff_phase'].iloc[-1]
                phase_names = {1: 'Accumulation', 2: 'Distribution', 3: 'Markup', 4: 'Markdown'}

                if current_phase in phase_names:
                    phase_name = phase_names[current_phase]
                    if current_phase == 1:  # Accumulation
                        st.markdown(f"""
                        <div class="wyckoff-indicator" style="background: linear-gradient(135deg, #00cec9 0%, #55a3ff 100%);">
                        🔵 ACCUMULATION PHASE<br>
                        Smart money quietly building positions
                        </div>
                        """, unsafe_allow_html=True)
                    elif current_phase == 2:  # Distribution
                        st.markdown(f"""
                        <div class="wyckoff-indicator" style="background: linear-gradient(135deg, #e17055 0%, #ff6b6b 100%);">
                        🔴 DISTRIBUTION PHASE<br>
                        Smart money distributing to retail
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.info("No Wyckoff analysis data available in this dataset")

        st.markdown('</div>', unsafe_allow_html=True)

    def create_risk_analysis_section(self, df):
        """Create risk analysis section"""
        st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
        st.markdown("#### ⚠️ Risk Analysis")

        if len(df) > 20:
            returns = df['close'].pct_change().dropna()

            # Risk metrics
            volatility = returns.std() * np.sqrt(252) * 100
            var_95 = np.percentile(returns, 5) * 100

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Annualized Volatility", f"{volatility:.2f}%")
                st.metric("VaR (95%)", f"{var_95:.3f}%")

            with col2:
                skewness = returns.skew()
                kurtosis = returns.kurtosis()
                st.metric("Skewness", f"{skewness:.3f}")
                st.metric("Kurtosis", f"{kurtosis:.3f}")

            # Drawdown analysis
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min() * 100

            st.metric("Maximum Drawdown", f"{max_drawdown:.2f}%")

        st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main application entry point"""
    dashboard = UltimateTradingDashboard()
    dashboard.create_main_dashboard()

if __name__ == "__main__":
    main()
