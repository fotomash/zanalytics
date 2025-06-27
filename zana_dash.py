# Create a complete dashboard file

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
from pathlib import Path
from datetime import datetime, timedelta
import warnings
import glob
from typing import Dict, List, Optional, Tuple, Any
import re
from scipy import stats
import importlib
warnings.filterwarnings('ignore')

# Import analyzer defaults for config-driven overlays
try:
    import ncOS_ultimate_microstructure_analyzer_DEFAULTS as analyzer_defaults
except ImportError:
    analyzer_defaults = None

class UltimateZANFLOWDashboard:
    def __init__(self, data_directory="./data"):
        self.data_dir = Path(data_directory)
        self.pairs_data = {}
        self.analysis_reports = {}
        self.smc_analysis = {}
        self.wyckoff_analysis = {}
        self.microstructure_data = {}
        self.tick_data = {}
        self.bar_data = {}
        # Store latest .txt and .json insights per pair
        self.latest_txt_reports = {}
        self.latest_json_insights = {}
        
        # Initialize session state
        if 'last_update' not in st.session_state:
            st.session_state.last_update = datetime.now()
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = False
        if 'chart_theme' not in st.session_state:
            st.session_state.chart_theme = 'plotly_dark'
        if 'lookback_bars' not in st.session_state:
            st.session_state.lookback_bars = 500
        if 'selected_pair' not in st.session_state:
            st.session_state.selected_pair = None
        if 'selected_timeframe' not in st.session_state:
            st.session_state.selected_timeframe = None
        if 'current_page' not in st.session_state:
            st.session_state.current_page = "Home"

    def load_all_data(self):

        # Find all pair directories
        if not os.path.exists(self.data_dir):
            return

        pair_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]

        for pair_dir in pair_dirs:
            # Normalize pair_name: strip '_ticks', '-ticks', '_tick', '-tick', '_TICKS', etc., and uppercase
            pair_name = re.sub(r'[_\-]?ticks?$', '', pair_dir.name, flags=re.IGNORECASE).upper()
            self.pairs_data[pair_name] = {}
            self.analysis_reports[pair_name] = {}

            # Print SMC files found in this pair_dir
            print(f"Pair: {pair_name} | Looking in: {pair_dir} | SMC files found:")
            for f in pair_dir.glob("*bars*processed.csv"):
                print("  -", f.name)

            # Load SMC CSVs matching *bars*processed.csv into self.smc_analysis
            smc_csv_files = list(pair_dir.glob("*bars*processed.csv"))
            self.smc_analysis[pair_name] = {}
            for smc_csv in smc_csv_files:
                m = re.search(r"bars_([0-9a-zA-Z]+)_csv_processed", smc_csv.name)
                tf = m.group(1) if m else "Unknown"
                try:
                    df = pd.read_csv(smc_csv, index_col=0, parse_dates=True)
                    self.smc_analysis.setdefault(pair_name, {})[tf] = df
                    print(f"Loaded SMC: {smc_csv.name} for {pair_name} TF {tf}")
                except Exception as e:
                    print(f"Error loading {smc_csv.name}: {e}")

            # Load CSV files for each timeframe (bar data only)
            csv_files = list(pair_dir.glob("*_COMPREHENSIVE_*.csv"))
            for csv_file in csv_files:
                # Skip summary/journal files
                if csv_file.name in {"COMPREHENSIVE_PROCESSING_SUMMARY.json", "processing_journal.json"}:
                    continue
                parts = csv_file.stem.split('_COMPREHENSIVE_')
                if len(parts) == 2:
                    timeframe = parts[1]
                    try:
                        df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
                        # Only load as bar data if it does NOT have both 'bid' and 'ask' columns
                        if not ('bid' in df.columns and 'ask' in df.columns):
                            self.pairs_data[pair_name][timeframe] = df
                            if pair_name not in self.bar_data:
                                self.bar_data[pair_name] = {}
                            self.bar_data[pair_name][timeframe] = df
                        else:
                            # If both 'bid' and 'ask' columns, treat as tick data
                            self.tick_data[pair_name] = df
                    except Exception:
                        continue

            # Load tick files matching *tick*processed.csv, excluding summary/journal files
            tick_files = [f for f in pair_dir.glob("*tick*processed.csv") if f.name not in {
                "COMPREHENSIVE_PROCESSING_SUMMARY.json", "processing_journal.json"}]
            for tick_file in tick_files:
                try:
                    df = pd.read_csv(tick_file, index_col=0, parse_dates=True)
                    if "bid" in df.columns and "ask" in df.columns:
                        self.tick_data[pair_name] = df
                    else:
                        print(f"Warning: Skipping {tick_file.name} — missing bid/ask columns.")
                except Exception as e:
                    print(f"Error loading {tick_file}: {e}")

            # Load analysis reports
            json_files = list(pair_dir.glob("*_ANALYSIS_REPORT.json"))
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        report_data = json.load(f)
                        self.analysis_reports[pair_name][json_file.stem] = report_data
                except Exception:
                    continue

            # Load latest .txt and .json (if present)
            txt_files = sorted(pair_dir.glob("*.txt"), key=lambda f: f.stat().st_mtime, reverse=True)
            json_files = sorted(pair_dir.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)

            if txt_files:
                try:
                    self.latest_txt_reports[pair_name] = txt_files[0].read_text()
                except Exception:
                    pass
            if json_files:
                try:
                    with open(json_files[0], 'r') as f:
                        self.latest_json_insights[pair_name] = json.load(f)
                except Exception:
                    pass

        # Also load data from CSV and Parquet files directly
        self.load_data_sources()

    def load_data_sources(self):

        if not os.path.exists(self.data_dir):
            return

        # Look for CSV and Parquet files
        file_patterns = ['*.csv', '*.parquet', '*.json']

        for pattern in file_patterns:
            files = glob.glob(os.path.join(self.data_dir, pattern))

            for file_path in files:
                try:
                    file_name = os.path.basename(file_path)
                    # Skip global summary/journal JSONs
                    if file_name in {"COMPREHENSIVE_PROCESSING_SUMMARY.json", "processing_journal.json"}:
                        continue
                    base_name = os.path.splitext(file_name)[0]

                    # Extract trading pair from filename (assuming format like EURUSD_H1.csv)
                    parts = base_name.split('_')
                    if len(parts) >= 2:
                        pair = parts[0]
                        timeframe = parts[1] if len(parts) > 1 else 'Unknown'
                    else:
                        pair = base_name
                        timeframe = 'Unknown'

                    # Normalize pair name to match load_all_data
                    pair = re.sub(r'[_\-]?ticks?$', '', pair, flags=re.IGNORECASE).upper()

                    # Skip if already loaded from directory structure
                    if pair in self.pairs_data and timeframe in self.pairs_data[pair]:
                        continue

                    # Load data based on file type
                    if file_path.endswith('.csv'):
                        df = pd.read_csv(file_path)
                    elif file_path.endswith('.parquet'):
                        df = pd.read_parquet(file_path)
                    elif file_path.endswith('.json'):
                        with open(file_path, 'r') as f:
                            json_data = json.load(f)
                        df = pd.DataFrame(json_data)

                    # Ensure datetime index
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df.set_index('timestamp', inplace=True)
                    elif 'datetime' in df.columns:
                        df['datetime'] = pd.to_datetime(df['datetime'])
                        df.set_index('datetime', inplace=True)
                    elif not isinstance(df.index, pd.DatetimeIndex):
                        try:
                            df.index = pd.to_datetime(df.index)
                        except:
                            df.reset_index(inplace=True)
                            df.index = pd.date_range(start='2023-01-01', periods=len(df), freq='1H')

                    # Store data
                    if pair not in self.pairs_data:
                        self.pairs_data[pair] = {}

                    # Only load bar data (OHLCV, no bid/ask) into pairs_data/bar_data
                    if 'bid' in df.columns and 'ask' in df.columns:
                        # Tick data
                        self.tick_data[pair] = df
                    else:
                        # Bar data
                        self.pairs_data[pair][timeframe] = df
                        if pair not in self.bar_data:
                            self.bar_data[pair] = {}
                        self.bar_data[pair][timeframe] = df

                except Exception as e:
                    print(f"Error loading {file_path}: {str(e)}")

    def create_main_dashboard(self):
        # Set page configuration
        st.set_page_config(
            page_title="ZANFLOW Ultimate Trading Dashboard",
            page_icon="🚀",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Custom CSS for better styling
        st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background: #f0f2f6;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #1e3c72;
}
.wyckoff-phase {
    padding: 0.5rem;
    border-radius: 5px;
    margin: 0.25rem 0;
    font-weight: bold;
}
.accumulation { background-color: #e8f5e8; color: #2e7d2e; }
.distribution { background-color: #ffe8e8; color: #cc0000; }
.markup { background-color: #e8f8ff; color: #0066cc; }
.markdown { background-color: #fff3e0; color: #e65100; }
.smc-signal {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 8px;
    text-align: center;
    color: white;
    font-weight: bold;
    margin: 0.25rem;
}
.risk-metric {
    background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
    padding: 0.8rem;
    border-radius: 6px;
    text-align: center;
    color: white;
    font-size: 0.9rem;
    margin: 0.2rem;
}
</style>
""", unsafe_allow_html=True)

        # Main header
        st.markdown("""
        <div class="main-header">
            <h1>🚀 ZANFLOW Ultimate Trading Analysis Platform</h1>
            <p>Comprehensive Market Microstructure • Smart Money Concepts • Wyckoff Analysis • Top-Down Analysis</p>
        </div>
        """, unsafe_allow_html=True)

        # Load data silently
        with st.spinner("Initializing analysis engine..."):
            self.load_all_data()

        # Create sidebar navigation
        self.create_sidebar_navigation()
        
        # Display the selected page
        if st.session_state.current_page == "Home":
            self.display_home_page()
        elif st.session_state.current_page == "Market Overview":
            self.display_market_overview()
        elif st.session_state.current_page == "Pair Analysis":
            self.display_pair_analysis()
        elif st.session_state.current_page == "SMC Analysis":
            self.display_smc_analysis()
        elif st.session_state.current_page == "Wyckoff Analysis":
            self.display_wyckoff_analysis()
        elif st.session_state.current_page == "Microstructure":
            self.display_microstructure_analysis()
        elif st.session_state.current_page == "Risk Analytics":
            self.display_risk_analytics()
        elif st.session_state.current_page == "Settings":
            self.display_settings_page()

    def create_sidebar_navigation(self):
        """Create sidebar navigation menu"""
        st.sidebar.title("🧭 Navigation")
        
        # Main navigation
        pages = [
            "Home",
            "Market Overview",
            "Pair Analysis",
            "SMC Analysis",
            "Wyckoff Analysis",
            "Microstructure",
            "Risk Analytics",
            "Settings"
        ]
        
        # Navigation buttons
        for page in pages:
            if st.sidebar.button(page, key=f"nav_{page}"):
                st.session_state.current_page = page
                
        st.sidebar.markdown("---")
        
        # Pair selection (available on all pages)
        available_pairs = list(self.pairs_data.keys())
        if available_pairs:
            selected_pair = st.sidebar.selectbox(
                "📈 Select Currency Pair",
                available_pairs,
                index=available_pairs.index(st.session_state.selected_pair) if st.session_state.selected_pair in available_pairs else 0,
                key="sidebar_pair_select"
            )
            st.session_state.selected_pair = selected_pair
            
            # Timeframe selection
            if selected_pair in self.pairs_data:
                available_timeframes = list(self.pairs_data[selected_pair].keys())
                if available_timeframes:
                    selected_timeframe = st.sidebar.selectbox(
                        "⏱️ Select Timeframe",
                        available_timeframes,
                        index=available_timeframes.index(st.session_state.selected_timeframe) if st.session_state.selected_timeframe in available_timeframes else 0,
                        key="sidebar_timeframe_select"
                    )
                    st.session_state.selected_timeframe = selected_timeframe
        
        # Display market status if pair is selected
        if st.session_state.selected_pair and st.session_state.selected_timeframe:
            if (st.session_state.selected_pair in self.pairs_data and 
                st.session_state.selected_timeframe in self.pairs_data[st.session_state.selected_pair]):
                
                df = self.pairs_data[st.session_state.selected_pair][st.session_state.selected_timeframe]
                
                if len(df) > 0:
                    latest_price = df['close'].iloc[-1]
                    price_change = ((df['close'].iloc[-1] / df['close'].iloc[-2]) - 1) * 100 if len(df) > 1 else 0
                    
                    st.sidebar.markdown("---")
                    st.sidebar.markdown("### 📊 Market Status")
                    
                    col1, col2 = st.sidebar.columns(2)
                    with col1:
                        st.metric("Price", f"{latest_price:.4f}")
                    with col2:
                        st.metric("Change", f"{price_change:+.2f}%")
        
        # Auto-refresh toggle
        st.sidebar.markdown("---")
        st.session_state.auto_refresh = st.sidebar.toggle("🔄 Auto Refresh", value=st.session_state.auto_refresh)
        
        # Last update time
        st.sidebar.markdown(f"📅 Last Update: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}")

    def display_home_page(self):
        """Display the home/landing page"""
        st.markdown("## 🏠 Welcome to ZANFLOW Ultimate Trading Dashboard")
        
        # Dashboard overview
        st.markdown("""
        This comprehensive trading analytics platform provides advanced analysis tools for traders:
        
        - **Market Overview**: Global market analysis and correlation studies
        - **Pair Analysis**: Detailed technical analysis for specific trading pairs
        - **SMC Analysis**: Smart Money Concepts including order blocks and fair value gaps
        - **Wyckoff Analysis**: Wyckoff method for market phase identification
        - **Microstructure**: Tick-level and market microstructure analysis
        - **Risk Analytics**: Risk management metrics and portfolio analysis
        """)
        
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Available Pairs", len(self.pairs_data))
        
        with col2:
            total_timeframes = sum(len(timeframes) for timeframes in self.pairs_data.values())
            st.metric("Total Datasets", total_timeframes)
        
        with col3:
            total_data_points = sum(
                len(df) for pair_data in self.pairs_data.values() 
                for df in pair_data.values()
            )
            st.metric("Total Bars", f"{total_data_points:,}")
        
        with col4:
            st.metric("Last Update", st.session_state.last_update.strftime('%H:%M:%S'))
        
        # Top movers
        st.markdown("### 📈 Top Movers")
        self.display_top_movers()
        
        # Getting started guide
        st.markdown("### 🚀 Getting Started")
        st.markdown("""
        1. **Select a trading pair** from the sidebar
        2. **Choose a timeframe** for analysis
        3. **Navigate** to different analysis pages using the sidebar menu
        4. **Customize** your view in the Settings page
        
        For best results, place your data files in the `./data` folder with these naming conventions:
        - `SYMBOL_TIMEFRAME.csv` (e.g., EURUSD_H1.csv, GBPUSD_D1.csv)
        - Files should contain OHLCV data with a timestamp column
        - For tick data, include 'bid' and 'ask' columns
        """)
        
        # Recent updates
        st.markdown("### 🔄 Recent Updates")
        st.markdown("""
        - Added multi-page navigation
        - Enhanced SMC analysis with order block detection
        - Improved Wyckoff phase identification
        - Added microstructure analysis for tick data
        - Integrated risk analytics dashboard
        """)

    def display_top_movers(self):
        """Display top moving pairs"""
        try:
            movers_data = []
            
            for pair, timeframes in self.pairs_data.items():
                # Use the shortest timeframe available for recent movement
                if timeframes:
                    tf = list(timeframes.keys())[0]
                    df = timeframes[tf]
                    
                    if len(df) > 10:
                        recent_data = df.tail(10)
                        performance = ((recent_data['close'].iloc[-1] / recent_data['close'].iloc[0]) - 1) * 100
                        
                        # Get additional metrics if available
                        rsi = recent_data['rsi_14'].iloc[-1] if 'rsi_14' in recent_data.columns else np.nan
                        
                        movers_data.append({
                            'Pair': pair,
                            'Timeframe': tf,
                            'Performance': performance,
                            'RSI': rsi
                        })
            
            if movers_data:
                movers_df = pd.DataFrame(movers_data)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📈 Top Gainers")
                    top_gainers = movers_df.nlargest(5, 'Performance')
                    for _, row in top_gainers.iterrows():
                        rsi_value = f"RSI: {row['RSI']:.1f}" if not np.isnan(row['RSI']) else ""
                        st.markdown(f"""
                        **{row['Pair']} ({row['Timeframe']})**: {row['Performance']:+.2f}% {rsi_value}
                        """)

                with col2:
                    st.markdown("#### 📉 Top Losers")
                    top_losers = movers_df.nsmallest(5, 'Performance')
                    for _, row in top_losers.iterrows():
                        rsi_value = f"RSI: {row['RSI']:.1f}" if not np.isnan(row['RSI']) else ""
                        st.markdown(f"""
                        **{row['Pair']} ({row['Timeframe']})**: {row['Performance']:+.2f}% {rsi_value}
                        """)
        
        except Exception as e:
            st.error(f"Error analyzing top movers: {str(e)}")

    def display_market_overview(self):
        """Display comprehensive market overview"""
        st.markdown("## 🌍 Market Overview & Analysis Summary")
        
        if not self.pairs_data:
            st.warning("No data available. Please add data files to the ./data directory.")
            return
        
        # Market statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Currency Pairs", len(self.pairs_data))
        
        with col2:
            total_timeframes = sum(len(timeframes) for timeframes in self.pairs_data.values())
            st.metric("Total Datasets", total_timeframes)
        
        with col3:
            total_data_points = sum(
                len(df) for pair_data in self.pairs_data.values() 
                for df in pair_data.values()
            )
            st.metric("Total Bars", f"{total_data_points:,}")
        
        with col4:
            # Calculate average indicators per dataset
            avg_indicators = np.mean([
                len([col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']])
                for pair_data in self.pairs_data.values() 
                for df in pair_data.values()
            ]) if self.pairs_data else 0
            st.metric("Avg Indicators", f"{int(avg_indicators)}")
        
        # Market heatmap
        st.markdown("### 🔥 Market Heatmap")
        self.create_market_heatmap()
        
        # Top movers
        st.markdown("### 📊 Top Movers Analysis")
        self.create_top_movers_analysis()
        
        # Correlation matrix
        st.markdown("### 🔗 Correlation Analysis")
        self.create_correlation_matrix()

    def create_market_heatmap(self):
        """Create market performance heatmap"""
        try:
            heatmap_data = []
            
            for pair, timeframes in self.pairs_data.items():
                pair_performance = {}
                
                for tf, df in timeframes.items():
                    if len(df) > 1:
                        performance = ((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100
                        pair_performance[tf] = performance
                
                if pair_performance:
                    heatmap_data.append({
                        'Pair': pair,
                        **pair_performance
                    })
            
            if heatmap_data:
                heatmap_df = pd.DataFrame(heatmap_data).set_index('Pair')
                
                fig = px.imshow(
                    heatmap_df.values,
                    x=heatmap_df.columns,
                    y=heatmap_df.index,
                    color_continuous_scale='RdYlGn',
                    aspect="auto",
                    title="Market Performance Heatmap (%)"
                )
                
                fig.update_layout(
                    template=st.session_state.get('chart_theme', 'plotly_dark'),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error creating heatmap: {str(e)}")

    def create_top_movers_analysis(self):
        """Analyze top moving pairs"""
        try:
            movers_data = []
            
            for pair, timeframes in self.pairs_data.items():
                # Use the shortest timeframe available
                if timeframes:
                    tf = list(timeframes.keys())[0]
                    df = timeframes[tf]
                    
                    if len(df) > 24:  # At least 24 bars of data
                        recent_data = df.tail(24)
                        performance = ((recent_data['close'].iloc[-1] / recent_data['close'].iloc[0]) - 1) * 100
                        volatility = recent_data['close'].pct_change().std() * 100
                        
                        # Get additional metrics if available
                        rsi = df['rsi_14'].iloc[-1] if 'rsi_14' in df.columns else np.nan
                        atr = df['atr_14'].iloc[-1] if 'atr_14' in df.columns else np.nan
                        
                        movers_data.append({
                            'Pair': pair,
                            'Timeframe': tf,
                            'Performance': performance,
                            'Volatility': volatility,
                            'RSI': rsi,
                            'ATR': atr,
                            'Current_Price': df['close'].iloc[-1]
                        })
            
            if movers_data:
                movers_df = pd.DataFrame(movers_data)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📈 Top Gainers")
                    top_gainers = movers_df.nlargest(5, 'Performance')
                    for _, row in top_gainers.iterrows():
                        rsi_value = f"RSI: {row['RSI']:.1f}" if not np.isnan(row['RSI']) else ""
                        st.markdown(f"""
                        **{row['Pair']} ({row['Timeframe']})**: {row['Performance']:+.2f}% | {rsi_value}
                        """)

                with col2:
                    st.markdown("#### 📉 Top Losers")
                    top_losers = movers_df.nsmallest(5, 'Performance')
                    for _, row in top_losers.iterrows():
                        rsi_value = f"RSI: {row['RSI']:.1f}" if not np.isnan(row['RSI']) else ""
                        st.markdown(f"""
                        **{row['Pair']} ({row['Timeframe']})**: {row['Performance']:+.2f}% | {rsi_value}
                        """)
        
        except Exception as e:
            st.error(f"Error analyzing top movers: {str(e)}")

    def create_correlation_matrix(self):
        """Create correlation matrix of pairs"""
        try:
            # Get price data for correlation
            price_data = {}
            
            for pair, timeframes in self.pairs_data.items():
                if timeframes:
                    # Use the first available timeframe
                    tf = list(timeframes.keys())[0]
                    df = timeframes[tf]
                    
                    if len(df) > 100:
                        # Get returns
                        returns = df['close'].pct_change().dropna()
                        if len(returns) > 50:
                            price_data[pair] = returns.tail(100)  # Last 100 returns
            
            if len(price_data) >= 2:
                # Align data
                aligned_data = pd.DataFrame(price_data).dropna()
                
                if len(aligned_data) > 10:
                    # Calculate correlation
                    correlation_matrix = aligned_data.corr()
                    
                    fig = px.imshow(
                        correlation_matrix.values,
                        x=correlation_matrix.columns,
                        y=correlation_matrix.index,
                        color_continuous_scale='RdBu',
                        aspect="auto",
                        title="Price Correlation Matrix"
                    )
                    
                    fig.update_layout(
                        template=st.session_state.get('chart_theme', 'plotly_dark'),
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error creating correlation matrix: {str(e)}")

    def display_pair_analysis(self):
        """Display comprehensive pair analysis"""
        st.markdown("## 📈 Pair Analysis")
        
        # Check if pair and timeframe are selected
        if not st.session_state.selected_pair or not st.session_state.selected_timeframe:
            st.warning("Please select a pair and timeframe from the sidebar.")
            return
        
        pair = st.session_state.selected_pair
        timeframe = st.session_state.selected_timeframe
        
        if pair not in self.pairs_data or timeframe not in self.pairs_data[pair]:
            st.error("Selected data not available")
            return
        
        df = self.pairs_data[pair][timeframe]
        lookback = st.session_state.get('lookback_bars', 500)
        
        # Use last N bars
        df_display = df.tail(lookback).copy()
        
        st.markdown(f"### 🚀 {pair} {timeframe} - Comprehensive Analysis")
        
        # Market status row
        self.display_market_status(df_display, pair)
        
        # Main price chart with comprehensive overlays
        self.create_ultimate_price_chart(df_display, pair, timeframe)
        
        # Technical analysis panels
        col1, col2 = st.columns(2)
        with col1:
            self.create_advanced_volume_analysis(df_display)
        with col2:
            self.create_risk_analysis(df_display)
        
        # Pattern analysis
        self.create_pattern_analysis(df_display)
        
        # Latest TXT insights
        if pair in self.latest_txt_reports:
            st.markdown("## 🧾 Latest Report Insights")
            st.expander("📄 View Latest TXT Report").markdown(f"```text\n{self.latest_txt_reports[pair]}\n```")
        
        # Advanced analytics
        self.create_advanced_analytics_panel(df_display)

    def display_market_status(self, df, pair):
        """Display comprehensive market status"""
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        current_price = df['close'].iloc[-1]
        price_change = df['close'].iloc[-1] - df['close'].iloc[-50] if len(df) > 50 else 0
        price_change_pct = (price_change / df['close'].iloc[-50]) * 100 if len(df) > 50 and df['close'].iloc[-50] != 0 else 0
        
        with col1:
            st.metric("Current Price", f"{current_price:.4f}", f"{price_change:+.4f}")
        
        with col2:
            st.metric("Change %", f"{price_change_pct:+.2f}%")
        
        with col3:
            atr = df['atr_14'].iloc[-1] if 'atr_14' in df.columns else 0
            st.metric("ATR (14)", f"{atr:.4f}")
        
        with col4:
            rsi = df['rsi_14'].iloc[-1] if 'rsi_14' in df.columns else 50
            rsi_color = "🟢" if rsi < 30 else "🔴" if rsi > 70 else "🟡"
            st.metric("RSI (14)", f"{rsi_color} {rsi:.1f}")
        
        with col5:
            # Market regime detection
            if 'ema_8' in df.columns and 'ema_21' in df.columns:
                trend = "🟢 BULL" if df['ema_8'].iloc[-1] > df['ema_21'].iloc[-1] else "🔴 BEAR"
            else:
                trend = "🟡 UNKNOWN"
            st.metric("Trend", trend)
        
        with col6:
            volatility = df['close'].pct_change().std() if len(df) > 1 else 0
            vol_regime = "🔥 HIGH" if volatility > 0.01 else "❄️ LOW"
            st.metric("Volatility", vol_regime)

    def create_ultimate_price_chart(self, df, pair, timeframe):
        """Create ultimate price chart with all overlays"""
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=[
                f"{pair} {timeframe} - Price Action & Smart Money Analysis",
                "Volume Profile",
                "Momentum Oscillators", 
                "Market Structure"
            ],
            vertical_spacing=0.05,
            row_heights=[0.5, 0.2, 0.15, 0.15],
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
                increasing_line_color='#00ff88',
                decreasing_line_color='#ff4444'
            ), row=1, col=1
        )
        
        # Moving averages with labels
        ma_colors = {'ema_8': '#ff6b6b', 'ema_21': '#4ecdc4', 'ema_55': '#45b7d1', 'sma_200': '#96ceb4'}
        for ma, color in ma_colors.items():
            if ma in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[ma],
                    mode='lines', name=ma.upper(),
                    line=dict(color=color, width=2)
                ), row=1, col=1)
        
        # Bollinger Bands
        if all(col in df.columns for col in ['BB_Upper_20', 'BB_Lower_20']):
            fig.add_trace(go.Scatter(
                x=df.index, y=df['BB_Upper_20'],
                mode='lines', name='BB Upper',
                line=dict(color='gray', width=1, dash='dash'),
                showlegend=False
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=df.index, y=df['BB_Lower_20'],
                mode='lines', name='BB Lower',
                line=dict(color='gray', width=1, dash='dash'),
                fill='tonexty', fillcolor='rgba(128,128,128,0.1)',
                showlegend=False
            ), row=1, col=1)
        
        # SMC Analysis overlays
        self.add_smc_overlays(fig, df, row=1)
        
        # Wyckoff Analysis overlays
        self.add_wyckoff_overlays(fig, df, row=1)
        
        # Volume analysis
        if 'volume' in df.columns:
            colors = ['green' if close >= open_val else 'red' 
                     for close, open_val in zip(df['close'], df['open'])]
            
            fig.add_trace(go.Bar(
                x=df.index, y=df['volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7
            ), row=2, col=1)
            
            # Volume moving average
            if 'volume_sma_20' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['volume_sma_20'],
                    mode='lines', name='Vol MA20',
                    line=dict(color='orange', width=2)
                ), row=2, col=1)
        
        # Momentum indicators
        if 'rsi_14' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df['rsi_14'],
                mode='lines', name='RSI 14',
                line=dict(color='purple', width=2)
            ), row=3, col=1)
            
            # RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1)
        
        # MACD
        if all(col in df.columns for col in ['MACD_12_26_9', 'MACDs_12_26_9']):
            fig.add_trace(go.Scatter(
                x=df.index, y=df['MACD_12_26_9'],
                mode='lines', name='MACD',
                line=dict(color='blue', width=2)
            ), row=4, col=1)
            
            fig.add_trace(go.Scatter(
                x=df.index, y=df['MACDs_12_26_9'],
                mode='lines', name='Signal',
                line=dict(color='red', width=2)
            ), row=4, col=1)
        
        # Update layout
        fig.update_layout(
            title=f"{pair} {timeframe} Ultimate Analysis",
            template=st.session_state.get('chart_theme', 'plotly_dark'),
            height=1000,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def add_smc_overlays(self, fig, df, row=1):
        """Add Smart Money Concepts overlays"""
        # Config-driven overlays using analyzer_defaults if available
        smc_features = getattr(analyzer_defaults, "SMC_FEATURES", {}) if analyzer_defaults else {}
        if smc_features:
            for col, meta in smc_features.items():
                if col in df.columns:
                    marker_type = meta.get('type', 'marker')
                    color = meta.get('color', 'lime')
                    name = meta.get('label', col)
                    # Only show markers for "True" rows if the column is boolean, else nonzero
                    if df[col].dtype == bool:
                        vals = df[df[col] == True]
                    else:
                        vals = df[df[col] != 0]
                    if not vals.empty:
                        # y_col: use meta, fallback to 'low'/'high' by bullish/bearish, or close
                        y_col = meta.get('y_col')
                        if not y_col:
                            if 'low' in df.columns and 'high' in df.columns:
                                if 'bullish' in col:
                                    y_col = 'low'
                                elif 'bearish' in col:
                                    y_col = 'high'
                                else:
                                    y_col = 'close'
                            else:
                                y_col = 'close'
                        fig.add_trace(go.Scatter(
                            x=vals.index, y=vals[y_col],
                            mode='markers', name=name,
                            marker=dict(
                                symbol=meta.get('symbol', 'circle'),
                                color=color,
                                size=meta.get('size', 10)
                            ),
                            showlegend=True
                        ), row=row, col=1)
        else:
            # Fallback to old logic if config missing
            # Fair Value Gaps
            if 'bullish_fvg' in df.columns:
                fvg_bullish = df[df['bullish_fvg'] == True]
                if not fvg_bullish.empty:
                    fig.add_trace(go.Scatter(
                        x=fvg_bullish.index, y=fvg_bullish['low'],
                        mode='markers', name='Bullish FVG',
                        marker=dict(symbol='triangle-up', color='lime', size=12),
                        showlegend=True
                    ), row=row, col=1)
            if 'bearish_fvg' in df.columns:
                fvg_bearish = df[df['bearish_fvg'] == True]
                if not fvg_bearish.empty:
                    fig.add_trace(go.Scatter(
                        x=fvg_bearish.index, y=fvg_bearish['high'],
                        mode='markers', name='Bearish FVG',
                        marker=dict(symbol='triangle-down', color='red', size=12),
                        showlegend=True
                    ), row=row, col=1)
            # Order Blocks
            if 'bullish_order_block' in df.columns:
                ob_bullish = df[df['bullish_order_block'] == True]
                if not ob_bullish.empty:
                    fig.add_trace(go.Scatter(
                        x=ob_bullish.index, y=ob_bullish['low'],
                        mode='markers', name='Bullish OB',
                        marker=dict(symbol='square', color='lightgreen', size=10),
                        showlegend=True
                    ), row=row, col=1)
            if 'bearish_order_block' in df.columns:
                ob_bearish = df[df['bearish_order_block'] == True]
                if not ob_bearish.empty:
                    fig.add_trace(go.Scatter(
                        x=ob_bearish.index, y=ob_bearish['high'],
                        mode='markers', name='Bearish OB',
                        marker=dict(symbol='square', color='lightcoral', size=10),
                        showlegend=True
                    ), row=row, col=1)
            # Structure breaks
            if 'structure_break' in df.columns:
                structure_breaks = df[df['structure_break'] == True]
                if not structure_breaks.empty:
                    fig.add_trace(go.Scatter(
                        x=structure_breaks.index, y=structure_breaks['close'],
                        mode='markers', name='Structure Break',
                        marker=dict(symbol='x', color='yellow', size=15),
                        showlegend=True
                    ), row=row, col=1)

    def add_wyckoff_overlays(self, fig, df, row=1):
        """Add Wyckoff analysis overlays"""
        wyckoff_features = getattr(analyzer_defaults, "WYCKOFF_FEATURES", {}) if analyzer_defaults else {}
        if wyckoff_features:
            for col, meta in wyckoff_features.items():
                if col in df.columns:
                    vals = df[df[col] != 0]
                    color = meta.get('color', 'blue')
                    label = meta.get('label', col)
                    if not vals.empty:
                        fig.add_trace(go.Scatter(
                            x=vals.index, y=vals['close'],
                            mode='markers', name=label,
                            marker=dict(color=color, size=meta.get('size', 8), opacity=0.7),
                            showlegend=True
                        ), row=row, col=1)
        else:
            # Fallback to old logic
            phase_colors = {
                1: 'blue',    # Accumulation
                2: 'red',     # Distribution  
                3: 'green',   # Markup
                4: 'orange'   # Markdown
            }
            if 'wyckoff_phase' in df.columns:
                for phase, color in phase_colors.items():
                    phase_data = df[df['wyckoff_phase'] == phase]
                    if not phase_data.empty:
                        phase_names = {1: 'Accumulation', 2: 'Distribution', 3: 'Markup', 4: 'Markdown'}
                        fig.add_trace(go.Scatter(
                            x=phase_data.index, y=phase_data['close'],
                            mode='markers', name=f'Wyckoff {phase_names[phase]}',
                            marker=dict(color=color, size=8, opacity=0.7),
                            showlegend=True
                        ), row=row, col=1)

    def create_advanced_volume_analysis(self, df):
        """Create advanced volume analysis"""
        st.markdown("### 📊 Advanced Volume Analysis")
        
        if 'volume' in df.columns:
            # Volume metrics
            col1, col2 = st.columns(2)
            
            with col1:
                avg_volume = df['volume'].mean()
                current_volume = df['volume'].iloc[-1]
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
                
                st.metric("Current Volume Ratio", f"{volume_ratio:.2f}x")
                
                # Volume trend
                volume_trend = df['volume'].rolling(20).mean().pct_change().iloc[-1] * 100 if len(df) > 20 else 0
                st.metric("Volume Trend (20)", f"{volume_trend:+.1f}%")
            
            with col2:
                # Price-Volume correlation
                if len(df) > 20:
                    price_change = df['close'].pct_change()
                    volume_change = df['volume'].pct_change()
                    correlation = price_change.corr(volume_change)
                    st.metric("Price-Volume Correlation", f"{correlation:.3f}")
                
                # Volume volatility
                volume_volatility = df['volume'].pct_change().std() if len(df) > 1 else 0
                st.metric("Volume Volatility", f"{volume_volatility:.3f}")
            
            # Volume profile
            self.create_volume_profile_chart(df)

    def create_volume_profile_chart(self, df):
        """Create volume profile chart"""
        try:
            # Simple volume profile
            price_bins = pd.cut(df['close'], bins=20)
            volume_profile = df.groupby(price_bins)['volume'].sum().sort_index()
            
            fig = go.Figure()
            
            # Convert intervals to midpoints for plotting
            bin_midpoints = [interval.mid for interval in volume_profile.index]
            
            fig.add_trace(go.Bar(
                x=volume_profile.values,
                y=bin_midpoints,
                orientation='h',
                name='Volume Profile',
                marker_color='rgba(55, 128, 191, 0.7)'
            ))
            
            fig.update_layout(
                title="Volume Profile",
                xaxis_title="Volume",
                yaxis_title="Price",
                template=st.session_state.get('chart_theme', 'plotly_dark'),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error creating volume profile: {str(e)}")

    def create_risk_analysis(self, df):
        """Create comprehensive risk analysis"""
        st.markdown("### ⚠️ Risk Analysis")
        
        if len(df) > 1:
            returns = df['close'].pct_change().dropna()
            
            # Risk metrics
            col1, col2 = st.columns(2)
            
            with col1:
                volatility = returns.std() * np.sqrt(252) * 100
                st.metric("Annualized Volatility", f"{volatility:.2f}%")
                
                var_95 = np.percentile(returns, 5) * 100
                st.metric("VaR (95%)", f"{var_95:.3f}%")
            
            with col2:
                skewness = returns.skew()
                st.metric("Return Skewness", f"{skewness:.3f}")
                
                kurtosis = returns.kurtosis()
                st.metric("Return Kurtosis", f"{kurtosis:.3f}")
            
            # Drawdown analysis
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min() * 100
            
            st.metric("Maximum Drawdown", f"{max_drawdown:.2f}%")
            
            # Risk visualization
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=drawdown.index,
                y=drawdown.values * 100,
                mode='lines',
                name='Drawdown %',
                fill='tonexty',
                fillcolor='rgba(255, 0, 0, 0.3)',
                line=dict(color='red')
            ))
            
            fig.update_layout(
                title="Drawdown Analysis",
                yaxis_title="Drawdown %",
                template=st.session_state.get('chart_theme', 'plotly_dark'),
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)

    def create_pattern_analysis(self, df):
        """Create comprehensive pattern analysis"""
        st.markdown("## 🎯 Pattern Recognition Analysis")
        
        # Candlestick patterns
        pattern_columns = [col for col in df.columns if any(pattern in col.lower() for pattern in 
                          ['doji', 'hammer', 'engulfing', 'shooting_star', 'morning_star', 'evening_star'])]
        
        if pattern_columns:
            st.markdown("### 🕯️ Candlestick Patterns")
            
            pattern_counts = {}
            for col in pattern_columns:
                # Count non-zero pattern occurrences
                count = (df[col] != 0).sum() if col in df.columns else 0
                if count > 0:
                    pattern_counts[col] = count
            
            if pattern_counts:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 🟢 Bullish Patterns")
                    bullish_patterns = {k: v for k, v in pattern_counts.items() 
                                      if any(bp in k.lower() for bp in ['hammer', 'morning_star'])}
                    for pattern, count in bullish_patterns.items():
                        st.markdown(f"**{pattern.replace('_', ' ').title()}**: {count} occurrences")
                
                with col2:
                    st.markdown("#### 🔴 Bearish Patterns")
                    bearish_patterns = {k: v for k, v in pattern_counts.items() 
                                      if any(bp in k.lower() for bp in ['shooting_star', 'evening_star'])}
                    for pattern, count in bearish_patterns.items():
                        st.markdown(f"**{pattern.replace('_', ' ').title()}**: {count} occurrences")
                
                # Show recent patterns
                st.markdown("#### 🔍 Recent Pattern Occurrences")
                recent_patterns = []
                for col in pattern_columns:
                    if col in df.columns:
                        recent_data = df[df[col] != 0].tail(5)
                        for idx, row in recent_data.iterrows():
                            recent_patterns.append({
                                'Date': idx.strftime('%Y-%m-%d %H:%M'),
                                'Pattern': col.replace('_', ' ').title(),
                                'Value': row[col],
                                'Price': row['close']
                            })
                
                if recent_patterns:
                    patterns_df = pd.DataFrame(recent_patterns).sort_values('Date', ascending=False)
                    st.dataframe(patterns_df.head(10), use_container_width=True)

    def create_advanced_analytics_panel(self, df):
        """Create advanced analytics panel"""
        st.markdown("## 🔬 Advanced Analytics")
        
        tab1, tab2, tab3 = st.tabs(["📊 Statistical Analysis", "🤖 ML Features", "🎯 Signal Analysis"])
        
        with tab1:
            self.create_statistical_analysis(df)
        
        with tab2:
            self.create_ml_features_analysis(df)
        
        with tab3:
            self.create_signal_analysis(df)

    def create_statistical_analysis(self, df):
        """Create statistical analysis"""
        if len(df) > 20:
            returns = df['close'].pct_change().dropna()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📈 Return Statistics")
                st.markdown(f"""
                - **Mean Return**: {returns.mean():.6f}
                - **Std Deviation**: {returns.std():.6f}
                - **Min Return**: {returns.min():.6f}
                - **Max Return**: {returns.max():.6f}
                - **Sharpe Ratio**: {(returns.mean() / returns.std()) * np.sqrt(252):.3f}
                """)
            
            with col2:
                st.markdown("#### 📊 Distribution Analysis")
                
                # Normality test (simplified)
                from scipy import stats
                
                try:
                    jb_stat, jb_pvalue = stats.jarque_bera(returns)
                    st.markdown(f"""
                    - **Jarque-Bera Stat**: {jb_stat:.3f}
                    - **P-value**: {jb_pvalue:.6f}
                    - **Normal Distribution**: {'Yes' if jb_pvalue > 0.05 else 'No'}
                    - **Excess Kurtosis**: {returns.kurtosis():.3f}
                    - **Skewness**: {returns.skew():.3f}
                    """)
                except:
                    st.info("Statistical tests unavailable")

    def create_ml_features_analysis(self, df):
        """Create ML features analysis"""
        st.markdown("#### 🤖 Machine Learning Feature Analysis")
        
        # Technical indicators summary
        technical_indicators = [col for col in df.columns 
                              if any(ind in col.upper() for ind in 
                                   ['RSI', 'MACD', 'ATR', 'BB', 'EMA', 'SMA'])]
        
        if technical_indicators:
            st.markdown(f"**Available Technical Indicators**: {len(technical_indicators)}")
            
            # Recent indicator values
            indicator_data = []
            for indicator in technical_indicators[:10]:  # Show first 10
                if indicator in df.columns and not pd.isna(df[indicator].iloc[-1]):
                    indicator_data.append({
                        'Indicator': indicator,
                        'Current Value': f"{df[indicator].iloc[-1]:.4f}",
                        'Change': f"{(df[indicator].iloc[-1] - df[indicator].iloc[-2]):.4f}" if len(df) > 1 else "N/A"
                    })
            
            if indicator_data:
                st.dataframe(pd.DataFrame(indicator_data), use_container_width=True)
        
        # Feature importance (simulated)
        st.markdown("#### 🎯 Feature Importance (Simulated)")
        
        feature_importance = {
            'RSI_14': 0.85,
            'MACD_12_26_9': 0.78,
            'Volume_Change': 0.72,
            'ATR_14': 0.68,
            'BB_Width': 0.65,
            'EMA_Cross': 0.62,
            'Price_Momentum': 0.58,
            'Volatility': 0.55
        }
        
        fig = px.bar(
            x=list(feature_importance.keys()),
            y=list(feature_importance.values()),
            title="Feature Importance for Price Prediction",
            labels={'x': 'Feature', 'y': 'Importance Score'},
            color=list(feature_importance.values()),
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            template=st.session_state.get('chart_theme', 'plotly_dark'),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def create_signal_analysis(self, df):
        """Create signal analysis"""
        st.markdown("#### 🎯 Trading Signal Analysis")
        
        # Check for signal columns
        signal_columns = [col for col in df.columns if 'signal' in col.lower()]
        
        if signal_columns:
            # Recent signals
            recent_signals = []
            for col in signal_columns:
                if col in df.columns:
                    signal_df = df[df[col] != 0].tail(5)
                    for idx, row in signal_df.iterrows():
                        signal_type = "Buy" if row[col] > 0 else "Sell" if row[col] < 0 else "None"
                        recent_signals.append({
                            'Date': idx.strftime('%Y-%m-%d %H:%M'),
                            'Signal': col,
                            'Type': signal_type,
                            'Value': row[col],
                            'Price': row['close'] if 'close' in row else None
                        })
            if recent_signals:
                st.markdown("##### Recent Signal Events")
                signals_df = pd.DataFrame(recent_signals).sort_values('Date', ascending=False)
                st.dataframe(signals_df.head(10), use_container_width=True)

    def display_advanced_smc_analysis(self, df, pair, timeframe):
        st.markdown("### 🧠 Advanced SMC Commentary")
        # Config-driven summary stats and detected SMC events
        smc_features = getattr(analyzer_defaults, "SMC_FEATURES", {}) if analyzer_defaults else {}
        smc_events = []
        if smc_features:
            for col, meta in smc_features.items():
                if col in df.columns:
                    label = meta.get('label', col)
                    if df[col].dtype == bool:
                        value = df[col].astype(int).sum()
                    else:
                        value = df[col].sum()
                    smc_events.append(f"{label}: {value}")
        else:
            # Fallback to old logic
            if 'bullish_fvg' in df.columns:
                smc_events.append(f"Bullish FVGs: {df['bullish_fvg'].sum()}")
            if 'bearish_fvg' in df.columns:
                smc_events.append(f"Bearish FVGs: {df['bearish_fvg'].sum()}")
            if 'bullish_order_block' in df.columns:
                smc_events.append(f"Bullish OBs: {df['bullish_order_block'].sum()}")
            if 'bearish_order_block' in df.columns:
                smc_events.append(f"Bearish OBs: {df['bearish_order_block'].sum()}")
            if 'structure_break' in df.columns:
                smc_events.append(f"Structure Breaks: {df['structure_break'].sum()}")
        st.info(" | ".join(smc_events) if smc_events else "No SMC events detected in this dataset.")

    def display_wyckoff_analysis(self):
        st.markdown("## 🏛️ Wyckoff Analysis")
        if not st.session_state.selected_pair or not st.session_state.selected_timeframe:
            st.warning("Please select a pair and timeframe from the sidebar.")
            return
        pair = st.session_state.selected_pair
        timeframe = st.session_state.selected_timeframe
        # Prefer SMC-processed data if available for Wyckoff analysis
        smc_df = None
        if pair in self.smc_analysis and self.smc_analysis[pair]:
            if timeframe in self.smc_analysis[pair]:
                smc_df = self.smc_analysis[pair][timeframe]
            else:
                # Use the first available SMC timeframe if exact match not found
                smc_df = next(iter(self.smc_analysis[pair].values()))
        if smc_df is not None:
            df = smc_df
        else:
            if pair not in self.pairs_data or timeframe not in self.pairs_data[pair]:
                st.error("Selected data not available")
                return
            df = self.pairs_data[pair][timeframe]
        lookback = st.session_state.get('lookback_bars', 500)
        df_display = df.tail(lookback).copy()
        st.markdown(f"### {pair} {timeframe} - Wyckoff Phase Analysis")
        self.create_wyckoff_dashboard(df_display, pair, timeframe)

    def create_wyckoff_dashboard(self, df, pair, timeframe):
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['open'], high=df['high'],
            low=df['low'], close=df['close'],
            name="Price"
        ))
        wyckoff_features = getattr(analyzer_defaults, "WYCKOFF_FEATURES", {}) if analyzer_defaults else {}
        if wyckoff_features:
            for col, meta in wyckoff_features.items():
                if col in df.columns:
                    vals = df[df[col] != 0]
                    color = meta.get('color', 'blue')
                    label = meta.get('label', col)
                    if not vals.empty:
                        fig.add_trace(go.Scatter(
                            x=vals.index, y=vals['close'],
                            mode='markers', name=label,
                            marker=dict(color=color, size=meta.get('size', 8), opacity=0.7)
                        ))
        else:
            if 'wyckoff_phase' in df.columns:
                phase_map = {1: 'Accumulation', 2: 'Distribution', 3: 'Markup', 4: 'Markdown'}
                colors = {1: 'blue', 2: 'red', 3: 'green', 4: 'orange'}
                for phase, name in phase_map.items():
                    phase_data = df[df['wyckoff_phase'] == phase]
                    if not phase_data.empty:
                        fig.add_trace(go.Scatter(
                            x=phase_data.index, y=phase_data['close'],
                            mode='markers', name=name,
                            marker=dict(color=colors[phase], size=8, opacity=0.7)
                        ))
        fig.update_layout(
            title=f"{pair} {timeframe} Wyckoff Phases",
            template=st.session_state.get('chart_theme', 'plotly_dark'),
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)

    def display_microstructure_analysis(self):
        st.markdown("## 🧬 Microstructure Analysis")
        if not st.session_state.selected_pair:
            st.warning("Please select a pair from the sidebar.")
            return
        pair = st.session_state.selected_pair
        if pair not in self.tick_data:
            st.warning("No tick data available for this pair.")
            return
        df = self.tick_data[pair]
        st.markdown(f"### {pair} - Tick Data Microstructure")
        st.write(df.tail(100))
        # --- Manipulation Detection Display ---
        manipulation_cols = [col for col in df.columns if "manipulation" in col.lower()]
        if manipulation_cols:
            st.markdown("### 🚨 Detected Manipulation Events")
            # Show the last 20 detected manipulation events in a table
            events = df[df[manipulation_cols[0]] != 0] if manipulation_cols else pd.DataFrame()
            if not events.empty:
                cols_to_show = [*manipulation_cols]
                if 'bid' in df.columns:
                    cols_to_show.append('bid')
                if 'ask' in df.columns:
                    cols_to_show.append('ask')
                st.dataframe(events[cols_to_show].tail(20), use_container_width=True)
            else:
                st.info("No recent manipulation signals detected in tick data.")
        else:
            st.info("No manipulation signal columns found in tick data.")

    def display_smc_analysis(self):
        st.markdown("## 🧠 SMC (Smart Money Concepts) Analysis")

        pair = st.session_state.selected_pair
        if not pair or pair not in self.smc_analysis:
            st.warning("No SMC data available for this pair.")
            return

        smc_timeframes = list(self.smc_analysis[pair].keys())
        if not smc_timeframes:
            st.warning("No SMC timeframes found for this pair.")
            return
        tf = st.selectbox("Select SMC Timeframe", smc_timeframes, key="smc_tf_select")
        df = self.smc_analysis[pair][tf]
        st.markdown(f"### {pair} SMC Analysis ({tf})")
        st.dataframe(df.tail(200), use_container_width=True)
        self.display_advanced_smc_analysis(df, pair, tf)

    def display_risk_analytics(self):
        st.markdown("## 🛡️ Risk Analytics")
        if not st.session_state.selected_pair or not st.session_state.selected_timeframe:
            st.warning("Please select a pair and timeframe from the sidebar.")
            return
        pair = st.session_state.selected_pair
        timeframe = st.session_state.selected_timeframe
        if pair not in self.pairs_data or timeframe not in self.pairs_data[pair]:
            st.error("Selected data not available")
            return
        df = self.pairs_data[pair][timeframe]
        self.create_risk_analysis(df)

    def display_settings_page(self):
        st.markdown("## ⚙️ Settings")
        st.session_state.chart_theme = st.selectbox(
            "Chart Theme",
            options=['plotly_dark', 'plotly_white', 'ggplot2', 'seaborn'],
            index=['plotly_dark', 'plotly_white', 'ggplot2', 'seaborn'].index(st.session_state.get('chart_theme', 'plotly_dark'))
        )
        st.session_state.lookback_bars = st.slider(
            "Lookback Bars for Analysis",
            min_value=100, max_value=2000, value=st.session_state.get('lookback_bars', 500), step=50
        )
        st.success("Settings updated! Changes will apply to all charts.")

# --- Streamlit App Runner ---
if __name__ == "__main__":
    dashboard = UltimateZANFLOWDashboard(data_directory="./data")
    dashboard.create_main_dashboard()

