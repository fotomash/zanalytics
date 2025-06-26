#!/usr/bin/env python3
"""
ZANFLOW v13 ULTIMATE COMPREHENSIVE Trading Dashboard
Displays ALL 153+ indicators and comprehensive market analysis
Enhanced version showing complete technical analysis suite
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
import sqlite3
from typing import Dict, List, Optional
import zipfile
import io
import os

warnings.filterwarnings('ignore')

class ZanflowUltimateComprehensiveDashboard:
    """
    Ultimate comprehensive trading dashboard showing ALL available data
    """

    def __init__(self, data_directory="/Users/tom/Documents/GitHub/zanalytics/data"):
        self.data_directory = Path(data_directory)
        self.pairs = []
        self.timeframes = []
        self.all_data = {}
        self.comprehensive_data = {}
        self.json_summaries = {}
        self.microstructure_reports = {}

    def load_all_comprehensive_data(self):
        """Load ALL comprehensive data including 153+ column datasets"""
        st.info("🔄 Loading comprehensive dataset with 153+ indicators...")

        # Load from current directory (uploaded files)
        current_dir = Path('.')

        # Look for comprehensive files
        comprehensive_files = list(current_dir.glob("*COMPREHENSIVE*.csv"))
        processed_files = list(current_dir.glob("*_csv_processed.csv"))
        json_files = list(current_dir.glob("*.json"))

        # Also check Archive.zip if available
        if (current_dir / "Archive.zip").exists():
            self.load_from_archive()

        # Load comprehensive CSV files
        for file in comprehensive_files:
            try:
                df = pd.read_csv(file)
                # Extract pair and timeframe from filename
                filename = file.stem
                if 'XAUUSD' in filename:
                    pair = 'XAUUSD'
                    if '1T' in filename:
                        timeframe = '1T'
                    elif '5T' in filename:
                        timeframe = '5T'
                    elif '15T' in filename:
                        timeframe = '15T'
                    elif '30T' in filename:
                        timeframe = '30T'
                    elif '1H' in filename:
                        timeframe = '1H'
                    else:
                        timeframe = 'Unknown'

                    key = f"{pair}_{timeframe}_COMPREHENSIVE"
                    self.comprehensive_data[key] = df

                    if pair not in self.pairs:
                        self.pairs.append(pair)
                    if timeframe not in self.timeframes:
                        self.timeframes.append(timeframe)

                    st.success(f"✅ Loaded {pair} {timeframe} comprehensive data: {df.shape[0]} rows, {df.shape[1]} columns")
            except Exception as e:
                st.error(f"❌ Error loading {file}: {e}")

        # Load processed files
        for file in processed_files:
            try:
                df = pd.read_csv(file)
                filename = file.stem
                # Extract info from filename
                parts = filename.split('_')
                if len(parts) >= 3:
                    pair = parts[0]
                    timeframe = parts[-2] + '_processed'
                    key = f"{pair}_{timeframe}"
                    self.all_data[key] = df

                    if pair not in self.pairs:
                        self.pairs.append(pair)

                    st.info(f"📈 Loaded {pair} {timeframe}: {df.shape[0]} rows, {df.shape[1]} columns")
            except Exception as e:
                st.warning(f"⚠️ Could not load {file}: {e}")

        # Load JSON summaries
        for file in json_files:
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    self.json_summaries[file.stem] = data
                    st.info(f"📋 Loaded JSON: {file.name}")
            except Exception as e:
                st.warning(f"⚠️ Could not load JSON {file}: {e}")

        if not self.comprehensive_data and not self.all_data:
            st.error("No data files found! Please ensure data files are available.")
            return False

        st.success(f"🎉 Total datasets loaded: {len(self.comprehensive_data) + len(self.all_data)}")
        return True

    def load_from_archive(self):
        """Load data from Archive.zip if available"""
        try:
            with zipfile.ZipFile('Archive.zip', 'r') as zip_ref:
                file_list = zip_ref.namelist()

                for file_path in file_list:
                    if file_path.endswith('.csv') and 'COMPREHENSIVE' in file_path:
                        try:
                            with zip_ref.open(file_path) as f:
                                df = pd.read_csv(f)
                                filename = file_path.split('/')[-1]
                                if 'XAUUSD' in filename:
                                    pair = 'XAUUSD'
                                    if '1T' in filename:
                                        timeframe = '1T'
                                    elif '5T' in filename:
                                        timeframe = '5T'
                                    elif '15T' in filename:
                                        timeframe = '15T'
                                    elif '30T' in filename:
                                        timeframe = '30T'
                                    else:
                                        timeframe = 'Archive'

                                    key = f"{pair}_{timeframe}_ARCHIVE"
                                    self.comprehensive_data[key] = df
                                    st.success(f"✅ Loaded from archive: {pair} {timeframe} - {df.shape[1]} columns")
                        except Exception as e:
                            st.warning(f"Could not load {file_path}: {e}")

                    elif file_path.endswith('.json'):
                        try:
                            with zip_ref.open(file_path) as f:
                                data = json.load(f)
                                filename = file_path.split('/')[-1].replace('.json', '')
                                self.json_summaries[f"archive_{filename}"] = data
                        except Exception as e:
                            pass

        except Exception as e:
            st.warning(f"Could not load from Archive.zip: {e}")

    def create_comprehensive_sidebar(self):
        """Enhanced sidebar with all controls"""
        st.sidebar.title("🎛️ COMPREHENSIVE Analysis Center")
        st.sidebar.markdown("### 📊 Data Selection")

        # Dataset selection
        available_datasets = list(self.comprehensive_data.keys()) + list(self.all_data.keys())
        if not available_datasets:
            st.sidebar.error("No datasets available")
            return None, None

        selected_dataset = st.sidebar.selectbox(
            "Select Dataset:",
            available_datasets,
            index=0
        )

        # Get the selected data
        if selected_dataset in self.comprehensive_data:
            df = self.comprehensive_data[selected_dataset]
            is_comprehensive = True
        else:
            df = self.all_data[selected_dataset]
            is_comprehensive = False

        # Display dataset info
        st.sidebar.info(f"""
        **Dataset:** {selected_dataset}
        **Rows:** {df.shape[0]:,}
        **Columns:** {df.shape[1]:,}
        **Type:** {'COMPREHENSIVE' if is_comprehensive else 'PROCESSED'}
        **Timeframe:** {df['timestamp'].min()} to {df['timestamp'].max() if 'timestamp' in df.columns else 'N/A'}
        """)

        # Analysis type selection
        st.sidebar.markdown("### 🔍 Analysis Type")
        analysis_types = [
            "📈 Complete Overview",
            "🎯 Technical Indicators Deep Dive", 
            "📊 Moving Averages Analysis",
            "⚡ Momentum Indicators",
            "📉 Volatility Analysis", 
            "🔄 Oscillators Dashboard",
            "🎪 Price Action Patterns",
            "🌊 Volume Analysis",
            "🎭 Market Regime Analysis",
            "🔬 Microstructure Analysis",
            "📋 JSON Reports Dashboard"
        ]

        selected_analysis = st.sidebar.selectbox(
            "Choose Analysis:",
            analysis_types
        )

        # Time range selection
        st.sidebar.markdown("### ⏰ Time Range")
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            min_date = df['timestamp'].min().date()
            max_date = df['timestamp'].max().date()

            date_range = st.sidebar.date_input(
                "Select Date Range:",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )

            if len(date_range) == 2:
                start_date, end_date = date_range
                df = df[(df['timestamp'].dt.date >= start_date) & 
                       (df['timestamp'].dt.date <= end_date)]

        # Advanced filters
        st.sidebar.markdown("### ⚙️ Advanced Filters")

        # Price range filter
        if 'close' in df.columns:
            price_min = float(df['close'].min())
            price_max = float(df['close'].max()) 
            price_range = st.sidebar.slider(
                "Price Range:",
                min_value=price_min,
                max_value=price_max,
                value=(price_min, price_max),
                step=0.01
            )
            df = df[(df['close'] >= price_range[0]) & (df['close'] <= price_range[1])]

        # Volume filter if available
        if 'volume' in df.columns:
            volume_threshold = st.sidebar.number_input(
                "Min Volume:",
                min_value=0,
                value=0,
                step=1000
            )
            df = df[df['volume'] >= volume_threshold]

        return df, selected_analysis

    def display_complete_overview(self, df):
        """Display complete overview with all available data"""
        st.title("📈 COMPLETE COMPREHENSIVE OVERVIEW")

        # Summary metrics
        col1, col2, col3, col4, col5 = st.columns(5)

        if 'close' in df.columns:
            latest_price = df['close'].iloc[-1]
            price_change = ((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]) * 100

            with col1:
                st.metric("Current Price", f"{latest_price:.2f}")
            with col2:
                st.metric("Total Change", f"{price_change:+.2f}%")
            with col3:
                st.metric("High", f"{df['high'].max():.2f}")
            with col4:
                st.metric("Low", f"{df['low'].min():.2f}")
            with col5:
                st.metric("Avg Volume", f"{df['volume'].mean():,.0f}" if 'volume' in df.columns else "N/A")

        # Dataset composition
        st.markdown("### 📊 Dataset Composition")

        # Categorize all columns
        columns = df.columns.tolist()
        categories = {
            "📈 Price Data": [col for col in columns if col.lower() in ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'spread']],
            "📊 Simple Moving Averages": [col for col in columns if 'sma_' in col.lower()],
            "📈 Exponential Moving Averages": [col for col in columns if 'ema_' in col.lower()],
            "📉 Weighted Moving Averages": [col for col in columns if 'wma_' in col.lower()],
            "⚡ RSI Indicators": [col for col in columns if 'rsi' in col.lower()],
            "🌊 MACD Indicators": [col for col in columns if 'macd' in col.lower()],
            "🎭 Stochastic Oscillators": [col for col in columns if 'stoc' in col.lower()],
            "📦 Bollinger Bands": [col for col in columns if 'bb_' in col.lower()],
            "🎯 ADX/DMI": [col for col in columns if any(x in col.lower() for x in ['adx', 'dmp', 'dmn'])],
            "🪂 Parabolic SAR": [col for col in columns if 'psar' in col.lower()],
            "🔮 Other Indicators": []
        }

        # Find uncategorized columns
        categorized = []
        for cat_cols in categories.values():
            categorized.extend(cat_cols)
        categories["🔮 Other Indicators"] = [col for col in columns if col not in categorized]

        # Display category breakdown
        cols = st.columns(3)
        for i, (category, cat_columns) in enumerate(categories.items()):
            with cols[i % 3]:
                st.info(f"**{category}**\n{len(cat_columns)} indicators")
                if len(cat_columns) > 0:
                    with st.expander(f"View {category} columns"):
                        for col in cat_columns:
                            st.write(f"• {col}")

        # Latest values dashboard
        st.markdown("### 🎛️ Latest Indicator Values")

        if len(df) > 0:
            latest_row = df.iloc[-1]

            # Create tabs for different indicator groups
            tabs = st.tabs(["📊 Moving Averages", "⚡ Momentum", "🎭 Oscillators", "📦 Bands", "🔮 Others"])

            with tabs[0]:  # Moving Averages
                ma_cols = categories["📊 Simple Moving Averages"] + categories["📈 Exponential Moving Averages"] + categories["📉 Weighted Moving Averages"]
                if ma_cols:
                    ma_data = []
                    for col in ma_cols:
                        if col in latest_row.index:
                            ma_data.append({
                                "Indicator": col,
                                "Value": f"{latest_row[col]:.4f}" if pd.notna(latest_row[col]) else "N/A",
                                "Type": "SMA" if "sma" in col.lower() else "EMA" if "ema" in col.lower() else "WMA"
                            })

                    if ma_data:
                        ma_df = pd.DataFrame(ma_data)
                        st.dataframe(ma_df, use_container_width=True)

            with tabs[1]:  # Momentum
                momentum_cols = categories["⚡ RSI Indicators"] + categories["🎯 ADX/DMI"]
                if momentum_cols:
                    momentum_data = []
                    for col in momentum_cols:
                        if col in latest_row.index:
                            momentum_data.append({
                                "Indicator": col,
                                "Value": f"{latest_row[col]:.4f}" if pd.notna(latest_row[col]) else "N/A"
                            })

                    if momentum_data:
                        momentum_df = pd.DataFrame(momentum_data)
                        st.dataframe(momentum_df, use_container_width=True)

            with tabs[2]:  # Oscillators
                osc_cols = categories["🎭 Stochastic Oscillators"] + categories["🌊 MACD Indicators"]
                if osc_cols:
                    osc_data = []
                    for col in osc_cols:
                        if col in latest_row.index:
                            osc_data.append({
                                "Indicator": col,
                                "Value": f"{latest_row[col]:.4f}" if pd.notna(latest_row[col]) else "N/A"
                            })

                    if osc_data:
                        osc_df = pd.DataFrame(osc_data)
                        st.dataframe(osc_df, use_container_width=True)

            with tabs[3]:  # Bands
                band_cols = categories["📦 Bollinger Bands"] + categories["🪂 Parabolic SAR"]
                if band_cols:
                    band_data = []
                    for col in band_cols:
                        if col in latest_row.index:
                            band_data.append({
                                "Indicator": col,
                                "Value": f"{latest_row[col]:.4f}" if pd.notna(latest_row[col]) else "N/A"
                            })

                    if band_data:
                        band_df = pd.DataFrame(band_data)
                        st.dataframe(band_df, use_container_width=True)

            with tabs[4]:  # Others
                other_cols = categories["🔮 Other Indicators"]
                if other_cols:
                    other_data = []
                    for col in other_cols[:50]:  # Limit to first 50 to avoid overwhelming
                        if col in latest_row.index:
                            other_data.append({
                                "Indicator": col,
                                "Value": f"{latest_row[col]:.4f}" if pd.notna(latest_row[col]) else "N/A"
                            })

                    if other_data:
                        other_df = pd.DataFrame(other_data)
                        st.dataframe(other_df, use_container_width=True)
                        if len(other_cols) > 50:
                            st.info(f"Showing first 50 of {len(other_cols)} other indicators")

    def display_technical_indicators_deep_dive(self, df):
        """Deep dive into all technical indicators"""
        st.title("🎯 TECHNICAL INDICATORS DEEP DIVE")

        # Get all numeric columns (excluding timestamp)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) == 0:
            st.error("No numeric indicators found!")
            return

        # Multi-select for indicators
        st.markdown("### 🔍 Select Indicators to Analyze")
        selected_indicators = st.multiselect(
            "Choose indicators (max 20 for performance):",
            numeric_cols,
            default=numeric_cols[:10] if len(numeric_cols) >= 10 else numeric_cols
        )

        if not selected_indicators:
            st.warning("Please select at least one indicator")
            return

        # Limit selection for performance
        if len(selected_indicators) > 20:
            selected_indicators = selected_indicators[:20]
            st.warning("Limited to first 20 indicators for performance")

        # Create comprehensive analysis
        st.markdown("### 📊 Indicator Statistics")

        # Statistics table
        stats_data = []
        for indicator in selected_indicators:
            if indicator in df.columns:
                series = df[indicator].dropna()
                if len(series) > 0:
                    stats_data.append({
                        "Indicator": indicator,
                        "Current": f"{series.iloc[-1]:.4f}",
                        "Mean": f"{series.mean():.4f}",
                        "Std": f"{series.std():.4f}",
                        "Min": f"{series.min():.4f}",
                        "Max": f"{series.max():.4f}",
                        "Range": f"{(series.max() - series.min()):.4f}"
                    })

        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True)

        # Correlation heatmap
        st.markdown("### 🔥 Indicator Correlation Heatmap")

        correlation_data = df[selected_indicators].corr()

        fig_corr = px.imshow(
            correlation_data,
            title="Indicator Correlation Matrix",
            color_continuous_scale="RdBu",
            aspect="auto"
        )
        fig_corr.update_layout(height=600)
        st.plotly_chart(fig_corr, use_container_width=True)

        # Time series plots
        st.markdown("### 📈 Indicator Time Series")

        # Group indicators into subplots
        n_indicators = len(selected_indicators)
        n_rows = min(4, n_indicators)  # Max 4 rows

        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            fig_ts = make_subplots(
                rows=n_rows,
                cols=1,
                subplot_titles=selected_indicators[:n_rows],
                vertical_spacing=0.05
            )

            for i, indicator in enumerate(selected_indicators[:n_rows]):
                fig_ts.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df[indicator],
                        name=indicator,
                        line=dict(width=1)
                    ),
                    row=i+1,
                    col=1
                )

            fig_ts.update_layout(
                height=800,
                title="Selected Indicators Over Time",
                showlegend=False
            )
            st.plotly_chart(fig_ts, use_container_width=True)

        # Distribution analysis
        st.markdown("### 📊 Indicator Distributions")

        cols = st.columns(2)
        for i, indicator in enumerate(selected_indicators[:6]):  # Show first 6 distributions
            with cols[i % 2]:
                fig_hist = px.histogram(
                    df,
                    x=indicator,
                    title=f"{indicator} Distribution",
                    nbins=50
                )
                fig_hist.update_layout(height=300)
                st.plotly_chart(fig_hist, use_container_width=True)

    def display_json_reports_dashboard(self):
        """Display all JSON analysis reports"""
        st.title("📋 JSON REPORTS DASHBOARD")

        if not self.json_summaries:
            st.warning("No JSON reports found!")
            return

        st.markdown(f"### 📊 Available Reports: {len(self.json_summaries)}")

        # Create tabs for different report types
        report_types = list(self.json_summaries.keys())

        for report_name, report_data in self.json_summaries.items():
            with st.expander(f"📋 {report_name}", expanded=False):
                st.json(report_data)

                # Try to extract and display key metrics if structured properly
                if isinstance(report_data, dict):
                    # Look for common structures
                    if 'data_info' in report_data:
                        st.markdown("#### 📊 Data Info")
                        st.json(report_data['data_info'])

                    if 'price_stats' in report_data:
                        st.markdown("#### 💰 Price Statistics")
                        st.json(report_data['price_stats'])

                    if 'signal_summary' in report_data:
                        st.markdown("#### 🎯 Signal Summary")
                        st.json(report_data['signal_summary'])

                    if 'market_regime' in report_data:
                        st.markdown("#### 🏛️ Market Regime")
                        st.json(report_data['market_regime'])

    def create_comprehensive_chart(self, df):
        """Create the most comprehensive chart possible"""
        if 'timestamp' in df.columns and all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Create subplot with multiple rows for different indicator groups
            fig = make_subplots(
                rows=6,
                cols=1,
                row_heights=[0.4, 0.15, 0.15, 0.15, 0.15, 0.15],
                subplot_titles=[
                    "Price Action with Moving Averages",
                    "RSI & Momentum",
                    "MACD",
                    "Stochastic",
                    "Bollinger Bands Position",
                    "Volume"
                ],
                vertical_spacing=0.02
            )

            # Main price chart
            fig.add_trace(
                go.Candlestick(
                    x=df['timestamp'],
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name="Price"
                ),
                row=1, col=1
            )

            # Add moving averages
            ma_colors = ['blue', 'red', 'green', 'orange', 'purple']
            ma_cols = [col for col in df.columns if any(x in col.lower() for x in ['sma_', 'ema_']) and any(str(x) in col for x in [5, 8, 13, 21, 34])]

            for i, ma_col in enumerate(ma_cols[:5]):  # Limit to 5 MAs
                if ma_col in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df['timestamp'],
                            y=df[ma_col],
                            name=ma_col,
                            line=dict(color=ma_colors[i % len(ma_colors)], width=1)
                        ),
                        row=1, col=1
                    )

            # RSI
            rsi_cols = [col for col in df.columns if 'rsi' in col.lower()]
            if rsi_cols:
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df[rsi_cols[0]],
                        name=rsi_cols[0].upper(),
                        line=dict(color='purple')
                    ),
                    row=2, col=1
                )
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

            # MACD
            macd_cols = [col for col in df.columns if 'macd' in col.lower()]
            if len(macd_cols) >= 2:
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df[macd_cols[0]],
                        name="MACD",
                        line=dict(color='blue')
                    ),
                    row=3, col=1
                )
                if len(macd_cols) > 1:
                    fig.add_trace(
                        go.Scatter(
                            x=df['timestamp'],
                            y=df[macd_cols[1]],
                            name="MACD Signal",
                            line=dict(color='red')
                        ),
                        row=3, col=1
                    )

            # Stochastic
            stoch_cols = [col for col in df.columns if 'stoc' in col.lower()]
            if stoch_cols:
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df[stoch_cols[0]],
                        name="Stochastic",
                        line=dict(color='orange')
                    ),
                    row=4, col=1
                )
                fig.add_hline(y=80, line_dash="dash", line_color="red", row=4, col=1)
                fig.add_hline(y=20, line_dash="dash", line_color="green", row=4, col=1)

            # Bollinger Bands Position
            bb_cols = [col for col in df.columns if 'bb_position' in col.lower()]
            if bb_cols:
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df[bb_cols[0]],
                        name="BB Position",
                        line=dict(color='cyan')
                    ),
                    row=5, col=1
                )
                fig.add_hline(y=1, line_dash="dash", line_color="red", row=5, col=1)
                fig.add_hline(y=0, line_dash="dash", line_color="green", row=5, col=1)

            # Volume
            if 'volume' in df.columns:
                fig.add_trace(
                    go.Bar(
                        x=df['timestamp'],
                        y=df['volume'],
                        name="Volume",
                        marker_color='lightblue'
                    ),
                    row=6, col=1
                )

            fig.update_layout(
                height=1200,
                title="Comprehensive Technical Analysis Chart",
                xaxis_rangeslider_visible=False
            )

            return fig

        return None

    def run_dashboard(self):
        """Main dashboard execution"""
        st.set_page_config(
            page_title="ZANFLOW v13 Ultimate Comprehensive Dashboard",
            page_icon="🎯",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Header
        st.title("🎯 ZANFLOW v13 ULTIMATE COMPREHENSIVE DASHBOARD")
        st.markdown("### 📊 Complete Technical Analysis Suite - ALL 153+ Indicators")

        # Load data
        if not hasattr(self, 'data_loaded'):
            if self.load_all_comprehensive_data():
                self.data_loaded = True
            else:
                st.stop()

        # Sidebar controls
        df, selected_analysis = self.create_comprehensive_sidebar()

        if df is None:
            st.error("No data available for analysis")
            st.stop()

        # Main content based on selected analysis
        if selected_analysis == "📈 Complete Overview":
            self.display_complete_overview(df)

        elif selected_analysis == "🎯 Technical Indicators Deep Dive":
            self.display_technical_indicators_deep_dive(df)

        elif selected_analysis == "📋 JSON Reports Dashboard":
            self.display_json_reports_dashboard()

        else:
            st.info(f"Analysis type '{selected_analysis}' coming soon!")
            # Fallback to complete overview
            self.display_complete_overview(df)

        # Comprehensive chart
        st.markdown("### 📈 COMPREHENSIVE ANALYSIS CHART")

        chart = self.create_comprehensive_chart(df)
        if chart:
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.warning("Could not create comprehensive chart - missing required price columns")

        # Footer
        st.markdown("---")
        st.markdown("### 🎛️ Dashboard Statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Datasets Loaded", len(self.comprehensive_data) + len(self.all_data))
        with col2:
            st.metric("JSON Reports", len(self.json_summaries))
        with col3:
            st.metric("Total Indicators", df.shape[1] if df is not None else 0)
        with col4:
            st.metric("Data Points", df.shape[0] if df is not None else 0)

# Initialize and run the dashboard
if __name__ == "__main__":
    dashboard = ZanflowUltimateComprehensiveDashboard()
    dashboard.run_dashboard()
