# Continuing from create_microstructure_analysis method...
#!/usr/bin/env python3
"""
ZANFLOW ULTIMATE MEGA DASHBOARD v16
The Complete All-in-One Trading Analysis Platform
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
import re
from typing import Dict, List, Optional, Tuple, Any
try:
    from scipy import stats
except ImportError:
    stats = None

warnings.filterwarnings('ignore')

class ZANFLOWUltimateMegaDashboard:
    def __init__(self, data_directory="."):
        """Initialize the MEGA dashboard with all features"""
        self.data_dir = Path(data_directory)
        
        # Data storage
        self.symbols_data = {}
        self.available_symbols = []
        self.available_timeframes = {}
        self.tick_data = {}
        self.bar_data = {}
        self.txt_reports = {}
        self.json_reports = {}
        self.microstructure_data = {}
        
        # Color scheme for consistent visualization
        self.colors = {
            'bullish': '#00ff88',
            'bearish': '#ff4444',
            'neutral': '#aaaaaa',
            'supply_zone': '#ff6b6b',
            'demand_zone': '#4ecdc4',
            'fair_value_gap': '#45b7d1',
            'liquidity': '#ffcc00',
            'manipulation': '#ff00ff',
            'wyckoff_accumulation': '#96ceb4',
            'wyckoff_distribution': '#ff9a8b',
            'bos': '#00ffff',
            'choch': '#ff00ff'
        }

    def scan_available_symbols(self):
        """Robust symbol scanner that works with all file naming conventions"""
        if not self.data_dir.exists():
            return False

        # Reset data structures
        self.symbols_data = {}
        self.tick_data = {}
        self.bar_data = {}
        self.microstructure_data = {}

        # Find all processed CSV files
        csv_files = list(self.data_dir.glob("*_processed.csv"))
        
        if not csv_files:
            return False

        # Process each file
        for file_path in csv_files:
            try:
                filename = file_path.name
                
                # Extract symbol (usually the first part before underscore)
                if 'XAUUSD' in filename:
                    symbol = 'XAUUSD'
                else:
                    symbol = filename.split('_')[0]
                
                # Initialize symbol data structures if needed
                if symbol not in self.symbols_data:
                    self.symbols_data[symbol] = {}
                if symbol not in self.tick_data:
                    self.tick_data[symbol] = {}
                if symbol not in self.bar_data:
                    self.bar_data[symbol] = {}
                
                # Determine if it's tick or bar data and extract timeframe
                if 'TICK' in filename:
                    # Tick data
                    if 'tick_tick' in filename:
                        timeframe = 'tick'
                    elif '1min' in filename:
                        timeframe = '1min'
                    elif '5min' in filename:
                        timeframe = '5min'
                    elif '15min' in filename:
                        timeframe = '15min'
                    elif '30min' in filename:
                        timeframe = '30min'
                    else:
                        # Default to the part after the last underscore before _processed
                        parts = filename.split('_')
                        timeframe = parts[-2] if len(parts) > 2 else 'unknown'
                    
                    self.tick_data[symbol][timeframe] = file_path
                    self.symbols_data[symbol][timeframe] = file_path
                    
                elif 'bars' in filename or 'M1' in filename:
                    # Bar data
                    if '1min' in filename:
                        timeframe = '1min'
                    elif '5min' in filename:
                        timeframe = '5min'
                    elif '15min' in filename:
                        timeframe = '15min'
                    elif '30min' in filename:
                        timeframe = '30min'
                    elif '1H' in filename:
                        timeframe = '1H'
                    elif '4H' in filename:
                        timeframe = '4H'
                    elif '1D' in filename:
                        timeframe = '1D'
                    else:
                        # Extract from filename
                        parts = filename.split('_')
                        timeframe = parts[-2] if len(parts) > 2 else 'unknown'
                    
                    self.bar_data[symbol][timeframe] = file_path
                    self.symbols_data[symbol][timeframe] = file_path
                else:
                    # Generic case - just use the last part before _processed
                    parts = filename.split('_')
                    timeframe = parts[-2] if len(parts) > 2 else 'unknown'
                    self.symbols_data[symbol][timeframe] = file_path
            
            except Exception as e:
                # Skip files that don't match expected patterns
                continue

        # Load TXT reports
        self.load_txt_reports()
        
        # Load microstructure data
        self.load_microstructure_data()
        
        # Update available symbols and timeframes
        self.available_symbols = list(self.symbols_data.keys())
        self.available_timeframes = {s: list(t.keys()) for s, t in self.symbols_data.items()}
        
        return len(self.available_symbols) > 0

    def load_txt_reports(self):
        """Load all TXT analysis reports"""
        txt_files = list(self.data_dir.glob("*.txt"))
        
        for txt_file in txt_files:
            try:
                with open(txt_file, 'r') as f:
                    content = f.read()
                
                # Try to determine which symbol this report belongs to
                filename = txt_file.name
                symbol = None
                
                for s in self.symbols_data.keys():
                    if s in filename:
                        symbol = s
                        break
                
                if symbol:
                    if symbol not in self.txt_reports:
                        self.txt_reports[symbol] = []
                    
                    self.txt_reports[symbol].append({
                        'filename': filename,
                        'content': content,
                        'timestamp': txt_file.stat().st_mtime
                    })
            except Exception:
                continue

    def load_microstructure_data(self):
        """Load microstructure analysis data"""
        # Look for JSON files with microstructure data
        json_files = list(self.data_dir.glob("*Microstructure*.json"))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Try to determine which symbol this data belongs to
                filename = json_file.name
                symbol = None
                
                for s in self.symbols_data.keys():
                    if s in filename:
                        symbol = s
                        break
                
                if symbol:
                    if symbol not in self.microstructure_data:
                        self.microstructure_data[symbol] = []
                    
                    self.microstructure_data[symbol].append({
                        'filename': filename,
                        'data': data,
                        'timestamp': json_file.stat().st_mtime
                    })
            except Exception:
                continue

    def load_symbol_data(self, symbol, timeframe):
        """Load data for a specific symbol and timeframe"""
        try:
            # Check if we have this data
            file_path = None
            
            # Try tick data first
            if symbol in self.tick_data and timeframe in self.tick_data[symbol]:
                file_path = self.tick_data[symbol][timeframe]
            # Then bar data
            elif symbol in self.bar_data and timeframe in self.bar_data[symbol]:
                file_path = self.bar_data[symbol][timeframe]
            # Finally general data
            elif symbol in self.symbols_data and timeframe in self.symbols_data[symbol]:
                file_path = self.symbols_data[symbol][timeframe]
            
            if file_path and file_path.exists():
                df = pd.read_csv(file_path)
                
                # Handle timestamp column
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                elif df.columns[0] == 'Unnamed: 0':
                    # First column is likely a timestamp
                    df.index = pd.to_datetime(df[df.columns[0]])
                    df = df.drop(df.columns[0], axis=1)
                
                return df
            
            return None
            
        except Exception as e:
            st.error(f"Error loading data for {symbol} {timeframe}: {e}")
            return None

    def load_latest_txt_report(self, symbol):
        """Load the latest TXT report for a symbol"""
        if symbol in self.txt_reports and self.txt_reports[symbol]:
            # Sort by timestamp and get the latest
            latest_report = sorted(self.txt_reports[symbol], key=lambda x: x['timestamp'], reverse=True)[0]
            return latest_report['content']
        
        # Fallback: try to find a report file
        txt_files = list(self.data_dir.glob(f"*{symbol}*Report*.txt"))
        if txt_files:
            latest_file = max(txt_files, key=os.path.getmtime)
            try:
                with open(latest_file, 'r') as f:
                    return f.read()
            except Exception:
                pass
        
        return None

    def calculate_market_metrics(self, df):
        """Calculate comprehensive market metrics"""
        try:
            if df is None or len(df) == 0:
                return {
                    'spread': 'N/A', 'atr': 'N/A', 'rsi': 'N/A', 
                    'manipulation': 'N/A', 'trend': 'N/A', 'volatility': 'N/A'
                }
            
            metrics = {}
            
            # Spread calculation
            if 'spread' in df.columns:
                metrics['spread'] = f"{df['spread'].iloc[-1]:.2f}"
            elif 'spread_price' in df.columns:
                metrics['spread'] = f"{df['spread_price'].iloc[-1]:.5f}"
            elif 'bid' in df.columns and 'ask' in df.columns:
                metrics['spread'] = f"{df['ask'].iloc[-1] - df['bid'].iloc[-1]:.5f}"
            else:
                metrics['spread'] = f"{(df['high'] - df['low']).tail(20).mean():.4f}"
            
            # ATR calculation
            atr_val = np.nan
            for col in ['ATR_14', 'atr_14', 'ATR', 'atr']:
                if col in df.columns and not pd.isna(df[col].iloc[-1]):
                    atr_val = df[col].iloc[-1]
                    break
            
            if pd.isna(atr_val) and len(df) > 14:
                # Calculate ATR manually
                high_low = df['high'] - df['low']
                high_close = abs(df['high'] - df['close'].shift(1))
                low_close = abs(df['low'] - df['close'].shift(1))
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                atr_val = true_range.rolling(14).mean().iloc[-1]
            
            metrics['atr'] = f"{atr_val:.4f}" if not pd.isna(atr_val) else "N/A"
            
            # RSI calculation
            rsi_val = np.nan
            for col in ['RSI_14', 'rsi_14', 'RSI', 'rsi']:
                if col in df.columns and not pd.isna(df[col].iloc[-1]):
                    rsi_val = df[col].iloc[-1]
                    break
            
            if pd.isna(rsi_val) and len(df) > 14:
                # Calculate RSI manually
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                rsi_val = 100 - (100 / (1 + rs)).iloc[-1]
            
            if pd.notna(rsi_val):
                rsi_color = "🟢" if rsi_val < 30 else "🔴" if rsi_val > 70 else "🟡"
                metrics['rsi'] = f"{rsi_color} {rsi_val:.1f}"
            else:
                metrics['rsi'] = "N/A"
            
            # Manipulation detection
            manipulation_score = 0
            for col in ['spoofing_detected', 'layering_detected', 'momentum_ignition']:
                if col in df.columns:
                    manipulation_score += df[col].tail(50).sum()
            
            if manipulation_score > 10:
                metrics['manipulation'] = "🔴 HIGH"
            elif manipulation_score > 5:
                metrics['manipulation'] = "🟡 MEDIUM"
            elif manipulation_score > 0:
                metrics['manipulation'] = "🟢 LOW"
            else:
                metrics['manipulation'] = "🟢 CLEAN"
            
            # Trend detection
            if len(df) >= 21:
                short_ma = df['close'].rolling(8).mean().iloc[-1]
                long_ma = df['close'].rolling(21).mean().iloc[-1]
                metrics['trend'] = "🟢 BULL" if short_ma > long_ma else "🔴 BEAR"
            else:
                metrics['trend'] = "🟡 N/A"
            
            # Volatility
            if len(df) > 20:
                vol = df['close'].pct_change().tail(20).std()
                vol_threshold = df['close'].pct_change().std()
                metrics['volatility'] = "🔥 HIGH" if vol > vol_threshold * 1.5 else "❄️ LOW"
            else:
                metrics['volatility'] = "🟡 N/A"
            
            return metrics
            
        except Exception as e:
            return {
                'spread': 'ERR', 'atr': 'ERR', 'rsi': 'ERR',
                'manipulation': 'ERR', 'trend': 'ERR', 'volatility': 'ERR'
            }

    def create_mega_dashboard(self):
        """Create the MEGA dashboard interface"""
        st.set_page_config(
            page_title="ZANFLOW MEGA Dashboard v16", 
            page_icon="🚀", 
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for professional styling
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(135deg, #1F2937 0%, #111827 100%);
            padding: 2rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        }
        .metric-card {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 5px solid #667eea;
            margin: 0.5rem 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .market-overview {
            background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
            border: 2px solid #ff9a8b;
        }
        .analysis-section {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            border-left: 3px solid #667eea;
        }
        .microstructure-section {
            background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
            border: 2px solid #5ee7df;
        }
        .tick-analysis {
            background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
            padding: 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
        }
        .manipulation-alert {
            background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
            padding: 1rem;
            border-radius: 8px;
            border: 2px solid #ff6b6b;
            margin: 1rem 0;
        }
        .smc-signal {
            background: linear-gradient(135deg, #a8e6cf 0%, #dcedc1 100%);
            padding: 0.8rem;
            border-radius: 8px;
            margin: 0.5rem 0;
            border-left: 4px solid #52b788;
        }
        .wyckoff-phase {
            padding: 0.8rem;
            border-radius: 8px;
            margin: 0.5rem 0;
            font-weight: bold;
            text-align: center;
        }
        .accumulation { background: linear-gradient(135deg, #e8f5e8 0%, #a8e6cf 100%); color: #2d5016; }
        .distribution { background: linear-gradient(135deg, #ffe8e8 0%, #ffcccb 100%); color: #8b0000; }
        .markup { background: linear-gradient(135deg, #e8f8ff 0%, #b3d9ff 100%); color: #003d82; }
        .markdown { background: linear-gradient(135deg, #fff3e0 0%, #ffe0b3 100%); color: #cc5500; }
        .success-metric { color: #28a745; font-weight: bold; }
        .warning-metric { color: #ffc107; font-weight: bold; }
        .danger-metric { color: #dc3545; font-weight: bold; }
        .info-card {
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
            padding: 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
            border-left: 4px solid #2196f3;
        }
        .advanced-section {
            background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
            border: 2px solid #9c27b0;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Main header
        st.markdown("""
        <div class="main-header">
            <h1>🚀 ZANFLOW ULTIMATE MEGA DASHBOARD v16</h1>
            <p><strong>The Complete Trading Intelligence Platform</strong></p>
            <p>Tick-Level Microstructure • Smart Money Concepts • Wyckoff Analysis • Multi-Timeframe Intelligence</p>
            <p>Market Manipulation Detection • Order Flow • Advanced Pattern Recognition • ML Features</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize data scanning
        if not hasattr(st.session_state, 'symbols_scanned'):
            with st.spinner("🔍 Initializing MEGA analysis engine..."):
                if self.scan_available_symbols():
                    st.session_state.symbols_scanned = True
                else:
                    st.error("❌ No trading data found. Please upload your '*_processed.csv' files.")
                    return
        
        # Create sidebar
        self.create_mega_sidebar()
        
        # Main content
        if st.session_state.get('selected_symbol') and st.session_state.get('selected_timeframe'):
            self.display_mega_analysis()
        else:
            self.display_mega_overview()

    def create_mega_sidebar(self):
        """Create the MEGA sidebar with all controls"""
        st.sidebar.markdown("# 🎛️ MEGA Control Center")
        
        # Symbol selection
        if self.available_symbols:
            selected_symbol = st.sidebar.selectbox(
                "📈 Select Trading Symbol",
                [""] + sorted(self.available_symbols),
                key="selected_symbol",
                help="Choose from all available symbols in your data"
            )
            
            if selected_symbol:
                # Timeframe selection
                available_tfs = self.available_timeframes.get(selected_symbol, [])
                
                # Categorize timeframes
                tick_tfs = [tf for tf in available_tfs if tf in ['tick', '1min', '5min', '15min', '30min'] and selected_symbol in self.tick_data and tf in self.tick_data[selected_symbol]]
                bar_tfs = [tf for tf in available_tfs if tf not in tick_tfs]
                
                if tick_tfs:
                    st.sidebar.markdown("**🔍 Tick Data Available**")
                
                selected_timeframe = st.sidebar.selectbox(
                    "⏱️ Select Timeframe",
                    [""] + sorted(available_tfs),
                    key="selected_timeframe",
                    help="Available timeframes for selected symbol"
                )
                
                if selected_timeframe:
                    # Load and display real-time metrics
                    df = self.load_symbol_data(selected_symbol, selected_timeframe)
                    if df is not None:
                        metrics = self.calculate_market_metrics(df)
                        
                        st.sidebar.markdown("---")
                        st.sidebar.markdown("### 🌍 Real-Time Market Status")
                        
                        # Enhanced metrics display
                        col1, col2 = st.sidebar.columns(2)
                        with col1:
                            st.metric("📏 Spread", metrics['spread'])
                            st.metric("📊 ATR", metrics['atr'])
                            st.metric("📈 Trend", metrics['trend'])
                        with col2:
                            st.metric("📈 RSI", metrics['rsi'])
                            st.metric("🛡️ Manipulation", metrics['manipulation'])
                            st.metric("🔥 Volatility", metrics['volatility'])
                        
                        # Data info
                        current_price = df['close'].iloc[-1] if 'close' in df.columns else df.get('mid_price', pd.Series([0])).iloc[-1]
                        
                        st.sidebar.markdown("---")
                        st.sidebar.markdown("### 📊 Data Overview")
                        
                        col1, col2 = st.sidebar.columns(2)
                        with col1:
                            st.metric("💰 Price", f"{current_price:.4f}")
                        with col2:
                            if len(df) > 1:
                                price_change = ((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100 if 'close' in df.columns else 0
                                st.metric("Δ %", f"{price_change:+.2f}%")
                        
                        st.sidebar.info(f"""
                        🔢 **Records**: {len(df):,}  
                        📅 **From**: {df.index.min().strftime('%Y-%m-%d %H:%M')}  
                        📅 **To**: {df.index.max().strftime('%Y-%m-%d %H:%M')}
                        """)
        
        # Analysis options
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🔬 Analysis Modules")
        
        # Core modules
        st.session_state['show_overview'] = st.sidebar.checkbox("🌍 Market Overview", True)
        st.session_state['show_price_analysis'] = st.sidebar.checkbox("📈 Advanced Price Analysis", True)
        
        # Microstructure
        with st.sidebar.expander("🔍 Microstructure Analysis"):
            st.session_state['show_microstructure'] = st.checkbox("Enable Microstructure", True)
            st.session_state['show_manipulation_detection'] = st.checkbox("🚨 Manipulation Detection", True)
            st.session_state['show_order_flow'] = st.checkbox("💹 Order Flow Analysis", True)
            st.session_state['show_tick_analysis'] = st.checkbox("⚡ Tick Analysis", True)
        
        # Technical Analysis
        with st.sidebar.expander("📊 Technical Analysis"):
            st.session_state['show_multitimeframe'] = st.checkbox("⏱️ Multi-Timeframe", True)
            st.session_state['show_indicators'] = st.checkbox("📉 Technical Indicators", True)
            st.session_state['show_patterns'] = st.checkbox("🎯 Pattern Recognition", True)
            st.session_state['show_volume'] = st.checkbox("📊 Volume Analysis", True)
        
        # Advanced Analysis
        with st.sidebar.expander("🧠 Advanced Analysis"):
            st.session_state['show_smc'] = st.checkbox("🧠 Smart Money Concepts", True)
            st.session_state['show_wyckoff'] = st.checkbox("📈 Wyckoff Analysis", True)
            st.session_state['show_liquidity'] = st.checkbox("💧 Liquidity Analysis", True)
            st.session_state['show_harmonic'] = st.checkbox("🎵 Harmonic Patterns", False)
        
        # Intelligence Features
        with st.sidebar.expander("🤖 Intelligence Features"):
            st.session_state['show_ml_features'] = st.checkbox("🤖 ML Features", True)
            st.session_state['show_statistical'] = st.checkbox("📊 Statistical Analysis", True)
            st.session_state['show_signal_analysis'] = st.checkbox("🎯 Signal Analysis", True)
            st.session_state['show_risk'] = st.checkbox("⚠️ Risk Metrics", True)
        
        # Reports
        with st.sidebar.expander("📄 Reports & Insights"):
            st.session_state['show_txt_analysis'] = st.checkbox("📄 TXT Reports", True)
            st.session_state['show_json_insights'] = st.checkbox("📊 JSON Insights", True)
            st.session_state['show_commentary'] = st.checkbox("💬 Rich Commentary", True)
        
        # Display Settings
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🎨 Display Settings")
        
        st.session_state['chart_theme'] = st.sidebar.selectbox(
            "Chart Theme", 
            ["plotly_dark", "plotly_white", "ggplot2", "seaborn", "plotly"],
            index=0
        )
        
        st.session_state['lookback_bars'] = st.sidebar.slider(
            "Lookback Bars", 50, 5000, 500
        )
        
        if any(s in self.tick_data for s in self.available_symbols):
            st.session_state['lookback_ticks'] = st.sidebar.slider(
                "Lookback Ticks", 100, 20000, 2000
            )

    def display_mega_overview(self):
        """Display MEGA market overview"""
        st.markdown("## 🌍 MEGA Market Overview & Intelligence Hub")
        
        if not self.available_symbols:
            st.warning("No symbols found. Please check your data files.")
            return
        
        # Market statistics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("📊 Total Symbols", len(self.available_symbols))
        
        with col2:
            total_timeframes = sum(len(tfs) for tfs in self.available_timeframes.values())
            st.metric("⏱️ Total Datasets", total_timeframes)
        
        with col3:
            tick_count = sum(len(self.tick_data.get(s, {})) for s in self.available_symbols)
            st.metric("🔍 Tick Datasets", tick_count)
        
        with col4:
            bar_count = sum(len(self.bar_data.get(s, {})) for s in self.available_symbols)
            st.metric("📊 Bar Datasets", bar_count)
        
        with col5:
            report_count = sum(len(self.txt_reports.get(s, [])) for s in self.available_symbols)
            st.metric("📄 Analysis Reports", report_count)
        
        # Symbol breakdown with enhanced display
        st.markdown("### 📊 Symbol Intelligence Dashboard")
        
        symbol_data = []
        for symbol in sorted(self.available_symbols):
            timeframes = self.available_timeframes.get(symbol, [])
            tick_tfs = [tf for tf in timeframes if symbol in self.tick_data and tf in self.tick_data[symbol]]
            bar_tfs = [tf for tf in timeframes if symbol in self.bar_data and tf in self.bar_data[symbol]]
            
            symbol_data.append({
                'Symbol': symbol,
                'Total Timeframes': len(timeframes),
                'Tick Data': len(tick_tfs),
                'Bar Data': len(bar_tfs),
                'Available Timeframes': ', '.join(sorted(timeframes)),
                'Has Reports': '✅' if symbol in self.txt_reports and self.txt_reports[symbol] else '❌'
            })
        
        if symbol_data:
            symbols_df = pd.DataFrame(symbol_data)
            st.dataframe(symbols_df, use_container_width=True, height=300)
        
        # Quick Launch Panel
        st.markdown("### 🚀 Quick Launch Panel")
        cols = st.columns(min(len(self.available_symbols), 4))
        
        for i, symbol in enumerate(sorted(self.available_symbols)[:4]):
            with cols[i]:
                if st.button(f"📈 Launch {symbol}", key=f"quick_{symbol}", use_container_width=True):
                    st.session_state['selected_symbol'] = symbol
                    if self.available_timeframes[symbol]:
                        # Prefer 1H or 15min for quick launch
                        preferred_tfs = ['1H', '15min', '5min', '1min']
                        for tf in preferred_tfs:
                            if tf in self.available_timeframes[symbol]:
                                st.session_state['selected_timeframe'] = tf
                                break
                        else:
                            st.session_state['selected_timeframe'] = sorted(self.available_timeframes[symbol])[0]
                    st.rerun()

    def display_mega_analysis(self):
        """Display MEGA comprehensive analysis"""
        symbol = st.session_state['selected_symbol']
        timeframe = st.session_state['selected_timeframe']
        
        # Load primary data
        df = self.load_symbol_data(symbol, timeframe)
        if df is None:
            st.error(f"Failed to load data for {symbol} {timeframe}")
            return
        
        # Apply lookback
        lookback = st.session_state.get('lookback_bars', 500)
        if 'tick' in timeframe:
            lookback = st.session_state.get('lookback_ticks', 2000)
        
        df_display = df.tail(lookback).copy()
        
        # Epic header with real-time status
        st.markdown(f"# 🚀 {symbol} - {timeframe} MEGA Analysis Suite")
        
        # Real-time market status bar
        if st.session_state.get('show_overview', True):
            self.display_mega_market_status(df_display, symbol)
        
        # Main price analysis
        if st.session_state.get('show_price_analysis', True):
            self.create_mega_price_chart(df_display, symbol, timeframe)
        
        # Create tabs for different analysis sections
        tabs_to_show = []
        if st.session_state.get('show_multitimeframe', True): tabs_to_show.append("⏱️ Multi-Timeframe")
        if st.session_state.get('show_microstructure', True): tabs_to_show.append("🔍 Microstructure & Order Flow")
        if st.session_state.get('show_smc', True): tabs_to_show.append("🧠 Smart Money Concepts")
        if st.session_state.get('show_wyckoff', True): tabs_to_show.append("📈 Wyckoff Analysis")
        if st.session_state.get('show_patterns', True): tabs_to_show.append("🎯 Pattern Recognition")
        if st.session_state.get('show_volume', True): tabs_to_show.append("📊 Volume Analysis")
        if st.session_state.get('show_ml_features', True) or st.session_state.get('show_statistical', True): 
            tabs_to_show.append("🤖 Advanced Analytics")
        if st.session_state.get('show_txt_analysis', True): tabs_to_show.append("📄 Analysis Reports")
        
        if tabs_to_show:
            tabs = st.tabs(tabs_to_show)
            tab_map = {name: tab for name, tab in zip(tabs_to_show, tabs)}
            
            if "⏱️ Multi-Timeframe" in tab_map:
                with tab_map["⏱️ Multi-Timeframe"]:
                    self.create_multi_timeframe_analysis(symbol)
            
            if "🔍 Microstructure & Order Flow" in tab_map:
                with tab_map["🔍 Microstructure & Order Flow"]:
                    self.create_microstructure_analysis(df_display, symbol, timeframe)
            
            if "🧠 Smart Money Concepts" in tab_map:
                with tab_map["🧠 Smart Money Concepts"]:
                    self.create_smc_analysis(df_display)
            
            if "📈 Wyckoff Analysis" in tab_map:
                with tab_map["📈 Wyckoff Analysis"]:
                    self.create_wyckoff_analysis(df_display)
            
            if "🎯 Pattern Recognition" in tab_map:
                with tab_map["🎯 Pattern Recognition"]:
                    self.create_pattern_analysis(df_display)
            
            if "📊 Volume Analysis" in tab_map:
                with tab_map["📊 Volume Analysis"]:
                    self.create_volume_analysis(df_display)
            
            if "🤖 Advanced Analytics" in tab_map:
                with tab_map["🤖 Advanced Analytics"]:
                    self.create_advanced_analytics(df_display)
            
            if "📄 Analysis Reports" in tab_map:
                with tab_map["📄 Analysis Reports"]:
                    self.display_txt_report_analysis(symbol)

    def display_mega_market_status(self, df, symbol):
        """Display comprehensive market status with rich information"""
        st.markdown('<div class="market-overview">', unsafe_allow_html=True)
        st.markdown("## 🌍 Real-Time Market Intelligence")
        
        # Calculate comprehensive metrics
        metrics = self.calculate_market_metrics(df)
        
        # Main metrics row
        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
        
        current_price = df['close'].iloc[-1] if 'close' in df.columns else df.get('mid_price', pd.Series([0])).iloc[-1]
        prev_price = df['close'].iloc[-2] if 'close' in df.columns and len(df) > 1 else current_price
        price_change = current_price - prev_price
        price_change_pct = (price_change / prev_price * 100) if prev_price != 0 else 0
        
        with col1:
            st.metric("💰 Price", f"{current_price:.4f}", f"{price_change:+.4f}")
        
        with col2:
            st.metric("📊 Change", f"{price_change_pct:+.2f}%")
        
        with col3:
            st.metric("📏 Spread", metrics['spread'])
        
        with col4:
            st.metric("📊 ATR", metrics['atr'])
        
        with col5:
            st.metric("📈 RSI", metrics['rsi'])
        
        with col6:
            st.metric("📈 Trend", metrics['trend'])
        
        with col7:
            st.metric("🔥 Volatility", metrics['volatility'])
        
        # Manipulation and market regime
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🛡️ Manipulation", metrics['manipulation'])
        
        with col2:
            # Volume analysis
            if 'volume' in df.columns and len(df) >= 20:
                recent_vol = df['volume'].tail(20).mean()
                avg_vol = df['volume'].mean()
                vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1
                vol_status = "📈 HIGH" if vol_ratio > 1.2 else "📉 LOW" if vol_ratio < 0.8 else "➡️ NORMAL"
                st.metric("📊 Volume", vol_status, f"{vol_ratio:.2f}x")
            else:
                st.metric("📊 Volume", "N/A")
        
        with col3:
            # Market phase
            if 'SMC_premium_zone' in df.columns:
                if df['SMC_premium_zone'].iloc[-1]:
                    market_zone = "🔴 PREMIUM"
                elif df.get('SMC_discount_zone', pd.Series([False])).iloc[-1]:
                    market_zone = "🟢 DISCOUNT"
                else:
                    market_zone = "🟡 EQUILIBRIUM"
            else:
                market_zone = "🟡 NEUTRAL"
            st.metric("🎯 Market Zone", market_zone)
        
        with col4:
            # Wyckoff phase
            if 'wyckoff_phase' in df.columns:
                phase_map = {1: "🔵 ACCUMULATION", 2: "🔴 DISTRIBUTION", 3: "🟢 MARKUP", 4: "🟠 MARKDOWN"}
                current_phase = df['wyckoff_phase'].iloc[-1] if not pd.isna(df['wyckoff_phase'].iloc[-1]) else 0
                wyckoff_status = phase_map.get(current_phase, "⚪ UNKNOWN")
            else:
                wyckoff_status = "⚪ N/A"
            st.metric("📈 Wyckoff", wyckoff_status)
        
        st.markdown('</div>', unsafe_allow_html=True)

    def create_mega_price_chart(self, df, symbol, timeframe):
        """Create the MEGA price chart with all overlays"""
        st.markdown("## 📈 Advanced Price Action & Technical Analysis")
        
        # Determine chart type based on data
        is_tick_data = 'mid_price' in df.columns or 'bid' in df.columns and 'ask' in df.columns
        
        # Create mega subplot
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=[
                f"{symbol} {timeframe} - {'Tick' if is_tick_data else 'Price'} Action & Smart Money Analysis",
                "Volume & Order Flow",
                "Momentum & Oscillators"
            ],
            vertical_spacing=0.05,
            row_heights=[0.6, 0.2, 0.2],
            shared_xaxes=True
        )
        
        # Main price/tick chart
        if is_tick_data:
            # Tick data visualization
            if 'mid_price' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['mid_price'],
                    mode='lines',
                    name='Mid Price',
                    line=dict(color='blue', width=2)
                ), row=1, col=1)
            
            if 'bid' in df.columns and 'ask' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['bid'],
                    mode='lines',
                    name='Bid',
                    line=dict(color='red', width=1),
                    opacity=0.7
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['ask'],
                    mode='lines',
                    name='Ask',
                    line=dict(color='green', width=1),
                    opacity=0.7
                ), row=1, col=1)
        else:
            # Candlestick for bar data
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name="Price",
                increasing_line_color=self.colors['bullish'],
                decreasing_line_color=self.colors['bearish']
            ), row=1, col=1)
            
            # Moving averages
            ma_colors = {
                'EMA_8': '#ff6b6b', 'ema_8': '#ff6b6b',
                'EMA_21': '#4ecdc4', 'ema_21': '#4ecdc4',
                'EMA_55': '#45b7d1', 'ema_55': '#45b7d1',
                'SMA_200': '#96ceb4', 'sma_200': '#96ceb4'
            }
            
            for ma_col, color in ma_colors.items():
                if ma_col in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df.index, y=df[ma_col],
                        mode='lines', name=ma_col.upper(),
                        line=dict(color=color, width=1.5)
                    ), row=1, col=1)
            
            # Bollinger Bands
            bb_upper_cols = [col for col in df.columns if 'BB_Upper' in col or 'bb_upper' in col]
            bb_lower_cols = [col for col in df.columns if 'BB_Lower' in col or 'bb_lower' in col]
            
            if bb_upper_cols and bb_lower_cols:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[bb_upper_cols[0]],
                    mode='lines', name='BB Upper',
                    line=dict(color='gray', width=1, dash='dash'),
                    showlegend=False
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[bb_lower_cols[0]],
                    mode='lines', name='BB Lower',
                    line=dict(color='gray', width=1, dash='dash'),
                    fill='tonexty', fillcolor='rgba(128,128,128,0.1)',
                    showlegend=False
                ), row=1, col=1)
        
        # Add SMC overlays
        self.add_smc_overlays(fig, df, row=1)
        
        # Volume
        if 'volume' in df.columns:
            colors = ['green' if c >= o else 'red' for c, o in zip(df['close'], df['open'])] if 'close' in df.columns and 'open' in df.columns else ['blue'] * len(df)
            
            fig.add_trace(go.Bar(
                x=df.index, y=df['volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7
            ), row=2, col=1)
        
        # RSI
        rsi_col = None
        for col in ['RSI_14', 'rsi_14', 'RSI', 'rsi']:
            if col in df.columns:
                rsi_col = col
                break
        
        if rsi_col:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[rsi_col],
                mode='lines', name='RSI',
                line=dict(color='purple', width=1.5)
            ), row=3, col=1)
            
            # RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1)
            fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.3, row=3, col=1)
        
        # MACD
        macd_col = None
        signal_col = None
        for col in ['MACD', 'macd', 'MACD_12_26_9']:
            if col in df.columns:
                macd_col = col
                break
        for col in ['MACD_Signal', 'macd_signal', 'MACDs_12_26_9']:
            if col in df.columns:
                signal_col = col
                break
        
        if macd_col and signal_col:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[macd_col],
                mode='lines', name='MACD',
                line=dict(color='blue', width=1.5)
            ), row=3, col=1)
            
            fig.add_trace(go.Scatter(
                x=df.index, y=df[signal_col],
                mode='lines', name='Signal',
                line=dict(color='red', width=1.5)
            ), row=3, col=1)
        
        # Update layout
        fig.update_layout(
            title=f"{symbol} {timeframe} - Advanced Technical Analysis",
            template=st.session_state.get('chart_theme', 'plotly_dark'),
            height=800,
            showlegend=True,
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Update y-axis labels
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="Oscillators", row=3, col=1)
        
        st.plotly_chart(fig, use_container_width=True)

    def add_smc_overlays(self, fig, df, row=1):
        """Add Smart Money Concepts overlays to chart"""
        # Fair Value Gaps
        for col in ['SMC_fvg_bullish', 'SMC_fvg_bearish']:
            if col in df.columns:
                fvg_points = df[df[col] == True]
                if not fvg_points.empty:
                    marker_symbol = 'triangle-up' if 'bullish' in col else 'triangle-down'
                    marker_color = self.colors['bullish'] if 'bullish' in col else self.colors['bearish']
                    
                    fig.add_trace(go.Scatter(
                        x=fvg_points.index,
                        y=fvg_points['low'] if 'bullish' in col else fvg_points['high'],
                        mode='markers',
                        name=f"{'Bullish' if 'bullish' in col else 'Bearish'} FVG",
                        marker=dict(symbol=marker_symbol, color=marker_color, size=10),
                        showlegend=True
                    ), row=row, col=1)
        
        # Order Blocks
        for col in ['SMC_bullish_ob', 'SMC_bearish_ob']:
            if col in df.columns:
                ob_points = df[df[col] == True]
                if not ob_points.empty:
                    marker_symbol = 'square' if 'bullish' in col else 'square'
                    marker_color = self.colors['bullish'] if 'bullish' in col else self.colors['bearish']
                    
                    fig.add_trace(go.Scatter(
                        x=ob_points.index,
                        y=ob_points['low'] if 'bullish' in col else ob_points['high'],
                        mode='markers',
                        name=f"{'Bullish' if 'bullish' in col else 'Bearish'} OB",
                        marker=dict(symbol=marker_symbol, color=marker_color, size=10),
                        showlegend=True
                    ), row=row, col=1)
        
        # Break of Structure
        for col in ['SMC_bos_bullish', 'SMC_bos_bearish']:
            if col in df.columns:
                bos_points = df[df[col] == True]
                if not bos_points.empty:
                    marker_color = self.colors['bullish'] if 'bullish' in col else self.colors['bearish']
                    
                    fig.add_trace(go.Scatter(
                        x=bos_points.index,
                        y=bos_points['close'],
                        mode='markers',
                        name=f"{'Bullish' if 'bullish' in col else 'Bearish'} BOS",
                        marker=dict(symbol='x', color=marker_color, size=12),
                        showlegend=True
                    ), row=row, col=1)

    def create_multi_timeframe_analysis(self, symbol):
        """Create multi-timeframe analysis"""
        st.subheader("Multi-Timeframe Trend Alignment")
        
        available_tfs = self.available_timeframes.get(symbol, [])
        if len(available_tfs) < 2:
            st.info("Multi-timeframe analysis requires at least two timeframes.")
            return
        
        # Load data for all timeframes
        tf_data = {}
        for tf in available_tfs:
            df = self.load_symbol_data(symbol, tf)
            if df is not None and len(df) > 0:
                tf_data[tf] = df
        
        if not tf_data:
            st.warning("No data available for multi-timeframe analysis.")
            return
        
        # Create trend alignment table
        trends = []
        for tf, df in tf_data.items():
            if len(df) >= 21:
                short_ma = df['close'].rolling(8).mean().iloc[-1]
                long_ma = df['close'].rolling(21).mean().iloc[-1]
                trend = "🟢 Bullish" if short_ma > long_ma else "🔴 Bearish"
                
                # Calculate additional metrics
                rsi = np.nan
                for col in ['RSI_14', 'rsi_14', 'RSI', 'rsi']:
                    if col in df.columns:
                        rsi = df[col].iloc[-1]
                        break
                
                trends.append({
                    'Timeframe': tf,
                    'Trend (8/21 EMA)': trend,
                    'Current Price': f"{df['close'].iloc[-1]:.4f}",
                    'RSI': f"{rsi:.1f}" if not pd.isna(rsi) else "N/A",
                    'Bars': len(df)
                })
        
        if trends:
            # Sort by timeframe duration (approximately)
            tf_order = {'tick': 0, '1min': 1, '5min': 2, '15min': 3, '30min': 4, '1H': 5, '4H': 6, '1D': 7}
            trends.sort(key=lambda x: tf_order.get(x['Timeframe'], 99))
            
            st.dataframe(pd.DataFrame(trends), use_container_width=True)
        
        # Create multi-timeframe chart
        st.subheader("Multi-Timeframe Performance Comparison")
        
        # Select timeframes to display (limit to 4 for clarity)
        display_tfs = list(tf_data.keys())[:4]
        
        # Create normalized price chart
        fig = go.Figure()
        
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
        
        for i, tf in enumerate(display_tfs):
            df = tf_data[tf]
            if len(df) > 1:
                # Normalize to percentage change from start
                normalized = ((df['close'] / df['close'].iloc[0]) - 1) * 100
                
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=normalized,
                    mode='lines',
                    name=tf,
                    line=dict(color=colors[i % len(colors)], width=2)
                ))
        
        fig.update_layout(
            title="Multi-Timeframe Performance Comparison",
            xaxis_title="Time",
            yaxis_title="Performance (%)",
            template=st.session_state.get('chart_theme', 'plotly_dark'),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def create_microstructure_analysis(self, df, symbol, timeframe):
        """Create comprehensive microstructure analysis"""
        st.subheader("Microstructure & Order Flow Analysis")
        
        # Check if we have tick data
        is_tick_data = 'bid' in df.columns and 'ask' in df.columns
        
        if is_tick_data:
            # Tick data analysis
            col1, col2, col3 = st.columns(3)
            
            with col1:
                spread = df['ask'] - df['bid']
                st.metric("Average Spread", f"{spread.mean():.5f}")
                st.metric("Spread Volatility", f"{spread.std():.5f}")
            
            with col2:
                if 'mid_price' in df.columns:
                    price_changes = df['mid_price'].diff().dropna()
                else:
                    price_changes = ((df['ask'] + df['bid']) / 2).diff().dropna()
                
                st.metric("Tick Volatility", f"{price_changes.std():.6f}")
                st.metric("Max Tick Move", f"{abs(price_changes).max():.6f}")
            
            with col3:
                # Manipulation detection
                manipulation_cols = ['spoofing_detected', 'layering_detected', 'momentum_ignition']
                manipulation_count = sum(df[col].sum() for col in manipulation_cols if col in df.columns)
                
                st.metric("Manipulation Events", f"{int(manipulation_count)}")
                
                if 'order_flow_imbalance' in df.columns:
                    avg_imbalance = df['order_flow_imbalance'].mean()
                    st.metric("Order Flow Imbalance", f"{avg_imbalance:.4f}")
            
            # Spread analysis chart
            st.subheader("Spread Analysis")
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df.index,
                y=spread,
                mode='lines',
                name='Spread',
                line=dict(color='purple', width=2)
            ))
            
            fig.update_layout(
                title="Bid-Ask Spread Over Time",
                xaxis_title="Time",
                yaxis_title="Spread",
                template=st.session_state.get('chart_theme', 'plotly_dark'),
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Price change distribution
            st.subheader("Tick-to-Tick Price Change Distribution")
            
            fig = px.histogram(
                price_changes,
                nbins=50,
                title="Distribution of Tick-to-Tick Price Changes",
                labels={'value': 'Price Change', 'count': 'Frequency'},
                template=st.session_state.get('chart_theme', 'plotly_dark')
            )
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Manipulation timeline if available
            manipulation_cols = ['spoofing_detected', 'layering_detected', 'momentum_ignition']
            if any(col in df.columns for col in manipulation_cols):
                st.subheader("Market Manipulation Timeline")
                
                manipulation_data = pd.DataFrame(index=df.index)
                
                for col in manipulation_cols:
                    if col in df.columns:
                        manipulation_data[col] = df[col]
                
                if not manipulation_data.empty:
                    fig = go.Figure()
                    
                    colors = {'spoofing_detected': 'red', 'layering_detected': 'orange', 'momentum_ignition': 'purple'}
                    
                    for col in manipulation_data.columns:
                        events = df[df[col] == True].index
                        
                        if len(events) > 0:
                            fig.add_trace(go.Scatter(
                                x=events,
                                y=[colors.get(col, 'blue')] * len(events),
                                mode='markers',
                                name=col.replace('_', ' ').title(),
                                marker=dict(symbol='square', size=10, color=colors.get(col, 'blue'))
                            ))
                    
                    fig.update_layout(
                        title="Market Manipulation Events Timeline",
                        xaxis_title="Time",
                        yaxis_title="Event Type",
                        template=st.session_state.get('chart_theme', 'plotly_dark'),
                        height=250,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        else:
            # Bar data microstructure analysis
            st.info("Detailed microstructure analysis requires tick data. Showing basic analysis for bar data.")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                price_changes = df['close'].pct_change().dropna()
                st.metric("Price Volatility", f"{price_changes.std() * 100:.4f}%")
            
            with col2:
                if 'volume' in df.columns:
                    vol_change = df['volume'].pct_change().dropna()
                    st.metric("Volume Volatility", f"{vol_change.std() * 100:.4f}%")
                else:
                    st.metric("Volume Volatility", "N/A")
            
            with col3:
                if 'high' in df.columns and 'low' in df.columns:
                    hl_range = (df['high'] - df['low']) / df['low'] * 100
                    st.metric("Avg H-L Range", f"{hl_range.mean():.4f}%")
                else:
                    st.metric("Avg H-L Range", "N/A")
            
            # Price change distribution
            st.subheader("Price Change Distribution")
            
            fig = px.histogram(
                price_changes * 100,
                nbins=50,
                title="Distribution of Price Changes (%)",
                labels={'value': 'Price Change (%)', 'count': 'Frequency'},
                template=st.session_state.get('chart_theme', 'plotly_dark')
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

    def create_smc_analysis(self, df_display):
        """Create comprehensive Smart Money Concepts analysis"""
        st.subheader("Smart Money Concepts Analysis")
        
        # Check for SMC columns
        smc_columns = [col for col in df_display.columns if 'SMC_' in col]
        
        if not smc_columns:
            st.info("No Smart Money Concepts data available in this dataset.")
            return
        
        # Create metrics for SMC indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Fair Value Gaps
            fvg_bullish = df_display['SMC_fvg_bullish'].sum() if 'SMC_fvg_bullish' in df_display.columns else 0
            fvg_bearish = df_display['SMC_fvg_bearish'].sum() if 'SMC_fvg_bearish' in df_display.columns else 0
            st.metric("Bullish FVG", f"{int(fvg_bullish)}")
            st.metric("Bearish FVG", f"{int(fvg_bearish)}")
        
        with col2:
            # Order Blocks
            ob_bullish = df_display['SMC_bullish_ob'].sum() if 'SMC_bullish_ob' in df_display.columns else 0
            ob_bearish = df_display['SMC_bearish_ob'].sum() if 'SMC_bearish_ob' in df_display.columns else 0
            st.metric("Bullish OB", f"{int(ob_bullish)}")
            st.metric("Bearish OB", f"{int(ob_bearish)}")
        
        with col3:
            # Break of Structure
            bos_bullish = df_display['SMC_bos_bullish'].sum() if 'SMC_bos_bullish' in df_display.columns else 0
            bos_bearish = df_display['SMC_bos_bearish'].sum() if 'SMC_bos_bearish' in df_display.columns else 0
            st.metric("Bullish BOS", f"{int(bos_bullish)}")
            st.metric("Bearish BOS", f"{int(bos_bearish)}")
        
        with col4:
            # Market Structure
            if 'SMC_structure' in df_display.columns:
                current_structure = df_display['SMC_structure'].iloc[-1]
                st.metric("Structure", current_structure.upper())
            
            # Liquidity
            if 'SMC_liquidity_grab' in df_display.columns:
                liquidity_grabs = df_display['SMC_liquidity_grab'].sum()
                st.metric("Liquidity Grabs", f"{int(liquidity_grabs)}")
        
        # SMC Timeline Chart
        st.subheader("SMC Events Timeline")
        
        fig = go.Figure()
        
        # Add price as background
        fig.add_trace(go.Scatter(
            x=df_display.index,
            y=df_display['close'],
            mode='lines',
            name='Price',
            line=dict(color='gray', width=1),
            opacity=0.5
        ))
        
        # Add SMC events
        if 'SMC_fvg_bullish' in df_display.columns:
            fvg_bullish_points = df_display[df_display['SMC_fvg_bullish'] == True]
            if not fvg_bullish_points.empty:
                fig.add_trace(go.Scatter(
                    x=fvg_bullish_points.index,
                    y=fvg_bullish_points['low'],
                    mode='markers',
                    name='Bullish FVG',
                    marker=dict(symbol='triangle-up', color='lime', size=12)
                ))
        
        if 'SMC_fvg_bearish' in df_display.columns:
            fvg_bearish_points = df_display[df_display['SMC_fvg_bearish'] == True]
            if not fvg_bearish_points.empty:
                fig.add_trace(go.Scatter(
                    x=fvg_bearish_points.index,
                    y=fvg_bearish_points['high'],
                    mode='markers',
                    name='Bearish FVG',
                    marker=dict(symbol='triangle-down', color='red', size=12)
                ))
        
        if 'SMC_bos_bullish' in df_display.columns:
            bos_bullish_points = df_display[df_display['SMC_bos_bullish'] == True]
            if not bos_bullish_points.empty:
                fig.add_trace(go.Scatter(
                    x=bos_bullish_points.index,
                    y=bos_bullish_points['close'],
                    mode='markers',
                    name='Bullish BOS',
                    marker=dict(symbol='x', color='cyan', size=15)
                ))
        
        fig.update_layout(
            title="Smart Money Concepts Timeline",
            xaxis_title="Time",
            yaxis_title="Price",
            template=st.session_state.get('chart_theme', 'plotly_dark'),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Market Zone Analysis
        st.subheader("Market Zone Analysis")
        
        zone_data = []
        
        if 'SMC_premium_zone' in df_display.columns:
            premium_count = df_display['SMC_premium_zone'].sum()
            zone_data.append({
                'Zone': 'Premium',
                'Count': int(premium_count),
                'Percentage': f"{(premium_count / len(df_display)) * 100:.1f}%"
            })
        
        if 'SMC_discount_zone' in df_display.columns:
            discount_count = df_display['SMC_discount_zone'].sum()
            zone_data.append({
                'Zone': 'Discount',
                'Count': int(discount_count),
                'Percentage': f"{(discount_count / len(df_display)) * 100:.1f}%"
            })
        
        if 'SMC_equilibrium' in df_display.columns:
            equilibrium_count = df_display['SMC_equilibrium'].sum()
            zone_data.append({
                'Zone': 'Equilibrium',
                'Count': int(equilibrium_count),
                'Percentage': f"{(equilibrium_count / len(df_display)) * 100:.1f}%"
            })
        
        if zone_data:
            zone_df = pd.DataFrame(zone_data)
            st.dataframe(zone_df, use_container_width=True)

    def create_wyckoff_analysis(self, df_display):
        """Create comprehensive Wyckoff analysis"""
        st.subheader("Wyckoff Method Analysis")
        
        # Check for Wyckoff columns
        wyckoff_columns = [col for col in df_display.columns if 'wyckoff' in col.lower()]
        
        if not wyckoff_columns:
            st.info("No Wyckoff analysis data available in this dataset.")
            return
        
        # Create metrics for Wyckoff indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Accumulation/Distribution
            if 'wyckoff_accumulation' in df_display.columns:
                accumulation = df_display['wyckoff_accumulation'].sum()
                st.metric("Accumulation Signals", f"{int(accumulation)}")
            
            if 'wyckoff_distribution' in df_display.columns:
                distribution = df_display['wyckoff_distribution'].sum()
                st.metric("Distribution Signals", f"{int(distribution)}")
        
        with col2:
            # Springs and Upthrusts
            if 'wyckoff_spring' in df_display.columns:
                springs = df_display['wyckoff_spring'].sum()
                st.metric("Springs", f"{int(springs)}")
            
            if 'wyckoff_upthrust' in df_display.columns:
                upthrusts = df_display['wyckoff_upthrust'].sum()
                st.metric("Upthrusts", f"{int(upthrusts)}")
        
        with col3:
            # No Demand/Supply
            if 'wyckoff_no_demand' in df_display.columns:
                no_demand = df_display['wyckoff_no_demand'].sum()
                st.metric("No Demand", f"{int(no_demand)}")
            
            if 'wyckoff_no_supply' in df_display.columns:
                no_supply = df_display['wyckoff_no_supply'].sum()
                st.metric("No Supply", f"{int(no_supply)}")
        
        with col4:
            # Effort vs Result
            if 'wyckoff_effort' in df_display.columns and 'wyckoff_result' in df_display.columns:
                latest_effort = df_display['wyckoff_effort'].iloc[-1] if not pd.isna(df_display['wyckoff_effort'].iloc[-1]) else 0
                latest_result = df_display['wyckoff_result'].iloc[-1] if not pd.isna(df_display['wyckoff_result'].iloc[-1]) else 0
                
                if latest_effort > latest_result:
                    effort_result = "📉 Divergence"
                else:
                    effort_result = "📈 Harmony"
                
                st.metric("Effort vs Result", effort_result)
        
        # Wyckoff Volume Analysis
        if 'wyckoff_spread' in df_display.columns and 'volume' in df_display.columns:
            st.subheader("Wyckoff Volume Spread Analysis")
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=["Price & Spread", "Volume"],
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3],
                shared_xaxes=True
            )
            
            # Price and spread
            fig.add_trace(go.Scatter(
                x=df_display.index,
                y=df_display['close'],
                mode='lines',
                name='Close Price',
                line=dict(color='blue', width=2)
            ), row=1, col=1)
            
            # Wyckoff spread
            fig.add_trace(go.Scatter(
                x=df_display.index,
                y=df_display['wyckoff_spread'],
                mode='lines',
                name='Wyckoff Spread',
                line=dict(color='orange', width=1),
                yaxis='y2'
            ), row=1, col=1)
            
            # Volume
            colors = ['green' if c >= o else 'red' for c, o in zip(df_display['close'], df_display['open'])]
            fig.add_trace(go.Bar(
                x=df_display.index,
                y=df_display['volume'],
                name='Volume',
                marker_color=colors
            ), row=2, col=1)
            
            fig.update_layout(
                title="Wyckoff Volume Spread Analysis",
                template=st.session_state.get('chart_theme', 'plotly_dark'),
                height=600,
                yaxis2=dict(overlaying='y', side='right', title='Spread')
            )
            
            st.plotly_chart(fig, use_container_width=True)

    def create_pattern_analysis(self, df_display):
        """Create pattern recognition analysis"""
        st.subheader("Pattern Recognition Analysis")
        
        # Check for pattern columns
        pattern_columns = [col for col in df_display.columns if 'Pattern_' in col or 'harmonic_' in col]
        
        if not pattern_columns:
            st.info("No pattern data available in this dataset.")
            return
        
        # Candlestick patterns
        candlestick_patterns = [col for col in pattern_columns if 'Pattern_' in col]
        if candlestick_patterns:
            st.markdown("#### 🕯️ Candlestick Patterns")
            
            pattern_counts = {}
            for pattern in candlestick_patterns:
                if pattern in df_display.columns:
                    count = df_display[pattern].sum() if df_display[pattern].dtype in ['int64', 'float64'] else 0
                    if count > 0:
                        pattern_counts[pattern.replace('Pattern_', '')] = int(count)
            
            if pattern_counts:
                # Display pattern counts
                pattern_df = pd.DataFrame(
                    list(pattern_counts.items()),
                    columns=['Pattern', 'Count']
                ).sort_values('Count', ascending=False)
                
                st.dataframe(pattern_df, use_container_width=True)
        
        # Harmonic patterns
        harmonic_patterns = [col for col in pattern_columns if 'harmonic_' in col and not 'score' in col]
        if harmonic_patterns:
            st.markdown("#### 🎵 Harmonic Patterns")
            
            harmonic_counts = {}
            for pattern in harmonic_patterns:
                if pattern in df_display.columns:
                    count = df_display[pattern].sum()
                    if count > 0:
                        # Get corresponding score
                        score_col = pattern + '_score'
                        if score_col in df_display.columns:
                            avg_score = df_display[df_display[pattern] == True][score_col].mean()
                            harmonic_counts[pattern.replace('harmonic_', '').upper()] = {
                                'Count': int(count),
                                'Avg Score': f"{avg_score:.2f}" if not pd.isna(avg_score) else "N/A"
                            }
            
            if harmonic_counts:
                harmonic_df = pd.DataFrame(harmonic_counts).T
                st.dataframe(harmonic_df, use_container_width=True)

    def create_volume_analysis(self, df_display):
        """Create advanced volume analysis"""
        st.subheader("Advanced Volume Analysis")
        
        if 'volume' not in df_display.columns:
            st.info("No volume data available.")
            return
        
        # Volume metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_volume = df_display['volume'].mean()
            st.metric("Average Volume", f"{avg_volume:,.0f}")
        
        with col2:
            recent_volume = df_display['volume'].tail(20).mean()
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
            st.metric("Recent Vol Ratio", f"{volume_ratio:.2f}x")
        
        with col3:
            volume_trend = "📈" if df_display['volume'].tail(20).mean() > df_display['volume'].tail(50).mean() else "📉"
            st.metric("Volume Trend", volume_trend)
        
        with col4:
            max_volume = df_display['volume'].max()
            max_volume_date = df_display['volume'].idxmax()
            st.metric("Peak Volume", f"{max_volume:,.0f}")
        
        # Volume Profile
        st.markdown("#### Volume Profile")
        
        # Create price bins
        price_range = df_display['high'].max() - df_display['low'].min()
        num_bins = 30
        bin_size = price_range / num_bins
        
        volume_profile = []
        for i in range(num_bins):
            bin_low = df_display['low'].min() + (i * bin_size)
            bin_high = bin_low + bin_size
            
            # Find bars where price traded in this range
            mask = ((df_display['low'] <= bin_high) & (df_display['high'] >= bin_low))
            bin_volume = df_display.loc[mask, 'volume'].sum()
            
            volume_profile.append({
                'Price': (bin_low + bin_high) / 2,
                'Volume': bin_volume
            })
        
        vp_df = pd.DataFrame(volume_profile)
        
        # Create horizontal bar chart for volume profile
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=vp_df['Volume'],
            y=vp_df['Price'],
            orientation='h',
            name='Volume Profile',
            marker_color='lightblue'
        ))
        
        # Add current price line
        current_price = df_display['close'].iloc[-1]
        fig.add_hline(y=current_price, line_dash="dash", line_color="red", 
                      annotation_text=f"Current Price: {current_price:.4f}")
        
        fig.update_layout(
            title="Volume Profile",
            xaxis_title="Volume",
            yaxis_title="Price",
            template=st.session_state.get('chart_theme', 'plotly_dark'),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def create_advanced_analytics(self, df_display):
        """Create advanced analytics section"""
        col1, col2 = st.columns(2)
        
        with col1:
            if st.session_state.get('show_ml_features', True):
                self.create_ml_features_dashboard(df_display)
        
        with col2:
            if st.session_state.get('show_statistical', True):
                self.create_statistical_dashboard(df_display)
        
        if st.session_state.get('show_risk', True):
            self.create_risk_metrics_dashboard(df_display)

    def create_ml_features_dashboard(self, df_display):
        """Create ML features dashboard"""
        st.markdown("#### 🤖 Machine Learning Features")
        
        # Count available features
        technical_features = [col for col in df_display.columns 
                            if any(ind in col.upper() for ind in ['RSI', 'MACD', 'ATR', 'BB', 'EMA', 'SMA', 'STOCH', 'ADX'])]
        
        pattern_features = [col for col in df_display.columns 
                          if any(pat in col.lower() for pat in ['pattern', 'harmonic', 'smc', 'wyckoff'])]
        
        microstructure_features = [col for col in df_display.columns 
                                 if any(ms in col.lower() for ms in ['spread', 'imbalance', 'manipulation', 'tick'])]
        
        feature_summary = {
            'Technical Indicators': len(technical_features),
            'Pattern Features': len(pattern_features),
            'Microstructure Features': len(microstructure_features),
            'Total Features': len(df_display.columns)
        }
        
        feature_df = pd.DataFrame(list(feature_summary.items()), columns=['Category', 'Count'])
        st.dataframe(feature_df, use_container_width=True)
        
        # Feature importance (simulated)
        if technical_features:
            st.markdown("##### Top Technical Features")
            
            # Calculate feature statistics
            feature_stats = []
            for feature in technical_features[:10]:  # Top 10
                if feature in df_display.columns and pd.api.types.is_numeric_dtype(df_display[feature]):
                    current_val = df_display[feature].iloc[-1]
                    mean_val = df_display[feature].mean()
                    std_val = df_display[feature].std()
                    
                    if not pd.isna(current_val) and std_val > 0:
                        z_score = (current_val - mean_val) / std_val
                        feature_stats.append({
                            'Feature': feature,
                            'Current': f"{current_val:.4f}",
                            'Z-Score': f"{z_score:.2f}"
                        })
            
            if feature_stats:
                stats_df = pd.DataFrame(feature_stats)
                st.dataframe(stats_df, use_container_width=True)

    def create_statistical_dashboard(self, df_display):
        """Create statistical analysis dashboard"""
        st.markdown("#### 📊 Statistical Analysis")
        
        if len(df_display) > 20:
            # Price returns
            returns = df_display['close'].pct_change().dropna()
            
            # Basic statistics
            stats_data = {
                'Mean Return': f"{returns.mean():.6f}",
                'Volatility': f"{returns.std():.6f}",
                'Sharpe Ratio': f"{(returns.mean() / returns.std()) * np.sqrt(252):.3f}",
                'Max Drawdown': f"{((df_display['close'] / df_display['close'].cummax()) - 1).min():.4f}",
                'Win Rate': f"{(returns > 0).sum() / len(returns) * 100:.1f}%"
            }
            
            stats_df = pd.DataFrame(list(stats_data.items()), columns=['Metric', 'Value'])
            st.dataframe(stats_df, use_container_width=True)
            
            # Distribution plot
            fig = px.histogram(
                returns * 100,
                nbins=30,
                title="Return Distribution (%)",
                labels={'value': 'Return (%)', 'count': 'Frequency'},
                template=st.session_state.get('chart_theme', 'plotly_dark')
            )
            
            # Add normal distribution overlay
            if stats:
                import scipy.stats as scipystats
                x_range = np.linspace(returns.min() * 100, returns.max() * 100, 100)
                normal_dist = scipystats.norm.pdf(x_range, returns.mean() * 100, returns.std() * 100)
                
                fig.add_trace(go.Scatter(
                    x=x_range,
                    y=normal_dist * len(returns) * (returns.max() - returns.min()) * 100 / 30,
                    mode='lines',
                    name='Normal Distribution',
                    line=dict(color='red', width=2)
                ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

    def create_risk_metrics_dashboard(self, df_display):
        """Create risk metrics dashboard"""
        st.markdown("#### ⚠️ Risk Metrics")
        
        # Calculate risk metrics
        returns = df_display['close'].pct_change().dropna()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Value at Risk (95% confidence)
            var_95 = np.percentile(returns, 5)
            st.metric("VaR (95%)", f"{var_95 * 100:.2f}%")
            
            # Conditional VaR
            cvar_95 = returns[returns <= var_95].mean()
            st.metric("CVaR (95%)", f"{cvar_95 * 100:.2f}%")
        
        with col2:
            # Maximum drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            
            st.metric("Max Drawdown", f"{max_drawdown * 100:.2f}%")
            
            # Drawdown duration
            drawdown_duration = (drawdown < 0).sum()
            st.metric("Drawdown Days", f"{drawdown_duration}")
        
        with col3:
            # Risk-adjusted returns
            sortino_ratio = returns.mean() / returns[returns < 0].std() * np.sqrt(252) if len(returns[returns < 0]) > 0 else 0
            st.metric("Sortino Ratio", f"{sortino_ratio:.3f}")
            
            # Calmar ratio
            calmar_ratio = (returns.mean() * 252) / abs(max_drawdown) if max_drawdown != 0 else 0
            st.metric("Calmar Ratio", f"{calmar_ratio:.3f}")

    def display_txt_report_analysis(self, symbol):
        """Display analysis from TXT reports"""
        txt_content = self.load_latest_txt_report(symbol)
        
        if txt_content:
            st.markdown("#### 📄 Latest Analysis Report")
            
            # Parse the report content
            sections = txt_content.split('\n\n')
            
            for section in sections:
                if section.strip():
                    lines = section.strip().split('\n')
                    if lines:
                        header = lines[0]
                        
                        # Format different sections
                        if 'MARKET OVERVIEW' in header:
                            with st.expander("🌍 Market Overview", expanded=True):
                                st.text('\n'.join(lines[1:]))
                        
                        elif 'TECHNICAL ANALYSIS' in header:
                            with st.expander("📊 Technical Analysis", expanded=True):
                                st.text('\n'.join(lines[1:]))
                        
                        elif 'TRADING STRATEGY' in header:
                            with st.expander("🎯 Trading Strategy", expanded=True):
                                st.text('\n'.join(lines[1:]))
                        
                        elif 'RISK MANAGEMENT' in header:
                            with st.expander("⚠️ Risk Management", expanded=True):
                                st.text('\n'.join(lines[1:]))
        else:
            st.info(f"No analysis report found for {symbol}")

# Main execution
def main():
    """Main function to run the dashboard"""
    dashboard = ZANFLOWUltimateMegaDashboard()
    dashboard.create_mega_dashboard()

if __name__ == "__main__":
    main()