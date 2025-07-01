# ncos_integrated_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add components and utils to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules from components and utils
from components.timeframe_converter import TimeframeConverter
from utils.smc_analyser import SMCAnalyzer
from utils.wyckoff_analyzer import WyckoffAnalyzer
from components.chart_builder import ChartBuilder
from components.technical_analysis import TechnicalAnalysis
from components.volume_profile_analyzer import VolumeProfileAnalyzer
from utils.data_processor import DataProcessor
from pathlib import Path
BAR_DATA_DIR = Path(st.secrets.get("bar_data_directory", "."))
# Page config
st.set_page_config(
    page_title="🎯 ncOS SMC Wyckoff Dashboard", 
    page_icon="🎯", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {padding: 0rem 1rem;}
    .block-container {padding-top: 1rem;}
    
    /* Header Styles */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.1);
        height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: bold;
        margin: 0.5rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #b8c7db;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Status Colors */
    .bullish { color: #00ff88; }
    .bearish { color: #ff3366; }
    .neutral { color: #ffa500; }
    
    /* Alert Box */
    .alert-box {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
        50% { box-shadow: 0 4px 20px rgba(255,107,107,0.5); }
        100% { box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
    }
    
    /* POI Card */
    .poi-card {
        background: rgba(30, 60, 114, 0.3);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.5rem 0;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(30, 60, 114, 0.5);
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

class NCOSDashboard:
    def __init__(self):
        # Initialize components
        self.data_processor = DataProcessor()
        self.timeframe_converter = TimeframeConverter()
        self.smc_analyzer = SMCAnalyzer()
        self.wyckoff_analyzer = WyckoffAnalyzer()
        self.chart_builder = ChartBuilder()
        self.tech_analyzer = TechnicalAnalysis()
        self.volume_analyzer = VolumeProfileAnalyzer()
        
        # Initialize session state
        self.initialize_session_state()
        
    def initialize_session_state(self):
        """Initialize session state variables"""
        defaults = {
            'selected_pair': 'ETHUSD',
            'selected_timeframe': 'M1',
            'data': None,
            'all_timeframes': {},
            'analysis_results': {},
            'last_update': None
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
                
    def load_and_process_data(self, pair):
        """Load and process data for the selected pair"""
        try:
            # First try to load from CSV file
            filename = BAR_DATA_DIR / f"{pair}_M1_bars.csv"
            if os.path.exists(filename):
                # Load tab‑separated CSV file
                df = pd.read_csv(filename, sep='\t', parse_dates=['timestamp'])
                # Type check after loading DataFrame from CSV
                if not isinstance(df, pd.DataFrame):
                    st.error("Loaded object is not a DataFrame. Please verify the file format and loader.")
                    return False
                df.columns = df.columns.str.strip()

                # Ensure proper column mapping (volume = tickvol)
                if 'tickvol' in df.columns and 'volume' in df.columns:
                    df['volume'] = df['tickvol']

                # Sort by timestamp in descending order (newest first)
                df = df.sort_values('timestamp', ascending=False)

                st.session_state.data = df

                # Ensure essential OHLC columns exist
                required_cols = ['open', 'high', 'low', 'close']
                if not all(col in df.columns for col in required_cols):
                    if 'bid' in df.columns and 'ask' in df.columns:
                        df['close'] = (df['bid'] + df['ask']) / 2
                        df['open'] = df['high'] = df['low'] = df['close']
                    elif 'price' in df.columns:
                        df['close'] = df['price']
                        df['open'] = df['high'] = df['low'] = df['close']
                    else:
                        st.error("CSV lacks standard OHLC or a recognisable price column (mid/last/bid/ask). Please verify your data.")
                        return False

                # Generate all timeframes
                if hasattr(self, 'timeframe_converter'):
                    st.session_state.all_timeframes = self.timeframe_converter.generate_all_timeframes(df)
                else:
                    # Fallback if component not available
                    st.session_state.all_timeframes = {'M1': df}

                # Record last update time
                st.session_state.last_update = datetime.now()

                return True

            # If file doesn't exist, try to use the data processor component
            elif hasattr(self, 'data_processor'):
                df = self.data_processor.load_data(pair, 'M1')
                # Type check after loading DataFrame from data_processor
                if not isinstance(df, pd.DataFrame):
                    st.error("Loaded object is not a DataFrame. Please verify the data processor output.")
                    return False
                # Fallback: ensure tab-separation if loading from CSV path
                if isinstance(df, str) and os.path.isfile(df):
                    df = pd.read_csv(df, sep='\t', parse_dates=['timestamp'])
                    if not isinstance(df, pd.DataFrame):
                        st.error("Loaded object is not a DataFrame. Please verify the file format and loader.")
                        return False
                    df.columns = df.columns.str.strip()
                if df is not None and not df.empty:
                    st.session_state.data = df

                    # Ensure essential OHLC columns exist
                    required_cols = ['open', 'high', 'low', 'close']
                    if not all(col in df.columns for col in required_cols):
                        if 'bid' in df.columns and 'ask' in df.columns:
                            df['close'] = (df['bid'] + df['ask']) / 2
                            df['open'] = df['high'] = df['low'] = df['close']
                        elif 'price' in df.columns:
                            df['close'] = df['price']
                            df['open'] = df['high'] = df['low'] = df['close']
                        else:
                            st.error("CSV lacks standard OHLC or a recognisable price column (mid/last/bid/ask). Please verify your data.")
                            return False

                    # Generate all timeframes
                    if hasattr(self, 'timeframe_converter'):
                        st.session_state.all_timeframes = self.timeframe_converter.generate_all_timeframes(df)
                    else:
                        st.session_state.all_timeframes = {'M1': df}

                    st.session_state.last_update = datetime.now()
                    return True

            # Fallback to sample data
            st.warning(f"No data found for {pair}. Using sample data.")
            df = self.generate_sample_data(pair)
            st.session_state.data = df
            st.session_state.all_timeframes = {'M1': df}
            st.session_state.last_update = datetime.now()
            return True

        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False

    def load_tick_data(self, pair):
        """Load tick-level data for the selected pair"""
        try:
            filename = BAR_DATA_DIR / f"{pair}_ticks.csv"
            if os.path.exists(filename):
                # Load tab‑separated CSV file
                df.columns = df.columns.str.strip()

                # Sort by timestamp in descending order
                df = df.sort_values('timestamp', ascending=False)

                st.session_state.tick_data = df
                return True

            st.session_state.tick_data = None
            return False

        except Exception as e:
            st.error(f"Error loading tick data: {str(e)}")
            return False
            
    def generate_sample_data(self, pair):
        """Generate sample data for testing"""
        dates = pd.date_range(end=datetime.now(), periods=1000, freq='1min')
        price_base = {'ETHUSD': 2500, 'BTCUSD': 45000}.get(pair, 100)
        
        # Generate realistic price movement
        returns = np.random.normal(0, 0.001, len(dates))
        price_series = price_base * np.exp(np.cumsum(returns))
        
        data = pd.DataFrame({
            'open': price_series * (1 + np.random.normal(0, 0.0005, len(dates))),
            'high': price_series * (1 + np.abs(np.random.normal(0, 0.001, len(dates)))),
            'low': price_series * (1 - np.abs(np.random.normal(0, 0.001, len(dates)))),
            'close': price_series,
            'volume': np.random.lognormal(10, 1, len(dates)),
            'tickvol': np.random.randint(50, 500, len(dates))
        }, index=dates)
        
        # Fix OHLC relationships
        data['high'] = data[['open', 'high', 'close']].max(axis=1)
        data['low'] = data[['open', 'low', 'close']].min(axis=1)
        
        return data
        
    def run_comprehensive_analysis(self, timeframe='M1'):
        """Run all analyses on selected timeframe"""
        try:
            # Get data for selected timeframe
            df = st.session_state.all_timeframes.get(timeframe) or st.session_state.data
            if not isinstance(df, pd.DataFrame):
                st.error("Error loading data: expected a valid DataFrame. Please check your data pipeline.")
                return False

            # Run SMC Analysis
            smc_results = self.smc_analyzer.analyze(df)
            
            # Run Wyckoff Analysis
            wyckoff_results = self.wyckoff_analyzer.analyze(df)
            
            # Run Technical Analysis
            if hasattr(self.tech_analyzer, 'calculate_indicators'):
                tech_results = self.tech_analyzer.calculate_indicators(df)
            else:
                tech_results = {}
                st.warning("Technical analysis module is missing 'calculate_indicators' method.")
            
            # Run Volume Profile Analysis
            volume_results = self.volume_analyzer.analyze(df)
            
            # Combine all results
            analysis_results = {
                'smc': smc_results,
                'wyckoff': wyckoff_results,
                'technical': tech_results,
                'volume_profile': volume_results,
                'data': df
            }
            
            # Calculate summary metrics
            analysis_results['summary'] = self.calculate_summary_metrics(analysis_results)
            
            st.session_state.analysis_results = analysis_results
            
            return True
            
        except Exception as e:
            st.error(f"Analysis error: {str(e)}")
            return False
            
    def calculate_summary_metrics(self, results):
        """Calculate summary metrics from all analyses"""
        summary = {}
        
        # Market Bias from SMC
        market_structure = results['smc'].get('market_structure', [])
        if market_structure:
            recent = market_structure[-10:] if len(market_structure) > 10 else market_structure
            bullish_count = sum(1 for s in recent if s.get('type') in ['HH', 'HL'])
            bearish_count = sum(1 for s in recent if s.get('type') in ['LH', 'LL'])
            
            if bullish_count > bearish_count * 1.5:
                summary['market_bias'] = 'BULLISH'
            elif bearish_count > bullish_count * 1.5:
                summary['market_bias'] = 'BEARISH'
            else:
                summary['market_bias'] = 'NEUTRAL'
        else:
            summary['market_bias'] = 'NEUTRAL'
            
        # Active POIs count
        active_pois = 0
        
        # Count order blocks
        order_blocks = results['smc'].get('order_blocks', [])
        active_pois += sum(1 for ob in order_blocks if ob.get('status') == 'active')
        
        # Count FVGs
        fvgs = results['smc'].get('fair_value_gaps', [])
        active_pois += sum(1 for fvg in fvgs if fvg.get('status') == 'unfilled')
        
        # Count liquidity zones
        liquidity = results['smc'].get('liquidity_zones', [])
        active_pois += len(liquidity)
        
        summary['active_pois'] = active_pois
        
        # Wyckoff Phase
        summary['wyckoff_phase'] = results['wyckoff'].get('current_phase', 'Unknown')
        
        # Risk Score
        if summary['wyckoff_phase'] in ['Distribution', 'Markdown']:
            summary['risk_score'] = 'HIGH'
        elif summary['market_bias'] == 'NEUTRAL':
            summary['risk_score'] = 'MEDIUM'
        else:
            summary['risk_score'] = 'LOW'
            
        return summary
        
    def render_header_section(self):
        """Render the header with key metrics"""
        st.markdown('<h1 class="main-header">🎯 ncOS SMC Wyckoff Dashboard</h1>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; color: #666; margin-bottom: 2rem;">Neural Core OS - Advanced Market Structure Analysis</p>', unsafe_allow_html=True)
        
        # Display current pair and timeframe
        col1, col2, col3 = st.columns([2, 3, 2])
        with col2:
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; background: rgba(255,255,255,0.05); border-radius: 10px;">
                <h3>{st.session_state.selected_pair} | {st.session_state.selected_timeframe}</h3>
                <p style="color: #888; margin: 0;">Last Update: {st.session_state.last_update.strftime('%H:%M:%S') if st.session_state.last_update else 'Never'}</p>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")
        
        # Key Metrics
        if 'summary' in st.session_state.analysis_results:
            summary = st.session_state.analysis_results['summary']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                bias = summary.get('market_bias', 'NEUTRAL')
                bias_class = bias.lower()
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Market Bias</div>
                    <div class="metric-value {bias_class}">{bias}</div>
                </div>
                """, unsafe_allow_html=True)
                
            with col2:
                pois = summary.get('active_pois', 0)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Active POIs</div>
                    <div class="metric-value">{pois}</div>
                </div>
                """, unsafe_allow_html=True)
                
            with col3:
                phase = summary.get('wyckoff_phase', 'Unknown')
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Wyckoff Phase</div>
                    <div class="metric-value" style="font-size: 1.8rem;">{phase.upper()}</div>
                </div>
                """, unsafe_allow_html=True)
                
            with col4:
                risk = summary.get('risk_score', 'MEDIUM')
                risk_class = 'bearish' if risk == 'HIGH' else 'bullish' if risk == 'LOW' else 'neutral'
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Risk Score</div>
                    <div class="metric-value {risk_class}">{risk}</div>
                </div>
                """, unsafe_allow_html=True)
                
    def render_advanced_price_action(self):
        """Render advanced price action tab"""
        if ('analysis_results' not in st.session_state
            or not isinstance(st.session_state.analysis_results, dict)
            or 'data' not in st.session_state.analysis_results
            or not isinstance(st.session_state.analysis_results['data'], pd.DataFrame)):
            st.warning("No analysis results available or data is invalid. Please update analysis.")
            return
            
        results = st.session_state.analysis_results
        df = results['data']
        
        # Chart controls
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            chart_type = st.selectbox("Chart Type", ["Candlestick", "Heikin Ashi", "Line", "Renko"])
        with col2:
            show_volume = st.checkbox("Show Volume", value=True)
        with col3:
            show_indicators = st.checkbox("Indicators", value=True)
            
        # Create main chart using ChartBuilder
        fig = self.chart_builder.create_main_chart(
            df, 
            st.session_state.selected_timeframe,
            chart_type=chart_type,
            show_volume=show_volume,
            analysis_results=results
        )
        
        # Add SMC overlays
        if results.get('smc'):
            # Add order blocks
            order_blocks = results['smc'].get('order_blocks', [])
            for ob in order_blocks:
                if ob.get('status') == 'active':
                    fig = self.chart_builder.add_order_block(fig, ob, df)
                    
            # Add FVGs
            fvgs = results['smc'].get('fair_value_gaps', [])
            for fvg in fvgs:
                if fvg.get('status') == 'unfilled':
                    fig = self.chart_builder.add_fair_value_gap(fig, fvg, df)
                    
        # Add Wyckoff annotations
        if results.get('wyckoff'):
            events = results['wyckoff'].get('events', [])
            for event in events[-10:]:  # Show last 10 events
                fig = self.chart_builder.add_wyckoff_event(fig, event)
                
        st.plotly_chart(fig, use_container_width=True)
        
    def render_smart_money_concepts(self):
        """Render SMC analysis tab"""
        if 'analysis_results' not in st.session_state or not st.session_state.analysis_results or 'data' not in st.session_state.analysis_results:
            st.warning("No analysis results available. Please update analysis.")
            return
            
        smc_results = st.session_state.analysis_results.get('smc', {})
        df = st.session_state.analysis_results['data']
        latest_price = df['close'].iloc[-1]
        
        # Order Blocks Section
        st.subheader("📦 Order Blocks")
        order_blocks = smc_results.get('order_blocks', [])
        
        if order_blocks:
            # Filter active order blocks
            active_obs = [ob for ob in order_blocks if ob.get('status') == 'active']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Active", len(active_obs))
            with col2:
                bullish_obs = len([ob for ob in active_obs if ob.get('type') == 'bullish'])
                st.metric("Bullish OBs", bullish_obs)
            with col3:
                bearish_obs = len([ob for ob in active_obs if ob.get('type') == 'bearish'])
                st.metric("Bearish OBs", bearish_obs)
            with col4:
                nearest_ob = min(active_obs, key=lambda x: abs((x['high'] + x['low'])/2 - latest_price)) if active_obs else None
                distance = abs((nearest_ob['high'] + nearest_ob['low'])/2 - latest_price) / latest_price * 100 if nearest_ob else 0
                st.metric("Nearest OB", f"{distance:.1f}%" if nearest_ob else "N/A")
                
            # Display active order blocks
            if active_obs:
                ob_df = pd.DataFrame(active_obs)
                ob_df['distance_pct'] = ob_df.apply(lambda x: abs((x['high'] + x['low'])/2 - latest_price) / latest_price * 100, axis=1)
                ob_df = ob_df.sort_values('distance_pct').head(5)
                
                for _, ob in ob_df.iterrows():
                    col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
                    with col1:
                        icon = "🟢" if ob['type'] == 'bullish' else "🔴"
                        st.markdown(f"{icon} **{ob['type'].upper()}**")
                    with col2:
                        st.markdown(f"Range: ${ob['low']:.2f} - ${ob['high']:.2f}")
                    with col3:
                        st.markdown(f"Strength: **{ob.get('strength', 'Medium')}**")
                    with col4:
                        st.markdown(f"{ob['distance_pct']:.1f}%")
                        
        # Fair Value Gaps Section
        st.subheader("🌊 Fair Value Gaps")
        fvgs = smc_results.get('fair_value_gaps', [])
        
        if fvgs:
            active_fvgs = [fvg for fvg in fvgs if fvg.get('status') == 'unfilled']
            recent_fvgs = [fvg for fvg in active_fvgs if pd.to_datetime(fvg['time']) > datetime.now() - timedelta(hours=1)]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Active FVGs", len(active_fvgs), f"{len(recent_fvgs)} new")
            with col2:
                avg_size = np.mean([fvg['size'] for fvg in active_fvgs]) if active_fvgs else 0
                st.metric("Avg Size", f"{avg_size:.1f} pips")
            with col3:
                filled_today = len([fvg for fvg in fvgs if fvg.get('status') == 'filled' and 
                                  pd.to_datetime(fvg.get('filled_time', '2000-01-01')) > datetime.now().replace(hour=0)])
                st.metric("Filled Today", filled_today)
            with col4:
                efficiency = (filled_today / len(fvgs) * 100) if fvgs else 0
                st.metric("Fill Rate", f"{efficiency:.0f}%")
                
            # Alert for new FVGs
            if recent_fvgs:
                latest_fvg = recent_fvgs[0]
                st.markdown(f"""
                <div class="alert-box">
                    <h4 style="margin: 0;">🚨 New FVG Detected!</h4>
                    <p style="margin: 0.5rem 0;">{latest_fvg['type'].capitalize()} FVG at ${latest_fvg['low']:.2f} - ${latest_fvg['high']:.2f}</p>
                    <p style="margin: 0; font-size: 0.9rem;">Size: {latest_fvg['size']:.1f} pips | Created: {pd.to_datetime(latest_fvg['time']).strftime('%H:%M')}</p>
                </div>
                """, unsafe_allow_html=True)
                
        # Liquidity Analysis
        st.subheader("💧 Liquidity Analysis")
        liquidity_zones = smc_results.get('liquidity_zones', [])
        
        if liquidity_zones:
            # Separate buy-side and sell-side
            bsl_zones = [lz for lz in liquidity_zones if lz['type'] == 'BSL' and lz['level'] > latest_price]
            ssl_zones = [lz for lz in liquidity_zones if lz['type'] == 'SSL' and lz['level'] < latest_price]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🎯 Buy-Side Liquidity (Above)**")
                if bsl_zones:
                    bsl_sorted = sorted(bsl_zones, key=lambda x: x['level'])[:3]
                    for lz in bsl_sorted:
                        distance = (lz['level'] - latest_price) / latest_price * 100
                        st.markdown(f"""
                        <div class="poi-card">
                            <div style="display: flex; justify-content: space-between;">
                                <span>Level: ${lz['level']:.2f}</span>
                                <span style="color: #00ff88;">+{distance:.2f}%</span>
                            </div>
                            <div style="font-size: 0.8rem; color: #888;">
                                Strength: {lz['strength']:.2f} | Index: {lz['index']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
            with col2:
                st.markdown("**🎯 Sell-Side Liquidity (Below)**")
                if ssl_zones:
                    ssl_sorted = sorted(ssl_zones, key=lambda x: x['level'], reverse=True)[:3]
                    for lz in ssl_sorted:
                        distance = (latest_price - lz['level']) / latest_price * 100
                        st.markdown(f"""
                        <div class="poi-card">
                            <div style="display: flex; justify-content: space-between;">
                                <span>Level: ${lz['level']:.2f}</span>
                                <span style="color: #ff3366;">-{distance:.2f}%</span>
                            </div>
                            <div style="font-size: 0.8rem; color: #888;">
                                Strength: {lz['strength']:.2f} | Index: {lz['index']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
    def render_wyckoff_analysis(self):
        """Render Wyckoff analysis tab"""
        if 'analysis_results' not in st.session_state or not st.session_state.analysis_results or 'data' not in st.session_state.analysis_results:
            st.warning("No analysis results available. Please update analysis.")
            return
            
        wyckoff = st.session_state.analysis_results.get('wyckoff', {})
        df = st.session_state.analysis_results['data']
        
        # Current Phase Display
        col1, col2 = st.columns([1, 2])
        
        with col1:
            phase = wyckoff.get('current_phase', 'Unknown')
            phase_colors = {
                'Accumulation': '#00ff88',
                'Markup': '#00cc66',
                'Distribution': '#ff3366',
                'Markdown': '#cc0033',
                'Unknown': '#666666'
            }
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1e3c72, #2a5298); 
                        padding: 2rem; border-radius: 15px; text-align: center;
                        border: 2px solid {phase_colors.get(phase, '#666')}>
                <h3 style="margin: 0; color: #888;">Current Phase</h3>
                <h1 style="margin: 0.5rem 0; color: {phase_colors.get(phase, '#666')};">{phase}</h1>
            </div>
            """, unsafe_allow_html=True)
            
            # Volume Analysis
            vol_analysis = wyckoff.get('volume_analysis', {})
            if vol_analysis:
                st.markdown("### 📊 Volume Characteristics")
                characteristics = []
                
                if vol_analysis.get('climactic_volume'):
                    characteristics.append("⚡ Climactic Volume Detected")
                if vol_analysis.get('no_demand'):
                    characteristics.append("🚫 No Demand Pattern")
                if vol_analysis.get('no_supply'):
                    characteristics.append("✅ No Supply Pattern")
                    
                for char in characteristics:
                    st.markdown(f"- {char}")
                    
        with col2:
            # Recent Wyckoff Events
            st.markdown("### 📌 Recent Wyckoff Events")
            events = wyckoff.get('events', [])
            
            if events:
                # Show last 5 events
                recent_events = events[-5:] if len(events) > 5 else events
                recent_events.reverse()  # Show newest first
                
                for event in recent_events:
                    event_time = pd.to_datetime(event.get('time', datetime.now()))
                    time_ago = (datetime.now() - event_time).total_seconds() / 60
                    
                    event_colors = {
                        'Spring': '#00ff88',
                        'Upthrust': '#ff3366',
                        'Sign of Strength': '#00cc66',
                        'Sign of Weakness': '#cc0033',
                        'Test': '#ffa500'
                    }
                    
                    st.markdown(f"""
                    <div class="poi-card" style="border-left: 4px solid {event_colors.get(event['event'], '#666')};">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <strong>{event['event']}</strong> at ${event.get('price', 0):.2f}
                            </div>
                            <div style="font-size: 0.8rem; color: #888;">
                                {int(time_ago)} min ago
                            </div>
                        </div>
                        <div style="font-size: 0.85rem; color: #aaa; margin-top: 0.3rem;">
                            {event.get('description', '')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No recent Wyckoff events detected")
                
        # Trading Ranges
        st.markdown("### 📈 Trading Ranges")
        trading_ranges = wyckoff.get('trading_ranges', pd.DataFrame())
        
        if not trading_ranges.empty:
            # Create range visualization
            fig = self.chart_builder.create_trading_range_chart(df, trading_ranges)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No significant trading ranges identified")
            
        # Composite Operator Analysis
        st.markdown("### 🎭 Composite Operator Footprint")
        co_analysis = wyckoff.get('composite_operator', {})
        
        if co_analysis:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                absorption = co_analysis.get('absorption_level', 0)
                st.metric("Absorption Level", f"{absorption:.1%}")
                
            with col2:
                distribution = co_analysis.get('distribution_intensity', 0)
                st.metric("Distribution Intensity", f"{distribution:.1%}")
                
            with col3:
                manipulation = co_analysis.get('manipulation_index', 0)
                st.metric("Manipulation Index", f"{manipulation:.2f}")
                
    def render_points_of_interest(self):
        """Render consolidated POIs"""
        if 'analysis_results' not in st.session_state or not st.session_state.analysis_results or 'data' not in st.session_state.analysis_results:
            st.warning("No analysis results available. Please update analysis.")
            return
            
        results = st.session_state.analysis_results
        df = results['data']
        latest_price = df['close'].iloc[-1]
        
        # Collect all POIs
        all_pois = []
        
        # Add Order Blocks
        if 'smc' in results:
            order_blocks = results['smc'].get('order_blocks', [])
            for ob in order_blocks:
                if ob.get('status') == 'active':
                    center = (ob['high'] + ob['low']) / 2
                    all_pois.append({
                        'type': 'Order Block',
                        'subtype': ob['type'],
                        'level': center,
                        'range': [ob['low'], ob['high']],
                        'distance': abs(center - latest_price) / latest_price * 100,
                        'strength': ob.get('strength', 'Medium'),
                        'time': ob.get('time')
                    })
                    
            # Add FVGs
            fvgs = results['smc'].get('fair_value_gaps', [])
            for fvg in fvgs:
                if fvg.get('status') == 'unfilled':
                    center = (fvg['high'] + fvg['low']) / 2
                    all_pois.append({
                        'type': 'Fair Value Gap',
                        'subtype': fvg['type'],
                        'level': center,
                        'range': [fvg['low'], fvg['high']],
                        'distance': abs(center - latest_price) / latest_price * 100,
                        'strength': 'High',
                        'time': fvg.get('time')
                    })
                    
            # Add Liquidity Zones
            liquidity = results['smc'].get('liquidity_zones', [])
            for lz in liquidity:
                all_pois.append({
                    'type': 'Liquidity Zone',
                    'subtype': 'buy-side' if lz['type'] == 'BSL' else 'sell-side',
                    'level': lz['level'],
                    'range': [lz['level'] * 0.999, lz['level'] * 1.001],
                    'distance': abs(lz['level'] - latest_price) / latest_price * 100,
                    'strength': f"{lz['strength']:.2f}",
                    'time': lz.get('time')
                })
                
        # Sort by distance
        all_pois.sort(key=lambda x: x['distance'])
        
        # Display filters
        col1, col2, col3 = st.columns(3)
        with col1:
            poi_types = st.multiselect("POI Types", 
                                     ["Order Block", "Fair Value Gap", "Liquidity Zone"],
                                     default=["Order Block", "Fair Value Gap", "Liquidity Zone"])
        with col2:
            max_distance = st.slider("Max Distance %", 0, 20, 10)
        with col3:
            sort_by = st.selectbox("Sort By", ["Distance", "Strength", "Time"])
            
        # Filter POIs
        filtered_pois = [poi for poi in all_pois 
                        if poi['type'] in poi_types and poi['distance'] <= max_distance]
                        
        # Sort based on selection
        if sort_by == "Strength":
            filtered_pois.sort(key=lambda x: x['strength'], reverse=True)
        elif sort_by == "Time":
            filtered_pois.sort(key=lambda x: x.get('time', datetime.min), reverse=True)
            
        # Display POIs
        st.markdown(f"### 🎯 Found {len(filtered_pois)} Points of Interest")
        
        if filtered_pois:
            # Above price POIs
            above_pois = [poi for poi in filtered_pois if poi['level'] > latest_price]
            below_pois = [poi for poi in filtered_pois if poi['level'] < latest_price]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📈 Above Current Price**")
                for poi in above_pois[:5]:
                    self.render_poi_card(poi, latest_price, 'above')
                    
            with col2:
                st.markdown("**📉 Below Current Price**")
                for poi in below_pois[:5]:
                    self.render_poi_card(poi, latest_price, 'below')
                    
    def render_poi_card(self, poi, current_price, position):
        """Render individual POI card"""
        icon_map = {
            'Order Block': '📦',
            'Fair Value Gap': '🌊',
            'Liquidity Zone': '💧'
        }
        
        color_map = {
            'bullish': '#00ff88',
            'bearish': '#ff3366',
            'buy-side': '#00cc66',
            'sell-side': '#cc0033'
        }
        
        icon = icon_map.get(poi['type'], '📍')
        color = color_map.get(poi['subtype'], '#ffa500')
        direction = '↑' if position == 'above' else '↓'
        
        st.markdown(f"""
        <div class="poi-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-size: 1.2rem;">{icon}</span>
                    <strong>{poi['type']}</strong>
                    <span style="color: {color}; font-size: 0.85rem;">({poi['subtype']})</span>
                </div>
                <div style="text-align: right;">
                    <div style="color: {color};">{direction} {poi['distance']:.2f}%</div>
                </div>
            </div>
            <div style="margin-top: 0.5rem; font-size: 0.9rem;">
                <div>Level: ${poi['level']:.2f}</div>
                <div style="color: #888;">Range: ${poi['range'][0]:.2f} - ${poi['range'][1]:.2f}</div>
                <div style="color: #888;">Strength: {poi['strength']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    def render_trade_setups(self):
        """Render potential trade setups"""
        if ('analysis_results' not in st.session_state
            or not isinstance(st.session_state.analysis_results, dict)
            or 'data' not in st.session_state.analysis_results
            or not isinstance(st.session_state.analysis_results['data'], pd.DataFrame)):
            st.warning("No analysis results available or data is invalid. Please update analysis.")
            return
            
        results = st.session_state.analysis_results
        df = results['data']
        latest_price = df['close'].iloc[-1]
        
        # Generate setups based on current conditions
        setups = self.generate_trade_setups(results, latest_price)
        
        st.markdown(f"### 🎯 {len(setups)} Potential Setups Identified")
        
        if setups:
            # Setup filters
            col1, col2 = st.columns(2)
            with col1:
                setup_types = st.multiselect("Setup Types", 
                                           list(set(s['type'] for s in setups)),
                                           default=list(set(s['type'] for s in setups)))
            with col2:
                min_rr = st.slider("Min Risk:Reward", 1.0, 5.0, 1.5, 0.5)
                
            # Filter setups
            filtered_setups = [s for s in setups 
                             if s['type'] in setup_types and s['risk_reward'] >= min_rr]
                             
            # Display setups
            for i, setup in enumerate(filtered_setups[:5]):
                self.render_trade_setup_card(setup, i+1)
                
        else:
            st.info("No high-probability setups detected at current levels")
            
    def generate_trade_setups(self, results, current_price):
        """Generate potential trade setups from analysis"""
        setups = []
        
        # Order Block Setups
        if 'smc' in results:
            order_blocks = results['smc'].get('order_blocks', [])
            for ob in order_blocks:
                if ob.get('status') == 'active':
                    # Bullish OB setup
                    if ob['type'] == 'bullish' and current_price <= ob['high'] * 1.02:
                        setups.append({
                            'type': 'Bullish OB Retest',
                            'direction': 'LONG',
                            'entry': ob['high'],
                            'stop': ob['low'] * 0.995,
                            'target1': ob['high'] * 1.015,
                            'target2': ob['high'] * 1.025,
                            'risk_reward': 2.0,
                            'confidence': 'High' if ob.get('strength') == 'High' else 'Medium',
                            'notes': f"Strong bullish order block at ${ob['high']:.2f}"
                        })
                        
            # FVG Setups
            fvgs = results['smc'].get('fair_value_gaps', [])
            for fvg in fvgs:
                if fvg.get('status') == 'unfilled':
                    # Bullish FVG fill setup
                    if fvg['type'] == 'bullish' and current_price >= fvg['low'] and current_price <= fvg['high']:
                        setups.append({
                            'type': 'Bullish FVG Fill',
                            'direction': 'LONG',
                            'entry': current_price,
                            'stop': fvg['low'] * 0.995,
                            'target1': fvg['high'] * 1.01,
                            'target2': fvg['high'] * 1.02,
                            'risk_reward': 1.5,
                            'confidence': 'Medium',
                            'notes': f"Price entering bullish FVG zone"
                        })
                        
        # Wyckoff Setups
        if 'wyckoff' in results:
            phase = results['wyckoff'].get('current_phase')
            events = results['wyckoff'].get('events', [])
            
            # Spring setup
            recent_spring = next((e for e in reversed(events) if e['event'] == 'Spring'), None)
            if recent_spring and phase == 'Accumulation':
                spring_time = pd.to_datetime(recent_spring.get('time', datetime.now()))
                # PATCH: robust handling for spring_time type
                delta = datetime.now() - spring_time if isinstance(spring_time, datetime) else timedelta(seconds=spring_time)
                if isinstance(delta, timedelta) and delta.total_seconds() < 3600:
                    setups.append({
                        'type': 'Wyckoff Spring',
                        'direction': 'LONG',
                        'entry': current_price,
                        'stop': recent_spring['price'] * 0.99,
                        'target1': current_price * 1.02,
                        'target2': current_price * 1.04,
                        'risk_reward': 3.0,
                        'confidence': 'High',
                        'notes': "Recent spring detected in accumulation phase"
                    })
                    
        return setups
        
    def render_trade_setup_card(self, setup, number):
        """Render individual trade setup card"""
        direction_color = '#00ff88' if setup['direction'] == 'LONG' else '#ff3366'
        confidence_colors = {'High': '#00ff88', 'Medium': '#ffa500', 'Low': '#ff3366'}
        
        # Calculate position metrics
        risk = abs(setup['entry'] - setup['stop'])
        reward1 = abs(setup['target1'] - setup['entry'])
        reward2 = abs(setup['target2'] - setup['entry'])
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1e3c72, #2a5298); 
                    padding: 1.5rem; border-radius: 15px; margin-bottom: 1rem;
                    border: 1px solid {direction_color};">
            <div style="display: flex; justify-content: space-between; align-items: start;">
                <div>
                    <h3 style="margin: 0; color: {direction_color};">
                        Setup #{number}: {setup['type']}
                    </h3>
                    <p style="margin: 0.5rem 0; color: #aaa;">{setup['notes']}</p>
                </div>
                <div style="text-align: right;">
                    <div style="background: {confidence_colors[setup['confidence']]}; 
                                color: white; padding: 0.3rem 0.8rem; 
                                border-radius: 5px; font-weight: bold;">
                        {setup['confidence']} Confidence
                    </div>
                    <div style="margin-top: 0.5rem; color: {direction_color}; font-size: 1.2rem;">
                        {setup['direction']}
                    </div>
                </div>
            </div>
            
            <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 1rem; margin-top: 1rem;">
                <div style="text-align: center;">
                    <div style="color: #888; font-size:
<div style="color: #888; font-size: 0.85rem;">Entry</div>
                    <div style="font-size: 1.2rem; font-weight: bold;">${setup['entry']:.2f}</div>
                </div>
                <div style="text-align: center;">
                    <div style="color: #888; font-size: 0.85rem;">Stop Loss</div>
                    <div style="font-size: 1.2rem; color: #ff3366;">${setup['stop']:.2f}</div>
                </div>
                <div style="text-align: center;">
                    <div style="color: #888; font-size: 0.85rem;">Target 1</div>
                    <div style="font-size: 1.2rem; color: #00ff88;">${setup['target1']:.2f}</div>
                </div>
                <div style="text-align: center;">
                    <div style="color: #888; font-size: 0.85rem;">Target 2</div>
                    <div style="font-size: 1.2rem; color: #00ff88;">${setup['target2']:.2f}</div>
                </div>
                <div style="text-align: center;">
                    <div style="color: #888; font-size: 0.85rem;">Risk:Reward</div>
                    <div style="font-size: 1.2rem; font-weight: bold;">1:{setup['risk_reward']:.1f}</div>
                </div>
            </div>
            
            <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.1);">
                <div style="display: flex; justify-content: space-between; font-size: 0.9rem;">
                    <div>Risk: ${risk:.2f} ({risk/setup['entry']*100:.2f}%)</div>
                    <div>Reward 1: ${reward1:.2f} ({reward1/setup['entry']*100:.2f}%)</div>
                    <div>Reward 2: ${reward2:.2f} ({reward2/setup['entry']*100:.2f}%)</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    def render_sidebar(self):
        """Render sidebar controls"""
        with st.sidebar:
            st.markdown("### 📊 Data Selection")
            
            # Pair selection
            pairs = {
                'Crypto': ['ETHUSD', 'BTCUSD', 'ETHUSDT', 'BTCUSDT'],
                'Forex': ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD'],
                'Indices': ['SPX', 'NDX', 'DJI']
            }
            
            category = st.selectbox("Category", list(pairs.keys()))
            selected_pair = st.selectbox("Pair", pairs[category], 
                                       index=pairs[category].index(st.session_state.selected_pair) 
                                       if st.session_state.selected_pair in pairs[category] else 0)
            
            # Timeframe selection
            available_timeframes = list(st.session_state.all_timeframes.keys()) if st.session_state.all_timeframes else ['M1']
            selected_timeframe = st.selectbox("Timeframe", available_timeframes,
                                            index=available_timeframes.index(st.session_state.selected_timeframe)
                                            if st.session_state.selected_timeframe in available_timeframes else 0)
            
            # Update button
            if st.button("🔄 Update Analysis", type="primary", use_container_width=True):
                # Update session state
                st.session_state.selected_pair = selected_pair
                st.session_state.selected_timeframe = selected_timeframe
                
                # Load data
                with st.spinner("Loading data..."):
                    if self.load_and_process_data(selected_pair):
                        # Run analysis
                        with st.spinner("Running comprehensive analysis..."):
                            if self.run_comprehensive_analysis(selected_timeframe):
                                st.success("✅ Analysis complete!")
                                st.rerun()
                            else:
                                st.error("Analysis failed")
                    else:
                        st.error("Failed to load data")
                        
            # Auto-refresh option
            st.divider()
            auto_refresh = st.checkbox("Auto-refresh", value=False)
            if auto_refresh:
                refresh_interval = st.slider("Refresh interval (seconds)", 10, 300, 60)
                st.info(f"Auto-refresh every {refresh_interval}s")
                
            # Analysis Settings
            st.divider()
            st.markdown("### ⚙️ Analysis Settings")
            
            with st.expander("SMC Settings"):
                ob_lookback = st.slider("Order Block Lookback", 20, 100, 50)
                fvg_min_size = st.slider("Min FVG Size (pips)", 1, 20, 5)
                liquidity_threshold = st.slider("Liquidity Threshold", 0.001, 0.01, 0.002, 0.001)
                
            with st.expander("Wyckoff Settings"):
                phase_lookback = st.slider("Phase Lookback", 50, 200, 100)
                volume_ma = st.slider("Volume MA Period", 10, 50, 20)
                
            # Export Options
            st.divider()
            st.markdown("### 💾 Export Options")
            
            if st.button("📊 Export Analysis Report", use_container_width=True):
                self.export_analysis_report()
                
            if st.button("📸 Save Chart Image", use_container_width=True):
                st.info("Chart saved to downloads folder")
                
    def export_analysis_report(self):
        """Export comprehensive analysis report"""
        if 'analysis_results' not in st.session_state:
            st.error("No analysis results to export")
            return
            
        # Create report content
        report = f"""
# ncOS Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Pair: {st.session_state.selected_pair}
Timeframe: {st.session_state.selected_timeframe}

## Summary Metrics
{'-' * 40}
"""
        
        summary = st.session_state.analysis_results.get('summary', {})
        for key, value in summary.items():
            report += f"{key.replace('_', ' ').title()}: {value}\n"
            
        # Add SMC analysis
        report += f"\n## Smart Money Concepts Analysis\n{'-' * 40}\n"
        smc = st.session_state.analysis_results.get('smc', {})
        
        # Order blocks
        order_blocks = smc.get('order_blocks', [])
        active_obs = [ob for ob in order_blocks if ob.get('status') == 'active']
        report += f"Active Order Blocks: {len(active_obs)}\n"
        
        # FVGs
        fvgs = smc.get('fair_value_gaps', [])
        active_fvgs = [fvg for fvg in fvgs if fvg.get('status') == 'unfilled']
        report += f"Unfilled Fair Value Gaps: {len(active_fvgs)}\n"
        
        # Wyckoff analysis
        report += f"\n## Wyckoff Analysis\n{'-' * 40}\n"
        wyckoff = st.session_state.analysis_results.get('wyckoff', {})
        report += f"Current Phase: {wyckoff.get('current_phase', 'Unknown')}\n"
        
        events = wyckoff.get('events', [])
        report += f"Recent Events: {len(events)}\n"
        
        # Save report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"ncOS_analysis_report_{st.session_state.selected_pair}_{timestamp}.txt"
        
        st.download_button(
            label="📥 Download Report",
            data=report,
            file_name=filename,
            mime="text/plain"
        )
        
    def run(self):
        """Main dashboard execution"""
        # Initial data load if needed
        if st.session_state.data is None:
            with st.spinner("Loading initial data..."):
                self.load_and_process_data(st.session_state.selected_pair)
                self.run_comprehensive_analysis(st.session_state.selected_timeframe)
                
        # Render sidebar
        self.render_sidebar()
        
        # Render header section
        self.render_header_section()
        
        # Main content tabs
        tabs = st.tabs([
            "📊 Advanced Price Action",
            "🏦 Smart Money Concepts", 
            "🎭 Wyckoff Analysis",
            "⚡ Points of Interest",
            "🎯 Trade Setups"
        ])
        
        with tabs[0]:
            self.render_advanced_price_action()
            
        with tabs[1]:
            self.render_smart_money_concepts()
            
        with tabs[2]:
            self.render_wyckoff_analysis()
            
        with tabs[3]:
            self.render_points_of_interest()
            
        with tabs[4]:
            self.render_trade_setups()
            
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; font-size: 0.8rem;">
            ncOS Dashboard v1.0 | Neural Core OS - Advanced Market Analysis
        </div>
        """, unsafe_allow_html=True)

# Main execution
if __name__ == "__main__":
    dashboard = NCOSDashboard()
    dashboard.run()