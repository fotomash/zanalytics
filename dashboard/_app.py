import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from utils.data_processor import DataProcessor
from utils.timeframe_converter import TimeframeConverter
from utils.technical_analysis import TechnicalAnalysis
from utils.smc_analyzer import SMCAnalyzer
from utils.wyckoff_analyzer import WyckoffAnalyzer
from utils.volume_profile import VolumeProfileAnalyzer
from components.chart_builder import ChartBuilder
from components.analysis_panel import AnalysisPanel

# Page configuration
st.set_page_config(
    page_title="Zanflow Analytics Dashboard",
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
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    div[data-testid="metric-container"] {
        background-color: rgba(28, 131, 225, 0.1);
        border: 1px solid rgba(28, 131, 225, 0.2);
        padding: 5px 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'timeframes' not in st.session_state:
    st.session_state.timeframes = {}
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}

# Header
st.title("🏛️ Zanflow Multi-Timeframe Analysis Dashboard")
st.markdown("### Institutional-Grade Market Analysis with Smart Money Concepts & Wyckoff Methodology")

# Sidebar
with st.sidebar:
    st.header("📁 Data Selection")

    # File selection
    data_files = [f for f in os.listdir('./data') if f.endswith(('.csv', '.txt'))]
    if not data_files:
        st.warning("No data files found in ./data folder")
        st.info("Please place your tab-separated CSV files in the ./data folder")
    else:
        selected_file = st.selectbox("Select Data File", data_files)

        # Load data button
        if st.button("🔄 Load Data", type="primary"):
            with st.spinner("Loading data..."):
                try:
                    # Load the data
                    file_path = os.path.join('./data', selected_file)
                    data_processor = DataProcessor()
                    st.session_state.data = data_processor.load_data(file_path)

                    # Generate timeframes
                    converter = TimeframeConverter()
                    st.session_state.timeframes = converter.generate_all_timeframes(st.session_state.data)

                    st.success(f"✅ Loaded {len(st.session_state.data)} bars")
                except Exception as e:
                    st.error(f"Error loading data: {str(e)}")

    if st.session_state.data is not None:
        st.divider()
        st.header("⚙️ Analysis Settings")

        # Timeframe selection
        available_timeframes = list(st.session_state.timeframes.keys())
        selected_timeframe = st.selectbox(
            "Select Timeframe",
            available_timeframes,
            index=available_timeframes.index('H1') if 'H1' in available_timeframes else 0
        )

        # Analysis options
        st.subheader("Analysis Modules")
        show_smc = st.checkbox("Smart Money Concepts", value=True)
        show_wyckoff = st.checkbox("Wyckoff Analysis", value=True)
        show_volume_profile = st.checkbox("Volume Profile", value=True)
        show_indicators = st.checkbox("Technical Indicators", value=True)

        # Visualization options
        st.subheader("Visualization Options")
        chart_type = st.selectbox("Chart Type", ["Candlestick", "Heikin Ashi", "Line"])
        show_volume = st.checkbox("Show Volume", value=True)
        show_orderflow = st.checkbox("Show Order Flow", value=True)

        # Run analysis button
        if st.button("🔍 Run Analysis", type="primary"):
            with st.spinner("Running comprehensive analysis..."):
                try:
                    # Get selected timeframe data
                    tf_data = st.session_state.timeframes[selected_timeframe]

                    # Initialize analyzers
                    smc = SMCAnalyzer() if show_smc else None
                    wyckoff = WyckoffAnalyzer() if show_wyckoff else None
                    volume_profile = VolumeProfileAnalyzer() if show_volume_profile else None
                    ta = TechnicalAnalysis() if show_indicators else None

                    # Run analyses
                    results = {}
                    if smc:
                        results['smc'] = smc.analyze(tf_data)
                    if wyckoff:
                        results['wyckoff'] = wyckoff.analyze(tf_data)
                    if volume_profile:
                        results['volume_profile'] = volume_profile.analyze(tf_data)
                    if ta:
                        results['indicators'] = ta.calculate_all(tf_data)

                    st.session_state.analysis_results = results
                    st.success("✅ Analysis complete!")
                except Exception as e:
                    st.error(f"Analysis error: {str(e)}")

# Main content area
if st.session_state.data is not None and st.session_state.timeframes:
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Chart Analysis", "🔄 Multi-Timeframe", "📈 Market Structure", "📋 Reports"])

    with tab1:
        # Main chart analysis
        if selected_timeframe in st.session_state.timeframes:
            tf_data = st.session_state.timeframes[selected_timeframe]

            # Metrics row
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                current_price = tf_data['close'].iloc[-1]
                price_change = ((current_price - tf_data['close'].iloc[-2]) / tf_data['close'].iloc[-2]) * 100
                st.metric("Current Price", f"${current_price:.5f}", f"{price_change:+.2f}%")
            with col2:
                if 'volume' in tf_data.columns:
                    st.metric("24h Volume", f"{tf_data['volume'].tail(24).sum():,.0f}")
                else:
                    st.metric("24h Volume", "N/A")
            with col3:
                st.metric("Volatility", f"{tf_data['close'].pct_change().std() * 100:.2f}%")
            with col4:
                high_24h = tf_data['high'].tail(24).max()
                st.metric("24h High", f"${high_24h:.5f}")
            with col5:
                low_24h = tf_data['low'].tail(24).min()
                st.metric("24h Low", f"${low_24h:.5f}")
            
            # Main chart
            chart_builder = ChartBuilder()
            fig = chart_builder.create_main_chart(
                tf_data, 
                selected_timeframe,
                chart_type=chart_type,
                show_volume=show_volume,
                analysis_results=st.session_state.analysis_results
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Analysis panel
            if st.session_state.analysis_results:
                analysis_panel = AnalysisPanel()
                analysis_panel.display_results(st.session_state.analysis_results)
    
    with tab2:
        # Multi-timeframe analysis
        st.header("Multi-Timeframe Analysis")
        
        # Select timeframes for comparison
        mtf_selection = st.multiselect(
            "Select Timeframes for Comparison",
            available_timeframes,
            default=['M15', 'H1', 'H4', 'D1'] if all(tf in available_timeframes for tf in ['M15', 'H1', 'H4', 'D1']) else available_timeframes[:4]
        )
        
        if mtf_selection:
            # Create multi-timeframe chart
            chart_builder = ChartBuilder()
            mtf_fig = chart_builder.create_mtf_chart(st.session_state.timeframes, mtf_selection)
            st.plotly_chart(mtf_fig, use_container_width=True)
            
            # MTF Analysis Summary
            st.subheader("Multi-Timeframe Confluence Analysis")
            mtf_analyzer = TechnicalAnalysis()
            confluence_results = mtf_analyzer.mtf_confluence_analysis(st.session_state.timeframes, mtf_selection)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Trend Alignment")
                for tf, trend in confluence_results['trends'].items():
                    emoji = "🟢" if trend == "Bullish" else "🔴" if trend == "Bearish" else "🟡"
                    st.write(f"{emoji} **{tf}**: {trend}")
            
            with col2:
                st.markdown("### Key Levels")
                for level_type, levels in confluence_results['key_levels'].items():
                    st.write(f"**{level_type}**:")
                    for level in levels[:5]:  # Show top 5 levels
                        st.write(f"  - ${level:.5f}")
    
    with tab3:
        # Market Structure Analysis
        st.header("Market Structure Analysis")
        
        if selected_timeframe in st.session_state.timeframes and st.session_state.analysis_results:
            tf_data = st.session_state.timeframes[selected_timeframe]
            
            # Structure visualization
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Smart Money Concepts")
                if 'smc' in st.session_state.analysis_results:
                    smc_results = st.session_state.analysis_results['smc']
                    
                    # Display liquidity zones
                    st.markdown("#### Liquidity Zones")
                    for zone in smc_results.get('liquidity_zones', [])[:5]:
                        zone_type = "🔴 Sell-side" if zone['type'] == 'SSL' else "🟢 Buy-side"
                        st.write(f"{zone_type}: ${zone['level']:.5f} (Strength: {zone['strength']:.2f})")
                    
                    # Display order blocks
                    st.markdown("#### Order Blocks")
                    for ob in smc_results.get('order_blocks', [])[:5]:
                        ob_type = "🟢 Bullish" if ob['type'] == 'bullish' else "🔴 Bearish"
                        st.write(f"{ob_type}: ${ob['start']:.5f} - ${ob['end']:.5f}")
            
            with col2:
                st.subheader("Wyckoff Analysis")
                if 'wyckoff' in st.session_state.analysis_results:
                    wyckoff_results = st.session_state.analysis_results['wyckoff']
                    
                    # Current phase
                    st.markdown("#### Current Phase")
                    phase = wyckoff_results.get('current_phase', 'Unknown')
                    phase_emoji = {
                        'Accumulation': '📈',
                        'Markup': '🚀',
                        'Distribution': '📉',
                        'Markdown': '💥'
                    }.get(phase, '❓')
                    st.write(f"{phase_emoji} **{phase}**")
                    
                    # Key events
                    st.markdown("#### Recent Events")
                    for event in wyckoff_results.get('events', [])[:5]:
                        st.write(f"• {event['type']} at ${event['price']:.5f}")
    
    with tab4:
        # Reports and Export
        st.header("Analysis Reports")
        
        # Generate report
        if st.button("📄 Generate Full Report"):
            with st.spinner("Generating comprehensive report..."):
                # Create report content
                report = f"""
# Zanflow Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Market Overview
- **Instrument**: {selected_file}
- **Timeframe**: {selected_timeframe}
- **Current Price**: ${tf_data['close'].iloc[-1]:.5f}
- **24h Change**: {((tf_data['close'].iloc[-1] - tf_data['close'].iloc[-24]) / tf_data['close'].iloc[-24] * 100):.2f}%

## Analysis Summary
"""
                
                # Add analysis results to report
                if st.session_state.analysis_results:
                    for module, results in st.session_state.analysis_results.items():
                        report += f"\\n### {module.upper()} Analysis\\n"
                        report += str(results)[:500] + "...\\n"
                
                # Display report
                st.text_area("Report Preview", report, height=400)
                
                # Download button
                st.download_button(
                    label="📥 Download Report",
                    data=report,
                    file_name=f"zanflow_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )

else:
    # Welcome screen
    st.info("👈 Please select and load a data file from the sidebar to begin analysis")
    
    # Instructions
    with st.expander("📖 How to Use This Dashboard"):
        st.markdown('''
        1. **Load Data**: Place your tab-separated CSV files in the `./data` folder and select from the sidebar
        2. **Configure Analysis**: Choose which analysis modules to run and visualization options
        3. **Run Analysis**: Click the "Run Analysis" button to process the data
        4. **Explore Results**: Navigate through the tabs to view different aspects of the analysis
        5. **Export Reports**: Generate and download comprehensive analysis reports
        
        ### Data Format
        Your CSV file s
        should be tab-separated with the following columns:
        - Date/Time
        - Open
        - High
        - Low
        - Close
        - Volume
        
        ### Available Analysis Modules
        - **Smart Money Concepts**: Liquidity zones, order blocks, fair value gaps
        - **Wyckoff Analysis**: Market phases, accumulation/distribution patterns
        - **Volume Profile**: Point of control, value areas, volume nodes
        - **Technical Indicators**: Moving averages, RSI, MACD, Bollinger Bands
''')
