
import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime
import glob
import os

# Configure dark theme
st.set_page_config(
    page_title="🚀 ZANFLOW v12 Ultimate Trading Analysis Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom dark theme CSS
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stApp {
        background: linear-gradient(135deg, #0E1117 0%, #1E2332 100%);
    }
    .css-1d391kg {
        background-color: #1E2332;
    }
    .metric-card {
        background: linear-gradient(135deg, #262730 0%, #2F3349 100%);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #404552;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        margin: 10px 0;
    }
    .manipulation-high {
        background: linear-gradient(135deg, #FF6B6B 0%, #EE5A24 100%);
        color: white;
    }
    .manipulation-medium {
        background: linear-gradient(135deg, #F39C12 0%, #E67E22 100%);
        color: white;
    }
    .manipulation-low {
        background: linear-gradient(135deg, #2ECC71 0%, #27AE60 100%);
        color: white;
    }
    .smc-bullish {
        background: linear-gradient(135deg, #00D2FF 0%, #3A7BD5 100%);
        color: white;
    }
    .smc-bearish {
        background: linear-gradient(135deg, #FF416C 0%, #FF4B2B 100%);
        color: white;
    }
    .wyckoff-accumulation {
        background: linear-gradient(135deg, #4ECDC4 0%, #44A08D 100%);
        color: white;
    }
    .wyckoff-distribution {
        background: linear-gradient(135deg, #FFA726 0%, #FF7043 100%);
        color: white;
    }
    .header-container {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    .data-source {
        background: rgba(64, 69, 82, 0.3);
        padding: 10px;
        border-radius: 8px;
        margin: 5px 0;
        font-size: 0.8em;
    }
</style>
""", unsafe_allow_html=True)

class ZanFlowDataScanner:
    def __init__(self):
        self.data_sources = {}
        self.loaded_data = {}

    def scan_all_directories(self):
        """Comprehensive scan of all data directories"""
        scan_results = {
            'json_files': [],
            'csv_files': [],
            'data_dirs': []
        }

        # Scan current directory and all subdirectories
        for root, dirs, files in os.walk('.'):
            # Track data directories
            for d in dirs:
                if 'data' in d.lower():
                    scan_results['data_dirs'].append(os.path.join(root, d))

            # Track data files
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith('.json'):
                    scan_results['json_files'].append(file_path)
                elif file.endswith('.csv'):
                    scan_results['csv_files'].append(file_path)

        return scan_results

    def load_analysis_data(self, scan_results):
        """Load all available analysis data"""
        loaded_data = {
            'tick_analysis': None,
            'microstructure': {},
            'comprehensive_analysis': {},
            'market_data': {}
        }

        # Try to load main analysis files
        for json_file in scan_results['json_files']:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)

                # Categorize by filename patterns
                filename = os.path.basename(json_file).lower()

                if 'tick_microstructure' in filename:
                    loaded_data['tick_analysis'] = data
                elif 'analysis_report' in filename:
                    pair = filename.split('_')[0].upper()
                    loaded_data['comprehensive_analysis'][pair] = data
                elif 'summary' in filename:
                    timeframe = self._extract_timeframe(filename)
                    loaded_data['microstructure'][timeframe] = data

            except Exception as e:
                st.sidebar.warning(f"⚠️ Could not load {json_file}: {str(e)[:50]}...")

        # Load CSV data for additional analysis
        for csv_file in scan_results['csv_files']:
            try:
                if os.path.getsize(csv_file) < 50*1024*1024:  # Only load files < 50MB
                    filename = os.path.basename(csv_file).lower()
                    if 'xauusd' in filename and 'processed' in filename:
                        timeframe = self._extract_timeframe(filename)
                        # Just load a sample for analysis
                        df = pd.read_csv(csv_file, nrows=1000)
                        loaded_data['market_data'][timeframe] = {
                            'shape': df.shape,
                            'columns': list(df.columns),
                            'latest_price': df['close'].iloc[-1] if 'close' in df.columns else None,
                            'file_path': csv_file
                        }
            except Exception as e:
                continue

        return loaded_data

    def _extract_timeframe(self, filename):
        """Extract timeframe from filename"""
        timeframes = ['1T', '5T', '15T', '30T', '1H', '4H', '1D', 'tick', '1min', '5min', '15min', '30min']
        for tf in timeframes:
            if tf.lower() in filename.lower():
                return tf
        return 'unknown'

class ZanFlowAnalyzer:
    def __init__(self, loaded_data):
        self.loaded_data = loaded_data

    def create_ispts_analysis(self):
        """Create ISPTS analysis from loaded data"""

        # Use tick analysis if available
        tick_data = self.loaded_data.get('tick_analysis')

        if tick_data:
            return self._create_real_ispts_analysis(tick_data)
        else:
            return self._create_enhanced_demo_analysis()

    def _create_real_ispts_analysis(self, data):
        """Create analysis from real data"""
        manipulation_score = 0
        sweep_events = 0
        bias = 'NEUTRAL'
        wyckoff_phase = 'Unknown'

        # Extract manipulation data
        if 'manipulation_detection' in data:
            manipulation_score = data['manipulation_detection'].get('manipulation_score', 0)
            sweep_events = data['manipulation_detection'].get('liquidity_sweeps', 0)

        # Extract SMC data
        if 'smc_analysis' in data:
            bias = data['smc_analysis'].get('bias', 'NEUTRAL')

        # Extract Wyckoff data
        if 'wyckoff_analysis' in data:
            wyckoff_phase = data['wyckoff_analysis'].get('dominant_phase', 'Unknown')

        return {
            'context_analyzer': {
                'htf_bias': bias,
                'range_status': 'ACTIVE',
                'manipulation_score': manipulation_score,
                'data_source': 'Real JSON Analysis'
            },
            'liquidity_engine': {
                'sweep_events': sweep_events,
                'inducements_high': data.get('inducement_analysis', {}).get('high_inducements', 0),
                'inducements_low': data.get('inducement_analysis', {}).get('low_inducements', 0),
                'idm_detected': data.get('inducement_analysis', {}).get('inducement_rate', 0) > 5,
                'data_source': 'Real Tick Analysis'
            },
            'structure_validator': {
                'choch_confirmed': bias != 'NEUTRAL',
                'bos_strength': 'HIGH' if manipulation_score > 40 else 'MEDIUM',
                'data_source': 'SMC Analysis'
            },
            'fvg_locator': {
                'bullish_fvgs': data.get('smc_analysis', {}).get('bullish_fvgs', 0),
                'bearish_fvgs': data.get('smc_analysis', {}).get('bearish_fvgs', 0),
                'active_pois': data.get('smc_analysis', {}).get('bullish_fvgs', 0) + data.get('smc_analysis', {}).get('bearish_fvgs', 0),
                'data_source': 'FVG Detection'
            },
            'risk_manager': {
                'manipulation_score': manipulation_score,
                'volatility_level': self._assess_volatility(data),
                'recommended_risk': 'REDUCED' if manipulation_score > 40 else 'NORMAL',
                'data_source': 'Risk Calculation'
            },
            'confluence_stacker': {
                'wyckoff_phase': wyckoff_phase,
                'session_alignment': self._get_session_status(),
                'mtf_confluence': self._calculate_mtf_score(data),
                'data_source': 'Multi-timeframe Analysis'
            }
        }

    def _create_enhanced_demo_analysis(self):
        """Enhanced demo with market data if available"""
        market_data = self.loaded_data.get('market_data', {})

        # Use real market data statistics if available
        latest_price = 2335.5
        volatility = 'MEDIUM'

        if market_data:
            for tf, data in market_data.items():
                if data.get('latest_price'):
                    latest_price = data['latest_price']
                    break

        return {
            'context_analyzer': {
                'htf_bias': 'BULLISH',
                'range_status': 'ACTIVE',
                'manipulation_score': 43.2,
                'current_price': latest_price,
                'data_source': f'Market Data ({len(market_data)} timeframes loaded)'
            },
            'liquidity_engine': {
                'sweep_events': 23,
                'inducements_high': 6,
                'inducements_low': 14,
                'idm_detected': True,
                'data_source': 'Pattern Recognition'
            },
            'structure_validator': {
                'choch_confirmed': True,
                'bos_strength': 'HIGH',
                'data_source': 'Structure Analysis'
            },
            'fvg_locator': {
                'bullish_fvgs': 3,
                'bearish_fvgs': 2,
                'active_pois': 5,
                'data_source': 'Gap Detection'
            },
            'risk_manager': {
                'manipulation_score': 43.2,
                'volatility_level': volatility,
                'recommended_risk': 'REDUCED',
                'data_source': 'Risk Assessment'
            },
            'confluence_stacker': {
                'wyckoff_phase': 'Accumulation',
                'session_alignment': self._get_session_status(),
                'mtf_confluence': 0.75,
                'data_source': 'Confluence Engine'
            }
        }

    def _assess_volatility(self, data):
        """Assess volatility from data"""
        if 'microstructure' in data:
            spread_vol = data['microstructure'].get('spread_volatility', 0)
            if spread_vol > 3:
                return 'HIGH'
            elif spread_vol > 1.5:
                return 'MEDIUM'
        return 'LOW'

    def _get_session_status(self):
        """Get current session"""
        current_hour = datetime.now().hour
        if 0 <= current_hour < 9:
            return 'ASIAN_SESSION'
        elif 9 <= current_hour < 17:
            return 'LONDON_SESSION'
        else:
            return 'NY_SESSION'

    def _calculate_mtf_score(self, data):
        """Calculate multi-timeframe confluence"""
        score = 0.5  # Base score

        if data.get('smc_analysis', {}).get('bias') == 'BULLISH':
            score += 0.2
        if data.get('manipulation_detection', {}).get('manipulation_score', 0) > 40:
            score += 0.15
        if data.get('wyckoff_analysis', {}).get('dominant_phase') == 'Accumulation':
            score += 0.15

        return min(score, 1.0)

def main():
    # Header
    st.markdown("""
    <div class="header-container">
        <h1>🚀 ZANFLOW v12 Ultimate Trading Analysis Platform</h1>
        <p><strong>Real-Time Data Scanner • ISPTS Framework • Multi-Directory Support</strong></p>
        <p><em>ENHANCED: Automatic ./data Directory Scanning</em></p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize scanner and analyzer
    scanner = ZanFlowDataScanner()

    # Scan all directories
    with st.spinner("🔍 Scanning directories for data files..."):
        scan_results = scanner.scan_all_directories()
        loaded_data = scanner.load_analysis_data(scan_results)

    # Display data sources in sidebar
    st.sidebar.markdown("## 📊 Data Sources")

    if scan_results['data_dirs']:
        st.sidebar.markdown("### 📁 Data Directories")
        for d in scan_results['data_dirs']:
            st.sidebar.markdown(f"- `{d}`")

    st.sidebar.markdown(f"### 📄 Files Found")
    st.sidebar.markdown(f"- **JSON Files**: {len(scan_results['json_files'])}")
    st.sidebar.markdown(f"- **CSV Files**: {len(scan_results['csv_files'])}")

    if loaded_data['tick_analysis']:
        st.sidebar.success("✅ Real tick analysis loaded")
    if loaded_data['comprehensive_analysis']:
        st.sidebar.success(f"✅ {len(loaded_data['comprehensive_analysis'])} pairs analyzed")
    if loaded_data['market_data']:
        st.sidebar.success(f"✅ {len(loaded_data['market_data'])} timeframes loaded")

    # Initialize analyzer with loaded data
    analyzer = ZanFlowAnalyzer(loaded_data)
    ispts_data = analyzer.create_ispts_analysis()

    # Main Dashboard Layout
    col1, col2, col3, col4 = st.columns(4)

    # Context Analyzer
    with col1:
        context = ispts_data['context_analyzer']
        bias = context['htf_bias']
        bias_class = 'smc-bullish' if bias == 'BULLISH' else 'smc-bearish'

        st.markdown(f"""
        <div class="metric-card {bias_class}">
            <h3>🧱 Context Analyzer</h3>
            <h2>{bias}</h2>
            <p>HTF Directional Bias</p>
            <hr>
            <p><strong>Manipulation:</strong> {context.get('manipulation_score', 'N/A')}%</p>
            <div class="data-source">📊 {context['data_source']}</div>
        </div>
        """, unsafe_allow_html=True)

    # Liquidity Engine
    with col2:
        liquidity = ispts_data['liquidity_engine']
        sweep_events = liquidity['sweep_events']
        idm_status = "DETECTED" if liquidity['idm_detected'] else "CLEAR"

        st.markdown(f"""
        <div class="metric-card manipulation-high">
            <h3>💧 Liquidity Engine</h3>
            <h2>{sweep_events}</h2>
            <p>Liquidity Sweeps</p>
            <hr>
            <p><strong>IDM Status:</strong> {idm_status}</p>
            <p><strong>H/L Traps:</strong> {liquidity['inducements_high']}/{liquidity['inducements_low']}</p>
            <div class="data-source">📊 {liquidity['data_source']}</div>
        </div>
        """, unsafe_allow_html=True)

    # FVG Locator
    with col3:
        fvg = ispts_data['fvg_locator']
        total_fvgs = fvg['active_pois']

        st.markdown(f"""
        <div class="metric-card smc-bullish">
            <h3>🧠 FVG Locator</h3>
            <h2>{total_fvgs}</h2>
            <p>Active POIs</p>
            <hr>
            <p><strong>Bullish FVGs:</strong> {fvg['bullish_fvgs']}</p>
            <p><strong>Bearish FVGs:</strong> {fvg['bearish_fvgs']}</p>
            <div class="data-source">📊 {fvg['data_source']}</div>
        </div>
        """, unsafe_allow_html=True)

    # Risk Manager
    with col4:
        risk = ispts_data['risk_manager']
        manipulation_score = risk['manipulation_score']
        risk_level = risk['recommended_risk']
        risk_class = 'manipulation-high' if manipulation_score > 40 else 'manipulation-medium'

        st.markdown(f"""
        <div class="metric-card {risk_class}">
            <h3>⚠️ Risk Manager</h3>
            <h2>{manipulation_score:.1f}%</h2>
            <p>Manipulation Score</p>
            <hr>
            <p><strong>Risk Mode:</strong> {risk_level}</p>
            <p><strong>Volatility:</strong> {risk['volatility_level']}</p>
            <div class="data-source">📊 {risk['data_source']}</div>
        </div>
        """, unsafe_allow_html=True)

    # Structure Validator & Confluence Stacker
    col5, col6 = st.columns(2)

    with col5:
        structure = ispts_data['structure_validator']
        choch_status = "CONFIRMED" if structure['choch_confirmed'] else "PENDING"

        st.markdown(f"""
        <div class="metric-card smc-bullish">
            <h3>🔀 Structure Validator</h3>
            <h2>{choch_status}</h2>
            <p>CHoCH Status</p>
            <hr>
            <p><strong>BoS Strength:</strong> {structure['bos_strength']}</p>
            <div class="data-source">📊 {structure['data_source']}</div>
        </div>
        """, unsafe_allow_html=True)

    with col6:
        confluence = ispts_data['confluence_stacker']
        wyckoff_phase = confluence['wyckoff_phase']
        confluence_score = int(confluence['mtf_confluence'] * 100)

        phase_class = 'wyckoff-accumulation' if 'Accumulation' in wyckoff_phase else 'wyckoff-distribution'

        st.markdown(f"""
        <div class="metric-card {phase_class}">
            <h3>📊 Confluence Stacker</h3>
            <h2>{confluence_score}%</h2>
            <p>MTF Confluence</p>
            <hr>
            <p><strong>Phase:</strong> {wyckoff_phase}</p>
            <p><strong>Session:</strong> {confluence['session_alignment'].replace('_', ' ')}</p>
            <div class="data-source">📊 {confluence['data_source']}</div>
        </div>
        """, unsafe_allow_html=True)

    # Data Summary
    st.markdown("---")
    st.markdown("## 📈 Data Summary")

    col7, col8, col9 = st.columns(3)

    with col7:
        st.markdown("### 📊 Loaded Analysis")
        if loaded_data['tick_analysis']:
            st.success("✅ Tick Microstructure Analysis")
        if loaded_data['comprehensive_analysis']:
            st.success(f"✅ {len(loaded_data['comprehensive_analysis'])} Currency Pairs")
        if loaded_data['microstructure']:
            st.success(f"✅ {len(loaded_data['microstructure'])} Timeframes")

    with col8:
        st.markdown("### 📁 Directory Scan")
        st.info(f"📂 {len(scan_results['data_dirs'])} data directories")
        st.info(f"📄 {len(scan_results['json_files'])} JSON files")
        st.info(f"📊 {len(scan_results['csv_files'])} CSV files")

    with col9:
        st.markdown("### 🎯 Current Analysis")
        analysis_source = "Real Data" if loaded_data['tick_analysis'] else "Enhanced Demo"
        st.success(f"🧠 ISPTS Framework: {analysis_source}")
        st.info(f"⏰ Last Updated: {datetime.now().strftime('%H:%M:%S')}")

    # File Explorer
    with st.expander("🔍 File Explorer", expanded=False):
        if scan_results['json_files']:
            st.markdown("### JSON Analysis Files")
            for f in scan_results['json_files']:
                file_size = os.path.getsize(f) / 1024  # KB
                st.markdown(f"- `{f}` ({file_size:.1f} KB)")

        if scan_results['csv_files']:
            st.markdown("### CSV Data Files")
            for f in scan_results['csv_files'][:20]:  # Show first 20
                try:
                    file_size = os.path.getsize(f) / (1024*1024)  # MB
                    st.markdown(f"- `{f}` ({file_size:.1f} MB)")
                except:
                    st.markdown(f"- `{f}` (size unknown)")

if __name__ == "__main__":
    main()
