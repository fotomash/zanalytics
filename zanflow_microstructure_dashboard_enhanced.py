
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import glob
import os
from datetime import datetime
import re
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="ZanFlow Advanced Microstructure Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
    }
    
    .analysis-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
    }
    
    .warning-card {
        background: linear-gradient(135deg, #ffa726 0%, #fb8c00 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
    }
    
    .success-card {
        background: linear-gradient(135deg, #66bb6a 0%, #43a047 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    h1 {
        color: #2E3192;
        text-align: center;
        font-size: 2.5rem;
        margin-bottom: 2rem;
    }
    
    .stMetric {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
</style>
""", unsafe_allow_html=True)

def find_latest_analysis_files(pair="XAUUSD"):
    """Find the latest microstructure analysis files for a given pair"""
    latest_files = {
        'txt': None,
        'json': None,
        'png': None
    }
    
    # Look in current directory and data subdirectories
    search_paths = [
        ".",
        f"./{pair}",
        f"./data/{pair}",
        "./data"
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            # Find microstructure analysis files
            txt_files = glob.glob(f"{path}/*Microstructure_Analysis*Report*.txt")
            json_files = glob.glob(f"{path}/*Microstructure_Analysis*.json")
            png_files = glob.glob(f"{path}/*Microstructure_Analysis*.png")
            
            # Get the latest files
            if txt_files:
                latest_files['txt'] = max(txt_files, key=os.path.getctime)
            if json_files:
                latest_files['json'] = max(json_files, key=os.path.getctime)
            if png_files:
                latest_files['png'] = max(png_files, key=os.path.getctime)
    
    return latest_files

def load_analysis_data(txt_file, json_file):
    """Load and parse analysis data from files"""
    analysis_data = {}
    
    # Load TXT report
    if txt_file and os.path.exists(txt_file):
        with open(txt_file, 'r') as f:
            analysis_data['txt_content'] = f.read()
    
    # Load JSON data
    if json_file and os.path.exists(json_file):
        with open(json_file, 'r') as f:
            analysis_data['json_data'] = json.load(f)
    
    return analysis_data

def parse_txt_analysis(txt_content):
    """Parse key metrics from TXT analysis report"""
    metrics = {}
    
    # Extract key values using regex
    patterns = {
        'total_ticks': r'Total Ticks Analyzed: (\d+)',
        'price_range': r'Price Range: ([\d.]+) - ([\d.]+)',
        'trend': r'Overall Trend: (\w+)',
        'avg_spread': r'Average Spread: ([\d.]+) pips',
        'manipulation_score': r'Manipulation Activity Score: ([\d.]+)%',
        'stop_hunts': r'Stop Hunts: (\d+) detected',
        'liquidity_sweeps': r'Liquidity Sweeps: (\d+) detected',
        'bullish_fvgs': r'Bullish Fair Value Gaps: (\d+)',
        'bearish_fvgs': r'Bearish Fair Value Gaps: (\d+)',
        'smc_bias': r'SMC Bias: (\w+)',
        'inducement_rate': r'Inducement Rate: ([\d.]+)%'
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, txt_content)
        if match:
            if key == 'price_range':
                metrics['price_min'] = float(match.group(1))
                metrics['price_max'] = float(match.group(2))
            elif key in ['total_ticks', 'stop_hunts', 'liquidity_sweeps', 'bullish_fvgs', 'bearish_fvgs']:
                metrics[key] = int(match.group(1))
            elif key in ['avg_spread', 'manipulation_score', 'inducement_rate']:
                metrics[key] = float(match.group(1))
            else:
                metrics[key] = match.group(1)
    
    return metrics

def explain_microstructure_analysis(metrics, json_data):
    """Generate detailed explanations of the microstructure analysis"""
    explanations = {}
    
    # Manipulation Analysis
    if 'manipulation_score' in metrics:
        score = metrics['manipulation_score']
        if score > 40:
            explanations['manipulation'] = {
                'level': 'HIGH',
                'color': 'red',
                'explanation': f"🚨 Heavy institutional activity detected ({score:.1f}%). Market makers and institutions are actively manipulating price through stop hunting and liquidity sweeps. Expect high volatility and potential false breakouts."
            }
        elif score > 20:
            explanations['manipulation'] = {
                'level': 'MEDIUM',
                'color': 'orange', 
                'explanation': f"⚠️ Moderate manipulation activity ({score:.1f}%). Some institutional order flow present. Watch for potential traps around key levels."
            }
        else:
            explanations['manipulation'] = {
                'level': 'LOW',
                'color': 'green',
                'explanation': f"✅ Low manipulation activity ({score:.1f}%). Market showing more organic price action with minimal institutional interference."
            }
    
    # SMC Analysis
    if 'smc_bias' in metrics:
        bias = metrics['smc_bias']
        bullish_fvgs = metrics.get('bullish_fvgs', 0)
        bearish_fvgs = metrics.get('bearish_fvgs', 0)
        
        explanations['smc'] = {
            'bias': bias,
            'explanation': f"📈 Smart Money Concepts show {bias} bias with {bullish_fvgs} bullish and {bearish_fvgs} bearish Fair Value Gaps. FVGs act as magnets for price and often provide excellent entry opportunities."
        }
    
    # Wyckoff Analysis
    if json_data and 'wyckoff' in json_data:
        phases = json_data['wyckoff']['phases']
        dominant_phase = max(phases.keys(), key=lambda k: phases[k]) if any(phases.values()) else 'Ranging'
        
        explanations['wyckoff'] = {
            'phase': dominant_phase,
            'explanation': f"🔄 Wyckoff Analysis indicates {dominant_phase} phase. This helps identify where institutional money is positioning for the next major move."
        }
    
    # Inducement Analysis
    if 'inducement_rate' in metrics:
        rate = metrics['inducement_rate']
        explanations['inducement'] = {
            'rate': rate,
            'explanation': f"🪤 Inducement rate of {rate:.1f}% shows how often price creates false signals to trap retail traders. Higher rates indicate more deceptive market conditions."
        }
    
    return explanations

def create_analysis_summary_cards(metrics, explanations):
    """Create summary cards for key analysis points"""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'manipulation' in explanations:
            level = explanations['manipulation']['level']
            score = metrics.get('manipulation_score', 0)
            if level == 'HIGH':
                st.markdown(f"""
                <div class="warning-card">
                    <h4>🚨 Manipulation Level</h4>
                    <h2>{level}</h2>
                    <p>{score:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="success-card">
                    <h4>✅ Manipulation Level</h4>
                    <h2>{level}</h2>
                    <p>{score:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        if 'smc' in explanations:
            bias = explanations['smc']['bias']
            fvgs = metrics.get('bullish_fvgs', 0) + metrics.get('bearish_fvgs', 0)
            st.markdown(f"""
            <div class="analysis-card">
                <h4>📈 SMC Bias</h4>
                <h2>{bias}</h2>
                <p>{fvgs} FVGs</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        if 'stop_hunts' in metrics:
            hunts = metrics['stop_hunts']
            st.markdown(f"""
            <div class="metric-card">
                <h4>🎯 Stop Hunts</h4>
                <h2>{hunts}</h2>
                <p>Detected Events</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        if 'liquidity_sweeps' in metrics:
            sweeps = metrics['liquidity_sweeps']
            st.markdown(f"""
            <div class="metric-card">
                <h4>💧 Liquidity Sweeps</h4>
                <h2>{sweeps}</h2>
                <p>Detected Events</p>
            </div>
            """, unsafe_allow_html=True)

def display_detailed_explanations(explanations):
    """Display detailed explanations of analysis results"""
    
    st.markdown("### 📋 Detailed Analysis Explanations")
    
    for analysis_type, data in explanations.items():
        with st.expander(f"📊 {analysis_type.upper()} Analysis", expanded=True):
            st.markdown(data['explanation'])

def create_strategy_recommendations(metrics, json_data):
    """Generate trading strategy recommendations based on analysis"""
    
    recommendations = []
    
    # Based on manipulation score
    if 'manipulation_score' in metrics:
        score = metrics['manipulation_score']
        if score > 40:
            recommendations.extend([
                "⚠️ Use wider stop losses due to high manipulation activity",
                "📉 Consider smaller position sizes in volatile conditions",
                "🎯 Wait for clear institutional direction before entering trades",
                "⏰ Avoid trading during high manipulation periods"
            ])
        else:
            recommendations.append("✅ Normal position sizing acceptable with standard stops")
    
    # Based on SMC bias
    if 'smc_bias' in metrics:
        bias = metrics['smc_bias']
        if bias == 'BULLISH':
            recommendations.extend([
                "📈 Focus on LONG setups at Fair Value Gap levels",
                "🔍 Look for bullish order blocks as entry zones",
                "📍 Target liquidity pools above recent highs"
            ])
        elif bias == 'BEARISH':
            recommendations.extend([
                "📉 Focus on SHORT setups at Fair Value Gap levels", 
                "🔍 Look for bearish order blocks as entry zones",
                "📍 Target liquidity pools below recent lows"
            ])
    
    # Based on Wyckoff phase
    if json_data and 'inducement' in json_data:
        strategy_insights = json_data['inducement'].get('strategy_insights', {})
        
        if 'v12_strategy' in strategy_insights:
            complexity = strategy_insights['v12_strategy'].get('market_complexity', 'MEDIUM')
            if complexity == 'HIGH':
                recommendations.append("🧠 High complexity market - use advanced multi-timeframe analysis")
            
    return recommendations

# Main Dashboard
def main():
    st.title("🎯 ZanFlow Advanced Microstructure Dashboard")
    
    # Sidebar for pair selection
    st.sidebar.title("📊 Analysis Controls")
    
    # Currency pair selection
    available_pairs = ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]
    selected_pair = st.sidebar.selectbox("Select Currency Pair", available_pairs)
    
    # Auto-refresh option
    auto_refresh = st.sidebar.checkbox("Auto-refresh every 30 seconds", value=False)
    
    if auto_refresh:
        st.rerun()
    
    # Find latest analysis files
    with st.spinner(f"Loading latest analysis for {selected_pair}..."):
        latest_files = find_latest_analysis_files(selected_pair)
    
    # Display file status
    st.sidebar.markdown("### 📁 File Status")
    for file_type, file_path in latest_files.items():
        if file_path:
            file_name = os.path.basename(file_path)
            st.sidebar.success(f"✅ {file_type.upper()}: {file_name}")
        else:
            st.sidebar.error(f"❌ No {file_type.upper()} file found")
    
    # Load and analyze data
    if latest_files['txt'] or latest_files['json']:
        analysis_data = load_analysis_data(latest_files['txt'], latest_files['json'])
        
        # Parse analysis
        metrics = {}
        if 'txt_content' in analysis_data:
            metrics = parse_txt_analysis(analysis_data['txt_content'])
        
        json_data = analysis_data.get('json_data', {})
        explanations = explain_microstructure_analysis(metrics, json_data)
        
        # Display summary cards
        create_analysis_summary_cards(metrics, explanations)
        
        # Display chart if available
        if latest_files['png']:
            st.markdown("### 📈 Microstructure Analysis Chart")
            st.image(latest_files['png'], use_column_width=True)
        
        # Two-column layout for detailed analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Display detailed explanations
            display_detailed_explanations(explanations)
            
            # Raw analysis metrics
            if metrics:
                with st.expander("📊 Raw Analysis Metrics"):
                    for key, value in metrics.items():
                        st.metric(key.replace('_', ' ').title(), value)
        
        with col2:
            # Strategy recommendations
            st.markdown("### 🎯 Trading Strategy Recommendations")
            recommendations = create_strategy_recommendations(metrics, json_data)
            
            for rec in recommendations:
                st.markdown(f"- {rec}")
            
            # JSON data display
            if json_data:
                with st.expander("🔧 JSON Analysis Data"):
                    st.json(json_data)
        
        # Full text report
        if 'txt_content' in analysis_data:
            with st.expander("📋 Full Analysis Report"):
                st.text(analysis_data['txt_content'])
        
        # Performance metrics
        st.markdown("### 📊 Analysis Performance")
        perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
        
        with perf_col1:
            st.metric("Total Ticks", metrics.get('total_ticks', 'N/A'))
        with perf_col2:
            st.metric("Price Range", f"{metrics.get('price_max', 0) - metrics.get('price_min', 0):.2f}" if 'price_max' in metrics else 'N/A')
        with perf_col3:
            st.metric("Average Spread", f"{metrics.get('avg_spread', 0):.2f} pips" if 'avg_spread' in metrics else 'N/A')
        with perf_col4:
            st.metric("Analysis Time", datetime.now().strftime("%H:%M:%S"))
    
    else:
        st.warning(f"⚠️ No microstructure analysis files found for {selected_pair}")
        st.info("📝 Make sure your analysis files follow the naming pattern: *Microstructure_Analysis*")
    
    # Footer
    st.markdown("---")
    st.markdown("**ZanFlow Microstructure Dashboard** - Advanced institutional flow analysis")

if __name__ == "__main__":
    main()
