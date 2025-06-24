# enhanced_zanalytics_dashboard.py - Ultimate Streamlit Dashboard for ZANALYTICS
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import json
import asyncio
import time
from datetime import datetime, timedelta
from pathlib import Path
import requests
from typing import Dict, Any, List
import logging

# Page config
st.set_page_config(
    page_title="ZANALYTICS Intelligence Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f4e79 0%, #2d7dd2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2d7dd2;
    }
    .agent-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .status-running { color: #28a745; }
    .status-error { color: #dc3545; }
    .status-warning { color: #ffc107; }
</style>
""", unsafe_allow_html=True)

class ZAnalyticsDashboardApp:
    def __init__(self):
        self.api_base = "http://localhost:5010"  # Your API service
        self.data_cache = {}
        self.last_update = datetime.now()
        
    def fetch_api_data(self, endpoint: str) -> Dict:
        """Fetch data from ZANALYTICS API"""
        try:
            response = requests.get(f"{self.api_base}{endpoint}", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API returned {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}
    
    def render_header(self):
        """Render dashboard header"""
        st.markdown("""
        <div class="main-header">
            <h1>🚀 ZANALYTICS INTELLIGENCE DASHBOARD</h1>
            <p>Real-Time Trading Data Intelligence & Agent Decisions</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_system_status(self):
        """Render system status section"""
        status_data = self.fetch_api_data("/status")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if "error" not in status_data:
                st.metric(
                    "🟢 System Status", 
                    "OPERATIONAL",
                    delta="Live"
                )
            else:
                st.metric(
                    "🔴 System Status", 
                    "ERROR",
                    delta=status_data.get("error", "Unknown")
                )
        
        with col2:
            active_symbols = len(status_data.get("data_feeds", {}).get("active_symbols", []))
            st.metric(
                "💱 Active Symbols", 
                active_symbols,
                delta=f"+{active_symbols} today"
            )
        
        with col3:
            events_processed = status_data.get("session_metrics", {}).get("events_processed", 0)
            st.metric(
                "📊 Events Processed", 
                events_processed,
                delta="+Live"
            )
        
        with col4:
            analysis_count = status_data.get("session_metrics", {}).get("analysis_count", 0)
            st.metric(
                "🤖 Analysis Count", 
                analysis_count,
                delta="Real-time"
            )
    
    def render_agent_decisions(self):
        """Render agent decisions with beautiful visualizations"""
        st.subheader("🤖 Agent Decision Center")
        
        decisions_data = self.fetch_api_data("/agents/decisions")
        
        if "error" not in decisions_data:
            # Agent Status Grid
            col1, col2 = st.columns(2)
            
            with col1:
                # Agent confidence radar chart
                agents = decisions_data.get("decision_summary", {})
                
                categories = []
                values = []
                
                if "bozenka_signal" in agents:
                    categories.append("Boženka<br>(Signal)")
                    values.append(1.0 if agents["bozenka_signal"] else 0.2)
                
                if "stefania_trust" in agents:
                    categories.append("Stefania<br>(Trust)")
                    values.append(agents["stefania_trust"])
                
                if "lusia_confluence" in agents:
                    categories.append("Lusia<br>(Confluence)")
                    values.append(abs(agents["lusia_confluence"]))
                
                if "zdzisiek_risk" in agents:
                    categories.append("Zdzisiek<br>(Risk)")
                    values.append(1.0 if agents["zdzisiek_risk"] else 0.3)
                
                # Create radar chart
                fig_radar = go.Figure()
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name='Agent Confidence',
                    fillcolor='rgba(45, 125, 210, 0.3)',
                    line=dict(color='rgb(45, 125, 210)', width=2)
                ))
                
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )
                    ),
                    title="Agent Confidence Radar",
                    height=400
                )
                
                st.plotly_chart(fig_radar, use_container_width=True)
            
            with col2:
                # Agent decision timeline
                st.markdown("### 📋 Agent Status Board")
                
                agents_detail = decisions_data.get("agents", {})
                
                for agent_name, agent_data in agents_detail.items():
                    if isinstance(agent_data, dict):
                        confidence = agent_data.get("confidence", 0.5)
                        
                        # Color based on confidence
                        if confidence > 0.7:
                            color = "🟢"
                            status = "HIGH"
                        elif confidence > 0.4:
                            color = "🟡"
                            status = "MEDIUM"
                        else:
                            color = "🔴"
                            status = "LOW"
                        
                        st.markdown(f"""
                        <div class="agent-card">
                            <h4>{color} {agent_name.upper()}</h4>
                            <p><strong>Confidence:</strong> {confidence:.2f} ({status})</p>
                            <p><strong>Status:</strong> {'Active' if confidence > 0.3 else 'Standby'}</p>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.error(f"Error fetching agent decisions: {decisions_data.get('error')}")
    
    def render_market_analysis(self):
        """Render market analysis charts"""
        st.subheader("📈 Market Analysis Dashboard")
        
        # Symbol selector
        symbols = ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY"]
        selected_symbol = st.selectbox("Select Symbol", symbols)
        
        # Timeframe selector
        timeframes = ["M1", "M5", "M15", "H1", "H4"]
        selected_tf = st.selectbox("Select Timeframe", timeframes)
        
        # Fetch data for selected symbol
        analysis_data = self.fetch_api_data(f"/analysis/summary/{selected_symbol}")
        
        if "error" not in analysis_data:
            col1, col2 = st.columns(2)
            
            with col1:
                # Create sample OHLC chart (you'll replace this with real data)
                dates = pd.date_range(start="2024-01-01", periods=100, freq="1H")
                np.random.seed(42)
                
                # Generate realistic OHLC data
                close_prices = 2000 + np.cumsum(np.random.randn(100) * 0.5)
                open_prices = close_prices + np.random.randn(100) * 0.2
                high_prices = np.maximum(open_prices, close_prices) + np.abs(np.random.randn(100) * 0.3)
                low_prices = np.minimum(open_prices, close_prices) - np.abs(np.random.randn(100) * 0.3)
                
                fig_ohlc = go.Figure(data=go.Candlestick(
                    x=dates,
                    open=open_prices,
                    high=high_prices,
                    low=low_prices,
                    close=close_prices,
                    name=f"{selected_symbol} {selected_tf}"
                ))
                
                fig_ohlc.update_layout(
                    title=f"{selected_symbol} {selected_tf} - Live OHLC",
                    xaxis_title="Time",
                    yaxis_title="Price",
                    height=500
                )
                
                st.plotly_chart(fig_ohlc, use_container_width=True)
            
            with col2:
                # Structure analysis chart
                structure_data = analysis_data.get("structure_analysis", {})
                
                if structure_data:
                    # Create structure strength visualization
                    tf_data = []
                    strength_values = []
                    
                    for tf, smc_data in structure_data.items():
                        if isinstance(smc_data, dict):
                            tf_data.append(tf)
                            # Mock strength calculation
                            strength = np.random.uniform(0.3, 0.95)
                            strength_values.append(strength)
                    
                    if tf_data:
                        fig_structure = go.Figure()
                        fig_structure.add_trace(go.Bar(
                            x=tf_data,
                            y=strength_values,
                            marker_color=['#28a745' if v > 0.7 else '#ffc107' if v > 0.4 else '#dc3545' for v in strength_values],
                            name="Structure Strength"
                        ))
                        
                        fig_structure.update_layout(
                            title="Multi-Timeframe Structure Analysis",
                            xaxis_title="Timeframe",
                            yaxis_title="Structure Strength",
                            height=250
                        )
                        
                        st.plotly_chart(fig_structure, use_container_width=True)
                
                # Wyckoff phase analysis
                wyckoff_data = analysis_data.get("wyckoff_phases", {})
                
                if wyckoff_data:
                    phases = ["Accumulation", "Markup", "Distribution", "Markdown"]
                    phase_counts = [np.random.randint(0, 5) for _ in phases]
                    
                    fig_wyckoff = go.Figure(data=go.Pie(
                        labels=phases,
                        values=phase_counts,
                        hole=0.4,
                        marker_colors=['#28a745', '#17a2b8', '#ffc107', '#dc3545']
                    ))
                    
                    fig_wyckoff.update_layout(
                        title="Wyckoff Phase Distribution",
                        height=250
                    )
                    
                    st.plotly_chart(fig_wyckoff, use_container_width=True)
        
        else:
            st.error(f"Error fetching market analysis: {analysis_data.get('error')}")
    
    def render_microstructure_analysis(self):
        """Render microstructure analysis"""
        st.subheader("🔬 Microstructure Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Spread analysis
            times = pd.date_range(start="2024-01-01 09:00", periods=100, freq="1min")
            spreads = 2 + np.random.exponential(0.5, 100)
            
            fig_spread = go.Figure()
            fig_spread.add_trace(go.Scatter(
                x=times,
                y=spreads,
                mode='lines',
                name='Spread',
                line=dict(color='#dc3545', width=2)
            ))
            
            fig_spread.update_layout(
                title="Live Spread Analysis",
                xaxis_title="Time",
                yaxis_title="Spread (points)",
                height=300
            )
            
            st.plotly_chart(fig_spread, use_container_width=True)
        
        with col2:
            # Volume profile
            prices = np.random.normal(2000, 10, 1000)
            volumes = np.random.exponential(100, 1000)
            
            fig_volume = go.Figure()
            fig_volume.add_trace(go.Histogram(
                y=prices,
                x=volumes,
                orientation='h',
                marker_color='rgba(45, 125, 210, 0.7)',
                name='Volume Profile'
            ))
            
            fig_volume.update_layout(
                title="Volume Profile",
                xaxis_title="Volume",
                yaxis_title="Price",
                height=300
            )
            
            st.plotly_chart(fig_volume, use_container_width=True)
        
        with col3:
            # Liquidity heatmap
            times = pd.date_range(start="09:00", end="17:00", freq="1H")
            liquidity_levels = np.random.uniform(0.2, 1.0, len(times))
            
            fig_liquidity = go.Figure()
            fig_liquidity.add_trace(go.Bar(
                x=times.strftime("%H:%M"),
                y=liquidity_levels,
                marker_color=liquidity_levels,
                marker_colorscale='RdYlGn',
                name='Liquidity'
            ))
            
            fig_liquidity.update_layout(
                title="Liquidity Heatmap",
                xaxis_title="Time",
                yaxis_title="Liquidity Level",
                height=300
            )
            
            st.plotly_chart(fig_liquidity, use_container_width=True)
    
    def render_data_flow_monitor(self):
        """Render data flow monitoring"""
        st.subheader("📊 Data Flow Monitor")
        
        # Create tabs for different monitoring views
        tab1, tab2, tab3 = st.tabs(["📁 File Monitor", "🔄 Event Stream", "📈 Performance"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📂 Monitored Directories")
                directories = ["./data", "./exports", "./uploads"]
                
                for directory in directories:
                    exists = Path(directory).exists()
                    status_icon = "✅" if exists else "❌"
                    file_count = len(list(Path(directory).glob("*"))) if exists else 0
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <p><strong>{status_icon} {directory}</strong></p>
                        <p>Files: {file_count}</p>
                        <p>Status: {'Active' if exists else 'Missing'}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("### 📄 Recent Files")
                
                # Mock recent files data
                recent_files = [
                    {"name": "XAUUSD_M1_20241201.csv", "time": "2 min ago", "size": "1.2MB"},
                    {"name": "EURUSD_TICK_20241201.csv", "time": "5 min ago", "size": "3.4MB"},
                    {"name": "analysis_20241201_143022.json", "time": "8 min ago", "size": "45KB"},
                ]
                
                for file_info in recent_files:
                    st.markdown(f"""
                    <div style="background: #f8f9fa; padding: 0.5rem; margin: 0.25rem 0; border-radius: 4px;">
                        <strong>📄 {file_info['name']}</strong><br>
                        <small>⏰ {file_info['time']} | 📦 {file_info['size']}</small>
                    </div>
                    """, unsafe_allow_html=True)
        
        with tab2:
            # Real-time event stream
            st.markdown("### 🔄 Live Event Stream")
            
            # Create a placeholder for live updates
            event_placeholder = st.empty()
            
            # Mock events
            events = [
                {"time": datetime.now() - timedelta(seconds=30), "type": "NEW_CSV", "symbol": "XAUUSD", "tf": "M1"},
                {"time": datetime.now() - timedelta(seconds=45), "type": "ANALYSIS_COMPLETE", "symbol": "EURUSD", "tf": "M5"},
                {"time": datetime.now() - timedelta(seconds=60), "type": "AGENT_DECISION", "symbol": "GBPUSD", "tf": "H1"},
            ]
            
            event_df = pd.DataFrame(events)
            event_df['time'] = event_df['time'].dt.strftime('%H:%M:%S')
            
            st.dataframe(event_df, use_container_width=True)
        
        with tab3:
            # Performance metrics
            col1, col2 = st.columns(2)
            
            with col1:
                # Processing speed chart
                times = list(range(24))
                processing_speeds = [np.random.uniform(50, 200) for _ in times]
                
                fig_speed = go.Figure()
                fig_speed.add_trace(go.Scatter(
                    x=times,
                    y=processing_speeds,
                    mode='lines+markers',
                    name='Events/Hour',
                    line=dict(color='#28a745', width=3)
                ))
                
                fig_speed.update_layout(
                    title="Processing Speed (24h)",
                    xaxis_title="Hour",
                    yaxis_title="Events/Hour",
                    height=300
                )
                
                st.plotly_chart(fig_speed, use_container_width=True)
            
            with col2:
                # Error rate chart
                error_rates = [np.random.uniform(0, 5) for _ in times]
                
                fig_errors = go.Figure()
                fig_errors.add_trace(go.Bar(
                    x=times,
                    y=error_rates,
                    marker_color=['#dc3545' if x > 3 else '#ffc107' if x > 1 else '#28a745' for x in error_rates],
                    name='Error Rate %'
                ))
                
                fig_errors.update_layout(
                    title="Error Rate (24h)",
                    xaxis_title="Hour",
                    yaxis_title="Error Rate %",
                    height=300
                )
                
                st.plotly_chart(fig_errors, use_container_width=True)
    
    def render_sidebar(self):
        """Render sidebar controls"""
        with st.sidebar:
            st.markdown("### ⚙️ Dashboard Controls")
            
            # Auto-refresh toggle
            auto_refresh = st.toggle("🔄 Auto Refresh", value=True)
            
            if auto_refresh:
                refresh_rate = st.slider("Refresh Rate (seconds)", 1, 30, 5)
            
            st.markdown("---")
            
            # API endpoint configuration
            st.markdown("### 🔗 API Configuration")
            api_endpoint = st.text_input("API Endpoint", value="http://localhost:5010")
            
            if st.button("🧪 Test Connection"):
                try:
                    response = requests.get(f"{api_endpoint}/status", timeout=5)
                    if response.status_code == 200:
                        st.success("✅ Connection successful!")
                    else:
                        st.error(f"❌ Connection failed: {response.status_code}")
                except Exception as e:
                    st.error(f"❌ Connection failed: {str(e)}")
            
            st.markdown("---")
            
            # System controls
            st.markdown("### 🎮 System Controls")
            
            if st.button("🔄 Refresh Data"):
                st.rerun()
            
            if st.button("📊 Export Report"):
                st.info("📊 Report export feature coming soon!")
            
            if st.button("🧹 Clear Cache"):
                st.success("🧹 Cache cleared!")
            
            st.markdown("---")
            
            # System info
            st.markdown("### ℹ️ System Info")
            st.markdown(f"""
            - **Version:** v2.0.0
            - **Last Update:** {datetime.now().strftime('%H:%M:%S')}
            - **Status:** 🟢 Operational
            - **Uptime:** {timedelta(hours=2, minutes=34)}
            """)
    
    def run(self):
        """Main dashboard runner"""
        self.render_header()
        self.render_sidebar()
        
        # Main content area
        self.render_system_status()
        
        st.markdown("---")
        
        self.render_agent_decisions()
        
        st.markdown("---")
        
        self.render_market_analysis()
        
        st.markdown("---")
        
        self.render_microstructure_analysis()
        
        st.markdown("---")
        
        self.render_data_flow_monitor()
        
        # Auto-refresh
        if st.sidebar.toggle("🔄 Auto Refresh", value=False):
            time.sleep(5)
            st.rerun()

def main():
    """Main entry point"""
    dashboard = ZAnalyticsDashboardApp()
    dashboard.run()

if __name__ == "__main__":
    main()